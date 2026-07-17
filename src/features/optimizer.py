#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AI Score Optimizer — 백테스트 데이터 기반 가중치 튜닝 '보고서' 엔진.

⚠️ 이 모듈은 AI Score 알고리즘을 절대 수정하지 않는다.
   동일한(현행) 알고리즘을 과거 데이터에 재실행해 컴포넌트별 점수를
   수집하고, 실제 수익률과 연결해 다음을 산출한다:

  ① Component Performance — 컴포넌트별 평균수익률·승률·기여도·상관관계
  ② Weight Suggestion     — 데이터 기반 추천 가중치 + 변경 이유
  ③ Correlation Matrix    — 컴포넌트 간 상관 → 중복 Feature 탐지
  ④ Feature Importance    — 컴포넌트 제거(ablation) 시 승률/수익 변화
  ⑤ Report                — Markdown / HTML / CSV 자동 생성

설계 노트
---------
- 일별 점수 파일(data/scores/*.json)에는 총점·근거만 있고 컴포넌트별
  점수가 없다 → 빌더가 ai2 git 히스토리 CSV에 현행 compute_ai_score를
  재실행해 rec-level 데이터셋(data/optimizer/component_data.csv)을 만든다.
  원본 CSV는 data/optimizer/raw/ 에 캐시(gitignore) → 재실행은 오프라인.
- 가중치 추천은 '문서화된 휴리스틱'이다:
    signal = 0.5·norm(상관계수) + 0.5·norm(승률 lift)
    추천 = clip(현행 × (1 + 0.4·signal), 3, 25) 후 합계 100으로 재정규화
  결정은 사람이 한다 — 이 모듈은 근거를 만들 뿐이다.
- Ablation은 점수만 다시 계산하는 게 아니라 '그 컴포넌트 없이 재선정'
  했을 때(일별 Top-N) 성과 변화를 본다 → 선택 효과까지 측정.
"""
from __future__ import annotations

import os
import io
import re
import csv
import json
import glob
import logging
import argparse
import datetime
import statistics
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features.ai_score import compute_ai_score, score_universe
from src.features.backtest import PriceBook

logger = logging.getLogger(__name__)

OPT_DIR = os.path.join('data', 'optimizer')
RAW_DIR = os.path.join(OPT_DIR, 'raw')
DATASET = os.path.join(OPT_DIR, 'component_data.csv')

API = 'https://api.github.com/repos/perseus2133-ai/ai2/commits'
RAW_URL = 'https://raw.githubusercontent.com/perseus2133-ai/ai2/{sha}/data/consensus_data.csv'

TOP_PER_DAY = 200          # 데이터셋에 담을 일별 상위 종목 수
SELECT_N = 20              # ablation 재선정 Top-N (백테스트와 동일 정책)
REDUNDANT_R = 0.7          # 중복 Feature 판정 임계 |r|
WEIGHT_MIN, WEIGHT_MAX = 3.0, 25.0
SIGNAL_GAIN = 0.4


# ============================================================
# 현행 가중치 introspection (상수 중복 정의 금지 — 알고리즘이 진실)
# ============================================================
def current_weights() -> dict[str, float]:
    """만점 합성 행에 현행 알고리즘을 실행해 컴포넌트 만점을 읽는다."""
    row = pd.Series({
        '종목코드': '000000', '종목명': '_probe_',
        '현재가': 105.0, '저항선': 100.0,
        '영업이익_2025': 100.0, '영업이익_최대성장률': 300.0,
        'PER': 10.0, '영업이익_성장률_2026': 100.0,
        '외인_5d': 1.0, '기관_5d': 1.0, '외인_20d': 1.0, '기관_20d': 1.0,
        '거래량배수': 3.5, 'RSI': np.nan,
        'MA_align': 'up', 'MACD_signal': 'bull', 'OBV_trend': 'up',
    })
    res = compute_ai_score(row, turnover_rank=1, acc_count=6)
    return {name: c.maximum for name, c in res.components.items()}


# ============================================================
# 데이터셋 빌더
# ============================================================
def _fetch_day_csv(day: str, sha: str) -> pd.DataFrame | None:
    """해당 거래일의 ai2 CSV (로컬 캐시 우선)."""
    import requests
    os.makedirs(RAW_DIR, exist_ok=True)
    cache = os.path.join(RAW_DIR, f'{day}.csv.gz')
    if os.path.exists(cache):
        return pd.read_csv(cache, dtype={'종목코드': str}, compression='gzip')
    r = requests.get(RAW_URL.format(sha=sha), timeout=60,
                     headers={'User-Agent': 'volume-radar'})
    if r.status_code != 200:
        logger.warning('%s: CSV 다운로드 실패(%s)', day, r.status_code)
        return None
    df = pd.read_csv(io.StringIO(r.content.decode('utf-8-sig')), dtype={'종목코드': str})
    df.to_csv(cache, index=False, compression='gzip', encoding='utf-8')
    return df


def _commit_map() -> dict[str, str]:
    import requests
    r = requests.get(API, params={'path': 'data/consensus_data.csv', 'per_page': 80},
                     timeout=30, headers={'User-Agent': 'volume-radar'})
    r.raise_for_status()
    out: dict[str, str] = {}
    for c in r.json():
        msg = c['commit']['message']
        if 'Auto crawl' not in msg:
            continue
        m = re.search(r'(\d{4}-\d{2}-\d{2})', msg)
        if not m:
            continue
        day = (datetime.date.fromisoformat(m.group(1))
               - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
        out.setdefault(day, c['sha'])
    return out


def dataset_rows_for_day(ai2_df: pd.DataFrame, day: str,
                         acc_counts: dict[str, int],
                         book: PriceBook,
                         horizons: tuple[int, ...] = (5, 10)) -> list[dict]:
    """하루치 rec-level 행 생성 (컴포넌트 + 보조 지표 + 미래 수익률)."""
    results = score_universe(ai2_df, acc_counts)[:TOP_PER_DAY]
    idx = ai2_df.copy()
    idx['종목코드'] = idx['종목코드'].astype(str).str.zfill(6)
    idx = idx.set_index('종목코드')

    rows = []
    for rank, r in enumerate(results, start=1):
        src = idx.loc[r.code] if r.code in idx.index else pd.Series(dtype=object)
        row: dict = {
            'date': day, 'code': r.code, 'name': r.name,
            'score': r.score, 'rank': rank, 'penalty': r.penalty,
            'avail_max': r.coverage_pct,
            'ind_ma_up': 1 if src.get('MA_align') == 'up' else 0,
            'ind_macd_bull': 1 if src.get('MACD_signal') in ('bull', 'bull_cross') else 0,
            'ind_obv_up': 1 if src.get('OBV_trend') == 'up' else 0,
            'rsi': pd.to_numeric(src.get('RSI'), errors='coerce'),
        }
        for comp, c in r.components.items():
            row[f'{comp}_earned'] = c.earned
            row[f'{comp}_max'] = c.maximum
            row[f'{comp}_avail'] = 1 if c.available else 0
        entry = book.close(r.code, day)
        for h in horizons:
            ret = None
            if entry:
                fday = book.future_day(day, h)
                if fday:
                    fclose = book.close(r.code, fday)
                    if fclose:
                        ret = round((fclose / entry - 1) * 100, 2)
            row[f'ret_{h}d'] = ret
        rows.append(row)
    return rows


def build_dataset(snap_dir: str = os.path.join('data', 'snapshots'),
                  rebuild: bool = False) -> pd.DataFrame:
    """전체 거래일 데이터셋 구축 (캐시 존재 시 재사용)."""
    if os.path.exists(DATASET) and not rebuild:
        return pd.read_csv(DATASET, dtype={'code': str})
    from backfill_scores import acc_counts_asof   # 재사용 (look-ahead 없음)
    book = PriceBook(snap_dir)
    cmap = _commit_map()
    all_rows: list[dict] = []
    for day in book.days:
        sha = cmap.get(day)
        if not sha:
            logger.warning('%s: 커밋 없음 — 스킵', day)
            continue
        df = _fetch_day_csv(day, sha)
        if df is None:
            continue
        rows = dataset_rows_for_day(df, day, acc_counts_asof(day), book)
        all_rows.extend(rows)
        logger.info('%s: %d행', day, len(rows))
    ds = pd.DataFrame(all_rows)
    os.makedirs(OPT_DIR, exist_ok=True)
    ds.to_csv(DATASET, index=False, encoding='utf-8-sig')
    return ds


# ============================================================
# ① Component Performance
# ============================================================
def component_performance(ds: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    ret_col = f'ret_{horizon}d'
    comps = sorted(c[:-7] for c in ds.columns if c.endswith('_earned'))
    rows = []
    for comp in comps:
        e, m, a = f'{comp}_earned', f'{comp}_max', f'{comp}_avail'
        sub = ds[(ds[a] == 1) & ds[ret_col].notna()]
        if sub.empty:
            rows.append({'component': comp, 'n': 0, 'available_pct': 0.0,
                         'avg_earned': None, 'corr_vs_ret': None,
                         'win_has': None, 'win_not': None, 'win_lift': None,
                         'mean_has': None, 'mean_not': None, 'note': '전량 결측 — 분석 불가'})
            continue
        has = sub[sub[e] >= sub[m] * 0.5]          # 컴포넌트 '보유' = 만점의 50%+
        not_ = sub[sub[e] < sub[m] * 0.5]
        corr = None
        if sub[e].nunique() > 1:
            corr = round(float(np.corrcoef(sub[e], sub[ret_col])[0, 1]), 3)
        def _win(x): return round(float((x[ret_col] > 0).mean() * 100), 1) if len(x) else None
        def _mean(x): return round(float(x[ret_col].mean()), 2) if len(x) else None
        win_h, win_n = _win(has), _win(not_)
        rows.append({
            'component': comp, 'n': len(sub),
            'available_pct': round(float((ds[a] == 1).mean() * 100), 1),
            'avg_earned': round(float(sub[e].mean()), 2),
            'corr_vs_ret': corr,
            'win_has': win_h, 'win_not': win_n,
            'win_lift': round(win_h - win_n, 1) if (win_h is not None and win_n is not None) else None,
            'mean_has': _mean(has), 'mean_not': _mean(not_),
            'note': '',
        })
    return pd.DataFrame(rows)


# ============================================================
# ② Weight Suggestion (문서화된 휴리스틱)
# ============================================================
def weight_suggestions(perf: pd.DataFrame,
                       weights: dict[str, float] | None = None) -> pd.DataFrame:
    weights = weights or current_weights()
    p = perf.set_index('component')

    corr = p['corr_vs_ret'].astype(float)
    lift = p['win_lift'].astype(float)
    c_max = max(corr.abs().max(), 1e-9)
    l_max = max(lift.abs().max(), 1e-9)

    rows = []
    for comp, cur in weights.items():
        if comp not in p.index or p.loc[comp, 'n'] == 0:
            rows.append({'component': comp, 'current': cur, 'suggested': cur,
                         'signal': None,
                         'reason': '데이터 결측으로 분석 불가 — 현행 유지 (복구 후 재평가)'})
            continue
        c = corr.get(comp)
        l = lift.get(comp)
        sig = 0.0
        parts = []
        if pd.notna(c):
            sig += 0.5 * (c / c_max)
            parts.append(f'수익률 상관 {c:+.2f}')
        if pd.notna(l):
            sig += 0.5 * (l / l_max)
            parts.append(f'보유 시 승률 {l:+.1f}%p')
        suggested = float(np.clip(cur * (1 + SIGNAL_GAIN * sig), WEIGHT_MIN, WEIGHT_MAX))
        direction = '증가' if suggested > cur + 0.5 else ('감소' if suggested < cur - 0.5 else '유지')
        rows.append({'component': comp, 'current': cur,
                     'suggested': suggested, 'signal': round(sig, 3),
                     'reason': f'{" · ".join(parts)} → {direction}'})
    out = pd.DataFrame(rows)

    # 분석 가능한 컴포넌트만 재정규화해 합계를 명목 100으로
    adj = out['signal'].notna()
    fixed = out.loc[~adj, 'current'].sum()
    target = sum(weights.values()) - fixed
    s = out.loc[adj, 'suggested'].sum()
    if s > 0 and target > 0:
        out.loc[adj, 'suggested'] = out.loc[adj, 'suggested'] / s * target
    out['suggested'] = out['suggested'].round(1)
    out['delta'] = (out['suggested'] - out['current']).round(1)
    return out


# ============================================================
# ③ Correlation Matrix
# ============================================================
def correlation_matrix(ds: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    comps = sorted(c[:-7] for c in ds.columns if c.endswith('_earned'))
    cols = []
    for comp in comps:
        col = ds[ds[f'{comp}_avail'] == 1][f'{comp}_earned']
        if col.nunique() > 1:
            cols.append(comp)
    mat = ds[[f'{c}_earned' for c in cols]].corr()
    mat.index = cols
    mat.columns = cols
    redundant = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = mat.loc[a, b]
            if pd.notna(r) and abs(r) >= REDUNDANT_R:
                redundant.append({'a': a, 'b': b, 'corr': round(float(r), 3),
                                  'suggestion': f'{a} ↔ {b} 상관 {r:.2f} — 중복 신호, '
                                                f'둘 중 하나 비중 축소 검토'})
    return mat.round(3), redundant


# ============================================================
# ④ Feature Importance (ablation — 재선정 기반)
# ============================================================
def _select_topn(ds: pd.DataFrame, score_col: str, n: int) -> pd.DataFrame:
    return (ds.sort_values(['date', score_col], ascending=[True, False])
              .groupby('date').head(n))


def ablation_importance(ds: pd.DataFrame, horizon: int = 5,
                        select_n: int = SELECT_N) -> pd.DataFrame:
    ret_col = f'ret_{horizon}d'
    comps = sorted(c[:-7] for c in ds.columns if c.endswith('_earned'))
    d = ds.copy()

    def metrics(sel: pd.DataFrame) -> tuple[float | None, float | None, int]:
        ok = sel[sel[ret_col].notna()]
        if ok.empty:
            return None, None, 0
        return (round(float((ok[ret_col] > 0).mean() * 100), 1),
                round(float(ok[ret_col].mean()), 2), len(ok))

    base_sel = _select_topn(d, 'score', select_n)
    base_win, base_mean, base_n = metrics(base_sel)

    rows = [{'component': '(baseline)', 'win_rate': base_win,
             'mean_ret': base_mean, 'n': base_n,
             'win_delta': 0.0, 'mean_delta': 0.0}]
    for comp in comps:
        e, m, a = f'{comp}_earned', f'{comp}_max', f'{comp}_avail'
        # 컴포넌트 제거 후 가용 만점 재계산 → 재정규화 점수
        earned_wo = d['score'] / 100 * d['avail_max'] + d['penalty'] - d[e].where(d[a] == 1, 0)
        avail_wo = d['avail_max'] - d[m].where(d[a] == 1, 0)
        d['_ablated'] = np.where(avail_wo > 0,
                                 (earned_wo / avail_wo * 100) - d['penalty'], 0.0)
        sel = _select_topn(d, '_ablated', select_n)
        win, mean, n = metrics(sel)
        rows.append({'component': comp, 'win_rate': win, 'mean_ret': mean, 'n': n,
                     'win_delta': (round(win - base_win, 1)
                                   if (win is not None and base_win is not None) else None),
                     'mean_delta': (round(mean - base_mean, 2)
                                    if (mean is not None and base_mean is not None) else None)})
    return pd.DataFrame(rows)


# ============================================================
# ⑤ Report 생성 (MD / HTML / CSV)
# ============================================================
def generate_reports(perf: pd.DataFrame, sugg: pd.DataFrame,
                     corr: pd.DataFrame, redundant: list[dict],
                     abl: pd.DataFrame, out_dir: str = OPT_DIR,
                     horizon: int = 5, meta: dict | None = None) -> dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    asof = datetime.date.today().isoformat()
    paths = {}

    # CSV
    paths['perf_csv'] = os.path.join(out_dir, 'component_performance.csv')
    perf.to_csv(paths['perf_csv'], index=False, encoding='utf-8-sig')
    paths['sugg_csv'] = os.path.join(out_dir, 'weight_suggestions.csv')
    sugg.to_csv(paths['sugg_csv'], index=False, encoding='utf-8-sig')
    paths['abl_csv'] = os.path.join(out_dir, 'feature_importance.csv')
    abl.to_csv(paths['abl_csv'], index=False, encoding='utf-8-sig')
    paths['corr_csv'] = os.path.join(out_dir, 'correlation_matrix.csv')
    corr.to_csv(paths['corr_csv'], encoding='utf-8-sig')

    # Markdown
    md = [f'# AI Score Optimizer Report ({asof})', '']
    if meta:
        md += [f"- 데이터: {meta.get('rows', '?')}행 · {meta.get('days', '?')}거래일 "
               f"· 기준 수익률 {horizon}거래일", '']
    md += ['> ⚠️ 본 보고서는 **제안**이다. AI Score 알고리즘은 수정되지 않았다.', '',
           '## ① 컴포넌트 성능', '', perf.to_markdown(index=False), '',
           '## ② 가중치 제안', '', sugg.to_markdown(index=False), '',
           '## ③ 컴포넌트 상관관계', '', corr.to_markdown(), '']
    if redundant:
        md += ['### 중복 신호 (|r| ≥ %.1f)' % REDUNDANT_R, '']
        md += [f"- {r['suggestion']}" for r in redundant]
    else:
        md += [f'중복 신호 없음 (모든 쌍 |r| < {REDUNDANT_R})']
    md += ['', '## ④ Feature Importance (제거 시 변화)', '',
           abl.to_markdown(index=False), '',
           '해석: win_delta > 0 → 그 컴포넌트를 빼고 뽑는 편이 승률이 높았다'
           ' (= 해당 기간 그 컴포넌트가 해로웠다는 신호).', '']
    paths['md'] = os.path.join(out_dir, f'{asof}_optimizer_report.md')
    with open(paths['md'], 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))

    # HTML (자립형, 다크 테마)
    style = ('<style>body{background:#1A2332;color:#E2E8F0;font-family:sans-serif;'
             'padding:24px;}table{border-collapse:collapse;margin:12px 0;}'
             'th,td{border:1px solid #4A5568;padding:6px 10px;font-size:13px;}'
             'th{background:#232E40;color:#94A3B8;}h1,h2{color:#62EFFF;}'
             'blockquote{color:#FBBF24;border-left:3px solid #FBBF24;padding-left:10px;}'
             '</style>')
    html_parts = [style, f'<h1>AI Score Optimizer Report ({asof})</h1>',
                  '<blockquote>본 보고서는 제안이다. AI Score 알고리즘은 수정되지 않았다.</blockquote>',
                  '<h2>① 컴포넌트 성능</h2>', perf.to_html(index=False),
                  '<h2>② 가중치 제안</h2>', sugg.to_html(index=False),
                  '<h2>③ 상관관계</h2>', corr.to_html(),
                  '<h2>④ Feature Importance</h2>', abl.to_html(index=False)]
    paths['html'] = os.path.join(out_dir, f'{asof}_optimizer_report.html')
    with open(paths['html'], 'w', encoding='utf-8') as f:
        f.write('\n'.join(html_parts))
    return paths


# ============================================================
# CLI
# ============================================================
def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    ap = argparse.ArgumentParser(description='AI Score Optimizer (보고서 전용)')
    ap.add_argument('--horizon', type=int, default=5, choices=(5, 10))
    ap.add_argument('--rebuild', action='store_true', help='데이터셋 재구축')
    args = ap.parse_args()

    ds = build_dataset(rebuild=args.rebuild)
    if ds.empty:
        print('데이터셋이 비어 있습니다.')
        return
    perf = component_performance(ds, args.horizon)
    sugg = weight_suggestions(perf)
    corr, redundant = correlation_matrix(ds)
    abl = ablation_importance(ds, args.horizon)
    meta = {'rows': len(ds), 'days': ds['date'].nunique()}
    paths = generate_reports(perf, sugg, corr, redundant, abl,
                             horizon=args.horizon, meta=meta)

    print(f"데이터셋 {len(ds)}행 · {meta['days']}거래일 · 기준 {args.horizon}일 수익률")
    print('\n[가중치 제안]')
    for _, r in sugg.iterrows():
        print(f"  {r['component']:8s} {r['current']:>4.0f} → {r['suggested']:>5.1f} "
              f"({r['delta']:+.1f})  {r['reason']}")
    if redundant:
        print('\n[중복 신호]')
        for r in redundant:
            print(f"  {r['suggestion']}")
    print('\n저장:', ', '.join(paths.values()))


if __name__ == '__main__':
    main()
