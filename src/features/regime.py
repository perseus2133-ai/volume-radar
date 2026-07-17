#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Market Regime Analyzer — 시장 상태별 AI Score 성능 분석.

⚠️ AI Score / Backtest 무수정 — 순수 분석 모듈 (import 재사용만).

시장 프록시
-----------
KOSPI 지수 데이터가 없으므로 스냅샷(전 종목 종가)에서 직접 구성한다:
  - ret_median: 전 종목 일별 수익률의 중앙값 (이상치 강건)
  - breadth   : 상승 종목 비율(%)

레짐 분류 (문서화된 규칙, 우선순위 순)
--------------------------------------
롤링 윈도우 W(기본 5거래일)의 누적 중앙값 수익률(cum)과
일별 수익률 표준편차(vol)로:
  1. High Volatility : vol ≥ 2.0%p          (방향 무관, 변동성 우선)
  2. Bull            : cum ≥ +2.0%
  3. Bear            : cum ≤ -2.0%
  4. Sideways        : 그 외
윈도우 미달 초기 거래일은 'Unknown' → 분석에서 제외.

분석
----
② 레짐 × 컴포넌트: 승률·평균수익률·MDD (컴포넌트 '보유' = 만점의 50%+,
   MDD는 이벤트타임 평균 자산곡선 — backtest와 동일 정의)
③ 레짐별 추천(일별 Top-N) 성능 비교
④ 레짐별 추천 가중치 (optimizer 함수 재사용, 적용하지 않음)
⑤ MD / HTML Report
"""
from __future__ import annotations

import os
import logging
import argparse
import datetime

import numpy as np
import pandas as pd

from src.features.backtest import PriceBook
from src.features.optimizer import (component_performance, weight_suggestions,
                                    current_weights, DATASET)

logger = logging.getLogger(__name__)

REGIME_DIR = os.path.join('data', 'regime')

WINDOW = 5
BULL_TH = 2.0        # 윈도우 누적 중앙값 수익률 %
BEAR_TH = -2.0
VOL_TH = 2.0         # 일별 수익률 std %p
MIN_N_WARN = 30      # 이보다 표본 적으면 보고서에 경고

REGIMES = ('Bull', 'Bear', 'Sideways', 'High Volatility')


# ============================================================
# ① 시장 프록시 + 레짐 분류
# ============================================================
def market_daily_returns(book: PriceBook) -> pd.DataFrame:
    """스냅샷에서 일별 시장 수익률 프록시 생성."""
    rows = []
    for prev, day in zip(book.days, book.days[1:]):
        p0, p1 = book._close[prev], book._close[day]
        rets = [(p1[c] / p0[c] - 1) * 100 for c in p1.keys() & p0.keys()
                if p0[c] > 0]
        if not rets:
            continue
        arr = np.array(rets)
        rows.append({'date': day,
                     'ret_median': round(float(np.median(arr)), 3),
                     'ret_mean': round(float(arr.mean()), 3),
                     'breadth': round(float((arr > 0).mean() * 100), 1),
                     'n_stocks': len(arr)})
    return pd.DataFrame(rows)


def classify_regimes(daily: pd.DataFrame, window: int = WINDOW,
                     bull_th: float = BULL_TH, bear_th: float = BEAR_TH,
                     vol_th: float = VOL_TH) -> pd.DataFrame:
    """거래일별 레짐 라벨. 윈도우 미달 초기일은 Unknown."""
    d = daily.copy().reset_index(drop=True)
    labels, cums, vols = [], [], []
    for i in range(len(d)):
        if i + 1 < window:
            labels.append('Unknown')
            cums.append(None)
            vols.append(None)
            continue
        w = d['ret_median'].iloc[i + 1 - window:i + 1]
        cum = float(((1 + w / 100).prod() - 1) * 100)
        vol = float(w.std(ddof=0))
        cums.append(round(cum, 2))
        vols.append(round(vol, 2))
        if vol >= vol_th:
            labels.append('High Volatility')
        elif cum >= bull_th:
            labels.append('Bull')
        elif cum <= bear_th:
            labels.append('Bear')
        else:
            labels.append('Sideways')
    d['cum_w'] = cums
    d['vol_w'] = vols
    d['regime'] = labels
    return d


# ============================================================
# 공통 헬퍼 — 이벤트타임 평균 곡선 MDD (backtest와 동일 정의)
# ============================================================
def _avg_curve_mdd(pairs: list[tuple[str, str]], book: PriceBook,
                   horizon: int) -> float | None:
    paths = []
    for code, day in pairs:
        p = book.path(code, day, horizon)
        if p and p[0] > 0:
            paths.append([v / p[0] for v in p])
    if not paths:
        return None
    curve = [sum(p[k] for p in paths) / len(paths) for k in range(horizon + 1)]
    peak, mdd = curve[0], 0.0
    for v in curve:
        peak = max(peak, v)
        mdd = min(mdd, (v / peak - 1) * 100)
    return round(mdd, 2)


def _join_regime(ds: pd.DataFrame, regimes: pd.DataFrame) -> pd.DataFrame:
    lab = regimes[['date', 'regime']]
    out = ds.merge(lab, on='date', how='left')
    return out[out['regime'].notna() & (out['regime'] != 'Unknown')]


# ============================================================
# ② 레짐 × 컴포넌트 성능
# ============================================================
def regime_component_stats(ds: pd.DataFrame, regimes: pd.DataFrame,
                           book: PriceBook, horizon: int = 5) -> pd.DataFrame:
    ret_col = f'ret_{horizon}d'
    d = _join_regime(ds, regimes)
    comps = sorted(c[:-7] for c in ds.columns if c.endswith('_earned'))
    rows = []
    for regime in REGIMES:
        sub_r = d[d['regime'] == regime]
        for comp in comps:
            e, m, a = f'{comp}_earned', f'{comp}_max', f'{comp}_avail'
            sub = sub_r[(sub_r[a] == 1) & sub_r[ret_col].notna()]
            held = sub[sub[e] >= sub[m] * 0.5]
            if held.empty:
                rows.append({'regime': regime, 'component': comp, 'n': 0,
                             'win_rate': None, 'mean_ret': None, 'mdd': None})
                continue
            mdd = _avg_curve_mdd(list(zip(held['code'], held['date'])), book, horizon)
            rows.append({
                'regime': regime, 'component': comp, 'n': len(held),
                'win_rate': round(float((held[ret_col] > 0).mean() * 100), 1),
                'mean_ret': round(float(held[ret_col].mean()), 2),
                'mdd': mdd,
            })
    return pd.DataFrame(rows)


# ============================================================
# ③ 레짐별 추천(Top-N) 성능
# ============================================================
def regime_recommendation_stats(ds: pd.DataFrame, regimes: pd.DataFrame,
                                book: PriceBook, horizon: int = 5,
                                top_n: int = 20) -> pd.DataFrame:
    ret_col = f'ret_{horizon}d'
    d = _join_regime(ds, regimes)
    sel = (d.sort_values(['date', 'score'], ascending=[True, False])
             .groupby('date').head(top_n))
    rows = []
    for regime in REGIMES:
        sub = sel[(sel['regime'] == regime) & sel[ret_col].notna()]
        days = sel[sel['regime'] == regime]['date'].nunique()
        if sub.empty:
            rows.append({'regime': regime, 'days': days, 'n': 0,
                         'win_rate': None, 'mean_ret': None,
                         'median_ret': None, 'mdd': None})
            continue
        mdd = _avg_curve_mdd(list(zip(sub['code'], sub['date'])), book, horizon)
        rows.append({
            'regime': regime, 'days': days, 'n': len(sub),
            'win_rate': round(float((sub[ret_col] > 0).mean() * 100), 1),
            'mean_ret': round(float(sub[ret_col].mean()), 2),
            'median_ret': round(float(sub[ret_col].median()), 2),
            'mdd': mdd,
        })
    return pd.DataFrame(rows)


# ============================================================
# ④ 레짐별 가중치 제안 (optimizer 재사용 — 적용 안 함)
# ============================================================
def regime_weight_suggestions(ds: pd.DataFrame, regimes: pd.DataFrame,
                              horizon: int = 5) -> dict[str, pd.DataFrame]:
    d = _join_regime(ds, regimes)
    out = {}
    weights = current_weights()
    for regime in REGIMES:
        sub = d[d['regime'] == regime].drop(columns=['regime'])
        if sub.empty:
            continue
        perf = component_performance(sub, horizon)
        out[regime] = weight_suggestions(perf, dict(weights))
    return out


# ============================================================
# ⑤ Report (MD / HTML)
# ============================================================
def generate_regime_report(daily: pd.DataFrame, comp_stats: pd.DataFrame,
                           rec_stats: pd.DataFrame,
                           weights_by_regime: dict[str, pd.DataFrame],
                           out_dir: str = REGIME_DIR,
                           horizon: int = 5) -> dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    asof = datetime.date.today().isoformat()
    paths = {}

    paths['daily_csv'] = os.path.join(out_dir, 'regime_daily.csv')
    daily.to_csv(paths['daily_csv'], index=False, encoding='utf-8-sig')
    paths['comp_csv'] = os.path.join(out_dir, 'regime_component_stats.csv')
    comp_stats.to_csv(paths['comp_csv'], index=False, encoding='utf-8-sig')
    paths['rec_csv'] = os.path.join(out_dir, 'regime_recommendation_stats.csv')
    rec_stats.to_csv(paths['rec_csv'], index=False, encoding='utf-8-sig')

    counts = daily[daily['regime'] != 'Unknown']['regime'].value_counts()
    small = counts[counts * 20 < MIN_N_WARN]

    md = [f'# Market Regime Report ({asof})', '',
          '> ⚠️ 분석 전용 — AI Score/Backtest는 수정되지 않았다. '
          '레짐별 가중치는 **제안**이며 적용되지 않는다.', '',
          f'- 기준 수익률: {horizon}거래일 · 레짐 윈도우: {WINDOW}거래일',
          f'- 분류 규칙: vol≥{VOL_TH}%p → High Volatility, '
          f'누적≥{BULL_TH}% → Bull, ≤{BEAR_TH}% → Bear, 그 외 Sideways', '',
          '## ① 거래일별 레짐', '',
          daily.to_markdown(index=False), '',
          '## ③ 레짐별 추천(Top-20) 성능', '',
          rec_stats.to_markdown(index=False), '',
          '## ② 레짐 × 컴포넌트 성능 (보유 시)', '',
          comp_stats.to_markdown(index=False), '',
          '## ④ 레짐별 추천 가중치 (미적용)', '']
    for regime, sugg in weights_by_regime.items():
        md += [f'### {regime}', '', sugg.to_markdown(index=False), '']
    if len(small):
        md += ['', '> ⚠️ 표본 부족 레짐: '
               + ', '.join(f'{k}({v}거래일)' for k, v in small.items())
               + ' — 통계 신뢰도 낮음, 데이터 축적 후 재평가 권장']
    paths['md'] = os.path.join(out_dir, f'{asof}_regime_report.md')
    with open(paths['md'], 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))

    style = ('<style>body{background:#1A2332;color:#E2E8F0;font-family:sans-serif;'
             'padding:24px;}table{border-collapse:collapse;margin:12px 0;}'
             'th,td{border:1px solid #4A5568;padding:6px 10px;font-size:13px;}'
             'th{background:#232E40;color:#94A3B8;}h1,h2,h3{color:#62EFFF;}'
             'blockquote{color:#FBBF24;border-left:3px solid #FBBF24;padding-left:10px;}'
             '</style>')
    html = [style, f'<h1>Market Regime Report ({asof})</h1>',
            '<blockquote>분석 전용 — 레짐별 가중치는 제안이며 적용되지 않는다.</blockquote>',
            '<h2>① 거래일별 레짐</h2>', daily.to_html(index=False),
            '<h2>③ 레짐별 추천(Top-20) 성능</h2>', rec_stats.to_html(index=False),
            '<h2>② 레짐 × 컴포넌트 성능</h2>', comp_stats.to_html(index=False),
            '<h2>④ 레짐별 추천 가중치 (미적용)</h2>']
    for regime, sugg in weights_by_regime.items():
        html += [f'<h3>{regime}</h3>', sugg.to_html(index=False)]
    paths['html'] = os.path.join(out_dir, f'{asof}_regime_report.html')
    with open(paths['html'], 'w', encoding='utf-8') as f:
        f.write('\n'.join(html))
    return paths


# ============================================================
# CLI
# ============================================================
def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    ap = argparse.ArgumentParser(description='Market Regime Analyzer (분석 전용)')
    ap.add_argument('--horizon', type=int, default=5, choices=(5, 10))
    ap.add_argument('--snap-dir', default=os.path.join('data', 'snapshots'))
    ap.add_argument('--dataset', default=DATASET)
    args = ap.parse_args()

    book = PriceBook(args.snap_dir)
    daily = market_daily_returns(book)
    regimes = classify_regimes(daily)
    if not os.path.exists(args.dataset):
        print('컴포넌트 데이터셋이 없습니다. 먼저 optimizer를 실행하세요:'
              ' python -m src.features.optimizer')
        return
    ds = pd.read_csv(args.dataset, dtype={'code': str})
    ds['code'] = ds['code'].astype(str).str.zfill(6)

    comp_stats = regime_component_stats(ds, regimes, book, args.horizon)
    rec_stats = regime_recommendation_stats(ds, regimes, book, args.horizon)
    weights = regime_weight_suggestions(ds, regimes, args.horizon)
    paths = generate_regime_report(regimes, comp_stats, rec_stats, weights,
                                   horizon=args.horizon)

    print('거래일별 레짐:')
    for _, r in regimes.iterrows():
        mark = {'Bull': '🟢', 'Bear': '🔴', 'Sideways': '⚪',
                'High Volatility': '🟡'}.get(r['regime'], '·')
        extra = (f" · 5일누적 {r['cum_w']:+.2f}% · vol {r['vol_w']}"
                 if pd.notna(r['cum_w']) else '')
        print(f"  {r['date']} {mark} {r['regime']:16s} "
              f"중앙값 {r['ret_median']:+.2f}% · 상승비율 {r['breadth']}%{extra}")
    print('\n레짐별 추천(Top-20) 성능:')
    for _, r in rec_stats.iterrows():
        if r['n']:
            print(f"  {r['regime']:16s} n={r['n']:<3} 승률 {r['win_rate']}% "
                  f"평균 {r['mean_ret']:+.2f}% MDD {r['mdd']}%")
        else:
            print(f"  {r['regime']:16s} 표본 없음")
    print('\n저장:', ', '.join(paths.values()))


if __name__ == '__main__':
    main()
