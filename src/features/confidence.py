#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Confidence Engine — 추천 신뢰도(0~100) + 설명가능성 리포트.

⚠️ AI Score / Backtest / Optimizer / Regime 무수정 — 산출물을 '읽어서'
   통합하는 신규 모듈. 새로운 투자 지표를 추가하지 않는다.

"왜 추천하는가"(reasons)는 이미 있다. 이 모듈은
"얼마나 믿을 수 있는가"를 5개 요인(각 0~20)으로 수치화한다:

  ① 표본 수        : 유사 과거 사례가 몇 건이나 뒷받침하는가
  ② 시장 국면 일치  : 현재 레짐에서 이 점수대가 역사적으로 통했는가
  ③ 점수 구간 성과  : (레짐 무관) 이 점수대의 역사적 승률
  ④ Feature 안정성 : 이 종목이 '보유한' 신호들이 역사적으로 유리했는가
  ⑤ 백테스트 일관성 : 전략의 일별 성과가 얼마나 일관적인가 (t-통계 유사)

원칙: 데이터가 부족하면 점수를 '낮게' 준다 — 불확실성을 숨기지 않는다.
현재처럼 급락장 15거래일뿐인 데이터에서는 낮은 신뢰도가 정답이다.

Historical Similarity: 컴포넌트 획득률 벡터(0~1) + RSI + 순위 백분위로
과거(date < 추천일) 사례와의 유클리드 거리 → 최근접 N건의
평균수익률/승률/최대낙폭 → Expected Metrics(기대수익·예상낙폭·R/R).
"""
from __future__ import annotations

import os
import json
import logging
import argparse
import datetime
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

from src.features.backtest import PriceBook
from src.features.optimizer import DATASET, component_performance
from src.features.regime import (market_daily_returns, classify_regimes,
                                 _avg_curve_mdd)

logger = logging.getLogger(__name__)

CONF_DIR = os.path.join('data', 'confidence')

SCORE_BANDS = (95, 90, 85, 80, 75)
N_SIMILAR = 20
MIN_SHARED_DIMS = 4
FACTOR_MAX = 20.0


# ============================================================
# 데이터 컨텍스트 (모든 엔진 산출물 read-only 로드)
# ============================================================
@dataclass
class Context:
    ds: pd.DataFrame              # optimizer 컴포넌트 데이터셋
    book: PriceBook
    regimes: pd.DataFrame         # date → regime
    perf: pd.DataFrame            # 컴포넌트 성능 (lift 테이블)
    comps: list[str]


def load_context(dataset_path: str = DATASET,
                 snap_dir: str = os.path.join('data', 'snapshots'),
                 horizon: int = 5) -> Context:
    ds = pd.read_csv(dataset_path, dtype={'code': str})
    ds['code'] = ds['code'].astype(str).str.zfill(6)
    book = PriceBook(snap_dir)
    regimes = classify_regimes(market_daily_returns(book))
    perf = component_performance(ds, horizon)
    comps = sorted(c[:-7] for c in ds.columns if c.endswith('_earned'))
    return Context(ds=ds, book=book, regimes=regimes, perf=perf, comps=comps)


def _regime_of(ctx: Context, date: str) -> str:
    row = ctx.regimes[ctx.regimes['date'] == date]
    return row.iloc[0]['regime'] if not row.empty else 'Unknown'


def _band_lo(score: float) -> float:
    for b in SCORE_BANDS:
        if score >= b:
            return float(b)
    return 0.0


# ============================================================
# 2. Historical Similarity
# ============================================================
def find_similar(ctx: Context, target: pd.Series, n: int = N_SIMILAR,
                 horizon: int = 5) -> pd.DataFrame:
    """target(데이터셋 행)과 유사한 '과거'(date < target.date) 사례 N건."""
    ret_col = f'ret_{horizon}d'
    hist = ctx.ds[(ctx.ds['date'] < target['date'])
                  & ctx.ds[ret_col].notna()].copy()
    if hist.empty:
        return hist

    dims: list[tuple[str, float, float]] = []   # (컬럼, target값, 스케일)
    for comp in ctx.comps:
        if target.get(f'{comp}_avail', 0) == 1:
            ratio = float(target[f'{comp}_earned']) / max(float(target[f'{comp}_max']), 1e-9)
            dims.append((comp, ratio, 1.0))
    rsi_t = pd.to_numeric(target.get('rsi'), errors='coerce')
    if pd.notna(rsi_t):
        dims.append(('__rsi', float(rsi_t) / 100.0, 1.0))
    dims.append(('__rank', float(target['rank']) / 200.0, 1.0))
    if len(dims) < MIN_SHARED_DIMS:
        return hist.iloc[0:0]

    d2 = np.zeros(len(hist))
    used = np.zeros(len(hist))
    for key, tval, _ in dims:
        if key == '__rsi':
            cand = pd.to_numeric(hist['rsi'], errors='coerce') / 100.0
            mask = cand.notna().to_numpy()
            vals = cand.fillna(0).to_numpy()
        elif key == '__rank':
            vals = (hist['rank'] / 200.0).to_numpy()
            mask = np.ones(len(hist), dtype=bool)
        else:
            mask = (hist[f'{key}_avail'] == 1).to_numpy()
            vals = (hist[f'{key}_earned'] /
                    hist[f'{key}_max'].clip(lower=1e-9)).to_numpy()
        diff = np.where(mask, vals - tval, 0.0)
        d2 += diff * diff
        used += mask.astype(float)
    ok = used >= MIN_SHARED_DIMS
    hist = hist[ok].copy()
    hist['distance'] = np.sqrt(d2[ok] / used[ok])
    return hist.nsmallest(n, 'distance')


def similarity_stats(ctx: Context, similar: pd.DataFrame,
                     horizon: int = 5) -> dict:
    ret_col = f'ret_{horizon}d'
    if similar.empty:
        return {'n': 0, 'avg_ret': None, 'win_rate': None, 'mdd': None, 'cases': []}
    rets = similar[ret_col]
    mdd = _avg_curve_mdd(list(zip(similar['code'], similar['date'])),
                         ctx.book, horizon)
    cases = [{'date': r['date'], 'code': r['code'], 'name': r['name'],
              'score': r['score'], 'ret': r[ret_col],
              'distance': round(float(r['distance']), 3)}
             for _, r in similar.head(5).iterrows()]
    return {'n': len(similar),
            'avg_ret': round(float(rets.mean()), 2),
            'win_rate': round(float((rets > 0).mean() * 100), 1),
            'mdd': mdd,
            'worst': round(float(rets.min()), 2),
            'best': round(float(rets.max()), 2),
            'cases': cases}


# ============================================================
# 3. Expected Metrics
# ============================================================
def expected_metrics(sim: dict) -> dict:
    if not sim['n']:
        return {'expected_return': None, 'expected_drawdown': None,
                'risk_reward': None}
    er = sim['avg_ret']
    dd = sim['mdd'] if sim['mdd'] is not None else sim.get('worst')
    rr = None
    if er is not None and dd not in (None, 0):
        rr = round(er / abs(dd), 2)
    return {'expected_return': er, 'expected_drawdown': dd, 'risk_reward': rr}


# ============================================================
# 1. Confidence Score — 5요인 × 20점
# ============================================================
def _scale(value: float | None, table: list[tuple[float, float]],
           default: float) -> float:
    """value ≥ 임계 → 점수 (내림차순 테이블)."""
    if value is None:
        return default
    for th, pts in table:
        if value >= th:
            return pts
    return table[-1][1]


@dataclass
class ConfidenceResult:
    total: float
    grade: str
    factors: dict[str, dict] = field(default_factory=dict)
    cautions: list[str] = field(default_factory=list)


def confidence_score(ctx: Context, target: pd.Series, sim: dict,
                     horizon: int = 5) -> ConfidenceResult:
    ret_col = f'ret_{horizon}d'
    hist = ctx.ds[(ctx.ds['date'] < target['date']) & ctx.ds[ret_col].notna()]
    factors: dict[str, dict] = {}
    cautions: list[str] = []

    # ① 표본 수 (유사 사례 건수)
    n = sim['n']
    pts = _scale(n, [(50, 20), (30, 15), (15, 10), (5, 5), (0, 2)], 2)
    factors['표본수'] = {'points': pts, 'max': FACTOR_MAX,
                       'note': f'유사 과거 사례 {n}건'}
    if n < 10:
        cautions.append(f'유사 사례 {n}건뿐 — 통계적 신뢰 낮음')

    # ② 시장 국면 일치도
    r_now = _regime_of(ctx, target['date'])
    band = _band_lo(target['score'])
    joined = hist.merge(ctx.regimes[['date', 'regime']], on='date', how='left')
    in_regime = joined[(joined['regime'] == r_now) & (joined['score'] >= band)]
    if r_now == 'Unknown' or len(in_regime) < 5:
        factors['국면일치'] = {'points': 5.0, 'max': FACTOR_MAX,
                           'note': f'현재 국면({r_now})의 동일 점수대 표본 부족'}
        cautions.append(f'현재 시장 국면({r_now})에서의 검증 데이터 부족')
    else:
        win = float((in_regime[ret_col] > 0).mean() * 100)
        pts = _scale(win, [(60, 20), (50, 14), (40, 8), (0, 4)], 4)
        factors['국면일치'] = {'points': pts, 'max': FACTOR_MAX,
                           'note': f'{r_now} 국면 동일 점수대 승률 {win:.0f}% (n={len(in_regime)})'}

    # ③ 점수 구간 성과 (레짐 무관)
    in_band = hist[hist['score'] >= band]
    if len(in_band) < 5:
        factors['구간성과'] = {'points': 5.0, 'max': FACTOR_MAX,
                           'note': f'{band:.0f}+ 구간 표본 부족'}
    else:
        win = float((in_band[ret_col] > 0).mean() * 100)
        pts = _scale(win, [(60, 20), (50, 14), (40, 8), (0, 4)], 4)
        factors['구간성과'] = {'points': pts, 'max': FACTOR_MAX,
                           'note': f'점수 {band:.0f}+ 역사적 승률 {win:.0f}% (n={len(in_band)})'}

    # ④ Feature 안정성 (보유 신호들의 역사적 lift)
    held = [c for c in ctx.comps
            if target.get(f'{c}_avail', 0) == 1
            and float(target[f'{c}_earned']) >= float(target[f'{c}_max']) * 0.5]
    perf_idx = ctx.perf.set_index('component')
    lifts = [float(perf_idx.loc[c, 'win_lift']) for c in held
             if c in perf_idx.index and pd.notna(perf_idx.loc[c, 'win_lift'])]
    if not lifts:
        factors['신호안정성'] = {'points': 8.0, 'max': FACTOR_MAX,
                            'note': '보유 신호의 성과 데이터 없음'}
    else:
        avg_lift = float(np.mean(lifts))
        pts = _scale(avg_lift, [(5, 20), (0, 12), (-5, 8), (-100, 4)], 8)
        factors['신호안정성'] = {'points': pts, 'max': FACTOR_MAX,
                            'note': f'보유 신호 {len(held)}개 평균 승률기여 {avg_lift:+.1f}%p'}
        bad = [c for c in held if c in perf_idx.index
               and pd.notna(perf_idx.loc[c, 'win_lift'])
               and float(perf_idx.loc[c, 'win_lift']) <= -5]
        if bad:
            cautions.append('역사적으로 불리했던 신호 보유: ' + ', '.join(bad))

    # ⑤ 백테스트 일관성 (일별 Top-20 평균수익률의 t-통계 유사값)
    daily = (hist.sort_values(['date', 'score'], ascending=[True, False])
                 .groupby('date').head(20).groupby('date')[ret_col].mean())
    if len(daily) < 3:
        factors['일관성'] = {'points': 5.0, 'max': FACTOR_MAX,
                          'note': '백테스트 거래일 부족'}
    else:
        t = float(daily.mean() / (daily.std(ddof=0) + 1e-9))
        pts = _scale(t, [(1.0, 20), (0.5, 16), (0.0, 12), (-0.5, 8), (-100, 4)], 4)
        factors['일관성'] = {'points': pts, 'max': FACTOR_MAX,
                          'note': f'일별 성과 t≈{t:.2f} ({len(daily)}거래일)'}
        if t < 0:
            cautions.append('최근 백테스트 구간에서 전략 평균 성과가 음(-)')

    total = round(sum(f['points'] for f in factors.values()), 1)
    grade = ('A' if total >= 80 else 'B' if total >= 65 else
             'C' if total >= 50 else 'D' if total >= 35 else 'F')
    rsi_t = pd.to_numeric(target.get('rsi'), errors='coerce')
    if pd.notna(rsi_t) and rsi_t >= 70:
        cautions.append(f'RSI {rsi_t:.0f} 과열권 — 단기 조정 위험')
    if total < 50:
        cautions.append('종합 신뢰도 낮음 — 참고용으로만 활용')
    return ConfidenceResult(total=total, grade=grade,
                            factors=factors, cautions=cautions)


# ============================================================
# 4. Explainability — 종목 단위 통합
# ============================================================
def analyze_stock(ctx: Context, target: pd.Series, reasons: list[str],
                  horizon: int = 5) -> dict:
    similar = find_similar(ctx, target, horizon=horizon)
    sim = similarity_stats(ctx, similar, horizon)
    conf = confidence_score(ctx, target, sim, horizon)
    exp = expected_metrics(sim)
    return {
        'date': target['date'], 'code': target['code'], 'name': target['name'],
        'score': float(target['score']), 'rank': int(target['rank']),
        'regime': _regime_of(ctx, target['date']),
        'reasons': reasons,
        'confidence': {'total': conf.total, 'grade': conf.grade,
                       'factors': conf.factors},
        'similar': sim,
        'expected': exp,
        'cautions': conf.cautions,
    }


def analyze_latest(ctx: Context, top_n: int = 10, horizon: int = 5,
                   scores_dir: str = os.path.join('data', 'scores')) -> list[dict]:
    latest = ctx.ds['date'].max()
    day = ctx.ds[ctx.ds['date'] == latest].sort_values('score', ascending=False)
    reasons_map: dict[str, list[str]] = {}
    sf = os.path.join(scores_dir, f'{latest}.json')
    if os.path.exists(sf):
        with open(sf, encoding='utf-8') as f:
            for s in json.load(f):
                reasons_map[str(s['code']).zfill(6)] = s.get('reasons', [])
    out = []
    for _, row in day.head(top_n).iterrows():
        out.append(analyze_stock(ctx, row, reasons_map.get(row['code'], []),
                                 horizon))
    return out


# ============================================================
# 5. Report (JSON / MD / HTML)
# ============================================================
def generate_reports(results: list[dict], out_dir: str = CONF_DIR,
                     horizon: int = 5) -> dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    asof = datetime.date.today().isoformat()
    paths = {}

    paths['json'] = os.path.join(out_dir, f'{asof}_confidence.json')
    with open(paths['json'], 'w', encoding='utf-8') as f:
        json.dump({'asof': asof, 'horizon': horizon, 'results': results},
                  f, ensure_ascii=False, indent=1)

    def stock_md(r: dict) -> list[str]:
        c = r['confidence']
        lines = [f"## {r['name']} ({r['code']}) — AI Score {r['score']:.1f} · "
                 f"신뢰도 {c['total']:.0f}/100 ({c['grade']})", '',
                 f"- 추천일: {r['date']} · 순위 {r['rank']}위 · 시장 국면 **{r['regime']}**",
                 f"- 추천 근거: {' · '.join(r['reasons']) or '-'}", '',
                 '| 신뢰도 요인 | 점수 | 설명 |', '|---|---|---|']
        for name, f_ in c['factors'].items():
            lines.append(f"| {name} | {f_['points']:.0f}/{f_['max']:.0f} | {f_['note']} |")
        s = r['similar']
        if s['n']:
            lines += ['', f"**유사 과거 사례 {s['n']}건**: 평균 {s['avg_ret']:+.2f}% · "
                      f"승률 {s['win_rate']}% · 최대낙폭 {s['mdd']}% "
                      f"(최악 {s['worst']:+.1f}% / 최선 {s['best']:+.1f}%)", '',
                      '| 유사사례 | 점수 | 수익률 | 거리 |', '|---|---|---|---|']
            for case in s['cases']:
                lines.append(f"| {case['date']} {case['name']} | {case['score']:.1f} "
                             f"| {case['ret']:+.2f}% | {case['distance']} |")
            e = r['expected']
            lines += ['', f"**기대치**: 수익 {e['expected_return']:+.2f}% · "
                      f"예상낙폭 {e['expected_drawdown']}% · "
                      f"R/R {e['risk_reward'] if e['risk_reward'] is not None else '-'}"]
        else:
            lines += ['', '유사 과거 사례 없음']
        if r['cautions']:
            lines += ['', '**⚠️ 주의사항**'] + [f'- {x}' for x in r['cautions']]
        return lines + ['', '---', '']

    md = [f'# 추천 신뢰도 리포트 ({asof})', '',
          f'> 기준 수익률: {horizon}거래일 · 신뢰도 = 표본수+국면일치+구간성과'
          f'+신호안정성+일관성 (각 20점)', '',
          '> ⚠️ AI Score/Backtest 무수정 — 기존 엔진 산출물의 통합 분석이다.', '']
    for r in results:
        md += stock_md(r)
    paths['md'] = os.path.join(out_dir, f'{asof}_confidence_report.md')
    with open(paths['md'], 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))

    style = ('<style>body{background:#1A2332;color:#E2E8F0;font-family:sans-serif;'
             'padding:24px;max-width:960px;margin:auto;}table{border-collapse:collapse;'
             'margin:10px 0;width:100%;}th,td{border:1px solid #4A5568;padding:6px 10px;'
             'font-size:13px;}th{background:#232E40;color:#94A3B8;}h1,h2{color:#62EFFF;}'
             '.grade{font-size:1.2em;font-weight:800;}.caution{color:#FBBF24;}'
             'hr{border-color:#4A5568;}</style>')
    html = [style, f'<h1>추천 신뢰도 리포트 ({asof})</h1>']
    for r in results:
        c = r['confidence']
        html.append(f"<h2>{r['name']} ({r['code']}) — Score {r['score']:.1f} · "
                    f"<span class='grade'>신뢰도 {c['total']:.0f} ({c['grade']})</span></h2>")
        html.append(f"<p>국면 {r['regime']} · 근거: {' · '.join(r['reasons']) or '-'}</p>")
        rows = ''.join(f"<tr><td>{k}</td><td>{v['points']:.0f}/{v['max']:.0f}</td>"
                       f"<td>{v['note']}</td></tr>" for k, v in c['factors'].items())
        html.append(f'<table><tr><th>요인</th><th>점수</th><th>설명</th></tr>{rows}</table>')
        if r['cautions']:
            html.append('<p class="caution">⚠️ ' + '<br>⚠️ '.join(r['cautions']) + '</p>')
        html.append('<hr>')
    paths['html'] = os.path.join(out_dir, f'{asof}_confidence_report.html')
    with open(paths['html'], 'w', encoding='utf-8') as f:
        f.write('\n'.join(html))
    return paths


# ============================================================
# CLI
# ============================================================
def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    ap = argparse.ArgumentParser(description='추천 신뢰도 엔진 (분석 전용)')
    ap.add_argument('--top-n', type=int, default=10)
    ap.add_argument('--horizon', type=int, default=5, choices=(5, 10))
    args = ap.parse_args()

    if not os.path.exists(DATASET):
        print('컴포넌트 데이터셋 없음 — 먼저 python -m src.features.optimizer 실행')
        return
    ctx = load_context(horizon=args.horizon)
    results = analyze_latest(ctx, top_n=args.top_n, horizon=args.horizon)
    paths = generate_reports(results, horizon=args.horizon)

    print(f"최신 추천 {len(results)}종목 신뢰도 (기준 {args.horizon}일):\n")
    for r in results:
        c = r['confidence']
        e = r['expected']
        exp = (f"기대 {e['expected_return']:+.1f}% / 낙폭 {e['expected_drawdown']}% "
               f"/ R/R {e['risk_reward']}" if e['expected_return'] is not None
               else '유사 사례 없음')
        print(f"  {r['name']:12s} Score {r['score']:5.1f} → 신뢰도 "
              f"{c['total']:4.0f} ({c['grade']}) · 유사 {r['similar']['n']:>2}건 · {exp}")
    print('\n저장:', ', '.join(paths.values()))


if __name__ == '__main__':
    main()
