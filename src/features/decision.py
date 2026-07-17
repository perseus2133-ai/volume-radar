#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Decision Engine — 5개 엔진 통합 최종 투자 의사결정.

⚠️ 기존 엔진(AI Score/Backtest/Optimizer/Regime/Confidence) 무수정.
   Confidence 엔진의 종목 분석 결과(analyze_latest)를 입력으로 받아
   최종 판단만 생성하는 신규 레이어.

Decision Score (0~100) — 문서화된 가중 결합
--------------------------------------------
  base = 0.35 × AI Score
       + 0.30 × Confidence
       + 0.20 × Historical Performance   (유사사례 승률·평균수익 합성)
       + 0.15 × Risk/Reward 점수화
  최종 = clip(base + 국면 조정, 0, 100)
  국면 조정: Bull +5 / Sideways 0 / Bear -5 / High Volatility -8 / Unknown -3

Historical Performance 점수화: 0.5×승률 + 0.5×clip(50 + 2×평균수익률, 0, 100)
  (유사 사례 없으면 50 = 중립)
R/R 점수화: ≥2→100, 1→75, 0→50, -1→25, ≤-2→0 (구간 선형보간, None→50)

Decision 사다리 (DScore 기준, 상위부터)
--------------------------------------
  STRONG BUY : DScore ≥ 75  AND R/R ≥ 1
  BUY        : DScore ≥ 65
  WATCH      : DScore ≥ 55
  HOLD       : DScore ≥ 45
  SELL       : DScore ≥ 35
  AVOID      : DScore < 35

가드레일(신뢰도·기대수익 조건은 여기로 일원화 — 하향만 가능):
  - Confidence < 65 → STRONG BUY 강등 (BUY)
  - Confidence < 50 → BUY 강등 (WATCH)   ※ 매수 판단은 C등급 이상만
  - 기대수익 < -5% → HOLD 이상 강등 (SELL)

Risk Level: 예상낙폭·R/R·국면·신뢰도로 LOW/MEDIUM/HIGH/VERY HIGH
"""
from __future__ import annotations

import os
import json
import logging
import argparse
import datetime

import numpy as np

logger = logging.getLogger(__name__)

DECISION_DIR = os.path.join('data', 'decision')

W_AI, W_CONF, W_HIST, W_RR = 0.35, 0.30, 0.20, 0.15
REGIME_ADJ = {'Bull': 5.0, 'Sideways': 0.0, 'Bear': -5.0,
              'High Volatility': -8.0, 'Unknown': -3.0}

DECISIONS = ('STRONG BUY', 'BUY', 'WATCH', 'HOLD', 'SELL', 'AVOID')


# ============================================================
# 점수화 헬퍼
# ============================================================
def hist_performance_score(similar: dict) -> tuple[float, str]:
    """유사 과거 사례 → 0~100. 사례 없으면 중립 50."""
    if not similar or not similar.get('n'):
        return 50.0, '유사 사례 없음 → 중립(50)'
    win = float(similar.get('win_rate') or 0)
    avg = float(similar.get('avg_ret') or 0)
    ret_part = float(np.clip(50 + 2 * avg, 0, 100))
    score = 0.5 * win + 0.5 * ret_part
    return round(score, 1), (f"유사 {similar['n']}건 승률 {win:.0f}% · "
                             f"평균 {avg:+.1f}%")


def rr_score(rr: float | None) -> tuple[float, str]:
    """Risk/Reward → 0~100 (구간 선형보간)."""
    if rr is None:
        return 50.0, 'R/R 산출 불가 → 중립(50)'
    pts_table = [(-2.0, 0.0), (-1.0, 25.0), (0.0, 50.0), (1.0, 75.0), (2.0, 100.0)]
    x = float(np.clip(rr, -2.0, 2.0))
    for (x0, y0), (x1, y1) in zip(pts_table, pts_table[1:]):
        if x0 <= x <= x1:
            y = y0 + (y1 - y0) * (x - x0) / (x1 - x0)
            return round(y, 1), f'R/R {rr:+.2f}'
    return 50.0, f'R/R {rr:+.2f}'


def risk_level(analysis: dict) -> str:
    exp = analysis.get('expected') or {}
    conf = analysis.get('confidence') or {}
    dd = exp.get('expected_drawdown')
    rr = exp.get('risk_reward')
    er = exp.get('expected_return')
    regime = analysis.get('regime', 'Unknown')

    very_high = ((dd is not None and dd <= -20)
                 or (conf.get('grade') == 'F' and er is not None and er < 0))
    if very_high:
        return 'VERY HIGH'
    if (dd is not None and dd <= -10) or regime == 'High Volatility':
        return 'HIGH'
    if (dd is not None and dd <= -5) or (rr is not None and rr < 1):
        return 'MEDIUM'
    return 'LOW'


def decision_grade(dscore: float) -> str:
    if dscore >= 85: return 'A+'
    if dscore >= 75: return 'A'
    if dscore >= 65: return 'B'
    if dscore >= 55: return 'C'
    if dscore >= 45: return 'D'
    return 'F'


# ============================================================
# 핵심 — decide() (순수 함수)
# ============================================================
def decide(analysis: dict) -> dict:
    """Confidence 분석 결과 1건 → 최종 의사결정."""
    ai = float(analysis['score'])
    conf = float(analysis['confidence']['total'])
    conf_grade = analysis['confidence']['grade']
    regime = analysis.get('regime', 'Unknown')
    exp = analysis.get('expected') or {}
    er = exp.get('expected_return')
    rr = exp.get('risk_reward')

    hist_pts, hist_note = hist_performance_score(analysis.get('similar') or {})
    rr_pts, rr_note = rr_score(rr)
    adj = REGIME_ADJ.get(regime, -3.0)

    base = W_AI * ai + W_CONF * conf + W_HIST * hist_pts + W_RR * rr_pts
    dscore = round(float(np.clip(base + adj, 0, 100)), 1)

    # 사다리 (DScore 기준)
    if dscore >= 75 and (rr is not None and rr >= 1):
        rec = 'STRONG BUY'
    elif dscore >= 65:
        rec = 'BUY'
    elif dscore >= 55:
        rec = 'WATCH'
    elif dscore >= 45:
        rec = 'HOLD'
    elif dscore >= 35:
        rec = 'SELL'
    else:
        rec = 'AVOID'

    # 가드레일 — 신뢰도/기대수익 조건 일원화 (하향만 가능)
    guardrails = []
    if rec == 'STRONG BUY' and conf < 65:
        rec = 'BUY'
        guardrails.append(f'신뢰도 {conf:.0f} < 65 → STRONG BUY 강등')
    if rec == 'BUY' and conf < 50:
        rec = 'WATCH'
        guardrails.append(f'신뢰도 {conf:.0f} < 50 → 매수 금지 (WATCH 강등)')
    if er is not None and er < -5 and rec in ('STRONG BUY', 'BUY', 'WATCH', 'HOLD'):
        rec = 'SELL'
        guardrails.append(f'기대수익 {er:+.1f}% < -5% → SELL 강등')

    risk = risk_level(analysis)

    # Reason Summary (한 문단 자동 생성)
    top_reasons = (analysis.get('reasons') or [])[:3]
    summary = (f"AI Score {ai:.0f}점({', '.join(top_reasons) if top_reasons else '근거 없음'}), "
               f"신뢰도 {conf:.0f}({conf_grade}), {hist_note}, {rr_note}, "
               f"국면 {regime}({adj:+.0f}) → Decision {dscore:.0f}점 = {rec}.")
    if guardrails:
        summary += ' 가드레일: ' + ' / '.join(guardrails)

    return {
        'date': analysis['date'], 'code': analysis['code'],
        'name': analysis['name'],
        'ai_score': ai, 'confidence': conf, 'confidence_grade': conf_grade,
        'regime': regime,
        'inputs': {'ai': round(W_AI * ai, 1), 'confidence': round(W_CONF * conf, 1),
                   'historical': round(W_HIST * hist_pts, 1),
                   'risk_reward': round(W_RR * rr_pts, 1), 'regime_adj': adj},
        'decision_score': dscore,
        'decision_grade': decision_grade(dscore),
        'recommendation': rec,
        'risk_level': risk,
        'reason_summary': summary,
        'guardrails': guardrails,
        'cautions': analysis.get('cautions', []),
    }


# ============================================================
# Report (JSON / MD / HTML)
# ============================================================
_REC_COLOR = {'STRONG BUY': '#10B981', 'BUY': '#34D399', 'WATCH': '#62EFFF',
              'HOLD': '#94A3B8', 'SELL': '#F59E0B', 'AVOID': '#F87171'}


def generate_reports(decisions: list[dict], out_dir: str = DECISION_DIR) -> dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    asof = datetime.date.today().isoformat()
    paths = {}

    paths['json'] = os.path.join(out_dir, f'{asof}_decisions.json')
    with open(paths['json'], 'w', encoding='utf-8') as f:
        json.dump({'asof': asof, 'decisions': decisions}, f,
                  ensure_ascii=False, indent=1)

    md = [f'# Decision Report ({asof})', '',
          '> 5개 엔진(AI Score·Backtest·Optimizer·Regime·Confidence) 통합 판단.',
          '> 기존 엔진 무수정 — 이 보고서는 자동 생성된 참고 자료다.', '',
          '| 종목 | DScore | 등급 | 판단 | 리스크 | AI | 신뢰도 | 국면 |',
          '|---|---|---|---|---|---|---|---|']
    for d in decisions:
        md.append(f"| {d['name']} | {d['decision_score']:.0f} | {d['decision_grade']} "
                  f"| **{d['recommendation']}** | {d['risk_level']} "
                  f"| {d['ai_score']:.0f} | {d['confidence']:.0f}({d['confidence_grade']}) "
                  f"| {d['regime']} |")
    md += ['', '## 종목별 판단 근거', '']
    for d in decisions:
        md += [f"### {d['name']} ({d['code']}) — {d['recommendation']}",
               '', d['reason_summary'], '']
        if d['cautions']:
            md += ['**주의사항**'] + [f'- {c}' for c in d['cautions']] + ['']
    paths['md'] = os.path.join(out_dir, f'{asof}_decision_report.md')
    with open(paths['md'], 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))

    style = ('<style>body{background:#1A2332;color:#E2E8F0;font-family:sans-serif;'
             'padding:24px;max-width:960px;margin:auto;}table{border-collapse:collapse;'
             'width:100%;margin:12px 0;}th,td{border:1px solid #4A5568;padding:7px 10px;'
             'font-size:13px;}th{background:#232E40;color:#94A3B8;}h1,h3{color:#62EFFF;}'
             '.rec{font-weight:800;}.caution{color:#FBBF24;font-size:12px;}</style>')
    rows = ''
    for d in decisions:
        color = _REC_COLOR.get(d['recommendation'], '#E2E8F0')
        rows += (f"<tr><td>{d['name']}</td><td>{d['decision_score']:.0f}</td>"
                 f"<td>{d['decision_grade']}</td>"
                 f"<td class='rec' style='color:{color}'>{d['recommendation']}</td>"
                 f"<td>{d['risk_level']}</td><td>{d['ai_score']:.0f}</td>"
                 f"<td>{d['confidence']:.0f}({d['confidence_grade']})</td>"
                 f"<td>{d['regime']}</td></tr>")
    html = [style, f'<h1>Decision Report ({asof})</h1>',
            '<table><tr><th>종목</th><th>DScore</th><th>등급</th><th>판단</th>'
            '<th>리스크</th><th>AI</th><th>신뢰도</th><th>국면</th></tr>',
            rows, '</table>']
    for d in decisions:
        html.append(f"<h3>{d['name']} — {d['recommendation']}</h3>"
                    f"<p>{d['reason_summary']}</p>")
        if d['cautions']:
            html.append('<p class="caution">⚠️ ' + '<br>⚠️ '.join(d['cautions']) + '</p>')
    paths['html'] = os.path.join(out_dir, f'{asof}_decision_report.html')
    with open(paths['html'], 'w', encoding='utf-8') as f:
        f.write('\n'.join(html))
    return paths


# ============================================================
# CLI 파이프라인
# ============================================================
def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    ap = argparse.ArgumentParser(description='Decision Engine (기존 엔진 무수정)')
    ap.add_argument('--top-n', type=int, default=10)
    ap.add_argument('--horizon', type=int, default=5, choices=(5, 10))
    args = ap.parse_args()

    from src.features.optimizer import DATASET
    if not os.path.exists(DATASET):
        print('컴포넌트 데이터셋 없음 — 먼저 python -m src.features.optimizer 실행')
        return
    from src.features.confidence import load_context, analyze_latest
    ctx = load_context(horizon=args.horizon)
    analyses = analyze_latest(ctx, top_n=args.top_n, horizon=args.horizon)
    decisions = [decide(a) for a in analyses]
    decisions.sort(key=lambda d: -d['decision_score'])
    paths = generate_reports(decisions)

    print(f'최종 판단 {len(decisions)}종목:\n')
    for d in decisions:
        print(f"  {d['recommendation']:10s} {d['name']:12s} DScore {d['decision_score']:5.1f} "
              f"({d['decision_grade']}) · 리스크 {d['risk_level']:9s} "
              f"· AI {d['ai_score']:.0f} · 신뢰도 {d['confidence']:.0f}({d['confidence_grade']})")
    print('\n저장:', ', '.join(paths.values()))


if __name__ == '__main__':
    main()
