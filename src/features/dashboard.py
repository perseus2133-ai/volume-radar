#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Daily AI Dashboard & Watchlist Engine — "오늘 무엇을 해야 하는가".

⚠️ 6개 엔진(AI Score/Backtest/Optimizer/Regime/Confidence/Decision)
   무수정 — 공개 함수 호출과 산출물 소비만 한다. 새 투자 알고리즘 없음.

① Daily Dashboard : 오늘 추천을 Decision 순서로 정렬해 한 화면에
② Watchlist Engine: "곧 BUY가 될" Tomorrow Candidates 자동 탐색
   - 최근 K일의 Decision 이력을 재계산(결정적·look-ahead 없음)해
     WATCH 연속 / 신뢰도·AI Score 상승 / 판단 개선 / 리스크 감소 /
     시장 국면 개선 신호를 집계. 신호 2개 이상 → 후보.
③ Daily Summary  : AI Morning Brief (시장 상태·판단 집계·강한/위험
   섹터·Top5 추천/주의)
④ Report         : data/dashboard/daily_dashboard.{json,md,html}
"""
from __future__ import annotations

import os
import json
import logging
import argparse
import datetime

import pandas as pd

from src.features.optimizer import DATASET
from src.features.confidence import load_context, analyze_stock, Context
from src.features.decision import decide

logger = logging.getLogger(__name__)

DASH_DIR = os.path.join('data', 'dashboard')
SCORES_DIR = os.path.join('data', 'scores')

DECISION_ORDER = ['STRONG BUY', 'BUY', 'WATCH', 'HOLD', 'SELL', 'AVOID']
DECISION_EMOJI = {'STRONG BUY': '🟢', 'BUY': '🟢', 'WATCH': '🟡',
                  'HOLD': '⚪', 'SELL': '🔴', 'AVOID': '⚫'}
RISK_ORDER = ['LOW', 'MEDIUM', 'HIGH', 'VERY HIGH']
REGIME_ORDER = ['Bear', 'High Volatility', 'Sideways', 'Bull']   # 나쁨→좋음

HISTORY_DAYS = 5          # Watchlist가 보는 이력 길이
MIN_SIGNALS = 2           # Tomorrow Candidate 최소 신호 수
RISE_TH = 3.0             # 신뢰도/AI Score '상승' 판정 최소 폭


def _d_rank(rec: str) -> int:
    """판단 순위 — 클수록 좋음 (AVOID=0 … STRONG BUY=5)."""
    return len(DECISION_ORDER) - 1 - DECISION_ORDER.index(rec)


# ============================================================
# Decision 이력 재계산 (엔진 공개 함수 재사용, look-ahead 없음)
# ============================================================
def _reasons_map(day: str) -> dict[str, list[str]]:
    path = os.path.join(SCORES_DIR, f'{day}.json')
    if not os.path.exists(path):
        return {}
    with open(path, encoding='utf-8') as f:
        return {str(s['code']).zfill(6): s.get('reasons', [])
                for s in json.load(f)}


def build_daily_decisions(ctx: Context, days: list[str],
                          top_n: int = 10, horizon: int = 5
                          ) -> dict[str, list[dict]]:
    """각 거래일의 상위 top_n 종목에 Confidence→Decision 파이프라인 적용."""
    out: dict[str, list[dict]] = {}
    for day in days:
        day_rows = (ctx.ds[ctx.ds['date'] == day]
                    .sort_values('score', ascending=False).head(top_n))
        if day_rows.empty:
            continue
        rmap = _reasons_map(day)
        decisions = []
        for _, row in day_rows.iterrows():
            analysis = analyze_stock(ctx, row, rmap.get(row['code'], []),
                                     horizon)
            d = decide(analysis)
            d['expected'] = analysis['expected']       # 대시보드 표시용
            decisions.append(d)
        out[day] = decisions
    return out


# ============================================================
# ① Dashboard 정렬
# ============================================================
def sort_dashboard(decisions: list[dict]) -> list[dict]:
    return sorted(decisions,
                  key=lambda d: (DECISION_ORDER.index(d['recommendation']),
                                 -d['decision_score']))


# ============================================================
# ② Watchlist Engine — Tomorrow Candidates
# ============================================================
def find_candidates(history: dict[str, list[dict]],
                    min_signals: int = MIN_SIGNALS) -> list[dict]:
    if not history:
        return []
    days = sorted(history.keys())
    today = days[-1]
    by_day = {d: {x['code']: x for x in history[d]} for d in days}
    today_map = by_day[today]

    def regime_rank(r: str) -> int:
        return REGIME_ORDER.index(r) if r in REGIME_ORDER else -1

    candidates = []
    for code, cur in today_map.items():
        if cur['recommendation'] in ('STRONG BUY', 'BUY'):
            continue                       # 이미 매수 판단 → 후보 아님
        signals: list[str] = []

        # WATCH 연속 (오늘 포함 역방향)
        streak = 0
        for d in reversed(days):
            rec = by_day[d].get(code, {}).get('recommendation')
            if rec == 'WATCH':
                streak += 1
            else:
                break
        if streak >= 2:
            signals.append(f'WATCH {streak}일 연속')

        # 직전 관측치와 비교 (오늘 제외 가장 최근)
        prev = None
        for d in reversed(days[:-1]):
            if code in by_day[d]:
                prev = by_day[d][code]
                break
        if prev:
            if cur['confidence'] - prev['confidence'] >= RISE_TH:
                signals.append(f"신뢰도 상승 {prev['confidence']:.0f}→{cur['confidence']:.0f}")
            if cur['ai_score'] - prev['ai_score'] >= RISE_TH:
                signals.append(f"AI Score 상승 {prev['ai_score']:.0f}→{cur['ai_score']:.0f}")
            if _d_rank(cur['recommendation']) > _d_rank(prev['recommendation']):
                signals.append(f"판단 개선 {prev['recommendation']}→{cur['recommendation']}")
            if (RISK_ORDER.index(cur['risk_level'])
                    < RISK_ORDER.index(prev['risk_level'])):
                signals.append(f"리스크 감소 {prev['risk_level']}→{cur['risk_level']}")
            if regime_rank(cur['regime']) > regime_rank(prev['regime']):
                signals.append(f"시장 국면 개선 {prev['regime']}→{cur['regime']}")

        if len(signals) >= min_signals:
            candidates.append({
                'code': code, 'name': cur['name'],
                'recommendation': cur['recommendation'],
                'decision_score': cur['decision_score'],
                'confidence': cur['confidence'],
                'signals': signals,
                'signal_count': len(signals),
            })
    candidates.sort(key=lambda c: (-c['signal_count'], -c['decision_score']))
    return candidates


# ============================================================
# 섹터 요약 (전체 시장 — ai2 CSV의 업종 + 스냅샷 등락)
# ============================================================
def sector_summary(ai2_df: pd.DataFrame | None, book,
                   top: int = 3, min_stocks: int = 3) -> dict:
    empty = {'strong': [], 'weak': [], 'note': '섹터 데이터 없음'}
    if ai2_df is None or len(book.days) < 2:
        return empty
    try:
        prev_day, today = book.days[-2], book.days[-1]
        df = ai2_df.copy()
        df['종목코드'] = df['종목코드'].astype(str).str.zfill(6)
        rows = []
        for _, r in df.iterrows():
            sec = r.get('업종')
            if not isinstance(sec, str) or not sec or sec == '기타':
                continue
            p1 = book.close(r['종목코드'], today)
            p0 = book.close(r['종목코드'], prev_day)
            if p0 and p1:
                ret = (p1 / p0 - 1) * 100
                # KRX 가격제한폭(±30%) 초과 = 감자/액면분할/재상장 등
                # 자본변동 이벤트 — 섹터 흐름과 무관하므로 제외
                if abs(ret) <= 30.0:
                    rows.append({'sector': sec, 'ret': ret})
        if not rows:
            return empty
        # 평균 대신 중앙값 — 개별 급등락 종목이 섹터를 왜곡하지 않도록
        g = (pd.DataFrame(rows).groupby('sector')
             .agg(avg_ret=('ret', 'median'), n=('ret', 'size')).reset_index())
        g = g[g['n'] >= min_stocks]
        g['avg_ret'] = g['avg_ret'].round(2)
        strong = g.nlargest(top, 'avg_ret').to_dict('records')
        weak = g.nsmallest(top, 'avg_ret').to_dict('records')
        return {'strong': strong, 'weak': weak, 'note': f'{today} 기준 전 종목'}
    except Exception:
        logger.exception('섹터 요약 실패')
        return empty


# ============================================================
# ③ AI Morning Brief
# ============================================================
def morning_brief(today: str, decisions: list[dict],
                  candidates: list[dict], sectors: dict,
                  regime: str) -> dict:
    counts = {rec: 0 for rec in DECISION_ORDER}
    for d in decisions:
        counts[d['recommendation']] += 1
    ranked = sort_dashboard(decisions)
    top5 = [{'name': d['name'], 'recommendation': d['recommendation'],
             'decision_score': d['decision_score']} for d in ranked[:5]]
    risk5 = [{'name': d['name'], 'recommendation': d['recommendation'],
              'risk_level': d['risk_level'],
              'decision_score': d['decision_score']} for d in ranked[-5:][::-1]]
    return {
        'date': today, 'regime': regime, 'counts': counts,
        'analyzed': len(decisions),
        'strong_sectors': sectors.get('strong', []),
        'weak_sectors': sectors.get('weak', []),
        'sector_note': sectors.get('note', ''),
        'top_picks': top5, 'risk_watch': risk5,
        'tomorrow_candidates': len(candidates),
    }


# ============================================================
# ④ Report (JSON / MD / HTML — 고정 파일명, 매일 덮어쓰기)
# ============================================================
def generate_reports(brief: dict, decisions: list[dict],
                     candidates: list[dict],
                     out_dir: str = DASH_DIR) -> dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    ranked = sort_dashboard(decisions)

    paths['json'] = os.path.join(out_dir, 'daily_dashboard.json')
    with open(paths['json'], 'w', encoding='utf-8') as f:
        json.dump({'brief': brief, 'dashboard': ranked,
                   'watchlist': candidates}, f, ensure_ascii=False, indent=1)

    # ── Markdown ──
    c = brief['counts']
    md = ['# ☀️ AI Morning Brief', '',
          f"**{brief['date']}** · 시장 국면 **{brief['regime']}**", '',
          f"| 🟢 BUY+ | 🟡 WATCH | ⚪ HOLD | 🔴 SELL | ⚫ AVOID |",
          '|---|---|---|---|---|',
          f"| {c['STRONG BUY'] + c['BUY']} | {c['WATCH']} | {c['HOLD']} "
          f"| {c['SELL']} | {c['AVOID']} |", '']
    if brief['strong_sectors']:
        md += ['**오늘 강한 섹터**: '
               + ', '.join(f"{s['sector']}({s['avg_ret']:+.1f}%)"
                           for s in brief['strong_sectors'])]
    if brief['weak_sectors']:
        md += ['**오늘 위험 섹터**: '
               + ', '.join(f"{s['sector']}({s['avg_ret']:+.1f}%)"
                           for s in brief['weak_sectors']), '']
    md += ['', '## 📋 Daily Dashboard', '',
           '| | 종목 | 판단 | DScore | AI | 신뢰도 | 국면 | 리스크 | 기대수익 | 예상낙폭 |',
           '|---|---|---|---|---|---|---|---|---|---|']
    for d in ranked:
        e = d.get('expected') or {}
        er = e.get('expected_return')
        dd = e.get('expected_drawdown')
        md.append(f"| {DECISION_EMOJI[d['recommendation']]} | {d['name']} "
                  f"| **{d['recommendation']}** | {d['decision_score']:.0f} "
                  f"| {d['ai_score']:.0f} | {d['confidence']:.0f} | {d['regime']} "
                  f"| {d['risk_level']} "
                  f"| {f'{er:+.1f}%' if er is not None else '-'} "
                  f"| {f'{dd:.1f}%' if dd is not None else '-'} |")
    md += ['', '## 🔭 Tomorrow Candidates (Watchlist)', '']
    if candidates:
        for w in candidates:
            md += [f"### {w['name']} ({w['code']}) — 현재 {w['recommendation']} "
                   f"· DScore {w['decision_score']:.0f}",
                   ''] + [f"- {s}" for s in w['signals']] + ['']
    else:
        md += ['후보 없음 — 개선 신호 2개 이상인 종목이 아직 없습니다.', '']
    paths['md'] = os.path.join(out_dir, 'daily_dashboard.md')
    with open(paths['md'], 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))

    # ── HTML (자립형 다크) ──
    style = ('<style>body{background:#1A2332;color:#E2E8F0;font-family:sans-serif;'
             'padding:24px;max-width:1000px;margin:auto;}table{border-collapse:collapse;'
             'width:100%;margin:12px 0;}th,td{border:1px solid #4A5568;padding:7px 10px;'
             'font-size:13px;}th{background:#232E40;color:#94A3B8;}h1,h2{color:#62EFFF;}'
             '.rec{font-weight:800;}.sig{color:#34D399;font-size:12px;}'
             '.metric{display:inline-block;background:#313B4D;border:1px solid #4A5568;'
             'border-radius:10px;padding:10px 18px;margin:4px;text-align:center;}'
             '.metric b{font-size:1.4em;display:block;}</style>')
    rec_color = {'STRONG BUY': '#10B981', 'BUY': '#34D399', 'WATCH': '#62EFFF',
                 'HOLD': '#94A3B8', 'SELL': '#F59E0B', 'AVOID': '#F87171'}
    html = [style, '<h1>☀️ AI Morning Brief</h1>',
            f"<p>{brief['date']} · 시장 국면 <b>{brief['regime']}</b></p>",
            '<div>']
    for label, key in [('BUY+', None), ('WATCH', 'WATCH'), ('HOLD', 'HOLD'),
                       ('SELL', 'SELL'), ('AVOID', 'AVOID')]:
        v = (c['STRONG BUY'] + c['BUY']) if key is None else c[key]
        html.append(f'<span class="metric">{label}<b>{v}</b></span>')
    html.append('</div>')
    if brief['strong_sectors']:
        html.append('<p>강한 섹터: ' + ', '.join(
            f"{s['sector']}({s['avg_ret']:+.1f}%)" for s in brief['strong_sectors']) + '</p>')
    if brief['weak_sectors']:
        html.append('<p>위험 섹터: ' + ', '.join(
            f"{s['sector']}({s['avg_ret']:+.1f}%)" for s in brief['weak_sectors']) + '</p>')
    rows = ''
    for d in ranked:
        e = d.get('expected') or {}
        er, dd = e.get('expected_return'), e.get('expected_drawdown')
        rows += (f"<tr><td>{DECISION_EMOJI[d['recommendation']]}</td>"
                 f"<td>{d['name']}</td>"
                 f"<td class='rec' style='color:{rec_color[d['recommendation']]}'>"
                 f"{d['recommendation']}</td>"
                 f"<td>{d['decision_score']:.0f}</td><td>{d['ai_score']:.0f}</td>"
                 f"<td>{d['confidence']:.0f}</td><td>{d['regime']}</td>"
                 f"<td>{d['risk_level']}</td>"
                 f"<td>{f'{er:+.1f}%' if er is not None else '-'}</td>"
                 f"<td>{f'{dd:.1f}%' if dd is not None else '-'}</td></tr>")
    html += ['<h2>📋 Daily Dashboard</h2>',
             '<table><tr><th></th><th>종목</th><th>판단</th><th>DScore</th>'
             '<th>AI</th><th>신뢰도</th><th>국면</th><th>리스크</th>'
             '<th>기대수익</th><th>예상낙폭</th></tr>', rows, '</table>',
             '<h2>🔭 Tomorrow Candidates</h2>']
    if candidates:
        for w in candidates:
            html.append(f"<h3>{w['name']} — 현재 {w['recommendation']}</h3>"
                        f"<p class='sig'>▲ " + '<br>▲ '.join(w['signals']) + '</p>')
    else:
        html.append('<p>후보 없음</p>')
    paths['html'] = os.path.join(out_dir, 'daily_dashboard.html')
    with open(paths['html'], 'w', encoding='utf-8') as f:
        f.write('\n'.join(html))
    return paths


# ============================================================
# CLI 파이프라인
# ============================================================
def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    ap = argparse.ArgumentParser(description='Daily AI Dashboard (엔진 무수정)')
    ap.add_argument('--top-n', type=int, default=10)
    ap.add_argument('--days', type=int, default=HISTORY_DAYS)
    ap.add_argument('--horizon', type=int, default=5, choices=(5, 10))
    args = ap.parse_args()

    if not os.path.exists(DATASET):
        print('컴포넌트 데이터셋 없음 — 먼저 python -m src.features.optimizer 실행')
        return
    ctx = load_context(horizon=args.horizon)
    all_days = sorted(ctx.ds['date'].unique())
    days = all_days[-args.days:]
    today = days[-1]

    history = build_daily_decisions(ctx, days, args.top_n, args.horizon)
    decisions = history.get(today, [])
    candidates = find_candidates(history)

    # 섹터: ai2 CSV (실패해도 대시보드는 계속)
    ai2_df = None
    try:
        from radar_core import fetch_ai2_csv
        ai2_df = fetch_ai2_csv()
    except Exception as e:
        logger.warning('ai2 CSV 로드 실패 — 섹터 요약 생략: %s', e)
    sectors = sector_summary(ai2_df, ctx.book)

    regime_row = ctx.regimes[ctx.regimes['date'] == today]
    regime = regime_row.iloc[0]['regime'] if not regime_row.empty else 'Unknown'
    brief = morning_brief(today, decisions, candidates, sectors, regime)
    paths = generate_reports(brief, decisions, candidates)

    c = brief['counts']
    print(f"☀️ AI Morning Brief — {today} · 국면 {regime}")
    print(f"  🟢 BUY+ {c['STRONG BUY'] + c['BUY']} · 🟡 WATCH {c['WATCH']} · "
          f"⚪ HOLD {c['HOLD']} · 🔴 SELL {c['SELL']} · ⚫ AVOID {c['AVOID']}")
    if brief['strong_sectors']:
        print('  강한 섹터:', ', '.join(f"{s['sector']}({s['avg_ret']:+.1f}%)"
                                    for s in brief['strong_sectors']))
    if brief['weak_sectors']:
        print('  위험 섹터:', ', '.join(f"{s['sector']}({s['avg_ret']:+.1f}%)"
                                    for s in brief['weak_sectors']))
    print(f"\n🔭 Tomorrow Candidates: {len(candidates)}종목")
    for w in candidates[:5]:
        print(f"  {w['name']:12s} ({w['recommendation']}) — {' · '.join(w['signals'])}")
    print('\n저장:', ', '.join(paths.values()))


if __name__ == '__main__':
    main()
