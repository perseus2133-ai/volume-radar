#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""매일 새벽(05:30 KST, ai2 크롤 완료 후) GitHub Actions에서 실행.

데이터 기준일 = 실행일 - 1일 (새벽 크롤이 담는 시세는 전날 종가).
기준일이 주말이거나 휴장(전 스냅샷과 거래량 동일)이면 스킵.
"""
import os
import sys
import json
import datetime

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from radar_core import (fetch_ai2_csv, build_daily_list, rebuild_accumulation,
                        snapshot_days, is_duplicate_of_prev, now_kst)
from src.features.ai_score import score_universe


def main():
    data_day = (now_kst().date() - datetime.timedelta(days=1))
    if data_day.weekday() >= 5:   # 토(5)/일(6) → 휴장
        print(f'{data_day} 은 주말 — 스킵')
        return
    day_str = data_day.strftime('%Y-%m-%d')

    days = snapshot_days()
    prev_day = days[-1] if days else None
    if prev_day == day_str:
        prev_candidates = [d for d in days if d < day_str]
        prev_day = prev_candidates[-1] if prev_candidates else None

    print(f'데이터 기준일: {day_str} (직전 거래일: {prev_day})')
    df = fetch_ai2_csv()
    print(f'ai2 CSV: {len(df)}종목')

    if is_duplicate_of_prev(df, prev_day):
        print('거래량 총합이 직전과 동일 — 휴장일로 판단, 스킵')
        return

    payload = build_daily_list(df, day_str, prev_day)
    print(f'오늘 리스트: {len(payload["entries"])}종목 '
          f'(거래대금 상위 {payload["top_n"]} 중 상승+실적 통과)')

    acc = rebuild_accumulation()
    top = acc['stocks'][:5]
    print(f'누적 랭킹 상위: ' + ', '.join(f"{s['name']}({s['count']}회)" for s in top))

    # AI Score: 채점 후 상위 200종목 저장 (data/scores/{거래일}.json)
    # → 일별 점수가 축적되어 다음 기능(백테스트)의 원료가 된다
    acc_counts = {s['code']: s['count'] for s in acc['stocks']}
    scores = score_universe(df, acc_counts)[:200]
    os.makedirs(os.path.join('data', 'scores'), exist_ok=True)
    with open(os.path.join('data', 'scores', f'{day_str}.json'), 'w', encoding='utf-8') as f:
        json.dump([{'code': r.code, 'name': r.name, 'score': r.score,
                    'stars': r.stars, 'coverage': r.coverage_pct,
                    'reasons': r.reasons()} for r in scores], f, ensure_ascii=False, indent=1)
    if scores:
        print(f'AI Score 저장: {len(scores)}종목 (1위 {scores[0].name} {scores[0].score}점)')


if __name__ == '__main__':
    main()
