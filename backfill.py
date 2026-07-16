#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""일회성 백필 — ai2 레포의 git 히스토리에서 과거 CSV를 받아
스냅샷/리스트/누적 랭킹을 소급 생성한다. (첫날부터 10일 누적이 작동하도록)

ai2의 '🤖 Auto crawl' 커밋(매일 새벽)을 날짜별로 하나씩 골라
그 시점의 consensus_data.csv 를 다운로드한다. 커밋이 새벽 D시에 담는
시세는 D-1 종가이므로 데이터 기준일 = 커밋일 - 1일.
"""
import io
import re
import sys
import datetime

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import pandas as pd
import requests

from radar_core import (build_daily_list, rebuild_accumulation,
                        snapshot_days, is_duplicate_of_prev)

API = 'https://api.github.com/repos/perseus2133-ai/ai2/commits'
RAW = 'https://raw.githubusercontent.com/perseus2133-ai/ai2/{sha}/data/consensus_data.csv'
N_DAYS = 15   # 소급할 캘린더 일수 (거래일로는 ~10일)


def main():
    r = requests.get(API, params={'path': 'data/consensus_data.csv', 'per_page': 60},
                     timeout=30, headers={'User-Agent': 'volume-radar'})
    r.raise_for_status()
    commits = r.json()

    # 날짜별 auto-crawl 커밋 1개씩 (가장 이른 새벽 커밋 우선)
    by_day = {}
    for c in commits:
        msg = c['commit']['message']
        if 'Auto crawl' not in msg:
            continue
        m = re.search(r'(\d{4}-\d{2}-\d{2})', msg)
        if not m:
            continue
        commit_day = datetime.date.fromisoformat(m.group(1))
        data_day = commit_day - datetime.timedelta(days=1)
        if data_day.weekday() >= 5:
            continue
        key = data_day.strftime('%Y-%m-%d')
        if key not in by_day:      # commits는 최신순 → 첫 매칭 유지
            by_day[key] = c['sha']

    days = sorted(by_day.keys())[-N_DAYS:]
    print(f'백필 대상 거래일: {len(days)}일 ({days[0]} ~ {days[-1]})')

    prev_day = None
    for day in days:
        sha = by_day[day]
        resp = requests.get(RAW.format(sha=sha), timeout=60,
                            headers={'User-Agent': 'volume-radar'})
        if resp.status_code != 200:
            print(f'  {day}: 다운로드 실패({resp.status_code}) — 스킵')
            continue
        df = pd.read_csv(io.StringIO(resp.content.decode('utf-8-sig')), dtype={'종목코드': str})
        df['종목코드'] = df['종목코드'].astype(str).str.zfill(6)
        for c in ('현재가', 'Recent_Volume', '시가총액'):
            df[c] = pd.to_numeric(df.get(c), errors='coerce')

        if is_duplicate_of_prev(df, prev_day):
            print(f'  {day}: 휴장(거래량 동일) — 스킵')
            continue
        payload = build_daily_list(df, day, prev_day)
        print(f'  {day}: 리스트 {len(payload["entries"])}종목 (직전 {prev_day})')
        prev_day = day

    acc = rebuild_accumulation()
    print(f'\n누적 랭킹 (최근 {len(acc["window_days"])}거래일) 상위 10:')
    for s in acc['stocks'][:10]:
        print(f"  {s['name']:12s} {s['count']}회 등장 · 평균 {s['avg_rank']:.0f}위 "
              f"· 연속 {s['streak']}일 · 최근 등락 {s['last_chg']:+.1f}%")


if __name__ == '__main__':
    main()
