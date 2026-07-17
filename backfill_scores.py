#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""일회성 — 과거 거래일의 AI Score 소급 생성 (백테스트 데이터 확보용).

ai2 git 히스토리에서 각 거래일의 CSV를 받아 **현재와 동일한 AI Score
알고리즘(무수정)** 으로 채점한다. 누적 등장 횟수(acc_counts)는 그 시점
이전 10개 리스트 파일에서 재구성 → look-ahead bias 없음.

이미 존재하는 점수 파일은 건드리지 않는다 (덮어쓰기 금지).
"""
import io
import os
import re
import sys
import json
import datetime

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import pandas as pd
import requests

from radar_core import LIST_DIR, SNAP_DIR, load_list, list_days
from src.features.ai_score import score_universe

API = 'https://api.github.com/repos/perseus2133-ai/ai2/commits'
RAW = 'https://raw.githubusercontent.com/perseus2133-ai/ai2/{sha}/data/consensus_data.csv'
SCORES_DIR = os.path.join('data', 'scores')
ACC_WINDOW = 10


def acc_counts_asof(day: str) -> dict[str, int]:
    """day '이전까지'(당일 포함) 최근 10개 리스트에서 등장 횟수 재구성."""
    days = [d for d in list_days() if d <= day][-ACC_WINDOW:]
    counts: dict[str, int] = {}
    for d in days:
        payload = load_list(d) or {}
        for e in payload.get('entries', []):
            counts[e['code']] = counts.get(e['code'], 0) + 1
    return counts


def main() -> None:
    snap_days = sorted(f[:-7] for f in os.listdir(SNAP_DIR) if f.endswith('.csv.gz'))
    have = {f[:-5] for f in os.listdir(SCORES_DIR)} if os.path.isdir(SCORES_DIR) else set()
    targets = [d for d in snap_days if d not in have]
    if not targets:
        print('생성할 날짜 없음 (모두 존재)')
        return
    print(f'소급 대상: {len(targets)}일 ({targets[0]} ~ {targets[-1]})')

    r = requests.get(API, params={'path': 'data/consensus_data.csv', 'per_page': 80},
                     timeout=30, headers={'User-Agent': 'volume-radar'})
    r.raise_for_status()
    by_day: dict[str, str] = {}
    for c in r.json():
        msg = c['commit']['message']
        if 'Auto crawl' not in msg:
            continue
        m = re.search(r'(\d{4}-\d{2}-\d{2})', msg)
        if not m:
            continue
        data_day = (datetime.date.fromisoformat(m.group(1))
                    - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
        by_day.setdefault(data_day, c['sha'])

    os.makedirs(SCORES_DIR, exist_ok=True)
    made = 0
    for day in targets:
        sha = by_day.get(day)
        if not sha:
            print(f'  {day}: 해당일 커밋 없음 — 스킵')
            continue
        resp = requests.get(RAW.format(sha=sha), timeout=60,
                            headers={'User-Agent': 'volume-radar'})
        if resp.status_code != 200:
            print(f'  {day}: CSV 다운로드 실패({resp.status_code}) — 스킵')
            continue
        df = pd.read_csv(io.StringIO(resp.content.decode('utf-8-sig')),
                         dtype={'종목코드': str})
        scores = score_universe(df, acc_counts_asof(day))[:200]
        with open(os.path.join(SCORES_DIR, f'{day}.json'), 'w', encoding='utf-8') as f:
            json.dump([{'code': s.code, 'name': s.name, 'score': s.score,
                        'stars': s.stars, 'coverage': s.coverage_pct,
                        'reasons': s.reasons()} for s in scores],
                      f, ensure_ascii=False, indent=1)
        made += 1
        print(f'  {day}: {len(scores)}종목 (1위 {scores[0].name} {scores[0].score}점)')
    print(f'\n완료: {made}일 생성')


if __name__ == '__main__':
    main()
