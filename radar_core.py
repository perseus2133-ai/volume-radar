#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Volume Radar 코어 — ai2 데이터 기반 거래대금 레이더
====================================================
크롤링 없음. ai2 레포의 consensus_data.csv(매일 새벽 자동 갱신)만 소비한다.

파이프라인 (매일):
  1. ai2 CSV fetch → 종가/거래량 스냅샷 저장 (data/snapshots/{거래일}.csv.gz)
  2. 전 거래일 스냅샷과 비교해 등락률 계산
  3. 거래대금(종가×거래량) 상위 TOP_N 중 [상승 + 실적 필터 통과]만 리스트화
     → data/lists/{거래일}.json
  4. 최근 10거래일 리스트로 누적 랭킹 재계산 → data/accumulation.json

거래일 규칙:
  - 새벽 크롤이 담는 시세는 '전날 종가' → 데이터 기준일 = 실행일 - 1일
  - 기준일이 토/일이면 스킵, 거래량 총합이 전 스냅샷과 동일하면(휴장) 스킵
"""
import os
import io
import json
import gzip
import datetime

import numpy as np
import pandas as pd
import requests

KST_OFFSET = datetime.timedelta(hours=9)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, 'data')
SNAP_DIR = os.path.join(DATA_DIR, 'snapshots')
LIST_DIR = os.path.join(DATA_DIR, 'lists')
ACC_FILE = os.path.join(DATA_DIR, 'accumulation.json')

AI2_RAW = 'https://raw.githubusercontent.com/perseus2133-ai/ai2/main/data/consensus_data.csv'

TOP_N = 100          # 거래대금 상위 몇 위까지 스캔할지
MAX_LIST = 60        # 리스트 최대 보존 종목 수
ACC_WINDOW = 10      # 누적 윈도우 (거래일)
MIN_GROWTH = 20.0    # 실적 필터: 최대성장률(매출 or 영업이익) 하한 (%)


def now_kst():
    return datetime.datetime.utcnow() + KST_OFFSET


# ============================================================
# 데이터 로드/저장
# ============================================================
def fetch_ai2_csv(url=AI2_RAW):
    r = requests.get(url, timeout=30, headers={'User-Agent': 'volume-radar'})
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.content.decode('utf-8-sig')), dtype={'종목코드': str})
    df['종목코드'] = df['종목코드'].astype(str).str.zfill(6)
    for c in ('현재가', 'Recent_Volume', '시가총액'):
        df[c] = pd.to_numeric(df.get(c), errors='coerce')
    return df


def save_snapshot(df, data_day):
    """종가/거래량 슬림 스냅샷 저장."""
    os.makedirs(SNAP_DIR, exist_ok=True)
    slim = df[['종목코드', '종목명', '시장', '현재가', 'Recent_Volume']].copy()
    slim.columns = ['code', 'name', 'market', 'close', 'volume']
    slim = slim[(slim['close'] > 0)]
    path = os.path.join(SNAP_DIR, f'{data_day}.csv.gz')
    slim.to_csv(path, index=False, compression='gzip', encoding='utf-8')
    return slim


def load_snapshot(data_day):
    path = os.path.join(SNAP_DIR, f'{data_day}.csv.gz')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, dtype={'code': str}, compression='gzip')
    df['code'] = df['code'].astype(str).str.zfill(6)
    return df


def snapshot_days():
    if not os.path.isdir(SNAP_DIR):
        return []
    return sorted(f[:-7] for f in os.listdir(SNAP_DIR) if f.endswith('.csv.gz'))


def list_days():
    if not os.path.isdir(LIST_DIR):
        return []
    return sorted(f[:-5] for f in os.listdir(LIST_DIR) if f.endswith('.json'))


def load_list(day):
    path = os.path.join(LIST_DIR, f'{day}.json')
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f)


# ============================================================
# 실적 필터
# ============================================================
def earnings_ok(row, min_growth=MIN_GROWTH):
    """영업이익 흑자(25E 또는 26E) + 성장률(매출/영업이익 최대) >= min_growth."""
    op25 = pd.to_numeric(row.get('영업이익_2025'), errors='coerce')
    op26 = pd.to_numeric(row.get('영업이익_2026'), errors='coerce')
    profitable = (pd.notna(op25) and op25 > 0) or (pd.notna(op26) and op26 > 0)
    if not profitable:
        return False
    g1 = pd.to_numeric(row.get('매출액_최대성장률'), errors='coerce')
    g2 = pd.to_numeric(row.get('영업이익_최대성장률'), errors='coerce')
    gmax = max([v for v in (g1, g2) if pd.notna(v)], default=np.nan)
    return pd.notna(gmax) and gmax >= min_growth


# ============================================================
# 일일 리스트 생성
# ============================================================
def build_daily_list(ai2_df, data_day, prev_day):
    """거래대금 상위 TOP_N 중 상승+실적 통과 종목 리스트를 생성/저장."""
    cur = save_snapshot(ai2_df, data_day)
    prev = load_snapshot(prev_day) if prev_day else None
    prev_close = dict(zip(prev['code'], prev['close'])) if prev is not None else {}

    df = ai2_df.copy()
    df['turnover'] = df['현재가'] * df['Recent_Volume']
    df = df[df['turnover'] > 0].sort_values('turnover', ascending=False).reset_index(drop=True)
    df['turnover_rank'] = df.index + 1

    entries = []
    for _, row in df.head(TOP_N).iterrows():
        code = row['종목코드']
        pc = prev_close.get(code)
        if pc is None or pc <= 0:
            continue                     # 전일 데이터 없으면 상승 판정 불가 → 제외
        chg = (row['현재가'] / pc - 1) * 100
        if chg <= 0:
            continue                     # 상승 종목만
        if not earnings_ok(row):
            continue                     # 실적 필터
        entries.append({
            'rank': int(row['turnover_rank']),
            'code': code,
            'name': row.get('종목명', ''),
            'market': row.get('시장', ''),
            'close': float(row['현재가']),
            'chg_pct': round(float(chg), 2),
            'turnover_억': round(float(row['turnover']) / 1e8, 1),
            'mcap_억': float(row.get('시가총액') or 0),
            '업종': row.get('업종', '') if isinstance(row.get('업종'), str) else '',
            'rev_gmax': _f(row.get('매출액_최대성장률')),
            'op_gmax': _f(row.get('영업이익_최대성장률')),
        })
        if len(entries) >= MAX_LIST:
            break

    os.makedirs(LIST_DIR, exist_ok=True)
    payload = {'date': data_day, 'prev_date': prev_day,
               'top_n': TOP_N, 'entries': entries}
    with open(os.path.join(LIST_DIR, f'{data_day}.json'), 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    return payload


def _f(v):
    v = pd.to_numeric(v, errors='coerce')
    return round(float(v), 1) if pd.notna(v) else None


# ============================================================
# 누적 랭킹
# ============================================================
def rebuild_accumulation(window=ACC_WINDOW):
    """최근 window 거래일 리스트에서 등장 횟수/평균 순위/스트릭 집계."""
    days = list_days()[-window:]
    acc = {}
    for d in days:
        payload = load_list(d)
        for e in (payload or {}).get('entries', []):
            slot = acc.setdefault(e['code'], {
                'name': e['name'], 'market': e['market'], '업종': e.get('업종', ''),
                'dates': [], 'ranks': [], 'last': None,
            })
            slot['dates'].append(d)
            slot['ranks'].append(e['rank'])
            slot['last'] = e
    result = []
    for code, s in acc.items():
        streak = 0
        for d in reversed(days):
            if d in s['dates']:
                streak += 1
            else:
                break
        result.append({
            'code': code, 'name': s['name'], 'market': s['market'], '업종': s['업종'],
            'count': len(s['dates']),
            'avg_rank': round(float(np.mean(s['ranks'])), 1),
            'best_rank': int(min(s['ranks'])),
            'streak': streak,
            'dates': s['dates'],
            'last_close': s['last']['close'],
            'last_chg': s['last']['chg_pct'],
            'last_turnover_억': s['last']['turnover_억'],
        })
    result.sort(key=lambda x: (-x['count'], x['avg_rank']))
    out = {'updated': now_kst().strftime('%Y-%m-%d %H:%M'),
           'window_days': days, 'stocks': result}
    with open(ACC_FILE, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    return out


# ============================================================
# 거래일 판정
# ============================================================
def is_duplicate_of_prev(ai2_df, prev_day):
    """휴장일 감지: 거래량 총합이 직전 스냅샷과 동일하면 같은 데이터."""
    prev = load_snapshot(prev_day) if prev_day else None
    if prev is None:
        return False
    cur_sum = int(pd.to_numeric(ai2_df['Recent_Volume'], errors='coerce').fillna(0).sum())
    prev_sum = int(prev['volume'].fillna(0).sum())
    return cur_sum == prev_sum
