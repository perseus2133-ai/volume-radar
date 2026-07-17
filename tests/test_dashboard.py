# -*- coding: utf-8 -*-
"""Daily AI Dashboard 단위 테스트 — 요구 7항목 전부 커버."""
import os
import json
import pandas as pd
import pytest

from src.features.backtest import PriceBook
from src.features.dashboard import (sort_dashboard, find_candidates,
                                    morning_brief, sector_summary,
                                    generate_reports, DECISION_ORDER)
from tests.test_backtest import write_snapshot, days


def dec(code='000001', name='S', rec='WATCH', dscore=60.0, ai=70.0,
        conf=50.0, regime='Sideways', risk='MEDIUM', er=1.0, dd=-3.0):
    return {'date': '2026-01-10', 'code': code, 'name': name,
            'ai_score': ai, 'confidence': conf, 'confidence_grade': 'C',
            'regime': regime, 'decision_score': dscore,
            'decision_grade': 'C', 'recommendation': rec,
            'risk_level': risk, 'reason_summary': '', 'guardrails': [],
            'cautions': [],
            'expected': {'expected_return': er, 'expected_drawdown': dd,
                         'risk_reward': 0.5}}


# ── 검증: 빈 데이터 ──────────────────────────────────────────
def test_empty_data(tmp_path):
    brief = morning_brief('2026-01-10', [], [], {'strong': [], 'weak': [],
                                                 'note': ''}, 'Unknown')
    assert brief['analyzed'] == 0
    assert all(v == 0 for v in brief['counts'].values())
    paths = generate_reports(brief, [], [], out_dir=str(tmp_path))
    for k in ('json', 'md', 'html'):
        assert os.path.exists(paths[k])
    assert find_candidates({}) == []


# ── 검증: BUY 없음 / WATCH 없음 ─────────────────────────────
def test_no_buy_no_watch():
    ds = [dec(rec='SELL', dscore=40), dec(code='000002', rec='AVOID', dscore=30)]
    brief = morning_brief('2026-01-10', ds, [], {'strong': [], 'weak': [],
                                                 'note': ''}, 'Bear')
    assert brief['counts']['BUY'] == 0 and brief['counts']['STRONG BUY'] == 0
    assert brief['counts']['WATCH'] == 0
    assert brief['counts']['SELL'] == 1 and brief['counts']['AVOID'] == 1
    # WATCH 없음 → 후보도 없음 (개선 신호 없는 히스토리)
    hist = {'2026-01-09': ds, '2026-01-10': ds}
    assert find_candidates(hist) == []


# ── 검증: 모든 Decision 존재 + 집계 ─────────────────────────
def test_all_decisions_counted():
    ds = [dec(code=f'{i:06d}', rec=r, dscore=90 - i * 10)
          for i, r in enumerate(DECISION_ORDER)]
    brief = morning_brief('2026-01-10', ds, [], {'strong': [], 'weak': [],
                                                 'note': ''}, 'Bull')
    assert all(brief['counts'][r] == 1 for r in DECISION_ORDER)
    assert brief['analyzed'] == 6
    assert len(brief['top_picks']) == 5
    assert brief['top_picks'][0]['recommendation'] == 'STRONG BUY'


# ── 검증: Dashboard 정렬 (판단 순서 → 점수 내림차순) ────────
def test_dashboard_sorting():
    ds = [dec(code='1', rec='SELL', dscore=44),
          dec(code='2', rec='STRONG BUY', dscore=80),
          dec(code='3', rec='WATCH', dscore=58),
          dec(code='4', rec='WATCH', dscore=62),
          dec(code='5', rec='BUY', dscore=70)]
    ranked = sort_dashboard(ds)
    assert [d['recommendation'] for d in ranked] == \
        ['STRONG BUY', 'BUY', 'WATCH', 'WATCH', 'SELL']
    assert ranked[2]['decision_score'] >= ranked[3]['decision_score']


# ── 검증: Watchlist 후보 검출 (신호 2개+ / 이유 출력) ───────
def test_watchlist_candidate_detection():
    h = {
        '2026-01-08': [dec(rec='WATCH', dscore=55, conf=40, ai=60)],
        '2026-01-09': [dec(rec='WATCH', dscore=57, conf=42, ai=62)],
        '2026-01-10': [dec(rec='WATCH', dscore=60, conf=48, ai=66)],
    }
    cands = find_candidates(h)
    assert len(cands) == 1
    c = cands[0]
    sig = ' | '.join(c['signals'])
    assert 'WATCH 3일 연속' in sig
    assert '신뢰도 상승' in sig
    assert 'AI Score 상승' in sig
    assert c['signal_count'] >= 3


def test_watchlist_excludes_buy_and_single_signal():
    # 이미 BUY → 제외
    h1 = {'2026-01-09': [dec(rec='WATCH')],
          '2026-01-10': [dec(rec='BUY', dscore=70, conf=60)]}
    assert find_candidates(h1) == []
    # 신호 1개(WATCH 2연속)뿐 → 최소 2개 미달로 제외
    h2 = {'2026-01-09': [dec(rec='WATCH', conf=50, ai=70)],
          '2026-01-10': [dec(rec='WATCH', conf=50, ai=70)]}
    assert find_candidates(h2, min_signals=2) == []
    # 같은 조건, 개선/리스크감소 추가 → 포함
    h3 = {'2026-01-09': [dec(rec='HOLD', conf=50, ai=70, risk='HIGH')],
          '2026-01-10': [dec(rec='WATCH', conf=50, ai=70, risk='MEDIUM')]}
    cands = find_candidates(h3)
    assert len(cands) == 1
    sig = ' | '.join(cands[0]['signals'])
    assert '판단 개선 HOLD→WATCH' in sig and '리스크 감소' in sig


# ── 검증: Morning Brief 생성 (섹터 포함) ────────────────────
def test_morning_brief_with_sectors(tmp_path):
    snap = str(tmp_path / 'snap')
    d2 = days(2)
    write_snapshot(snap, d2[0], {'000001': 100.0, '000002': 100.0,
                                 '000003': 100.0, '000004': 100.0,
                                 '000005': 100.0, '000006': 100.0})
    write_snapshot(snap, d2[1], {'000001': 110.0, '000002': 108.0,
                                 '000003': 109.0, '000004': 95.0,
                                 '000005': 94.0, '000006': 96.0})
    book = PriceBook(snap)
    ai2 = pd.DataFrame([
        {'종목코드': '000001', '업종': '전력'}, {'종목코드': '000002', '업종': '전력'},
        {'종목코드': '000003', '업종': '전력'}, {'종목코드': '000004', '업종': '바이오'},
        {'종목코드': '000005', '업종': '바이오'}, {'종목코드': '000006', '업종': '바이오'},
    ])
    sectors = sector_summary(ai2, book, top=1, min_stocks=3)
    assert sectors['strong'][0]['sector'] == '전력'
    assert sectors['strong'][0]['avg_ret'] == pytest.approx(9.0)
    assert sectors['weak'][0]['sector'] == '바이오'

    ds = [dec(rec='BUY', dscore=70, conf=60)]
    brief = morning_brief(d2[1], ds, [], sectors, 'Bull')
    assert brief['regime'] == 'Bull'
    assert brief['strong_sectors'][0]['sector'] == '전력'

    paths = generate_reports(brief, ds, [], out_dir=str(tmp_path / 'out'))
    md = open(paths['md'], encoding='utf-8').read()
    assert 'AI Morning Brief' in md
    assert '전력' in md and '바이오' in md
    assert 'Daily Dashboard' in md and 'Tomorrow Candidates' in md
    data = json.load(open(paths['json'], encoding='utf-8'))
    assert data['brief']['counts']['BUY'] == 1


# ── 섹터 데이터 없음 → 우아한 생략 ──────────────────────────
def test_sector_summary_graceful_without_data(tmp_path):
    snap = str(tmp_path / 'snap')
    write_snapshot(snap, days(1)[0], {'000001': 100.0})
    s = sector_summary(None, PriceBook(snap))
    assert s['strong'] == [] and s['weak'] == []
