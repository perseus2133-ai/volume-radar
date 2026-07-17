# -*- coding: utf-8 -*-
"""Backtest Engine v1 단위 테스트 — 합성 데이터로 완전 격리 검증."""
import os
import json
import gzip
import pandas as pd
import pytest

from src.features.backtest import (BacktestConfig, PriceBook, run_backtest,
                                   save_report, categorize_reason)


# ── 합성 데이터 헬퍼 ─────────────────────────────────────────
def write_snapshot(snap_dir, day, prices: dict[str, float]):
    os.makedirs(snap_dir, exist_ok=True)
    df = pd.DataFrame([{'code': c, 'name': f'N{c}', 'market': 'KOSPI',
                        'close': p, 'volume': 1000} for c, p in prices.items()])
    df.to_csv(os.path.join(snap_dir, f'{day}.csv.gz'),
              index=False, compression='gzip', encoding='utf-8')


def write_scores(scores_dir, day, entries):
    os.makedirs(scores_dir, exist_ok=True)
    with open(os.path.join(scores_dir, f'{day}.json'), 'w', encoding='utf-8') as f:
        json.dump(entries, f, ensure_ascii=False)


def days(n, start=1):
    """2026-01-{start}.. 형태의 n개 유사 거래일."""
    return [f'2026-01-{d:02d}' for d in range(start, start + n)]


def score_entry(code='000001', name='A', score=90.0, reasons=None):
    return {'code': code, 'name': name, 'score': score, 'stars': 5,
            'coverage': 100.0, 'reasons': reasons or ['60일 신고가 돌파']}


# ── 검증 항목: 빈 데이터 ─────────────────────────────────────
def test_empty_data(tmp_path):
    cfg = BacktestConfig(start='2026-01-01', end='2026-01-31')
    result = run_backtest(cfg, str(tmp_path / 's'), str(tmp_path / 'p'))
    assert result.records == []
    assert result.summary[5].count == 0


# ── 검증 항목: 하루 데이터 (미래 없음 → insufficient) ───────
def test_single_day_insufficient(tmp_path):
    snap, scores = str(tmp_path / 'snap'), str(tmp_path / 'scores')
    write_snapshot(snap, '2026-01-01', {'000001': 100.0})
    write_scores(scores, '2026-01-01', [score_entry()])
    result = run_backtest(BacktestConfig('2026-01-01', '2026-01-01'), scores, snap)
    assert len(result.records) == 1
    assert result.records[0].returns[5].status == 'insufficient'
    assert result.summary[5].count == 0
    assert result.summary[5].insufficient == 1


# ── 검증 항목: 여러 거래일 — 수익률·승률·MDD 정밀 검증 ──────
def test_multi_day_exact_returns(tmp_path):
    snap, scores = str(tmp_path / 'snap'), str(tmp_path / 'scores')
    ds = days(7)
    # A: 100 → (5거래일 후) 110  = +10%
    # B: 200 → 180              = -10%
    path_a = [100, 102, 104, 106, 108, 110, 111]
    path_b = [200, 195, 190, 185, 182, 180, 179]
    for i, d in enumerate(ds):
        write_snapshot(snap, d, {'000001': path_a[i], '000002': path_b[i]})
    write_scores(scores, ds[0], [
        score_entry('000001', 'A', 95.0, ['60일 신고가 돌파']),
        score_entry('000002', 'B', 80.0, ['거래량 3.0배 폭증']),
    ])
    result = run_backtest(BacktestConfig(ds[0], ds[0]), scores, snap)
    r_a = next(r for r in result.records if r.code == '000001')
    r_b = next(r for r in result.records if r.code == '000002')
    assert r_a.returns[5].ret_pct == pytest.approx(10.0)
    assert r_b.returns[5].ret_pct == pytest.approx(-10.0)
    s5 = result.summary[5]
    assert s5.count == 2
    assert s5.mean == pytest.approx(0.0)
    assert s5.win_rate == pytest.approx(50.0)
    assert s5.max_gain == pytest.approx(10.0)
    assert s5.max_loss == pytest.approx(-10.0)
    # MDD: 평균 곡선 [1.0, .9975, .995, .9925, .995, 1.0] → 최저 -0.75%
    assert s5.mdd == pytest.approx(-0.75, abs=0.05)
    # 점수 구간: 95+ 는 A만, 80+ 는 둘 다
    assert result.bands['95+']['recommendations'] == 1
    assert result.bands['80+']['recommendations'] == 2
    # 근거 집계: 신고가 카테고리 평균 +10%
    assert result.reasons['신고가 돌파/근접']['mean_5d'] == pytest.approx(10.0)
    assert result.reasons['거래량 폭증']['mean_5d'] == pytest.approx(-10.0)


# ── 검증 항목: 거래정지 (목표일에 종목 없음) ─────────────────
def test_suspended_at_target(tmp_path):
    snap, scores = str(tmp_path / 'snap'), str(tmp_path / 'scores')
    ds = days(7)
    for i, d in enumerate(ds):
        prices = {'000001': 100.0 + i}
        if i < 3:                       # 3일차 이후 사라짐 (정지/상폐)
            prices['000009'] = 50.0
        write_snapshot(snap, d, prices)
    write_scores(scores, ds[0], [score_entry('000009', '정지주', 90.0)])
    result = run_backtest(BacktestConfig(ds[0], ds[0]), scores, snap)
    assert result.records[0].returns[5].status == 'suspended'
    assert result.summary[5].suspended == 1
    assert result.summary[5].count == 0


# ── 검증 항목: 상장폐지 예외 (추천일 가격 자체가 없음) ───────
def test_missing_entry_price(tmp_path):
    snap, scores = str(tmp_path / 'snap'), str(tmp_path / 'scores')
    ds = days(7)
    for d in ds:
        write_snapshot(snap, d, {'000001': 100.0})
    write_scores(scores, ds[0], [score_entry('999999', '유령주', 99.0)])
    result = run_backtest(BacktestConfig(ds[0], ds[0]), scores, snap)
    r = result.records[0]
    assert r.entry_close is None
    assert all(rr.status == 'suspended' for rr in r.returns.values())


# ── 검증 항목: 누락 데이터 (중간일 결측 → carry-forward) ─────
def test_mid_gap_carry_forward(tmp_path):
    snap, scores = str(tmp_path / 'snap'), str(tmp_path / 'scores')
    ds = days(7)
    for i, d in enumerate(ds):
        prices = {'000001': 100.0 + i * 2}
        if i == 2:                      # 중간 하루만 결측 (일시 정지)
            prices.pop('000001')
        prices['000002'] = 500.0
        write_snapshot(snap, d, prices)
    write_scores(scores, ds[0], [score_entry('000001', 'A', 90.0)])
    result = run_backtest(BacktestConfig(ds[0], ds[0]), scores, snap)
    rr = result.records[0].returns[5]
    assert rr.status == 'ok'            # 목표일엔 가격 존재 → 정상 계산
    assert rr.ret_pct == pytest.approx(10.0)   # 100 → 110


# ── 검증 항목: 잘못된 날짜 ───────────────────────────────────
def test_invalid_dates(tmp_path):
    with pytest.raises(ValueError):
        run_backtest(BacktestConfig('2026-02-01', '2026-01-01'),
                     str(tmp_path), str(tmp_path))
    with pytest.raises(ValueError):
        run_backtest(BacktestConfig('2026/01/01', '2026-01-31'),
                     str(tmp_path), str(tmp_path))
    with pytest.raises(ValueError):
        run_backtest(BacktestConfig('2026-01-01', '2026-01-31', top_n=0),
                     str(tmp_path), str(tmp_path))


# ── min_score / top_n 필터 ──────────────────────────────────
def test_min_score_and_top_n(tmp_path):
    snap, scores = str(tmp_path / 'snap'), str(tmp_path / 'scores')
    ds = days(7)
    for d in ds:
        write_snapshot(snap, d, {f'00000{i}': 100.0 for i in range(1, 6)})
    write_scores(scores, ds[0], [
        score_entry(f'00000{i}', f'S{i}', 100.0 - i * 10) for i in range(1, 6)
    ])  # 점수: 90, 80, 70, 60, 50
    r1 = run_backtest(BacktestConfig(ds[0], ds[0], min_score=75), scores, snap)
    assert len(r1.records) == 2
    r2 = run_backtest(BacktestConfig(ds[0], ds[0], top_n=3), scores, snap)
    assert len(r2.records) == 3
    assert [r.rank for r in r2.records] == [1, 2, 3]


# ── 저장 (⑤): 4개 파일 생성 ─────────────────────────────────
def test_save_report_files(tmp_path):
    snap, scores = str(tmp_path / 'snap'), str(tmp_path / 'scores')
    ds = days(7)
    for i, d in enumerate(ds):
        write_snapshot(snap, d, {'000001': 100.0 + i})
    write_scores(scores, ds[0], [score_entry()])
    result = run_backtest(BacktestConfig(ds[0], ds[-1]), scores, snap)
    paths = save_report(result, str(tmp_path / 'out'), asof='2026-01-31')
    for key in ('report', 'summary', 'performance', 'score_analysis'):
        assert os.path.exists(paths[key]), key
    with open(paths['report'], encoding='utf-8') as f:
        rpt = json.load(f)
    assert rpt['summary']['5']['count'] == 1


# ── 근거 카테고리 정규화 ─────────────────────────────────────
def test_categorize_reason():
    assert categorize_reason('거래대금 5위') == '거래대금 상위'
    assert categorize_reason('영업이익 성장 +391%') == '실적 성장'
    assert categorize_reason('PEG 0.31 (저평가)') == '저평가(PEG)'
    assert categorize_reason('외인+기관 동시 순매수') == '수급(외인/기관)'
    assert categorize_reason('10거래일 중 5회 등장') == '반복 등장'
    assert categorize_reason('완전 새로운 근거') == '기타'
