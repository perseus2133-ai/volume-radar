# -*- coding: utf-8 -*-
"""AI Score 단위 테스트."""
import numpy as np
import pandas as pd
import pytest

from src.features.ai_score import (compute_ai_score, score_universe, _stars,
                                   RSI_OVERHEAT)


def make_row(**kw) -> pd.Series:
    """기본은 전부 결측인 합성 종목 행."""
    base = {
        '종목코드': '000001', '종목명': '테스트',
        '현재가': np.nan, 'Recent_Volume': np.nan,
        '영업이익_2025': np.nan, '영업이익_2026': np.nan,
        '영업이익_최대성장률': np.nan, '영업이익_성장률_2025': np.nan,
        '영업이익_성장률_2026': np.nan, 'PER': np.nan,
        '외인_5d': np.nan, '외인_20d': np.nan, '기관_5d': np.nan, '기관_20d': np.nan,
        '저항선': np.nan, '거래량배수': np.nan, 'RSI': np.nan,
        'MA_align': '', 'MACD_signal': '', 'OBV_trend': '',
    }
    base.update(kw)
    return pd.Series(base)


def perfect_row(**kw) -> pd.Series:
    """모든 컴포넌트 만점인 행."""
    row = make_row(
        현재가=105.0, 저항선=100.0,                 # 신고가 돌파 10
        영업이익_2025=100.0,                         # 흑자 +4
        영업이익_최대성장률=300.0,                   # 성장 16 → 실적 20
        PER=10.0, 영업이익_성장률_2026=100.0,        # PEG 0.05 → 10
        외인_5d=1.0, 기관_5d=1.0, 외인_20d=1.0, 기관_20d=1.0,  # 수급 15
        거래량배수=3.5,                              # 10
        MA_align='up', MACD_signal='bull', OBV_trend='up',      # 10
    )
    for k, v in kw.items():
        row[k] = v
    return row


def test_perfect_score_is_100_and_5_stars():
    r = compute_ai_score(perfect_row(), turnover_rank=5, acc_count=6)
    assert r.score == 100.0
    assert r.stars == 5
    assert r.coverage_pct == 100.0


def test_missing_flow_renormalizes_not_zero():
    """수급 전량 결측(현재 ai2 상태)이어도 나머지로 100점 환산."""
    row = perfect_row(외인_5d=np.nan, 기관_5d=np.nan,
                      외인_20d=np.nan, 기관_20d=np.nan)
    r = compute_ai_score(row, turnover_rank=5, acc_count=6)
    assert r.coverage_pct == 85.0          # 100 - 수급 15
    assert r.score == 100.0                # 85/85 → 재정규화 100
    assert not r.components['수급'].available


def test_overheat_penalty_is_5_points():
    hot = compute_ai_score(perfect_row(RSI=RSI_OVERHEAT), 5, 6)
    cool = compute_ai_score(perfect_row(RSI=60.0), 5, 6)
    assert cool.score - hot.score == pytest.approx(5.0)
    assert hot.penalty == 5.0


def test_empty_row_scores_zero_without_error():
    r = compute_ai_score(make_row(), turnover_rank=np.nan, acc_count=0)
    assert r.score == 0.0
    assert r.stars == 1


def test_star_boundaries():
    assert _stars(90) == 5
    assert _stars(89.9) == 4
    assert _stars(80) == 4
    assert _stars(65) == 3
    assert _stars(50) == 2
    assert _stars(49.9) == 1


def test_reasons_contains_notes():
    r = compute_ai_score(perfect_row(), 5, 6)
    joined = ' '.join(r.reasons())
    assert '신고가 돌파' in joined
    assert '순매수' in joined
    assert '정배열' in joined


def test_score_universe_liquidity_floor_and_sorting():
    df = pd.DataFrame([
        # A: 거래대금 5e9 (통과), 좋은 지표
        dict(make_row(종목코드='000010', 종목명='A', 현재가=5000.0,
                      Recent_Volume=1_000_000, 영업이익_최대성장률=200.0,
                      영업이익_2025=100.0, 저항선=5000.0, 거래량배수=3.0)),
        # B: 거래대금 4e9 (통과), 약한 지표
        dict(make_row(종목코드='000020', 종목명='B', 현재가=4000.0,
                      Recent_Volume=1_000_000, 영업이익_최대성장률=30.0,
                      영업이익_2025=10.0)),
        # C: 거래대금 1e8 (하한 미달 → 제외)
        dict(make_row(종목코드='000030', 종목명='C', 현재가=100.0,
                      Recent_Volume=1_000_000)),
    ])
    results = score_universe(df, acc_counts={'000010': 5})
    codes = [r.code for r in results]
    assert '000030' not in codes
    assert len(results) == 2
    assert results[0].score >= results[1].score
    assert results[0].code == '000010'
