# -*- coding: utf-8 -*-
"""Market Regime Analyzer 단위 테스트 — 합성 데이터로 규칙 검증."""
import os
import pandas as pd
import pytest

from src.features.backtest import PriceBook
from src.features.regime import (market_daily_returns, classify_regimes,
                                 regime_component_stats,
                                 regime_recommendation_stats,
                                 regime_weight_suggestions,
                                 generate_regime_report)
from tests.test_backtest import write_snapshot, days


# ── 합성 시장: 알려진 일별 중앙값 수익률 시퀀스 ──────────────
def daily_df(rets):
    return pd.DataFrame([{'date': f'2026-01-{i+2:02d}', 'ret_median': r,
                          'ret_mean': r, 'breadth': 50.0, 'n_stocks': 100}
                         for i, r in enumerate(rets)])


# ── ① 레짐 분류 규칙 ────────────────────────────────────────
def test_classify_bull():
    d = classify_regimes(daily_df([0.5] * 8), window=5)   # 5일 누적 ≈ +2.5%
    assert (d['regime'].iloc[:4] == 'Unknown').all()      # 윈도우 미달
    assert (d['regime'].iloc[4:] == 'Bull').all()


def test_classify_bear():
    d = classify_regimes(daily_df([-0.5] * 8), window=5)  # 누적 ≈ -2.5%
    assert (d['regime'].iloc[4:] == 'Bear').all()


def test_classify_sideways():
    d = classify_regimes(daily_df([0.1, -0.1] * 4), window=5)
    assert (d['regime'].iloc[4:] == 'Sideways').all()


def test_classify_high_vol_takes_priority():
    # 강한 상승이지만 변동성 극단 → High Volatility 우선
    d = classify_regimes(daily_df([5.0, -3.0, 5.0, -3.0, 5.0]), window=5)
    assert d['regime'].iloc[4] == 'High Volatility'


# ── 시장 프록시 (스냅샷 → 일별 수익률) ──────────────────────
def test_market_daily_returns(tmp_path):
    snap = str(tmp_path / 'snap')
    ds = days(3)
    write_snapshot(snap, ds[0], {'000001': 100.0, '000002': 200.0})
    write_snapshot(snap, ds[1], {'000001': 110.0, '000002': 200.0})  # +10%, 0%
    write_snapshot(snap, ds[2], {'000001': 110.0, '000002': 180.0})  # 0%, -10%
    daily = market_daily_returns(PriceBook(snap))
    assert len(daily) == 2
    assert daily.iloc[0]['ret_median'] == pytest.approx(5.0)   # (+10, 0) 중앙값
    assert daily.iloc[0]['breadth'] == pytest.approx(50.0)
    assert daily.iloc[1]['ret_median'] == pytest.approx(-5.0)


# ── ②③④ 통합: 레짐별 통계 (Bull에서 좋고 Bear에서 나쁜 컴포넌트) ──
def synth_env(tmp_path):
    """8거래일: 앞 절반 Bull, 뒤 절반 Bear로 강제 라벨링할 합성 환경."""
    snap = str(tmp_path / 'snap')
    ds_days = days(14)
    # 가격: 종목 1개(더미) — MDD/수익률 경로용
    for i, d in enumerate(ds_days):
        write_snapshot(snap, d, {'000001': 100.0 + i, '000002': 100.0 - i * 0.5})
    book = PriceBook(snap)
    regimes = pd.DataFrame([
        {'date': d, 'regime': ('Bull' if i < 7 else 'Bear'),
         'ret_median': 0, 'ret_mean': 0, 'breadth': 50, 'n_stocks': 2,
         'cum_w': 0, 'vol_w': 0}
        for i, d in enumerate(ds_days)])
    rows = []
    for i, d in enumerate(ds_days[:8]):        # 미래 5거래일 확보되는 날만
        bull = i < 7
        for j, code in enumerate(('000001', '000002')):
            up = code == '000001'              # 000001은 상승 경로, 000002는 하락
            rows.append({
                'date': d, 'code': code, 'name': f'S{j}',
                'score': 90 - j * 10, 'rank': j + 1, 'penalty': 0.0,
                'avail_max': 20.0,
                # compX: 상승 종목에서만 보유 → Bull/Bear 모두 승률 차이 검증용
                'compX_earned': 10.0 if up else 0.0, 'compX_max': 10.0, 'compX_avail': 1,
                'compY_earned': 5.0, 'compY_max': 10.0, 'compY_avail': 1,
                'ret_5d': (5.0 if up else -2.5), 'ret_10d': None,
            })
    return book, regimes, pd.DataFrame(rows)


def test_regime_component_and_rec_stats(tmp_path):
    book, regimes, ds = synth_env(tmp_path)
    comp = regime_component_stats(ds, regimes, book, horizon=5)
    bull_x = comp[(comp['regime'] == 'Bull') & (comp['component'] == 'compX')].iloc[0]
    assert bull_x['n'] > 0
    assert bull_x['win_rate'] == pytest.approx(100.0)   # compX 보유 = 상승 종목만
    assert bull_x['mean_ret'] == pytest.approx(5.0)
    assert bull_x['mdd'] is not None

    rec = regime_recommendation_stats(ds, regimes, book, horizon=5, top_n=2)
    bull = rec[rec['regime'] == 'Bull'].iloc[0]
    assert bull['n'] == 14                               # 7일 × 2종목
    assert bull['win_rate'] == pytest.approx(50.0)
    side = rec[rec['regime'] == 'Sideways'].iloc[0]
    assert side['n'] == 0                                # 표본 없음 → 안전 처리


def test_regime_weight_suggestions(tmp_path):
    book, regimes, ds = synth_env(tmp_path)
    weights = regime_weight_suggestions(ds, regimes, horizon=5)
    assert 'Bull' in weights and 'Bear' in weights
    assert 'Sideways' not in weights                     # 표본 없는 레짐 제외
    for sugg in weights.values():
        assert {'component', 'current', 'suggested', 'reason'} <= set(sugg.columns)


def test_generate_regime_report(tmp_path):
    book, regimes, ds = synth_env(tmp_path)
    comp = regime_component_stats(ds, regimes, book, 5)
    rec = regime_recommendation_stats(ds, regimes, book, 5, top_n=2)
    weights = regime_weight_suggestions(ds, regimes, 5)
    paths = generate_regime_report(regimes, comp, rec, weights,
                                   out_dir=str(tmp_path / 'out'))
    for key in ('md', 'html', 'daily_csv', 'comp_csv', 'rec_csv'):
        assert os.path.exists(paths[key]), key
    md = open(paths['md'], encoding='utf-8').read()
    assert '레짐별 추천 가중치 (미적용)' in md
    assert '수정되지 않았다' in md
