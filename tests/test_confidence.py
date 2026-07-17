# -*- coding: utf-8 -*-
"""Confidence Engine 단위 테스트 — 합성 컨텍스트로 격리 검증."""
import os
import json
import numpy as np
import pandas as pd
import pytest

from src.features.backtest import PriceBook
from src.features.confidence import (Context, find_similar, similarity_stats,
                                     expected_metrics, confidence_score,
                                     analyze_stock, generate_reports, _band_lo)
from src.features.optimizer import component_performance
from tests.test_backtest import write_snapshot, days


def make_ds_row(date, code, name='S', score=85.0, rank=1,
                compA=1.0, compB=0.0, rsi=50.0, ret5=None):
    """compA/compB는 획득률(0~1)."""
    return {
        'date': date, 'code': code, 'name': name, 'score': score,
        'rank': rank, 'penalty': 0.0, 'avail_max': 20.0,
        'compA_earned': compA * 10, 'compA_max': 10.0, 'compA_avail': 1,
        'compB_earned': compB * 10, 'compB_max': 10.0, 'compB_avail': 1,
        'ind_ma_up': 0, 'ind_macd_bull': 0, 'ind_obv_up': 0,
        'rsi': rsi, 'ret_5d': ret5, 'ret_10d': None,
    }


def synth_context(tmp_path, n_days=10, winners=True) -> Context:
    """compA 보유 종목이 이기는(winners=True) 합성 세계."""
    snap = str(tmp_path / 'snap')
    ds_days = days(n_days + 6)
    for i, d in enumerate(ds_days):
        write_snapshot(snap, d, {'000001': 100.0 + i, '000002': 100.0 - i * 0.3,
                                 '000003': 100.0})
    book = PriceBook(snap)
    rows = []
    for i, d in enumerate(ds_days[:n_days]):
        ret_good = 5.0 if winners else -5.0
        rows.append(make_ds_row(d, '000001', 'GOOD', 90.0, 1,
                                compA=1.0, compB=0.0, ret5=ret_good))
        rows.append(make_ds_row(d, '000002', 'BAD', 76.0, 2,
                                compA=0.0, compB=1.0, ret5=-3.0))
    # 타깃(마지막 날, 미래 수익 미상)
    rows.append(make_ds_row(ds_days[n_days], '000003', 'TARGET', 90.0, 1,
                            compA=1.0, compB=0.0, ret5=None))
    ds = pd.DataFrame(rows)
    regimes = pd.DataFrame([{'date': d, 'regime': 'Sideways', 'ret_median': 0,
                             'ret_mean': 0, 'breadth': 50, 'n_stocks': 3,
                             'cum_w': 0.0, 'vol_w': 0.5}
                            for d in ds_days])
    comps = ['compA', 'compB']
    perf = component_performance(ds, 5)
    return Context(ds=ds, book=book, regimes=regimes, perf=perf, comps=comps)


def target_of(ctx: Context) -> pd.Series:
    return ctx.ds[ctx.ds['name'] == 'TARGET'].iloc[0]


# ── Historical Similarity ───────────────────────────────────
def test_similar_finds_identical_profile_first(tmp_path):
    ctx = synth_context(tmp_path)
    sim = find_similar(ctx, target_of(ctx), n=5)
    assert len(sim) == 5
    # 타깃(compA=1.0)과 동일 프로필인 GOOD(000001)이 최근접이어야 함
    assert (sim.iloc[0]['code'] == '000001')
    assert sim.iloc[0]['distance'] < sim.iloc[-1]['distance'] + 1e-9
    # 과거만: 타깃 날짜의 행은 후보에 없음
    assert (sim['date'] < target_of(ctx)['date']).all()


def test_similarity_stats_and_expected_metrics(tmp_path):
    ctx = synth_context(tmp_path)
    sim_df = find_similar(ctx, target_of(ctx), n=10)
    good_only = sim_df[sim_df['code'] == '000001']
    stats = similarity_stats(ctx, good_only, 5)
    assert stats['avg_ret'] == pytest.approx(5.0)
    assert stats['win_rate'] == pytest.approx(100.0)
    assert stats['mdd'] is not None
    exp = expected_metrics(stats)
    assert exp['expected_return'] == pytest.approx(5.0)
    assert exp['risk_reward'] is None or exp['risk_reward'] > 0


# ── Confidence Score ────────────────────────────────────────
def test_confidence_higher_in_winning_world(tmp_path):
    ctx_w = synth_context(tmp_path / 'w', winners=True)
    ctx_l = synth_context(tmp_path / 'l', winners=False)
    sim_w = similarity_stats(ctx_w, find_similar(ctx_w, target_of(ctx_w)), 5)
    sim_l = similarity_stats(ctx_l, find_similar(ctx_l, target_of(ctx_l)), 5)
    conf_w = confidence_score(ctx_w, target_of(ctx_w), sim_w, 5)
    conf_l = confidence_score(ctx_l, target_of(ctx_l), sim_l, 5)
    assert conf_w.total > conf_l.total          # 이기는 세계에서 신뢰도 ↑
    assert 0 <= conf_l.total <= 100
    assert 0 <= conf_w.total <= 100
    assert set(conf_w.factors) == {'표본수', '국면일치', '구간성과',
                                   '신호안정성', '일관성'}


def test_confidence_empty_history_is_low_with_cautions(tmp_path):
    ctx = synth_context(tmp_path)
    ctx.ds = ctx.ds[ctx.ds['name'] == 'TARGET'].copy()   # 과거 전무
    sim = similarity_stats(ctx, find_similar(ctx, target_of(ctx)), 5)
    conf = confidence_score(ctx, target_of(ctx), sim, 5)
    assert conf.total < 50
    assert conf.grade in ('D', 'F')
    assert any('유사 사례' in c for c in conf.cautions)
    assert any('신뢰도 낮음' in c for c in conf.cautions)


def test_bad_held_signal_caution(tmp_path):
    """타깃이 '역사적으로 불리한' compB를 보유하면 주의사항에 떠야 함."""
    ctx = synth_context(tmp_path)
    tgt = target_of(ctx).copy()
    tgt['compA_earned'], tgt['compB_earned'] = 0.0, 10.0   # compB 보유로 변경
    sim = similarity_stats(ctx, find_similar(ctx, tgt), 5)
    conf = confidence_score(ctx, tgt, sim, 5)
    assert any('불리했던 신호' in c and 'compB' in c for c in conf.cautions)


def test_band_lo():
    assert _band_lo(97.0) == 95.0
    assert _band_lo(84.9) == 80.0
    assert _band_lo(60.0) == 0.0


# ── Explainability + Report ─────────────────────────────────
def test_analyze_stock_and_reports(tmp_path):
    ctx = synth_context(tmp_path)
    r = analyze_stock(ctx, target_of(ctx), reasons=['테스트 근거'], horizon=5)
    assert r['name'] == 'TARGET'
    assert r['regime'] == 'Sideways'
    assert r['reasons'] == ['테스트 근거']
    assert 'total' in r['confidence'] and 'grade' in r['confidence']
    assert r['similar']['n'] > 0

    paths = generate_reports([r], out_dir=str(tmp_path / 'out'), horizon=5)
    for key in ('json', 'md', 'html'):
        assert os.path.exists(paths[key]), key
    data = json.load(open(paths['json'], encoding='utf-8'))
    assert data['results'][0]['code'] == '000003'
    md = open(paths['md'], encoding='utf-8').read()
    assert '신뢰도' in md and '유사 과거 사례' in md
    # 주의사항 섹션은 cautions가 있을 때만 렌더 (이기는 세계 = 없어도 정상)
    assert ('주의사항' in md) == bool(r['cautions'])
