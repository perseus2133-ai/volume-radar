# -*- coding: utf-8 -*-
"""AI Score Optimizer 단위 테스트 — 합성 데이터셋으로 격리 검증."""
import os
import numpy as np
import pandas as pd
import pytest

from src.features.optimizer import (component_performance, weight_suggestions,
                                    correlation_matrix, ablation_importance,
                                    generate_reports, current_weights)


# ── 합성 데이터셋 ────────────────────────────────────────────
def synth_dataset(n_days=5, n_per_day=30, seed=42) -> pd.DataFrame:
    """규칙이 알려진 데이터: compA는 수익과 강한 양(+)의 관계,
    compB는 강한 음(-)의 관계, compC는 compA의 복제(중복 신호)."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        date = f'2026-01-{d+1:02d}'
        for i in range(n_per_day):
            a = float(rng.uniform(0, 10))          # compA earned (max 10)
            b = float(rng.uniform(0, 10))          # compB earned (max 10)
            c = a                                   # compC = compA 복제 → r=1.0
            ret = a - b + float(rng.normal(0, 0.5))   # 수익률: A 좋고 B 나쁨
            score = (a + b + c) / 30 * 100
            rows.append({
                'date': date, 'code': f'{i:06d}', 'name': f'S{i}',
                'score': round(score, 1), 'rank': i + 1, 'penalty': 0.0,
                'avail_max': 30.0,
                'compA_earned': a, 'compA_max': 10.0, 'compA_avail': 1,
                'compB_earned': b, 'compB_max': 10.0, 'compB_avail': 1,
                'compC_earned': c, 'compC_max': 10.0, 'compC_avail': 1,
                'compD_earned': 0.0, 'compD_max': 10.0, 'compD_avail': 0,  # 전량 결측
                'ind_ma_up': 0, 'ind_macd_bull': 0, 'ind_obv_up': 0, 'rsi': 50.0,
                'ret_5d': round(ret, 2), 'ret_10d': round(ret * 1.5, 2),
            })
    return pd.DataFrame(rows)


# ── ① Component Performance ────────────────────────────────
def test_component_performance_signs():
    ds = synth_dataset()
    perf = component_performance(ds, horizon=5).set_index('component')
    assert perf.loc['compA', 'corr_vs_ret'] > 0.5      # A: 강한 양의 상관
    assert perf.loc['compB', 'corr_vs_ret'] < -0.5     # B: 강한 음의 상관
    assert perf.loc['compA', 'win_lift'] > 0           # A 보유 시 승률 상승
    assert perf.loc['compB', 'win_lift'] < 0
    assert perf.loc['compD', 'n'] == 0                 # 결측 → 분석 불가
    assert '결측' in perf.loc['compD', 'note']


# ── ② Weight Suggestion ─────────────────────────────────────
def test_weight_suggestions_direction_and_renorm():
    ds = synth_dataset()
    perf = component_performance(ds, horizon=5)
    weights = {'compA': 10.0, 'compB': 10.0, 'compC': 10.0, 'compD': 10.0}
    sugg = weight_suggestions(perf, weights).set_index('component')
    assert sugg.loc['compA', 'suggested'] > sugg.loc['compA', 'current'] - 0.01
    assert sugg.loc['compB', 'suggested'] < sugg.loc['compB', 'current']
    assert sugg.loc['compD', 'suggested'] == 10.0      # 결측 → 현행 유지
    assert '분석 불가' in sugg.loc['compD', 'reason']
    # 합계는 명목 총합(40)과 일치 (±0.5 라운딩 허용)
    assert sugg['suggested'].sum() == pytest.approx(40.0, abs=0.5)
    assert '상관' in sugg.loc['compA', 'reason']


# ── ③ Correlation Matrix ────────────────────────────────────
def test_correlation_redundant_pair():
    ds = synth_dataset()
    mat, redundant = correlation_matrix(ds)
    assert mat.loc['compA', 'compC'] == pytest.approx(1.0)
    pairs = {(r['a'], r['b']) for r in redundant}
    assert ('compA', 'compC') in pairs
    assert all('비중 축소' in r['suggestion'] for r in redundant)


# ── ④ Feature Importance (ablation) ─────────────────────────
def test_ablation_removing_harmful_improves():
    ds = synth_dataset(n_days=6, n_per_day=40)
    abl = ablation_importance(ds, horizon=5, select_n=10).set_index('component')
    base_win = abl.loc['(baseline)', 'win_rate']
    # 해로운 compB 제거 → 승률 개선 (win_delta > 0)
    assert abl.loc['compB', 'win_delta'] > 0
    # 이로운 compA 제거 → 승률 악화
    assert abl.loc['compA', 'win_delta'] < 0
    assert abl.loc['(baseline)', 'win_delta'] == 0.0
    assert base_win is not None


# ── ⑤ Report 생성 ───────────────────────────────────────────
def test_generate_reports_files(tmp_path):
    ds = synth_dataset()
    perf = component_performance(ds, 5)
    sugg = weight_suggestions(perf, {'compA': 10, 'compB': 10, 'compC': 10, 'compD': 10})
    mat, red = correlation_matrix(ds)
    abl = ablation_importance(ds, 5, select_n=10)
    paths = generate_reports(perf, sugg, mat, red, abl, out_dir=str(tmp_path),
                             meta={'rows': len(ds), 'days': 5})
    for key in ('md', 'html', 'perf_csv', 'sugg_csv', 'abl_csv', 'corr_csv'):
        assert os.path.exists(paths[key]), key
    md = open(paths['md'], encoding='utf-8').read()
    assert '① 컴포넌트 성능' in md
    assert '② 가중치 제안' in md
    assert '알고리즘은 수정되지 않았다' in md
    html = open(paths['html'], encoding='utf-8').read()
    assert '<table' in html


# ── 빈 데이터 ────────────────────────────────────────────────
def test_empty_dataset_graceful():
    ds = synth_dataset(n_days=1, n_per_day=3)
    ds['ret_5d'] = None            # 수익률 전무 (미래 미축적)
    perf = component_performance(ds, 5)
    assert (perf['n'] == 0).all()
    sugg = weight_suggestions(perf, {'compA': 10, 'compB': 10,
                                     'compC': 10, 'compD': 10})
    assert (sugg['suggested'] == sugg['current']).all()   # 전부 현행 유지


# ── 현행 가중치 introspection ────────────────────────────────
def test_current_weights_match_algorithm():
    w = current_weights()
    assert w == {'거래대금': 15, '실적': 20, '밸류': 10, '수급': 15,
                 '신고가': 10, '거래량증가': 10, '차트': 10, '누적등장': 10}
    assert sum(w.values()) == 100
