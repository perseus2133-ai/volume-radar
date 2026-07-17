# -*- coding: utf-8 -*-
"""Decision Engine 단위 테스트 — 합성 분석 결과로 사다리/가드레일 검증."""
import os
import json
import pytest

from src.features.decision import (decide, generate_reports, risk_level,
                                   decision_grade, hist_performance_score,
                                   rr_score, DECISIONS)


def analysis(score=90.0, conf=80.0, grade='A', regime='Bull',
             n=30, win=70.0, avg=5.0, mdd=-3.0,
             er=5.0, dd=-3.0, rr=1.5, reasons=None, cautions=None):
    return {
        'date': '2026-01-10', 'code': '000001', 'name': 'T', 'rank': 1,
        'score': score,
        'confidence': {'total': conf, 'grade': grade, 'factors': {}},
        'regime': regime,
        'similar': {'n': n, 'avg_ret': avg, 'win_rate': win, 'mdd': mdd,
                    'worst': -8.0, 'best': 12.0, 'cases': []},
        'expected': {'expected_return': er, 'expected_drawdown': dd,
                     'risk_reward': rr},
        'reasons': reasons or ['거래대금 5위', '실적 성장 +100%'],
        'cautions': cautions or [],
    }


# ── 사다리: 6개 판단 전부 도달 가능 ─────────────────────────
def test_strong_buy_path():
    d = decide(analysis())
    assert d['recommendation'] == 'STRONG BUY'
    assert d['decision_grade'] in ('A', 'A+')
    assert d['risk_level'] == 'LOW'
    assert 0 <= d['decision_score'] <= 100


def test_buy_path():
    d = decide(analysis(score=75, conf=60, grade='C', regime='Sideways',
                        win=60, avg=3.0, rr=1.2, er=3.0, dd=-4.0))
    assert d['recommendation'] == 'BUY'
    assert d['guardrails'] == []


def test_watch_path():
    d = decide(analysis(score=68, conf=45, grade='D', regime='Sideways',
                        win=52, avg=0.5, rr=0.4, er=0.5, dd=-6.0))
    assert d['recommendation'] == 'WATCH'


def test_hold_sell_avoid_paths():
    hold = decide(analysis(score=55, conf=40, grade='D', regime='Sideways',
                           win=40, avg=-1.0, rr=-0.3, er=-1.0, dd=-8.0))
    assert hold['recommendation'] == 'HOLD'
    sell = decide(analysis(score=50, conf=38, grade='D', regime='Sideways',
                           win=35, avg=-2.0, rr=-0.6, er=-2.0, dd=-9.0))
    assert sell['recommendation'] == 'SELL'
    avoid = decide(analysis(score=20, conf=20, grade='F', regime='Bear',
                            win=10, avg=-10.0, rr=-2.0, er=-10.0, dd=-25.0))
    assert avoid['recommendation'] == 'AVOID'


# ── 가드레일 ────────────────────────────────────────────────
def test_guardrail_low_confidence_blocks_buy():
    """DScore는 매수권이지만 신뢰도 F → WATCH 강등."""
    d = decide(analysis(conf=30, grade='F'))    # 나머지는 전부 강세 입력
    assert d['recommendation'] == 'WATCH'
    assert any('매수 금지' in g for g in d['guardrails'])


def test_guardrail_mid_confidence_demotes_strong_buy():
    """SB 조건 충족이지만 신뢰도 55(<65) → BUY 강등."""
    d = decide(analysis(score=95, conf=55, grade='C', win=75, avg=6.0, rr=1.5))
    assert d['recommendation'] == 'BUY'
    assert any('STRONG BUY 강등' in g for g in d['guardrails'])


def test_guardrail_negative_expected_forces_sell():
    d = decide(analysis(score=70, conf=55, grade='C', er=-8.0,
                        avg=-8.0, win=25, rr=-0.9, dd=-9.0))
    assert d['recommendation'] == 'SELL'
    assert any('기대수익' in g for g in d['guardrails'])


# ── 국면 조정 ───────────────────────────────────────────────
def test_regime_adjustment_bull_vs_highvol():
    bull = decide(analysis(regime='Bull'))
    hv = decide(analysis(regime='High Volatility'))
    assert bull['decision_score'] - hv['decision_score'] == pytest.approx(13.0)
    assert bull['inputs']['regime_adj'] == 5.0
    assert hv['inputs']['regime_adj'] == -8.0


# ── 점수화 헬퍼 ─────────────────────────────────────────────
def test_hist_and_rr_scores():
    pts, _ = hist_performance_score({'n': 10, 'win_rate': 60.0, 'avg_ret': 5.0})
    assert pts == pytest.approx(0.5 * 60 + 0.5 * 60)
    assert hist_performance_score({})[0] == 50.0
    assert rr_score(None)[0] == 50.0
    assert rr_score(2.5)[0] == 100.0
    assert rr_score(1.0)[0] == 75.0
    assert rr_score(-2.5)[0] == 0.0
    assert rr_score(0.5)[0] == pytest.approx(62.5)


def test_risk_levels():
    assert risk_level(analysis(dd=-2.0, rr=1.5)) == 'LOW'
    assert risk_level(analysis(dd=-7.0)) == 'MEDIUM'
    assert risk_level(analysis(dd=-12.0)) == 'HIGH'
    assert risk_level(analysis(regime='High Volatility', dd=-2.0)) == 'HIGH'
    assert risk_level(analysis(dd=-25.0)) == 'VERY HIGH'
    assert risk_level(analysis(grade='F', er=-5.0, dd=-3.0)) == 'VERY HIGH'


def test_decision_grades():
    assert decision_grade(90) == 'A+'
    assert decision_grade(75) == 'A'
    assert decision_grade(65) == 'B'
    assert decision_grade(55) == 'C'
    assert decision_grade(45) == 'D'
    assert decision_grade(44.9) == 'F'


# ── Reason Summary + Report ─────────────────────────────────
def test_reason_summary_contents():
    d = decide(analysis())
    s = d['reason_summary']
    assert 'AI Score' in s and '신뢰도' in s and 'Bull' in s
    assert 'STRONG BUY' in s
    assert '거래대금 5위' in s


def test_generate_reports(tmp_path):
    ds = [decide(analysis()),
          decide(analysis(score=20, conf=20, grade='F', regime='Bear',
                          win=10, avg=-10.0, rr=-2.0, er=-10.0, dd=-25.0,
                          cautions=['테스트 주의']))]
    paths = generate_reports(ds, out_dir=str(tmp_path))
    for key in ('json', 'md', 'html'):
        assert os.path.exists(paths[key]), key
    data = json.load(open(paths['json'], encoding='utf-8'))
    assert len(data['decisions']) == 2
    md = open(paths['md'], encoding='utf-8').read()
    assert 'STRONG BUY' in md and 'AVOID' in md and '테스트 주의' in md
    html = open(paths['html'], encoding='utf-8').read()
    assert 'Decision Report' in html


def test_all_decisions_are_known():
    for kw in [dict(), dict(score=20, conf=20, grade='F', regime='Bear',
                            win=5, avg=-12.0, rr=-2.0, er=-12.0, dd=-30.0)]:
        assert decide(analysis(**kw))['recommendation'] in DECISIONS
