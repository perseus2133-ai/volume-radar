#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AI Score — 종목별 0~100점 종합 점수 (순수 함수, 파일 IO 없음).

컴포넌트별 (획득, 만점, 가용) 구조. 데이터 결측 컴포넌트는 제외하고
가용 만점 합 기준으로 100점 재정규화 → 커버리지와 함께 반환.
수급(외인/기관)처럼 소스 사정으로 통째로 비는 데이터가 있어도
점수 체계가 죽지 않도록 설계되었다.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RSI_OVERHEAT = 78.0
GROWTH_CAP = 300.0


@dataclass
class Component:
    earned: float
    maximum: float
    available: bool
    note: str = ""


@dataclass
class ScoreResult:
    code: str
    name: str
    score: float                 # 0~100 (재정규화 후)
    stars: int                   # 1~5
    coverage_pct: float          # 가용 만점 합 (명목 100 기준)
    components: dict[str, Component] = field(default_factory=dict)
    penalty: float = 0.0

    def reasons(self) -> list[str]:
        """리포트용 근거 문자열 (기능 ⑥ AI 리포트에서 재사용)."""
        out = []
        for label, c in self.components.items():
            if c.available and c.note:
                out.append(c.note)
        if self.penalty > 0:
            out.append(f"과열 페널티 -{self.penalty:.0f} (RSI 고점)")
        return out


def _num(v) -> float:
    v = pd.to_numeric(v, errors="coerce")
    return float(v) if pd.notna(v) else np.nan


def _c_turnover(rank: float) -> Component:
    if np.isnan(rank):
        return Component(0, 15, False)
    if rank <= 10:
        return Component(15, 15, True, f"거래대금 {rank:.0f}위")
    if rank <= 30:
        return Component(12, 15, True, f"거래대금 {rank:.0f}위")
    if rank <= 100:
        return Component(8, 15, True, f"거래대금 {rank:.0f}위")
    if rank <= 300:
        return Component(4, 15, True)
    return Component(0, 15, True)


def _c_earnings(op_gmax: float, profitable: bool) -> Component:
    if np.isnan(op_gmax):
        return Component(0, 20, False)
    pts = min(max(op_gmax, 0), GROWTH_CAP) / GROWTH_CAP * 16 + (4 if profitable else 0)
    note = f"영업이익 성장 +{op_gmax:.0f}%" if op_gmax >= 30 else ""
    return Component(round(pts, 1), 20, True, note)


def _c_value(per: float, g26: float) -> Component:
    if np.isnan(per) or per <= 0 or np.isnan(g26) or g26 <= 0:
        return Component(0, 10, False)
    peg = (per / (1 + g26 / 100)) / g26
    if peg <= 0.5:
        return Component(10, 10, True, f"PEG {peg:.2f} (저평가)")
    if peg <= 1.0:
        return Component(8, 10, True, f"PEG {peg:.2f}")
    if peg <= 1.5:
        return Component(5, 10, True)
    if peg <= 2.0:
        return Component(3, 10, True)
    return Component(1, 10, True)


def _c_flow(f5: float, i5: float, f20: float, i20: float) -> Component:
    vals = [f5, i5, f20, i20]
    if all(np.isnan(v) for v in vals):
        return Component(0, 15, False, "수급 데이터 없음")
    pts, notes = 0.0, []
    if not np.isnan(f5) and f5 > 0:
        pts += 4.5
        notes.append("외인 5일 순매수")
    if not np.isnan(i5) and i5 > 0:
        pts += 4.5
        notes.append("기관 5일 순매수")
    if not np.isnan(f20) and f20 > 0:
        pts += 3.0
    if not np.isnan(i20) and i20 > 0:
        pts += 3.0
    if len(notes) == 2:
        notes = ["외인+기관 동시 순매수"]
    return Component(pts, 15, True, " · ".join(notes))


def _c_high(close: float, high60: float) -> Component:
    if np.isnan(close) or np.isnan(high60) or high60 <= 0:
        return Component(0, 10, False)
    ratio = close / high60
    if ratio >= 1.0:
        return Component(10, 10, True, "60일 신고가 돌파")
    if ratio >= 0.97:
        return Component(7, 10, True, "60일 고가 근접")
    if ratio >= 0.90:
        return Component(4, 10, True)
    return Component(0, 10, True)


def _c_volume(mult: float) -> Component:
    if np.isnan(mult):
        return Component(0, 10, False)
    if mult >= 3.0:
        return Component(10, 10, True, f"거래량 {mult:.1f}배 폭증")
    if mult >= 2.0:
        return Component(7, 10, True, f"거래량 {mult:.1f}배")
    if mult >= 1.5:
        return Component(4, 10, True)
    return Component(0, 10, True)


def _c_chart(ma: str, macd: str, obv: str) -> Component:
    known = any(isinstance(v, str) and v for v in (ma, macd, obv))
    if not known:
        return Component(0, 10, False)
    pts, notes = 0.0, []
    if ma == "up":
        pts += 4
        notes.append("정배열")
    if macd in ("bull", "bull_cross"):
        pts += 3
        notes.append("MACD 강세")
    if obv == "up":
        pts += 3
        notes.append("OBV 매집")
    return Component(pts, 10, True, " · ".join(notes))


def _c_persistence(count: int) -> Component:
    if count >= 5:
        return Component(10, 10, True, f"10거래일 중 {count}회 등장")
    if count >= 3:
        return Component(7, 10, True, f"10거래일 중 {count}회 등장")
    if count >= 2:
        return Component(4, 10, True)
    if count >= 1:
        return Component(2, 10, True)
    return Component(0, 10, True)


def _stars(score: float) -> int:
    if score >= 90:
        return 5
    if score >= 80:
        return 4
    if score >= 65:
        return 3
    if score >= 50:
        return 2
    return 1


def compute_ai_score(row: pd.Series, turnover_rank: float,
                     acc_count: int = 0) -> ScoreResult:
    """단일 종목 AI Score. row = ai2 CSV 행 (pd.Series)."""
    op25, op26 = _num(row.get("영업이익_2025")), _num(row.get("영업이익_2026"))
    profitable = bool((not np.isnan(op25) and op25 > 0)
                      or (not np.isnan(op26) and op26 > 0))
    g26 = _num(row.get("영업이익_성장률_2026"))
    if np.isnan(g26):
        g26 = _num(row.get("영업이익_성장률_2025"))

    comps = {
        "거래대금": _c_turnover(turnover_rank),
        "실적": _c_earnings(_num(row.get("영업이익_최대성장률")), profitable),
        "밸류": _c_value(_num(row.get("PER")), g26),
        "수급": _c_flow(_num(row.get("외인_5d")), _num(row.get("기관_5d")),
                        _num(row.get("외인_20d")), _num(row.get("기관_20d"))),
        "신고가": _c_high(_num(row.get("현재가")), _num(row.get("저항선"))),
        "거래량증가": _c_volume(_num(row.get("거래량배수"))),
        "차트": _c_chart(row.get("MA_align", ""), row.get("MACD_signal", ""),
                         row.get("OBV_trend", "")),
        "누적등장": _c_persistence(acc_count),
    }

    avail_max = sum(c.maximum for c in comps.values() if c.available)
    earned = sum(c.earned for c in comps.values() if c.available)

    penalty = 0.0
    rsi = _num(row.get("RSI"))
    if not np.isnan(rsi) and rsi >= RSI_OVERHEAT:
        penalty = 5.0

    score = (earned / avail_max * 100.0 - penalty) if avail_max > 0 else 0.0
    score = round(max(0.0, min(100.0, score)), 1)

    return ScoreResult(
        code=str(row.get("종목코드", "")).zfill(6),
        name=str(row.get("종목명", "")),
        score=score, stars=_stars(score),
        coverage_pct=round(avail_max, 1),
        components=comps, penalty=penalty,
    )


def score_universe(ai2_df: pd.DataFrame, acc_counts: dict[str, int],
                   min_turnover_won: float = 3e9) -> list[ScoreResult]:
    """전 종목 채점 (거래대금 하한으로 유동성 미달 제외). 점수 내림차순."""
    df = ai2_df.copy()
    df["종목코드"] = df["종목코드"].astype(str).str.zfill(6)
    df["_turnover"] = (pd.to_numeric(df["현재가"], errors="coerce")
                       * pd.to_numeric(df["Recent_Volume"], errors="coerce"))
    df = df[df["_turnover"] > 0].sort_values("_turnover", ascending=False)
    df["_rank"] = range(1, len(df) + 1)

    results: list[ScoreResult] = []
    for _, row in df[df["_turnover"] >= min_turnover_won].iterrows():
        try:
            r = compute_ai_score(row, float(row["_rank"]),
                                 acc_counts.get(row["종목코드"], 0))
            results.append(r)
        except Exception:
            logger.exception("채점 실패: %s", row.get("종목코드"))
    results.sort(key=lambda r: r.score, reverse=True)
    return results
