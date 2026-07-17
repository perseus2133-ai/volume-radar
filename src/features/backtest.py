#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backtest Engine v1 — AI Score 추천의 사후 성과 검증.

설계 원칙
---------
- 데이터 주입식(디렉토리 경로 인자) → 단위 테스트에서 합성 데이터로 완전 검증
- 가격 소스: data/snapshots/{거래일}.csv.gz (종가) — 거래일 인덱스 기반
  으로 +5/+10/+20/+60 '거래일' 수익률 계산 (달력일 아님)
- 미래 스냅샷에 종목이 없으면 거래정지/상장폐지로 간주 → 통계에서 제외
  하되 별도 카운트 (survivorship bias를 숨기지 않는다)
- 미래 거래일이 아직 부족하면 status='insufficient' → 데이터가 쌓이면
  같은 명령으로 자동 확장
- MDD: 이벤트타임(추천일=0) 정렬 후 완주 추천들의 평균 자산곡선에서
  peak 대비 최대 낙폭
"""
from __future__ import annotations

import os
import csv
import json
import glob
import logging
import argparse
import datetime
import statistics
from dataclasses import dataclass, field, asdict

import pandas as pd

logger = logging.getLogger(__name__)

HORIZONS = (5, 10, 20, 60)
SCORE_BANDS = (95, 90, 85, 80, 75)

# 추천 근거 문자열 → 카테고리 정규화 (기능④ 집계 구조)
REASON_CATEGORIES = [
    ('거래대금', '거래대금 상위'),
    ('영업이익 성장', '실적 성장'),
    ('PEG', '저평가(PEG)'),
    ('신고가', '신고가 돌파/근접'),
    ('거래량', '거래량 폭증'),
    ('정배열', '정배열'),
    ('MACD', 'MACD 강세'),
    ('OBV', 'OBV 매집'),
    ('순매수', '수급(외인/기관)'),
    ('등장', '반복 등장'),
    ('과열', '과열 페널티'),
]


def categorize_reason(reason: str) -> str:
    for key, cat in REASON_CATEGORIES:
        if key in reason:
            return cat
    return '기타'


# ============================================================
# 가격북 — 스냅샷 로드 & 거래일 산술
# ============================================================
class PriceBook:
    """스냅샷 디렉토리에서 {거래일: {코드: 종가}} 를 로드."""

    def __init__(self, snap_dir: str):
        self.days: list[str] = []
        self._close: dict[str, dict[str, float]] = {}
        for path in sorted(glob.glob(os.path.join(snap_dir, '*.csv.gz'))):
            day = os.path.basename(path)[:-7]
            try:
                df = pd.read_csv(path, dtype={'code': str}, compression='gzip')
            except Exception:
                logger.warning('스냅샷 로드 실패: %s', path)
                continue
            df['code'] = df['code'].astype(str).str.zfill(6)
            closes = pd.to_numeric(df['close'], errors='coerce')
            self._close[day] = {c: float(p) for c, p in zip(df['code'], closes)
                                if pd.notna(p) and p > 0}
            self.days.append(day)
        self._idx = {d: i for i, d in enumerate(self.days)}

    def close(self, code: str, day: str) -> float | None:
        return self._close.get(day, {}).get(code)

    def future_day(self, day: str, horizon: int) -> str | None:
        i = self._idx.get(day)
        if i is None or i + horizon >= len(self.days):
            return None
        return self.days[i + horizon]

    def path(self, code: str, day: str, horizon: int) -> list[float] | None:
        """추천일부터 horizon 거래일까지의 종가 경로 (결측일은 직전가 carry).
        완주 불가(마지막 날 상장 없음)면 None."""
        i = self._idx.get(day)
        if i is None or i + horizon >= len(self.days):
            return None
        entry = self.close(code, day)
        if not entry:
            return None
        out, last = [], entry
        for d in self.days[i:i + horizon + 1]:
            p = self.close(code, d)
            if p:
                last = p
            out.append(last)
        # 최종일에 실제 가격이 없으면 거래정지/상폐 취급
        if self.close(code, self.days[i + horizon]) is None:
            return None
        return out


# ============================================================
# 데이터 모델
# ============================================================
@dataclass
class RecReturn:
    horizon: int
    status: str                  # 'ok' | 'insufficient' | 'suspended'
    ret_pct: float | None = None


@dataclass
class Recommendation:
    date: str
    code: str
    name: str
    score: float
    stars: int
    rank: int                    # 추천 당시 점수 순위 (파일 내 순번)
    reasons: list[str]
    entry_close: float | None
    returns: dict[int, RecReturn] = field(default_factory=dict)


@dataclass
class HorizonStats:
    horizon: int
    count: int = 0
    suspended: int = 0
    insufficient: int = 0
    mean: float | None = None
    median: float | None = None
    std: float | None = None
    win_rate: float | None = None
    max_gain: float | None = None
    max_loss: float | None = None
    mdd: float | None = None


@dataclass
class BacktestConfig:
    start: str
    end: str
    min_score: float = 0.0
    top_n: int | None = None
    horizons: tuple[int, ...] = HORIZONS

    def validate(self) -> None:
        try:
            s = datetime.date.fromisoformat(self.start)
            e = datetime.date.fromisoformat(self.end)
        except ValueError as exc:
            raise ValueError(f'날짜 형식 오류 (YYYY-MM-DD): {exc}') from exc
        if s > e:
            raise ValueError(f'시작일({self.start})이 종료일({self.end})보다 늦습니다')
        if self.top_n is not None and self.top_n <= 0:
            raise ValueError('top_n은 1 이상이어야 합니다')


# ============================================================
# 엔진
# ============================================================
def load_recommendations(cfg: BacktestConfig, scores_dir: str) -> list[Recommendation]:
    recs: list[Recommendation] = []
    for path in sorted(glob.glob(os.path.join(scores_dir, '*.json'))):
        day = os.path.basename(path)[:-5]
        if not (cfg.start <= day <= cfg.end):
            continue
        try:
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            logger.warning('점수 파일 로드 실패: %s', path)
            continue
        for rank, s in enumerate(data, start=1):
            if s.get('score', 0) < cfg.min_score:
                continue
            if cfg.top_n and rank > cfg.top_n:
                break            # 파일은 점수 내림차순 → rank 초과 시 중단
            recs.append(Recommendation(
                date=day, code=str(s['code']).zfill(6), name=s.get('name', ''),
                score=float(s['score']), stars=int(s.get('stars', 0)),
                rank=rank, reasons=list(s.get('reasons', [])), entry_close=None,
            ))
    return recs


def compute_returns(recs: list[Recommendation], book: PriceBook,
                    horizons: tuple[int, ...]) -> list[Recommendation]:
    out = []
    for r in recs:
        entry = book.close(r.code, r.date)
        if entry is None:
            # 추천일 가격 자체가 없음 (데이터 누락) → 전 구간 산출 불가
            r.entry_close = None
            r.returns = {h: RecReturn(h, 'suspended') for h in horizons}
            out.append(r)
            continue
        r.entry_close = entry
        for h in horizons:
            fday = book.future_day(r.date, h)
            if fday is None:
                r.returns[h] = RecReturn(h, 'insufficient')
                continue
            fclose = book.close(r.code, fday)
            if fclose is None:
                r.returns[h] = RecReturn(h, 'suspended')
                continue
            r.returns[h] = RecReturn(h, 'ok', round((fclose / entry - 1) * 100, 2))
        out.append(r)
    return out


def _equity_curve_mdd(recs: list[Recommendation], book: PriceBook, horizon: int) -> float | None:
    """이벤트타임 평균 자산곡선의 최대 낙폭(%). 완주(ok) 추천만 포함."""
    paths = []
    for r in recs:
        if r.returns.get(horizon) and r.returns[horizon].status == 'ok':
            p = book.path(r.code, r.date, horizon)
            if p and p[0] > 0:
                paths.append([v / p[0] for v in p])
    if not paths:
        return None
    n = horizon + 1
    curve = [sum(p[k] for p in paths) / len(paths) for k in range(n)]
    peak, mdd = curve[0], 0.0
    for v in curve:
        peak = max(peak, v)
        mdd = min(mdd, (v / peak - 1) * 100)
    return round(mdd, 2)


def horizon_stats(recs: list[Recommendation], book: PriceBook, horizon: int) -> HorizonStats:
    st = HorizonStats(horizon=horizon)
    rets = []
    for r in recs:
        rr = r.returns.get(horizon)
        if rr is None:
            continue
        if rr.status == 'ok':
            rets.append(rr.ret_pct)
        elif rr.status == 'suspended':
            st.suspended += 1
        else:
            st.insufficient += 1
    st.count = len(rets)
    if rets:
        st.mean = round(statistics.mean(rets), 2)
        st.median = round(statistics.median(rets), 2)
        st.std = round(statistics.pstdev(rets), 2) if len(rets) > 1 else 0.0
        st.win_rate = round(sum(1 for x in rets if x > 0) / len(rets) * 100, 1)
        st.max_gain = max(rets)
        st.max_loss = min(rets)
        st.mdd = _equity_curve_mdd(recs, book, horizon)
    return st


def band_stats(recs: list[Recommendation], book: PriceBook,
               horizons: tuple[int, ...]) -> dict[str, dict]:
    out = {}
    for band in SCORE_BANDS:
        subset = [r for r in recs if r.score >= band]
        out[f'{band}+'] = {
            'recommendations': len(subset),
            'horizons': {h: asdict(horizon_stats(subset, book, h)) for h in horizons},
        }
    return out


def reason_stats(recs: list[Recommendation], horizons: tuple[int, ...]) -> dict[str, dict]:
    """기능④: 근거 카테고리별 성과 집계 구조."""
    agg: dict[str, dict] = {}
    for r in recs:
        cats = {categorize_reason(x) for x in r.reasons}
        for cat in cats:
            slot = agg.setdefault(cat, {'count': 0,
                                        **{f'rets_{h}': [] for h in horizons}})
            slot['count'] += 1
            for h in horizons:
                rr = r.returns.get(h)
                if rr and rr.status == 'ok':
                    slot[f'rets_{h}'].append(rr.ret_pct)
    out = {}
    for cat, slot in agg.items():
        row = {'count': slot['count']}
        for h in horizons:
            rets = slot[f'rets_{h}']
            row[f'mean_{h}d'] = round(statistics.mean(rets), 2) if rets else None
            row[f'win_{h}d'] = (round(sum(1 for x in rets if x > 0) / len(rets) * 100, 1)
                                if rets else None)
            row[f'n_{h}d'] = len(rets)
        out[cat] = row
    return dict(sorted(out.items(), key=lambda kv: -kv[1]['count']))


@dataclass
class BacktestResult:
    config: dict
    records: list[Recommendation]
    summary: dict[int, HorizonStats]
    bands: dict[str, dict]
    reasons: dict[str, dict]
    trading_days_available: int


def run_backtest(cfg: BacktestConfig, scores_dir: str, snap_dir: str) -> BacktestResult:
    cfg.validate()
    book = PriceBook(snap_dir)
    recs = load_recommendations(cfg, scores_dir)
    recs = compute_returns(recs, book, cfg.horizons)
    summary = {h: horizon_stats(recs, book, h) for h in cfg.horizons}
    return BacktestResult(
        config={'start': cfg.start, 'end': cfg.end,
                'min_score': cfg.min_score, 'top_n': cfg.top_n},
        records=recs,
        summary=summary,
        bands=band_stats(recs, book, cfg.horizons),
        reasons=reason_stats(recs, cfg.horizons),
        trading_days_available=len(book.days),
    )


# ============================================================
# 결과 저장 (⑤)
# ============================================================
def save_report(result: BacktestResult, out_dir: str,
                asof: str | None = None) -> dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    asof = asof or datetime.date.today().isoformat()
    paths = {}

    # report.json (전체)
    paths['report'] = os.path.join(out_dir, f'{asof}_report.json')
    with open(paths['report'], 'w', encoding='utf-8') as f:
        json.dump({
            'config': result.config,
            'trading_days_available': result.trading_days_available,
            'summary': {h: asdict(s) for h, s in result.summary.items()},
            'bands': result.bands,
            'reasons': result.reasons,
            'records': [
                {'date': r.date, 'code': r.code, 'name': r.name,
                 'score': r.score, 'stars': r.stars, 'rank': r.rank,
                 'reasons': r.reasons, 'entry_close': r.entry_close,
                 'returns': {h: asdict(rr) for h, rr in r.returns.items()}}
                for r in result.records],
        }, f, ensure_ascii=False, indent=1)

    # summary.csv (구간별 성과)
    paths['summary'] = os.path.join(out_dir, 'summary.csv')
    with open(paths['summary'], 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['horizon_days', 'count', 'mean_pct', 'median_pct', 'std',
                    'win_rate_pct', 'max_gain_pct', 'max_loss_pct', 'mdd_pct',
                    'suspended', 'insufficient'])
        for h, s in result.summary.items():
            w.writerow([h, s.count, s.mean, s.median, s.std, s.win_rate,
                        s.max_gain, s.max_loss, s.mdd, s.suspended, s.insufficient])

    # performance.csv (추천별)
    paths['performance'] = os.path.join(out_dir, 'performance.csv')
    with open(paths['performance'], 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        head = ['date', 'code', 'name', 'score', 'stars', 'rank', 'entry_close']
        head += sum(([f'ret_{h}d', f'status_{h}d'] for h in HORIZONS), [])
        head += ['reasons']
        w.writerow(head)
        for r in result.records:
            row = [r.date, r.code, r.name, r.score, r.stars, r.rank, r.entry_close]
            for h in HORIZONS:
                rr = r.returns.get(h)
                row += [rr.ret_pct if rr else None, rr.status if rr else None]
            row.append(' | '.join(r.reasons))
            w.writerow(row)

    # score_analysis.csv (점수 구간별)
    paths['score_analysis'] = os.path.join(out_dir, 'score_analysis.csv')
    with open(paths['score_analysis'], 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['band', 'recommendations', 'horizon_days', 'count',
                    'mean_pct', 'win_rate_pct', 'max_loss_pct', 'mdd_pct'])
        for band, info in result.bands.items():
            for h, s in info['horizons'].items():
                w.writerow([band, info['recommendations'], h, s['count'],
                            s['mean'], s['win_rate'], s['max_loss'], s['mdd']])
    return paths


# ============================================================
# CLI
# ============================================================
def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    ap = argparse.ArgumentParser(description='AI Score 백테스트')
    ap.add_argument('--start', required=True)
    ap.add_argument('--end', required=True)
    ap.add_argument('--min-score', type=float, default=0.0)
    ap.add_argument('--top-n', type=int, default=None)
    ap.add_argument('--scores-dir', default=os.path.join('data', 'scores'))
    ap.add_argument('--snap-dir', default=os.path.join('data', 'snapshots'))
    ap.add_argument('--out-dir', default=os.path.join('data', 'backtest'))
    args = ap.parse_args()

    cfg = BacktestConfig(start=args.start, end=args.end,
                         min_score=args.min_score, top_n=args.top_n)
    result = run_backtest(cfg, args.scores_dir, args.snap_dir)
    paths = save_report(result, args.out_dir)

    print(f"추천 {len(result.records)}건 · 가용 거래일 {result.trading_days_available}일")
    for h, s in result.summary.items():
        if s.count:
            print(f"  {h:>2}일: n={s.count:<3} 평균 {s.mean:+.2f}% · 승률 {s.win_rate:.0f}% "
                  f"· 최대 {s.max_gain:+.1f}/{s.max_loss:+.1f}% · MDD {s.mdd}%")
        else:
            print(f"  {h:>2}일: 데이터 부족 (미래 거래일 미축적 {s.insufficient}건)")
    print('저장:', ', '.join(paths.values()))


if __name__ == '__main__':
    main()
