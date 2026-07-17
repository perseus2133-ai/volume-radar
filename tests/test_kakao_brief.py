# -*- coding: utf-8 -*-
"""카카오 브리핑 메시지 포맷/전송조건 단위 테스트 (네트워크 없음)."""
import datetime

from src.features.kakao_brief import build_text, should_send, MAX_TEXT


def brief(date='2026-07-16', regime='High Volatility', buy=0, watch=0,
          hold=0, sell=4, avoid=6, strong=None, weak=None, picks=None):
    return {
        'date': date, 'regime': regime,
        'counts': {'STRONG BUY': 0, 'BUY': buy, 'WATCH': watch,
                   'HOLD': hold, 'SELL': sell, 'AVOID': avoid},
        'strong_sectors': strong or [],
        'weak_sectors': weak or [],
        'top_picks': picks or [],
    }


def test_text_within_kakao_limit_and_contents():
    b = brief(strong=[{'sector': '가정용품', 'avg_ret': 3.2},
                      {'sector': '무선통신서비스', 'avg_ret': 3.0}],
              weak=[{'sector': '반도체와반도체장비', 'avg_ret': -6.2}])
    t = build_text(b, watchlist=[{'name': '한화오션'}, {'name': 'S-Oil'}])
    assert len(t) <= MAX_TEXT
    assert '07/16' in t
    assert '변동성' in t                      # regime 한글 매핑
    assert '매도4' in t and '회피6' in t
    assert '가정용품' in t
    assert '내일후보 2' in t and '한화오션' in t


def test_text_buy_day_shows_picks():
    b = brief(buy=2, picks=[{'name': '삼성전자'}, {'name': 'S-Oil'}])
    t = build_text(b)
    assert '매수2' in t
    assert '추천:' in t and '삼성전자' in t


def test_text_never_exceeds_limit_with_long_names():
    b = brief(strong=[{'sector': '아주아주아주긴업종이름' * 3, 'avg_ret': 1.0}] * 3,
              weak=[{'sector': '매우매우매우긴업종이름' * 3, 'avg_ret': -1.0}] * 3,
              buy=3, picks=[{'name': '아주아주긴종목이름입니다' * 2}] * 5)
    t = build_text(b, watchlist=[{'name': '엄청나게긴종목이름' * 2}] * 9)
    assert len(t) <= MAX_TEXT


def test_should_send_only_for_yesterday():
    today = datetime.date(2026, 7, 17)
    ok, _ = should_send(brief(date='2026-07-16'), today)
    assert ok
    ok2, why = should_send(brief(date='2026-07-14'), today)   # 주말 지난 중복
    assert not ok2 and '스킵' in why
