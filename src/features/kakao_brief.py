#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""카카오톡 Morning Brief 전송 — '나에게 보내기' API.

대시보드(daily_dashboard.json)를 카톡 메시지로 요약해 매일 아침 전송한다.
카카오 메모 API의 text 템플릿은 200자 제한 → 핵심만 압축 + 대시보드 링크.

필요 환경변수 (GitHub Secrets):
  KAKAO_REST_KEY       카카오 개발자 앱의 REST API 키
  KAKAO_REFRESH_TOKEN  최초 1회 kakao_auth.py 로 발급
  BRIEF_URL (선택)     버튼 링크 (기본: GitHub의 daily_dashboard.md)

토큰 수명: access token ~6시간 → 매 전송마다 refresh token으로 재발급.
refresh token은 2개월 유효, 잔여 1개월 미만이면 갱신 응답에 새 토큰이
포함됨 → 로그에 경고 출력 (GitHub Secret 수동 갱신 필요, 월 1회 미만).

전송 스킵 규칙: 브리핑의 데이터 기준일이 '어제(KST)'가 아니면 중복
브리핑(주말/공휴일)으로 보고 스킵. --force 로 무시 가능.
"""
from __future__ import annotations

import os
import json
import logging
import argparse
import datetime
from zoneinfo import ZoneInfo

import requests

logger = logging.getLogger(__name__)

KST = ZoneInfo('Asia/Seoul')
DASH_JSON = os.path.join('data', 'dashboard', 'daily_dashboard.json')
TOKEN_URL = 'https://kauth.kakao.com/oauth/token'
SEND_URL = 'https://kapi.kakao.com/v2/api/talk/memo/default/send'
DEFAULT_BRIEF_URL = ('https://github.com/perseus2133-ai/volume-radar'
                     '/blob/main/data/dashboard/daily_dashboard.md')
MAX_TEXT = 200          # 카카오 text 템플릿 제한


# ============================================================
# 메시지 포맷
# ============================================================
def build_text(brief: dict, watchlist: list[dict] | None = None) -> str:
    """Morning Brief → 200자 이내 카톡 텍스트."""
    c = brief.get('counts', {})
    buy = c.get('STRONG BUY', 0) + c.get('BUY', 0)
    date = brief.get('date', '')[5:].replace('-', '/')
    regime_map = {'High Volatility': '변동성', 'Bull': '상승', 'Bear': '하락',
                  'Sideways': '횡보', 'Unknown': '?'}
    regime = regime_map.get(brief.get('regime', ''), brief.get('regime', '?'))

    lines = [f"☀️AI브리핑 {date} 국면:{regime}",
             f"매수{buy} 관심{c.get('WATCH', 0)} 보류{c.get('HOLD', 0)} "
             f"매도{c.get('SELL', 0)} 회피{c.get('AVOID', 0)}"]

    strong = brief.get('strong_sectors') or []
    if strong:
        lines.append('강세: ' + '·'.join(s['sector'][:6] for s in strong[:2]))
    weak = brief.get('weak_sectors') or []
    if weak:
        lines.append('약세: ' + '·'.join(s['sector'][:6] for s in weak[:2]))

    picks = brief.get('top_picks') or []
    if picks and buy > 0:
        lines.append('추천: ' + '·'.join(p['name'][:6] for p in picks[:3]))
    if watchlist:
        lines.append(f"🔭내일후보 {len(watchlist)}: "
                     + '·'.join(w['name'][:6] for w in watchlist[:2]))

    text = '\n'.join(lines)
    return text[:MAX_TEXT]


# ============================================================
# 카카오 API
# ============================================================
def refresh_access_token(rest_key: str, refresh_token: str,
                         client_secret: str = '') -> tuple[str, str | None]:
    """access token 재발급. 반환: (access, 새 refresh 또는 None).
    앱에서 '클라이언트 시크릿 활성화'가 ON이면 client_secret 필수."""
    payload = {
        'grant_type': 'refresh_token',
        'client_id': rest_key,
        'refresh_token': refresh_token,
    }
    if client_secret:
        payload['client_secret'] = client_secret
    r = requests.post(TOKEN_URL, data=payload, timeout=15)
    r.raise_for_status()
    d = r.json()
    return d['access_token'], d.get('refresh_token')


def send_memo(access_token: str, text: str, url: str) -> None:
    template = {
        'object_type': 'text',
        'text': text,
        'link': {'web_url': url, 'mobile_web_url': url},
        'button_title': '대시보드 보기',
    }
    r = requests.post(SEND_URL,
                      headers={'Authorization': f'Bearer {access_token}'},
                      data={'template_object': json.dumps(template,
                                                          ensure_ascii=False)},
                      timeout=15)
    r.raise_for_status()
    if r.json().get('result_code') != 0:
        raise RuntimeError(f'카카오 전송 실패: {r.text}')


def should_send(brief: dict, today: datetime.date | None = None) -> tuple[bool, str]:
    """브리핑 데이터 기준일 == 어제(KST) 일 때만 전송 (주말/공휴일 중복 방지)."""
    today = today or datetime.datetime.now(KST).date()
    bdate = brief.get('date', '')
    expect = (today - datetime.timedelta(days=1)).isoformat()
    if bdate != expect:
        return False, f'브리핑 기준일 {bdate} ≠ 어제 {expect} — 휴장/중복으로 스킵'
    return True, 'ok'


# ============================================================
# CLI
# ============================================================
def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    ap = argparse.ArgumentParser(description='카카오톡 Morning Brief 전송')
    ap.add_argument('--dry-run', action='store_true', help='전송 없이 메시지만 출력')
    ap.add_argument('--force', action='store_true', help='기준일 검사 무시')
    args = ap.parse_args()

    if not os.path.exists(DASH_JSON):
        print('대시보드 파일 없음 — 스킵')
        return
    with open(DASH_JSON, encoding='utf-8') as f:
        dash = json.load(f)
    brief = dash.get('brief', {})
    text = build_text(brief, dash.get('watchlist'))

    ok, why = should_send(brief)
    if not ok and not args.force:
        print(f'전송 스킵: {why}')
        return

    print(f'--- 메시지 ({len(text)}자) ---\n{text}\n---')
    if args.dry_run:
        return

    rest_key = os.environ.get('KAKAO_REST_KEY', '')
    refresh = os.environ.get('KAKAO_REFRESH_TOKEN', '')
    client_secret = os.environ.get('KAKAO_CLIENT_SECRET', '')
    if not rest_key or not refresh:
        print('KAKAO_REST_KEY / KAKAO_REFRESH_TOKEN 미설정 — 스킵 '
              '(설정법: KAKAO_SETUP.md)')
        return

    access, new_refresh = refresh_access_token(rest_key, refresh, client_secret)
    url = os.environ.get('BRIEF_URL', DEFAULT_BRIEF_URL)
    send_memo(access, text, url)
    print('✅ 카카오톡 전송 완료')
    if new_refresh:
        print('⚠️⚠️ 새 REFRESH TOKEN 발급됨 — GitHub Secret '
              'KAKAO_REFRESH_TOKEN 을 아래 값으로 교체하세요:')
        print(new_refresh)


if __name__ == '__main__':
    main()
