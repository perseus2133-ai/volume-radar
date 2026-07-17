#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""카카오 최초 1회 인증 헬퍼 — refresh token 발급.

사전 준비 (KAKAO_SETUP.md 참고):
  1. developers.kakao.com 에서 앱 생성 → REST API 키 확보
  2. 앱 설정 > 플랫폼 > Web > http://localhost:3000 등록
  3. 제품 설정 > 카카오 로그인 활성화 + Redirect URI http://localhost:3000
  4. 동의항목 > '카카오톡 메시지 전송(talk_message)' 선택 동의 설정

실행: python kakao_auth.py
  → 브라우저가 열리면 로그인/동의 → localhost:3000 으로 이동된 주소창의
    code=XXXX 값을 복사해 붙여넣기 → refresh token 출력
"""
import sys
import webbrowser

import requests

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

REDIRECT = 'http://localhost:3000'


def main() -> None:
    rest_key = input('REST API 키 입력: ').strip()
    auth_url = ('https://kauth.kakao.com/oauth/authorize'
                f'?client_id={rest_key}&redirect_uri={REDIRECT}'
                '&response_type=code&scope=talk_message')
    print('\n브라우저에서 카카오 로그인/동의를 진행하세요:')
    print(' ', auth_url)
    webbrowser.open(auth_url)
    print('\n동의 후 이동된 주소는 http://localhost:3000/?code=XXXX 형태입니다')
    print('(페이지가 안 열려도 정상 — 주소창의 code 값만 복사)')
    code = input('code 값 입력: ').strip()

    r = requests.post('https://kauth.kakao.com/oauth/token', data={
        'grant_type': 'authorization_code',
        'client_id': rest_key,
        'redirect_uri': REDIRECT,
        'code': code,
    }, timeout=15)
    if r.status_code != 200:
        print('❌ 토큰 발급 실패:', r.text)
        return
    d = r.json()
    print('\n✅ 발급 완료! GitHub 레포 Settings → Secrets → Actions 에 등록:')
    print(f'  KAKAO_REST_KEY      = {rest_key}')
    print(f"  KAKAO_REFRESH_TOKEN = {d['refresh_token']}")

    # 즉시 테스트 발송
    if input('\n지금 테스트 메시지를 보낼까요? (y/n): ').strip().lower() == 'y':
        import json as _json
        template = {'object_type': 'text',
                    'text': '📡 Volume Radar 카카오 연동 테스트 성공!',
                    'link': {'web_url': 'https://github.com',
                             'mobile_web_url': 'https://github.com'}}
        t = requests.post('https://kapi.kakao.com/v2/api/talk/memo/default/send',
                          headers={'Authorization': f"Bearer {d['access_token']}"},
                          data={'template_object': _json.dumps(template,
                                                               ensure_ascii=False)},
                          timeout=15)
        print('✅ 카톡 확인!' if t.status_code == 200 else f'❌ 실패: {t.text}')


if __name__ == '__main__':
    main()
