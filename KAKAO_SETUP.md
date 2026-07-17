# 📨 카카오톡 아침 브리핑 설정 (최초 1회, ~10분)

매일 새벽 자동 크롤 직후, AI Morning Brief가 **내 카카오톡("나에게 보내기")**으로 도착합니다.

```
☀️AI브리핑 07/16 국면:변동성
매수0 관심0 보류0 매도4 회피6
강세: 가정용품·무선통신
약세: 반도체와반·전기장비
          [대시보드 보기]
```

## 1단계 — 카카오 개발자 앱 만들기 (~3분)

1. https://developers.kakao.com 접속 → 카카오 계정 로그인
2. **내 애플리케이션 → 애플리케이션 추가** (이름: `volume-radar` 아무거나)
3. **앱 키** 메뉴에서 **REST API 키** 복사해두기
4. **플랫폼 → Web 플랫폼 등록** → 사이트 도메인: `http://localhost:3000`
5. **제품 설정 → 카카오 로그인** → 활성화 ON → Redirect URI: `http://localhost:3000`
6. **카카오 로그인 → 동의항목** → **카카오톡 메시지 전송(talk_message)** → "선택 동의" 설정

## 2단계 — 최초 인증 (이 컴퓨터에서, ~2분)

```
cd C:\Users\김종민\Documents\volume-radar
python kakao_auth.py
```

- REST API 키 붙여넣기 → 브라우저 열림 → 카카오 로그인/동의
- 이동된 주소창의 `code=XXXX` 값 복사해 붙여넣기
- **테스트 메시지 y** → 카톡 도착 확인
- 화면에 출력된 두 값을 복사

## 3단계 — GitHub Secrets 등록 (~2분)

https://github.com/perseus2133-ai/volume-radar/settings/secrets/actions → **New repository secret** 2개:

| Name | Value |
|---|---|
| `KAKAO_REST_KEY` | (2단계에서 출력된 값) |
| `KAKAO_REFRESH_TOKEN` | (2단계에서 출력된 값) |
| `KAKAO_CLIENT_SECRET` | (앱의 '클라이언트 시크릿 활성화'가 ON인 경우만 — 보안 탭의 코드) |

**끝.** 다음 새벽 05:30 자동 크롤부터 카톡이 옵니다.

## 운영 참고

- **주말/공휴일 자동 스킵**: 브리핑 기준일이 '어제 거래일'이 아니면 안 보냄 (중복 방지)
- **토큰 수명**: refresh token은 2개월 유효. 만료 1개월 전부터 자동 갱신되며,
  갱신 시 Actions 로그에 `⚠️ 새 REFRESH TOKEN` 이 출력됨 → 그 값으로
  Secret만 교체 (월 1회 미만, 안 하면 만료 시 카톡만 안 오고 나머지는 정상)
- **수동 테스트**: `python -m src.features.kakao_brief --dry-run` (메시지 미리보기)
- Secrets 미설정 상태에서는 해당 단계가 조용히 스킵됨 (다른 기능 무영향)
