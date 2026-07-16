#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Volume Radar — 거래대금 상위 + 상승 + 실적 종목 레이더 (Streamlit)"""
import os
import json
import datetime

import pandas as pd
import streamlit as st

from radar_core import list_days, load_list, ACC_FILE

st.set_page_config(page_title='Volume Radar', page_icon='📡',
                   layout='wide', initial_sidebar_state='expanded')

st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] { background-color:#1A2332 !important; color:#E2E8F0 !important; }
[data-testid="stSidebar"] > div:first-child { background-color:#141B28 !important; }
[data-testid="stSidebar"] * { color:#CBD5E0 !important; }
h1,h2,h3,h4 { color:#FFFFFF !important; }
.radar-title { font-weight:900; font-size:2.0rem;
  background:linear-gradient(to right,#62efff,#ffb3fd);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.badge-new { background:#10B981; color:#fff; padding:1px 7px; border-radius:4px; font-size:0.72rem; font-weight:700; }
.badge-streak { background:rgba(98,239,255,0.15); color:#62EFFF; border:1px solid rgba(98,239,255,0.35);
  padding:1px 7px; border-radius:4px; font-size:0.72rem; font-weight:700; }
#MainMenu, footer, header { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


# ── 비밀번호 게이트 ─────────────────────────────────────────
def check_password():
    def entered():
        st.session_state['pw_ok'] = (st.session_state.get('pw') == '9084')
        if 'pw' in st.session_state:
            del st.session_state['pw']
    if st.session_state.get('pw_ok'):
        return True
    st.markdown('<div class="radar-title">📡 Volume Radar</div>', unsafe_allow_html=True)
    st.caption('거래대금 상위 · 상승 · 실적 종목 레이더')
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.text_input('비밀번호', type='password', key='pw', on_change=entered)
        if 'pw_ok' in st.session_state and not st.session_state['pw_ok']:
            st.error('비밀번호가 일치하지 않습니다')
    return False


def main():
    if not check_password():
        return

    st.markdown('<div class="radar-title">📡 Volume Radar</div>', unsafe_allow_html=True)
    st.caption('전 거래일 거래대금 상위 100 중 [상승 + 실적(흑자·성장률 20%↑)] 종목 · 매일 새벽 자동 갱신')

    days = list_days()
    if not days:
        st.info('아직 데이터가 없습니다. 새벽 자동 갱신 후 표시됩니다.')
        return

    acc = {}
    if os.path.exists(ACC_FILE):
        with open(ACC_FILE, encoding='utf-8') as f:
            acc = json.load(f)

    with st.sidebar:
        st.markdown('## ⚙️ 설정')
        sel_day = st.selectbox('조회 거래일', days[::-1], index=0)
        min_count = st.slider('누적 랭킹: 최소 등장 횟수', 1, 10, 2)
        market_sel = st.multiselect('시장', ['KOSPI', 'KOSDAQ'], default=['KOSPI', 'KOSDAQ'])
        st.markdown('---')
        st.caption(f'데이터: {len(days)}거래일 축적\n\n'
                   f'윈도우: 최근 {len(acc.get("window_days", []))}거래일\n\n'
                   f'갱신: {acc.get("updated", "-")}')

    tab_today, tab_acc, tab_matrix = st.tabs(['📅 일별 리스트', '🏆 누적 랭킹 (10거래일)', '📈 등장 매트릭스'])

    # ── 탭 1: 일별 리스트 ──────────────────────────────────
    with tab_today:
        payload = load_list(sel_day) or {}
        entries = [e for e in payload.get('entries', []) if e['market'] in market_sel]
        prev_payload = None
        idx = days.index(sel_day)
        if idx > 0:
            prev_payload = load_list(days[idx - 1])
        prev_codes = {e['code'] for e in (prev_payload or {}).get('entries', [])}

        st.markdown(f'### {sel_day} — {len(entries)}종목')
        if not entries:
            st.info('해당일 조건 통과 종목 없음')
        else:
            rows = []
            for e in entries:
                rows.append({
                    '순위': e['rank'],
                    'NEW': '' if e['code'] in prev_codes else '🆕',
                    '종목명': e['name'], '코드': e['code'], '시장': e['market'],
                    '업종': e.get('업종', ''),
                    '종가': f"{e['close']:,.0f}",
                    '등락률(%)': e['chg_pct'],
                    '거래대금(억)': f"{e['turnover_억']:,.0f}",
                    '시총(억)': f"{e['mcap_억']:,.0f}",
                    '매출MAX성장(%)': e.get('rev_gmax'),
                    '영업이익MAX성장(%)': e.get('op_gmax'),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True,
                         hide_index=True, height=620)
            st.caption('🆕 = 직전 거래일 리스트에 없던 신규 진입 · 순위 = 전체 시장 거래대금 순위')

    # ── 탭 2: 누적 랭킹 ────────────────────────────────────
    with tab_acc:
        stocks = [s for s in acc.get('stocks', [])
                  if s['count'] >= min_count and s['market'] in market_sel]
        st.markdown(f'### 최근 {len(acc.get("window_days", []))}거래일 · 등장 {min_count}회 이상 — {len(stocks)}종목')
        if not stocks:
            st.info('조건에 맞는 종목 없음')
        else:
            rows = []
            for s in stocks:
                rows.append({
                    '등장': f"{s['count']}회",
                    '연속': f"{s['streak']}일" if s['streak'] > 1 else '-',
                    '종목명': s['name'], '코드': s['code'], '시장': s['market'],
                    '업종': s.get('업종', ''),
                    '평균순위': s['avg_rank'], '최고순위': s['best_rank'],
                    '최근종가': f"{s['last_close']:,.0f}",
                    '최근등락(%)': s['last_chg'],
                    '최근거래대금(억)': f"{s['last_turnover_억']:,.0f}",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True,
                         hide_index=True, height=620)
            st.caption('정렬: 등장 횟수 ↓, 평균 순위 ↑ — 반복 등장 + 상위권일수록 지속적 자금 유입')

    # ── 탭 3: 등장 매트릭스 ────────────────────────────────
    with tab_matrix:
        window = acc.get('window_days', [])
        stocks = [s for s in acc.get('stocks', [])
                  if s['count'] >= min_count and s['market'] in market_sel]
        st.markdown(f'### 종목 × 거래일 등장 매트릭스 (등장 {min_count}회 이상)')
        if not stocks or not window:
            st.info('데이터 부족')
        else:
            rows = []
            for s in stocks[:60]:
                row = {'종목명': s['name'], '등장': s['count']}
                for d in window:
                    row[d[5:]] = '●' if d in s['dates'] else ''
                rows.append(row)
            st.dataframe(pd.DataFrame(rows), use_container_width=True,
                         hide_index=True, height=620)
            st.caption('● = 그 거래일 리스트에 등장 — 오른쪽으로 연속되면 지속 유입, 띄엄띄엄이면 이벤트성')


if __name__ == '__main__':
    main()
