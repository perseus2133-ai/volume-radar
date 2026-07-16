#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Volume Radar — 거래대금 상위 + 상승 + 실적 종목 레이더 (Streamlit)"""
import os
import json

import pandas as pd
import streamlit as st

from radar_core import list_days, load_list, ACC_FILE

st.set_page_config(page_title='Volume Radar', page_icon='📡',
                   layout='wide', initial_sidebar_state='expanded')

# ============================================================
# 커스텀 CSS — ai2 퀀트 터미널 스타일
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&family=JetBrains+Mono:wght@400;700;800&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', 'Pretendard', sans-serif;
    background-color: #1A2332 !important;
    color: #E2E8F0 !important;
}
[data-testid="stSidebar"] > div:first-child { background-color: #141B28 !important; }
[data-testid="stSidebar"] * { color: #CBD5E0 !important; }
h1,h2,h3,h4 { color: #FFFFFF !important; }

/* 헤더 */
.hero { border-bottom: 1px solid #4A5568; padding-bottom: 14px; margin-bottom: 18px; }
.radar-title {
    font-weight: 900; font-size: 2.1rem; display: flex; align-items: center; gap: 10px;
    background: linear-gradient(to right, #62efff, #ffb3fd, #ffeead, #62efff);
    background-size: 400% 400%;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    animation: rainbow 6s ease infinite;
}
@keyframes rainbow { 0%{background-position:0% 50%;} 50%{background-position:100% 50%;} 100%{background-position:0% 50%;} }
.hero p { color: #94A3B8; font-size: 0.92rem; margin: 4px 0 0 0; font-family: 'JetBrains Mono', monospace; }

/* 메트릭 카드 */
.metric-card {
    text-align: center; background: linear-gradient(135deg, #3F4C60, #313B4D);
    border-radius: 12px; padding: 14px 10px; border: 1px solid #4A5568;
    box-shadow: 0 6px 14px rgba(0,0,0,0.25);
}
.metric-label { color: #94A3B8; font-size: 0.78rem; margin-bottom: 3px; }
.metric-value { font-family: 'JetBrains Mono', monospace; font-weight: 800; font-size: 1.35rem; color: #FFFFFF; }
.metric-value.cyan { color: #62EFFF; }

/* 탭 */
div[data-baseweb="tab-list"] { gap: 10px; }
button[data-baseweb="tab"] {
    background: linear-gradient(135deg, #2D3748, #313B4D) !important;
    border: 1px solid #4A5568 !important; border-radius: 10px !important;
    padding: 10px 20px !important;
}
button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] > p {
    color: #94A3B8 !important; font-weight: 700 !important; font-size: 1.02rem !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #1D3557, #457B9D) !important;
    border-color: #62efff !important;
    box-shadow: 0 6px 12px rgba(98,239,255,0.18) !important;
}
button[data-baseweb="tab"][aria-selected="true"] > div[data-testid="stMarkdownContainer"] > p {
    color: #FFFFFF !important; font-weight: 900 !important;
}

/* 시장 선택 라디오 */
div[role="radiogroup"] label { color: #CBD5E0 !important; }

/* 데이터 테이블 (커스텀 HTML) */
.tbl-wrap { overflow-x: auto; border: 1px solid #4A5568; border-radius: 12px;
    background: linear-gradient(135deg, #333E52 0%, #2B3547 100%);
    box-shadow: 0 8px 18px rgba(0,0,0,0.28); }
table.radar { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
table.radar th {
    position: sticky; top: 0; background: #232E40; color: #94A3B8;
    font-size: 0.74rem; font-weight: 700; text-align: right;
    padding: 10px 12px; border-bottom: 1px solid #4A5568; white-space: nowrap;
}
table.radar th.l, table.radar td.l { text-align: left; }
table.radar th.c, table.radar td.c { text-align: center; }
table.radar td {
    padding: 9px 12px; text-align: right; white-space: nowrap;
    border-bottom: 1px solid rgba(74,85,104,0.35);
    font-family: 'JetBrains Mono', monospace; font-size: 0.86rem;
}
table.radar tr:hover td { background: rgba(98,239,255,0.05); }
table.radar tr:last-child td { border-bottom: none; }

a.stk { color: #FFFFFF !important; font-weight: 700; text-decoration: none !important;
    font-family: 'Inter', sans-serif; border-bottom: 1px dashed rgba(98,239,255,0.35); }
a.stk:hover { color: #62EFFF !important; border-bottom-color: #62EFFF; }

.b-kospi { background: linear-gradient(135deg,#FBBF24,#F59E0B); color: #1A1C24;
    font-weight: 800; font-size: 0.68rem; padding: 2px 8px; border-radius: 4px; font-family:'Inter'; }
.b-kosdaq { background: linear-gradient(135deg,#FB923C,#F97316); color: #1A1C24;
    font-weight: 800; font-size: 0.68rem; padding: 2px 8px; border-radius: 4px; font-family:'Inter'; }
.b-new { background: #10B981; color: #fff; padding: 2px 7px; border-radius: 4px;
    font-size: 0.7rem; font-weight: 800; font-family:'Inter'; }
.b-streak { background: rgba(98,239,255,0.12); color: #62EFFF;
    border: 1px solid rgba(98,239,255,0.35); padding: 1px 7px; border-radius: 4px;
    font-size: 0.72rem; font-weight: 700; }
.b-count { background: rgba(167,139,250,0.15); color: #C4B5FD;
    border: 1px solid rgba(167,139,250,0.4); padding: 1px 8px; border-radius: 4px;
    font-size: 0.78rem; font-weight: 800; }
.up   { color: #FF6B6B; font-weight: 700; }
.dn   { color: #4A90E2; font-weight: 700; }
.dim  { color: #64748B; }
.est  { color: #62EFFF; }
.dot  { color: #62EFFF; font-size: 0.95rem; }
.dot-off { color: #33415580; }

.sect-note { color: #94A3B8; font-size: 0.8rem; margin: 8px 2px 0 2px; }
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

NAVER = 'https://finance.naver.com/item/main.naver?code={code}'


# ============================================================
# 헬퍼
# ============================================================
def stk(code, name):
    return f'<a class="stk" href="{NAVER.format(code=code)}" target="_blank">{name}</a>'


def badge_market(m):
    return f'<span class="b-kospi">KOSPI</span>' if m == 'KOSPI' else f'<span class="b-kosdaq">KOSDAQ</span>'


def fmt_chg(v):
    if v is None:
        return '<span class="dim">-</span>'
    cls = 'up' if v > 0 else ('dn' if v < 0 else 'dim')
    arrow = '▲' if v > 0 else ('▼' if v < 0 else '')
    return f'<span class="{cls}">{arrow} {v:+.2f}%</span>'


def fmt_growth(v):
    if v is None:
        return '<span class="dim">-</span>'
    if v >= 100:
        return f'<span class="up">🔥 {v:,.0f}%</span>'
    return f'<span class="est">{v:,.0f}%</span>'


def fmt_turnover(v):
    if v is None:
        return '-'
    if v >= 10000:
        return f'{v/10000:,.1f}조'
    return f'{v:,.0f}억'


def html_table(header_cells, body_rows):
    thead = ''.join(header_cells)
    tbody = ''.join(body_rows)
    return f'<div class="tbl-wrap"><table class="radar"><thead><tr>{thead}</tr></thead><tbody>{tbody}</tbody></table></div>'


# ============================================================
# 비밀번호 게이트
# ============================================================
def check_password():
    def entered():
        st.session_state['pw_ok'] = (st.session_state.get('pw') == '9084')
        if 'pw' in st.session_state:
            del st.session_state['pw']
    if st.session_state.get('pw_ok'):
        return True
    st.markdown('<div class="hero"><div class="radar-title">📡 VOLUME RADAR</div>'
                '<p>거래대금 레이더 · 인증 필요</p></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.text_input('비밀번호', type='password', key='pw', on_change=entered)
        if 'pw_ok' in st.session_state and not st.session_state['pw_ok']:
            st.error('비밀번호가 일치하지 않습니다')
    return False


# ============================================================
# 메인
# ============================================================
def main():
    if not check_password():
        return

    st.markdown('<div class="hero"><div class="radar-title">📡 VOLUME RADAR</div>'
                '<p>전 거래일 거래대금 상위 100 중 [상승 + 실적] 종목 · 매일 새벽 자동 갱신</p></div>',
                unsafe_allow_html=True)

    days = list_days()
    if not days:
        st.info('아직 데이터가 없습니다. 새벽 자동 갱신 후 표시됩니다.')
        return

    acc = {}
    if os.path.exists(ACC_FILE):
        with open(ACC_FILE, encoding='utf-8') as f:
            acc = json.load(f)

    # ── 사이드바 ──
    with st.sidebar:
        st.markdown('## ⚙️ 설정')
        sel_day = st.selectbox('조회 거래일', days[::-1], index=0)
        min_count = st.slider('누적: 최소 등장 횟수', 1, 10, 2)
        st.markdown('---')
        st.caption(f'축적: {len(days)}거래일 · 윈도우 {len(acc.get("window_days", []))}거래일\n\n'
                   f'갱신: {acc.get("updated", "-")}')

    # ── 시장 선택 (전체/KOSPI/KOSDAQ 분리) ──
    market_mode = st.radio('시장', ['전체', 'KOSPI', 'KOSDAQ'],
                           horizontal=True, label_visibility='collapsed')
    def market_ok(m):
        return market_mode == '전체' or m == market_mode

    # ── 메트릭 카드 ──
    latest = load_list(days[-1]) or {}
    lat_entries = latest.get('entries', [])
    kp = sum(1 for e in lat_entries if e['market'] == 'KOSPI')
    kd = sum(1 for e in lat_entries if e['market'] == 'KOSDAQ')
    acc_n = sum(1 for s in acc.get('stocks', []) if s['count'] >= 2)
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(f'<div class="metric-card"><div class="metric-label">최신 리스트 ({days[-1][5:]})</div>'
                f'<div class="metric-value cyan">{len(lat_entries)}</div>'
                f'<div class="metric-label">종목</div></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="metric-card"><div class="metric-label">KOSPI / KOSDAQ</div>'
                f'<div class="metric-value">{kp} / {kd}</div>'
                f'<div class="metric-label">종목</div></div>', unsafe_allow_html=True)
    m3.markdown(f'<div class="metric-card"><div class="metric-label">반복 등장 (2회+)</div>'
                f'<div class="metric-value cyan">{acc_n}</div>'
                f'<div class="metric-label">/ 최근 {len(acc.get("window_days", []))}거래일</div></div>',
                unsafe_allow_html=True)
    m4.markdown(f'<div class="metric-card"><div class="metric-label">데이터 축적</div>'
                f'<div class="metric-value">{len(days)}</div>'
                f'<div class="metric-label">거래일</div></div>', unsafe_allow_html=True)
    st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)

    tab_today, tab_acc, tab_matrix = st.tabs(['📅 일별 리스트', '🏆 누적 랭킹', '📈 등장 매트릭스'])

    # ════════════════════════════════════════════════════════
    # 탭 1 — 일별 리스트
    # ════════════════════════════════════════════════════════
    with tab_today:
        payload = load_list(sel_day) or {}
        entries = [e for e in payload.get('entries', []) if market_ok(e['market'])]
        idx = days.index(sel_day)
        prev_codes = set()
        if idx > 0:
            prev_codes = {e['code'] for e in (load_list(days[idx - 1]) or {}).get('entries', [])}

        st.markdown(f'<div style="color:#62EFFF;font-family:\'JetBrains Mono\',monospace;'
                    f'font-size:0.9rem;margin-bottom:10px;">&gt; {sel_day} · {len(entries)}종목 '
                    f'({market_mode})</div>', unsafe_allow_html=True)
        if not entries:
            st.info('해당일 조건 통과 종목 없음')
        else:
            head = ['<th class="c">순위</th>', '<th class="c"></th>', '<th class="l">종목명</th>',
                    '<th class="c">시장</th>', '<th class="l">업종</th>', '<th>종가</th>',
                    '<th>등락률</th>', '<th>거래대금</th>', '<th>시총</th>',
                    '<th>매출MAX</th>', '<th>영업이익MAX</th>']
            rows = []
            for e in entries:
                new = '' if e['code'] in prev_codes else '<span class="b-new">NEW</span>'
                rows.append(
                    f'<tr><td class="c dim">{e["rank"]}</td>'
                    f'<td class="c">{new}</td>'
                    f'<td class="l">{stk(e["code"], e["name"])}</td>'
                    f'<td class="c">{badge_market(e["market"])}</td>'
                    f'<td class="l dim" style="font-family:Inter;font-size:0.78rem;">{e.get("업종") or "-"}</td>'
                    f'<td>{e["close"]:,.0f}</td>'
                    f'<td>{fmt_chg(e["chg_pct"])}</td>'
                    f'<td>{fmt_turnover(e["turnover_억"])}</td>'
                    f'<td class="dim">{fmt_turnover(e["mcap_억"])}</td>'
                    f'<td>{fmt_growth(e.get("rev_gmax"))}</td>'
                    f'<td>{fmt_growth(e.get("op_gmax"))}</td></tr>'
                )
            st.markdown(html_table(head, rows), unsafe_allow_html=True)
            st.markdown('<div class="sect-note">순위 = 전체 시장 거래대금 순위 · '
                        'NEW = 직전 거래일 리스트에 없던 신규 진입 · 종목명 클릭 → 네이버 증권</div>',
                        unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # 탭 2 — 누적 랭킹
    # ════════════════════════════════════════════════════════
    with tab_acc:
        stocks = [s for s in acc.get('stocks', [])
                  if s['count'] >= min_count and market_ok(s['market'])]
        st.markdown(f'<div style="color:#62EFFF;font-family:\'JetBrains Mono\',monospace;'
                    f'font-size:0.9rem;margin-bottom:10px;">&gt; 최근 {len(acc.get("window_days", []))}거래일 · '
                    f'등장 {min_count}회+ · {len(stocks)}종목 ({market_mode})</div>',
                    unsafe_allow_html=True)
        if not stocks:
            st.info('조건에 맞는 종목 없음')
        else:
            head = ['<th class="c">등장</th>', '<th class="c">연속</th>', '<th class="l">종목명</th>',
                    '<th class="c">시장</th>', '<th class="l">업종</th>',
                    '<th>평균순위</th>', '<th>최고순위</th>', '<th>최근종가</th>',
                    '<th>최근등락</th>', '<th>최근거래대금</th>']
            rows = []
            for s in stocks:
                streak = (f'<span class="b-streak">{s["streak"]}일 연속</span>'
                          if s['streak'] > 1 else '<span class="dim">-</span>')
                rows.append(
                    f'<tr><td class="c"><span class="b-count">{s["count"]}회</span></td>'
                    f'<td class="c">{streak}</td>'
                    f'<td class="l">{stk(s["code"], s["name"])}</td>'
                    f'<td class="c">{badge_market(s["market"])}</td>'
                    f'<td class="l dim" style="font-family:Inter;font-size:0.78rem;">{s.get("업종") or "-"}</td>'
                    f'<td>{s["avg_rank"]:.0f}위</td>'
                    f'<td class="est">{s["best_rank"]}위</td>'
                    f'<td>{s["last_close"]:,.0f}</td>'
                    f'<td>{fmt_chg(s["last_chg"])}</td>'
                    f'<td>{fmt_turnover(s["last_turnover_억"])}</td></tr>'
                )
            st.markdown(html_table(head, rows), unsafe_allow_html=True)
            st.markdown('<div class="sect-note">정렬: 등장 횟수 ↓ → 평균 순위 ↑ — '
                        '반복 등장 + 상위권일수록 지속적 자금 유입 신호</div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # 탭 3 — 등장 매트릭스
    # ════════════════════════════════════════════════════════
    with tab_matrix:
        window = acc.get('window_days', [])
        stocks = [s for s in acc.get('stocks', [])
                  if s['count'] >= min_count and market_ok(s['market'])]
        st.markdown(f'<div style="color:#62EFFF;font-family:\'JetBrains Mono\',monospace;'
                    f'font-size:0.9rem;margin-bottom:10px;">&gt; 종목 × 거래일 등장 매트릭스 '
                    f'({market_mode})</div>', unsafe_allow_html=True)
        if not stocks or not window:
            st.info('데이터 부족')
        else:
            head = ['<th class="l">종목명</th>', '<th class="c">시장</th>', '<th class="c">등장</th>'] + \
                   [f'<th class="c">{d[5:]}</th>' for d in window]
            rows = []
            for s in stocks[:60]:
                dots = ''.join(
                    f'<td class="c">{"<span class=\'dot\'>●</span>" if d in s["dates"] else "<span class=\'dot-off\'>·</span>"}</td>'
                    for d in window)
                rows.append(
                    f'<tr><td class="l">{stk(s["code"], s["name"])}</td>'
                    f'<td class="c">{badge_market(s["market"])}</td>'
                    f'<td class="c"><span class="b-count">{s["count"]}</span></td>'
                    f'{dots}</tr>'
                )
            st.markdown(html_table(head, rows), unsafe_allow_html=True)
            st.markdown('<div class="sect-note">● = 그 거래일 리스트 등장 — '
                        '오른쪽으로 연속되면 지속 유입, 띄엄띄엄이면 이벤트성</div>', unsafe_allow_html=True)


if __name__ == '__main__':
    main()
