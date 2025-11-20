import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date

# ==========================================
# 1. 視覺設定：純白專業風格 (UI 修復版)
# ==========================================
st.set_page_config(page_title="個人投資資產管理", layout="wide", page_icon="📈")

st.markdown("""
    <style>
    /* --- 全局設定 --- */
    .stApp { background-color: #ffffff; }
    
    /* --- 文字設定 (深色字體) --- */
    h1, h2, h3, h4, h5, h6 {
        color: #111827 !important;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700 !important;
    }
    p, div, span, label, li {
        color: #374151 !important;
    }
    
    /* --- 側邊欄優化 --- */
    [data-testid="stSidebar"] {
        background-color: #f9fafb !important;
        border-right: 1px solid #e5e7eb;
    }
    
    /* --- 指標卡片 (Metric) --- */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    [data-testid="stMetricValue"] {
        color: #2563eb !important; /* 專業藍 */
        font-weight: 800 !important;
    }
    
    /* --- 按鈕樣式強制修復 (白字藍底) --- */
    div.stButton > button {
        background-color: #2563eb !important; /* 深藍底 */
        color: #ffffff !important; /* 白字 */
        border: none;
        font-weight: 600;
        border-radius: 6px;
        transition: background-color 0.3s;
    }
    div.stButton > button:hover {
        background-color: #1d4ed8 !important; /* 懸停加深 */
        color: #ffffff !important;
    }
    div.stButton > button p {
        color: #ffffff !important; /* 強制內部文字變白 */
    }
    
    /* --- 表格樣式 --- */
    thead tr th:first-child {display:none}
    tbody th {display:none}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 核心邏輯 (含風險分級算法)
# ==========================================

if 'portfolio' not in st.session_state:
    st.session_state['portfolio'] = pd.DataFrame([
        {'Ticker': 'NVDA', 'Cost': 450.0, 'Shares': 10, 'Date': date(2023, 1, 15)},
        {'Ticker': 'AAPL', 'Cost': 170.0, 'Shares': 20, 'Date': date(2023, 6, 1)},
        {'Ticker': 'TSLA', 'Cost': 200.0, 'Shares': 15, 'Date': date(2022, 11, 20)}
    ])
if 'cash' not in st.session_state:
    st.session_state['cash'] = 10000.0

# --- Beta 風險分級函數 (新增功能) ---
def classify_risk(beta):
    if pd.isna(beta): return "未知 (Unknown)"
    if beta < 0.8: return "🛡️ 低波動 (保守型)"
    if beta > 1.3: return "⚡ 高波動 (積極型)"
    return "⚖️ 中波動 (穩健型)"

# --- 獲取數據函數 ---
@st.cache_data(ttl=3600)
def get_stock_data_stable(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            'sector': info.get('sector', '其他'),
            'pe': info.get('trailingPE', None),
            'beta': info.get('beta', 1.0),
        }
    except:
        return {'sector': '未知', 'pe': None, 'beta': 1.0}

# --- CAGR 計算 ---
def calculate_cagr(current_price, cost, buy_date):
    if cost == 0: return 0
    days = (date.today() - buy_date).days
    if days <= 0: return 0
    years = days / 365.25
    if years < 1: return (current_price - cost) / cost
    try: return (current_price / cost) ** (1 / years) - 1
    except: return 0

# ==========================================
# 3. 側邊欄 (UI 已修復)
# ==========================================
with st.sidebar:
    st.header("⚙️ 投資組合設定")
    
    st.subheader("💵 現金管理")
    new_cash = st.number_input("現金餘額 (USD)", value=st.session_state['cash'], step=100.0)
    if new_cash != st.session_state['cash']:
        st.session_state['cash'] = new_cash
        st.rerun()
        
    st.divider()
    
    st.subheader("➕ 新增/更新持倉")
    with st.form("add_pos"):
        # 這裡刪除了 "(如 AAPL)" 並調整了排版
        t_in = st.text_input("代號").upper()
        
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            c_in = st.number_input("成本", min_value=0.0, step=0.1)
        with col_s2:
            s_in = st.number_input("股數", min_value=0.0, step=1.0)
            
        d_in = st.date_input("買入日期", value=date.today())
        
        # 按鈕 CSS 已修復，字體會是白色
        if st.form_submit_button("確認送出", use_container_width=True):
            if t_in and s_in > 0:
                df = st.session_state['portfolio']
                if t_in in df['Ticker'].values:
                    df = df[df['Ticker'] != t_in]
                new_row = pd.DataFrame([{'Ticker': t_in, 'Cost': c_in, 'Shares': s_in, 'Date': d_in}])
                st.session_state['portfolio'] = pd.concat([df, new_row], ignore_index=True)
                st.success("已更新")
                st.rerun()

    if not st.session_state['portfolio'].empty:
        st.divider()
        st.write("選擇要刪除的股票")
        del_ticker = st.selectbox("刪除股票", st.session_state['portfolio']['Ticker'].unique(), label_visibility="collapsed")
        if st.button("🗑️ 刪除選定股票", use_container_width=True):
            st.session_state['portfolio'] = st.session_state['portfolio'][st.session_state['portfolio']['Ticker'] != del_ticker]
            st.rerun()

# ==========================================
# 4. 主畫面數據處理
# ==========================================
st.title("📊 個人投資資產分析")
st.caption(f"數據更新時間: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

df = st.session_state['portfolio'].copy()
total_history = pd.DataFrame()

if not df.empty:
    tickers = df['Ticker'].tolist()
    
    # 1. 獲取價格與歷史
    try:
        hist_data = yf.download(tickers, period="1y", progress=False)['Close']
        current_prices = {}
        if isinstance(hist_data, pd.DataFrame) and not hist_data.empty:
            for t in tickers:
                current_prices[t] = hist_data[t].iloc[-1] if t in hist_data.columns else 0
            stock_val_hist = (hist_data * df.set_index('Ticker')['Shares']).sum(axis=1)
            total_history = stock_val_hist + st.session_state['cash']
        elif isinstance(hist_data, pd.Series):
            current_prices[tickers[0]] = hist_data.iloc[-1]
            total_history = (hist_data * df.iloc[0]['Shares']) + st.session_state['cash']
    except:
        current_prices = {t: 0 for t in tickers}

    # 2. 獲取 Meta Data
    meta_map = {t: get_stock_data_stable(t) for t in tickers}
    
    df['Sector'] = df['Ticker'].map(lambda x: meta_map[x]['sector'])
    df['Beta'] = df['Ticker'].map(lambda x: meta_map[x]['beta'])
    
    # --- 計算風險分級 ---
    df['Risk Level'] = df['Beta'].apply(classify_risk)
    
    # 3. 計算財務數據
    df['Current Price'] = df['Ticker'].map(current_prices)
    df['Market Value'] = df['Current Price'] * df['Shares']
    df['Profit'] = (df['Current Price'] - df['Cost']) * df['Shares']
    df['Return %'] = df['Profit'] / (df['Cost'] * df['Shares']) * 100
    df['CAGR %'] = df.apply(lambda x: calculate_cagr(x['Current Price'], x['Cost'], x['Date']), axis=1) * 100

    total_stock_val = df['Market Value'].sum()
    total_profit = df['Profit'].sum()
else:
    total_stock_val = 0
    total_profit = 0

total_assets = total_stock_val + st.session_state['cash']
cash_ratio = (st.session_state['cash'] / total_assets * 100) if total_assets > 0 else 0

# ==========================================
# 5. 儀表板 Metrics
# ==========================================
col1, col2, col3, col4 = st.columns(4)
col1.metric("總資產 (Total Assets)", f"${total_assets:,.0f}")
col2.metric("總損益 (Total P/L)", f"${total_profit:,.0f}", delta_color="normal")
col3.metric("股票市值", f"${total_stock_val:,.0f}")
col4.metric("現金水位", f"{cash_ratio:.1f}%")

st.divider()

# ==========================================
# 6. 圖表區 (重新佈局以確保對齊)
# ==========================================

# 第一排：資產成長走勢 (全寬，確保不被壓縮)
st.subheader("📈 資產成長走勢")
if not total_history.empty:
    fig_area = px.area(x=total_history.index, y=total_history.values)
    fig_area.update_layout(
        plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(l=0,r=0,t=10,b=0), height=300, # 固定高度
        xaxis=dict(showgrid=False, title=""),
        yaxis=dict(showgrid=True, gridcolor='#f3f4f6', title="資產價值 (USD)")
    )
    fig_area.update_traces(line_color='#2563eb', fillcolor='rgba(37, 99, 235, 0.1)')
    st.plotly_chart(fig_area, use_container_width=True)
else:
    st.info("暫無歷史數據")

st.write("") # 間隔

# 第二排：產業分散度 + 風險分佈 (並排顯示，高度對齊)
c_sector, c_risk = st.columns(2)

with c_sector:
    st.subheader("🍰 產業分散度 (Sector)")
    if not df.empty:
        # 處理空值
        clean_df = df.copy()
        clean_df['Sector'] = clean_df['Sector'].fillna('其他')
        
        fig_pie = px.pie(clean_df, values='Market Value', names='Sector', hole=0.5,
                         color_discrete_sequence=px.colors.qualitative.Set2)
        fig_pie.update_layout(margin=dict(l=0,r=0,t=20,b=0), height=350, showlegend=True)
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.caption("無持倉數據")

with c_risk:
    st.subheader("🛡️ 風險屬性分佈 (Risk)")
    if not df.empty:
        # 根據風險等級匯總資產
        risk_dist = df.groupby('Risk Level')['Market Value'].sum().reset_index()
        # 加入現金 (視為零風險/現金)
        if st.session_state['cash'] > 0:
            cash_row = pd.DataFrame([{'Risk Level': '💵 現金 (Cash)', 'Market Value': st.session_state['cash']}])
            risk_dist = pd.concat([risk_dist, cash_row], ignore_index=True)

        fig_risk = px.pie(risk_dist, values='Market Value', names='Risk Level', hole=0.5,
                          color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_risk.update_layout(margin=dict(l=0,r=0,t=20,b=0), height=350, showlegend=True)
        st.plotly_chart(fig_risk, use_container_width=True)
    else:
        st.caption("無持倉數據")

st.divider()

# ==========================================
# 7. 底部：持倉詳情表
# ==========================================
st.subheader("📋 持倉詳細績效表")
if not df.empty:
    display_df = df[['Ticker', 'Sector', 'Risk Level', 'Date', 'Cost', 'Current Price', 'Shares', 'Market Value', 'Profit', 'Return %', 'CAGR %', 'Beta']]
    
    st.dataframe(
        display_df,
        column_config={
            "Ticker": "代號",
            "Sector": "產業",
            "Risk Level": "風險屬性", # 新增欄位
            "Date": st.column_config.DateColumn("買入日期"),
            "Cost": st.column_config.NumberColumn("成本", format="$%.2f"),
            "Current Price": st.column_config.NumberColumn("現價", format="$%.2f"),
            "Shares": st.column_config.NumberColumn("股數", format="%.0f"),
            "Market Value": st.column_config.NumberColumn("市值", format="$%.0f"),
            "Profit": st.column_config.NumberColumn("損益", format="$%.0f"),
            "Return %": st.column_config.NumberColumn("報酬率", format="%.2f%%"),
            "CAGR %": st.column_config.NumberColumn("年化(CAGR)", format="%.2f%%"),
            "Beta": st.column_config.NumberColumn("Beta", format="%.2f"),
        },
        hide_index=True,
        use_container_width=True
    )
else:
    st.info("暫無持倉，請從左側新增。")
