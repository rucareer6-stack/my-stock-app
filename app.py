import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date

# ==========================================
# 1. 視覺設定：純白專業風格
# ==========================================
st.set_page_config(page_title="個人投資資產管理", layout="wide", page_icon="📈")

st.markdown("""
    <style>
    /* --- 全局設定 --- */
    .stApp { background-color: #ffffff; }
    
    /* --- 文字設定 --- */
    h1, h2, h3, h4, h5, h6 {
        color: #111827 !important;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700 !important;
    }
    p, div, span, label, li {
        color: #374151 !important;
    }
    
    /* --- 側邊欄 --- */
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
    
    /* --- 按鈕樣式 (白字藍底) --- */
    div.stButton > button {
        background-color: #2563eb !important;
        color: #ffffff !important;
        border: none;
        font-weight: 600;
        border-radius: 6px;
        transition: background-color 0.3s;
    }
    div.stButton > button:hover {
        background-color: #1d4ed8 !important;
        color: #ffffff !important;
    }
    div.stButton > button p {
        color: #ffffff !important;
    }
    
    /* --- 表格樣式 --- */
    thead tr th:first-child {display:none}
    tbody th {display:none}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 核心邏輯與計算工具
# ==========================================

if 'portfolio' not in st.session_state:
    st.session_state['portfolio'] = pd.DataFrame([
        {'Ticker': 'NVDA', 'Cost': 450.0, 'Shares': 10, 'Date': date(2023, 1, 15)},
        {'Ticker': 'AAPL', 'Cost': 170.0, 'Shares': 20, 'Date': date(2023, 6, 1)},
        {'Ticker': 'TSLA', 'Cost': 200.0, 'Shares': 15, 'Date': date(2022, 11, 20)}
    ])
if 'cash' not in st.session_state:
    st.session_state['cash'] = 10000.0

# --- 輔助函數 ---
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_cagr(current_price, cost, buy_date):
    if cost == 0: return 0
    days = (date.today() - buy_date).days
    if days <= 0: return 0
    years = days / 365.25
    if years < 1: return (current_price - cost) / cost
    try: return (current_price / cost) ** (1 / years) - 1
    except: return 0

def classify_risk(beta):
    if pd.isna(beta): return "未知"
    if beta < 0.8: return "🛡️ 低波動 (保守)"
    if beta > 1.3: return "⚡ 高波動 (積極)"
    return "⚖️ 中波動 (穩健)"

@st.cache_data(ttl=3600)
def get_stock_data(ticker):
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

# ==========================================
# 3. 側邊欄
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
        t_in = st.text_input("代號").upper()
        col_s1, col_s2 = st.columns(2)
        with col_s1: c_in = st.number_input("成本", min_value=0.0, step=0.1)
        with col_s2: s_in = st.number_input("股數", min_value=0.0, step=1.0)
        d_in = st.date_input("買入日期", value=date.today())
        
        if st.form_submit_button("確認送出", use_container_width=True):
            if t_in and s_in > 0:
                df = st.session_state['portfolio']
                if t_in in df['Ticker'].values: df = df[df['Ticker'] != t_in]
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

    meta_map = {t: get_stock_data(t) for t in tickers}
    
    df['Sector'] = df['Ticker'].map(lambda x: meta_map[x]['sector'])
    df['Beta'] = df['Ticker'].map(lambda x: meta_map[x]['beta'])
    df['PE'] = df['Ticker'].map(lambda x: meta_map[x]['pe'])
    df['Risk Level'] = df['Beta'].apply(classify_risk)
    
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
# 6. 圖表區 (對齊修復版)
# ==========================================
st.subheader("📈 資產成長走勢")
if not total_history.empty:
    fig_area = px.area(x=total_history.index, y=total_history.values)
    fig_area.update_layout(
        plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(l=0,r=0,t=10,b=0), height=300,
        xaxis=dict(showgrid=False, title=""),
        yaxis=dict(showgrid=True, gridcolor='#f3f4f6', title="資產價值 (USD)")
    )
    fig_area.update_traces(line_color='#2563eb', fillcolor='rgba(37, 99, 235, 0.1)')
    st.plotly_chart(fig_area, use_container_width=True)

st.write("")

# 並排圓餅圖 (強制對齊：Legend 在底部 + 固定高度)
c_sector, c_risk = st.columns(2)

with c_sector:
    st.subheader("🍰 產業分散度")
    if not df.empty:
        clean_df = df.copy()
        clean_df['Sector'] = clean_df['Sector'].fillna('其他')
        fig_pie = px.pie(clean_df, values='Market Value', names='Sector', hole=0.5,
                         color_discrete_sequence=px.colors.qualitative.Set2)
        # 關鍵修復：legend orientation="h" (水平置底)
        fig_pie.update_layout(
            margin=dict(l=20,r=20,t=0,b=0), 
            height=350, 
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("無數據")

with c_risk:
    st.subheader("🛡️ 風險屬性分佈")
    if not df.empty:
        risk_dist = df.groupby('Risk Level')['Market Value'].sum().reset_index()
        if st.session_state['cash'] > 0:
            cash_row = pd.DataFrame([{'Risk Level': '💵 現金 (Cash)', 'Market Value': st.session_state['cash']}])
            risk_dist = pd.concat([risk_dist, cash_row], ignore_index=True)

        fig_risk = px.pie(risk_dist, values='Market Value', names='Risk Level', hole=0.5,
                          color_discrete_sequence=px.colors.qualitative.Pastel)
        # 關鍵修復：同步設定，確保高度一致
        fig_risk.update_layout(
            margin=dict(l=20,r=20,t=0,b=0), 
            height=350, 
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_risk, use_container_width=True)
    else:
        st.info("無數據")

st.divider()

# ==========================================
# 7. 持倉詳細績效表
# ==========================================
st.subheader("📋 持倉詳細績效表")
if not df.empty:
    display_df = df[['Ticker', 'Sector', 'Risk Level', 'Date', 'Cost', 'Current Price', 'Shares', 'Market Value', 'Profit', 'Return %', 'CAGR %', 'Beta']]
    
    st.dataframe(
        display_df,
        column_config={
            "Ticker": "代號", "Sector": "產業", "Risk Level": "風險",
            "Date": st.column_config.DateColumn("買入日"),
            "Cost": st.column_config.NumberColumn("成本", format="$%.2f"),
            "Current Price": st.column_config.NumberColumn("現價", format="$%.2f"),
            "Shares": st.column_config.NumberColumn("股數", format="%.0f"),
            "Market Value": st.column_config.NumberColumn("市值", format="$%.0f"),
            "Profit": st.column_config.NumberColumn("損益", format="$%.0f"),
            "Return %": st.column_config.NumberColumn("報酬%", format="%.2f%%"),
            "CAGR %": st.column_config.NumberColumn("年化%", format="%.2f%%"),
            "Beta": st.column_config.NumberColumn("Beta", format="%.2f"),
        },
        hide_index=True, use_container_width=True
    )

st.divider()

# ==========================================
# 8. [已恢復] 個股深度分析 (RSI/Beta/PE/Chart)
# ==========================================
st.subheader("🔍 個股深度分析 (基本面/技術面)")

if not df.empty:
    sel_ticker = st.selectbox("選擇要分析的股票：", df['Ticker'].unique())
    row = df[df['Ticker'] == sel_ticker].iloc[0]
    
    # 獲取 6 個月數據計算 RSI
    stock = yf.Ticker(sel_ticker)
    hist = stock.history(period="6mo")
    hist['MA20'] = hist['Close'].rolling(20).mean()
    hist['RSI'] = calculate_rsi(hist['Close'])
    curr_rsi = hist['RSI'].iloc[-1]
    
    # 佈局：左圖表，右指標
    c_chart_deep, c_metrics_deep = st.columns([2, 1])
    
    with c_chart_deep:
        st.markdown(f"**{sel_ticker} K線與均線 (Daily)**")
        fig_k = go.Figure()
        fig_k.add_trace(go.Candlestick(x=hist.index,
                        open=hist['Open'], high=hist['High'],
                        low=hist['Low'], close=hist['Close'], name='Price'))
        fig_k.add_trace(go.Scatter(x=hist.index, y=hist['MA20'], line=dict(color='orange', width=1), name='MA 20'))
        fig_k.update_layout(xaxis_rangeslider_visible=False, height=400,
                            plot_bgcolor='white', margin=dict(l=10, r=0, t=10, b=20),
                            legend=dict(orientation="h", y=1, x=0))
        st.plotly_chart(fig_k, use_container_width=True)
        
    with c_metrics_deep:
        st.markdown("**關鍵指標儀表板**")
        
        # 使用 Grid 顯示
        m1, m2 = st.columns(2)
        m1.metric("現價", f"${row['Current Price']:.2f}")
        
        pe_val = row['PE']
        pe_show = f"{pe_val:.1f}" if pd.notnull(pe_val) else "N/A"
        m2.metric("本益比 (P/E)", pe_show)
        
        st.markdown("---")
        
        m3, m4 = st.columns(2)
        # RSI 顏色邏輯
        rsi_col = "inverse" if curr_rsi > 70 else ("normal" if curr_rsi < 30 else "off")
        rsi_state = "過熱" if curr_rsi > 70 else ("超賣" if curr_rsi < 30 else "中性")
        m3.metric("RSI (14)", f"{curr_rsi:.1f}", delta=rsi_state, delta_color=rsi_col)
        
        m4.metric("Beta (波動)", f"{row['Beta']:.2f}", help=">1 代表波動大於大盤")
        
        st.markdown("---")
        st.caption(f"所屬產業: {row['Sector']}")
        st.caption(f"風險屬性: {row['Risk Level']}")

else:
    st.info("暫無持倉數據，無法進行深度分析。")
