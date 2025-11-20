import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date

# ==========================================
# 1. 視覺設定：純白專業風格 (Light Mode)
# ==========================================
st.set_page_config(page_title="個人投資資產管理", layout="wide", page_icon="📈")

st.markdown("""
    <style>
    /* --- 全局設定 --- */
    .stApp { background-color: #ffffff; }
    
    /* --- 文字設定 --- */
    h1, h2, h3, h4, h5, h6 {
        color: #111827 !important; /* 深黑色標題 */
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700 !important;
    }
    p, div, span, label, li {
        color: #4b5563 !important; /* 深灰色內文 */
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
    [data-testid="stMetricLabel"] {
        color: #6b7280 !important;
    }
    
    /* --- 按鈕 --- */
    .stButton > button {
        background-color: #2563eb !important;
        color: white !important;
        border: none;
        font-weight: 600;
        border-radius: 6px;
        transition: background-color 0.3s;
    }
    .stButton > button:hover {
        background-color: #1d4ed8 !important;
    }
    
    /* --- 表格頭部顏色 --- */
    thead tr th:first-child {display:none}
    tbody th {display:none}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 核心邏輯與計算工具 (本地計算，保證穩定)
# ==========================================

# 初始化 Session State
if 'portfolio' not in st.session_state:
    st.session_state['portfolio'] = pd.DataFrame([
        {'Ticker': 'NVDA', 'Cost': 450.0, 'Shares': 10, 'Date': date(2023, 1, 15)},
        {'Ticker': 'AAPL', 'Cost': 170.0, 'Shares': 20, 'Date': date(2023, 6, 1)},
        {'Ticker': 'TSLA', 'Cost': 200.0, 'Shares': 15, 'Date': date(2022, 11, 20)}
    ])
if 'cash' not in st.session_state:
    st.session_state['cash'] = 10000.0

# --- 計算 RSI (相對強弱指標) ---
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- 計算 CAGR (年化報酬率) ---
def calculate_cagr(current_price, cost, buy_date):
    if cost == 0: return 0
    days = (date.today() - buy_date).days
    if days <= 0: return 0
    years = days / 365.25
    
    # 未滿一年顯示簡單報酬，滿一年顯示複利年化
    if years < 1:
        return (current_price - cost) / cost
    else:
        try:
            return (current_price / cost) ** (1 / years) - 1
        except:
            return 0

# --- 獲取股票詳細數據 (基本面 + 技術面) ---
@st.cache_data(ttl=3600)
def get_stock_data_extended(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            'sector': info.get('sector', '其他'),
            'pe': info.get('trailingPE', None),      # 本益比
            'eps': info.get('trailingEps', None),    # 每股盈餘
            'beta': info.get('beta', 1.0),           # 波動率
            'mkt_cap': info.get('marketCap', 0),     # 市值
            'yield': info.get('dividendYield', 0),   # 殖利率
            'high52': info.get('fiftyTwoWeekHigh', 0),
            'low52': info.get('fiftyTwoWeekLow', 0),
        }
    except:
        return {
            'sector': '未知', 'pe': None, 'eps': None, 'beta': 1.0, 
            'mkt_cap': 0, 'yield': 0, 'high52': 0, 'low52': 0
        }

# ==========================================
# 3. 側邊欄：輸入與設定
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
        col_a, col_b = st.columns(2)
        with col_a:
            t_in = st.text_input("代號 (如 AAPL)").upper()
            c_in = st.number_input("成本價", min_value=0.0, step=0.1)
        with col_b:
            s_in = st.number_input("股數", min_value=0.0, step=1.0)
            d_in = st.date_input("買入日期", value=date.today())
            
        if st.form_submit_button("確認送出", use_container_width=True):
            if t_in and s_in > 0:
                df = st.session_state['portfolio']
                # 若存在則覆蓋，不存在則新增
                if t_in in df['Ticker'].values:
                    df = df[df['Ticker'] != t_in]
                
                new_row = pd.DataFrame([{'Ticker': t_in, 'Cost': c_in, 'Shares': s_in, 'Date': d_in}])
                st.session_state['portfolio'] = pd.concat([df, new_row], ignore_index=True)
                st.success(f"已更新 {t_in}")
                st.rerun()

    if not st.session_state['portfolio'].empty:
        st.divider()
        del_ticker = st.selectbox("選擇要刪除的股票", st.session_state['portfolio']['Ticker'].unique())
        if st.button("🗑️ 刪除選定股票", use_container_width=True):
            st.session_state['portfolio'] = st.session_state['portfolio'][st.session_state['portfolio']['Ticker'] != del_ticker]
            st.rerun()

# ==========================================
# 4. 主畫面：數據處理
# ==========================================
st.title("📊 個人投資資產分析")
st.caption(f"數據更新時間: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

df = st.session_state['portfolio'].copy()
total_history = pd.DataFrame() # 用於畫資產圖

if not df.empty:
    tickers = df['Ticker'].tolist()
    
    # --- 1. 批量獲取價格與歷史 (最穩定的方法) ---
    try:
        # 下載 1 年數據
        hist_data = yf.download(tickers, period="1y", progress=False)['Close']
        
        current_prices = {}
        # 處理單支股票與多支股票的格式差異
        if isinstance(hist_data, pd.DataFrame) and not hist_data.empty:
            for t in tickers:
                current_prices[t] = hist_data[t].iloc[-1] if t in hist_data.columns else 0
            
            # 計算資產歷史走勢 (模擬回測)
            stock_val_hist = (hist_data * df.set_index('Ticker')['Shares']).sum(axis=1)
            total_history = stock_val_hist + st.session_state['cash']
            
        elif isinstance(hist_data, pd.Series):
            current_prices[tickers[0]] = hist_data.iloc[-1]
            total_history = (hist_data * df.iloc[0]['Shares']) + st.session_state['cash']
    except:
        current_prices = {t: 0 for t in tickers}
        st.error("⚠️ 數據連線緩慢，顯示部分資訊")

    # --- 2. 獲取進階財務數據 (Meta) ---
    meta_map = {t: get_stock_data_extended(t) for t in tickers}
    
    # 將數據併入 DataFrame
    df['Sector'] = df['Ticker'].map(lambda x: meta_map[x]['sector'])
    df['PE'] = df['Ticker'].map(lambda x: meta_map[x]['pe'])
    df['Beta'] = df['Ticker'].map(lambda x: meta_map[x]['beta'])
    
    # --- 3. 計算績效 ---
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
# 5. 頂部儀表板 (Assets & Allocation)
# ==========================================
col1, col2, col3, col4 = st.columns(4)
col1.metric("總資產 (Total Assets)", f"${total_assets:,.0f}")
col2.metric("總損益 (Total P/L)", f"${total_profit:,.0f}", delta_color="normal")
col3.metric("股票市值", f"${total_stock_val:,.0f}")
col4.metric("現金水位", f"{cash_ratio:.1f}%")

# 圖表區：左邊資產走勢，右邊產業分佈
c_chart, c_pie = st.columns([2, 1])

with c_chart:
    if not total_history.empty:
        st.subheader("📈 資產成長走勢")
        # 使用 Plotly Area Chart
        fig_area = px.area(x=total_history.index, y=total_history.values)
        fig_area.update_layout(
            plot_bgcolor='white', paper_bgcolor='white',
            margin=dict(l=0,r=0,t=10,b=0), height=280,
            xaxis=dict(showgrid=False, title=""),
            yaxis=dict(showgrid=True, gridcolor='#f3f4f6', title="資產價值 (USD)")
        )
        fig_area.update_traces(line_color='#2563eb', fillcolor='rgba(37, 99, 235, 0.1)')
        st.plotly_chart(fig_area, use_container_width=True)

with c_pie:
    if not df.empty:
        st.subheader("🍰 產業分散度")
        # 處理未知產業
        clean_df = df.copy()
        clean_df['Sector'] = clean_df['Sector'].fillna('其他')
        
        fig_pie = px.pie(clean_df, values='Market Value', names='Sector', hole=0.5, 
                         color_discrete_sequence=px.colors.qualitative.Prism)
        fig_pie.update_layout(margin=dict(l=0,r=0,t=10,b=0), height=280, showlegend=True)
        st.plotly_chart(fig_pie, use_container_width=True)

st.divider()

# ==========================================
# 6. 中間：持倉績效表 (標準財務報表)
# ==========================================
st.subheader("📋 持倉詳細績效表")
if not df.empty:
    # 整理要顯示的欄位
    display_df = df[['Ticker', 'Sector', 'Date', 'Cost', 'Current Price', 'Shares', 'Market Value', 'Profit', 'Return %', 'CAGR %', 'PE', 'Beta']]
    
    st.dataframe(
        display_df,
        column_config={
            "Ticker": "代號",
            "Sector": "產業",
            "Date": st.column_config.DateColumn("買入日期"),
            "Cost": st.column_config.NumberColumn("成本", format="$%.2f"),
            "Current Price": st.column_config.NumberColumn("現價", format="$%.2f"),
            "Shares": st.column_config.NumberColumn("股數", format="%.0f"),
            "Market Value": st.column_config.NumberColumn("市值", format="$%.0f"),
            "Profit": st.column_config.NumberColumn("損益", format="$%.0f"),
            "Return %": st.column_config.NumberColumn("報酬率", format="%.2f%%"),
            "CAGR %": st.column_config.NumberColumn("年化(CAGR)", format="%.2f%%"),
            "PE": st.column_config.NumberColumn("P/E", format="%.1f"),
            "Beta": st.column_config.NumberColumn("波動率(Beta)", format="%.2f"),
        },
        hide_index=True,
        use_container_width=True
    )
else:
    st.info("暫無持倉，請從左側新增。")

st.divider()

# ==========================================
# 7. 底部：個股深度分析儀表板 (替代 AI 區塊)
# ==========================================
st.subheader("🔍 個股深度財務分析")

if not df.empty:
    # 1. 選擇股票
    sel_ticker = st.selectbox("選擇要深入分析的股票：", df['Ticker'].unique())
    
    # 取得該股資料
    row = df[df['Ticker'] == sel_ticker].iloc[0]
    meta = meta_map[sel_ticker]
    
    # 下載技術面歷史資料 (半年)
    stock = yf.Ticker(sel_ticker)
    hist = stock.history(period="6mo")
    
    # 計算技術指標
    hist['MA20'] = hist['Close'].rolling(20).mean()
    hist['MA60'] = hist['Close'].rolling(60).mean()
    hist['RSI'] = calculate_rsi(hist['Close'])
    curr_rsi = hist['RSI'].iloc[-1]
    
    # --- 版面規劃：左邊 (K線+均線)，右邊 (財務指標網格) ---
    col_chart_deep, col_metrics_deep = st.columns([2, 1])
    
    with col_chart_deep:
        st.markdown(f"**{sel_ticker} 股價走勢與均線 (Daily)**")
        
        fig_k = go.Figure()
        # K 線
        fig_k.add_trace(go.Candlestick(x=hist.index,
                        open=hist['Open'], high=hist['High'],
                        low=hist['Low'], close=hist['Close'], name='Price'))
        # 均線
        fig_k.add_trace(go.Scatter(x=hist.index, y=hist['MA20'], line=dict(color='orange', width=1), name='MA 20'))
        fig_k.add_trace(go.Scatter(x=hist.index, y=hist['MA60'], line=dict(color='blue', width=1), name='MA 60'))
        
        fig_k.update_layout(
            xaxis_rangeslider_visible=False, height=400,
            plot_bgcolor='white', paper_bgcolor='white',
            margin=dict(l=10, r=0, t=10, b=10),
            legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, xanchor="left")
        )
        st.plotly_chart(fig_k, use_container_width=True)

    with col_metrics_deep:
        st.markdown("**基本面與技術面總覽**")
        
        # 使用 2x2 或 2x3 的 Grid 顯示關鍵數據
        g1, g2 = st.columns(2)
        
        # 1. 技術面 RSI
        rsi_color = "normal" # 預設黑/灰
        if curr_rsi > 70: rsi_msg = "超買 (過熱)"; rsi_val_color = "#ef4444" # 紅
        elif curr_rsi < 30: rsi_msg = "超賣 (低檔)"; rsi_val_color = "#10b981" # 綠
        else: rsi_msg = "中性區間"; rsi_val_color = "#3b82f6" # 藍
            
        with g1:
            st.metric("RSI (14)", f"{curr_rsi:.1f}", delta=rsi_msg, delta_color="off")
            
        # 2. 基本面 P/E
        pe_val = meta['pe']
        pe_display = f"{pe_val:.1f}" if pe_val else "N/A"
        with g2:
            st.metric("本益比 (P/E)", pe_display, help="越高代表股價越昂貴")
            
        st.markdown("---")
        
        # 3. 更多指標
        m1, m2 = st.columns(2)
        m1.metric("Beta (波動率)", f"{meta['beta']:.2f}", help=">1 代表比大盤波動大")
        
        # 距離 52 週高點的回撤
        high52 = meta['high52']
        curr_p = row['Current Price']
        if high52 > 0:
            drawdown = (curr_p - high52) / high52 * 100
            m2.metric("距 52 週高點", f"{drawdown:.1f}%", help="從最高點回跌的幅度")
