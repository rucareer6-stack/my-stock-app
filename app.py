import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import google.generativeai as genai
from datetime import datetime

# ==========================================
# 1. 頁面配置與 CSS (視覺核心)
# ==========================================
st.set_page_config(page_title="AI 投資戰情室 Pro", layout="wide", page_icon="📉")

# 深色主題 CSS 強制覆蓋
st.markdown("""
    <style>
    /* --- 全局設定 --- */
    .stApp {
        background-color: #0b1120; /* 極深藍黑 (參考圖底色) */
    }
    
    /* 文字顏色強制覆蓋 */
    h1, h2, h3, h4, h5, h6, p, div, span, label {
        color: #e2e8f0 !important; /* 淺灰白 */
        font-family: 'Inter', sans-serif;
    }
    
    /* --- 卡片風格 (Glassmorphism) --- */
    div[data-testid="stMetric"], .css-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);
    }
    
    /* Metric 數值顏色 */
    div[data-testid="stMetricValue"] {
        color: #38bdf8 !important; /* 霓虹藍 */
        font-weight: 700;
    }
    
    /* --- 按鈕風格 (仿照熱門題材) --- */
    .stButton > button {
        width: 100%;
        background-color: #1e293b;
        color: #94a3b8;
        border: 1px solid #334155;
        border-radius: 8px;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #2563eb;
        color: white;
        border-color: #3b82f6;
    }
    
    /* --- 表格風格 --- */
    div[data-testid="stDataFrame"] {
        background-color: #1e293b;
        border-radius: 10px;
    }
    
    /* 去除頂部空白 */
    .block-container {
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 狀態管理 (Navigation & Data)
# ==========================================

# 初始化 Session State
if 'page' not in st.session_state:
    st.session_state['page'] = 'dashboard' # 預設首頁
if 'selected_ticker' not in st.session_state:
    st.session_state['selected_ticker'] = 'NVDA'
if 'portfolio' not in st.session_state:
    # 預設持倉數據
    st.session_state['portfolio'] = pd.DataFrame([
        {'Ticker': 'NVDA', 'Cost': 450.0, 'Shares': 20, 'Sector': '半導體'},
        {'Ticker': 'TSLA', 'Cost': 180.0, 'Shares': 15, 'Sector': '電動車'},
        {'Ticker': 'AAPL', 'Cost': 175.0, 'Shares': 30, 'Sector': '消費電子'},
        {'Ticker': 'PLTR', 'Cost': 15.0, 'Shares': 100, 'Sector': 'AI 軟體'},
    ])
if 'cash' not in st.session_state:
    st.session_state['cash'] = 25000.0
if 'gemini_api_key' not in st.session_state:
    st.session_state['gemini_api_key'] = ""

# 導航函數
def go_to_page(page_name, ticker=None):
    st.session_state['page'] = page_name
    if ticker:
        st.session_state['selected_ticker'] = ticker
    st.rerun()

# ==========================================
# 3. 工具函數 (計算與 API)
# ==========================================

@st.cache_data(ttl=1800)
def get_market_news():
    """獲取模擬市場焦點 (取代 Google Trends)"""
    try:
        # 抓取熱門科技股新聞作為市場風向
        tickers = yf.Tickers("NVDA AAPL TSLA")
        news_list = []
        for t in ["NVDA", "AAPL", "TSLA"]:
            news = tickers.tickers[t].news
            if news:
                for n in news[:2]: # 各取2則
                    news_list.append(f"🔥 [{t}] {n['title']}")
        return news_list if news_list else ["系統暫時無法獲取即時新聞"]
    except:
        return ["無法連接市場數據服務"]

@st.cache_data(ttl=3600)
def get_stock_metrics(ticker):
    """計算年化報酬與基本面"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5y")
        info = stock.info
        
        if hist.empty: return None
        
        current = hist['Close'].iloc[-1]
        
        def cagr(years):
            days = years * 252
            if len(hist) < days: return None
            start = hist['Close'].iloc[-days]
            return ((current / start) ** (1/years)) - 1

        return {
            "1Y": cagr(1), "3Y": cagr(3), "5Y": cagr(5),
            "PE": info.get('trailingPE'), "Beta": info.get('beta'),
            "High52": info.get('fiftyTwoWeekHigh'), "Low52": info.get('fiftyTwoWeekLow'),
            "Target": info.get('targetMeanPrice'), "Summary": info.get('longBusinessSummary')
        }
    except:
        return None

# ==========================================
# 4. 頁面：Dashboard (儀表板)
# ==========================================
if st.session_state['page'] == 'dashboard':
    
    # --- Header ---
    c1, c2 = st.columns([3, 1])
    with c1:
        st.title("洞悉 美股未來")
        st.caption("全方位 AI 投資管理系統")
    with c2:
        # API Key 設定 (隱藏式)
        with st.expander("🔑 設定 API Key"):
            key = st.text_input("Gemini Key", value=st.session_state['gemini_api_key'], type="password")
            if key: st.session_state['gemini_api_key'] = key

    # --- 搜尋與熱門題材 (仿圖 4/5) ---
    st.markdown("#### ⚡ 熱門題材 (Hot Themes)")
    
    # 使用 Columns 模擬按鈕列
    bc1, bc2, bc3, bc4, bc5 = st.columns(5)
    if bc1.button("🤖 AI 伺服器", help="NVDA, SMCI, DELL"):
        st.toast("已切換關注：AI 伺服器板塊")
    if bc2.button("⚙️ 先進製程", help="TSM, ASML"):
        st.toast("已切換關注：先進製程")
    if bc3.button("🚗 電動車", help="TSLA, RIVN"):
        st.toast("已切換關注：電動車產業")
    if bc4.button("☁️ 雲端運算", help="MSFT, AMZN, GOOGL"):
        st.toast("已切換關注：雲端運算")
    if bc5.button("📱 消費電子", help="AAPL"):
        st.toast("已切換關注：消費電子")

    st.markdown("---")

    # --- 資產總覽卡片 (可點擊跳轉) ---
    # 計算資產
    df = st.session_state['portfolio'].copy()
    
    # 預加載現價
    prices = {}
    for t in df['Ticker']:
        try:
            prices[t] = yf.Ticker(t).fast_info['last_price']
        except:
            prices[t] = 0
            
    df['Price'] = df['Ticker'].map(prices)
    df['Value'] = df['Price'] * df['Shares']
    df['Profit'] = (df['Price'] - df['Cost']) * df['Shares']
    
    total_stock = df['Value'].sum()
    total_cash = st.session_state['cash']
    total_assets = total_stock + total_cash
    cash_pct = (total_cash / total_assets * 100) if total_assets > 0 else 0

    # 使用 columns 佈局模擬 Dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("總資產", f"${total_assets:,.0f}")
        if st.button("查看資產詳情 >"):
            go_to_page('details')
            
    with col2:
        st.metric("未實現損益", f"${df['Profit'].sum():,.0f}", delta_color="normal")
    
    with col3:
        st.metric("現金水位", f"{cash_pct:.1f}%")
        st.caption("建議水位: 20-30%")
        
    with col4:
        st.metric("持有檔數", f"{len(df)} 檔")
        if st.button("AI 投資建議 >"):
            go_to_page('ai_report')

    # --- 輿情跑馬燈 ---
    st.markdown("#### 📰 今日市場焦點")
    news = get_market_news()
    for n in news:
        st.info(n)

    # --- 持倉概況 (快速瀏覽) ---
    st.markdown("#### 💼 我的持倉 (點擊代號分析)")
    
    # 製作一個可點擊的列表
    cols = st.columns(len(df))
    for i, row in df.iterrows():
        with cols[i % 4]: # 每行顯示4個
             # 模擬卡片
            st.markdown(f"""
            <div class="css-card" style="text-align: center; margin-bottom: 10px;">
                <h3 style="color: #38bdf8 !important;">{row['Ticker']}</h3>
                <p>${row['Price']:.2f}</p>
                <span style="color: {'#4ade80' if row['Profit']>0 else '#f87171'}">
                    {'+' if row['Profit']>0 else ''}{row['Profit']:.0f}
                </span>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"分析 {row['Ticker']}", key=f"btn_{row['Ticker']}"):
                go_to_page('analysis', row['Ticker'])

# ==========================================
# 5. 頁面：持倉細節 (Details)
# ==========================================
elif st.session_state['page'] == 'details':
    st.button("← 返回總覽", on_click=lambda: go_to_page('dashboard'))
    st.title("📋 持倉與績效深度報表")
    
    # 更新現金
    new_cash = st.number_input("調整現金餘額 (USD)", value=st.session_state['cash'])
    if new_cash != st.session_state['cash']:
        st.session_state['cash'] = new_cash
        st.rerun()

    df = st.session_state['portfolio'].copy()
    
    # 計算詳細數據
    display_data = []
    with st.spinner("正在計算年化報酬 (CAGR)..."):
        for idx, row in df.iterrows():
            metrics = get_stock_metrics(row['Ticker'])
            current_price = yf.Ticker(row['Ticker']).fast_info['last_price']
            val = current_price * row['Shares']
            prof = val - (row['Cost'] * row['Shares'])
            
            display_data.append({
                "代號": row['Ticker'],
                "現價": f"${current_price:.2f}",
                "成本": f"${row['Cost']:.2f}",
                "市值": f"${val:,.0f}",
                "損益": f"${prof:,.0f}",
                "1年報酬": f"{metrics['1Y']*100:.1f}%" if metrics and metrics['1Y'] else "-",
                "3年報酬": f"{metrics['3Y']*100:.1f}%" if metrics and metrics['3Y'] else "-",
                "5年報酬": f"{metrics['5Y']*100:.1f}%" if metrics and metrics['5Y'] else "-",
                "Beta": f"{metrics['Beta']:.2f}" if metrics else "-"
            })
    
    st.dataframe(pd.DataFrame(display_data), use_container_width=True, height=500)

# ==========================================
# 6. 頁面：個股深度分析 (Analysis)
# ==========================================
elif st.session_state['page'] == 'analysis':
    ticker = st.session_state['selected_ticker']
    
    c_head_1, c_head_2 = st.columns([1, 5])
    with c_head_1:
        st.button("← 返回", on_click=lambda: go_to_page('dashboard'))
    with c_head_2:
        st.title(f"{ticker} 深度戰情分析")

    # 獲取數據
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y")
    info = stock.info
    
    # --- Layout 仿圖 3 (左圖右資訊) ---
    col_chart, col_info = st.columns([2, 1])
    
    with col_chart:
        st.subheader("📊 真實 K 線 (Real-time K-Line)")
        
        # 繪製專業 K 線圖
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=hist.index,
                        open=hist['Open'], high=hist['High'],
                        low=hist['Low'], close=hist['Close'],
                        name='K線',
                        increasing_line_color='#22c55e', decreasing_line_color='#ef4444'))
        
        # 增加均線
        hist['MA20'] = hist['Close'].rolling(20).mean()
        hist['MA60'] = hist['Close'].rolling(60).mean()
        fig.add_trace(go.Scatter(x=hist.index, y=hist['MA20'], line=dict(color='orange', width=1), name='MA20'))
        fig.add_trace(go.Scatter(x=hist.index, y=hist['MA60'], line=dict(color='#38bdf8', width=1), name='MA60'))
        
        # 深色圖表設定
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15, 23, 42, 0.5)',
            font=dict(color='#94a3b8'),
            xaxis_rangeslider_visible=False,
            margin=dict(l=10, r=10, t=30, b=10),
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # AI 基本面短評
        st.markdown("### 🤖 AI 基本面摘要")
        st.info(info.get('longBusinessSummary', '無資料'))

    with col_info:
        # 右側資訊欄 (仿圖 3 右側)
        
        # 1. 當前價格大字
        current_price = hist['Close'].iloc[-1]
        change = current_price - hist['Open'].iloc[-1]
        color = "#22c55e" if change > 0 else "#ef4444"
        
        st.markdown(f"""
        <div style="background-color: #1e293b; padding: 15px; border-radius: 10px; text-align: center; border: 1px solid {color};">
            <h1 style="color: {color} !important; margin:0;">${current_price:.2f}</h1>
            <p style="color: {color} !important; margin:0;">{change:+.2f} ({change/current_price*100:.2f}%)</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("") # spacer

        # 2. 儀表板 (Gauge) - 使用 Plotly
        rsi = 100 - (100 / (1 + (hist['Close'].diff().clip(lower=0).rolling(14).mean() / hist['Close'].diff().clip(upper=0).abs().rolling(14).mean()).iloc[-1]))
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = rsi,
            title = {'text': "AI 信心指標 (RSI)"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "#38bdf8"},
                'steps': [
                    {'range': [0, 30], 'color': "rgba(34, 197, 94, 0.3)"},
                    {'range': [70, 100], 'color': "rgba(239, 68, 68, 0.3)"}],
            }
        ))
        fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"}, height=250, margin=dict(l=20, r=20, t=0, b=0))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # 3. 關鍵價位 (Support/Resistance)
        high_52 = info.get('fiftyTwoWeekHigh', 0)
        low_52 = info.get('fiftyTwoWeekLow', 0)
        
        st.markdown("#### 🗝️ 關鍵價位 (Levels)")
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #94a3b8;">
            <span>Support (52L)</span>
            <span>Resistance (52H)</span>
        </div>
        <div style="background: #334155; height: 6px; border-radius: 3px; position: relative; margin: 5px 0 15px 0;">
            <div style="background: #38bdf8; width: {(current_price-low_52)/(high_52-low_52)*100}%; height: 100%; border-radius: 3px;"></div>
        </div>
        <div style="display: flex; justify-content: space-between; font-weight: bold;">
            <span style="color: #22c55e;">${low_52}</span>
            <span style="color: #ef4444;">${high_52}</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.metric("本益比 (P/E)", f"{info.get('trailingPE', 'N/A')}")
        st.metric("目標價", f"${info.get('targetMeanPrice', 'N/A')}")

# ==========================================
# 7. 頁面：AI 智囊報告 (AI Report)
# ==========================================
elif st.session_state['page'] == 'ai_report':
    st.button("← 返回總覽", on_click=lambda: go_to_page('dashboard'))
    st.title("🤖 Gemini 深度智囊報告")
    
    if not st.session_state['gemini_api_key']:
        st.warning("⚠️ 請先在首頁設定 Gemini API Key 才能啟用此功能。")
    else:
        # 生成報告
        if st.button("✨ 啟動 AI 分析 (分析持倉與現金水位)"):
            with st.spinner("AI 正在閱讀您的投資組合..."):
                try:
                    genai.configure(api_key=st.session_state['gemini_api_key'])
                    model = genai.GenerativeModel('gemini-pro')
                    
                    # 準備資料
                    pf = st.session_state['portfolio']
                    cash = st.session_state['cash']
                    total = pf['Shares'] * pf['Ticker'].apply(lambda x: yf.Ticker(x).fast_info['last_price'])
                    pf_text = pf.to_string()
                    
                    prompt = f"""
                    你是一個專業的避險基金經理。
                    使用者目前現金: ${cash}
                    持有股票: 
                    {pf_text}
                    
                    請用繁體中文，輸出一段專業的投資建議，包含：
                    1. **資金效率分析** (現金是否太多？)
                    2. **板塊風險** (是否太集中某產業？)
                    3. **AI 建議板塊** (根據目前缺口，建議關注哪些互補板塊？)
                    請用 Markdown 格式，條列式輸出，語氣要像彭博社報告。
                    """
                    
                    response = model.generate_content(prompt)
                    st.markdown("### 📝 投資總結")
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"AI 分析發生錯誤: {e}")
