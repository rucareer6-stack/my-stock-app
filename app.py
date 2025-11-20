import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
from datetime import datetime, timedelta

# --- 1. 頁面與 UI 設定 (深藍色調) ---
st.set_page_config(page_title="美股 AI 智囊戰情室", layout="wide", page_icon="📈")

# 自定義 CSS: 深藍色主題
st.markdown("""
    <style>
    /* 全局背景 */
    .stApp {
        background-color: #0f172a; /* 深藍色背景 */
        color: #e2e8f0; /* 淺灰白文字 */
    }
    /* 側邊欄背景 */
    [data-testid="stSidebar"] {
        background-color: #1e293b;
    }
    /* 卡片/區塊背景 */
    div[data-testid="stMetric"], div.stDataFrame, div.stPlotlyChart {
        background-color: #1e293b;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.3);
    }
    /* 輸入框優化 */
    .stTextInput > div > div > input {
        color: #ffffff;
    }
    h1, h2, h3 {
        color: #38bdf8 !important; /* 標題亮藍色 */
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. 初始化 Session State ---
if 'portfolio' not in st.session_state:
    # 範例數據
    st.session_state['portfolio'] = pd.DataFrame([
        {'Ticker': 'NVDA', 'Cost': 450.0, 'Shares': 10},
        {'Ticker': 'AAPL', 'Cost': 170.0, 'Shares': 20},
        {'Ticker': 'TSLA', 'Cost': 200.0, 'Shares': 15}
    ])

if 'cash' not in st.session_state:
    st.session_state['cash'] = 15000.0

if 'gemini_api_key' not in st.session_state:
    st.session_state['gemini_api_key'] = ""

# --- 3. 核心函數 ---

# 獲取年化報酬率 (CAGR)
@st.cache_data(ttl=3600)
def get_stock_performance(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5y")
        if hist.empty:
            return None, None, None
        
        current = hist['Close'].iloc[-1]
        
        def calc_cagr(years):
            if len(hist) < years * 252: return None # 數據不足
            start_price = hist['Close'].iloc[-int(years*252)]
            return ((current / start_price) ** (1/years)) - 1

        r1y = calc_cagr(1)
        r3y = calc_cagr(3)
        r5y = calc_cagr(5)
        
        return r1y, r3y, r5y
    except:
        return None, None, None

# 獲取市場熱門新聞 (模擬 Google 趨勢)
@st.cache_data(ttl=3600)
def get_market_trends():
    try:
        # 使用 SPY 和 QQQ 的新聞作為市場熱點代理
        spy = yf.Ticker("SPY")
        news = spy.news[:5] # 取前5則
        trends = []
        for n in news:
            trends.append(f"🔥 {n['title']}")
        return trends
    except:
        return ["無法獲取即時新聞"]

# 計算 RSI
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- 4. 側邊欄導航與設定 ---
with st.sidebar:
    st.title("🚀 導航中心")
    page = st.radio("前往頁面", ["🏠 資產總覽 (Dashboard)", "📋 持倉細節與績效", "🔍 個股深度分析", "🤖 AI 智囊報告"])
    
    st.markdown("---")
    st.subheader("⚙️ 設定")
    
    # API Key 輸入
    api_key_input = st.text_input("輸入 Gemini API Key (用於 AI 分析)", type="password", value=st.session_state['gemini_api_key'])
    if api_key_input:
        st.session_state['gemini_api_key'] = api_key_input

    # 簡單的持倉管理 (保留在側邊欄以便隨時新增)
    with st.expander("快速新增交易"):
        new_ticker = st.text_input("代碼", placeholder="NVDA").upper()
        new_cost = st.number_input("成本價", min_value=0.0)
        new_shares = st.number_input("股數", min_value=0.0)
        if st.button("加入"):
            new_row = {'Ticker': new_ticker, 'Cost': new_cost, 'Shares': new_shares}
            df = st.session_state['portfolio']
            if new_ticker in df['Ticker'].values:
                df.loc[df['Ticker'] == new_ticker, ['Cost', 'Shares']] = [new_cost, new_shares]
            else:
                st.session_state['portfolio'] = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            st.success("已更新")
            st.rerun()
            
    # 現金更新
    cash_input = st.number_input("更新現金餘額", value=st.session_state['cash'])
    if cash_input != st.session_state['cash']:
        st.session_state['cash'] = cash_input
        st.rerun()

# --- 數據預處理 (所有頁面共用) ---
df_port = st.session_state['portfolio'].copy()
if not df_port.empty:
    tickers = df_port['Ticker'].tolist()
    # 簡單緩存價格獲取
    current_prices = {}
    sectors = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            current_prices[t] = info.get('currentPrice') or info.get('previousClose')
            sectors[t] = info.get('sector', 'Unknown')
        except:
            current_prices[t] = 0
            sectors[t] = 'Unknown'
            
    df_port['Current Price'] = df_port['Ticker'].map(current_prices)
    df_port['Market Value'] = df_port['Current Price'] * df_port['Shares']
    df_port['Profit'] = (df_port['Current Price'] - df_port['Cost']) * df_port['Shares']
    df_port['Sector'] = df_port['Ticker'].map(sectors)

total_stock_val = df_port['Market Value'].sum() if not df_port.empty else 0
total_cash = st.session_state['cash']
total_assets = total_stock_val + total_cash
cash_ratio = (total_cash / total_assets * 100) if total_assets > 0 else 0

# --- 5. 頁面邏輯 ---

# === PAGE 1: 資產總覽 ===
if page == "🏠 資產總覽 (Dashboard)":
    st.title("🏠 資產總覽與分配")
    
    # 輿情跑馬燈
    st.subheader("🔥 今日市場焦點 (Google Trends / News)")
    trends = get_market_trends()
    st.info(" | ".join(trends))

    # 關鍵指標
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💰 總資產", f"${total_assets:,.0f}")
    col2.metric("📈 總未實現損益", f"${df_port['Profit'].sum():,.0f}", delta_color="normal")
    col3.metric("💵 現金水位", f"{cash_ratio:.1f}%")
    col4.metric("🏦 股票市值", f"${total_stock_val:,.0f}")

    # 現金水位警告
    if cash_ratio < 10:
        st.warning("⚠️ 現金水位過低 (<10%)，建議風險控管。")
    elif cash_ratio > 50:
        st.info("💡 現金充裕 (>50%)，可關注 AI 建議的加碼機會。")

    # 圖表區
    c1, c2 = st.columns(2)
    with c1:
        # 資產配置圓餅圖
        fig_alloc = px.pie(names=['股票', '現金'], values=[total_stock_val, total_cash], 
                           title="資產配置", hole=0.5, color_discrete_sequence=['#38bdf8', '#94a3b8'])
        fig_alloc.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig_alloc, use_container_width=True)
    
    with c2:
        # 持倉佔比
        if not df_port.empty:
            fig_hold = px.pie(df_port, values='Market Value', names='Ticker', 
                              title="持股權重分析", color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_hold.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
            st.plotly_chart(fig_hold, use_container_width=True)

# === PAGE 2: 持倉細節與績效 ===
elif page == "📋 持倉細節與績效":
    st.title("📋 持倉深度報表")
    
    if df_port.empty:
        st.write("暫無持倉。")
    else:
        # 計算年化報酬並加入 Table
        st.write("正在計算歷史年化報酬，請稍候...")
        
        perf_data = []
        for t in df_port['Ticker']:
            r1, r3, r5 = get_stock_performance(t)
            perf_data.append({
                '1Y Return': f"{r1*100:.1f}%" if r1 else "N/A",
                '3Y Return': f"{r3*100:.1f}%" if r3 else "N/A",
                '5Y Return': f"{r5*100:.1f}%" if r5 else "N/A"
            })
        
        df_perf = pd.DataFrame(perf_data)
        df_display = pd.concat([df_port, df_perf], axis=1)
        
        # 顯示表格
        st.dataframe(
            df_display[['Ticker', 'Shares', 'Cost', 'Current Price', 'Profit', '1Y Return', '3Y Return', '5Y Return', 'Sector']],
            use_container_width=True,
            height=400
        )
        st.caption("* 1Y/3Y/5Y Return 為該股票本身的年化報酬率 (CAGR)，非您的持有報酬。")

# === PAGE 3: 個股深度分析 ===
elif page == "🔍 個股深度分析":
    st.title("🔍 個股全方位分析")
    
    if df_port.empty:
        st.warning("請先新增持倉。")
    else:
        selected = st.selectbox("選擇股票", df_port['Ticker'].unique())
        
        if selected:
            stock = yf.Ticker(selected)
            info = stock.info
            hist = stock.history(period="1y")
            
            # 1. 頂部數據條
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("今日開盤", f"${hist['Open'].iloc[-1]:.2f}")
            m2.metric("今日最高", f"${hist['High'].iloc[-1]:.2f}")
            m3.metric("今日最低", f"${hist['Low'].iloc[-1]:.2f}")
            m4.metric("市值", f"${info.get('marketCap', 0)/1e9:.1f}B")
            m5.metric("本益比 P/E", f"{info.get('trailingPE', 'N/A')}")

            # 2. K線圖與技術指標
            hist['SMA20'] = hist['Close'].rolling(20).mean()
            hist['SMA60'] = hist['Close'].rolling(60).mean()
            
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=hist.index,
                            open=hist['Open'], high=hist['High'],
                            low=hist['Low'], close=hist['Close'], name='Price'))
            fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA20'], line=dict(color='orange', width=1), name='月線 (20MA)'))
            fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA60'], line=dict(color='purple', width=1), name='季線 (60MA)'))
            
            fig.update_layout(title=f"{selected} 股價走勢", 
                              paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                              font_color='white', height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # 3. 基本面敘述與關鍵價位
            c_left, c_right = st.columns([3, 2])
            with c_left:
                st.subheader("📜 公司業務與基本面")
                st.write(info.get('longBusinessSummary', '無詳細敘述。'))
                
            with c_right:
                st.subheader("🎯 關鍵統計")
                st.write(f"**52週高點:** ${info.get('fiftyTwoWeekHigh')}")
                st.write(f"**52週低點:** ${info.get('fiftyTwoWeekLow')}")
                st.write(f"**分析師目標價:** ${info.get('targetMeanPrice', 'N/A')}")
                st.write(f"**機構持股:** {info.get('heldPercentInstitutions', 0)*100:.1f}%")

# === PAGE 4: AI 智囊報告 ===
elif page == "🤖 AI 智囊報告":
    st.title("🤖 Gemini AI 投資顧問")
    
    st.markdown("""
    此功能將整合您的 **持倉數據** 與 **現金水位**，並發送給 Google Gemini 模型，
    為您生成一份客製化的投資建議與板塊分析。
    """)
    
    if not st.session_state['gemini_api_key']:
        st.error("🔴 請先在左側側邊欄輸入您的 Gemini API Key。")
        st.markdown("[點此免費申請 Google Gemini API Key](https://aistudio.google.com/app/apikey)")
    else:
        if st.button("✨ 生成深度智囊報告"):
            with st.spinner("AI 正在分析您的資產組合與市場數據..."):
                try:
                    # 準備 Prompt 數據
                    genai.configure(api_key=st.session_state['gemini_api_key'])
                    model = genai.GenerativeModel('gemini-pro')
                    
                    portfolio_str = df_port[['Ticker', 'Shares', 'Cost', 'Profit', 'Sector']].to_string()
                    prompt = f"""
                    角色：你是一位專業的華爾街投資顧問。
                    
                    使用者資產狀況：
                    1. 總資產：${total_assets} USD
                    2. 現金水位：{cash_ratio:.1f}% (金額：${total_cash})
                    3. 持倉詳情：
                    {portfolio_str}
                    
                    請提供以下分析（請用繁體中文，語氣專業且具建設性）：
                    1. **投資組合體檢**：評論目前的產業分散度與風險（例如是否太集中科技股）。
                    2. **現金水位建議**：根據目前的現金比例，建議應該加碼還是保留現金？
                    3. **操作建議**：針對持有的股票，提供簡單的操作建議（續抱/減碼/加碼）。
                    4. **關注板塊**：根據目前的持倉缺口，建議未來可以關注哪些互補的板塊或 ETF。
                    """
                    
                    response = model.generate_content(prompt)
                    st.success("分析完成！")
                    st.markdown("### 📋 AI 深度分析報告")
                    st.markdown("---")
                    st.markdown(response.text)
                    
                except Exception as e:
                    st.error(f"AI 分析失敗，請檢查 API Key 是否正確。\n錯誤訊息: {e}")
