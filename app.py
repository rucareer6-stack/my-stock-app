import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import google.generativeai as genai
from io import BytesIO

# ==========================================
# 1. 頁面配置與 CSS (視覺核心 - 深色戰情室版)
# ==========================================
st.set_page_config(page_title="AI 投資戰情室 Ultimate", layout="wide", page_icon="📉")

# 強制深色主題 CSS
st.markdown("""
    <style>
    /* --- 全局背景 --- */
    .stApp {
        background-color: #0b1120; /* 深藍黑背景 */
    }
    
    /* --- 文字顏色強制反白 --- */
    h1, h2, h3, h4, h5, h6, p, div, span, label, li {
        color: #e2e8f0 !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* --- 按鈕優化 (解決白色突兀問題) --- */
    div.stButton > button {
        background-color: #1e293b !important;
        color: #38bdf8 !important; /* 霓虹藍字 */
        border: 1px solid #334155 !important;
        border-radius: 8px;
        transition: all 0.3s ease;
        font-weight: 600;
    }
    div.stButton > button:hover {
        background-color: #38bdf8 !important;
        color: #0f172a !important; /* 懸停變黑字 */
        border-color: #38bdf8 !important;
        transform: translateY(-2px);
    }
    
    /* --- 卡片風格 (Glassmorphism) --- */
    div[data-testid="stMetric"], div.stDataFrame {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    /* --- Metric 數值顏色 --- */
    div[data-testid="stMetricValue"] {
        color: #38bdf8 !important;
        font-size: 28px !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
    }
    
    /* --- 表格優化 --- */
    [data-testid="stDataFrame"] {
        border: none;
    }
    
    /* --- 側邊欄 --- */
    [data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #1e293b;
    }
    
    /* --- 消除頂部留白 --- */
    .block-container {
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 狀態管理 (Session State)
# ==========================================
if 'page' not in st.session_state:
    st.session_state['page'] = 'dashboard'
if 'selected_ticker' not in st.session_state:
    st.session_state['selected_ticker'] = 'NVDA'
if 'gemini_api_key' not in st.session_state:
    st.session_state['gemini_api_key'] = ""
if 'cash' not in st.session_state:
    st.session_state['cash'] = 25000.0
if 'portfolio' not in st.session_state:
    # 預設持倉
    st.session_state['portfolio'] = pd.DataFrame([
        {'Ticker': 'NVDA', 'Cost': 450.0, 'Shares': 20, 'Sector': '半導體'},
        {'Ticker': 'TSLA', 'Cost': 180.0, 'Shares': 15, 'Sector': '電動車'},
        {'Ticker': 'AAPL', 'Cost': 175.0, 'Shares': 30, 'Sector': '消費電子'},
        {'Ticker': 'PLTR', 'Cost': 15.0, 'Shares': 100, 'Sector': 'AI 軟體'},
    ])

def go_to_page(page_name, ticker=None):
    st.session_state['page'] = page_name
    if ticker:
        st.session_state['selected_ticker'] = ticker
    st.rerun()

# ==========================================
# 3. 核心功能函數
# ==========================================

@st.cache_data(ttl=1800)
def get_safe_market_news():
    """獲取市場新聞 (帶有容錯機制)"""
    try:
        # 嘗試獲取 QQQ (那斯達克 ETF) 的新聞，通常比較豐富
        ticker = yf.Ticker("QQQ")
        news = ticker.news
        if news and len(news) > 0:
            formatted_news = []
            for n in news[:3]:
                formatted_news.append(f"🔥 {n['title']}")
            return formatted_news
    except:
        pass
    
    # 如果失敗，返回靜態的熱門話題 (確保 UI 不崩壞)
    return [
        "🔥 NVIDIA 發布最新 AI 晶片，市場預期強勁",
        "⚡ 特斯拉 Cybertruck 產能提升，股價震盪",
        "📈 聯準會暗示降息路徑，科技股受惠"
    ]

def get_stock_info_safe(ticker):
    """獲取個股資訊 (處理數字格式與空值)"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="1y")
        
        if hist.empty: return None
        
        current_price = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
        change_pct = (current_price - prev_close) / prev_close * 100

        # 安全獲取數據並格式化
        def safe_fmt(key, fmt="{:.2f}"):
            val = info.get(key)
            return fmt.format(val) if val is not None and isinstance(val, (int, float)) else "N/A"

        return {
            "price": current_price,
            "change_pct": change_pct,
            "pe": safe_fmt('trailingPE'),
            "target": safe_fmt('targetMeanPrice'),
            "high52": safe_fmt('fiftyTwoWeekHigh'),
            "low52": safe_fmt('fiftyTwoWeekLow'),
            "beta": safe_fmt('beta'),
            "summary": info.get('longBusinessSummary', 'No summary available.'),
            "hist": hist
        }
    except Exception as e:
        return None

# ==========================================
# 4. 側邊欄 (資料管理)
# ==========================================
with st.sidebar:
    st.title("⚙️ 控制台")
    
    # API Key
    st.caption("AI 分析功能需設定 API Key")
    key_input = st.text_input("Gemini API Key", value=st.session_state['gemini_api_key'], type="password")
    if key_input: st.session_state['gemini_api_key'] = key_input
    
    st.markdown("---")
    st.subheader("💾 資料備份/還原")
    
    # 匯出 CSV
    csv = st.session_state['portfolio'].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ 下載持倉備份 (CSV)",
        data=csv,
        file_name='my_portfolio.csv',
        mime='text/csv',
    )
    
    # 匯入 CSV
    uploaded_file = st.file_uploader("⬆️ 上傳備份檔", type=['csv'])
    if uploaded_file is not None:
        try:
            df_uploaded = pd.read_csv(uploaded_file)
            required_cols = ['Ticker', 'Cost', 'Shares']
            if all(col in df_uploaded.columns for col in required_cols):
                st.session_state['portfolio'] = df_uploaded
                st.success("資料已還原！")
                st.rerun()
            else:
                st.error("格式錯誤：CSV 必須包含 Ticker, Cost, Shares")
        except:
            st.error("讀取失敗")

# ==========================================
# 5. 頁面：Dashboard (首頁)
# ==========================================
if st.session_state['page'] == 'dashboard':
    
    st.title("洞悉 美股未來")
    st.caption("全方位 AI 投資管理系統 V2.0")

    # --- 熱門題材按鈕列 (修復樣式) ---
    st.subheader("⚡ 熱門題材 (Hot Themes)")
    b1, b2, b3, b4, b5 = st.columns(5)
    
    # 使用 callback 函數避免頁面重新載入太慢
    def toast_msg(msg):
        st.toast(f"已切換關注：{msg}", icon="✅")

    if b1.button("🤖 AI 伺服器"): toast_msg("AI 伺服器")
    if b2.button("⚙️ 先進製程"): toast_msg("先進製程")
    if b3.button("🚗 電動車"): toast_msg("電動車")
    if b4.button("☁️ 雲端運算"): toast_msg("雲端運算")
    if b5.button("💊 生技醫療"): toast_msg("生技醫療")

    st.write("") # Spacer

    # --- 資產計算 ---
    pf = st.session_state['portfolio'].copy()
    
    # 批量獲取現價 (優化速度)
    tickers_str = " ".join(pf['Ticker'].tolist())
    if tickers_str:
        live_data = yf.download(tickers_str, period="1d", progress=False)['Close']
        # 處理單支股票 vs 多支股票的數據結構差異
        current_prices = {}
        if isinstance(live_data, pd.DataFrame) and not live_data.empty:
             # 取最後一筆非 NaN 的數據
            last_row = live_data.iloc[-1]
            for t in pf['Ticker']:
                try:
                    current_prices[t] = last_row[t]
                except:
                    current_prices[t] = 0
        elif isinstance(live_data, pd.Series):
             current_prices[pf['Ticker'][0]] = live_data.iloc[-1]
    else:
        current_prices = {}

    pf['Price'] = pf['Ticker'].map(current_prices).fillna(0)
    pf['Value'] = pf['Price'] * pf['Shares']
    pf['Profit'] = (pf['Price'] - pf['Cost']) * pf['Shares']
    
    total_assets = pf['Value'].sum() + st.session_state['cash']
    total_profit = pf['Profit'].sum()
    cash_ratio = (st.session_state['cash'] / total_assets * 100) if total_assets > 0 else 0

    # --- 資產儀表板 ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("總資產", f"${total_assets:,.0f}")
    c2.metric("總未實現損益", f"${total_profit:,.0f}", delta_color="normal")
    c3.metric("現金水位", f"{cash_ratio:.1f}%")
    c4.metric("持有檔數", f"{len(pf)} 檔")
    
    if c1.button("查看資產詳情 >"): go_to_page('details')
    if c4.button("AI 投資診斷 >"): go_to_page('ai_report')

    # --- 市場焦點 (修復數據源) ---
    st.markdown("### 📰 今日市場焦點")
    news_items = get_safe_market_news()
    for item in news_items:
        st.info(item)

    # --- 持倉卡片區 ---
    st.markdown("### 💼 我的持倉 (點擊分析)")
    cols = st.columns(4)
    for i, row in pf.iterrows():
        with cols[i % 4]:
            # 自定義卡片 HTML
            color = "#4ade80" if row['Profit'] > 0 else "#f87171"
            st.markdown(f"""
            <div style="background-color: #1e293b; border:1px solid #334155; border-radius:10px; padding:15px; margin-bottom:10px; text-align:center;">
                <h3 style="color:#38bdf8 !important; margin:0;">{row['Ticker']}</h3>
                <p style="font-size:14px; color:#94a3b8 !important;">${row['Price']:.2f}</p>
                <p style="color:{color} !important; font-weight:bold;">{'+' if row['Profit']>0 else ''}{row['Profit']:.0f}</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"分析 {row['Ticker']}", key=f"btn_{row['Ticker']}"):
                go_to_page('analysis', row['Ticker'])

# ==========================================
# 6. 頁面：持倉細節 (Details)
# ==========================================
elif st.session_state['page'] == 'details':
    st.button("← 返回首頁", on_click=lambda: go_to_page('dashboard'))
    st.title("📋 持倉詳細報表")
    
    # 現金調整
    c_input = st.number_input("調整現金餘額 (USD)", value=st.session_state['cash'])
    if c_input != st.session_state['cash']:
        st.session_state['cash'] = c_input
        st.rerun()

    # 顯示表格 (修復小數點)
    pf = st.session_state['portfolio'].copy()
    # 重新獲取價格以確保準確
    tickers_str = " ".join(pf['Ticker'].tolist())
    if tickers_str:
        data = yf.download(tickers_str, period="1d", progress=False)['Close']
        # Logic to handle series vs dataframe
        last_prices = data.iloc[-1] if isinstance(data, pd.DataFrame) else data
        
        current_p = []
        if isinstance(last_prices, pd.Series):
            for t in pf['Ticker']:
                current_p.append(last_prices.get(t, 0))
        else:
             current_p.append(last_prices) # Single stock case
             
        pf['Current Price'] = current_p
    
    pf['Market Value'] = pf['Current Price'] * pf['Shares']
    pf['Profit/Loss'] = (pf['Current Price'] - pf['Cost']) * pf['Shares']
    pf['Return %'] = (pf['Profit/Loss'] / (pf['Cost'] * pf['Shares']) * 100).fillna(0)

    # 格式化顯示
    st.dataframe(
        pf.style.format({
            "Cost": "${:.2f}",
            "Shares": "{:.0f}",
            "Current Price": "${:.2f}",
            "Market Value": "${:,.0f}",
            "Profit/Loss": "${:,.0f}",
            "Return %": "{:.2f}%"
        }),
        use_container_width=True,
        height=500
    )

# ==========================================
# 7. 頁面：個股分析 (Analysis - 核心修復)
# ==========================================
elif st.session_state['page'] == 'analysis':
    ticker = st.session_state['selected_ticker']
    
    # 頂部導航
    c_back, c_title = st.columns([1, 6])
    with c_back:
        st.button("← 返回", on_click=lambda: go_to_page('dashboard'))
    with c_title:
        st.title(f"{ticker} 深度戰情分析")

    # 獲取數據
    data = get_stock_info_safe(ticker)
    
    if not data:
        st.error("無法獲取數據，請稍後再試。")
    else:
        # --- 版面配置：左圖表，右數據 ---
        col_chart, col_metrics = st.columns([2, 1])
        
        with col_chart:
            # K 線圖
            st.subheader("📊 股價走勢 (K-Line)")
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=data['hist'].index,
                            open=data['hist']['Open'], high=data['hist']['High'],
                            low=data['hist']['Low'], close=data['hist']['Close'],
                            name='Price'))
            
            # 均線
            ma20 = data['hist']['Close'].rolling(20).mean()
            fig.add_trace(go.Scatter(x=data['hist'].index, y=ma20, line=dict(color='orange', width=1.5), name='MA20'))
            
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(20, 30, 50, 0.5)',
                xaxis_rangeslider_visible=False,
                font=dict(color='#94a3b8'),
                height=450,
                margin=dict(l=10, r=10, t=30, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # --- 這裡解決文字閱讀問題 ---
            st.subheader("🤖 公司簡介 (AI 翻譯)")
            
            # 顯示原始英文 (摺疊)
            with st.expander("📄 顯示原始英文簡介"):
                st.write(data['summary'])
            
            # AI 翻譯按鈕
            if st.button("✨ 點擊使用 AI 翻譯/摘要 (需 API Key)"):
                if not st.session_state['gemini_api_key']:
                    st.warning("請先在側邊欄設定 Gemini API Key")
                else:
                    with st.spinner("AI 正在閱讀並翻譯..."):
                        try:
                            genai.configure(api_key=st.session_state['gemini_api_key'])
                            model = genai.GenerativeModel('gemini-pro')
                            prompt = f"請將以下公司簡介翻譯成繁體中文，並用條列式列出 3 個核心業務重點：\n{data['summary']}"
                            response = model.generate_content(prompt)
                            st.success("翻譯完成")
                            st.markdown(f"""
                            <div style="background-color:#1e293b; padding:15px; border-radius:10px; border-left: 4px solid #38bdf8;">
                                {response.text}
                            </div>
                            """, unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"AI 服務暫時無法使用: {e}")

        with col_metrics:
            # 右側數據儀表板
            
            # 1. 現價大字卡 (帶顏色)
            color = "#22c55e" if data['change_pct'] >= 0 else "#ef4444"
            st.markdown(f"""
            <div style="background-color: #1e293b; padding: 20px; border-radius: 12px; text-align: center; border: 1px solid {color}; box-shadow: 0 0 15px {color}40;">
                <h1 style="color: {color} !important; margin:0; font-size: 48px;">${data['price']:.2f}</h1>
                <p style="color: {color} !important; margin:0; font-size: 18px;">{data['change_pct']:+.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("")
            
            # 2. 關鍵基本面 (修復小數點)
            st.markdown("#### 🗝️ 關鍵指標")
            m1, m2 = st.columns(2)
            m1.metric("本益比 P/E", data['pe'])
            m2.metric("Beta 係數", data['beta'])
            m3, m4 = st.columns(2)
            m3.metric("目標價", f"${data['target']}" if data['target'] != 'N/A' else 'N/A')
            m4.metric("52週高", f"${data['high52']}")
            
            # 3. 信心儀表板 (RSI)
            diff = data['hist']['Close'].diff()
            gain = diff.where(diff > 0, 0).rolling(14).mean()
            loss = -diff.where(diff < 0, 0).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1]
            
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = rsi,
                title = {'text': "RSI 強度", 'font': {'color': '#e2e8f0'}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickcolor': "white"},
                    'bar': {'color': "#38bdf8"},
                    'bgcolor': "#1e293b",
                    'steps': [
                        {'range': [0, 30], 'color': "rgba(34, 197, 94, 0.3)"},
                        {'range': [70, 100], 'color': "rgba(239, 68, 68, 0.3)"}],
                }
            ))
            fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"}, height=250, margin=dict(t=40, b=10, l=20, r=20))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            st.caption("RSI > 70 超買 (紅區)，< 30 超賣 (綠區)")

# ==========================================
# 8. 頁面：AI 診斷報告
# ==========================================
elif st.session_state['page'] == 'ai_report':
    st.button("← 返回首頁", on_click=lambda: go_to_page('dashboard'))
    st.title("🤖 Gemini 投資組合診斷")
    
    if not st.session_state['gemini_api_key']:
        st.warning("⚠️ 請先在側邊欄輸入 API Key")
        st.markdown("[點此獲取 Google Gemini API Key](https://aistudio.google.com/app/apikey)")
    else:
        st.markdown("""
        <div style="background-color:#1e293b; padding:20px; border-radius:10px;">
            此功能將掃描您的持倉結構與現金水位，提供專業的：<br>
            1. <b>風險評估</b> (集中度分析)<br>
            2. <b>操作建議</b> (加減碼時機)<br>
            3. <b>機會發現</b> (建議關注板塊)
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 開始 AI 深度分析"):
            with st.spinner("AI 基金經理人正在分析中..."):
                try:
                    genai.configure(api_key=st.session_state['gemini_api_key'])
                    model = genai.GenerativeModel('gemini-pro')
                    
                    pf = st.session_state['portfolio']
                    cash = st.session_state['cash']
                    
                    prompt = f"""
                    角色：華爾街資深避險基金經理。
                    用戶資產：現金 ${cash}。
                    持倉列表：
                    {pf.to_string()}
                    
                    請用繁體中文、Markdown 格式，輸出一份嚴謹的投資建議報告。
                    重點包含：
                    1. 現金水位評點 (是否過高/過低？)
                    2. 持股健檢 (有無過度集中風險？)
                    3. 下一步行動建議 (具體的加碼/減碼方向)
                    """
                    response = model.generate_content(prompt)
                    
                    st.markdown("---")
                    st.markdown(response.text)
                    st.success("分析完成")
                except Exception as e:
                    st.error(f"分析失敗: {e}")
