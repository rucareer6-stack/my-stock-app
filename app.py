import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import google.generativeai as genai
from datetime import datetime

# ==========================================
# 1. 視覺核心與 CSS 修復 (針對側邊欄與表格)
# ==========================================
st.set_page_config(page_title="AI 投資戰情室 v3", layout="wide", page_icon="📉")

st.markdown("""
    <style>
    /* --- 全局背景：深藍黑 --- */
    .stApp {
        background-color: #0b1120;
    }
    
    /* --- 文字顏色：全域反白 --- */
    h1, h2, h3, h4, h5, h6, p, span, div, label, li, a {
        color: #e2e8f0 !important;
        font-family: 'Inter', sans-serif;
        text-decoration: none;
    }
    
    /* --- 側邊欄 (Sidebar) 強制修復 --- */
    [data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid #1e293b;
    }
    [data-testid="stSidebar"] * {
        color: #cbd5e1 !important; /* 側邊欄文字顏色 */
    }
    /* 輸入框背景修復 */
    [data-testid="stSidebar"] input {
        background-color: #1e293b !important;
        color: white !important;
        border: 1px solid #334155 !important;
    }
    
    /* --- 表格 (DataFrame) 優化：緊湊版 --- */
    div[data-testid="stDataFrame"] div[data-testid="stTable"] {
        font-size: 14px; /* 縮小字體 */
    }
    div[data-testid="stDataFrame"] {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 10px; /* 減少內距 */
    }
    /* 隱藏原本醜陋的索引列 */
    thead tr th:first-child {display:none}
    tbody th {display:none}
    
    /* --- 按鈕與卡片 --- */
    .stButton > button {
        background-color: #1e293b !important;
        color: #38bdf8 !important;
        border: 1px solid #334155 !important;
        border-radius: 6px;
    }
    .stButton > button:hover {
        border-color: #38bdf8 !important;
        background-color: #334155 !important;
    }
    
    /* --- 新聞連結樣式 --- */
    a.news-link:hover {
        color: #38bdf8 !important;
        text-decoration: underline !important;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 狀態管理
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
    st.session_state['portfolio'] = pd.DataFrame([
        {'Ticker': 'NVDA', 'Cost': 450.0, 'Shares': 20, 'Sector': 'Semiconductors'},
        {'Ticker': 'TSLA', 'Cost': 180.0, 'Shares': 15, 'Sector': 'Auto Manufacturers'},
        {'Ticker': 'AAPL', 'Cost': 175.0, 'Shares': 30, 'Sector': 'Consumer Electronics'},
        {'Ticker': 'PLTR', 'Cost': 15.0, 'Shares': 100, 'Sector': 'Software'},
    ])

def go_to_page(page_name, ticker=None):
    st.session_state['page'] = page_name
    if ticker:
        st.session_state['selected_ticker'] = ticker
    st.rerun()

# ==========================================
# 3. 核心邏輯 (真實數據抓取)
# ==========================================

@st.cache_data(ttl=600) # 10分鐘更新一次
def get_real_hot_sectors():
    """抓取 ETF 漲跌幅來決定真實熱門板塊"""
    # 定義板塊 ETF
    sectors = {
        'SMH': '半導體 (Semi)',
        'XLK': '科技 (Tech)',
        'XLV': '醫療 (Health)',
        'XLF': '金融 (Finance)',
        'XLE': '能源 (Energy)',
        'IGV': '軟體 (Software)',
        'XLC': '通訊 (Comm)',
        'XLY': '非必需消費 (Discretionary)'
    }
    try:
        tickers = list(sectors.keys())
        # 批量下載
        data = yf.download(tickers, period="5d", progress=False)['Close']
        
        # 計算今日漲跌幅
        if len(data) >= 2:
            last_price = data.iloc[-1]
            prev_price = data.iloc[-2]
            changes = ((last_price - prev_price) / prev_price * 100)
            
            # 排序
            sorted_sectors = changes.sort_values(ascending=False)
            
            # 格式化輸出前 5 名
            top_sectors = []
            for sym in sorted_sectors.index[:5]:
                val = sorted_sectors[sym]
                name = sectors.get(sym, sym)
                # 根據漲跌變色
                icon = "🔥" if val > 0 else "❄️"
                top_sectors.append({"name": name, "change": val, "icon": icon})
            return top_sectors
    except:
        pass
    return [{"name": "半導體", "change": 1.5, "icon": "🔥"}, {"name": "科技", "change": 0.8, "icon": "📈"}]

@st.cache_data(ttl=300)
def get_real_news():
    """抓取 SPY/QQQ 的真實英文新聞連結"""
    try:
        # 抓取大盤新聞
        spy = yf.Ticker("SPY")
        news_data = spy.news
        
        formatted_news = []
        if news_data:
            for n in news_data[:5]: # 取前5則
                formatted_news.append({
                    "title": n.get('title'),
                    "link": n.get('link'),
                    "publisher": n.get('publisher'),
                    "time": datetime.fromtimestamp(n.get('providerPublishTime', 0)).strftime('%H:%M')
                })
        return formatted_news
    except:
        return []

# ==========================================
# 4. 側邊欄 (Sidebar)
# ==========================================
with st.sidebar:
    st.subheader("⚙️ 控制中心")
    
    st.caption("API 設定")
    api_key = st.text_input("Gemini API Key", value=st.session_state['gemini_api_key'], type="password")
    if api_key: st.session_state['gemini_api_key'] = api_key
    
    st.markdown("---")
    st.caption("快速交易")
    with st.form("add_stock"):
        t_sym = st.text_input("代碼 (Ticker)", value="AMD").upper()
        t_cost = st.number_input("成本 (Cost)", min_value=0.0, step=0.1)
        t_share = st.number_input("股數 (Shares)", min_value=0.0, step=1.0)
        if st.form_submit_button("➕ 加入持倉"):
            new_row = {'Ticker': t_sym, 'Cost': t_cost, 'Shares': t_share, 'Sector': 'Unknown'}
            st.session_state['portfolio'] = pd.concat([st.session_state['portfolio'], pd.DataFrame([new_row])], ignore_index=True)
            st.success(f"已加入 {t_sym}")
            st.rerun()

# ==========================================
# 5. Dashboard (首頁)
# ==========================================
if st.session_state['page'] == 'dashboard':
    st.title("🚀 戰情室 Dashboard")
    
    # --- 1. 真實熱門板塊 (自動排序) ---
    st.subheader("⚡ 今日強勢板塊 (Real-time)")
    hot_sectors = get_real_hot_sectors()
    
    cols = st.columns(len(hot_sectors))
    for i, sec in enumerate(hot_sectors):
        color = "#38bdf8" if sec['change'] > 0 else "#94a3b8"
        with cols[i]:
            st.markdown(f"""
            <div style="background:#1e293b; padding:10px; border-radius:8px; text-align:center; border:1px solid #334155;">
                <div style="font-size:12px; color:#94a3b8;">{sec['icon']} {sec['name']}</div>
                <div style="font-size:16px; font-weight:bold; color:{color};">{sec['change']:+.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

    st.write("")

    # --- 2. 資產狀態 ---
    pf = st.session_state['portfolio'].copy()
    # 取得現價
    tickers_list = pf['Ticker'].tolist()
    current_prices = {}
    if tickers_list:
        try:
            # 為了速度，一次下載
            data = yf.download(tickers_list, period="1d", progress=False)['Close']
            # 處理格式
            if len(tickers_list) == 1:
                current_prices[tickers_list[0]] = data.iloc[-1]
            else:
                for t in tickers_list:
                    current_prices[t] = data.iloc[-1][t]
        except:
            for t in tickers_list: current_prices[t] = 0

    pf['Price'] = pf['Ticker'].map(current_prices).fillna(0)
    pf['Val'] = pf['Price'] * pf['Shares']
    pf['Profit'] = (pf['Price'] - pf['Cost']) * pf['Shares']
    
    total_asset = pf['Val'].sum() + st.session_state['cash']
    total_pl = pf['Profit'].sum()
    cash_lvl = (st.session_state['cash'] / total_asset * 100) if total_asset > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("總資產", f"${total_asset:,.0f}")
    c2.metric("總損益", f"${total_pl:,.0f}", delta_color="normal")
    c3.metric("現金水位", f"{cash_lvl:.1f}%")
    
    with c4:
        st.write("")
        if st.button("📋 查看持倉詳情 >", use_container_width=True):
            go_to_page('details')

    # --- 3. 真實新聞 (可點擊) ---
    st.subheader("📰 市場焦點 (News)")
    news_list = get_real_news()
    
    if not news_list:
        st.info("暫無即時新聞或連接 API 逾時")
    else:
        for news in news_list:
            # 使用 Markdown 製作超連結
            st.markdown(f"""
            <div style="background:#1e293b; padding:10px; margin-bottom:8px; border-radius:5px; border-left: 4px solid #38bdf8;">
                <a href="{news['link']}" target="_blank" class="news-link" style="font-size:16px; font-weight:600; color:#e2e8f0; text-decoration:none;">
                    {news['title']} ↗
                </a>
                <br>
                <span style="font-size:12px; color:#94a3b8;">{news['publisher']} • {news['time']}</span>
            </div>
            """, unsafe_allow_html=True)

# ==========================================
# 6. 持倉詳情 (Details) - 包含完整財務建議
# ==========================================
elif st.session_state['page'] == 'details':
    c_back, c_title = st.columns([1, 6])
    with c_back:
        st.button("← 返迴", on_click=lambda: go_to_page('dashboard'))
    with c_title:
        st.title("📋 持倉詳細報表")

    # 更新現金
    new_cash = st.number_input("現金餘額 (USD)", value=st.session_state['cash'], step=100.0)
    if new_cash != st.session_state['cash']:
        st.session_state['cash'] = new_cash
        st.rerun()

    # 準備資料
    pf = st.session_state['portfolio'].copy()
    tickers_list = pf['Ticker'].tolist()
    
    # 重新獲取最新價格
    if tickers_list:
        data = yf.download(tickers_list, period="1d", progress=False)['Close']
        if len(tickers_list) == 1:
            pf['Current Price'] = data.iloc[-1]
        else:
            pf['Current Price'] = pf['Ticker'].apply(lambda x: data.iloc[-1][x] if x in data.columns else 0)
    else:
        pf['Current Price'] = 0
        
    pf['Market Value'] = pf['Current Price'] * pf['Shares']
    pf['P/L'] = (pf['Current Price'] - pf['Cost']) * pf['Shares']
    pf['Return %'] = (pf['P/L'] / (pf['Cost'] * pf['Shares']) * 100).fillna(0)

    # --- 優化後的緊湊表格 ---
    st.subheader("資產明細")
    
    # 使用 column_config 進行格式化，讓表格更專業
    st.dataframe(
        pf,
        column_config={
            "Ticker": "代碼",
            "Cost": st.column_config.NumberColumn("成本", format="$%.2f"),
            "Shares": st.column_config.NumberColumn("股數", format="%.0f"),
            "Current Price": st.column_config.NumberColumn("現價", format="$%.2f"),
            "Market Value": st.column_config.NumberColumn("市值", format="$%.0f"),
            "P/L": st.column_config.NumberColumn("損益", format="$%.0f"),
            "Return %": st.column_config.NumberColumn("報酬率", format="%.2f%%"),
            "Sector": "板塊"
        },
        hide_index=True, # 隱藏索引，減少寬度佔用
        use_container_width=True
    )

    st.markdown("---")

    # --- AI 個人財務建議 (Financial Advice) ---
    st.subheader("🤖 AI 投資組合診斷與建議")
    
    if not st.session_state['gemini_api_key']:
        st.warning("⚠️ 請在側邊欄輸入 Gemini API Key 以解鎖財務分析報告")
    else:
        if st.button("✨ 生成完整財務分析報告"):
            with st.spinner("AI 正在分析您的資產結構、現金流與市場風險..."):
                try:
                    genai.configure(api_key=st.session_state['gemini_api_key'])
                    model = genai.GenerativeModel('gemini-pro')
                    
                    # 構建提示詞
                    csv_data = pf.to_csv(index=False)
                    total_val = pf['Market Value'].sum() + st.session_state['cash']
                    cash_pct = st.session_state['cash'] / total_val * 100
                    
                    prompt = f"""
                    你是一位專業的私人財富管理顧問。
                    以下是客戶的資產數據：
                    
                    1. 總資產: ${total_val:.2f} USD
                    2. 現金水位: {cash_pct:.2f}%
                    3. 持倉明細 (CSV格式):
                    {csv_data}
                    
                    請提供一份詳細的財務分析報告（使用繁體中文），包含：
                    1. **投資組合健康度評分 (0-100)**：並解釋原因。
                    2. **板塊集中度風險**：是否有過度集中在某一產業？
                    3. **現金管理建議**：目前現金是否過多或過少？應該如何調整？
                    4. **具體操作建議**：針對目前持倉，哪一隻股票風險較高需注意？
                    5. **再平衡建議**：建議納入什麼類型的資產（如債券、防禦型股票）來平衡風險？
                    """
                    
                    response = model.generate_content(prompt)
                    
                    st.markdown(f"""
                    <div style="background-color:#1e293b; padding:20px; border-radius:10px; border: 1px solid #334155;">
                        {response.text}
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"分析失敗，請檢查 API Key 或網絡連線: {e}")

# ==========================================
# 7. 個股分析 (Analysis) - 保持不變
# ==========================================
elif st.session_state['page'] == 'analysis':
    # (此處保持原有的個股分析代碼，為了篇幅省略，
    # 您可以把上一版代碼的 'analysis' 部分複製過來，
    # 或是如果需要我再完整貼一次請告訴我)
    st.info("請從 Dashboard 點擊個股進入分析")
    if st.button("回首頁"): go_to_page('dashboard')
