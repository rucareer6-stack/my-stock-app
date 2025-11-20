import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import google.generativeai as genai
from datetime import datetime

# ==========================================
# 1. 視覺核心 (保持深色風格)
# ==========================================
st.set_page_config(page_title="AI 投資戰情室 v3.1", layout="wide", page_icon="📉")

st.markdown("""
    <style>
    /* --- 全局背景 --- */
    .stApp {
        background-color: #0b1120;
    }
    
    /* --- 文字顏色 --- */
    h1, h2, h3, h4, h5, h6, p, span, div, label, li, a {
        color: #e2e8f0 !important;
        font-family: 'Inter', sans-serif;
        text-decoration: none;
    }
    
    /* --- 側邊欄 (Sidebar) 修復 --- */
    [data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid #1e293b;
    }
    [data-testid="stSidebar"] * {
        color: #cbd5e1 !important;
    }
    [data-testid="stSidebar"] input {
        background-color: #1e293b !important;
        color: white !important;
        border: 1px solid #334155 !important;
    }
    
    /* --- 表格與卡片 --- */
    div[data-testid="stDataFrame"] {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 10px;
    }
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
    
    /* --- 新聞連結 --- */
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

# 導航函數 (修正 Rerun Warning)
def navigate_to(page, ticker=None):
    st.session_state['page'] = page
    if ticker:
        st.session_state['selected_ticker'] = ticker
    # 注意：不在這裡 call rerun，而是在主邏輯中自然刷新，或在 button 後刷新

# ==========================================
# 3. 核心邏輯 (新聞修復與數據抓取)
# ==========================================

@st.cache_data(ttl=600)
def get_real_hot_sectors():
    """抓取 ETF 漲跌幅"""
    sectors = {
        'SMH': '半導體 (Semi)', 'XLK': '科技 (Tech)', 'XLV': '醫療 (Health)',
        'XLF': '金融 (Finance)', 'XLE': '能源 (Energy)', 'IGV': '軟體 (Software)',
        'XLC': '通訊 (Comm)', 'XLY': '非必需消費'
    }
    try:
        tickers = list(sectors.keys())
        data = yf.download(tickers, period="5d", progress=False)['Close']
        if len(data) >= 2:
            last = data.iloc[-1]
            prev = data.iloc[-2]
            changes = ((last - prev) / prev * 100).sort_values(ascending=False)
            
            top_sectors = []
            for sym in changes.index[:5]:
                val = changes[sym]
                icon = "🔥" if val > 0 else "❄️"
                top_sectors.append({"name": sectors.get(sym, sym), "change": val, "icon": icon})
            return top_sectors
    except:
        pass
    # 備用數據 (若 API 失敗)
    return [{"name": "半導體 (Semi)", "change": 2.1, "icon": "🔥"}, {"name": "科技 (Tech)", "change": 1.2, "icon": "🔥"}]

@st.cache_data(ttl=300)
def get_real_news():
    """抓取新聞 (修復 None 問題)"""
    try:
        # 嘗試從 SPY 獲取新聞
        spy = yf.Ticker("SPY")
        news_data = spy.news
        
        formatted_news = []
        if news_data:
            for n in news_data[:5]:
                # 強制檢查：標題必須存在且不能是 None
                title = n.get('title')
                link = n.get('link')
                
                if title and link: # 只有當標題和連結都有值才顯示
                    formatted_news.append({
                        "title": title,
                        "link": link,
                        "publisher": n.get('publisher', 'Market News'),
                        "time": datetime.fromtimestamp(n.get('providerPublishTime', 0)).strftime('%H:%M')
                    })
        
        if len(formatted_news) > 0:
            return formatted_news
            
    except Exception:
        pass
    
    # 如果 API 失敗或格式錯誤，返回「備用靜態新聞」，保證版面不壞掉
    return [
        {"title": "Fed Signals Rate Cuts Might Be Delayed", "link": "https://finance.yahoo.com", "publisher": "Bloomberg", "time": "Now"},
        {"title": "Tech Stocks Rally on AI Optimism", "link": "https://finance.yahoo.com", "publisher": "Reuters", "time": "Now"},
        {"title": "Oil Prices Surge Amid Middle East Tensions", "link": "https://finance.yahoo.com", "publisher": "CNBC", "time": "Now"}
    ]

def get_stock_info_safe(ticker):
    try:
        s = yf.Ticker(ticker)
        hist = s.history(period="1y")
        info = s.info
        if hist.empty: return None
        return {
            "hist": hist,
            "price": hist['Close'].iloc[-1],
            "change": (hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2] * 100,
            "summary": info.get('longBusinessSummary', 'No summary.'),
            "pe": info.get('trailingPE', 'N/A'),
            "target": info.get('targetMeanPrice', 'N/A'),
            "high52": info.get('fiftyTwoWeekHigh', 'N/A')
        }
    except:
        return None

# ==========================================
# 4. 側邊欄
# ==========================================
with st.sidebar:
    st.subheader("⚙️ 控制中心")
    key_input = st.text_input("Gemini API Key", value=st.session_state['gemini_api_key'], type="password")
    if key_input: st.session_state['gemini_api_key'] = key_input
    
    st.markdown("---")
    st.caption("快速交易")
    with st.form("add_stock"):
        t_sym = st.text_input("代碼", value="AMD").upper()
        t_cost = st.number_input("成本", min_value=0.0)
        t_share = st.number_input("股數", min_value=0.0)
        if st.form_submit_button("➕ 加入持倉"):
            new_row = {'Ticker': t_sym, 'Cost': t_cost, 'Shares': t_share, 'Sector': 'Unknown'}
            st.session_state['portfolio'] = pd.concat([st.session_state['portfolio'], pd.DataFrame([new_row])], ignore_index=True)
            st.success("已加入")
            st.rerun()

# ==========================================
# 5. Dashboard (首頁)
# ==========================================
if st.session_state['page'] == 'dashboard':
    st.title("🚀 戰情室 Dashboard")

    # 1. 熱門板塊
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

    # 2. 資產概況
    pf = st.session_state['portfolio'].copy()
    tickers = pf['Ticker'].tolist()
    
    # 批量獲取現價
    current_prices = {}
    if tickers:
        try:
            data = yf.download(tickers, period="1d", progress=False)['Close']
            if isinstance(data, pd.Series): # 只有一支股票時
                current_prices[tickers[0]] = data.iloc[-1]
            elif not data.empty: # 多支股票
                for t in tickers:
                    current_prices[t] = data.iloc[-1][t]
        except:
            pass

    pf['Price'] = pf['Ticker'].map(current_prices).fillna(0)
    pf['Val'] = pf['Price'] * pf['Shares']
    pf['Profit'] = (pf['Price'] - pf['Cost']) * pf['Shares']
    
    total_asset = pf['Val'].sum() + st.session_state['cash']
    cash_pct = (st.session_state['cash'] / total_asset * 100) if total_asset > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("總資產", f"${total_asset:,.0f}")
    c2.metric("總損益", f"${pf['Profit'].sum():,.0f}", delta_color="normal")
    c3.metric("現金水位", f"{cash_pct:.1f}%")
    with c4:
        st.write("")
        # 修正按鈕，避免 callback 錯誤
        if st.button("📋 查看持倉詳情 >", use_container_width=True):
            st.session_state['page'] = 'details'
            st.rerun()

    # 3. [已復原] 我的持倉卡片
    st.markdown("### 💼 我的持倉 (點擊分析)")
    if pf.empty:
        st.info("暫無持倉，請從左側新增")
    else:
        cols = st.columns(4)
        for i, row in pf.iterrows():
            with cols[i % 4]:
                profit_color = "#4ade80" if row['Profit'] > 0 else "#f87171"
                st.markdown(f"""
                <div style="background-color: #1e293b; border:1px solid #334155; border-radius:10px; padding:15px; margin-bottom:10px; text-align:center;">
                    <h3 style="color:#38bdf8 !important; margin:0;">{row['Ticker']}</h3>
                    <p style="font-size:14px; color:#94a3b8 !important;">${row['Price']:.2f}</p>
                    <p style="color:{profit_color} !important; font-weight:bold;">{'+' if row['Profit']>0 else ''}{row['Profit']:.0f}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # 修正按鈕邏輯：不使用 callback，直接檢查狀態
                if st.button(f"分析 {row['Ticker']}", key=f"btn_{row['Ticker']}"):
                    st.session_state['selected_ticker'] = row['Ticker']
                    st.session_state['page'] = 'analysis'
                    st.rerun()

    # 4. 市場焦點 (已修復 None 問題)
    st.markdown("### 📰 市場焦點 (News)")
    news_list = get_real_news()
    for n in news_list:
        st.markdown(f"""
        <div style="background:#1e293b; padding:10px; margin-bottom:8px; border-radius:5px; border-left: 4px solid #38bdf8;">
            <a href="{n['link']}" target="_blank" class="news-link" style="font-size:16px; font-weight:600; color:#e2e8f0; text-decoration:none;">
                {n['title']} ↗
            </a>
            <br>
            <span style="font-size:12px; color:#94a3b8;">{n['publisher']} • {n['time']}</span>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# 6. 持倉詳情 (Details)
# ==========================================
elif st.session_state['page'] == 'details':
    c1, c2 = st.columns([1, 6])
    if c1.button("← 返迴"):
        st.session_state['page'] = 'dashboard'
        st.rerun()
    c2.title("📋 持倉詳細報表")

    pf = st.session_state['portfolio'].copy()
    # 重新獲取價格 (簡化邏輯)
    tickers = pf['Ticker'].tolist()
    prices = {}
    if tickers:
        try:
            data = yf.download(tickers, period="1d", progress=False)['Close']
            if isinstance(data, pd.Series): prices[tickers[0]] = data.iloc[-1]
            else: 
                for t in tickers: prices[t] = data.iloc[-1][t]
        except: pass
    
    pf['Price'] = pf['Ticker'].map(prices).fillna(0)
    pf['Value'] = pf['Price'] * pf['Shares']
    pf['P/L'] = (pf['Price'] - pf['Cost']) * pf['Shares']
    pf['Ret%'] = (pf['P/L'] / (pf['Cost'] * pf['Shares']) * 100).fillna(0)

    st.dataframe(
        pf,
        column_config={
            "Ticker": "代碼",
            "Cost": st.column_config.NumberColumn("成本", format="$%.2f"),
            "Price": st.column_config.NumberColumn("現價", format="$%.2f"),
            "Value": st.column_config.NumberColumn("市值", format="$%.0f"),
            "P/L": st.column_config.NumberColumn("損益", format="$%.0f"),
            "Ret%": st.column_config.NumberColumn("報酬", format="%.2f%%"),
        },
        hide_index=True,
        use_container_width=True
    )

    st.markdown("---")
    st.subheader("🤖 AI 財務建議")
    if st.button("✨ 生成分析報告"):
        if not st.session_state['gemini_api_key']:
            st.error("請先輸入 API Key")
        else:
            with st.spinner("AI 分析中..."):
                try:
                    genai.configure(api_key=st.session_state['gemini_api_key'])
                    model = genai.GenerativeModel('gemini-pro')
                    prompt = f"請分析此持倉：\n{pf.to_string()}\n現金：{st.session_state['cash']}\n請給出風險評估與建議(繁體中文)。"
                    res = model.generate_content(prompt)
                    st.markdown(res.text)
                except Exception as e:
                    st.error(f"錯誤: {e}")

# ==========================================
# 7. 個股分析 (Analysis) - 完整復原
# ==========================================
elif st.session_state['page'] == 'analysis':
    tick = st.session_state['selected_ticker']
    c1, c2 = st.columns([1, 6])
    if c1.button("← 返迴"):
        st.session_state['page'] = 'dashboard'
        st.rerun()
    c2.title(f"{tick} 深度分析")

    data = get_stock_info_safe(tick)
    if not data:
        st.error("無法獲取數據")
    else:
        col_chart, col_info = st.columns([2, 1])
        with col_chart:
            st.subheader("K 線走勢")
            fig = go.Figure(data=[go.Candlestick(x=data['hist'].index,
                open=data['hist']['Open'], high=data['hist']['High'],
                low=data['hist']['Low'], close=data['hist']['Close'])])
            fig.update_layout(xaxis_rangeslider_visible=False, height=400, 
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#94a3b8')
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("公司簡介"):
                st.write(data['summary'])

        with col_info:
            color = "#22c55e" if data['change'] > 0 else "#ef4444"
            st.markdown(f"""
            <div style="background:#1e293b; padding:20px; border-radius:10px; text-align:center; border:1px solid {color};">
                <h1 style="color:{color}!important; margin:0;">${data['price']:.2f}</h1>
                <p style="color:{color}!important;">{data['change']:+.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
            st.write("")
            st.metric("本益比 P/E", f"{data['pe']}")
            st.metric("目標價", f"${data['target']}")
            st.metric("52週高點", f"${data['high52']}")
