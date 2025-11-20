import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
from datetime import datetime, date

# ==========================================
# 1. 視覺設定：純白高對比 (Light Mode)
# ==========================================
st.set_page_config(page_title="美股資產戰情室 (Pro)", layout="wide", page_icon="📊")

st.markdown("""
    <style>
    /* 全局白底 */
    .stApp { background-color: #ffffff; }
    
    /* 文字深灰黑 */
    h1, h2, h3, h4, h5, h6 { color: #111827 !important; font-weight: 700 !important; }
    p, div, span, label, li { color: #374151 !important; }
    
    /* 側邊欄淺灰 */
    [data-testid="stSidebar"] { background-color: #f9fafb !important; border-right: 1px solid #e5e7eb; }
    
    /* 卡片與區塊陰影 */
    div.css-card {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* Metric 數值優化 */
    [data-testid="stMetricValue"] { color: #2563eb !important; font-weight: 800 !important; }
    
    /* 按鈕優化 */
    .stButton > button {
        background-color: #2563eb !important;
        color: white !important;
        border-radius: 8px;
        border: none;
    }
    .stButton > button:hover { background-color: #1d4ed8 !important; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 核心邏輯與 API 修復
# ==========================================

if 'portfolio' not in st.session_state:
    st.session_state['portfolio'] = pd.DataFrame([
        {'Ticker': 'NVDA', 'Cost': 450.0, 'Shares': 10, 'Date': date(2023, 1, 15)},
        {'Ticker': 'AAPL', 'Cost': 170.0, 'Shares': 20, 'Date': date(2023, 6, 1)},
        {'Ticker': 'TSLA', 'Cost': 200.0, 'Shares': 15, 'Date': date(2022, 11, 20)}
    ])
if 'cash' not in st.session_state:
    st.session_state['cash'] = 10000.0
if 'gemini_api_key' not in st.session_state:
    st.session_state['gemini_api_key'] = ""

# --- 快取函數：獲取產業與公司資訊 (避免卡頓) ---
@st.cache_data(ttl=86400) # 緩存 24 小時
def get_stock_meta(ticker):
    try:
        info = yf.Ticker(ticker).info
        return {
            'sector': info.get('sector', '其他'),
            'industry': info.get('industry', 'N/A'),
            'beta': info.get('beta', 0)
        }
    except:
        return {'sector': '未知', 'industry': 'N/A', 'beta': 0}

# --- 計算年化報酬 (CAGR) ---
def calculate_cagr(end, start, start_date):
    if start == 0: return 0
    days = (date.today() - start_date).days
    if days <= 0: return 0
    years = days / 365.25
    if years < 1: return (end - start) / start # 未滿一年顯示簡單報酬
    try:
        return (end / start) ** (1 / years) - 1
    except:
        return 0

# ==========================================
# 3. 側邊欄設定
# ==========================================
with st.sidebar:
    st.header("⚙️ 投資設定")
    api_key = st.text_input("Gemini API Key", value=st.session_state['gemini_api_key'], type="password")
    if api_key: st.session_state['gemini_api_key'] = api_key
    
    st.divider()
    
    st.subheader("💰 現金管理")
    new_cash = st.number_input("現金餘額 (USD)", value=st.session_state['cash'], step=100.0)
    if new_cash != st.session_state['cash']:
        st.session_state['cash'] = new_cash
        st.rerun()
        
    st.divider()
    st.subheader("➕ 新增持倉")
    with st.form("add"):
        t = st.text_input("代碼").upper()
        c = st.number_input("成本", min_value=0.0, step=0.1)
        s = st.number_input("股數", min_value=0.0, step=1.0)
        d = st.date_input("買入日", value=date.today())
        if st.form_submit_button("存入"):
            if t and s > 0:
                df = st.session_state['portfolio']
                # 如果已有該股，刪除舊的 (簡單覆蓋邏輯)
                if t in df['Ticker'].values:
                    df = df[df['Ticker'] != t]
                
                new_row = pd.DataFrame([{'Ticker': t, 'Cost': c, 'Shares': s, 'Date': d}])
                st.session_state['portfolio'] = pd.concat([df, new_row], ignore_index=True)
                st.rerun()

    if not st.session_state['portfolio'].empty:
        st.divider()
        if st.button("🗑️ 刪除選定股票"):
             # 這裡可做更細緻的刪除，先做清空示範
             pass 

# ==========================================
# 4. 主畫面數據處理
# ==========================================
st.title("📊 個人美股資產戰情室")

df = st.session_state['portfolio'].copy()
total_history = pd.DataFrame()

if not df.empty:
    tickers = df['Ticker'].tolist()
    
    # 1. 批量獲取現價與歷史 (畫圖用)
    try:
        data = yf.download(tickers, period="1y", progress=False)['Close']
        # 處理現價
        current_prices = {}
        if isinstance(data, pd.DataFrame) and not data.empty:
            last_row = data.iloc[-1]
            for t in tickers:
                current_prices[t] = last_row.get(t, 0)
            # 簡易回測數據
            stock_hist = (data * df.set_index('Ticker')['Shares']).sum(axis=1)
            total_history = stock_hist + st.session_state['cash']
        elif isinstance(data, pd.Series):
            current_prices[tickers[0]] = data.iloc[-1]
            total_history = (data * df.iloc[0]['Shares']) + st.session_state['cash']
    except:
        current_prices = {t:0 for t in tickers}

    # 2. 獲取產業資訊 (Meta Data)
    meta_data = [get_stock_meta(t) for t in tickers]
    df['Sector'] = [m['sector'] for m in meta_data]
    df['Industry'] = [m['industry'] for m in meta_data]

    # 3. 計算財務指標
    df['Current Price'] = df['Ticker'].map(current_prices)
    df['Market Value'] = df['Current Price'] * df['Shares']
    df['Profit'] = (df['Current Price'] - df['Cost']) * df['Shares']
    df['Return %'] = df['Profit'] / (df['Cost'] * df['Shares']) * 100
    df['CAGR %'] = df.apply(lambda x: calculate_cagr(x['Current Price'], x['Cost'], x['Date']), axis=1) * 100

    total_stock = df['Market Value'].sum()
    total_profit = df['Profit'].sum()
else:
    total_stock = 0
    total_profit = 0

total_assets = total_stock + st.session_state['cash']
cash_ratio = (st.session_state['cash'] / total_assets * 100) if total_assets > 0 else 0

# ==========================================
# 5. 上半部：資產圖表區 (資產走勢 + 圓餅圖)
# ==========================================
m1, m2, m3, m4 = st.columns(4)
m1.metric("總資產", f"${total_assets:,.0f}")
m2.metric("總損益", f"${total_profit:,.0f}", delta_color="normal")
m3.metric("股票市值", f"${total_stock:,.0f}")
m4.metric("現金水位", f"{cash_ratio:.1f}%")

col_chart, col_pie = st.columns([2, 1])

with col_chart:
    if not total_history.empty:
        st.subheader("📈 總資產走勢")
        fig_area = px.area(x=total_history.index, y=total_history.values)
        fig_area.update_layout(
            plot_bgcolor='white', paper_bgcolor='white', margin=dict(l=0,r=0,t=0,b=0), height=250,
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#f3f4f6')
        )
        fig_area.update_traces(line_color='#2563eb', fillcolor='rgba(37, 99, 235, 0.1)')
        st.plotly_chart(fig_area, use_container_width=True)

with col_pie:
    if not df.empty:
        st.subheader("🍰 資產/產業分佈")
        # 依產業分類
        fig_pie = px.pie(df, values='Market Value', names='Sector', hole=0.4, 
                         color_discrete_sequence=px.colors.qualitative.Set2)
        fig_pie.update_layout(margin=dict(l=0,r=0,t=30,b=0), height=250)
        st.plotly_chart(fig_pie, use_container_width=True)

st.divider()

# ==========================================
# 6. 中間：持倉詳細列表
# ==========================================
st.subheader("📋 持倉績效表")
if not df.empty:
    st.dataframe(
        df[['Ticker', 'Sector', 'Date', 'Cost', 'Current Price', 'Shares', 'Market Value', 'Profit', 'Return %', 'CAGR %']],
        column_config={
            "Ticker": "代號",
            "Sector": "產業",
            "Date": st.column_config.DateColumn("買入日"),
            "Cost": st.column_config.NumberColumn("成本", format="$%.2f"),
            "Current Price": st.column_config.NumberColumn("現價", format="$%.2f"),
            "Shares": st.column_config.NumberColumn("股數", format="%.0f"),
            "Market Value": st.column_config.NumberColumn("市值", format="$%.0f"),
            "Profit": st.column_config.NumberColumn("損益", format="$%.0f"),
            "Return %": st.column_config.NumberColumn("報酬率", format="%.2f%%"),
            "CAGR %": st.column_config.NumberColumn("年化(CAGR)", format="%.2f%%"),
        },
        use_container_width=True,
        hide_index=True
    )

# ==========================================
# 7. 下半部：個股深度分析 (K線 + AI)
# ==========================================
st.markdown("---")
st.subheader("🔍 個股深度診斷")

if not df.empty:
    # 選擇股票
    selected_t = st.selectbox("選擇要分析的持股：", df['Ticker'].unique())
    
    # 抓取該股資料
    row = df[df['Ticker'] == selected_t].iloc[0]
    
    # 佈局：左邊 AI 文字，右邊 K 線圖
    c_ai, c_k = st.columns([1, 2])
    
    with c_k:
        st.markdown(f"#### {selected_t} 近半年走勢")
        try:
            stock_k = yf.Ticker(selected_t)
            hist_k = stock_k.history(period="6mo")
            
            fig_k = go.Figure(data=[go.Candlestick(x=hist_k.index,
                            open=hist_k['Open'], high=hist_k['High'],
                            low=hist_k['Low'], close=hist_k['Close'])])
            fig_k.update_layout(xaxis_rangeslider_visible=False, height=350,
                                margin=dict(l=20, r=0, t=20, b=20),
                                plot_bgcolor='white', paper_bgcolor='white')
            st.plotly_chart(fig_k, use_container_width=True)
        except:
            st.error("無法載入 K 線圖")

    with c_ai:
        st.markdown(f"#### 🤖 AI 分析報告")
        st.markdown(f"**產業**：{row['Sector']} | **現價**：${row['Current Price']:.2f}")
        
        if st.button(f"✨ 分析 {selected_t} (Gemini 1.5)"):
            if not st.session_state['gemini_api_key']:
                st.warning("請輸入 API Key")
            else:
                with st.spinner("AI 正在讀取財報與走勢..."):
                    try:
                        # 重要：這裡換成了 gemini-1.5-flash，解決 404 錯誤
                        genai.configure(api_key=st.session_state['gemini_api_key'])
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        
                        prompt = f"""
                        請分析美股 {selected_t} (產業: {row['Sector']})。
                        我的成本: {row['Cost']}, 現價: {row['Current Price']}, 帳面報酬: {row['Return %']:.2f}%。
                        請用繁體中文提供：
                        1. 該公司近期的基本面強弱。
                        2. 技術面簡單評點。
                        3. 針對我的成本位，建議的操作策略（續抱/減碼/加碼）。
                        """
                        res = model.generate_content(prompt)
                        st.success("分析完成")
                        st.markdown(f"""
                        <div style="background-color:#f3f4f6; padding:15px; border-radius:10px; height:300px; overflow-y:auto;">
                            {res.text}
                        </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"API 錯誤 (請檢查 Key 是否正確): {e}")
else:
    st.info("暫無持倉可分析")
