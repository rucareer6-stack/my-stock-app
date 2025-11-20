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
st.set_page_config(page_title="美股資產戰情室 (Pro v7)", layout="wide", page_icon="📊")

st.markdown("""
    <style>
    /* 全局設定 */
    .stApp { background-color: #ffffff; }
    h1, h2, h3, h4, h5, h6 { color: #111827 !important; font-weight: 700 !important; }
    p, div, span, label, li { color: #374151 !important; }
    
    /* 側邊欄 */
    [data-testid="stSidebar"] { background-color: #f9fafb !important; border-right: 1px solid #e5e7eb; }
    
    /* 指標卡片優化 */
    div[data-testid="stMetric"] {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 10px;
    }
    [data-testid="stMetricValue"] { color: #2563eb !important; font-weight: 800 !important; }
    
    /* 按鈕 */
    .stButton > button {
        background-color: #2563eb !important;
        color: white !important;
        border-radius: 6px;
        border: none;
        font-weight: 600;
    }
    .stButton > button:hover { background-color: #1d4ed8 !important; }
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
if 'gemini_api_key' not in st.session_state:
    st.session_state['gemini_api_key'] = ""

# --- RSI 計算函數 (不依賴 TA-Lib，純 Pandas 實作) ---
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --- 獲取個股詳細資訊 (含 P/E, Sector) ---
@st.cache_data(ttl=3600)
def get_stock_details(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            'sector': info.get('sector', '其他'),
            'pe': info.get('trailingPE', None), # 本益比
            'forward_pe': info.get('forwardPE', None),
            'beta': info.get('beta', 0),
            'price': info.get('currentPrice', 0)
        }
    except:
        return {'sector': '未知', 'pe': None, 'forward_pe': None, 'beta': 0, 'price': 0}

# --- AI 呼叫函數 (含自動錯誤修復/模型切換) ---
def call_gemini_safe(api_key, prompt):
    genai.configure(api_key=api_key)
    
    # 定義嘗試的模型順序
    models_to_try = ['gemini-1.5-flash', 'gemini-pro', 'gemini-1.0-pro']
    
    last_error = ""
    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            last_error = str(e)
            continue # 嘗試下一個模型
            
    raise Exception(f"所有模型皆嘗試失敗。最後錯誤: {last_error}")

# --- CAGR 計算 ---
def calculate_cagr(end, start, start_date):
    if start == 0: return 0
    days = (date.today() - start_date).days
    if days <= 0: return 0
    years = days / 365.25
    if years < 1: return (end - start) / start
    try:
        return (end / start) ** (1 / years) - 1
    except:
        return 0

# ==========================================
# 3. 側邊欄
# ==========================================
with st.sidebar:
    st.header("⚙️ 投資設定")
    api_key = st.text_input("Gemini API Key", value=st.session_state['gemini_api_key'], type="password")
    if api_key: st.session_state['gemini_api_key'] = api_key
    
    st.divider()
    st.subheader("💵 現金管理")
    new_cash = st.number_input("現金 (USD)", value=st.session_state['cash'], step=100.0)
    if new_cash != st.session_state['cash']:
        st.session_state['cash'] = new_cash
        st.rerun()
        
    st.divider()
    st.subheader("➕ 持倉操作")
    with st.form("add"):
        t = st.text_input("代碼").upper()
        c = st.number_input("成本", min_value=0.0, step=0.1)
        s = st.number_input("股數", min_value=0.0, step=1.0)
        d = st.date_input("買入日", value=date.today())
        if st.form_submit_button("存入"):
            if t and s > 0:
                df = st.session_state['portfolio']
                if t in df['Ticker'].values:
                    df = df[df['Ticker'] != t]
                new_row = pd.DataFrame([{'Ticker': t, 'Cost': c, 'Shares': s, 'Date': d}])
                st.session_state['portfolio'] = pd.concat([df, new_row], ignore_index=True)
                st.rerun()

    if not st.session_state['portfolio'].empty:
        st.divider()
        del_t = st.selectbox("選擇刪除", st.session_state['portfolio']['Ticker'].unique())
        if st.button("🗑️ 刪除"):
            st.session_state['portfolio'] = st.session_state['portfolio'][st.session_state['portfolio']['Ticker'] != del_t]
            st.rerun()

# ==========================================
# 4. 主畫面數據準備
# ==========================================
st.title("📊 個人美股資產戰情室")

df = st.session_state['portfolio'].copy()
total_hist = pd.DataFrame()

if not df.empty:
    tickers = df['Ticker'].tolist()
    
    # 1. 獲取歷史數據 (用於畫圖與算 RSI)
    try:
        # 多抓一些數據以利計算指標
        hist_data = yf.download(tickers, period="1y", progress=False)['Close']
        
        current_prices = {}
        if isinstance(hist_data, pd.DataFrame) and not hist_data.empty:
            for t in tickers:
                current_prices[t] = hist_data[t].iloc[-1] if t in hist_data.columns else 0
            # 簡易回測
            stock_val_hist = (hist_data * df.set_index('Ticker')['Shares']).sum(axis=1)
            total_hist = stock_val_hist + st.session_state['cash']
        elif isinstance(hist_data, pd.Series):
            current_prices[tickers[0]] = hist_data.iloc[-1]
            total_hist = (hist_data * df.iloc[0]['Shares']) + st.session_state['cash']
    except:
        current_prices = {t:0 for t in tickers}

    # 2. 補充基本面資料 (Sector, PE)
    details_map = {t: get_stock_details(t) for t in tickers}
    
    df['Sector'] = df['Ticker'].map(lambda x: details_map[x]['sector'])
    df['PE'] = df['Ticker'].map(lambda x: details_map[x]['pe'])
    
    # 3. 計算績效
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
# 5. 儀表板 (Assets & Chart)
# ==========================================
m1, m2, m3, m4 = st.columns(4)
m1.metric("總資產", f"${total_assets:,.0f}")
m2.metric("總損益", f"${total_profit:,.0f}", delta_color="normal")
m3.metric("股票市值", f"${total_stock:,.0f}")
m4.metric("現金水位", f"{cash_ratio:.1f}%")

c_chart, c_pie = st.columns([2, 1])
with c_chart:
    if not total_hist.empty:
        st.subheader("📈 資產歷史走勢")
        fig = px.area(x=total_hist.index, y=total_hist.values)
        fig.update_layout(plot_bgcolor='white', margin=dict(l=0,r=0,t=0,b=0), height=250,
                          xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#f3f4f6'))
        fig.update_traces(line_color='#2563eb', fillcolor='rgba(37, 99, 235, 0.1)')
        st.plotly_chart(fig, use_container_width=True)

with c_pie:
    if not df.empty:
        st.subheader("🍰 產業配置")
        fig_p = px.pie(df, values='Market Value', names='Sector', hole=0.4)
        fig_p.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=250)
        st.plotly_chart(fig_p, use_container_width=True)

st.divider()

# ==========================================
# 6. 持倉列表
# ==========================================
st.subheader("📋 持倉詳細績效")
if not df.empty:
    st.dataframe(
        df,
        column_config={
            "Ticker": "代號", "Sector": "產業", "Date": st.column_config.DateColumn("買入日"),
            "Cost": st.column_config.NumberColumn("成本", format="$%.2f"),
            "Current Price": st.column_config.NumberColumn("現價", format="$%.2f"),
            "Shares": st.column_config.NumberColumn("股數", format="%.0f"),
            "Market Value": st.column_config.NumberColumn("市值", format="$%.0f"),
            "Profit": st.column_config.NumberColumn("損益", format="$%.0f"),
            "Return %": st.column_config.NumberColumn("報酬%", format="%.2f%%"),
            "CAGR %": st.column_config.NumberColumn("年化%", format="%.2f%%"),
            "PE": st.column_config.NumberColumn("P/E", format="%.1f"),
        },
        hide_index=True, use_container_width=True
    )

st.divider()

# ==========================================
# 7. 個股深度診斷 (含 RSI, PE, AI)
# ==========================================
st.subheader("🔍 個股深度診斷 (含 RSI & P/E)")

if not df.empty:
    # 1. 選擇股票
    sel_ticker = st.selectbox("選擇分析標的：", df['Ticker'].unique())
    row = df[df['Ticker'] == sel_ticker].iloc[0]
    
    # 2. 獲取數據 (K線 & RSI)
    stock = yf.Ticker(sel_ticker)
    hist = stock.history(period="6mo")
    
    # 計算 RSI
    hist['RSI'] = calculate_rsi(hist['Close'])
    curr_rsi = hist['RSI'].iloc[-1]
    curr_pe = row['PE'] if pd.notnull(row['PE']) else "N/A"
    
    # 3. 指標顯示區 (三欄)
    k1, k2, k3 = st.columns(3)
    k1.metric("現價", f"${row['Current Price']:.2f}")
    
    # P/E 顏色判斷
    pe_val = row['PE'] if pd.notnull(row['PE']) else 0
    pe_color = "normal"
    if pe_val > 0:
        if pe_val > 30: pe_label = "偏高"; pe_color="off" # 紅色概念(需自訂CSS，這裡用off模擬灰或紅)
        elif pe_val < 15: pe_label = "便宜"
        else: pe_label = "合理"
    else: pe_label = "N/A"
    k2.metric("本益比 (P/E)", f"{curr_pe}", delta=pe_label, delta_color="off")
    
    # RSI 顏色判斷
    if curr_rsi > 70: rsi_state = "超買 (過熱)"; rsi_color = "inverse" # inverse 在 light mode 也是紅色
    elif curr_rsi < 30: rsi_state = "超賣 (低檔)"; rsi_color = "normal" # 綠色
    else: rsi_state = "中性"; rsi_color = "off"
    k3.metric("RSI (14)", f"{curr_rsi:.1f}", delta=rsi_state, delta_color=rsi_color)

    # 4. 圖表與 AI 區
    c_kline, c_ai = st.columns([2, 1])
    
    with c_kline:
        st.caption(f"{sel_ticker} K 線圖")
        fig_k = go.Figure(data=[go.Candlestick(x=hist.index,
                        open=hist['Open'], high=hist['High'],
                        low=hist['Low'], close=hist['Close'])])
        fig_k.update_layout(xaxis_rangeslider_visible=False, height=350, 
                            margin=dict(l=10,r=0,t=10,b=10), plot_bgcolor='white')
        st.plotly_chart(fig_k, use_container_width=True)
        
    with c_ai:
        st.caption("🤖 AI 綜合分析")
        if st.button(f"分析 {sel_ticker} (含技術指標)", use_container_width=True):
            if not st.session_state['gemini_api_key']:
                st.error("請輸入 API Key")
            else:
                with st.spinner("AI 正在讀取 RSI, P/E 與 財報..."):
                    try:
                        prompt = f"""
                        請分析美股 {sel_ticker}。
                        數據：
                        - 現價: {row['Current Price']}
                        - 成本: {row['Cost']}
                        - RSI(14): {curr_rsi:.1f} (技術面)
                        - P/E Ratio: {curr_pe} (基本面)
                        - 產業: {row['Sector']}
                        
                        請用繁體中文提供：
                        1. **基本面短評**：P/E 是否合理？
                        2. **技術面短評**：RSI 水位代表什麼？
                        3. **操作建議**：針對我的成本，建議加碼、續抱或減碼？
                        """
                        # 呼叫安全的 Gemini 函數
                        res_text = call_gemini_safe(st.session_state['gemini_api_key'], prompt)
                        
                        st.markdown(f"""
                        <div style="background-color:#f0f9ff; padding:15px; border-radius:10px; border:1px solid #bae6fd; font-size:14px;">
                            {res_text}
                        </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"AI 服務失敗: {e}")
else:
    st.info("暫無持倉")
