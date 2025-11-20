import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
from datetime import datetime, date

# ==========================================
# 1. 視覺設定 (Light Mode)
# ==========================================
st.set_page_config(page_title="美股資產戰情室 (Pro v8)", layout="wide", page_icon="📊")

st.markdown("""
    <style>
    .stApp { background-color: #ffffff; }
    h1, h2, h3, h4, h5, h6 { color: #111827 !important; font-weight: 700 !important; }
    p, div, span, label, li { color: #374151 !important; }
    [data-testid="stSidebar"] { background-color: #f9fafb !important; border-right: 1px solid #e5e7eb; }
    div[data-testid="stMetric"] {
        background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px;
    }
    [data-testid="stMetricValue"] { color: #2563eb !important; font-weight: 800 !important; }
    .stButton > button {
        background-color: #2563eb !important; color: white !important; border-radius: 6px; border: none;
    }
    .stButton > button:hover { background-color: #1d4ed8 !important; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 核心邏輯
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
if 'available_models' not in st.session_state:
    st.session_state['available_models'] = []

# --- RSI 計算 ---
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- 獲取股票資訊 ---
@st.cache_data(ttl=3600)
def get_stock_details(ticker):
    try:
        info = yf.Ticker(ticker).info
        return {
            'sector': info.get('sector', '其他'),
            'pe': info.get('trailingPE', None),
            'price': info.get('currentPrice', 0)
        }
    except:
        return {'sector': '未知', 'pe': None, 'price': 0}

# --- AI 呼叫 (修復版) ---
def call_gemini_v8(api_key, prompt):
    genai.configure(api_key=api_key)
    
    # 直接指定目前最穩定的模型名稱
    # 如果您的 Key 是舊的，可能需要改用 'gemini-pro'
    target_model = 'gemini-1.5-flash' 
    
    try:
        model = genai.GenerativeModel(target_model)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        # 如果 1.5-flash 失敗，嘗試回退到舊版
        try:
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)
            return response.text
        except:
            raise Exception(f"AI 連線失敗: {e}\n(請檢查 API Key 或更新 google-generativeai 套件)")

# ==========================================
# 3. 側邊欄 (含模型診斷)
# ==========================================
with st.sidebar:
    st.header("⚙️ 投資設定")
    api_key = st.text_input("Gemini API Key", value=st.session_state['gemini_api_key'], type="password")
    if api_key: st.session_state['gemini_api_key'] = api_key

    # --- 新增：API 診斷按鈕 ---
    if st.button("🔍 檢查 API 可用模型"):
        if not api_key:
            st.error("請先輸入 API Key")
        else:
            try:
                genai.configure(api_key=api_key)
                models = list(genai.list_models())
                # 過濾出支援生成的模型
                valid_models = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
                st.session_state['available_models'] = valid_models
                st.success(f"連線成功！可用模型：{len(valid_models)} 個")
            except Exception as e:
                st.error(f"連線失敗：{e}")

    if st.session_state['available_models']:
        with st.expander("查看可用模型列表"):
            st.write(st.session_state['available_models'])
            
    st.divider()
    st.subheader("💵 現金管理")
    new_cash = st.number_input("現金 (USD)", value=st.session_state['cash'], step=100.0)
    if new_cash != st.session_state['cash']:
        st.session_state['cash'] = new_cash
        st.rerun()
        
    st.divider()
    st.subheader("➕ 新增/更新持倉")
    with st.form("add"):
        t = st.text_input("代碼").upper()
        c = st.number_input("成本", min_value=0.0, step=0.1)
        s = st.number_input("股數", min_value=0.0, step=1.0)
        d = st.date_input("買入日", value=date.today())
        if st.form_submit_button("存入"):
            if t and s > 0:
                df = st.session_state['portfolio']
                if t in df['Ticker'].values: df = df[df['Ticker'] != t]
                new_row = pd.DataFrame([{'Ticker': t, 'Cost': c, 'Shares': s, 'Date': d}])
                st.session_state['portfolio'] = pd.concat([df, new_row], ignore_index=True)
                st.rerun()

    if not st.session_state['portfolio'].empty:
        st.divider()
        del_t = st.selectbox("刪除股票", st.session_state['portfolio']['Ticker'].unique())
        if st.button("🗑️ 刪除"):
            st.session_state['portfolio'] = st.session_state['portfolio'][st.session_state['portfolio']['Ticker'] != del_t]
            st.rerun()

# ==========================================
# 4. 主畫面
# ==========================================
st.title("📊 個人美股資產戰情室")

df = st.session_state['portfolio'].copy()
total_hist = pd.DataFrame()

if not df.empty:
    tickers = df['Ticker'].tolist()
    try:
        hist_data = yf.download(tickers, period="1y", progress=False)['Close']
        current_prices = {}
        if isinstance(hist_data, pd.DataFrame) and not hist_data.empty:
            for t in tickers:
                current_prices[t] = hist_data[t].iloc[-1] if t in hist_data.columns else 0
            stock_val_hist = (hist_data * df.set_index('Ticker')['Shares']).sum(axis=1)
            total_hist = stock_val_hist + st.session_state['cash']
        elif isinstance(hist_data, pd.Series):
            current_prices[tickers[0]] = hist_data.iloc[-1]
            total_hist = (hist_data * df.iloc[0]['Shares']) + st.session_state['cash']
    except:
        current_prices = {t:0 for t in tickers}

    details_map = {t: get_stock_details(t) for t in tickers}
    df['Sector'] = df['Ticker'].map(lambda x: details_map[x]['sector'])
    df['PE'] = df['Ticker'].map(lambda x: details_map[x]['pe'])
    df['Current Price'] = df['Ticker'].map(current_prices)
    df['Market Value'] = df['Current Price'] * df['Shares']
    df['Profit'] = (df['Current Price'] - df['Cost']) * df['Shares']
    df['Return %'] = df['Profit'] / (df['Cost'] * df['Shares']) * 100
    
    # CAGR
    def calc_cagr(row):
        days = (date.today() - row['Date']).days
        if days < 365: return row['Return %'] / 100 # 未滿一年顯示簡單報酬
        return (row['Current Price']/row['Cost'])**(365/days) - 1
    df['CAGR %'] = df.apply(calc_cagr, axis=1) * 100

    total_stock = df['Market Value'].sum()
    total_profit = df['Profit'].sum()
else:
    total_stock = 0
    total_profit = 0

total_assets = total_stock + st.session_state['cash']
cash_ratio = (st.session_state['cash'] / total_assets * 100) if total_assets > 0 else 0

# --- 儀表板 ---
m1, m2, m3, m4 = st.columns(4)
m1.metric("總資產", f"${total_assets:,.0f}")
m2.metric("總損益", f"${total_profit:,.0f}", delta_color="normal")
m3.metric("股票市值", f"${total_stock:,.0f}")
m4.metric("現金水位", f"{cash_ratio:.1f}%")

c_chart, c_pie = st.columns([2, 1])
with c_chart:
    if not total_hist.empty:
        st.subheader("📈 資產走勢")
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

# --- 持倉表 ---
st.subheader("📋 持倉績效 (含年化報酬)")
if not df.empty:
    st.dataframe(df, column_config={
        "Ticker": "代號", "Sector": "產業", "Date": st.column_config.DateColumn("買入日"),
        "Cost": st.column_config.NumberColumn("成本", format="$%.2f"),
        "Current Price": st.column_config.NumberColumn("現價", format="$%.2f"),
        "Shares": st.column_config.NumberColumn("股數", format="%.0f"),
        "Market Value": st.column_config.NumberColumn("市值", format="$%.0f"),
        "Profit": st.column_config.NumberColumn("損益", format="$%.0f"),
        "Return %": st.column_config.NumberColumn("報酬%", format="%.2f%%"),
        "CAGR %": st.column_config.NumberColumn("年化%", format="%.2f%%"),
        "PE": st.column_config.NumberColumn("P/E", format="%.1f"),
    }, hide_index=True, use_container_width=True)

st.divider()

# --- 個股診斷 ---
st.subheader("🔍 個股深度診斷 (RSI + P/E)")
if not df.empty:
    sel_ticker = st.selectbox("選擇分析標的：", df['Ticker'].unique())
    row = df[df['Ticker'] == sel_ticker].iloc[0]
    
    stock = yf.Ticker(sel_ticker)
    hist = stock.history(period="6mo")
    hist['RSI'] = calculate_rsi(hist['Close'])
    curr_rsi = hist['RSI'].iloc[-1]
    curr_pe = row['PE'] if pd.notnull(row['PE']) else "N/A"
    
    k1, k2, k3 = st.columns(3)
    k1.metric("現價", f"${row['Current Price']:.2f}")
    k2.metric("本益比 (P/E)", f"{curr_pe}")
    
    rsi_color = "inverse" if curr_rsi > 70 else ("normal" if curr_rsi < 30 else "off")
    rsi_state = "超買" if curr_rsi > 70 else ("超賣" if curr_rsi < 30 else "中性")
    k3.metric("RSI (14)", f"{curr_rsi:.1f}", delta=rsi_state, delta_color=rsi_color)

    c_k, c_ai = st.columns([2, 1])
    with c_k:
        fig_k = go.Figure(data=[go.Candlestick(x=hist.index,
                        open=hist['Open'], high=hist['High'],
                        low=hist['Low'], close=hist['Close'])])
        fig_k.update_layout(xaxis_rangeslider_visible=False, height=350, margin=dict(l=10,r=0,t=10,b=10))
        st.plotly_chart(fig_k, use_container_width=True)
        
    with c_ai:
        if st.button(f"✨ AI 分析 {sel_ticker}"):
            if not st.session_state['gemini_api_key']:
                st.error("請輸入 API Key")
            else:
                with st.spinner("AI 分析中..."):
                    try:
                        prompt = f"""
                        請分析美股 {sel_ticker}。
                        數據：現價 {row['Current Price']}, RSI {curr_rsi:.1f}, P/E {curr_pe}。
                        請提供：
                        1. 技術面 RSI 解讀
                        2. 基本面 P/E 評價
                        3. 操作建議
                        """
                        res = call_gemini_v8(st.session_state['gemini_api_key'], prompt)
                        st.info(res)
                    except Exception as e:
                        st.error(e)
else:
    st.info("暫無持倉")
