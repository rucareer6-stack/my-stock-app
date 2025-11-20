import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import google.generativeai as genai

# ==========================================
# 1. 基礎設定 (只保留最必要的 CSS)
# ==========================================
st.set_page_config(page_title="個人美股投資管理", layout="wide", page_icon="📈")

# 僅調整背景色，不強制修改元件結構，確保穩定性
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
    }
    h1, h2, h3, p, div, span, label {
        color: #e0e0e0 !important;
    }
    /* 讓 Metric 數值更明顯 */
    [data-testid="stMetricValue"] {
        color: #4facfe !important;
    }
    /* 側邊欄微調 */
    [data-testid="stSidebar"] {
        background-color: #161b22;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 核心數據邏輯 (最穩定的 Session State)
# ==========================================
if 'portfolio' not in st.session_state:
    # 預設範例資料
    st.session_state['portfolio'] = pd.DataFrame([
        {'Ticker': 'NVDA', 'Cost': 450.0, 'Shares': 10},
        {'Ticker': 'AAPL', 'Cost': 175.0, 'Shares': 20},
        {'Ticker': 'TSLA', 'Cost': 200.0, 'Shares': 15}
    ])

if 'cash' not in st.session_state:
    st.session_state['cash'] = 10000.0

if 'gemini_api_key' not in st.session_state:
    st.session_state['gemini_api_key'] = ""

# ==========================================
# 3. 側邊欄：最純粹的輸入介面
# ==========================================
with st.sidebar:
    st.header("⚙️ 投資設定")
    
    # API Key
    api_key = st.text_input("Gemini API Key (選填)", value=st.session_state['gemini_api_key'], type="password")
    if api_key: st.session_state['gemini_api_key'] = api_key
    
    st.divider()
    
    # 現金管理
    st.subheader("💰 現金管理")
    new_cash = st.number_input("目前現金餘額 (USD)", value=st.session_state['cash'], step=100.0)
    if new_cash != st.session_state['cash']:
        st.session_state['cash'] = new_cash
        st.rerun()
    
    st.divider()

    # 新增持倉
    st.subheader("➕ 新增/更新持倉")
    with st.form("add_position"):
        ticker_in = st.text_input("股票代號 (例如 NVDA)").upper()
        cost_in = st.number_input("平均成本", min_value=0.0, step=0.1)
        shares_in = st.number_input("持有股數", min_value=0.0, step=1.0)
        
        submitted = st.form_submit_button("確認送出")
        if submitted and ticker_in and shares_in > 0:
            # 邏輯：有就更新，沒有就新增
            df = st.session_state['portfolio']
            new_data = {'Ticker': ticker_in, 'Cost': cost_in, 'Shares': shares_in}
            
            if ticker_in in df['Ticker'].values:
                df.loc[df['Ticker'] == ticker_in, ['Cost', 'Shares']] = [cost_in, shares_in]
                st.success(f"已更新 {ticker_in}")
            else:
                st.session_state['portfolio'] = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
                st.success(f"已新增 {ticker_in}")
            st.rerun()

    # 刪除功能
    if not st.session_state['portfolio'].empty:
        st.divider()
        to_del = st.selectbox("刪除股票", st.session_state['portfolio']['Ticker'].unique())
        if st.button("刪除選定項目"):
            st.session_state['portfolio'] = st.session_state['portfolio'][st.session_state['portfolio']['Ticker'] != to_del]
            st.rerun()

# ==========================================
# 4. 主畫面：直接顯示數據，不搞花俏導航
# ==========================================
st.title("📊 個人美股資產總覽")

# --- 數據計算區 ---
df = st.session_state['portfolio'].copy()
if not df.empty:
    # 批量獲取現價 (最快最穩的方法)
    ticker_list = df['Ticker'].tolist()
    try:
        if len(ticker_list) == 1:
            stock = yf.Ticker(ticker_list[0])
            current_price = stock.history(period='1d')['Close'].iloc[-1]
            price_map = {ticker_list[0]: current_price}
        else:
            data = yf.download(ticker_list, period="1d", progress=False)['Close']
            price_map = data.iloc[-1].to_dict()
    except:
        price_map = {} # 避免報錯
        st.error("無法連接 Yahoo Finance，顯示持倉成本。")

    # 映射價格
    df['Current Price'] = df['Ticker'].map(price_map).fillna(df['Cost']) # 若抓不到就用成本價暫代
    df['Market Value'] = df['Current Price'] * df['Shares']
    df['Profit'] = (df['Current Price'] - df['Cost']) * df['Shares']
    df['Return %'] = (df['Profit'] / (df['Cost'] * df['Shares']) * 100).fillna(0)
    
    total_stock_val = df['Market Value'].sum()
    total_profit = df['Profit'].sum()
else:
    total_stock_val = 0
    total_profit = 0

total_cash = st.session_state['cash']
total_assets = total_stock_val + total_cash
cash_ratio = (total_cash / total_assets * 100) if total_assets > 0 else 0

# --- 儀表板 Metrics ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("總資產 (Total Assets)", f"${total_assets:,.0f}")
col2.metric("總損益 (P/L)", f"${total_profit:,.0f}", delta_color="normal")
col3.metric("股票市值 (Stock Value)", f"${total_stock_val:,.0f}")
col4.metric("現金水位 (Cash)", f"{cash_ratio:.1f}%")

# --- 現金水位條 ---
if cash_ratio < 10:
    st.warning(f"⚠️ 現金水位偏低 ({cash_ratio:.1f}%)")
else:
    st.progress(min(cash_ratio/100, 1.0), text=f"目前現金佔比: {cash_ratio:.1f}%")

st.divider()

# --- 持倉表格 (乾淨、原生、好讀) ---
st.subheader("📋 持倉明細")
if not df.empty:
    # 使用 Streamlit 原生表格設定，最穩定
    st.dataframe(
        df,
        column_config={
            "Ticker": "代號",
            "Cost": st.column_config.NumberColumn("平均成本", format="$%.2f"),
            "Shares": st.column_config.NumberColumn("股數", format="%.0f"),
            "Current Price": st.column_config.NumberColumn("現價", format="$%.2f"),
            "Market Value": st.column_config.NumberColumn("市值", format="$%.0f"),
            "Profit": st.column_config.NumberColumn("損益", format="$%.0f"),
            "Return %": st.column_config.NumberColumn("報酬率", format="%.2f%%"),
        },
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("目前沒有持倉，請從左側新增。")

# --- 個股分析區塊 (直接選，不跳頁) ---
st.divider()
st.subheader("📈 個股快速分析")

if not df.empty:
    selected_ticker = st.selectbox("選擇要查看的股票：", df['Ticker'].unique())
    
    if selected_ticker:
        col_k, col_info = st.columns([2, 1])
        
        # 獲取資料
        stock = yf.Ticker(selected_ticker)
        hist = stock.history(period="6mo")
        info = stock.info
        
        with col_k:
            # 簡單明瞭的 K 線圖
            fig = go.Figure(data=[go.Candlestick(x=hist.index,
                            open=hist['Open'], high=hist['High'],
                            low=hist['Low'], close=hist['Close'], name="K線")])
            fig.update_layout(title=f"{selected_ticker} 近半年走勢", xaxis_rangeslider_visible=False, height=400,
                              template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
        with col_info:
            st.markdown(f"### {selected_ticker}")
            st.write(f"**產業：** {info.get('sector', 'N/A')}")
            st.write(f"**本益比 (P/E)：** {info.get('trailingPE', 'N/A')}")
            st.write(f"**52週高點：** ${info.get('fiftyTwoWeekHigh', 'N/A')}")
            st.write(f"**分析師目標價：** ${info.get('targetMeanPrice', 'N/A')}")
            
            # AI 分析按鈕 (只有按下去才觸發，不自動觸發以免報錯)
            if st.session_state['gemini_api_key']:
                if st.button(f"🤖 AI 分析 {selected_ticker}"):
                    with st.spinner("AI 正在思考..."):
                        try:
                            genai.configure(api_key=st.session_state['gemini_api_key'])
                            model = genai.GenerativeModel('gemini-pro')
                            prompt = f"請用繁體中文簡短分析美股 {selected_ticker} 的基本面與近期風險。"
                            res = model.generate_content(prompt)
                            st.info(res.text)
                        except Exception as e:
                            st.error(f"AI 分析失敗: {e}")

# --- AI 投資建議 (可選) ---
st.divider()
with st.expander("✨ 投資組合 AI 總體建議 (點擊展開)"):
    if st.button("生成投資建議報告"):
        if not st.session_state['gemini_api_key']:
            st.warning("請先在左側輸入 Gemini API Key")
        else:
            with st.spinner("正在分析您的資產配置..."):
                try:
                    genai.configure(api_key=st.session_state['gemini_api_key'])
                    model = genai.GenerativeModel('gemini-pro')
                    
                    pf_csv = df.to_string()
                    prompt = f"""
                    用戶總資產: {total_assets} USD
                    現金水位: {cash_ratio:.1f}%
                    持倉:
                    {pf_csv}
                    請給出 3 點具體的投資調整建議 (繁體中文)。
                    """
                    res = model.generate_content(prompt)
                    st.markdown(res.text)
                except Exception as e:
                    st.error("分析失敗")
