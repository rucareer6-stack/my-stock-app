import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 設定頁面 ---
st.set_page_config(page_title="美股投資戰情室", layout="wide")

# --- 初始化 Session State (用於暫存數據) ---
if 'portfolio' not in st.session_state:
    st.session_state['portfolio'] = pd.DataFrame(columns=['Ticker', 'Cost', 'Shares'])

if 'cash' not in st.session_state:
    st.session_state['cash'] = 10000.0  # 預設現金

# --- 輔助函數：計算 RSI ---
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- 側邊欄：輸入區 ---
st.sidebar.header("📝 投資組合管理")

# 1. 設定現金
st.sidebar.subheader("1. 現金管理")
cash_input = st.sidebar.number_input("目前持有現金 (USD)", value=st.session_state['cash'], step=100.0)
if cash_input != st.session_state['cash']:
    st.session_state['cash'] = cash_input
    st.rerun()

# 2. 新增持倉
st.sidebar.subheader("2. 新增/更新 持倉")
ticker = st.sidebar.text_input("美股代號 (如 AAPL)", value="").upper()
cost = st.sidebar.number_input("平均成本 (USD)", value=0.0, step=0.1)
shares = st.sidebar.number_input("持有股數", value=0.0, step=1.0)

if st.sidebar.button("加入 / 更新持倉"):
    if ticker and shares > 0:
        new_row = {'Ticker': ticker, 'Cost': cost, 'Shares': shares}
        # 如果已存在則更新，否則新增
        df = st.session_state['portfolio']
        if ticker in df['Ticker'].values:
            df.loc[df['Ticker'] == ticker, ['Cost', 'Shares']] = [cost, shares]
        else:
            st.session_state['portfolio'] = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        st.success(f"已更新 {ticker}")
        st.rerun()

# 刪除持倉功能
if not st.session_state['portfolio'].empty:
    st.sidebar.subheader("管理現有持倉")
    to_delete = st.sidebar.selectbox("選擇要刪除的股票", st.session_state['portfolio']['Ticker'].unique())
    if st.sidebar.button("刪除選定股票"):
        st.session_state['portfolio'] = st.session_state['portfolio'][st.session_state['portfolio']['Ticker'] != to_delete]
        st.rerun()

# --- 主畫面邏輯 ---
st.title("📈 個人美股投資管理分析")

# 如果沒有持倉
if st.session_state['portfolio'].empty and st.session_state['cash'] == 0:
    st.info("👈 請從側邊欄加入您的第一支股票或設定現金！")
else:
    # --- 獲取即時數據 ---
    portfolio = st.session_state['portfolio'].copy()
    tickers = portfolio['Ticker'].tolist()
    
    market_data = {}
    sectors = {}
    
    if tickers:
        # 批量下載數據
        data = yf.download(tickers, period="1d", progress=False)['Close']
        # 獲取個股詳細資訊 (Sector, etc.) - 需要逐個獲取
        for t in tickers:
            try:
                stock_info = yf.Ticker(t).info
                current_price = stock_info.get('currentPrice') or stock_info.get('previousClose')
                market_data[t] = current_price
                sectors[t] = stock_info.get('sector', 'Unknown')
            except:
                market_data[t] = 0
                sectors[t] = 'Unknown'

    # 計算市值與損益
    portfolio['Current Price'] = portfolio['Ticker'].map(market_data)
    portfolio['Market Value'] = portfolio['Current Price'] * portfolio['Shares']
    portfolio['Profit/Loss'] = (portfolio['Current Price'] - portfolio['Cost']) * portfolio['Shares']
    portfolio['Return %'] = ((portfolio['Current Price'] - portfolio['Cost']) / portfolio['Cost']) * 100
    portfolio['Sector'] = portfolio['Ticker'].map(sectors)

    # --- 總體儀表板 ---
    total_stock_value = portfolio['Market Value'].sum()
    total_cash = st.session_state['cash']
    total_assets = total_stock_value + total_cash
    total_pl = portfolio['Profit/Loss'].sum()
    cash_position = (total_cash / total_assets) * 100 if total_assets > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("總資產 (Total Assets)", f"${total_assets:,.2f}")
    col2.metric("總損益 (Total P/L)", f"${total_pl:,.2f}", delta_color="normal")
    col3.metric("股票市值 (Stock Value)", f"${total_stock_value:,.2f}")
    col4.metric("現金水位 (Cash Level)", f"{cash_position:.1f}%")

    # 現金水位進度條
    st.write("現金水位健康度:")
    if cash_position < 10:
        st.progress(cash_position / 100)
        st.warning("⚠️ 現金水位低於 10%，風險承受力較低，建議保留部分現金以便逢低加碼。")
    else:
        st.progress(cash_position / 100)
        st.success("✅ 現金水位健康。")

    # --- 持倉列表 ---
    st.subheader("📋 持倉細節")
    st.dataframe(portfolio.style.format({
        'Cost': '${:.2f}',
        'Shares': '{:.0f}',
        'Current Price': '${:.2f}',
        'Market Value': '${:.2f}',
        'Profit/Loss': '${:.2f}',
        'Return %': '{:.2f}%'
    }))

    # --- 投資組合分析 (圖表) ---
    st.subheader("📊 投資組合分析")
    c1, c2 = st.columns(2)
    
    with c1:
        # 資產分佈 (股票 vs 現金)
        labels = ['Stocks', 'Cash']
        values = [total_stock_value, total_cash]
        fig_alloc = px.pie(names=labels, values=values, title="資產配置 (現金 vs 股票)", hole=0.4)
        st.plotly_chart(fig_alloc, use_container_width=True)

    with c2:
        if not portfolio.empty:
            # 產業分佈
            fig_sector = px.pie(portfolio, values='Market Value', names='Sector', title="產業板塊分佈")
            st.plotly_chart(fig_sector, use_container_width=True)

    # --- 個股深度分析 (技術 + 基本) ---
    st.subheader("🔍 個股深度分析 (技術 & 基本面)")
    
    if not portfolio.empty:
        selected_ticker = st.selectbox("選擇要分析的股票", tickers)
        
        if selected_ticker:
            stock = yf.Ticker(selected_ticker)
            
            # 獲取歷史數據
            hist = stock.history(period="6mo")
            info = stock.info
            
            # 計算指標
            hist['SMA20'] = hist['Close'].rolling(window=20).mean()
            hist['SMA50'] = hist['Close'].rolling(window=50).mean()
            hist['RSI'] = calculate_rsi(hist)
            
            # 1. 基本面數據卡片
            bc1, bc2, bc3, bc4 = st.columns(4)
            bc1.metric("本益比 (P/E)", f"{info.get('trailingPE', 'N/A')}")
            bc2.metric("殖利率 (Yield)", f"{info.get('dividendYield', 0)*100:.2f}%" if info.get('dividendYield') else "N/A")
            bc3.metric("52週高點", f"${info.get('fiftyTwoWeekHigh', 'N/A')}")
            bc4.metric("Beta (波動率)", f"{info.get('beta', 'N/A')}")

            # 2. 技術面圖表 (K線 + 均線)
            fig_tech = go.Figure()
            fig_tech.add_trace(go.Candlestick(x=hist.index,
                            open=hist['Open'], high=hist['High'],
                            low=hist['Low'], close=hist['Close'], name='K線'))
            fig_tech.add_trace(go.Scatter(x=hist.index, y=hist['SMA20'], mode='lines', name='SMA 20 (月線)', line=dict(color='orange')))
            fig_tech.add_trace(go.Scatter(x=hist.index, y=hist['SMA50'], mode='lines', name='SMA 50 (季線)', line=dict(color='blue')))
            
            fig_tech.update_layout(title=f"{selected_ticker} 股價走勢與均線", xaxis_title="日期", yaxis_title="價格")
            st.plotly_chart(fig_tech, use_container_width=True)
            
            # 3. RSI 指標
            current_rsi = hist['RSI'].iloc[-1]
            st.write(f"**目前 RSI (14): {current_rsi:.2f}**")
            if current_rsi > 70:
                st.error("🔴 RSI 高於 70，處於超買區，注意回調風險。")
            elif current_rsi < 30:
                st.success("🟢 RSI 低於 30，處於超賣區，可能反彈。")
            else:
                st.info("⚪ RSI 處於中性區間。")
