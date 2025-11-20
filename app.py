import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
from datetime import datetime, date
import numpy as np

# ==========================================
# 1. 視覺設定：純白簡約風格 (Light Mode)
# ==========================================
st.set_page_config(page_title="個人美股資產管理 (Light)", layout="wide", page_icon="📊")

st.markdown("""
    <style>
    /* --- 全局背景：純白 --- */
    .stApp {
        background-color: #ffffff;
    }
    
    /* --- 文字顏色：深灰/黑 (高對比) --- */
    h1, h2, h3, h4, h5, h6 {
        color: #1f2937 !important; /* 深灰 */
        font-weight: 700 !important;
    }
    p, div, span, label, li {
        color: #374151 !important; /* 次深灰 */
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    
    /* --- 側邊欄：淺灰底 --- */
    [data-testid="stSidebar"] {
        background-color: #f3f4f6 !important;
        border-right: 1px solid #e5e7eb;
    }
    
    /* --- Metric 指標卡片 --- */
    div[data-testid="stMetric"] {
        background-color: #f9fafb; /* 非常淺的灰 */
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    [data-testid="stMetricLabel"] {
        color: #6b7280 !important; /* 標籤淺灰 */
    }
    [data-testid="stMetricValue"] {
        color: #111827 !important; /* 數值純黑 */
        font-weight: 800 !important;
    }
    
    /* --- 表格優化 (白底黑字) --- */
    div[data-testid="stDataFrame"] {
        border: 1px solid #e5e7eb;
    }
    
    /* --- 按鈕風格 (藍色強調) --- */
    .stButton > button {
        background-color: #2563eb !important; /* 亮藍 */
        color: white !important;
        border: none !important;
        border-radius: 6px;
    }
    .stButton > button:hover {
        background-color: #1d4ed8 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 核心邏輯與計算
# ==========================================

# 初始化 Session State
if 'portfolio' not in st.session_state:
    # 預設資料 (包含買入日期，用於計算年化)
    st.session_state['portfolio'] = pd.DataFrame([
        {'Ticker': 'NVDA', 'Cost': 450.0, 'Shares': 10, 'Date': date(2023, 1, 15)},
        {'Ticker': 'AAPL', 'Cost': 170.0, 'Shares': 20, 'Date': date(2023, 6, 1)},
        {'Ticker': 'TSLA', 'Cost': 200.0, 'Shares': 15, 'Date': date(2022, 11, 20)}
    ])

if 'cash' not in st.session_state:
    st.session_state['cash'] = 10000.0

if 'gemini_api_key' not in st.session_state:
    st.session_state['gemini_api_key'] = ""

# 年化報酬率計算函數 (CAGR)
def calculate_cagr(end_price, start_price, start_date):
    if start_price == 0: return 0
    days_held = (date.today() - start_date).days
    if days_held <= 0: return 0
    years = days_held / 365.25
    
    # 如果持有不到一年，直接顯示簡單報酬率，避免年化數值過於誇張
    if years < 1:
        return (end_price - start_price) / start_price
    
    try:
        cagr = (end_price / start_price) ** (1 / years) - 1
        return cagr
    except:
        return 0

# ==========================================
# 3. 側邊欄：輸入區
# ==========================================
with st.sidebar:
    st.header("⚙️ 設定與交易")
    
    # API Key
    api_key = st.text_input("Gemini API Key (選填)", value=st.session_state['gemini_api_key'], type="password")
    if api_key: st.session_state['gemini_api_key'] = api_key
    
    st.markdown("---")
    st.subheader("💵 現金管理")
    new_cash = st.number_input("現金餘額 (USD)", value=st.session_state['cash'], step=100.0)
    if new_cash != st.session_state['cash']:
        st.session_state['cash'] = new_cash
        st.rerun()
        
    st.markdown("---")
    st.subheader("➕ 新增/更新持倉")
    
    with st.form("add_pos"):
        t_in = st.text_input("股票代號").upper()
        c_in = st.number_input("平均成本", min_value=0.0, step=0.1)
        s_in = st.number_input("持有股數", min_value=0.0, step=1.0)
        d_in = st.date_input("買入日期 (用於算年化)", value=date.today())
        
        if st.form_submit_button("確認送出"):
            if t_in and s_in > 0:
                df = st.session_state['portfolio']
                new_row = {'Ticker': t_in, 'Cost': c_in, 'Shares': s_in, 'Date': d_in}
                
                # 如果已存在，更新資料 (包含日期)
                if t_in in df['Ticker'].values:
                    # 更新該行的所有欄位
                    df.loc[df['Ticker'] == t_in, ['Cost', 'Shares', 'Date']] = [c_in, s_in, d_in]
                    st.success(f"已更新 {t_in}")
                else:
                    st.session_state['portfolio'] = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                    st.success(f"已新增 {t_in}")
                st.rerun()

    # 刪除區塊
    if not st.session_state['portfolio'].empty:
        st.markdown("---")
        to_del = st.selectbox("選擇刪除", st.session_state['portfolio']['Ticker'].unique())
        if st.button("🗑️ 刪除"):
            st.session_state['portfolio'] = st.session_state['portfolio'][st.session_state['portfolio']['Ticker'] != to_del]
            st.rerun()

# ==========================================
# 4. 主畫面：白底高對比
# ==========================================
st.title("📈 個人美股資產總覽")
st.caption(f"最後更新: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# --- 數據處理與計算 ---
df = st.session_state['portfolio'].copy()
total_assets_history = pd.DataFrame() # 用於畫圖

if not df.empty:
    tickers = df['Ticker'].tolist()
    
    # 1. 獲取現價
    try:
        # 下載過去一年的數據，用於畫資產走勢圖
        hist_data = yf.download(tickers, period="1y", progress=False)['Close']
        
        # 處理單支股票與多支股票的格式差異
        current_prices = {}
        if isinstance(hist_data, pd.DataFrame) and not hist_data.empty:
            # 多支股票
            for t in tickers:
                if t in hist_data.columns:
                    current_prices[t] = hist_data[t].iloc[-1]
                else:
                    current_prices[t] = 0
            # 準備畫圖數據：計算每日總資產
            # 邏輯：假設過去一年都持有這些股數 (這是簡易回測邏輯)
            stock_history_val = (hist_data * df.set_index('Ticker')['Shares']).sum(axis=1)
            total_assets_history = stock_history_val + st.session_state['cash']
            
        elif isinstance(hist_data, pd.Series):
            # 單支股票
            current_prices[tickers[0]] = hist_data.iloc[-1]
            total_assets_history = (hist_data * df.iloc[0]['Shares']) + st.session_state['cash']
            
    except:
        current_prices = {t: 0 for t in tickers}
        st.error("⚠️ 數據連線異常，請稍後再試")

    # 2. 整合數據
    df['Current Price'] = df['Ticker'].map(current_prices)
    df['Market Value'] = df['Current Price'] * df['Shares']
    df['Total Profit'] = (df['Current Price'] - df['Cost']) * df['Shares']
    df['Return %'] = (df['Total Profit'] / (df['Cost'] * df['Shares']) * 100)
    
    # 3. 計算年化報酬 (CAGR)
    df['CAGR %'] = df.apply(lambda x: calculate_cagr(x['Current Price'], x['Cost'], x['Date']), axis=1) * 100

    total_stock_val = df['Market Value'].sum()
    total_profit = df['Total Profit'].sum()
else:
    total_stock_val = 0
    total_profit = 0

total_cash = st.session_state['cash']
total_assets = total_stock_val + total_cash
cash_ratio = (total_cash / total_assets * 100) if total_assets > 0 else 0

# --- A. 總資產折線圖 (放在最顯眼位置) ---
# 如果有歷史數據，繪製圖表
if not total_assets_history.empty:
    st.subheader("💰 總資產歷史走勢 (模擬回測)")
    
    # 使用 Plotly 繪製
    fig = px.area(
        x=total_assets_history.index, 
        y=total_assets_history.values,
        labels={'x': '日期', 'y': '總資產 (USD)'},
    )
    
    # 白底圖表設定
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font_color='#374151',
        xaxis=dict(showgrid=True, gridcolor='#f3f4f6'),
        yaxis=dict(showgrid=True, gridcolor='#f3f4f6'),
        margin=dict(l=0, r=0, t=0, b=0),
        height=300
    )
    fig.update_traces(line_color='#2563eb', fill='tozeroy', fillcolor='rgba(37, 99, 235, 0.1)')
    st.plotly_chart(fig, use_container_width=True)

# --- B. 關鍵 Metrics ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("總資產 (Total Assets)", f"${total_assets:,.0f}")
col2.metric("總損益 (Total P/L)", f"${total_profit:,.0f}", delta_color="normal")
col3.metric("股票市值", f"${total_stock_val:,.0f}")
col4.metric("現金水位", f"{cash_ratio:.1f}%")

# 現金條
st.write(f"**現金佔比: {cash_ratio:.1f}%**")
st.progress(min(cash_ratio/100, 1.0))

st.divider()

# --- C. 持倉明細 (新增年化報酬欄位) ---
st.subheader("📋 持倉詳細績效")

if not df.empty:
    # 格式化顯示
    display_df = df.copy()
    
    # 使用 column_config 製作漂亮的表格
    st.dataframe(
        display_df,
        column_config={
            "Ticker": "代號",
            "Date": st.column_config.DateColumn("買入日期"),
            "Cost": st.column_config.NumberColumn("成本價", format="$%.2f"),
            "Current Price": st.column_config.NumberColumn("現價", format="$%.2f"),
            "Shares": st.column_config.NumberColumn("股數", format="%.0f"),
            "Market Value": st.column_config.NumberColumn("市值", format="$%.0f"),
            "Total Profit": st.column_config.NumberColumn("總損益", format="$%.0f"),
            "Return %": st.column_config.NumberColumn("總報酬率", format="%.2f%%"),
            "CAGR %": st.column_config.NumberColumn("年化報酬 (CAGR)", format="%.2f%%", help="根據持有天數計算的複利年化報酬"),
        },
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("暫無持倉，請從左側側邊欄新增。")

# --- D. AI 顧問區 ---
st.divider()
st.subheader("🤖 AI 投資分析")

if not df.empty:
    ticker_selected = st.selectbox("選擇要分析的股票", df['Ticker'].unique())
    
    if st.button("生成分析與建議"):
        if not st.session_state['gemini_api_key']:
            st.warning("請先在側邊欄輸入 API Key")
        else:
            with st.spinner("AI 正在分析基本面與財報數據..."):
                try:
                    genai.configure(api_key=st.session_state['gemini_api_key'])
                    model = genai.GenerativeModel('gemini-pro')
                    
                    # 獲取該股數據
                    stock_row = df[df['Ticker'] == ticker_selected].iloc[0]
                    prompt = f"""
                    請用繁體中文分析美股 {ticker_selected}。
                    
                    我的持倉狀況：
                    - 成本: {stock_row['Cost']}
                    - 現價: {stock_row['Current Price']}
                    - 報酬率: {stock_row['Return %']:.2f}%
                    - 持有時間: 從 {stock_row['Date']} 至今
                    
                    請提供：
                    1. 短評目前該公司的基本面狀況。
                    2. 針對我的獲利狀況，建議續抱還是獲利了結？
                    """
                    res = model.generate_content(prompt)
                    st.markdown(f"""
                    <div style="background-color:#f3f4f6; padding:20px; border-radius:10px; border-left:5px solid #2563eb;">
                        {res.text}
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"分析失敗: {e}")
