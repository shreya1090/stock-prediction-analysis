import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import glob
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="AI-Powered Live Stock Tracker",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .big-font {
        font-size:50px !important;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# ===== UPDATED: NEW MODEL LOADING FUNCTION =====
@st.cache_resource
def load_stock_models(models_dir="models"):
    """Load all trained stock models and scalers"""
    try:
        models = {}
        scalers = {}
        
        # Find all .keras files
        model_files = glob.glob(f"{models_dir}/*_model.keras")
        
        if not model_files:
            st.warning(f"⚠️ No models found in '{models_dir}/' folder")
            return models, scalers
        
        for model_file in model_files:
            # Extract stock name from filename
            stock_name = os.path.basename(model_file).replace('_model.keras', '')
            scaler_file = model_file.replace('_model.keras', '_scaler.pkl')
            
            try:
                # Load model
                model = load_model(model_file, compile=False)
                model.compile(optimizer='adam', loss='mse')
                models[stock_name] = model
                
                # Load scaler
                if os.path.exists(scaler_file):
                    with open(scaler_file, 'rb') as f:
                        scalers[stock_name] = pickle.load(f)
                else:
                    st.warning(f"⚠️ Scaler not found for {stock_name}")
                    
            except Exception as e:
                st.warning(f"⚠️ Error loading {stock_name}: {e}")
        
        if models:
            st.success(f"✅ Loaded {len(models)} stock models successfully!")
        
        return models, scalers
        
    except Exception as e:
        st.error(f"❌ Model loading error: {str(e)}")
        return {}, {}

@st.cache_data
def load_historical_data(data_dir="data"):
    """Load all historical CSV files"""
    try:
        csv_files = glob.glob(f"{data_dir}/*.csv")
        stocks_data = {}
        for f in csv_files:
            stock_name = os.path.basename(f).replace(".csv", "")
            stocks_data[stock_name] = f
        return stocks_data
    except Exception as e:
        st.warning(f"Historical data not found: {e}")
        return {}

# Stock symbol mapping
STOCK_MAPPING = {
    "RELIANCE": "RELIANCE",
    "TCS": "TCS",
    "HDFCBANK": "HDFCBANK",
    "INFY": "INFY",
    "ICICIBANK": "ICICIBANK",
    "WIPRO": "WIPRO",
    "ITC": "ITC",
    "SBIN": "SBIN",
    "BHARTIARTL": "BHARTIARTL",
    "MARUTI": "MARUTI",
    "TATAMOTORS": "TATAMOTORS",
    "ADANIPORTS": "ADANIPORTS",
    "AXISBANK": "AXISBANK",
    "BAJAJFINSV": "BAJAJFINSV",
    "BAJFINANCE": "BAJFINANCE",
    "BAJAJ-AUTO": "BAJAJ-AUTO",
    "BRITANNIA": "BRITANNIA",
    "CIPLA": "CIPLA",
    "COALINDIA": "COALINDIA",
    "DRREDDY": "DRREDDY",
    "EICHERMOT": "EICHERMOT",
    "GRASIM": "GRASIM",
    "HCLTECH": "HCLTECH",
    "HEROMOTOCO": "HEROMOTOCO",
    "HINDALCO": "HINDALCO",
    "HINDUNILVR": "HINDUNILVR",
    "INDUSINDBK": "INDUSINDBK",
    "JSWSTEEL": "JSWSTEEL",
    "KOTAKBANK": "KOTAKBANK",
    "LT": "LT",
    "MM": "M&M",
    "NESTLEIND": "NESTLEIND",
    "NTPC": "NTPC",
    "ONGC": "ONGC",
    "POWERGRID": "POWERGRID",
    "SHREECEM": "SHREECEM",
    "SUNPHARMA": "SUNPHARMA",
    "TATASTEEL": "TATASTEEL",
    "TECHM": "TECHM",
    "TITAN": "TITAN",
    "ULTRACEMCO": "ULTRACEMCO",
    "UPL": "UPL",
    "VEDL": "VEDL"
}

def fetch_nse_stock(symbol):
    """Fetch stock data from NSE India"""
    try:
        symbol = symbol.replace('.NS', '').upper()
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        
        session = requests.Session()
        session.get('https://www.nseindia.com', headers=headers, timeout=10)
        
        url = f'https://www.nseindia.com/api/quote-equity?symbol={symbol}'
        response = session.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            price_info = data.get('priceInfo', {})
            
            return {
                'symbol': symbol,
                'name': data.get('info', {}).get('companyName', symbol),
                'price': price_info.get('lastPrice', 0),
                'change': price_info.get('change', 0),
                'pChange': price_info.get('pChange', 0),
                'open': price_info.get('open', 0),
                'high': price_info.get('intraDayHighLow', {}).get('max', 0),
                'low': price_info.get('intraDayHighLow', {}).get('min', 0),
                'previousClose': price_info.get('previousClose', 0),
                'volume': data.get('preOpenMarket', {}).get('totalTradedVolume', 0),
                'week52High': price_info.get('weekHighLow', {}).get('max', 0),
                'week52Low': price_info.get('weekHighLow', {}).get('min', 0),
                'success': True
            }
        else:
            return {'success': False, 'error': 'Failed to fetch data from NSE'}
            
    except Exception as e:
        return {'success': False, 'error': str(e)}

# ===== UPDATED: NEW PREDICTION FUNCTIONS =====
def predict_next_price(model, scaler, df, seq_len=60):
    """Predict next price using stock-specific model and scaler"""
    if model is None or scaler is None or len(df) < seq_len:
        return None
    
    try:
        # Scale the data
        close_data = df['Close'].values[-seq_len:].reshape(-1, 1)
        scaled_data = scaler.transform(close_data)
        
        # Prepare sequence
        seq = scaled_data.reshape(1, seq_len, 1)
        
        # Predict
        pred_scaled = model.predict(seq, verbose=0)
        next_price = scaler.inverse_transform(pred_scaled)[0][0]
        
        return next_price
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

def predict_future_prices(model, scaler, df, days=30, seq_len=60):
    """Predict multiple future prices"""
    if model is None or scaler is None or len(df) < seq_len:
        return []
    
    try:
        predictions = []
        
        # Start with last 60 days
        close_data = df['Close'].values[-seq_len:].reshape(-1, 1)
        scaled_data = scaler.transform(close_data)
        current_seq = scaled_data.reshape(1, seq_len, 1)
        
        for _ in range(days):
            # Predict next value
            pred_scaled = model.predict(current_seq, verbose=0)
            next_price = scaler.inverse_transform(pred_scaled)[0][0]
            predictions.append(next_price)
            
            # Update sequence
            new_val = pred_scaled[0][0]
            current_seq = np.append(current_seq[0][1:], [[new_val]], axis=0).reshape(1, seq_len, 1)
        
        return predictions
    except Exception as e:
        st.error(f"Future prediction error: {e}")
        return []

# ===== MAIN APP =====
st.title("🤖 AI-Powered Live Stock Tracker")
st.markdown("*Real-time data + LSTM predictions with individual stock models*")
st.markdown("---")

# Load models and data
models, scalers = load_stock_models()
historical_stocks = load_historical_data()

with st.sidebar:
    st.header("🔍 Stock Selection")
    
    has_historical = len(historical_stocks) > 0
    has_models = len(models) > 0
    
    if has_historical:
        st.success(f"✅ {len(historical_stocks)} stocks with historical data")
    else:
        st.warning("⚠️ No historical data found in 'data/' folder")
    
    if has_models:
        st.success(f"✅ {len(models)} AI models loaded")
    else:
        st.warning("⚠️ No AI models found in 'models/' folder")
    
    st.markdown("---")
    
    if has_historical:
        stock_options = sorted(list(historical_stocks.keys()))
        selected_csv_name = st.selectbox("Select Stock", stock_options)
        stock_symbol = STOCK_MAPPING.get(selected_csv_name, selected_csv_name)
    else:
        st.subheader("Manual Entry")
        stock_symbol = st.text_input(
            "Enter NSE Stock Symbol",
            placeholder="e.g., RELIANCE, TCS, INFY"
        )
        selected_csv_name = None
    
    st.markdown("---")
    st.subheader("⚙️ View Options")
    show_live = st.checkbox("Show Live Data", value=True)
    show_historical = st.checkbox("Show Historical Data", value=has_historical)
    show_predictions = st.checkbox("Show AI Predictions", value=has_models)
    
    if show_predictions and has_models:
        prediction_days = st.slider("Prediction Days", 1, 90, 30)
    
    st.markdown("---")
    
    st.subheader("🔄 Auto Refresh")
    auto_refresh = st.checkbox("Enable Auto Refresh", value=False)
    if auto_refresh:
        refresh_interval = st.slider("Interval (seconds)", 10, 120, 30)
    
    if st.button("🔄 Refresh Now"):
        st.rerun()

if stock_symbol or selected_csv_name:
    
    # Check if model exists for this stock
    has_model_for_stock = selected_csv_name in models and selected_csv_name in scalers
    
    if not has_model_for_stock and show_predictions:
        st.warning(f"⚠️ No AI model found for {selected_csv_name}. Predictions disabled.")
        show_predictions = False
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📊 Live Dashboard", "📈 Historical Analysis", "🤖 AI Insights"])
    
    with tab1:
        if show_live:
            with st.spinner(f"Fetching live data for {stock_symbol}..."):
                stock_data = fetch_nse_stock(stock_symbol)
                
                if stock_data.get('success'):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            label=stock_data.get('name', stock_symbol),
                            value=f"₹{stock_data['price']:.2f}",
                            delta=f"{stock_data['change']:.2f} ({stock_data['pChange']:.2f}%)"
                        )
                    
                    with col2:
                        st.metric("Day High", f"₹{stock_data['high']:.2f}")
                    
                    with col3:
                        st.metric("Day Low", f"₹{stock_data['low']:.2f}")
                    
                    with col4:
                        volume = stock_data.get('volume', 0)
                        st.metric("Volume", f"{volume:,.0f}")
                    
                    st.markdown("---")
                    
                    st.subheader("📊 Today's Price Movement")
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=['Previous Close', 'Open', 'Low', 'Current', 'High'],
                        y=[stock_data['previousClose'], stock_data['open'], 
                           stock_data['low'], stock_data['price'], stock_data['high']],
                        mode='lines+markers',
                        name='Price',
                        line=dict(color='#1f77b4', width=3),
                        marker=dict(size=12)
                    ))
                    
                    fig.update_layout(
                        yaxis_title='Price (₹)',
                        template='plotly_white',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.error(f"❌ Error: {stock_data.get('error')}")
        else:
            st.info("📊 Enable 'Show Live Data' in sidebar")
    
    with tab2:
        if show_historical and selected_csv_name and selected_csv_name in historical_stocks:
            st.subheader(f"📈 Historical Data: {selected_csv_name}")
            
            file_path = historical_stocks[selected_csv_name]
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            df.set_index('Date', inplace=True)
            df.ffill(inplace=True)
            
            st.write(f"**Data Range:** {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
            st.write(f"**Total Trading Days:** {len(df)}")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Average Close", f"₹{df['Close'].mean():.2f}")
            with col2:
                st.metric("All-Time High", f"₹{df['High'].max():.2f}")
            with col3:
                st.metric("All-Time Low", f"₹{df['Low'].min():.2f}")
            with col4:
                st.metric("Volatility (Std)", f"₹{df['Close'].std():.2f}")
            
            st.markdown("---")
            st.subheader("📊 Price History & AI Predictions")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['Close'],
                mode='lines',
                name='Historical Close Price',
                line=dict(color='#2E86AB', width=2)
            ))
            
            # Add predictions if model exists
            if show_predictions and has_model_for_stock:
                model = models[selected_csv_name]
                scaler = scalers[selected_csv_name]
                
                future_prices = predict_future_prices(model, scaler, df, days=prediction_days)
                
                if future_prices:
                    last_date = df.index[-1]
                    future_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                                 periods=len(future_prices), freq='D')
                    
                    fig.add_trace(go.Scatter(
                        x=future_dates,
                        y=future_prices,
                        mode='lines',
                        name='AI Prediction',
                        line=dict(color='#FF6B35', width=2, dash='dash')
                    ))
                    
                    st.success(f"✅ Predicted {len(future_prices)} future prices using {selected_csv_name} model")
            
            fig.update_layout(
                yaxis_title='Price (₹)',
                xaxis_title='Date',
                template='plotly_white',
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("📋 View Recent Data"):
                st.dataframe(df.tail(20)[['Open', 'High', 'Low', 'Close', 'Volume']], 
                           use_container_width=True)
    
    with tab3:
        st.subheader("🤖 AI-Powered Insights")
        
        if has_model_for_stock and selected_csv_name in historical_stocks:
            file_path = historical_stocks[selected_csv_name]
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            df.set_index('Date', inplace=True)
            df.ffill(inplace=True)
            
            model = models[selected_csv_name]
            scaler = scalers[selected_csv_name]
            
            st.subheader("📅 Next Trading Day Prediction")
            
            next_price = predict_next_price(model, scaler, df)
            
            if next_price:
                current_price = df['Close'].iloc[-1]
                price_change = next_price - current_price
                price_change_pct = (price_change / current_price) * 100
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Current Price", f"₹{current_price:.2f}")
                with col2:
                    st.metric("Predicted Price", f"₹{next_price:.2f}", 
                             delta=f"{price_change:.2f} ({price_change_pct:.2f}%)")
                with col3:
                    if price_change_pct > 1:
                        st.success("🟢 STRONG BUY")
                    elif price_change_pct > 0:
                        st.info("🟡 BUY")
                    elif price_change_pct > -1:
                        st.warning("🟡 HOLD")
                    else:
                        st.error("🔴 SELL")
            
            st.markdown("---")
            
            with st.expander("ℹ️ About This Model"):
                st.write(f"**Stock:** {selected_csv_name}")
                st.write(f"**Model Type:** LSTM (Long Short-Term Memory)")
                st.write(f"**Training Data:** Historical data from {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
                st.write(f"**Sequence Length:** 60 days")
                st.write("**Note:** This is a stock-specific model trained on this company's historical patterns")
        
        else:
            st.warning(f"⚠️ AI model not available for {selected_csv_name}")
            st.info("Available models: " + ", ".join(sorted(models.keys())))

    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

else:
    st.info("👈 Please select a stock from the sidebar!")
    
    st.subheader("📁 Setup Instructions:")
    st.write("""
    **For your NEW trained models:**
    
    1. Create a `models/` folder in your Streamlit app directory
    2. Copy all your trained models from Google Drive:
       - `RELIANCE_model.keras` and `RELIANCE_scaler.pkl`
       - `TCS_model.keras` and `TCS_scaler.pkl`
       - ... (all 54 stocks)
    3. Create a `data/` folder and add CSV files with historical data
    4. CSV files should match model names (e.g., `RELIANCE.csv`)
    5. Select a stock from the sidebar
    """)