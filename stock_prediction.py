import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pickle
import os

# Page configuration
st.set_page_config(page_title="Stock Price Prediction", page_icon="📈", layout="wide")

st.title("📈 Realistic Stock Price Prediction System")
st.markdown("*Using LSTM Neural Networks with Proper Scaling*")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("🔍 Stock Selection")
    
    # Popular stocks
    popular_stocks = {
        "": "",
        "Reliance": "RELIANCE.NS",
        "TCS": "TCS.NS",
        "HDFC Bank": "HDFCBANK.NS",
        "Infosys": "INFY.NS",
        "ICICI Bank": "ICICIBANK.NS",
        "Wipro": "WIPRO.NS",
        "ITC": "ITC.NS",
        "Adani Ports": "ADANIPORTS.NS",
        "Asian Paints": "ASIANPAINT.NS",
        "Axis Bank": "AXISBANK.NS"
    }
    
    selected = st.selectbox("Select Stock:", list(popular_stocks.keys()))
    
    if selected:
        stock_symbol = popular_stocks[selected]
    else:
        stock_symbol = st.text_input("Or enter symbol:", "RELIANCE.NS")
    
    st.markdown("---")
    
    # Model parameters
    st.subheader("⚙️ Model Settings")
    lookback_days = st.slider("Lookback Days", 30, 120, 60)
    epochs = st.slider("Training Epochs", 10, 100, 50)
    
    st.markdown("---")
    train_button = st.button("🚀 Train Model", type="primary", use_container_width=True)
    predict_button = st.button("🔮 Predict Next Day", use_container_width=True)

# Function to fetch stock data
@st.cache_data(ttl=3600)
def fetch_stock_data(symbol, period="5y"):
    """Fetch historical stock data"""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        if df.empty:
            return None
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Function to prepare data for LSTM
def prepare_lstm_data(data, lookback=60):
    """Prepare data for LSTM with proper scaling"""
    
    # Use Close prices
    prices = data['Close'].values.reshape(-1, 1)
    
    # Create and fit scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(prices)
    
    # Create sequences
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    return X, y, scaler, prices

# Function to create LSTM model
def create_lstm_model(lookback):
    """Create LSTM model architecture"""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to make prediction with validation
def make_realistic_prediction(model, scaler, last_sequence, current_price):
    """Make prediction and validate it's realistic"""
    
    # Make prediction (this will be scaled between 0-1)
    scaled_prediction = model.predict(last_sequence, verbose=0)
    
    # Inverse transform to get actual price
    predicted_price = scaler.inverse_transform(scaled_prediction)[0][0]
    
    # Calculate percentage change
    price_change_percent = ((predicted_price - current_price) / current_price) * 100
    
    # Validation: If change is unrealistic (>15%), cap it
    max_change_percent = 15.0  # Maximum 15% change per day
    
    if abs(price_change_percent) > max_change_percent:
        st.warning(f"⚠️ Model predicted {price_change_percent:.2f}% change - capping to realistic range")
        
        # Cap the change to max_change_percent
        if price_change_percent > 0:
            predicted_price = current_price * (1 + max_change_percent/100)
        else:
            predicted_price = current_price * (1 - max_change_percent/100)
        
        price_change_percent = ((predicted_price - current_price) / current_price) * 100
    
    return predicted_price, price_change_percent

# Main App Logic
if stock_symbol:
    
    # Fetch data
    with st.spinner("Fetching stock data..."):
        df = fetch_stock_data(stock_symbol)
    
    if df is not None and not df.empty:
        
        # Display current stock info
        current_price = df['Close'].iloc[-1]
        prev_close = df['Close'].iloc[-2]
        change = current_price - prev_close
        change_percent = (change / prev_close) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Price", f"₹{current_price:.2f}", f"{change:.2f} ({change_percent:.2f}%)")
        with col2:
            st.metric("Day High", f"₹{df['High'].iloc[-1]:.2f}")
        with col3:
            st.metric("Day Low", f"₹{df['Low'].iloc[-1]:.2f}")
        with col4:
            st.metric("Volume", f"{df['Volume'].iloc[-1]:,.0f}")
        
        st.markdown("---")
        
        # Historical price chart
        st.subheader("📊 Historical Price Chart")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='#1f77b4', width=2)
        ))
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Price (₹)',
            template='plotly_white',
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Train Model
        if train_button:
            st.subheader("🔧 Training LSTM Model")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Prepare data
                status_text.text("Preparing data...")
                progress_bar.progress(20)
                
                X, y, scaler, prices = prepare_lstm_data(df, lookback=lookback_days)
                
                # Split data
                train_size = int(len(X) * 0.8)
                X_train, X_test = X[:train_size], X[train_size:]
                y_train, y_test = y[:train_size], y[train_size:]
                
                status_text.text(f"Training on {len(X_train)} samples...")
                progress_bar.progress(40)
                
                # Create and train model
                model = create_lstm_model(lookback_days)
                
                history = model.fit(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=32,
                    validation_data=(X_test, y_test),
                    verbose=0
                )
                
                progress_bar.progress(80)
                status_text.text("Saving model...")
                
                # Save model and scaler
                model_path = f"model_{stock_symbol.replace('.NS', '')}.h5"
                scaler_path = f"scaler_{stock_symbol.replace('.NS', '')}.pkl"
                
                model.save(model_path)
                with open(scaler_path, 'wb') as f:
                    pickle.dump(scaler, f)
                
                progress_bar.progress(100)
                status_text.text("✅ Model trained successfully!")
                
                # Store in session state
                st.session_state['model'] = model
                st.session_state['scaler'] = scaler
                st.session_state['lookback'] = lookback_days
                st.session_state['stock_symbol'] = stock_symbol
                
                st.success(f"✅ Model trained and saved successfully!")
                st.info(f"Final Loss: {history.history['loss'][-1]:.6f}")
                
                # Plot training history
                col1, col2 = st.columns(2)
                with col1:
                    fig_loss = go.Figure()
                    fig_loss.add_trace(go.Scatter(y=history.history['loss'], name='Training Loss'))
                    fig_loss.add_trace(go.Scatter(y=history.history['val_loss'], name='Validation Loss'))
                    fig_loss.update_layout(title='Model Loss', xaxis_title='Epoch', yaxis_title='Loss')
                    st.plotly_chart(fig_loss, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")
                progress_bar.empty()
                status_text.empty()
        
        # Make Prediction
        if predict_button:
            st.subheader("🔮 Next Trading Day Prediction")
            
            try:
                # Try to load existing model
                model_path = f"model_{stock_symbol.replace('.NS', '')}.h5"
                scaler_path = f"scaler_{stock_symbol.replace('.NS', '')}.pkl"
                
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    model = load_model(model_path)
                    with open(scaler_path, 'rb') as f:
                        scaler = pickle.load(f)
                    lookback = lookback_days
                    
                elif 'model' in st.session_state:
                    model = st.session_state['model']
                    scaler = st.session_state['scaler']
                    lookback = st.session_state['lookback']
                else:
                    st.warning("⚠️ Please train the model first!")
                    st.stop()
                
                # Prepare last sequence for prediction
                last_prices = df['Close'].values[-lookback:].reshape(-1, 1)
                scaled_last = scaler.transform(last_prices)
                last_sequence = scaled_last.reshape(1, lookback, 1)
                
                # Make realistic prediction
                predicted_price, price_change_percent = make_realistic_prediction(
                    model, scaler, last_sequence, current_price
                )
                
                # Determine signal
                if price_change_percent > 2:
                    signal = "🟢 Strong Bullish"
                    signal_color = "green"
                elif price_change_percent > 0:
                    signal = "🟢 Bullish"
                    signal_color = "lightgreen"
                elif price_change_percent < -2:
                    signal = "🔴 Strong Bearish"
                    signal_color = "red"
                else:
                    signal = "🔴 Bearish"
                    signal_color = "lightcoral"
                
                # Display prediction
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("### Current Price")
                    st.markdown(f"# ₹{current_price:.2f}")
                
                with col2:
                    st.markdown("### Predicted Price")
                    st.markdown(f"# ₹{predicted_price:.2f}")
                    st.markdown(f"**Change:** ₹{predicted_price - current_price:.2f} ({price_change_percent:+.2f}%)")
                
                with col3:
                    st.markdown("### Signal")
                    st.markdown(f"<h2 style='color: {signal_color};'>{signal}</h2>", unsafe_allow_html=True)
                
                # Prediction visualization
                st.markdown("---")
                st.subheader("📈 Prediction Visualization")
                
                # Show last 30 days + prediction
                last_30_days = df['Close'].iloc[-30:].copy()
                
                fig = go.Figure()
                
                # Historical data
                fig.add_trace(go.Scatter(
                    x=last_30_days.index,
                    y=last_30_days.values,
                    mode='lines',
                    name='Historical',
                    line=dict(color='blue', width=2)
                ))
                
                # Prediction point
                next_day = last_30_days.index[-1] + timedelta(days=1)
                fig.add_trace(go.Scatter(
                    x=[last_30_days.index[-1], next_day],
                    y=[current_price, predicted_price],
                    mode='lines+markers',
                    name='Prediction',
                    line=dict(color='red', width=2, dash='dash'),
                    marker=dict(size=10, color='red')
                ))
                
                fig.update_layout(
                    title='Price Prediction for Next Trading Day',
                    xaxis_title='Date',
                    yaxis_title='Price (₹)',
                    template='plotly_white',
                    height=400,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Confidence and warnings
                st.markdown("---")
                st.subheader("⚠️ Important Notes")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.info("""
                    **Prediction Details:**
                    - Based on last {} days of data
                    - Uses LSTM neural network
                    - Predictions capped at ±15% change
                    - For reference only, not financial advice
                    """.format(lookback))
                
                with col2:
                    st.warning("""
                    **Disclaimer:**
                    - Stock market predictions are inherently uncertain
                    - Past performance doesn't guarantee future results
                    - Always do your own research
                    - Consult financial advisors for decisions
                    """)
                
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
                st.info("Please try training the model again.")
        
    else:
        st.error("❌ Could not fetch stock data. Please check the symbol.")

else:
    st.info("👈 Please select a stock from the sidebar to begin!")
    
    st.markdown("""
    ## 📚 How to Use:
    
    1. **Select a stock** from the sidebar dropdown
    2. **Adjust model parameters** (optional):
       - Lookback Days: How many past days to consider
       - Training Epochs: More epochs = better learning (but slower)
    3. **Click "Train Model"** to build the prediction model
    4. **Click "Predict Next Day"** to see tomorrow's price prediction
    
    ## 🎯 Features:
    
    ✅ **Realistic Predictions** - Capped at ±15% daily change
    ✅ **Proper Data Scaling** - Correct scaler usage
    ✅ **Model Validation** - Training/validation split
    ✅ **Visual Analysis** - Interactive charts
    ✅ **Buy/Sell Signals** - Based on predicted movement
    
    ## ⚙️ How It Works:
    
    - Uses **LSTM (Long Short-Term Memory)** neural networks
    - Trained on historical closing prices
    - Applies **MinMaxScaler** for proper normalization
    - Validates predictions to ensure realistic ranges
    - Saves trained models for reuse
    
    ## ⚠️ Disclaimer:
    
    This tool is for **educational purposes only**. Stock market predictions 
    are inherently uncertain. Always consult with financial advisors and do 
    your own research before making investment decisions.
    """)