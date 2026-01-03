📈 AI-Powered Live Stock Tracker

An end-to-end AI-based stock analysis & prediction platform that combines:

• Real-time NSE stock data scraping
• Interactive historical analysis dashboards
• Deep Learning (LSTM) future price forecasting
• Intelligent AI trend insights

Built completely in Python + Streamlit + TensorFlow, without any paid APIs.

🚀 Project Overview

This application provides a live stock market dashboard for Indian NSE stocks with integrated AI-powered future price predictions using an LSTM neural network trained on NIFTY50 historical data (2004–2021).

It allows users to:

Track real-time NSE stock prices

Analyze long-term historical market behavior

Predict next-day & future stock prices

Receive AI-generated bullish/bearish insights

🧠 Key Features
📊 Live Market Dashboard

Real-time NSE web scraping (No API key required)

Live price, volume, day high/low

Intraday price movement visualization

Auto-refresh capability

📈 Historical Market Analysis

Load & analyze multiple stock CSV datasets

Interactive Plotly charts

Moving Averages (MA50, MA200)

Volatility & performance metrics

🤖 AI Stock Prediction Engine

Deep Learning LSTM model

Next trading day prediction

Multi-day future forecasting (up to 90 days)

Bullish / Bearish signal detection

💬 AI Stock Assistant

Ask natural language questions like:

“What is the trend?”

“Will the stock go up?”

“What is tomorrow’s prediction?”

AI explains patterns, trends & signals

🛠 Tech Stack
Layer	Technologies
Frontend	Streamlit
Visualization	Plotly
Backend	Python
AI / ML	TensorFlow (LSTM)
Data Processing	Pandas, NumPy, Scikit-Learn
Live Data	NSE Web Scraping
Model Type	LSTM Neural Network
📂 Folder Structure
AI-Stock-Tracker/
│
├── app.py
├── models/
│   └── lstm_model_clean.h5
│
├── data/
│   ├── RELIANCE.csv
│   ├── TCS.csv
│   ├── INFY.csv
│   └── ...
│
├── requirements.txt
└── README.md

📥 Installation
1️⃣ Clone the Repository
git clone https://github.com/yourusername/AI-Stock-Tracker.git
cd AI-Stock-Tracker

2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate # Linux/Mac

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Add Model & Data

Place LSTM model at:

models/lstm_model_clean.h5


Place historical stock CSV files inside:

data/


CSV Format:

Date, Open, High, Low, Close, Volume

▶️ Run the App
streamlit run app.py


Open in browser:

http://localhost:8501

📊 AI Model Details
Attribute	Value
Model	LSTM
Training Data	NIFTY50 (2004–2021)
Sequence Length	60 days
Prediction Horizon	1–90 days
Output	Stock Close Price
⚠️ Disclaimer

This application is strictly for educational & research purposes only.
It does not provide financial advice. Always consult certified financial advisors before investing.

✨ Why This Project Is Valuable

Real-world finance + AI integration

Shows end-to-end ML product development