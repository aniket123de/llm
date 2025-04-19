# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import yfinance as yf
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Stock Recommendation API")

# CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Asset Options and Category Mapping
# -------------------------------
asset_options = [
    "Stocks (Large-cap)",     # 0
    "Stocks (Mid-cap)",       # 1
    "Stocks (Small-cap)"      # 2
]

category_stocks = {
    "Long-Term Investor": ["AAPL", "MSFT", "TCS.NS", "INFY.NS", "RELIANCE.NS"],
    "Swing Trader": ["SBIN.NS", "HDFCBANK.NS", "AXISBANK.NS", "ITC.NS"],
    "Day Trader": ["ADANIENT.NS", "TATAMOTORS.NS", "VEDL.NS", "YESBANK.NS"],
    "Experimental Trader": [
        "IDEA.NS",
        "RTNPOWER.NS",
        "GTLINFRA.NS",
        "SUZLON.NS"
    ],
    "Balanced Investor": ["NIFTYBEES.NS", "ICICIBANK.NS", "HCLTECH.NS"]
}

# -------------------------------
# Pydantic Models
# -------------------------------
class UserInput(BaseModel):
    goal: int
    risk: int
    freq: int
    assets_indices: List[int]
    vol: int
    horizon: int
    decision: int
    emotion: int
    capital: int
    style: int

class StockRecommendation(BaseModel):
    ticker: str

class RecommendationResponse(BaseModel):
    category: str
    recommendations: List[StockRecommendation]

# -------------------------------
# Helper Functions
# -------------------------------
def categorize_user(user_vector):
    goal = user_vector["goal"]
    risk = user_vector["risk"]
    freq = user_vector["freq"]
    horizon = user_vector["horizon"]
    capital = user_vector["capital"]

    if goal == 0 and risk <= 1 and horizon == 3:
        return "Long-Term Investor"
    elif goal == 2 and risk >= 2:
        return "Day Trader"
    elif freq == 1 and risk == 1:
        return "Swing Trader"
    elif capital == 0 and risk >= 2:
        return "Experimental Trader"
    else:
        return "Balanced Investor"

def fetch_stock_data(ticker, period="3mo"):
    try:
        df = yf.download(ticker, period=period, interval="1d", progress=False)
        return df[['Open', 'Close', 'Volume']]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching data for {ticker}: {str(e)}")

def is_stock_upward_trending(df):
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    latest_ma20 = df['MA20'].iloc[-1]
    latest_ma50 = df['MA50'].iloc[-1]
    return latest_ma20 > latest_ma50

# -------------------------------
# API Endpoints
# -------------------------------
@app.get("/")
def read_root():
    return {"message": "Stock Recommendation API is running"}

@app.get("/asset-options")
def get_asset_options():
    return {"asset_options": asset_options}

@app.post("/recommend", response_model=RecommendationResponse)
def recommend_stocks(user_input: UserInput):
    user_vector = user_input.dict()
    category = categorize_user(user_vector)
    tickers = category_stocks.get(category, [])
    
    results = [StockRecommendation(ticker=t) for t in tickers]
    return RecommendationResponse(category=category, recommendations=results)

@app.get("/stock/{ticker}")
def get_stock_details(ticker: str, period: str = "3mo"):
    try:
        df = fetch_stock_data(ticker, period)
        trending = is_stock_upward_trending(df)
        return {
            "ticker": ticker,
            "trending": trending,
            "status": "Uptrend" if trending else "Not Trending",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
