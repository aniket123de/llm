from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Tuple, Optional
import numpy as np
import yfinance as yf
import logging
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Investment Profile API", 
              description="API for analyzing user investment profiles and recommending stocks",
              version="1.0.0")

# Add CORS middleware to allow React Native app to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins - adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Assets multi-select reference
asset_options = [
    "Stocks (Large-cap)",     # 0
    "Stocks (Mid-cap)",       # 1
    "Stocks (Small-cap)",     # 2
]

# Stock recommendations for each category
category_stocks = {
    "Long-Term Investor": ["AAPL", "MSFT", "TCS.NS", "INFY.NS", "RELIANCE.NS"],
    "Swing Trader": ["SBIN.NS", "HDFCBANK.NS", "AXISBANK.NS", "ITC.NS"],
    "Day Trader": ["ADANIENT.NS", "TATAMOTORS.NS", "VEDL.NS", "YESBANK.NS"],
    "Experimental Trader": ["DOGE-USD", "BTC-USD", "PENNY.STOCK"],  # dummy entry
    "Balanced Investor": ["NIFTYBEES.NS", "ICICIBANK.NS", "HCLTECH.NS"]
}

# Pydantic models for request/response validation
class UserProfile(BaseModel):
    goal: int
    risk: int
    frequency: int
    assets: List[int]
    volatility_reaction: int
    time_horizon: int
    decision_making: int
    emotion: int
    capital: int
    trading_style: int

class StockAnalysis(BaseModel):
    ticker: str

class UserCategorization(BaseModel):
    category: str
    recommendations: List[StockAnalysis]


def categorize_user(user_vector):
    """Categorize user based on their inputs"""
    try:
        goal = user_vector[0]
        risk = user_vector[1]
        freq = user_vector[2]
        horizon = user_vector[len(asset_options)+4]
        capital = user_vector[-2]

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
    except Exception as e:
        logger.error(f"Error in user categorization: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error categorizing user: {str(e)}")


def fetch_stock_data(ticker="AAPL", period="6mo", interval="1d"):
    """Fetch stock data with error handling"""
    try:
        data = yf.download(ticker, period=period, interval=interval)
        if data.empty:
            raise ValueError(f"No data found for ticker {ticker}")
        return data[['Open', 'Close', 'Volume']]
    except Exception as e:
        logger.error(f"Error fetching stock data for {ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching stock data for {ticker}: {str(e)}")


def preprocess_lstm_data(df):
    """Preprocess data for LSTM model"""
    try:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df)

        X, y = [], []
        seq_len = 60
        for i in range(seq_len, len(scaled)):
            X.append(scaled[i-seq_len:i])
            y.append(scaled[i, 1])

        return np.array(X), np.array(y), scaler
    except Exception as e:
        logger.error(f"Error in data preprocessing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in data preprocessing: {str(e)}")


def build_lstm_model(input_shape):
    """Build LSTM model with error handling"""
    try:
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(50))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    except Exception as e:
        logger.error(f"Error building LSTM model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error building LSTM model: {str(e)}")


def is_stock_upward_trending(df):
    """Check if stock is trending upward with error handling"""
    try:
        df['MA20'] = df['Close'].rolling(20).mean()
        df['MA50'] = df['Close'].rolling(50).mean()
        latest_ma20 = df['MA20'].iloc[-1]
        latest_ma50 = df['MA50'].iloc[-1]
        return latest_ma20 > latest_ma50
    except Exception as e:
        logger.error(f"Error analyzing stock trend: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing stock trend: {str(e)}")


def recommend_stocks(category):
    """Recommend stocks based on category with error handling"""
    try:
        tickers = category_stocks.get(category, [])
        logger.info(f"Checking trends for category: {category}...")

        results = []
        for ticker in tickers:
            try:
                df = fetch_stock_data(ticker)
                results.append(StockAnalysis(
                    ticker=ticker, 
                ))
            except Exception as e:
                logger.warning(f"Error analyzing {ticker}: {str(e)}")
                results.append(StockAnalysis(
                    ticker=ticker, 
                ))

        return results
    except Exception as e:
        logger.error(f"Error in stock recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in stock recommendations: {str(e)}")


@app.get("/")
def read_root():
    """Root endpoint with API information"""
    return {
        "api": "Investment Profile API",
        "version": "1.0.0",
        "endpoints": {
            "/asset_options": "Get list of available asset options",
            "/analyze_profile": "Submit user profile and get investment categorization with recommendations"
        }
    }


@app.get("/asset_options", response_model=List[str])
def get_asset_options():
    """Endpoint to return available asset options"""
    return asset_options


@app.post("/analyze_profile", response_model=UserCategorization)
def analyze_profile(user_profile: UserProfile):
    """Analyze user profile and recommend stocks"""
    try:
        # Convert user profile to vector format
        assets_vector = [1 if i in user_profile.assets else 0 for i in range(len(asset_options))]
        user_vector = [
            user_profile.goal,
            user_profile.risk,
            user_profile.frequency,
            *assets_vector,
            user_profile.volatility_reaction,
            user_profile.time_horizon,
            user_profile.decision_making,
            user_profile.emotion,
            user_profile.capital,
            user_profile.trading_style
        ]
        
        # Categorize user
        category = categorize_user(user_vector)
        
        # Get stock recommendations
        stocks = recommend_stocks(category)
        
        return UserCategorization(
            category=category,
            recommendations=stocks
        )
    except Exception as e:
        logger.error(f"Error analyzing profile: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing profile: {str(e)}")


# Additional endpoint to get questionnaire structure
@app.get("/questionnaire")
def get_questionnaire():
    """Return the structure of the questionnaire for the frontend"""
    return {
        "questions": [
            {
                "id": "goal",
                "title": "What is your primary investment goal?",
                "options": [
                    {"value": 0, "label": "Wealth Creation (Long-term growth)"},
                    {"value": 1, "label": "Passive Income (Regular returns)"},
                    {"value": 2, "label": "Short-Term Gains (Quick profits)"},
                    {"value": 3, "label": "Learning (Educational purpose)"}
                ]
            },
            {
                "id": "risk",
                "title": "What is your risk tolerance level?",
                "options": [
                    {"value": 0, "label": "Conservative (Minimal risk)"},
                    {"value": 1, "label": "Moderate (Balanced approach)"},
                    {"value": 2, "label": "Aggressive (Higher risk for higher returns)"},
                    {"value": 3, "label": "Speculative (Willing to take substantial risks)"}
                ]
            },
            {
                "id": "frequency",
                "title": "How often do you plan to trade?",
                "options": [
                    {"value": 0, "label": "Daily (Active trading)"},
                    {"value": 1, "label": "Weekly (Regular monitoring)"},
                    {"value": 2, "label": "Monthly (Periodic review)"},
                    {"value": 3, "label": "Occasionally (As opportunities arise)"}
                ]
            },
            {
                "id": "assets",
                "title": "Which assets are you interested in? (Select all that apply)",
                "multiSelect": True,
                "options": [
                    {"value": 0, "label": "Stocks (Large-cap)"},
                    {"value": 1, "label": "Stocks (Mid-cap)"},
                    {"value": 2, "label": "Stocks (Small-cap)"}
                ]
            },
            {
                "id": "volatility_reaction",
                "title": "How do you typically react to market volatility?",
                "options": [
                    {"value": 0, "label": "Hold (Stay invested)"},
                    {"value": 1, "label": "Buy the dip (See opportunity)"},
                    {"value": 2, "label": "Sell to cut losses (Risk averse)"},
                    {"value": 3, "label": "Hedge (Seek protection)"}
                ]
            },
            {
                "id": "time_horizon",
                "title": "What is your investment time horizon?",
                "options": [
                    {"value": 0, "label": "Less than 1 month (Very short-term)"},
                    {"value": 1, "label": "1–6 months (Short-term)"},
                    {"value": 2, "label": "6 months–3 years (Medium-term)"},
                    {"value": 3, "label": "3+ years (Long-term)"}
                ]
            },
            {
                "id": "decision_making",
                "title": "How do you make your trading decisions?",
                "options": [
                    {"value": 0, "label": "Technical Analysis"},
                    {"value": 1, "label": "Fundamental Analysis"},
                    {"value": 2, "label": "News and Events"},
                    {"value": 3, "label": "Social Media/Community"},
                    {"value": 4, "label": "Algorithm/Bot Based"}
                ]
            },
            {
                "id": "emotion",
                "title": "What emotion primarily drives your trading decisions?",
                "options": [
                    {"value": 0, "label": "Confidence (Based on research)"},
                    {"value": 1, "label": "Fear (Of missing out or losing)"},
                    {"value": 2, "label": "Greed (Desire for high returns)"},
                    {"value": 3, "label": "FOMO (Fear of missing out)"}
                ]
            },
            {
                "id": "capital",
                "title": "What is your starting capital range?",
                "options": [
                    {"value": 0, "label": "Less than ₹10,000"},
                    {"value": 1, "label": "₹10,000 – ₹50,000"},
                    {"value": 2, "label": "₹50,000 – ₹2,00,000"},
                    {"value": 3, "label": "More than ₹2,00,000"}
                ]
            },
            {
                "id": "trading_style",
                "title": "What is your preferred trading approach?",
                "options": [
                    {"value": 0, "label": "Fully Automated (Algorithm-based)"},
                    {"value": 1, "label": "Semi-Automated (Mix of manual and automated)"},
                    {"value": 2, "label": "Manual (Complete manual control)"}
                ]
            }
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
