"""
Application configuration and constants.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_ARTIFACTS_DIR = BASE_DIR / "model_artifacts"

# Sentiment model path (ProsusAI/finbert downloaded from Colab)
SENTIMENT_MODEL_DIR = MODEL_ARTIFACTS_DIR / "sentiment_model" / "finbert_sentiment_model"

# Cross-day decay parameter (matching Colab training)
# Formula: decayed[t] = sentiment[t] + exp(-LAMBDA) * decayed[t-1]
SENTIMENT_DECAY_LAMBDA = 0.3

# Model configuration (loaded from config.pkl at runtime)
LOOKBACK = 30  # Default, will be overwritten from config.pkl
HORIZON = 14   # Default, will be overwritten from config.pkl

# Feature definitions (must match training exactly)
LAGS = [1, 2, 3, 5, 7, 10, 14]
VOL_WINDOWS = [5, 10, 14]
EMA_WINDOWS = [3, 7, 14]

# Sentiment columns (matches Colab preprocessing)
SENT_COLS = [
    'daily_sentiment_decay',
    'news_volume',
    'log_news_volume',
    'decayed_news_volume',
    'high_news_regime'
]



# Yahoo Finance ticker for Brent Crude Oil Futures
BRENT_TICKER = "BZ=F"

# API settings
API_TITLE = "Oil Price Prediction API"
API_DESCRIPTION = "14-day Brent oil price forecasting using VMD-based ensemble model"
API_VERSION = "1.0.0"

# NewsAPI configuration (loaded from environment variables)
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
NEWSDATA_KEY = os.getenv("NEWSDATA_KEY", "")

# Sentiment analysis mode: 'simple' or 'finbert'
# Set to 'finbert' to use the custom ProsusAI/finbert model
SENTIMENT_MODE = os.getenv("SENTIMENT_MODE", "finbert")


