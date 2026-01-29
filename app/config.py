"""
Application configuration and constants.
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_ARTIFACTS_DIR = BASE_DIR / "model_artifacts"

# Model configuration (loaded from config.pkl at runtime)
LOOKBACK = 30  # Default, will be overwritten from config.pkl
HORIZON = 14   # Default, will be overwritten from config.pkl

# Feature definitions (must match training exactly)
LAGS = [1, 2, 3, 5, 7, 10, 14]
VOL_WINDOWS = [5, 10, 14]
EMA_WINDOWS = [3, 7, 14]

# Sentiment columns (for Phase 2)
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

# NewsAPI configuration (primary)
NEWSAPI_KEY = "e3a34ba3e5aa43e59b473f2547503d5a"

# NewsData.io configuration (backup option)
NEWSDATA_KEY = "pub_dd9bcf0c5327499098e18d5c979e1abb"


