"""
News fetching and sentiment computation service.

Uses NewsAPI to fetch oil-related news and computes sentiment features.
Uses custom ProsusAI/finbert model for sentiment analysis (matching Colab training).

SENTIMENT COMPUTATION (matching Colab exactly):
1. Each article gets a sentiment score using finbert_sentiment_continuous()
2. Daily sentiment = simple mean of all article scores (NO within-day decay!)
3. Cross-day decay is applied later in sentiment_service.py
"""
import os
import math
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from app.config import NEWSAPI_KEY as CONFIG_NEWSAPI_KEY, SENTIMENT_MODE

logger = logging.getLogger(__name__)

# NewsAPI configuration
NEWSAPI_BASE_URL = "https://newsapi.org/v2/everything"
NEWSAPI_KEY = CONFIG_NEWSAPI_KEY  # From config.py

# Oil-related search terms (simplified for better results)
OIL_SEARCH_QUERY = 'oil price OR crude oil OR brent OR OPEC'


def fetch_oil_news(
    date: str = None,
    api_key: str = None,
    page_size: int = 100
) -> List[Dict[str, Any]]:
    """
    Fetch oil-related news articles from NewsAPI.
    
    Args:
        date: Date to fetch news for (YYYY-MM-DD). Default: yesterday.
        api_key: NewsAPI key. Falls back to NEWSAPI_KEY from config.
        page_size: Number of articles to fetch (max 100).
    
    Returns:
        List of article dictionaries with title, description, publishedAt, source.
    
    Raises:
        ValueError: If no API key is provided.
    """
    import requests
    
    key = api_key or NEWSAPI_KEY
    if not key:
        raise ValueError(
            "NewsAPI key required. Set NEWSAPI_KEY in config.py or pass api_key parameter."
        )
    
    # Default to yesterday's date
    if date is None:
        date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    # First try date-specific fetch
    articles = _fetch_news_for_date(date, key, page_size)
    
    # If no articles found, try fetching recent news without date filter
    if not articles:
        logger.warning(f"No articles for {date}, fetching recent news instead")
        articles = _fetch_recent_news(key, page_size)
    
    return articles


def _fetch_news_for_date(date: str, api_key: str, page_size: int) -> List[Dict[str, Any]]:
    """Fetch news for a specific date."""
    import requests
    
    target_date = datetime.strptime(date, "%Y-%m-%d")
    from_date = target_date.strftime("%Y-%m-%d")
    to_date = (target_date + timedelta(days=1)).strftime("%Y-%m-%d")
    
    params = {
        "q": OIL_SEARCH_QUERY,
        "from": from_date,
        "to": to_date,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": api_key
    }
    
    try:
        response = requests.get(NEWSAPI_BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data.get("status") != "ok":
            logger.error(f"NewsAPI error: {data.get('message')}")
            return []
        
        articles = data.get("articles", [])
        logger.info(f"Fetched {len(articles)} articles for {date}")
        return articles
    
    except Exception as e:
        logger.error(f"Error fetching news: {e}")
        return []


def _fetch_recent_news(api_key: str, page_size: int = 100) -> List[Dict[str, Any]]:
    """Fetch recent oil news without date filter (for NewsAPI free tier compatibility)."""
    import requests
    
    params = {
        "q": OIL_SEARCH_QUERY,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": api_key
    }
    
    try:
        response = requests.get(NEWSAPI_BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data.get("status") != "ok":
            logger.error(f"NewsAPI error: {data.get('message')}")
            return []
        
        articles = data.get("articles", [])
        logger.info(f"Fetched {len(articles)} recent articles")
        return articles
    
    except Exception as e:
        logger.error(f"Error fetching recent news: {e}")
        return []


def analyze_sentiment_simple(text: str) -> float:
    """
    Simple keyword-based sentiment analysis for oil news.
    Fallback when custom model is not available.
    
    Returns:
        Sentiment score between -1 (negative) and 1 (positive)
    """
    if not text:
        return 0.0
    
    text_lower = text.lower()
    
    # Positive keywords for oil market
    positive_words = [
        "surge", "soar", "rally", "gain", "rise", "jump", "increase",
        "boost", "growth", "recovery", "demand", "bullish", "optimism",
        "supply cut", "production cut", "higher", "up"
    ]
    
    # Negative keywords for oil market
    negative_words = [
        "fall", "drop", "decline", "crash", "plunge", "slump", "tumble",
        "bearish", "oversupply", "glut", "recession", "weak", "lower",
        "down", "loss", "concern", "fear", "crisis"
    ]
    
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    total = pos_count + neg_count
    if total == 0:
        return 0.0
    
    return (pos_count - neg_count) / total


def analyze_sentiment(text: str, mode: str = None) -> float:
    """
    Analyze sentiment of text using the specified mode.
    
    Args:
        text: Text to analyze
        mode: 'simple' or 'finbert'. If None, uses SENTIMENT_MODE from config.
    
    Returns:
        Sentiment score between -1 (negative) and 1 (positive)
    """
    if mode is None:
        mode = SENTIMENT_MODE
    
    if mode == "finbert":
        try:
            from app.services.finbert_analyzer import analyze_sentiment_finbert, is_finbert_available
            
            if is_finbert_available():
                return analyze_sentiment_finbert(text)
            else:
                logger.warning("Custom FinBERT model not available, falling back to simple sentiment")
                return analyze_sentiment_simple(text)
        except Exception as e:
            logger.warning(f"FinBERT analysis failed: {e}, falling back to simple sentiment")
            return analyze_sentiment_simple(text)
    else:
        return analyze_sentiment_simple(text)


def compute_sentiment_features(
    articles: List[Dict[str, Any]],
    sentiment_mode: str = None
) -> Dict[str, Any]:
    """
    Compute sentiment features from a list of articles.
    
    IMPORTANT: This matches Colab training preprocessing exactly:
    - daily_sentiment: Simple mean of all article sentiments (NO within-day decay!)
    - news_volume: Number of articles
    - log_news_volume: log(news_volume + 1)
    - decayed_news_volume: EWM of volume (for feature matching)
    - high_news_regime: Binary flag (1 if volume > 30 articles)
    
    Cross-day decay formula (s[t] + exp(-0.3) * s[t-1]) is applied 
    LATER in sentiment_service.py, NOT here!
    
    Args:
        articles: List of article dictionaries from NewsAPI
        sentiment_mode: 'simple' or 'finbert' (default: from config)
    
    Returns:
        Dictionary with sentiment features
    """
    if not articles:
        return {
            "daily_sentiment_decay": 0.0,
            "news_volume": 0,
            "log_news_volume": 0.0,
            "decayed_news_volume": 0.0,
            "high_news_regime": 0
        }
    
    # Collect all article sentiments
    all_sentiments = []
    
    # Check if we should use FinBERT batch processing
    use_finbert = (sentiment_mode == "finbert") if sentiment_mode else (SENTIMENT_MODE == "finbert")
    
    if use_finbert:
        try:
            from app.services.finbert_analyzer import analyze_batch_finbert, is_finbert_available
            
            if is_finbert_available():
                # Prepare texts for batch processing
                texts = []
                for article in articles:
                    title = article.get("title", "") or ""
                    description = article.get("description", "") or ""
                    content = article.get("content", "") or ""
                    # Combine title + description (matching Colab)
                    text = f"{title}. {description}"
                    if content and len(content) > len(description):
                        text = f"{title}. {content}"
                    texts.append(text)
                
                logger.info(f"Analyzing {len(texts)} articles with custom FinBERT...")
                all_sentiments = analyze_batch_finbert(texts)
                logger.info(f"FinBERT analysis complete. Mean sentiment: {np.mean(all_sentiments):.4f}")
            else:
                logger.warning("Custom FinBERT not available, using simple sentiment")
                use_finbert = False
        except Exception as e:
            logger.warning(f"FinBERT batch analysis failed: {e}, using simple sentiment")
            use_finbert = False
    
    # Fallback to simple sentiment
    if not use_finbert or not all_sentiments:
        for article in articles:
            title = article.get("title", "") or ""
            description = article.get("description", "") or ""
            text = f"{title} {description}"
            sentiment = analyze_sentiment_simple(text)
            all_sentiments.append(sentiment)
    
    # Compute features
    news_volume = len(articles)
    log_news_volume = math.log(news_volume + 1)
    
    # CRITICAL: Simple mean - matching Colab training exactly!
    # Colab code: daily_sentiment = df.groupby('date')['sentiment'].mean()
    daily_sentiment = float(np.mean(all_sentiments)) if all_sentiments else 0.0
    
    # Decayed volume (for feature compatibility, uses EWM-like approach)
    # This is a single-day estimate; proper EWM needs history
    decayed_news_volume = float(news_volume) * 0.5  # Approximate
    
    # High news regime: flag if more than 30 articles
    high_news_regime = 1 if news_volume > 30 else 0
    
    logger.info(f"Computed sentiment: {daily_sentiment:.4f} from {news_volume} articles")
    
    return {
        "daily_sentiment_decay": round(daily_sentiment, 6),
        "news_volume": news_volume,
        "log_news_volume": round(log_news_volume, 6),
        "decayed_news_volume": round(decayed_news_volume, 6),
        "high_news_regime": high_news_regime
    }


def fetch_and_compute_sentiment(
    date: str = None,
    api_key: str = None,
    sentiment_mode: str = None
) -> Dict[str, Any]:
    """
    Convenience function: Fetch news and compute sentiment features.
    
    Args:
        date: Date to analyze (YYYY-MM-DD). Default: yesterday.
        api_key: NewsAPI key.
        sentiment_mode: 'simple' or 'finbert'. Default: from config.
    
    Returns:
        Dictionary with date and all sentiment features.
    """
    if date is None:
        date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    articles = fetch_oil_news(date=date, api_key=api_key)
    features = compute_sentiment_features(articles, sentiment_mode=sentiment_mode)
    
    return {
        "date": date,
        **features
    }


# For CLI usage
if __name__ == "__main__":
    import sys
    import json
    
    # Check for API key
    if not NEWSAPI_KEY:
        print("Error: Set NEWSAPI_KEY environment variable")
        print("  Windows: set NEWSAPI_KEY=your_key_here")
        print("  Linux:   export NEWSAPI_KEY=your_key_here")
        sys.exit(1)
    
    # Get date from command line or use yesterday
    date = sys.argv[1] if len(sys.argv) > 1 else None
    
    result = fetch_and_compute_sentiment(date=date)
    print(json.dumps(result, indent=2))
