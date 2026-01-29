"""
News fetching and sentiment computation service.

Uses NewsAPI to fetch oil-related news and computes sentiment features.
"""
import os
import math
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import requests

from app.config import NEWSAPI_KEY as CONFIG_NEWSAPI_KEY

logger = logging.getLogger(__name__)

# NewsAPI configuration
NEWSAPI_BASE_URL = "https://newsapi.org/v2/everything"
NEWSAPI_KEY = CONFIG_NEWSAPI_KEY  # From config.py

# Oil-related search terms (simplified for better results)
OIL_SEARCH_QUERY = 'oil price OR crude oil OR brent OR OPEC'

# Sentiment decay parameter (from training)
SENTIMENT_DECAY_RATE = 0.1


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
    
    except requests.RequestException as e:
        logger.error(f"Error fetching news: {e}")
        return []


def _fetch_recent_news(api_key: str, page_size: int = 100) -> List[Dict[str, Any]]:
    """Fetch recent oil news without date filter (for NewsAPI free tier compatibility)."""
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
    
    except requests.RequestException as e:
        logger.error(f"Error fetching recent news: {e}")
        return []


def analyze_sentiment_simple(text: str) -> float:
    """
    Simple keyword-based sentiment analysis for oil news.
    
    For production, consider using:
    - FinBERT (financial sentiment)
    - VADER with custom lexicon
    - OpenAI/Claude API
    
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


def compute_sentiment_features(
    articles: List[Dict[str, Any]],
    decay_rate: float = SENTIMENT_DECAY_RATE
) -> Dict[str, Any]:
    """
    Compute sentiment features from a list of articles.
    
    This replicates the training feature computation:
    - daily_sentiment_decay: Decay-weighted average sentiment
    - news_volume: Number of articles
    - log_news_volume: Log-transformed volume
    - decayed_news_volume: Decay-weighted volume
    - high_news_regime: Binary flag for high activity (>median articles)
    
    Args:
        articles: List of article dictionaries
        decay_rate: Decay rate for time weighting (default: 0.1)
    
    Returns:
        Dictionary with all 5 sentiment features
    """
    if not articles:
        return {
            "daily_sentiment_decay": 0.0,
            "news_volume": 0,
            "log_news_volume": 0.0,
            "decayed_news_volume": 0.0,
            "high_news_regime": 0
        }
    
    # Get current time for decay calculation
    now = datetime.now()
    
    total_weight = 0.0
    weighted_sentiment_sum = 0.0
    decayed_volume = 0.0
    
    for article in articles:
        # Get article timestamp
        pub_time_str = article.get("publishedAt", "")
        try:
            # Parse ISO format: 2026-01-29T15:30:00Z
            pub_time = datetime.fromisoformat(pub_time_str.replace("Z", "+00:00"))
            pub_time = pub_time.replace(tzinfo=None)  # Remove timezone for comparison
        except (ValueError, AttributeError):
            pub_time = now - timedelta(hours=12)  # Default to 12 hours ago
        
        # Calculate hours since publication
        hours_ago = (now - pub_time).total_seconds() / 3600
        hours_ago = max(0, hours_ago)  # Ensure non-negative
        
        # Calculate decay weight
        weight = math.exp(-decay_rate * hours_ago)
        
        # Analyze sentiment from title and description
        title = article.get("title", "") or ""
        description = article.get("description", "") or ""
        text = f"{title} {description}"
        
        sentiment = analyze_sentiment_simple(text)
        
        # Accumulate weighted sentiment
        weighted_sentiment_sum += weight * sentiment
        total_weight += weight
        decayed_volume += weight
    
    # Compute final features
    news_volume = len(articles)
    log_news_volume = math.log(news_volume + 1)  # +1 to avoid log(0)
    
    # Decay-weighted average sentiment
    daily_sentiment_decay = weighted_sentiment_sum / total_weight if total_weight > 0 else 0.0
    
    # High news regime: typical oil news volume is ~20-50 articles/day
    # Flag as high if more than 30 articles
    high_news_regime = 1 if news_volume > 30 else 0
    
    return {
        "daily_sentiment_decay": round(daily_sentiment_decay, 6),
        "news_volume": news_volume,
        "log_news_volume": round(log_news_volume, 6),
        "decayed_news_volume": round(decayed_volume, 6),
        "high_news_regime": high_news_regime
    }


def fetch_and_compute_sentiment(
    date: str = None,
    api_key: str = None
) -> Dict[str, Any]:
    """
    Convenience function: Fetch news and compute sentiment features.
    
    Args:
        date: Date to analyze (YYYY-MM-DD). Default: yesterday.
        api_key: NewsAPI key.
    
    Returns:
        Dictionary with date and all 5 sentiment features.
    """
    if date is None:
        date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    articles = fetch_oil_news(date=date, api_key=api_key)
    features = compute_sentiment_features(articles)
    
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
