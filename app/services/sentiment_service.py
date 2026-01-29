"""
Sentiment service - handles sentiment data management and processing.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import logging

from app.database import (
    get_sentiment_history,
    add_sentiment,
    add_bulk_sentiment,
    get_latest_sentiment,
    get_sentiment_count
)
from app.config import EMA_WINDOWS

logger = logging.getLogger(__name__)


class SentimentService:
    """
    Service for managing sentiment data.
    
    Handles:
    - Adding daily sentiment
    - Retrieving sentiment history
    - Computing EMA features
    - Applying lag (sentiment shifted by 1 day)
    """
    
    def add_daily_sentiment(
        self,
        date_str: str,
        daily_sentiment_decay: float,
        news_volume: int,
        log_news_volume: float,
        decayed_news_volume: float,
        high_news_regime: int
    ) -> Dict[str, Any]:
        """
        Add sentiment data for a specific date.
        
        Args:
            date_str: Date in YYYY-MM-DD format
            daily_sentiment_decay: Decay-weighted sentiment score
            news_volume: Number of news articles
            log_news_volume: Log-transformed volume
            decayed_news_volume: Decay-weighted volume
            high_news_regime: Binary flag (0 or 1)
        
        Returns:
            Status dictionary
        """
        # Validate high_news_regime is 0 or 1
        if high_news_regime not in [0, 1]:
            raise ValueError("high_news_regime must be 0 or 1")
        
        add_sentiment(
            date_str=date_str,
            daily_sentiment_decay=daily_sentiment_decay,
            news_volume=news_volume,
            log_news_volume=log_news_volume,
            decayed_news_volume=decayed_news_volume,
            high_news_regime=high_news_regime
        )
        
        return {
            "success": True,
            "message": f"Sentiment added for {date_str}",
            "total_records": get_sentiment_count()
        }
    
    def add_bulk_sentiment(self, sentiment_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Add multiple sentiment records at once.
        
        Args:
            sentiment_list: List of sentiment dictionaries
        
        Returns:
            Status dictionary
        """
        count = add_bulk_sentiment(sentiment_list)
        
        return {
            "success": True,
            "records_added": count,
            "total_records": get_sentiment_count()
        }
    
    def get_sentiment_window(self, days: int = 60) -> pd.DataFrame:
        """
        Get sentiment history for feature engineering.
        
        Args:
            days: Number of days of history
        
        Returns:
            DataFrame with sentiment data (already lagged by storage date)
        """
        df = get_sentiment_history(days=days)
        
        if df.empty:
            logger.warning("No sentiment data available, using zeros")
            return pd.DataFrame()
        
        return df
    
    def compute_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all sentiment features including EMAs.
        
        The model expects:
        - Base: daily_sentiment_decay, news_volume, log_news_volume, 
                decayed_news_volume, high_news_regime
        - EMAs: *_ema_3, *_ema_7, *_ema_14 for 4 base columns
        
        Args:
            df: DataFrame with base sentiment columns
        
        Returns:
            DataFrame with all sentiment features
        """
        if df.empty:
            return df
        
        result = df.copy()
        
        # Columns to compute EMAs for
        ema_cols = [
            'daily_sentiment_decay',
            'news_volume',
            'log_news_volume',
            'decayed_news_volume'
        ]
        
        # Compute EMAs
        for col in ema_cols:
            if col in result.columns:
                for window in EMA_WINDOWS:
                    result[f'{col}_ema_{window}'] = (
                        result[col].ewm(span=window, adjust=False).mean()
                    )
        
        return result
    
    def merge_with_prices(
        self,
        price_df: pd.DataFrame,
        sentiment_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge sentiment data with price data.
        
        Sentiment is automatically lagged by 1 day during this merge:
        - Price for day T is matched with sentiment from day T-1
        
        Args:
            price_df: DataFrame with 'date' and price features
            sentiment_df: DataFrame with 'date' and sentiment columns
        
        Returns:
            Merged DataFrame
        """
        if sentiment_df.empty:
            logger.info("No sentiment data, returning price data with zero sentiment")
            return price_df
        
        # Ensure date columns are datetime
        price_df = price_df.copy()
        sentiment_df = sentiment_df.copy()
        
        price_df['date'] = pd.to_datetime(price_df['date'])
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
        
        # Shift sentiment by 1 day (today's prediction uses yesterday's sentiment)
        # This is done by shifting the date forward by 1 day in sentiment_df
        sentiment_df['date'] = sentiment_df['date'] + pd.Timedelta(days=1)
        
        # Merge on date
        merged = price_df.merge(sentiment_df, on='date', how='left')
        
        # Fill missing sentiment with 0 (matches training behavior)
        sentiment_cols = [
            'daily_sentiment_decay', 'news_volume', 'log_news_volume',
            'decayed_news_volume', 'high_news_regime'
        ]
        
        # Also include EMA columns
        for col in merged.columns:
            if any(base in col for base in sentiment_cols):
                merged[col] = merged[col].fillna(0)
        
        return merged
    
    def get_latest_info(self) -> Dict[str, Any]:
        """Get information about latest sentiment data."""
        latest = get_latest_sentiment()
        count = get_sentiment_count()
        
        return {
            "total_records": count,
            "latest_date": latest['date'] if latest else None,
            "latest_sentiment": latest
        }


# Global singleton
sentiment_service = SentimentService()
