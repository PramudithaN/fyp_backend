"""
Sentiment service - handles sentiment data management and processing.

IMPORTANT: Cross-day decay matching Colab training:
- LAMBDA = 0.3
- Formula: decayed[t] = sentiment[t] + exp(-LAMBDA) * decayed[t-1]
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import logging
import math

from app.database import (
    get_sentiment_history,
    add_sentiment,
    add_bulk_sentiment,
    get_latest_sentiment,
    get_sentiment_count,
    get_sentiment_for_dates
)
from app.config import EMA_WINDOWS
from app.services.news_fetcher import fetch_and_compute_sentiment

logger = logging.getLogger(__name__)

# Cross-day decay parameter (matches Colab training)
SENTIMENT_DECAY_LAMBDA = 0.3


class SentimentService:
    """
    Service for managing sentiment data.
    
    Handles:
    - Adding daily sentiment
    - Retrieving sentiment history
    - Computing cross-day decay (matching Colab LAMBDA=0.3)
    - Computing EMA features
    - Applying lag (sentiment shifted by 1 day)
    """
    
    def __init__(self):
        """Initialize the sentiment service."""
        self.decay_lambda = SENTIMENT_DECAY_LAMBDA
    
    def add_daily_sentiment(
        self,
        date_str: str,
        daily_sentiment: float,
        news_volume: int,
        log_news_volume: float,
        decayed_news_volume: float,
        high_news_regime: int
    ) -> Dict[str, Any]:
        """
        Add sentiment data for a specific date.
        
        The daily_sentiment should be the simple mean from news_fetcher.
        Cross-day decay will be computed when retrieving features.
        
        Args:
            date_str: Date in YYYY-MM-DD format
            daily_sentiment: Simple mean of article sentiments (NOT yet decayed)
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
        
        # Store the raw daily_sentiment (not decayed yet)
        # The database column is still named 'daily_sentiment_decay' for compatibility
        # but we store the raw value here. Decay is applied at retrieval time.
        add_sentiment(
            date_str=date_str,
            daily_sentiment_decay=daily_sentiment,  # Raw sentiment, decay applied later
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
    
    def apply_cross_day_decay(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply cross-day exponential decay to sentiment scores.
        
        This EXACTLY matches the Colab training formula:
            LAMBDA = 0.3
            decayed[t] = sentiment[t] + exp(-LAMBDA) * decayed[t-1]
        
        Args:
            df: DataFrame with 'daily_sentiment_decay' column (raw daily means)
        
        Returns:
            DataFrame with decayed sentiment applied
        """
        if df.empty or 'daily_sentiment_decay' not in df.columns:
            return df
        
        result = df.copy()
        
        # Sort by date to ensure correct order
        if 'date' in result.columns:
            result = result.sort_values('date').reset_index(drop=True)
        
        # Get raw sentiment values
        raw_sentiment = result['daily_sentiment_decay'].values.astype(float)
        
        # Apply exponential decay (matching Colab exactly)
        # decayed[t] = sentiment[t] + exp(-LAMBDA) * decayed[t-1]
        decayed = np.zeros(len(raw_sentiment))
        decay_factor = np.exp(-self.decay_lambda)  # exp(-0.3) ≈ 0.7408
        
        for t in range(len(raw_sentiment)):
            if t == 0:
                decayed[t] = raw_sentiment[t]
            else:
                decayed[t] = raw_sentiment[t] + decay_factor * decayed[t - 1]
        
        # Replace with decayed values
        result['daily_sentiment_decay'] = decayed
        
        logger.debug(f"Applied cross-day decay (lambda={self.decay_lambda})")
        
        return result
    
    def get_sentiment_window(self, days: int = 60, end_date: pd.Timestamp = None) -> pd.DataFrame:
        """
        Get sentiment history for feature engineering.
        
        Args:
            days: Number of days of history
            end_date: Optional end date (defaults to today if None)
        
        Returns:
            DataFrame with sentiment data
        """
        if end_date is not None:
            # Use date-range query for backtesting
            from datetime import timedelta
            start_date = (end_date - timedelta(days=days)).strftime("%Y-%m-%d")
            end_date_str = end_date.strftime("%Y-%m-%d")
            df = get_sentiment_for_dates(start_date, end_date_str)
        else:
            df = get_sentiment_history(days=days)
        
        if df.empty:
            logger.warning("No sentiment data available, using zeros")
            return pd.DataFrame()
        
        # Return raw sentiment data (decay already applied during FinBERT processing)
        return df
    
    def compute_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all sentiment features including EMAs.
        
        The model expects:
        - Base: daily_sentiment_decay, news_volume, log_news_volume, 
                decayed_news_volume, high_news_regime
        - EMAs: *_ema_3, *_ema_7, *_ema_14 for 4 base columns
        
        Args:
            df: DataFrame with base sentiment columns (decay already applied)
        
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
        
        # Convert to just date (no time/timezone) for robust merging
        price_df['date'] = pd.to_datetime(price_df['date']).dt.date
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date']).dt.date
        
        # Shift sentiment by 1 day (today's prediction uses yesterday's sentiment)
        # This is done by shifting the date forward by 1 day in sentiment_df
        sentiment_df['date'] = sentiment_df['date'] + pd.Timedelta(days=1)
        
        # Filter sentiment to match price dates (plus buffer for EMA)
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

    def ensure_history(self, dates: List[Any]) -> pd.DataFrame:
        """
        Ensure sentiment history exists for the given dates.
        Fetches and computes if missing.
        
        Args:
            dates: List of dates (datetime or string)
            
        Returns:
            DataFrame with sentiment for those dates
        """
        if not dates:
            return pd.DataFrame()
            
        # Convert all to date objects for comparison
        target_dates = [pd.to_datetime(d).date() for d in dates]
        start_date = min(target_dates)
        end_date = max(target_dates)
        
        # Fetch existing from DB
        existing_df = get_sentiment_for_dates(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        existing_dates = set()
        if not existing_df.empty:
            existing_dates = set(pd.to_datetime(existing_df['date']).dt.date)
            
        # Identify missing dates
        missing_dates = [d for d in target_dates if d not in existing_dates]
        
        if missing_dates:
            logger.info(f"Filling sentiment history for {len(missing_dates)} missing dates")
            for d in missing_dates:
                date_str = d.strftime("%Y-%m-%d")
                try:
                    logger.info(f"Auto-fetching sentiment for {date_str}...")
                    result = fetch_and_compute_sentiment(date=date_str)
                    if result:
                        self.add_daily_sentiment(
                            date_str=date_str,
                            daily_sentiment=result['daily_sentiment_decay'],
                            news_volume=result['news_volume'],
                            log_news_volume=result['log_news_volume'],
                            decayed_news_volume=result['decayed_news_volume'],
                            high_news_regime=result['high_news_regime']
                        )
                except Exception as e:
                    logger.error(f"Failed to auto-fetch sentiment for {date_str}: {e}")
            
            # Re-fetch from DB after filling
            existing_df = get_sentiment_for_dates(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
            
        return existing_df
    
    def get_latest_info(self) -> Dict[str, Any]:
        """Get information about latest sentiment data."""
        latest = get_latest_sentiment()
        count = get_sentiment_count()
        
        return {
            "total_records": count,
            "latest_date": latest['date'] if latest else None,
            "latest_sentiment": latest
        }
    
    def compute_single_day_decayed_sentiment(
        self,
        current_sentiment: float,
        previous_decayed: Optional[float] = None
    ) -> float:
        """
        Compute decayed sentiment for a single day.
        
        Useful for real-time updates without reprocessing entire history.
        
        Formula: decayed[t] = sentiment[t] + exp(-LAMBDA) * decayed[t-1]
        
        Args:
            current_sentiment: Today's raw daily sentiment (simple mean)
            previous_decayed: Yesterday's decayed sentiment value
        
        Returns:
            Today's decayed sentiment value
        """
        if previous_decayed is None:
            return current_sentiment
        
        decay_factor = np.exp(-self.decay_lambda)  # exp(-0.3) ≈ 0.7408
        return current_sentiment + decay_factor * previous_decayed


# Global singleton
sentiment_service = SentimentService()
