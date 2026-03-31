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
    get_sentiment_for_dates,
    get_news_articles,
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
        high_news_regime: int,
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
            high_news_regime=high_news_regime,
        )

        return {
            "success": True,
            "message": f"Sentiment added for {date_str}",
            "total_records": get_sentiment_count(),
        }

    def add_bulk_sentiment(
        self, sentiment_list: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
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
            "total_records": get_sentiment_count(),
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
        if df.empty or "daily_sentiment_decay" not in df.columns:
            return df

        result = df.copy()

        # Sort by date to ensure correct order
        if "date" in result.columns:
            result = result.sort_values("date").reset_index(drop=True)

        # Get raw sentiment values
        raw_sentiment = result["daily_sentiment_decay"].values.astype(float)

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
        result["daily_sentiment_decay"] = decayed

        logger.debug(f"Applied cross-day decay (lambda={self.decay_lambda})")

        return result

    def get_sentiment_window(
        self, days: int = 60, end_date: pd.Timestamp = None
    ) -> pd.DataFrame:
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
            "daily_sentiment_decay",
            "news_volume",
            "log_news_volume",
            "decayed_news_volume",
        ]

        # Compute EMAs
        for col in ema_cols:
            if col in result.columns:
                for window in EMA_WINDOWS:
                    result[f"{col}_ema_{window}"] = (
                        result[col].ewm(span=window, adjust=False).mean()
                    )

        return result

    def merge_with_prices(
        self, price_df: pd.DataFrame, sentiment_df: pd.DataFrame
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
        price_df["date"] = pd.to_datetime(price_df["date"]).dt.date
        sentiment_df["date"] = pd.to_datetime(sentiment_df["date"]).dt.date

        # Shift sentiment by 1 day (today's prediction uses yesterday's sentiment)
        # This is done by shifting the date forward by 1 day in sentiment_df
        sentiment_df["date"] = sentiment_df["date"] + pd.Timedelta(days=1)

        # Filter sentiment to match price dates (plus buffer for EMA)
        # Merge on date
        merged = price_df.merge(sentiment_df, on="date", how="left")

        # Fill missing sentiment with 0 (matches training behavior)
        sentiment_cols = [
            "daily_sentiment_decay",
            "news_volume",
            "log_news_volume",
            "decayed_news_volume",
            "high_news_regime",
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
        existing_df = get_sentiment_for_dates(
            start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
        )
        existing_dates = set()
        if not existing_df.empty:
            existing_dates = set(pd.to_datetime(existing_df["date"]).dt.date)

        # Identify missing dates
        missing_dates = [d for d in target_dates if d not in existing_dates]

        if missing_dates:
            logger.info(
                f"Filling sentiment history for {len(missing_dates)} missing dates"
            )
            for d in missing_dates:
                date_str = d.strftime("%Y-%m-%d")
                try:
                    logger.info(f"Auto-fetching sentiment for {date_str}...")
                    result = fetch_and_compute_sentiment(date=date_str)
                    if result:
                        self.add_daily_sentiment(
                            date_str=date_str,
                            daily_sentiment=result["daily_sentiment_decay"],
                            news_volume=result["news_volume"],
                            log_news_volume=result["log_news_volume"],
                            decayed_news_volume=result["decayed_news_volume"],
                            high_news_regime=result["high_news_regime"],
                        )
                except Exception as e:
                    logger.error(f"Failed to auto-fetch sentiment for {date_str}: {e}")

            # Re-fetch from DB after filling
            existing_df = get_sentiment_for_dates(
                start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
            )

        return existing_df

    def get_latest_info(self) -> Dict[str, Any]:
        """Get information about latest sentiment data."""
        latest = get_latest_sentiment()
        count = get_sentiment_count()

        return {
            "total_records": count,
            "latest_date": latest["date"] if latest else None,
            "latest_sentiment": latest,
        }

    def _sentiment_meta(
        self,
        days: int,
        records: int,
        start: Optional[str],
        end: Optional[str],
    ) -> Dict[str, Any]:
        decay_factor = float(np.exp(-self.decay_lambda))
        return {
            "requested_days": days,
            "actual_records": records,
            "start_date": start,
            "end_date": end,
            "decay_lambda": self.decay_lambda,
            "decay_factor": decay_factor,
            "decay_formula": (
                "decayed[t] = raw_sentiment[t] + exp(-lambda) * decayed[t-1]"
            ),
            "ema_windows": EMA_WINDOWS,
        }

    def _empty_overview_payload(self, days: int) -> Dict[str, Any]:
        return {
            "success": True,
            "meta": self._sentiment_meta(days, 0, None, None),
            "summary": {
                "latest_raw_sentiment": None,
                "latest_decayed_sentiment": None,
                "average_raw_sentiment": None,
                "average_decayed_sentiment": None,
                "average_news_volume": None,
                "high_news_regime_days": 0,
                "positive_days": 0,
                "negative_days": 0,
                "neutral_days": 0,
                "latest_trend": "neutral",
            },
            "timeline": [],
        }

    def _load_sentiment_df(self, days: int, end_date: Optional[str]) -> pd.DataFrame:
        if end_date:
            parsed_end = pd.to_datetime(end_date)
            start_date = (parsed_end - timedelta(days=days - 1)).strftime("%Y-%m-%d")
            end_date_str = parsed_end.strftime("%Y-%m-%d")
            return get_sentiment_for_dates(start_date, end_date_str)
        return get_sentiment_history(days=days)

    def _build_ema_map(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        ema_df = df[
            [
                "date",
                "cross_day_decayed_sentiment",
                "news_volume",
                "log_news_volume",
                "decayed_news_volume",
            ]
        ].copy()
        ema_df = ema_df.rename(
            columns={"cross_day_decayed_sentiment": "daily_sentiment_decay"}
        )
        ema_df = self.compute_sentiment_features(ema_df)

        ema_map: Dict[str, Dict[str, float]] = {}
        for _, row in ema_df.iterrows():
            date_key = pd.to_datetime(row["date"]).strftime("%Y-%m-%d")
            payload: Dict[str, float] = {}
            for window in EMA_WINDOWS:
                payload[f"daily_sentiment_decay_ema_{window}"] = float(
                    row.get(f"daily_sentiment_decay_ema_{window}", 0.0)
                )
                payload[f"news_volume_ema_{window}"] = float(
                    row.get(f"news_volume_ema_{window}", 0.0)
                )
                payload[f"log_news_volume_ema_{window}"] = float(
                    row.get(f"log_news_volume_ema_{window}", 0.0)
                )
                payload[f"decayed_news_volume_ema_{window}"] = float(
                    row.get(f"decayed_news_volume_ema_{window}", 0.0)
                )
            ema_map[date_key] = payload

        return ema_map

    @staticmethod
    def _ranked_headlines(date_key: str, headlines_per_day: int) -> List[Dict[str, Any]]:
        articles = get_news_articles(date_key)
        ranked = sorted(
            articles,
            key=lambda art: abs(float(art.get("sentiment_score") or 0.0)),
            reverse=True,
        )
        return [
            {
                "title": article.get("title") or "",
                "source": article.get("source"),
                "sentiment_score": article.get("sentiment_score"),
                "published_at": article.get("published_at"),
                "url": article.get("url"),
            }
            for article in ranked[:headlines_per_day]
        ]

    @staticmethod
    def _trend_label(decayed_series: pd.Series) -> str:
        slope_window = min(5, len(decayed_series))
        recent = decayed_series.tail(slope_window)
        signal = float(recent.iloc[-1] - recent.iloc[0]) if len(recent) > 1 else 0.0
        if signal > 0.05:
            return "bullish"
        if signal < -0.05:
            return "bearish"
        return "neutral"

    def get_frontend_sentiment_overview(
        self,
        days: int = 60,
        end_date: Optional[str] = None,
        include_headlines: bool = True,
        headlines_per_day: int = 3,
    ) -> Dict[str, Any]:
        """Build a frontend-oriented sentiment payload with decay analytics."""
        if days < 1:
            raise ValueError("days must be >= 1")
        if headlines_per_day < 1:
            raise ValueError("headlines_per_day must be >= 1")

        raw_df = self._load_sentiment_df(days, end_date)
        if raw_df.empty:
            return self._empty_overview_payload(days)

        df = raw_df.copy().sort_values("date").reset_index(drop=True)
        if "daily_sentiment" not in df.columns:
            raise ValueError("Sentiment data does not contain daily_sentiment column")

        df["raw_daily_sentiment"] = df["daily_sentiment"].astype(float)
        decay_factor = float(np.exp(-self.decay_lambda))
        decayed_values: List[float] = []
        for idx, raw_value in enumerate(df["raw_daily_sentiment"].tolist()):
            if idx == 0:
                decayed_values.append(float(raw_value))
            else:
                decayed_values.append(float(raw_value) + decay_factor * decayed_values[-1])
        df["cross_day_decayed_sentiment"] = decayed_values
        df["sentiment_change_vs_prev_day"] = (
            df["raw_daily_sentiment"].diff().fillna(0.0)
        )
        df["decayed_sentiment_change_vs_prev_day"] = (
            df["cross_day_decayed_sentiment"].diff().fillna(0.0)
        )

        ema_map = self._build_ema_map(df)

        timeline: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            date_key = pd.to_datetime(row["date"]).strftime("%Y-%m-%d")
            ema_payload = ema_map.get(date_key, {})
            headlines = (
                self._ranked_headlines(date_key, headlines_per_day)
                if include_headlines
                else []
            )

            timeline.append(
                {
                    "date": date_key,
                    "raw_daily_sentiment": float(row["raw_daily_sentiment"]),
                    "cross_day_decayed_sentiment": float(
                        row["cross_day_decayed_sentiment"]
                    ),
                    "sentiment_change_vs_prev_day": float(
                        row["sentiment_change_vs_prev_day"]
                    ),
                    "decayed_sentiment_change_vs_prev_day": float(
                        row["decayed_sentiment_change_vs_prev_day"]
                    ),
                    "news_volume": int(row["news_volume"]),
                    "log_news_volume": float(row["log_news_volume"]),
                    "decayed_news_volume": float(row["decayed_news_volume"]),
                    "high_news_regime": bool(int(row["high_news_regime"])),
                    "ema": ema_payload,
                    "headlines": headlines,
                }
            )

        latest_trend = self._trend_label(df["cross_day_decayed_sentiment"])

        positive_days = int((df["raw_daily_sentiment"] > 0.05).sum())
        negative_days = int((df["raw_daily_sentiment"] < -0.05).sum())
        neutral_days = int(len(df) - positive_days - negative_days)

        return {
            "success": True,
            "meta": self._sentiment_meta(
                days,
                int(len(df)),
                pd.to_datetime(df["date"].iloc[0]).strftime("%Y-%m-%d"),
                pd.to_datetime(df["date"].iloc[-1]).strftime("%Y-%m-%d"),
            ),
            "summary": {
                "latest_raw_sentiment": float(df["raw_daily_sentiment"].iloc[-1]),
                "latest_decayed_sentiment": float(
                    df["cross_day_decayed_sentiment"].iloc[-1]
                ),
                "average_raw_sentiment": float(df["raw_daily_sentiment"].mean()),
                "average_decayed_sentiment": float(
                    df["cross_day_decayed_sentiment"].mean()
                ),
                "average_news_volume": float(df["news_volume"].mean()),
                "high_news_regime_days": int(df["high_news_regime"].sum()),
                "positive_days": positive_days,
                "negative_days": negative_days,
                "neutral_days": neutral_days,
                "latest_trend": latest_trend,
            },
            "timeline": timeline,
        }

    def apply_no_news_decay(self, date_str: str) -> Dict[str, Any]:
        """
        Apply sentiment decay for days with no news articles.

        Instead of storing 0.0, retrieves the previous day's raw sentiment
        and decays it: decayed = exp(-LAMBDA) * previous_sentiment.
        This prevents abrupt sentiment drops on no-news days.

        Args:
            date_str: Date in YYYY-MM-DD format

        Returns:
            Dict with the decayed sentiment value stored
        """
        decay_factor = np.exp(-self.decay_lambda)  # exp(-0.3) ≈ 0.7408

        # Find previous day's sentiment
        prev_date = (pd.to_datetime(date_str) - pd.Timedelta(days=1)).strftime(
            "%Y-%m-%d"
        )
        prev_df = get_sentiment_for_dates(prev_date, prev_date)

        if not prev_df.empty:
            prev_sentiment = float(prev_df["daily_sentiment"].iloc[0])
            decayed_sentiment = decay_factor * prev_sentiment
        else:
            # No previous data either — use 0.0
            decayed_sentiment = 0.0

        # Store with zero news volume
        add_sentiment(
            date_str=date_str,
            daily_sentiment_decay=decayed_sentiment,
            news_volume=0,
            log_news_volume=0.0,
            decayed_news_volume=0.0,
            high_news_regime=0,
        )

        logger.info(
            f"No-news decay for {date_str}: previous={prev_date}, "
            f"decayed_sentiment={decayed_sentiment:.4f}"
        )

        return {
            "date": date_str,
            "decayed_sentiment": decayed_sentiment,
            "previous_date": prev_date,
            "decay_factor": decay_factor,
        }

    def compute_single_day_decayed_sentiment(
        self, current_sentiment: float, previous_decayed: Optional[float] = None
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
