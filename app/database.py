"""
Database layer for sentiment storage using SQLite.

NOTE: The database stores raw daily_sentiment (simple mean).
Cross-day decay is applied at retrieval time by sentiment_service.py
"""
import sqlite3
from pathlib import Path
from datetime import datetime, date
from typing import List, Dict, Optional, Any
import pandas as pd
import logging

from app.config import BASE_DIR

logger = logging.getLogger(__name__)

# Database file location
DB_PATH = BASE_DIR / "sentiment_history.db"


def get_connection() -> sqlite3.Connection:
    """Get database connection."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_database() -> None:
    """Initialize the sentiment database with required tables."""
    logger.info(f"Initializing sentiment database at {DB_PATH}")
    
    conn = get_connection()
    cursor = conn.cursor()
    
    # Create sentiment history table
    # Note: daily_sentiment_decay column stores the RAW daily sentiment
    # (simple mean of article scores). Cross-day decay is applied at read time.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sentiment_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT UNIQUE NOT NULL,
            daily_sentiment_decay REAL NOT NULL,
            news_volume INTEGER NOT NULL,
            log_news_volume REAL NOT NULL,
            decayed_news_volume REAL NOT NULL,
            high_news_regime INTEGER NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create index on date for faster queries
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_sentiment_date ON sentiment_history(date)
    """)
    
    conn.commit()
    conn.close()
    logger.info("Sentiment database initialized successfully")


def add_sentiment(
    date_str: str,
    daily_sentiment_decay: float,
    news_volume: int,
    log_news_volume: float,
    decayed_news_volume: float,
    high_news_regime: int
) -> bool:
    """
    Add or update sentiment data for a specific date.
    
    Args:
        date_str: Date in YYYY-MM-DD format
        daily_sentiment_decay: Raw daily sentiment (simple mean, no cross-day decay)
        news_volume: Number of news articles
        log_news_volume: Log-transformed volume
        decayed_news_volume: EWM-based volume estimate
        high_news_regime: Binary flag (0 or 1)
    
    Returns:
        True if successful
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            INSERT OR REPLACE INTO sentiment_history 
            (date, daily_sentiment_decay, news_volume, log_news_volume, 
             decayed_news_volume, high_news_regime)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (date_str, daily_sentiment_decay, news_volume, log_news_volume,
              decayed_news_volume, high_news_regime))
        
        conn.commit()
        logger.info(f"Added sentiment for {date_str}: {daily_sentiment_decay:.4f}")
        return True
    
    except Exception as e:
        logger.error(f"Error adding sentiment: {e}")
        conn.rollback()
        raise
    
    finally:
        conn.close()


def add_bulk_sentiment(sentiment_list: List[Dict[str, Any]]) -> int:
    """
    Add multiple sentiment records at once.
    
    Args:
        sentiment_list: List of sentiment dictionaries
    
    Returns:
        Number of records added
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    count = 0
    try:
        for record in sentiment_list:
            # Support both 'daily_sentiment' and 'daily_sentiment_decay' keys
            sentiment_value = record.get('daily_sentiment_decay', 
                                        record.get('daily_sentiment', 0.0))
            
            cursor.execute("""
                INSERT OR REPLACE INTO sentiment_history 
                (date, daily_sentiment_decay, news_volume, log_news_volume, 
                 decayed_news_volume, high_news_regime)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                record['date'],
                sentiment_value,
                record['news_volume'],
                record['log_news_volume'],
                record['decayed_news_volume'],
                record['high_news_regime']
            ))
            count += 1
        
        conn.commit()
        logger.info(f"Added {count} sentiment records")
        return count
    
    except Exception as e:
        logger.error(f"Error in bulk add: {e}")
        conn.rollback()
        raise
    
    finally:
        conn.close()


def get_sentiment_history(days: int = 60) -> pd.DataFrame:
    """
    Get sentiment history for the last N days.
    
    Note: Returns column as 'daily_sentiment' for feature_engineering.py
    which will apply its own alpha decay.
    
    Args:
        days: Number of days of history to retrieve
    
    Returns:
        DataFrame with sentiment data (daily_sentiment column = raw daily mean)
    """
    conn = get_connection()
    
    # Rename column to daily_sentiment for feature_engineering.py compatibility
    query = """
        SELECT date, daily_sentiment_decay as daily_sentiment, news_volume, log_news_volume,
               decayed_news_volume, high_news_regime
        FROM sentiment_history
        ORDER BY date DESC
        LIMIT ?
    """
    
    df = pd.read_sql_query(query, conn, params=(days,))
    conn.close()
    
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
    
    return df


def get_sentiment_for_dates(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Get sentiment data for a specific date range.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    
    Returns:
        DataFrame with sentiment data
    """
    conn = get_connection()
    
    query = """
        SELECT date, daily_sentiment_decay as daily_sentiment, news_volume, log_news_volume,
               decayed_news_volume, high_news_regime
        FROM sentiment_history
        WHERE date >= ? AND date <= ?
        ORDER BY date
    """
    
    df = pd.read_sql_query(query, conn, params=(start_date, end_date))
    conn.close()
    
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
    
    return df


def get_latest_sentiment() -> Optional[Dict[str, Any]]:
    """Get the most recent sentiment record."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT date, daily_sentiment_decay, news_volume, log_news_volume,
               decayed_news_volume, high_news_regime
        FROM sentiment_history
        ORDER BY date DESC
        LIMIT 1
    """)
    
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return dict(row)
    return None


def get_sentiment_count() -> int:
    """Get total number of sentiment records."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM sentiment_history")
    count = cursor.fetchone()[0]
    conn.close()
    return count


def clear_sentiment_history() -> int:
    """
    Clear all sentiment records from the database.
    
    Returns:
        Number of records deleted
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM sentiment_history")
    count = cursor.fetchone()[0]
    
    cursor.execute("DELETE FROM sentiment_history")
    conn.commit()
    conn.close()
    
    logger.info(f"Cleared {count} sentiment records")
    return count
