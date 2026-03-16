"""
Database layer for storing sentiment, prices, news articles, and predictions.

NOTE: The sentiment_history table stores raw daily_sentiment (simple mean).
Cross-day decay is applied at retrieval time by sentiment_service.py

Database: Turso (libsql) — remote, persistent across deploys.
"""

import json
import os
from datetime import datetime, date
from typing import List, Dict, Optional, Any
import pandas as pd
import logging

try:
    import libsql_experimental as libsql
    _USE_EXPERIMENTAL_LIBSQL = True
except ModuleNotFoundError:
    import libsql_client
    _USE_EXPERIMENTAL_LIBSQL = False

logger = logging.getLogger(__name__)


class _LibsqlClientCursor:
    """Minimal DB-API-like cursor shim for libsql_client sync client."""

    def __init__(self, client):
        self._client = client
        self._rows: List[Any] = []
        self._row_index = 0
        self.description = None
        self.rowcount = 0
        self.lastrowid = None

    def execute(self, query: str, params=None):
        result = self._client.execute(query, params or [])

        self._rows = list(result) if result is not None else []
        self._row_index = 0

        columns = getattr(result, "columns", None)
        if columns:
            self.description = [(col,) for col in columns]
        else:
            self.description = None

        self.rowcount = int(getattr(result, "rows_affected", 0) or 0)
        self.lastrowid = getattr(result, "last_insert_rowid", None)
        return self

    def fetchone(self):
        if self._row_index >= len(self._rows):
            return None
        row = self._rows[self._row_index]
        self._row_index += 1
        return row

    def fetchall(self):
        if self._row_index >= len(self._rows):
            return []
        rows = self._rows[self._row_index :]
        self._row_index = len(self._rows)
        return rows


class _LibsqlClientConnection:
    """Minimal connection shim for libsql_client to match existing code paths."""

    def __init__(self, client):
        self._client = client

    def cursor(self):
        return _LibsqlClientCursor(self._client)

    def commit(self):
        # libsql_client executes statements immediately; no explicit commit needed.
        return None

    def rollback(self):
        # libsql_client has no transaction in this shim path.
        return None

    def close(self):
        close_fn = getattr(self._client, "close", None)
        if callable(close_fn):
            close_fn()


def get_connection():
    """Get Turso (libsql) database connection."""
    url = os.environ.get("TURSO_DATABASE_URL")
    auth_token = os.environ.get("TURSO_AUTH_TOKEN", "")

    if _USE_EXPERIMENTAL_LIBSQL:
        return libsql.connect(database=url, auth_token=auth_token)

    if url and url.startswith("libsql://"):
        url = url.replace("libsql://", "https://", 1)

    logger.warning(
        "libsql_experimental unavailable; using libsql_client sync fallback"
    )
    client = libsql_client.create_client_sync(url=url, auth_token=auth_token)
    return _LibsqlClientConnection(client)


def _fetchone_dict(cursor) -> Optional[Dict[str, Any]]:
    """Fetch one row as a dict using cursor column names."""
    if cursor.description is None:
        return None
    cols = [d[0] for d in cursor.description]
    row = cursor.fetchone()
    return dict(zip(cols, row)) if row else None


def _fetchall_dicts(cursor) -> List[Dict[str, Any]]:
    """Fetch all rows as dicts using cursor column names."""
    if cursor.description is None:
        return []
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]


def _query_to_df(conn, query: str, params=None) -> pd.DataFrame:
    """Execute a SELECT query and return results as a DataFrame."""
    cursor = conn.cursor()
    cursor.execute(query, params or [])
    if cursor.description is None:
        return pd.DataFrame()
    cols = [d[0] for d in cursor.description]
    return pd.DataFrame(cursor.fetchall(), columns=cols)


def init_database() -> None:
    """Initialize the sentiment database with required tables."""
    logger.info("Initializing Turso database")

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

    # Create prices table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT UNIQUE NOT NULL,
            price REAL NOT NULL,
            source TEXT DEFAULT 'yahoo_finance',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_prices_date ON prices(date)
    """)

    # Create news_articles table (one row per article, with per-article sentiment)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS news_articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            article_date TEXT NOT NULL,
            title TEXT NOT NULL,
            description TEXT,
            url TEXT UNIQUE,
            source TEXT,
            published_at TEXT,
            sentiment_score REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_articles_date ON news_articles(article_date)
    """)

    # Create predictions table (stores each 14-day forecast run)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            generated_at TEXT NOT NULL,
            last_price_date TEXT NOT NULL,
            last_price REAL NOT NULL,
            forecasts TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()
    logger.info("Database initialized successfully (sentiment, prices, articles, predictions)")


def add_sentiment(
    date_str: str,
    daily_sentiment_decay: float,
    news_volume: int,
    log_news_volume: float,
    decayed_news_volume: float,
    high_news_regime: int,
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
        cursor.execute(
            """
            INSERT OR REPLACE INTO sentiment_history 
            (date, daily_sentiment_decay, news_volume, log_news_volume, 
             decayed_news_volume, high_news_regime)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                date_str,
                daily_sentiment_decay,
                news_volume,
                log_news_volume,
                decayed_news_volume,
                high_news_regime,
            ),
        )

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
            sentiment_value = record.get(
                "daily_sentiment_decay", record.get("daily_sentiment", 0.0)
            )

            cursor.execute(
                """
                INSERT OR REPLACE INTO sentiment_history 
                (date, daily_sentiment_decay, news_volume, log_news_volume, 
                 decayed_news_volume, high_news_regime)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    record["date"],
                    sentiment_value,
                    record["news_volume"],
                    record["log_news_volume"],
                    record["decayed_news_volume"],
                    record["high_news_regime"],
                ),
            )
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

    df = _query_to_df(conn, query, params=(days,))
    conn.close()

    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

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

    df = _query_to_df(conn, query, params=(start_date, end_date))
    conn.close()

    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])

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

    result = _fetchone_dict(cursor)
    conn.close()
    return result


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


# ---------------------------------------------------------------------------
# Price functions
# ---------------------------------------------------------------------------


def add_price(date_str: str, price: float, source: str = "yahoo_finance") -> bool:
    """Insert or replace a single daily price record."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT OR REPLACE INTO prices (date, price, source) VALUES (?, ?, ?)",
            (date_str, price, source),
        )
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Error adding price for {date_str}: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


def add_bulk_prices(price_records: List[Dict[str, Any]]) -> int:
    """
    Insert or replace multiple price records.

    Each record must have 'date' and 'price' keys.
    Optional 'source' key (defaults to 'yahoo_finance').
    """
    conn = get_connection()
    cursor = conn.cursor()
    count = 0
    try:
        for rec in price_records:
            cursor.execute(
                "INSERT OR REPLACE INTO prices (date, price, source) VALUES (?, ?, ?)",
                (rec["date"], rec["price"], rec.get("source", "yahoo_finance")),
            )
            count += 1
        conn.commit()
        logger.info(f"Saved {count} price records")
        return count
    except Exception as e:
        logger.error(f"Error in bulk add prices: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


def get_prices(days: int = 90) -> pd.DataFrame:
    """Return the most recent N days of stored price data."""
    conn = get_connection()
    df = _query_to_df(
        conn,
        "SELECT date, price, source FROM prices ORDER BY date DESC LIMIT ?",
        params=(days,),
    )
    conn.close()
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# News article functions
# ---------------------------------------------------------------------------


def add_news_articles(article_date: str, articles: List[Dict[str, Any]]) -> int:
    """
    Store a list of news articles for a given date.

    Each article dict should contain:
        title, description, url, source, published_at, sentiment_score
    Duplicate URLs are silently ignored (INSERT OR IGNORE).

    Returns:
        Number of new rows inserted.
    """
    conn = get_connection()
    cursor = conn.cursor()
    count = 0
    try:
        for art in articles:
            cursor.execute(
                """
                INSERT OR IGNORE INTO news_articles
                    (article_date, title, description, url, source, published_at, sentiment_score)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    article_date,
                    art.get("title", ""),
                    art.get("description", ""),
                    art.get("url"),
                    art.get("source", ""),
                    art.get("published_at", ""),
                    art.get("sentiment_score"),
                ),
            )
            if cursor.rowcount:
                count += 1
        conn.commit()
        logger.info(f"Saved {count} new articles for {article_date}")
        return count
    except Exception as e:
        logger.error(f"Error saving articles for {article_date}: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


def get_news_articles(article_date: str) -> List[Dict[str, Any]]:
    """Return all stored articles for a specific date."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, article_date, title, description, url, source, published_at, sentiment_score, created_at
        FROM news_articles WHERE article_date = ? ORDER BY id
        """,
        (article_date,),
    )
    rows = _fetchall_dicts(cursor)
    conn.close()
    return rows


def get_recent_news_articles(days: int = 7) -> List[Dict[str, Any]]:
    """Return articles from the most recent N distinct dates."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, article_date, title, description, url, source, published_at, sentiment_score
        FROM news_articles
        WHERE article_date IN (
            SELECT DISTINCT article_date FROM news_articles ORDER BY article_date DESC LIMIT ?
        )
        ORDER BY article_date DESC, id
        """,
        (days,),
    )
    rows = _fetchall_dicts(cursor)
    conn.close()
    return rows


# ---------------------------------------------------------------------------
# Prediction functions
# ---------------------------------------------------------------------------


def add_prediction(
    generated_at: str,
    last_price_date: str,
    last_price: float,
    forecasts: List[Dict[str, Any]],
) -> int:
    """
    Persist a 14-day forecast run.

    Args:
        generated_at: ISO timestamp of when the prediction was made.
        last_price_date: Date string of the last known price used.
        last_price: Last known price value.
        forecasts: List of ForecastDay dicts (serialized as JSON).

    Returns:
        Row id of the inserted record.
    """
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO predictions (generated_at, last_price_date, last_price, forecasts)
            VALUES (?, ?, ?, ?)
            """,
            (generated_at, last_price_date, last_price, json.dumps(forecasts)),
        )
        row_id = cursor.lastrowid
        conn.commit()
        logger.info(f"Saved prediction run id={row_id} generated_at={generated_at}")
        return row_id
    except Exception as e:
        logger.error(f"Error saving prediction: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


def get_latest_prediction() -> Optional[Dict[str, Any]]:
    """Return the most recently stored prediction run."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM predictions ORDER BY id DESC LIMIT 1"
    )
    result = _fetchone_dict(cursor)
    conn.close()
    if result:
        result["forecasts"] = json.loads(result["forecasts"])
        return result
    return None


def get_prediction_history(limit: int = 10) -> List[Dict[str, Any]]:
    """Return the last N prediction runs (most recent first)."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
    )
    rows = []
    for rec in _fetchall_dicts(cursor):
        rec["forecasts"] = json.loads(rec["forecasts"])
        rows.append(rec)
    conn.close()
    return rows
