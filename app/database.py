"""
Database layer for storing sentiment, prices, news articles, and predictions.

NOTE: The sentiment_history table stores raw daily_sentiment (simple mean).
Cross-day decay is applied at retrieval time by sentiment_service.py

Database: Turso (libsql) — remote, persistent across deploys.
"""

import json
import os
import math
import re
from collections import defaultdict
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

DATE_BETWEEN_CLAUSE = " WHERE date >= ? AND date <= ?"
DATE_FROM_CLAUSE = " WHERE date >= ?"
DATE_TO_CLAUSE = " WHERE date <= ?"


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
        result = self._client.execute(query, params or ())

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
    cursor.execute(query, params or ())
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

    # Create historical prices table (dataset imports)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS historical_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT UNIQUE NOT NULL,
            price REAL NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            volume REAL,
            change_pct REAL,
            source TEXT DEFAULT 'historical_import',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_historical_prices_date ON historical_prices(date)
    """)

    # Create historical news features table (dataset imports)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS historical_news_features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT UNIQUE NOT NULL,
            daily_sentiment_decay REAL NOT NULL,
            news_volume REAL NOT NULL,
            log_news_volume REAL NOT NULL,
            decayed_news_volume REAL NOT NULL,
            high_news_regime INTEGER NOT NULL,
            source TEXT DEFAULT 'historical_import',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_historical_news_features_date ON historical_news_features(date)
    """)

    # Convenience full-outer-date view across both historical tables.
    cursor.execute("""
        CREATE VIEW IF NOT EXISTS historical_features_combined AS
        SELECT
            hp.date AS date,
            hp.price AS price,
            hp.open AS open,
            hp.high AS high,
            hp.low AS low,
            hp.volume AS volume,
            hp.change_pct AS change_pct,
            hnf.daily_sentiment_decay AS daily_sentiment_decay,
            hnf.news_volume AS news_volume,
            hnf.log_news_volume AS log_news_volume,
            hnf.decayed_news_volume AS decayed_news_volume,
            hnf.high_news_regime AS high_news_regime
        FROM historical_prices hp
        LEFT JOIN historical_news_features hnf ON hp.date = hnf.date
        UNION ALL
        SELECT
            hnf.date AS date,
            hp.price AS price,
            hp.open AS open,
            hp.high AS high,
            hp.low AS low,
            hp.volume AS volume,
            hp.change_pct AS change_pct,
            hnf.daily_sentiment_decay AS daily_sentiment_decay,
            hnf.news_volume AS news_volume,
            hnf.log_news_volume AS log_news_volume,
            hnf.decayed_news_volume AS decayed_news_volume,
            hnf.high_news_regime AS high_news_regime
        FROM historical_news_features hnf
        LEFT JOIN historical_prices hp ON hp.date = hnf.date
        WHERE hp.date IS NULL
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


def _to_float(value: Any) -> Optional[float]:
    """Convert mixed numeric text formats (%, K/M/B suffixes, commas) to float."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        numeric_value = float(value)
        return numeric_value if math.isfinite(numeric_value) else None

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "-"}:
        return None

    multiplier = 1.0
    suffix = text[-1].upper()
    if suffix in {"K", "M", "B"}:
        text = text[:-1]
        multiplier = {"K": 1_000.0, "M": 1_000_000.0, "B": 1_000_000_000.0}[suffix]

    text = text.replace(",", "").replace("%", "")
    text = re.sub(r"[^0-9.+-]", "", text)

    if not text:
        return None

    try:
        numeric_value = float(text) * multiplier
        return numeric_value if math.isfinite(numeric_value) else None
    except ValueError:
        return None


def add_bulk_historical_prices(
    price_records: List[Dict[str, Any]],
    default_source: str = "historical_import",
) -> int:
    """
    Insert or replace multiple historical price records.

    Required keys per record:
    - date
    - price
    Optional keys:
    - open, high, low, volume, change_pct, source
    """
    conn = get_connection()
    cursor = conn.cursor()
    count = 0
    try:
        chunk_size = 100
        for i in range(0, len(price_records), chunk_size):
            chunk = price_records[i : i + chunk_size]
            values = []
            for rec in chunk:
                values.extend(
                    [
                        rec["date"],
                        _to_float(rec["price"]),
                        _to_float(rec.get("open")),
                        _to_float(rec.get("high")),
                        _to_float(rec.get("low")),
                        _to_float(rec.get("volume")),
                        _to_float(rec.get("change_pct")),
                        rec.get("source", default_source),
                    ]
                )

            placeholders = ", ".join(["(?, ?, ?, ?, ?, ?, ?, ?)"] * len(chunk))
            query = (
                "INSERT OR REPLACE INTO historical_prices "
                "(date, price, open, high, low, volume, change_pct, source) "
                f"VALUES {placeholders}"
            )
            cursor.execute(query, tuple(values))
            count += len(chunk)

        conn.commit()
        logger.info(f"Saved {count} historical price records")
        return count
    except Exception as e:
        logger.error(f"Error in bulk add historical prices: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


def add_bulk_historical_news_features(
    feature_records: List[Dict[str, Any]],
    default_source: str = "historical_import",
) -> int:
    """
    Insert or replace multiple historical news feature rows.

    Required keys per record:
    - date, daily_sentiment_decay, news_volume, log_news_volume,
      decayed_news_volume, high_news_regime
    """
    conn = get_connection()
    cursor = conn.cursor()
    count = 0
    try:
        chunk_size = 100
        for i in range(0, len(feature_records), chunk_size):
            chunk = feature_records[i : i + chunk_size]
            values = []
            for rec in chunk:
                values.extend(
                    [
                        rec["date"],
                        _to_float(rec["daily_sentiment_decay"]),
                        _to_float(rec["news_volume"]),
                        _to_float(rec["log_news_volume"]),
                        _to_float(rec["decayed_news_volume"]),
                        int(float(rec["high_news_regime"])),
                        rec.get("source", default_source),
                    ]
                )

            placeholders = ", ".join(["(?, ?, ?, ?, ?, ?, ?)"] * len(chunk))
            query = (
                "INSERT OR REPLACE INTO historical_news_features "
                "(date, daily_sentiment_decay, news_volume, log_news_volume, "
                "decayed_news_volume, high_news_regime, source) "
                f"VALUES {placeholders}"
            )
            cursor.execute(query, tuple(values))
            count += len(chunk)

        conn.commit()
        logger.info(f"Saved {count} historical news feature records")
        return count
    except Exception as e:
        logger.error(f"Error in bulk add historical news features: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


def get_historical_features_combined(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: Optional[int] = None,
    offset: int = 0,
) -> pd.DataFrame:
    """Return joined historical price and news features over an optional date range."""
    conn = get_connection()

    base_query = """
        SELECT date, price, open, high, low, volume, change_pct,
               daily_sentiment_decay, news_volume, log_news_volume,
               decayed_news_volume, high_news_regime
        FROM historical_features_combined
    """
    params: List[Any] = []

    if start_date and end_date:
        base_query += DATE_BETWEEN_CLAUSE
        params.extend([start_date, end_date])
    elif start_date:
        base_query += DATE_FROM_CLAUSE
        params.append(start_date)
    elif end_date:
        base_query += DATE_TO_CLAUSE
        params.append(end_date)

    base_query += " ORDER BY date"

    if limit is not None:
        base_query += " LIMIT ? OFFSET ?"
        params.extend([limit, offset])

    df = _query_to_df(conn, base_query, tuple(params))
    conn.close()

    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])

    return df


def get_historical_features_combined_count(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> int:
    """Return total row count in combined historical view for an optional date range."""
    conn = get_connection()
    cursor = conn.cursor()

    query = "SELECT COUNT(*) FROM historical_features_combined"
    params: List[Any] = []
    if start_date and end_date:
        query += DATE_BETWEEN_CLAUSE
        params.extend([start_date, end_date])
    elif start_date:
        query += DATE_FROM_CLAUSE
        params.append(start_date)
    elif end_date:
        query += DATE_TO_CLAUSE
        params.append(end_date)

    cursor.execute(query, tuple(params))
    count = cursor.fetchone()[0]
    conn.close()
    return int(count)


def get_historical_prices(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: Optional[int] = None,
    offset: int = 0,
) -> pd.DataFrame:
    """Return historical price dataset over an optional date range."""
    conn = get_connection()

    base_query = """
        SELECT date, price, open, high, low, volume, change_pct, source
        FROM historical_prices
    """
    params: List[Any] = []

    if start_date and end_date:
        base_query += DATE_BETWEEN_CLAUSE
        params.extend([start_date, end_date])
    elif start_date:
        base_query += DATE_FROM_CLAUSE
        params.append(start_date)
    elif end_date:
        base_query += DATE_TO_CLAUSE
        params.append(end_date)

    base_query += " ORDER BY date"

    if limit is not None:
        base_query += " LIMIT ? OFFSET ?"
        params.extend([limit, offset])

    df = _query_to_df(conn, base_query, tuple(params))
    conn.close()

    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])

    return df


def get_historical_prices_count(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> int:
    """Return total historical price rows for an optional date range."""
    conn = get_connection()
    cursor = conn.cursor()

    query = "SELECT COUNT(*) FROM historical_prices"
    params: List[Any] = []
    if start_date and end_date:
        query += DATE_BETWEEN_CLAUSE
        params.extend([start_date, end_date])
    elif start_date:
        query += DATE_FROM_CLAUSE
        params.append(start_date)
    elif end_date:
        query += DATE_TO_CLAUSE
        params.append(end_date)

    cursor.execute(query, tuple(params))
    count = cursor.fetchone()[0]
    conn.close()
    return int(count)


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


def _compute_quantile(values: List[float], q: float) -> float:
    """Compute a quantile from a non-empty numeric list using linear interpolation."""
    if not values:
        return 0.0

    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])

    pos = (len(sorted_vals) - 1) * q
    lower_idx = int(math.floor(pos))
    upper_idx = int(math.ceil(pos))

    if lower_idx == upper_idx:
        return float(sorted_vals[lower_idx])

    lower_val = sorted_vals[lower_idx]
    upper_val = sorted_vals[upper_idx]
    weight = pos - lower_idx
    return float(lower_val + (upper_val - lower_val) * weight)


def _collect_relative_errors_by_horizon(
    prediction_runs: pd.DataFrame,
    actual_price_by_date: Dict[date, float],
) -> Dict[int, List[float]]:
    """Build empirical signed relative errors grouped by forecast horizon."""
    errors_by_horizon: Dict[int, List[float]] = defaultdict(list)

    if prediction_runs.empty:
        return errors_by_horizon

    for _, row in prediction_runs.iterrows():
        reference_date = _parse_prediction_reference_date(
            last_price_date_raw=row.get("last_price_date"),
            generated_at_raw=row.get("generated_at"),
        )
        if reference_date is None:
            continue

        forecasts = _parse_forecasts_blob(row.get("forecasts"))
        for forecast in forecasts:
            parsed = _parse_single_forecast_observation(
                forecast=forecast,
                reference_date=reference_date,
                generated_at_raw=row.get("generated_at"),
                cutoff_date=date.today(),
            )
            if parsed is None:
                continue

            target_date, pred_entry = parsed
            actual_price = actual_price_by_date.get(target_date)
            predicted_price = float(pred_entry.get("forecasted_price", 0.0))
            horizon = int(pred_entry.get("horizon", 14))

            if actual_price is None or predicted_price <= 0:
                continue

            rel_error = (actual_price - predicted_price) / predicted_price
            errors_by_horizon[horizon].append(float(rel_error))

    return errors_by_horizon


def _calibration_pool_for_horizon(
    errors_by_horizon: Dict[int, List[float]],
    horizon: int,
    min_samples: int,
) -> List[float]:
    """Get calibration samples for a horizon, widening to neighbors and global pool if needed."""
    direct = list(errors_by_horizon.get(horizon, []))
    if len(direct) >= min_samples:
        return direct

    pooled = list(direct)
    max_h = 14
    for radius in range(1, max_h):
        left = horizon - radius
        right = horizon + radius
        if left >= 1:
            pooled.extend(errors_by_horizon.get(left, []))
        if right <= max_h:
            pooled.extend(errors_by_horizon.get(right, []))
        if len(pooled) >= min_samples:
            return pooled

    global_pool: List[float] = []
    for h in range(1, max_h + 1):
        global_pool.extend(errors_by_horizon.get(h, []))

    if global_pool:
        return global_pool

    # Final deterministic fallback if no calibration data exists at all.
    return [-0.03, -0.015, 0.0, 0.015, 0.03]


def get_latest_prediction_fan_chart(min_samples_per_horizon: int = 20) -> Dict[str, Any]:
    """
    Return fan chart data for the latest prediction run using empirical error calibration.

    Uncertainty bands are derived from historical signed relative errors
    ((actual - predicted) / predicted), grouped by horizon.
    """
    latest = get_latest_prediction()
    if not latest:
        raise ValueError("No stored prediction runs available")

    conn = get_connection()
    try:
        prediction_runs = _query_to_df(
            conn,
            """
            SELECT generated_at, last_price_date, forecasts
            FROM predictions
            ORDER BY generated_at ASC
            """,
        )
        actual_prices_df = _query_to_df(
            conn,
            """
            SELECT date, price
            FROM prices
            ORDER BY date ASC
            """,
        )
    finally:
        conn.close()

    actual_price_by_date: Dict[date, float] = {}
    if not actual_prices_df.empty:
        actual_prices_df["date"] = pd.to_datetime(actual_prices_df["date"]).dt.date
        actual_price_by_date = {
            row["date"]: float(row["price"]) for _, row in actual_prices_df.iterrows()
        }

    errors_by_horizon = _collect_relative_errors_by_horizon(
        prediction_runs=prediction_runs,
        actual_price_by_date=actual_price_by_date,
    )

    fan_points: List[Dict[str, Any]] = []
    latest_forecasts = latest.get("forecasts", []) if isinstance(latest, dict) else []

    for item in latest_forecasts:
        date_str = str(item.get("date"))
        horizon = int(item.get("horizon", 14))
        point_forecast = float(item.get("forecasted_price", 0.0))

        samples = _calibration_pool_for_horizon(
            errors_by_horizon=errors_by_horizon,
            horizon=horizon,
            min_samples=max(1, int(min_samples_per_horizon)),
        )

        q10 = _compute_quantile(samples, 0.10)
        q25 = _compute_quantile(samples, 0.25)
        q50 = _compute_quantile(samples, 0.50)
        q75 = _compute_quantile(samples, 0.75)
        q90 = _compute_quantile(samples, 0.90)

        def _apply(q: float) -> float:
            return round(max(0.01, point_forecast * (1.0 + q)), 2)

        fan_points.append(
            {
                "date": date_str,
                "horizon": horizon,
                "point_forecast": round(point_forecast, 2),
                "p10": _apply(q10),
                "p25": _apply(q25),
                "p50": _apply(q50),
                "p75": _apply(q75),
                "p90": _apply(q90),
                "sample_count": len(samples),
            }
        )

    return {
        "generated_at": str(latest.get("generated_at", "")),
        "last_price_date": str(latest.get("last_price_date", "")),
        "last_price": float(latest.get("last_price", 0.0)),
        "calibration_method": (
            "Empirical horizon-wise quantiles from historical signed relative forecast "
            "errors; neighbor/global pooling fallback when samples are sparse."
        ),
        "fan": fan_points,
    }


def _empty_comparison_payload(cutoff_date: date) -> Dict[str, Any]:
    """Return a standard empty comparison payload."""
    return {
        "end_date": cutoff_date.strftime("%Y-%m-%d"),
        "rows": [],
        "metrics": {
            "compared_days": 0,
            "mae": None,
            "rmse": None,
            "mape": None,
        },
    }


def _parse_generated_date(generated_at_raw: Any) -> Optional[date]:
    """Parse generated_at timestamp into date, supporting trailing Z."""
    if not generated_at_raw:
        return None
    raw = str(generated_at_raw)
    try:
        return datetime.fromisoformat(raw).date()
    except ValueError:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).date()


def _collect_predictions_by_target_date(
    prediction_runs: pd.DataFrame, cutoff_date: date
) -> Dict[date, List[Dict[str, Any]]]:
    """Collect all valid prediction observations keyed by target date."""
    per_date_predictions: Dict[date, List[Dict[str, Any]]] = {}

    if prediction_runs.empty:
        return per_date_predictions

    for _, row in prediction_runs.iterrows():
        generated_at_raw = row.get("generated_at")
        reference_date = _parse_prediction_reference_date(
            last_price_date_raw=row.get("last_price_date"),
            generated_at_raw=generated_at_raw,
        )
        if reference_date is None:
            continue

        forecasts = _parse_forecasts_blob(row.get("forecasts"))
        for forecast in forecasts:
            parsed = _parse_single_forecast_observation(
                forecast=forecast,
                reference_date=reference_date,
                generated_at_raw=generated_at_raw,
                cutoff_date=cutoff_date,
            )
            if parsed is None:
                continue

            target_date, prediction_entry = parsed
            per_date_predictions.setdefault(target_date, []).append(prediction_entry)

    return per_date_predictions


def _parse_forecasts_blob(forecasts_blob: Any) -> List[Dict[str, Any]]:
    """Parse serialized forecasts JSON into a list."""
    try:
        parsed = json.loads(forecasts_blob or "[]")
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def _parse_single_forecast_observation(
    forecast: Dict[str, Any],
    reference_date: date,
    generated_at_raw: Any,
    cutoff_date: date,
) -> Optional[tuple]:
    """Parse and validate one forecast item into (target_date, entry)."""
    target_date_raw = forecast.get("date")
    pred_price_raw = forecast.get("forecasted_price")
    if target_date_raw is None or pred_price_raw is None:
        return None

    try:
        target_date = datetime.strptime(str(target_date_raw), "%Y-%m-%d").date()
        pred_price = float(pred_price_raw)
    except (ValueError, TypeError):
        return None

    if target_date > cutoff_date or reference_date > target_date:
        return None

    try:
        horizon = int(forecast.get("horizon"))
    except (TypeError, ValueError):
        horizon = 14

    return (
        target_date,
        {
            "forecasted_price": pred_price,
            "horizon": max(1, horizon),
            "generated_at": str(generated_at_raw),
        },
    )


def _parse_prediction_reference_date(
    last_price_date_raw: Any, generated_at_raw: Any
) -> Optional[date]:
    """
    Parse prediction reference date.

    Prefer last_price_date (the date the forecast starts from); fallback to generated_at.
    """
    if last_price_date_raw:
        try:
            return datetime.strptime(str(last_price_date_raw), "%Y-%m-%d").date()
        except ValueError:
            pass

    return _parse_generated_date(generated_at_raw)


def _aggregate_predictions(preds: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate multiple predictions for a date."""
    weights = [1.0 / float(p["horizon"]) for p in preds]
    weighted_sum = sum(p["forecasted_price"] * w for p, w in zip(preds, weights))
    total_weight = sum(weights)
    weighted_predicted = weighted_sum / total_weight if total_weight > 0 else 0.0

    sorted_prices = sorted(p["forecasted_price"] for p in preds)
    n = len(sorted_prices)
    if n % 2 == 1:
        median_predicted = sorted_prices[n // 2]
    else:
        median_predicted = (sorted_prices[n // 2 - 1] + sorted_prices[n // 2]) / 2.0

    latest_pred_rec = max(preds, key=lambda p: p.get("generated_at", ""))

    return {
        "weighted_predicted": weighted_predicted,
        "median_predicted": median_predicted,
        "latest_predicted": float(latest_pred_rec["forecasted_price"]),
        "prediction_count": len(preds),
    }


def _build_comparison_rows(
    actual_df: pd.DataFrame, per_date_predictions: Dict[date, List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """Build day-level comparison rows for dates that have both actual and predicted values."""
    rows: List[Dict[str, Any]] = []

    for _, rec in actual_df.iterrows():
        target_date = rec["date"]
        preds = per_date_predictions.get(target_date, [])
        if not preds:
            continue

        actual_price = float(rec["price"])
        aggregated = _aggregate_predictions(preds)

        weighted_predicted = aggregated["weighted_predicted"]
        error = actual_price - weighted_predicted
        abs_error = abs(error)
        pct_error = abs_error / actual_price * 100.0 if actual_price else None

        rows.append(
            {
                "date": target_date.strftime("%Y-%m-%d"),
                "actual_price": round(actual_price, 2),
                "predicted_price": round(weighted_predicted, 2),
                "predicted_price_median": round(aggregated["median_predicted"], 2),
                "predicted_price_latest": round(aggregated["latest_predicted"], 2),
                "prediction_count": aggregated["prediction_count"],
                "error": round(error, 2),
                "abs_error": round(abs_error, 2),
                "abs_pct_error": round(pct_error, 2) if pct_error is not None else None,
            }
        )

    return rows


def _compute_comparison_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute MAE, RMSE, and MAPE from comparison rows."""
    compared_days = len(rows)
    mae = sum(r["abs_error"] for r in rows) / compared_days
    rmse = math.sqrt(sum((r["error"] ** 2) for r in rows) / compared_days)

    mape_values = [r["abs_pct_error"] for r in rows if r["abs_pct_error"] is not None]
    mape = sum(mape_values) / len(mape_values) if mape_values else None

    return {
        "compared_days": compared_days,
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mape": round(mape, 4) if mape is not None else None,
    }


def get_actual_vs_predicted_until(
    end_date: Optional[str] = None,
    start_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compare stored actual prices with aggregated stored predictions up to end_date.

    Multiple forecasts for the same target date are aggregated using a horizon-weighted
    mean (weight = 1 / horizon), which prioritizes shorter-horizon forecasts.

    Args:
        end_date: Optional YYYY-MM-DD end date. Defaults to today.
        start_date: Optional YYYY-MM-DD start date. Defaults to earliest available.

    Returns:
        Dict with comparison rows and error summary metrics.
    """
    cutoff_date = date.today() if end_date is None else datetime.strptime(end_date, "%Y-%m-%d").date()
    start_bound = (
        datetime.strptime(start_date, "%Y-%m-%d").date() if start_date is not None else None
    )

    conn = get_connection()
    try:
        # Fetch actual prices up to the requested cutoff date.
        if start_bound is None:
            actual_df = _query_to_df(
                conn,
                """
                SELECT date, price
                FROM prices
                WHERE date <= ?
                ORDER BY date ASC
                """,
                params=(cutoff_date.strftime("%Y-%m-%d"),),
            )
        else:
            actual_df = _query_to_df(
                conn,
                """
                SELECT date, price
                FROM prices
                WHERE date >= ? AND date <= ?
                ORDER BY date ASC
                """,
                params=(start_bound.strftime("%Y-%m-%d"), cutoff_date.strftime("%Y-%m-%d")),
            )

        if actual_df.empty:
            return _empty_comparison_payload(cutoff_date)

        prediction_runs = _query_to_df(
            conn,
            """
            SELECT generated_at, last_price_date, forecasts
            FROM predictions
            ORDER BY generated_at ASC
            """,
        )
    finally:
        conn.close()

    actual_df["date"] = pd.to_datetime(actual_df["date"]).dt.date

    per_date_predictions = _collect_predictions_by_target_date(prediction_runs, cutoff_date)
    rows = _build_comparison_rows(actual_df, per_date_predictions)

    if not rows:
        return _empty_comparison_payload(cutoff_date)

    return {
        "end_date": cutoff_date.strftime("%Y-%m-%d"),
        "rows": rows,
        "metrics": _compute_comparison_metrics(rows),
    }
