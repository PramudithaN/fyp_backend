"""
Database layer for storing sentiment, prices, news articles, and predictions.

NOTE: The sentiment_history table stores raw daily_sentiment (simple mean).
Cross-day decay is applied at retrieval time by sentiment_service.py

Database: Turso (libsql) — remote, persistent across deploys.
"""

import json
import os
import math
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
