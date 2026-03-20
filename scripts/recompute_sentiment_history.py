"""
Recompute sentiment_history from news_articles in Turso.

Process:
1. Clear sentiment_history table
2. Query all news_articles grouped by article_date
3. For each article, compute sentiment (FinBERT with fallback to simple)
4. Average sentiments per day
5. Apply cross-day decay (LAMBDA=0.3)
6. Store in sentiment_history with all columns

Usage:
    python scripts/recompute_sentiment_history.py
    python scripts/recompute_sentiment_history.py --batch-size 50 --chunk-days 30
"""

from __future__ import annotations

import argparse
import logging
import sys
import os
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

logger = logging.getLogger(__name__)

TURSO_URL = os.environ.get("TURSO_DATABASE_URL", "")
TURSO_TOKEN = os.environ.get("TURSO_AUTH_TOKEN", "")

# Sentiment decay parameter
SENTIMENT_DECAY_LAMBDA = 0.3

try:
    import libsql_experimental as _libsql  # type: ignore[import-not-found]

    _TURSO_DRIVER = "libsql_experimental"

    def _connect_turso(url: str, token: str):
        return _libsql.connect(database=url, auth_token=token)

    def _execute(conn, query: str, params=()):
        cur = conn.cursor()
        cur.execute(query, params)
        return cur

    def _commit(conn):
        conn.commit()

except ImportError:
    import libsql_client

    _TURSO_DRIVER = "libsql_client"

    def _normalize_turso_url(url: str) -> str:
        if url.startswith("libsql://"):
            return "https://" + url[len("libsql://"):]
        return url

    def _connect_turso(url: str, token: str):
        return libsql_client.create_client_sync(
            url=_normalize_turso_url(url),
            auth_token=token,
        )

    def _execute(conn, query: str, params=()):
        return conn.execute(query, params)

    def _commit(_conn):
        return None


def _ensure_turso_env() -> None:
    if not TURSO_URL:
        raise RuntimeError("TURSO_DATABASE_URL is not set")
    if not TURSO_TOKEN:
        raise RuntimeError("TURSO_AUTH_TOKEN is not set")


def _get_sentiment_score(text: str, use_fallback_only: bool = False) -> float:
    """
    Compute sentiment score for text using FinBERT with chunking and fallback.
    
    Args:
        text: Text to analyze
        use_fallback_only: If True, skip FinBERT and use simple sentiment only
    
    Returns:
        Sentiment score between -1 and 1
    """
    if not text or len(text.strip()) < 10:
        return 0.0

    if not use_fallback_only:
        try:
            from app.services.finbert_analyzer import analyze_sentiment_finbert
            
            # Try FinBERT analysis
            score = analyze_sentiment_finbert(text)
            return score
        
        except Exception as e:
            logger.debug(f"FinBERT analysis failed ({type(e).__name__}), falling back to simple sentiment")

    # Fallback to simple keyword-based sentiment
    try:
        from app.services.news_fetcher import analyze_sentiment_simple
        return analyze_sentiment_simple(text)
    except Exception as e:
        logger.warning(f"Even simple sentiment failed: {e}, returning 0.0")
        return 0.0


def _clear_sentiment_history(conn) -> int:
    """Delete all rows from sentiment_history."""
    cur = _execute(conn, "DELETE FROM sentiment_history")
    _commit(conn)
    count = getattr(cur, "rowcount", None)
    if count is None:
        count = 0
    logger.info(f"Cleared sentiment_history: {count} rows deleted")
    return count


def _fetch_articles_by_date(conn) -> Dict[str, List[Dict[str, Any]]]:
    """
    Fetch all news_articles from Turso, grouped by article_date.
    
    Returns:
        Dict mapping date string to list of article dicts
    """
    query = """
        SELECT article_date, title, description, url, source, published_at, sentiment_score
        FROM news_articles
        ORDER BY article_date, id
    """
    
    res = _execute(conn, query)
    
    # Handle both libsql_experimental (cursor with fetchall) and libsql_client (ResultSet)
    if hasattr(res, 'fetchall'):
        rows = res.fetchall()
    else:
        # libsql_client returns an iterable ResultSet
        rows = list(res)
    
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        article_date = row[0]
        article = {
            "title": row[1],
            "description": row[2],
            "url": row[3],
            "source": row[4],
            "published_at": row[5],
            "existing_sentiment": row[6],  # Already stored score
        }
        grouped.setdefault(article_date, []).append(article)
    
    logger.info(f"Fetched {sum(len(v) for v in grouped.values())} articles across {len(grouped)} days")
    return grouped


def _compute_sentiment_for_articles(
    articles: List[Dict[str, Any]],
    batch_size: int = 16,
    use_fallback_only: bool = False,
) -> List[float]:
    """
    Compute sentiment scores using batch FinBERT inference (much faster than one-at-a-time).
    Falls back to simple keyword-based sentiment if FinBERT unavailable.
    
    Args:
        articles: List of article dicts with title, description
        batch_size: Batch size for FinBERT (16 is optimal for CPU)
        use_fallback_only: Skip FinBERT, use simple keyword sentiment only
    
    Returns:
        List of sentiment scores
    """
    texts = []
    for art in articles:
        title = art.get("title", "") or ""
        description = art.get("description", "") or ""
        texts.append(f"{title}. {description}")

    if not use_fallback_only:
        try:
            from app.services.finbert_analyzer import analyze_batch_finbert
            scores = analyze_batch_finbert(texts, batch_size=batch_size)
            if len(scores) == len(texts):
                return scores
        except Exception as e:
            logger.warning(f"FinBERT batch failed: {e}, using simple sentiment fallback")

    # Fallback: simple keyword-based sentiment per article
    from app.services.news_fetcher import analyze_sentiment_simple
    return [analyze_sentiment_simple(text) for text in texts]


def _compute_daily_sentiment(articles: List[Dict[str, Any]], scores: List[float]) -> float:
    """
    Compute daily sentiment as simple mean of article scores.
    
    Args:
        articles: List of articles (for logging)
        scores: List of sentiment scores
    
    Returns:
        Daily sentiment value (mean)
    """
    if not scores:
        return 0.0
    
    daily_sentiment = float(np.mean(scores))
    logger.info(f"Daily sentiment for {len(articles)} articles: {daily_sentiment:.6f}")
    return daily_sentiment


def _compute_voice_features(num_articles: int) -> Dict[str, float]:
    """Compute volume-related features matching news_fetcher.py logic."""
    import math
    
    log_volume = math.log(num_articles + 1)
    decayed_volume = float(num_articles) * 0.5  # Approximate EWM for matching training features
    high_regime = 1 if num_articles > 30 else 0
    
    return {
        "news_volume": num_articles,
        "log_news_volume": log_volume,
        "decayed_news_volume": decayed_volume,
        "high_news_regime": high_regime,
    }


def _apply_cross_day_decay(sentiment_by_date: Dict[str, float]) -> Dict[str, float]:
    """
    Apply exponential cross-day decay to sentiment time series.
    
    Formula: decayed[t] = sentiment[t] + exp(-LAMBDA) * decayed[t-1]
    
    Args:
        sentiment_by_date: Dict mapping date string to raw sentiment
    
    Returns:
        Dict mapping date string to decayed sentiment
    """
    sorted_dates = sorted(sentiment_by_date.keys())
    
    decayed = {}
    decay_factor = np.exp(-SENTIMENT_DECAY_LAMBDA)
    
    for i, date_str in enumerate(sorted_dates):
        raw = sentiment_by_date[date_str]
        
        if i == 0:
            decayed[date_str] = raw
        else:
            prev_date = sorted_dates[i - 1]
            decayed[date_str] = raw + decay_factor * decayed[prev_date]
    
    return decayed


def _insert_sentiment_records(
    conn,
    sentiment_records: List[Dict[str, Any]],
) -> int:
    """Insert sentiment records into sentiment_history table."""
    count = 0
    
    for rec in sentiment_records:
        _execute(
            conn,
            """
            INSERT INTO sentiment_history
                (date, daily_sentiment_decay, news_volume, log_news_volume,
                 decayed_news_volume, high_news_regime)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                rec["date"],
                rec["daily_sentiment_decay"],
                rec["news_volume"],
                rec["log_news_volume"],
                rec["decayed_news_volume"],
                rec["high_news_regime"],
            ),
        )
        # For libsql_experimental, rowcount may be available; for libsql_client, just count attempts
        count += 1
    
    _commit(conn)
    logger.info(f"Inserted {count} sentiment records into sentiment_history")
    return count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recompute sentiment_history from news_articles in Turso"
    )
    parser.add_argument("--batch-size", type=int, default=50, help="Articles per batch")
    parser.add_argument("--chunk-days", type=int, default=30, help="Process in daily chunks")
    parser.add_argument("--use-fallback-only", action="store_true", help="Skip FinBERT, use simple sentiment only")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    _ensure_turso_env()
    conn = _connect_turso(TURSO_URL, TURSO_TOKEN)
    logger.info(f"Connected to Turso via {_TURSO_DRIVER}")

    try:
        # Step 1: Clear sentiment_history
        logger.info("Step 1: Clearing sentiment_history table")
        _clear_sentiment_history(conn)

        # Step 2: Fetch all articles grouped by date
        logger.info("Step 2: Fetching articles from news_articles table")
        articles_by_date = _fetch_articles_by_date(conn)
        
        if not articles_by_date:
            logger.warning("No articles found in news_articles table")
            return 0

        # Step 3: Compute sentiments and daily aggregates
        logger.info("Step 3: Computing sentiment scores")
        sentiment_by_date = {}
        volume_info_by_date = {}
        
        sorted_dates = sorted(articles_by_date.keys())
        
        for i, date_str in enumerate(sorted_dates):
            articles = articles_by_date[date_str]
            
            logger.info(f"Processing {date_str} ({i+1}/{len(sorted_dates)}) with {len(articles)} articles")
            
            # Compute individual article sentiments
            scores = _compute_sentiment_for_articles(
                articles,
                batch_size=args.batch_size,
                use_fallback_only=args.use_fallback_only,
            )
            
            # Daily sentiment = mean of scores
            daily_sent = _compute_daily_sentiment(articles, scores)
            sentiment_by_date[date_str] = daily_sent
            
            # Volume features
            volume_info_by_date[date_str] = _compute_voice_features(len(articles))

        # Step 4: Apply cross-day decay
        logger.info("Step 4: Applying cross-day exponential decay")
        decayed_sentiment = _apply_cross_day_decay(sentiment_by_date)

        # Step 5: Build records and insert
        logger.info("Step 5: Storing sentiment records in Turso")
        records = []
        
        for date_str in sorted_dates:
            records.append({
                "date": date_str,
                "daily_sentiment_decay": decayed_sentiment[date_str],
                **volume_info_by_date[date_str],
            })

        _insert_sentiment_records(conn, records)

        logger.info("=== Recomputation complete ===")
        logger.info(f"Processed {len(sorted_dates)} days with {sum(len(v) for v in articles_by_date.values())} articles")
        
        return 0

    except Exception:
        logger.error("Recomputation failed", exc_info=True)
        return 1
    
    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
