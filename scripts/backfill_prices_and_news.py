"""
One-time backfill for Turso: prices + news_articles for a date range.

Default target range:
- start: 2025-12-01
- end: today

What this script does:
1. Fetches Brent prices from Yahoo Finance and upserts them into prices table.
2. Scrapes 4 configured news sites across the range and stores per-article rows.
3. Computes per-article sentiment scores before storing (same path as scheduler).

Usage:
    python scripts/backfill_prices_and_news.py
    python scripts/backfill_prices_and_news.py --start-date 2025-12-01 --end-date 2026-03-16
    python scripts/backfill_prices_and_news.py --max-pages-per-site 50
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


import os
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from app.services.price_fetcher import fetch_latest_prices
from app.services.news_scraper import scrape_all_sources_multiday, scrape_all_sources
from app.services.news_fetcher import compute_sentiment_features_with_articles


logger = logging.getLogger(__name__)


TURSO_URL = os.environ.get("TURSO_DATABASE_URL", "")
TURSO_TOKEN = os.environ.get("TURSO_AUTH_TOKEN", "")


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


def _ensure_schema(conn) -> None:
    _execute(conn, """
        CREATE TABLE IF NOT EXISTS prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT UNIQUE NOT NULL,
            price REAL NOT NULL,
            source TEXT DEFAULT 'yahoo_finance',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    _execute(conn, "CREATE INDEX IF NOT EXISTS idx_prices_date ON prices(date)")

    _execute(conn, """
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
    _execute(conn, "CREATE INDEX IF NOT EXISTS idx_articles_date ON news_articles(article_date)")

    _commit(conn)


def _upsert_bulk_prices(conn, records: List[Dict[str, Any]]) -> int:
    count = 0
    for rec in records:
        _execute(
            conn,
            "INSERT OR REPLACE INTO prices (date, price, source) VALUES (?, ?, ?)",
            (rec["date"], rec["price"], rec.get("source", "yahoo_finance")),
        )
        count += 1
    _commit(conn)
    return count


def _insert_news_articles(conn, article_date: str, articles: List[Dict[str, Any]]) -> int:
    count = 0
    for art in articles:
        url = art.get("url")
        already_exists = False
        if url:
            exists_res = _execute(
                conn,
                "SELECT 1 FROM news_articles WHERE url = ? LIMIT 1",
                (url,),
            )
            exists_rows = getattr(exists_res, "fetchall", lambda: [])()
            already_exists = bool(exists_rows)

        if already_exists:
            continue

        _execute(
            conn,
            """
            INSERT OR IGNORE INTO news_articles
                (article_date, title, description, url, source, published_at, sentiment_score)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                article_date,
                art.get("title", ""),
                art.get("description", ""),
                url,
                art.get("source", ""),
                art.get("published_at", ""),
                art.get("sentiment_score"),
            ),
        )

        # rowcount is not reliable on some Turso drivers; count by URL pre-check.
        count += 1

    _commit(conn)
    return count


def _parse_date(value: str) -> datetime.date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _daterange(start_date, end_date):
    day = start_date
    while day <= end_date:
        yield day
        day += timedelta(days=1)


def _backfill_prices(conn, start_date, end_date) -> Dict[str, Any]:
    lookback_days = (end_date - start_date).days + 10
    prices_df = fetch_latest_prices(lookback_days=lookback_days, end_date=datetime.combine(end_date, datetime.min.time()))

    filtered = prices_df[
        (prices_df["date"].dt.date >= start_date)
        & (prices_df["date"].dt.date <= end_date)
    ].copy()

    records = [
        {
            "date": d.date().strftime("%Y-%m-%d"),
            "price": float(p),
            "source": "yahoo_finance",
        }
        for d, p in zip(filtered["date"], filtered["price"])
    ]

    inserted = _upsert_bulk_prices(conn, records) if records else 0
    return {
        "requested_start": start_date.strftime("%Y-%m-%d"),
        "requested_end": end_date.strftime("%Y-%m-%d"),
        "fetched_trading_days": len(records),
        "upserted_rows": inserted,
        "first_price_date": records[0]["date"] if records else None,
        "last_price_date": records[-1]["date"] if records else None,
    }


def _backfill_news(conn, start_date, end_date, max_pages_per_site: int) -> Dict[str, Any]:
    today = datetime.now().date()
    if end_date > today:
        end_date = today

    # scrape_all_sources_multiday is relative to today and excludes day 0,
    # so include enough days to cover the requested start date.
    days_back = (today - start_date).days + 1
    days_back = max(days_back, 1)

    logger.info(
        "Scraping multiday news for %d days back (max_pages_per_site=%d)",
        days_back,
        max_pages_per_site,
    )
    grouped = scrape_all_sources_multiday(
        days_back=days_back,
        max_pages_per_site=max_pages_per_site,
    )

    # scrape_all_sources_multiday excludes day 0 (today), so include it explicitly.
    if end_date == today:
        today_str = today.strftime("%Y-%m-%d")
        try:
            grouped[today_str] = scrape_all_sources(
                target_date=today_str,
                max_pages=max_pages_per_site,
            )
            logger.info("Included same-day scrape for %s", today_str)
        except Exception as e:
            logger.warning("Failed same-day scrape for %s: %s", today_str, e)

    total_raw_articles = 0
    total_inserted_articles = 0
    days_with_articles = 0
    days_without_articles = 0

    for day in _daterange(start_date, end_date):
        date_str = day.strftime("%Y-%m-%d")
        articles = grouped.get(date_str, [])
        total_raw_articles += len(articles)

        if not articles:
            days_without_articles += 1
            continue

        days_with_articles += 1

        # Keep same enrichment path as scheduler so sentiment_score is populated.
        _, enriched = compute_sentiment_features_with_articles(articles)
        inserted = _insert_news_articles(conn, date_str, enriched)
        total_inserted_articles += inserted

        logger.info(
            "News backfill %s: raw=%d inserted=%d",
            date_str,
            len(articles),
            inserted,
        )

    return {
        "requested_start": start_date.strftime("%Y-%m-%d"),
        "requested_end": end_date.strftime("%Y-%m-%d"),
        "days_with_articles": days_with_articles,
        "days_without_articles": days_without_articles,
        "raw_articles_seen": total_raw_articles,
        "inserted_articles": total_inserted_articles,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill Turso prices and news_articles")
    parser.add_argument("--start-date", default="2025-12-01", help="YYYY-MM-DD")
    parser.add_argument("--end-date", default=datetime.now().strftime("%Y-%m-%d"), help="YYYY-MM-DD")
    parser.add_argument("--max-pages-per-site", type=int, default=50, help="1-50")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    start_date = _parse_date(args.start_date)
    end_date = _parse_date(args.end_date)

    if start_date > end_date:
        raise ValueError("start-date must be <= end-date")

    max_pages = max(1, min(int(args.max_pages_per_site), 50))

    _ensure_turso_env()
    conn = _connect_turso(TURSO_URL, TURSO_TOKEN)
    logger.info("Connected to Turso via %s", _TURSO_DRIVER)
    _ensure_schema(conn)

    logger.info(
        "Starting backfill for range %s to %s",
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
    )

    price_summary = _backfill_prices(conn, start_date, end_date)
    news_summary = _backfill_news(conn, start_date, end_date, max_pages)

    logger.info("Backfill complete")
    logger.info("Price summary: %s", price_summary)
    logger.info("News summary: %s", news_summary)

    print("\n=== Backfill Summary ===")
    print("Prices:", price_summary)
    print("News:", news_summary)

    try:
        conn.close()
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
