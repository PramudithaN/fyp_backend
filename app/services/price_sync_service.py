"""
Price sync service — keeps the prices table current by fetching from Yahoo Finance.

Functions:
- backfill_prices_to_today()  : fills any gap between the last stored date and today.
- run_daily_price_sync()       : thin wrapper used by the scheduler (same logic,
                                  extra logging/return value for job tracking).
"""

import logging
from datetime import datetime, timedelta, date
from typing import Dict, Any

import pandas as pd

from app.database import add_bulk_prices, get_latest_price_date
from app.services.price_fetcher import fetch_latest_prices

logger = logging.getLogger(__name__)

# How far back to fetch when the prices table is completely empty.
_BOOTSTRAP_LOOKBACK_DAYS = 730


def backfill_prices_to_today() -> Dict[str, Any]:
    """
    Ensure the prices table is current up to today.

    Steps:
    1. Determine the latest date already stored in the prices table.
    2. If the table is empty, bootstrap with _BOOTSTRAP_LOOKBACK_DAYS of history.
    3. Fetch all trading days from Yahoo Finance since (latest_stored + 1) through today.
    4. Upsert the fetched rows into the prices table.

    Returns:
        Summary dict with keys: status, latest_stored_before, latest_stored_after,
        rows_fetched, rows_saved.
    """
    result: Dict[str, Any] = {
        "status": "success",
        "latest_stored_before": None,
        "latest_stored_after": None,
        "rows_fetched": 0,
        "rows_saved": 0,
    }

    # ── Step 1: find out how far our DB goes ──────────────────────────────────
    latest_stored = get_latest_price_date()
    result["latest_stored_before"] = latest_stored
    today_str = date.today().isoformat()
    today_dt = datetime.today()

    if latest_stored and latest_stored >= today_str:
        logger.info("Prices already up to date (latest=%s)", latest_stored)
        result["latest_stored_after"] = latest_stored
        result["status"] = "already_current"
        return result

    # ── Step 2: choose lookback window ────────────────────────────────────────
    if latest_stored:
        last_dt = datetime.strptime(latest_stored, "%Y-%m-%d")
        gap_days = (today_dt - last_dt).days
        # Fetch enough to cover the gap plus a small buffer.
        lookback_days = max(gap_days + 10, 30)
        logger.info(
            "Price gap detected: latest=%s, today=%s — fetching %d days",
            latest_stored, today_str, lookback_days,
        )
    else:
        lookback_days = _BOOTSTRAP_LOOKBACK_DAYS
        logger.info(
            "Prices table empty — bootstrapping with %d days of history",
            lookback_days,
        )

    # ── Step 3: fetch from Yahoo Finance ─────────────────────────────────────
    try:
        df = fetch_latest_prices(lookback_days=lookback_days, end_date=today_dt)
    except Exception as exc:
        logger.error("Yahoo Finance fetch failed during price sync: %s", exc, exc_info=True)
        result["status"] = "error"
        result["error"] = str(exc)
        return result

    result["rows_fetched"] = len(df)

    if df.empty:
        logger.warning("Yahoo Finance returned no rows for the requested window")
        result["status"] = "no_data"
        return result

    # ── Step 4: upsert into DB ────────────────────────────────────────────────
    # Only insert rows strictly after the previously stored date to avoid
    # unnecessary writes, but "INSERT OR REPLACE" is safe regardless.
    price_records = [
        {
            "date": pd.to_datetime(row["date"]).strftime("%Y-%m-%d"),
            "price": float(row["price"]),
            "source": "yahoo_finance",
        }
        for _, row in df.iterrows()
    ]

    try:
        saved = add_bulk_prices(price_records)
        result["rows_saved"] = saved
    except Exception as exc:
        logger.error("DB write failed during price sync: %s", exc, exc_info=True)
        result["status"] = "db_error"
        result["error"] = str(exc)
        return result

    result["latest_stored_after"] = max(r["date"] for r in price_records)
    logger.info(
        "Price sync complete: fetched=%d saved=%d latest=%s",
        result["rows_fetched"],
        result["rows_saved"],
        result["latest_stored_after"],
    )
    return result


def run_daily_price_sync() -> Dict[str, Any]:
    """
    Entry point called by the APScheduler daily cron job.

    Delegates to backfill_prices_to_today() and enriches the result dict with
    a timestamp for job-tracking purposes.
    """
    logger.info("[PriceSyncJob] Starting daily price sync")
    started_at = datetime.utcnow().isoformat()

    result = backfill_prices_to_today()
    result["started_at"] = started_at
    result["completed_at"] = datetime.utcnow().isoformat()

    if result["status"] == "success":
        logger.info(
            "[PriceSyncJob] Done — saved=%d latest=%s",
            result.get("rows_saved", 0),
            result.get("latest_stored_after"),
        )
    elif result["status"] == "already_current":
        logger.info("[PriceSyncJob] Prices already current, nothing to do")
    else:
        logger.warning("[PriceSyncJob] Finished with status=%s", result["status"])

    return result
