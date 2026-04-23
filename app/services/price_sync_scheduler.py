"""
Daily price sync scheduler.

Runs once per day (configurable via env vars) to keep the prices table
current with the latest Brent crude closing prices from Yahoo Finance.

The scheduler is separate from the prediction scheduler so prices stay fresh
even when the prediction job is disabled or fails.
"""

import logging
from typing import Optional
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from app.config import (
    PRICE_SYNC_SCHEDULE_ENABLED,
    PRICE_SYNC_SCHEDULE_HOUR,
    PRICE_SYNC_SCHEDULE_MINUTE,
    PRICE_SYNC_SCHEDULE_TIMEZONE,
)

logger = logging.getLogger(__name__)

_scheduler: Optional[AsyncIOScheduler] = None


def init_price_sync_scheduler() -> Optional[AsyncIOScheduler]:
    """Initialize and start the daily price sync scheduler."""
    global _scheduler

    if not PRICE_SYNC_SCHEDULE_ENABLED:
        logger.info("Price sync scheduler disabled by config (PRICE_SYNC_SCHEDULE_ENABLED=false)")
        return None

    if _scheduler is not None and _scheduler.running:
        logger.info("Price sync scheduler already running")
        return _scheduler

    tz = ZoneInfo(PRICE_SYNC_SCHEDULE_TIMEZONE)
    _scheduler = AsyncIOScheduler(timezone=tz)

    _scheduler.add_job(
        _run_sync_job,
        CronTrigger(
            hour=PRICE_SYNC_SCHEDULE_HOUR,
            minute=PRICE_SYNC_SCHEDULE_MINUTE,
            timezone=tz,
        ),
        id="daily_price_sync",
        name="Daily Brent Price Sync",
        replace_existing=True,
    )

    _scheduler.start()
    logger.info(
        "Price sync scheduler started — runs daily at %02d:%02d %s",
        PRICE_SYNC_SCHEDULE_HOUR,
        PRICE_SYNC_SCHEDULE_MINUTE,
        PRICE_SYNC_SCHEDULE_TIMEZONE,
    )
    return _scheduler


def shutdown_price_sync_scheduler() -> None:
    """Gracefully shut down the price sync scheduler if running."""
    global _scheduler

    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=True)
        logger.info("Price sync scheduler shut down")
    _scheduler = None


def trigger_price_sync_now() -> dict:
    """Run a price sync immediately (used by admin endpoint / startup backfill)."""
    from app.services.price_sync_service import run_daily_price_sync

    logger.info("Manual price sync triggered")
    try:
        return run_daily_price_sync()
    except Exception as exc:
        logger.error("Manual price sync failed: %s", exc, exc_info=True)
        return {"status": "error", "error": str(exc)}


def _run_sync_job() -> None:
    """Scheduler callback — wraps run_daily_price_sync for APScheduler."""
    from app.services.price_sync_service import run_daily_price_sync

    try:
        run_daily_price_sync()
    except Exception as exc:
        logger.error("Scheduled price sync job failed: %s", exc, exc_info=True)
