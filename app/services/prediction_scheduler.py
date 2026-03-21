"""
Daily locked prediction scheduler.

Runs once after market close and stores exactly one locked forecast row per day.
"""

import logging
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from app.config import (
    PREDICTION_LOCK_SCHEDULE_ENABLED,
    PREDICTION_LOCK_SCHEDULE_HOUR,
    PREDICTION_LOCK_SCHEDULE_MINUTE,
    PREDICTION_LOCK_SCHEDULE_TIMEZONE,
)
from app.database import upsert_daily_prediction
from app.services.prediction import prediction_service
from app.services.price_fetcher import fetch_latest_prices

logger = logging.getLogger(__name__)

_scheduler: AsyncIOScheduler | None = None


def _to_yyyymmdd(ts: pd.Timestamp) -> str:
    return pd.to_datetime(ts).strftime("%Y-%m-%d")


def _resolve_previous_trading_close(prices: pd.DataFrame, prediction_date: str) -> tuple[str, float]:
    """
    Select the latest available trading close strictly before prediction_date.
    """
    if prices.empty:
        raise ValueError("No price data available to compute locked prediction")

    working = prices[["date", "price"]].copy()
    working["date"] = pd.to_datetime(working["date"]).dt.tz_localize(None)
    working = working.sort_values("date").reset_index(drop=True)

    prediction_dt = pd.to_datetime(prediction_date)
    candidates = working[working["date"] < prediction_dt]
    if candidates.empty:
        # Fallback to latest available row if historical window is too narrow.
        row = working.iloc[-1]
    else:
        row = candidates.iloc[-1]

    return _to_yyyymmdd(row["date"]), round(float(row["price"]), 2)


def run_daily_prediction_job(now_local: datetime | None = None) -> dict:
    """
    Generate and upsert the single locked forecast row for the current local day.

    Returns:
        Summary dict containing lock metadata and persistence outcome.
    """
    tz = ZoneInfo(PREDICTION_LOCK_SCHEDULE_TIMEZONE)
    local_now = now_local.astimezone(tz) if now_local is not None else datetime.now(tz)
    prediction_date = local_now.strftime("%Y-%m-%d")

    logger.info("Daily locked prediction job started for prediction_date=%s", prediction_date)

    prices = fetch_latest_prices(lookback_days=180, end_date=local_now.replace(tzinfo=None))
    based_on_price_date, based_on_price = _resolve_previous_trading_close(
        prices=prices,
        prediction_date=prediction_date,
    )

    # Model receives historical prices up to and including the based-on close date.
    model_prices = prices.copy()
    model_prices["date"] = pd.to_datetime(model_prices["date"]).dt.tz_localize(None)
    model_prices = model_prices[model_prices["date"] <= pd.to_datetime(based_on_price_date)]
    model_prices = model_prices.sort_values("date").reset_index(drop=True)

    if model_prices.empty:
        raise ValueError("No model input prices available for locked daily prediction")

    forecasts = prediction_service.predict(prices=model_prices)

    # Keep API response dates aligned to the based-on close (first forecast = next business day).
    current_date = pd.to_datetime(based_on_price_date)
    aligned_forecasts = []
    for forecast in forecasts:
        current_date += pd.Timedelta(days=1)
        while current_date.weekday() >= 5:
            current_date += pd.Timedelta(days=1)

        aligned_forecasts.append(
            {
                **forecast,
                "date": current_date.strftime("%Y-%m-%d"),
            }
        )

    locked_at = datetime.now().isoformat()
    row_id = upsert_daily_prediction(
        prediction_date=prediction_date,
        based_on_price_date=based_on_price_date,
        based_on_price=based_on_price,
        forecasts=aligned_forecasts,
        locked_at=locked_at,
    )

    result = {
        "status": "success",
        "prediction_date": prediction_date,
        "based_on_price_date": based_on_price_date,
        "based_on_price": based_on_price,
        "locked_at": locked_at,
        "row_id": row_id,
        "forecast_points": len(aligned_forecasts),
    }
    logger.info("Daily locked prediction job completed: %s", result)
    return result


def init_prediction_scheduler() -> AsyncIOScheduler | None:
    """Initialize and start the daily locked prediction scheduler."""
    global _scheduler

    if not PREDICTION_LOCK_SCHEDULE_ENABLED:
        logger.info("Locked prediction scheduler disabled by config")
        return None

    if _scheduler is not None and _scheduler.running:
        logger.info("Locked prediction scheduler already running")
        return _scheduler

    tz = ZoneInfo(PREDICTION_LOCK_SCHEDULE_TIMEZONE)
    _scheduler = AsyncIOScheduler(timezone=tz)

    _scheduler.add_job(
        run_daily_prediction_job,
        CronTrigger(
            hour=PREDICTION_LOCK_SCHEDULE_HOUR,
            minute=PREDICTION_LOCK_SCHEDULE_MINUTE,
            timezone=tz,
        ),
        id="daily_locked_prediction",
        name="Daily Locked Oil Forecast",
        replace_existing=True,
    )

    _scheduler.start()
    logger.info(
        "Locked prediction scheduler started at %02d:%02d %s",
        PREDICTION_LOCK_SCHEDULE_HOUR,
        PREDICTION_LOCK_SCHEDULE_MINUTE,
        PREDICTION_LOCK_SCHEDULE_TIMEZONE,
    )
    return _scheduler


def shutdown_prediction_scheduler() -> None:
    """Shutdown locked prediction scheduler if running."""
    global _scheduler

    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=True)
        logger.info("Locked prediction scheduler shut down")
    _scheduler = None


def trigger_prediction_job_now() -> dict:
    """Manual trigger helper for admin/testing usage."""
    try:
        return run_daily_prediction_job()
    except Exception as exc:
        logger.error("Manual daily prediction trigger failed: %s", exc, exc_info=True)
        return {"status": "failed", "error": str(exc)}
