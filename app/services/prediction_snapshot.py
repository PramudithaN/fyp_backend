"""Shared locked prediction snapshot helpers.

This module centralizes how the backend selects and normalizes the canonical
forecast snapshot so all APIs can use the same source of truth.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

from app.config import (
    PREDICTION_CLOSE_LOCK_BUFFER_MINUTES,
    PREDICTION_LOCK_SCHEDULE_TIMEZONE,
)
from app.database import get_latest_locked_prediction, get_prediction_for_date
from app.services.price_fetcher import get_canonical_prediction_date

logger = logging.getLogger(__name__)


class LockedPredictionUnavailableError(RuntimeError):
    """Raised when no locked prediction snapshot is available."""


def current_prediction_date_local() -> str:
    """Return canonical prediction key in configured timezone using Yahoo close cutoff."""
    tz = ZoneInfo(PREDICTION_LOCK_SCHEDULE_TIMEZONE)
    return get_canonical_prediction_date(
        target_timezone=PREDICTION_LOCK_SCHEDULE_TIMEZONE,
        close_lock_buffer_minutes=PREDICTION_CLOSE_LOCK_BUFFER_MINUTES,
        now_target=datetime.now(tz),
    )


def _normalize_locked_record(record: Dict[str, Any], source: str) -> Dict[str, Any]:
    """Normalize a DB locked-prediction row to a canonical snapshot shape."""
    based_on_price_date = record.get("based_on_price_date") or record.get("last_price_date")
    based_on_price = record.get("based_on_price")
    if based_on_price is None:
        based_on_price = record.get("last_price", 0.0)

    forecasts = record.get("forecasts") or []

    return {
        "source": source,
        "prediction_date": str(record.get("prediction_date") or ""),
        "generated_at": str(record.get("locked_at") or record.get("generated_at") or ""),
        "last_price_date": str(based_on_price_date),
        "last_price": float(based_on_price),
        "based_on_price_date": str(based_on_price_date),
        "based_on_price": float(based_on_price),
        "forecasts": forecasts,
    }


def get_locked_prediction_snapshot(
    prediction_date: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Return canonical locked forecast snapshot.

    Selection order mirrors `/predict`:
    1. Locked row for requested/local prediction date.
    2. Latest available locked row.
    """
    target_date = prediction_date or current_prediction_date_local()

    try:
        today_record = get_prediction_for_date(target_date)
    except Exception as db_err:
        logger.warning("Failed reading locked prediction for %s: %s", target_date, db_err)
        today_record = None

    if today_record:
        return _normalize_locked_record(today_record, source="locked_for_date")

    try:
        latest_record = get_latest_locked_prediction()
    except Exception as db_err:
        logger.warning("Failed reading latest locked prediction: %s", db_err)
        latest_record = None

    if latest_record:
        return _normalize_locked_record(latest_record, source="locked_latest")

    return None


def get_required_locked_prediction_snapshot(
    prediction_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Return locked snapshot or raise a domain-specific error."""
    snapshot = get_locked_prediction_snapshot(prediction_date=prediction_date)
    if snapshot is None:
        raise LockedPredictionUnavailableError(
            "No locked daily forecast available yet"
        )
    return snapshot
