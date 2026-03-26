"""Helpers for in-memory prediction from uploaded Excel price windows."""

from __future__ import annotations

from io import BytesIO
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from app.database import get_prices_for_date_range, get_sentiment_for_dates
from app.models.model_loader import model_artifacts
from app.services.prediction import prediction_service

SUPPORTED_EXTENSIONS = (".xlsx", ".xls")
_DATE_COL_CANDIDATES = ("date", "day", "timestamp", "datetime")
_PRICE_COL_CANDIDATES = (
    "price",
    "close",
    "closing_price",
    "brent_price",
    "value",
)


def build_upload_excel_template_bytes(lookback_days: int) -> bytes:
    """Build a strict Excel template with required columns for upload."""
    days = max(int(lookback_days), 1)
    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=days - 1)

    rows = []
    for idx in range(days):
        current_date = start_date + timedelta(days=idx)
        rows.append(
            {
                "date": current_date.strftime("%Y-%m-%d"),
                "price": "",
            }
        )

    template_df = pd.DataFrame(rows)
    instructions_df = pd.DataFrame(
        {
            "rule": [
                "Required columns",
                "Date format",
                "Price rule",
                "Duplicates",
                "Expected rows",
            ],
            "value": [
                "date, price",
                "YYYY-MM-DD",
                "price must be numeric and > 0",
                "If duplicate dates exist, latest row wins",
                f"Upload up to {days} rows (model lookback={days})",
            ],
        }
    )

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        template_df.to_excel(writer, sheet_name="template", index=False)
        instructions_df.to_excel(writer, sheet_name="instructions", index=False)

    return buffer.getvalue()


def _normalize_col_name(col: str) -> str:
    return str(col).strip().lower().replace(" ", "_")


def _pick_column(columns: list[str], candidates: tuple[str, ...]) -> Optional[str]:
    normalized = {_normalize_col_name(col): col for col in columns}
    for cand in candidates:
        if cand in normalized:
            return normalized[cand]
    return None


def _is_blank_cell(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    return bool(pd.isna(value))


def _summarize_row_errors(
    messages: list[str], total_rows: int, shown_rows: int = 10
) -> str:
    if not messages:
        return ""

    visible = messages[:shown_rows]
    summary = " ; ".join(visible)
    hidden = max(total_rows - shown_rows, 0)
    if hidden > 0:
        summary += f" ; ... and {hidden} more row error(s)"
    return summary


def _parse_date_cell(raw_date: Any) -> tuple[Optional[pd.Timestamp], Optional[str]]:
    if _is_blank_cell(raw_date):
        return None, "date is required"

    if isinstance(raw_date, pd.Timestamp):
        return raw_date, None

    if isinstance(raw_date, datetime):
        return pd.Timestamp(raw_date), None

    if isinstance(raw_date, str):
        value = raw_date.strip()
        try:
            parsed_date = datetime.strptime(value, "%Y-%m-%d")
            return pd.Timestamp(parsed_date), None
        except ValueError:
            return None, "date must be in YYYY-MM-DD format"

    return None, "date must be in YYYY-MM-DD format"


def _parse_price_cell(raw_price: Any) -> tuple[Optional[float], Optional[str]]:
    if _is_blank_cell(raw_price):
        return None, None

    try:
        parsed_price = (
            float(raw_price.strip()) if isinstance(raw_price, str) else float(raw_price)
        )
    except (TypeError, ValueError):
        return None, "price must be a numeric value greater than 0"

    if not np.isfinite(parsed_price) or parsed_price <= 0:
        return None, "price must be a numeric value greater than 0"

    return parsed_price, None


def _extract_validated_rows(
    non_empty_rows: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[str]]:
    row_errors: list[str] = []
    valid_rows: list[dict[str, Any]] = []

    for original_idx, row in non_empty_rows.iterrows():
        row_idx = int(original_idx) + 2
        issues: list[str] = []

        parsed_price, price_issue = _parse_price_cell(row["price"])
        if price_issue:
            issues.append(price_issue)

        # Missing prices are allowed per-row; these rows are ignored.
        if parsed_price is None and not price_issue:
            continue

        parsed_date, date_issue = _parse_date_cell(row["date"])
        if date_issue:
            issues.append(date_issue)

        if issues:
            row_errors.append(f"Row {row_idx}: {', '.join(issues)}")
            continue

        valid_rows.append(
            {
                "date": pd.to_datetime(parsed_date).normalize(),
                "price": parsed_price,
            }
        )

    return valid_rows, row_errors


def parse_uploaded_price_excel(
    file_bytes: bytes,
    filename: Optional[str],
    max_days: Optional[int] = None,
) -> pd.DataFrame:
    """Parse uploaded Excel and return standardized [date, price] rows."""
    safe_name = (filename or "").lower()
    if safe_name and not safe_name.endswith(SUPPORTED_EXTENSIONS):
        raise ValueError("File must be an Excel file (.xlsx or .xls)")

    try:
        raw = pd.read_excel(BytesIO(file_bytes))
    except Exception as exc:
        raise ValueError(f"Failed to parse Excel file: {exc}") from exc

    if raw.empty:
        raise ValueError("Uploaded Excel file is empty")

    date_col = _pick_column(list(raw.columns), _DATE_COL_CANDIDATES)
    price_col = _pick_column(list(raw.columns), _PRICE_COL_CANDIDATES)

    if date_col is None or price_col is None:
        raise ValueError(
            "Excel must include date and price columns. "
            "Accepted date columns: date/day/timestamp. "
            "Accepted price columns: price/close/closing_price."
        )

    parsed = raw[[date_col, price_col]].copy()
    parsed.columns = ["date", "price"]

    non_empty_rows = parsed[
        ~(parsed["date"].apply(_is_blank_cell) & parsed["price"].apply(_is_blank_cell))
    ].copy()

    if non_empty_rows.empty:
        raise ValueError(
            "No price data rows found. Add at least one row with both date and price values."
        )

    if non_empty_rows["price"].apply(_is_blank_cell).all():
        raise ValueError(
            "All price values are missing. Add at least one numeric price greater than 0."
        )

    if max_days is not None and len(non_empty_rows) > int(max_days):
        raise ValueError(
            f"Upload supports at most {int(max_days)} days, but {len(non_empty_rows)} rows were provided."
        )

    valid_rows, row_errors = _extract_validated_rows(non_empty_rows)

    if row_errors:
        error_summary = _summarize_row_errors(row_errors, len(row_errors))
        raise ValueError(f"Upload validation failed: {error_summary}")

    if not valid_rows:
        raise ValueError(
            "No valid date/price rows found in uploaded Excel. Ensure date uses YYYY-MM-DD and price is numeric > 0."
        )

    out = pd.DataFrame(valid_rows)
    out = out.drop_duplicates(subset=["date"], keep="last")
    out = out.sort_values("date").reset_index(drop=True)
    return out


def _build_lookback_price_window(
    uploaded_prices: pd.DataFrame,
    lookback_days: int,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Build a full lookback window by preferring uploaded prices and filling from DB."""
    end_date = pd.to_datetime(uploaded_prices["date"].max()).normalize()
    start_date = end_date - pd.Timedelta(days=lookback_days - 1)

    window_dates = pd.date_range(start=start_date, end=end_date, freq="D")
    window_df = pd.DataFrame({"date": window_dates})

    db_prices = get_prices_for_date_range(
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
    )

    uploaded = uploaded_prices.rename(columns={"price": "uploaded_price"})
    uploaded = uploaded.drop_duplicates(subset=["date"], keep="last")
    merged = window_df.merge(uploaded, on="date", how="left", validate="one_to_one")

    if not db_prices.empty:
        db_subset = db_prices[["date", "price"]].rename(columns={"price": "db_price"})
        db_subset = db_subset.drop_duplicates(subset=["date"], keep="last")
        merged = merged.merge(db_subset, on="date", how="left", validate="one_to_one")
    else:
        merged["db_price"] = np.nan

    merged["price"] = merged["uploaded_price"].combine_first(merged["db_price"])
    raw_missing_count = int(merged["price"].isna().sum())

    merged["price"] = merged["price"].ffill().bfill()
    remaining_missing_count = int(merged["price"].isna().sum())
    if remaining_missing_count > 0:
        raise ValueError(
            "Insufficient data to build full lookback window. "
            "Upload more rows or ensure DB has enough nearby history."
        )

    source = np.where(
        merged["uploaded_price"].notna(),
        "uploaded",
        np.where(merged["db_price"].notna(), "database", "carry_fill"),
    )

    out = merged[["date", "price"]].copy()
    out["source"] = source

    filled_from_db = int(
        (merged["uploaded_price"].isna() & merged["db_price"].notna()).sum()
    )
    filled_by_carry = max(raw_missing_count - remaining_missing_count, 0)

    stats: Dict[str, Any] = {
        "lookback_days": lookback_days,
        "window_start": start_date.strftime("%Y-%m-%d"),
        "window_end": end_date.strftime("%Y-%m-%d"),
        "uploaded_rows_used": int((merged["uploaded_price"].notna()).sum()),
        "filled_from_database": filled_from_db,
        "filled_by_carry": int(filled_by_carry),
    }

    return out, stats


def _build_sentiment_window(
    start_date: pd.Timestamp, end_date: pd.Timestamp
) -> pd.DataFrame:
    """Fetch and align sentiment rows for the same period used by prediction."""
    sentiment_start = (start_date - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    sentiment_end = (end_date - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    sentiment_df = get_sentiment_for_dates(sentiment_start, sentiment_end)

    target_dates = pd.date_range(
        start=start_date - pd.Timedelta(days=1),
        end=end_date - pd.Timedelta(days=1),
        freq="D",
    )
    target = pd.DataFrame({"date": target_dates})

    if sentiment_df is None or sentiment_df.empty:
        target["daily_sentiment"] = 0.0
        target["news_volume"] = 0.0
        target["log_news_volume"] = 0.0
        target["decayed_news_volume"] = 0.0
        target["high_news_regime"] = 0
        return target

    sent = sentiment_df.copy()
    sent["date"] = pd.to_datetime(sent["date"]).dt.normalize()

    sent = sent.drop_duplicates(subset=["date"], keep="last")
    merged = target.merge(sent, on="date", how="left", validate="one_to_one")

    for col in [
        "daily_sentiment",
        "news_volume",
        "log_news_volume",
        "decayed_news_volume",
    ]:
        if col not in merged.columns:
            merged[col] = 0.0
        merged[col] = (
            pd.to_numeric(merged[col], errors="coerce").ffill().bfill().fillna(0.0)
        )

    if "high_news_regime" not in merged.columns:
        merged["high_news_regime"] = 0
    merged["high_news_regime"] = (
        pd.to_numeric(merged["high_news_regime"], errors="coerce")
        .fillna(0)
        .round()
        .clip(lower=0, upper=1)
        .astype(int)
    )

    return merged


def run_prediction_from_uploaded_excel(
    file_bytes: bytes,
    filename: Optional[str],
) -> Dict[str, Any]:
    """Run prediction from uploaded Excel without storing uploaded rows in DB."""
    lookback = int(model_artifacts.lookback)
    uploaded_df = parse_uploaded_price_excel(
        file_bytes=file_bytes,
        filename=filename,
        max_days=lookback,
    )

    price_window_df, upload_stats = _build_lookback_price_window(
        uploaded_prices=uploaded_df,
        lookback_days=lookback,
    )

    start_date = pd.to_datetime(price_window_df["date"].min()).normalize()
    end_date = pd.to_datetime(price_window_df["date"].max()).normalize()

    sentiment_window_df = _build_sentiment_window(start_date, end_date)

    forecasts = prediction_service.predict(
        prices=price_window_df[["date", "price"]].copy(),
        sentiment_df=sentiment_window_df,
    )

    return {
        "data_source": "Uploaded Excel + Database Backfill + Sentiment History",
        "last_price_date": end_date.strftime("%Y-%m-%d"),
        "last_price": round(float(price_window_df["price"].iloc[-1]), 2),
        "forecasts": [
            f.model_dump() if hasattr(f, "model_dump") else dict(f) for f in forecasts
        ],
        "upload_window": upload_stats,
        "resolved_price_window": [
            {
                "date": pd.to_datetime(row["date"]).strftime("%Y-%m-%d"),
                "price": round(float(row["price"]), 4),
                "source": str(row["source"]),
            }
            for _, row in price_window_df.iterrows()
        ],
    }
