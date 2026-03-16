"""
Import historical price and news-feature datasets into Turso.

Default approach:
- Store price history in historical_prices
- Store news features in historical_news_features
- Use historical_features_combined SQL view when you need merged features by date

Usage:
    python scripts/import_historical_data.py \
      --price-file "D:/IIT/FYP/IPD/Datasets/Brent Oil Futures Historical Data.csv" \
      --news-file "D:/IIT/FYP/IPD/Datasets/data_with_news_features.csv"

Supports .csv, .xlsx, and .xls input files.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from app.database import (
    init_database,
    add_bulk_historical_prices,
    add_bulk_historical_news_features,
)

logger = logging.getLogger(__name__)


def _read_dataset(file_path: str) -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    raise ValueError(
        f"Unsupported file extension: {suffix}. Use .csv, .xlsx, or .xls"
    )


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def _normalize_date_column(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    return parsed.dt.strftime("%Y-%m-%d")


def _price_records_from_df(df: pd.DataFrame, source_label: str) -> List[Dict[str, Any]]:
    df = _normalize_columns(df)

    rename_map = {
        "close": "price",
        "close/last": "price",
        "last": "price",
        "vol.": "volume",
        "change %": "change_pct",
        "change%": "change_pct",
    }
    df = df.rename(columns=rename_map)

    required = {"date", "price"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Price file missing required columns: {sorted(missing)}")

    df["date"] = _normalize_date_column(df["date"])
    df = df[df["date"].notna()].copy()

    for col in ["open", "high", "low", "volume", "change_pct"]:
        if col not in df.columns:
            df[col] = None

    records = []
    for _, row in df.iterrows():
        records.append(
            {
                "date": row["date"],
                "price": row["price"],
                "open": row.get("open"),
                "high": row.get("high"),
                "low": row.get("low"),
                "volume": row.get("volume"),
                "change_pct": row.get("change_pct"),
                "source": source_label,
            }
        )

    return records


def _news_feature_records_from_df(
    df: pd.DataFrame,
    source_label: str,
) -> List[Dict[str, Any]]:
    df = _normalize_columns(df)

    required = {
        "date",
        "daily_sentiment_decay",
        "news_volume",
        "log_news_volume",
        "decayed_news_volume",
        "high_news_regime",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"News feature file missing required columns: {sorted(missing)}")

    df["date"] = _normalize_date_column(df["date"])
    df = df[df["date"].notna()].copy()

    records = []
    for _, row in df.iterrows():
        records.append(
            {
                "date": row["date"],
                "daily_sentiment_decay": row["daily_sentiment_decay"],
                "news_volume": row["news_volume"],
                "log_news_volume": row["log_news_volume"],
                "decayed_news_volume": row["decayed_news_volume"],
                "high_news_regime": row["high_news_regime"],
                "source": source_label,
            }
        )

    return records


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import historical price and news-feature datasets into Turso"
    )
    parser.add_argument("--price-file", required=True, help="Path to price CSV/Excel")
    parser.add_argument("--news-file", required=True, help="Path to news features CSV/Excel")
    parser.add_argument(
        "--source-label",
        default="historical_import",
        help="Source label stored in historical tables",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("Initializing database schema")
    init_database()

    logger.info("Reading price file: %s", args.price_file)
    price_df = _read_dataset(args.price_file)
    price_records = _price_records_from_df(price_df, args.source_label)

    logger.info("Reading news feature file: %s", args.news_file)
    news_df = _read_dataset(args.news_file)
    news_records = _news_feature_records_from_df(news_df, args.source_label)

    inserted_prices = add_bulk_historical_prices(
        price_records,
        default_source=args.source_label,
    )
    inserted_news = add_bulk_historical_news_features(
        news_records,
        default_source=args.source_label,
    )

    logger.info("Import completed")
    logger.info("Historical prices upserted: %d", inserted_prices)
    logger.info("Historical news features upserted: %d", inserted_news)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
