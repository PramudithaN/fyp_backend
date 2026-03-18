"""
Backfill image URLs for existing news articles using headline keyword extraction + Pexels.

Usage examples:
    python scripts/backfill_news_images.py
    python scripts/backfill_news_images.py --start-date 2025-12-01
    python scripts/backfill_news_images.py --start-date 2025-12-01 --end-date 2026-03-18
    python scripts/backfill_news_images.py --start-date 2025-12-01 --reset

Flags:
    --reset   Clear existing image_url values in the date range before re-fetching,
              so already-processed rows are also re-evaluated with the latest query logic.
"""

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from app.services.news_image_backfill import (
    backfill_news_image_urls,
    validate_backfill_date,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill image_url for historical news rows")
    parser.add_argument("--start-date", default="2025-12-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="End date YYYY-MM-DD (defaults to today)")
    parser.add_argument("--limit", type=int, default=None, help="Optional max rows to process")
    parser.add_argument(
        "--reset",
        action="store_true",
        default=False,
        help="Clear existing image_url values in the date range before re-fetching",
    )
    args = parser.parse_args()

    try:
        start_date = validate_backfill_date(args.start_date)
        end_date = validate_backfill_date(args.end_date) if args.end_date else None
    except ValueError:
        logger.error("Invalid date format. Expected YYYY-MM-DD")
        return 1

    summary = backfill_news_image_urls(
        start_date=start_date,
        end_date=end_date,
        limit=args.limit,
        reset=args.reset,
    )
    logger.info("Backfill summary: %s", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
