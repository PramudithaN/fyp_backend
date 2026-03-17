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
from datetime import datetime
from typing import Dict

from app.database import (
    init_database,
    get_news_articles_missing_image_url,
    update_news_article_image_url,
    clear_news_article_image_urls,
)
from app.services.news_fetcher import _resolve_image_url_from_headline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _validate_date(date_str: str) -> str:
    datetime.strptime(date_str, "%Y-%m-%d")
    return date_str


def run_backfill(
    start_date: str,
    end_date: str | None,
    limit: int | None,
    reset: bool = False,
) -> dict:
    init_database()

    if reset:
        cleared = clear_news_article_image_urls(start_date=start_date, end_date=end_date)
        logger.info("Reset %d existing image_url values (date range: %s → %s)", cleared, start_date, end_date or "today")

    missing = get_news_articles_missing_image_url(
        start_date=start_date,
        end_date=end_date,
        limit=limit,
    )

    logger.info("Found %d articles to process", len(missing))
    if not missing:
        return {
            "total_processed": 0,
            "updated": 0,
            "no_result": 0,
            "errors": 0,
        }

    # Shared query→url cache so the same Pexels query isn't called twice.
    pexels_cache: Dict[str, str] = {}
    updated = 0
    no_result = 0
    errors = 0

    for idx, row in enumerate(missing, start=1):
        article_id = int(row["id"])
        title = str(row.get("title") or "")

        try:
            image_url = _resolve_image_url_from_headline(title, cache=pexels_cache)

            if image_url:
                if update_news_article_image_url(article_id, image_url):
                    updated += 1
                else:
                    no_result += 1
            else:
                no_result += 1

            if idx % 50 == 0:
                logger.info(
                    "Progress %d/%d | updated=%d | no_result=%d | errors=%d",
                    idx,
                    len(missing),
                    updated,
                    no_result,
                    errors,
                )
        except Exception as exc:
            errors += 1
            logger.warning("Failed row id=%s title=%r: %s", article_id, title[:100], exc)

    return {
        "total_processed": len(missing),
        "updated": updated,
        "no_result": no_result,
        "errors": errors,
    }


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
        start_date = _validate_date(args.start_date)
        end_date = _validate_date(args.end_date) if args.end_date else None
    except ValueError:
        logger.error("Invalid date format. Expected YYYY-MM-DD")
        return 1

    summary = run_backfill(start_date=start_date, end_date=end_date, limit=args.limit, reset=args.reset)
    logger.info("Backfill summary: %s", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
