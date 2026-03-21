import logging
from datetime import datetime
from typing import Any, Dict

from app.database import (
    clear_news_article_image_urls,
    get_news_articles_missing_image_url,
    init_database,
    update_news_article_image_url,
)
from app.services.news_fetcher import _resolve_image_url_from_headline

logger = logging.getLogger(__name__)


def validate_backfill_date(date_str: str) -> str:
    datetime.strptime(date_str, "%Y-%m-%d")
    return date_str


def backfill_news_image_urls(
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int | None = None,
    reset: bool = False,
) -> dict:
    init_database()

    if reset:
        cleared = clear_news_article_image_urls(start_date=start_date, end_date=end_date)
        logger.info(
            "Reset %d existing image_url values (date range: %s -> %s)",
            cleared,
            start_date or "beginning",
            end_date or "latest",
        )

    missing = get_news_articles_missing_image_url(
        start_date=start_date,
        end_date=end_date,
        limit=limit,
    )

    logger.info("Found %d articles with missing image_url", len(missing))
    if not missing:
        return {
            "start_date": start_date,
            "end_date": end_date,
            "limit": limit,
            "reset": reset,
            "total_processed": 0,
            "updated": 0,
            "no_result": 0,
            "errors": 0,
        }

    pexels_cache: Dict[str, Any] = {}
    updated = 0
    no_result = 0
    errors = 0

    for idx, row in enumerate(missing, start=1):
        article_id = int(row["id"])
        title = str(row.get("title") or "")

        try:
            image_url = _resolve_image_url_from_headline(title=title, cache=pexels_cache)

            if image_url and update_news_article_image_url(article_id, image_url):
                updated += 1
            else:
                no_result += 1

            if idx % 50 == 0:
                logger.info(
                    "Image backfill progress %d/%d | updated=%d | no_result=%d | errors=%d",
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
        "start_date": start_date,
        "end_date": end_date,
        "limit": limit,
        "reset": reset,
        "total_processed": len(missing),
        "updated": updated,
        "no_result": no_result,
        "errors": errors,
    }