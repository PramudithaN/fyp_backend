"""
Standalone daily scraper script.

Invoked by the Render Cron Job (or any external scheduler) once per day.
Runs the full scraping + sentiment pipeline and exits.

Usage:
    python scripts/run_scraper.py [YYYY-MM-DD]

If no date is provided, defaults to yesterday's date.
Exit codes: 0 = success, 1 = error
"""

import sys
import logging
from datetime import datetime, timedelta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    target_date = None
    if len(sys.argv) > 1:
        raw = sys.argv[1]
        try:
            datetime.strptime(raw, "%Y-%m-%d")
            target_date = raw
        except ValueError:
            logger.error("Invalid date argument '%s'. Expected YYYY-MM-DD.", raw)
            return 1

    if target_date is None:
        target_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    logger.info("=== Daily scraper starting for %s ===", target_date)

    try:
        # Bootstrap DB so it exists before we write to it
        from app.database import init_database
        init_database()

        from app.services.scraper_scheduler import run_scraper_now
        result = run_scraper_now(target_date=target_date)
    except Exception:
        logger.error("Scraper script failed with an unexpected error", exc_info=True)
        return 1

    status = result.get("status")
    articles = result.get("articles_found", 0)
    sentiment = result.get("sentiment_value")
    decay = result.get("decay_applied", False)
    error = result.get("error")

    if status == "success":
        if decay:
            logger.info(
                "=== Done: no articles found — decay applied, sentiment=%.4f ===",
                sentiment or 0.0,
            )
        else:
            logger.info(
                "=== Done: %d articles, sentiment=%.4f ===",
                articles,
                sentiment or 0.0,
            )
        return 0
    else:
        logger.error("=== Scrape failed: %s ===", error)
        return 1


if __name__ == "__main__":
    sys.exit(main())
