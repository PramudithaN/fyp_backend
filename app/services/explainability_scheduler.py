"""
Daily explainability job scheduler using APScheduler.

Runs the explainability pipeline once per day at 06:00 UTC.
Integrates with FastAPI lifespan to start/stop scheduler with app.
"""

import logging
from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

logger = logging.getLogger(__name__)

# Global scheduler instance
_scheduler: AsyncIOScheduler | None = None


def get_scheduler() -> AsyncIOScheduler | None:
    """Get the global scheduler instance."""
    return _scheduler


def init_scheduler() -> AsyncIOScheduler:
    """
    Initialize and start the APScheduler.

    Should be called during FastAPI lifespan startup.
    """
    global _scheduler

    if _scheduler is not None and _scheduler.running:
        logger.info("Scheduler already running, skipping init")
        return _scheduler

    _scheduler = AsyncIOScheduler()

    # Primary job: Daily explainability at 06:00 UTC
    _scheduler.add_job(
        run_daily_explainability_job,
        CronTrigger(hour=6, minute=0),
        id="daily_explainability",
        name="Daily Explainability Pipeline (Primary)",
        replace_existing=True,
    )

    # Retry job: If prices delayed, retry at 08:00 UTC
    _scheduler.add_job(
        run_daily_explainability_job,
        CronTrigger(hour=8, minute=0),
        id="daily_explainability_retry",
        name="Daily Explainability Pipeline (Retry)",
        replace_existing=True,
    )

    _scheduler.start()
    logger.info(
        "APScheduler started with daily explainability jobs:"
        "\n  - Primary: 06:00 UTC"
        "\n  - Retry: 08:00 UTC (if prices delayed)"
    )

    return _scheduler


def shutdown_scheduler() -> None:
    """
    Shutdown the APScheduler.

    Should be called during FastAPI lifespan shutdown.
    """
    global _scheduler

    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=True)
        logger.info("APScheduler shut down successfully")
        _scheduler = None


def run_daily_explainability_job() -> None:
    """
    Execute the daily explainability pipeline.

    This is the job function scheduled by APScheduler.
    Runs at 06:00 UTC daily.
    
    If prices aren't available yet, logs a warning and defers.
    Subsequent retries can happen via manual trigger or scheduler.
    """
    logger.info(f"Daily explainability job triggered at {datetime.now().isoformat()}")

    try:
        from app.services.explainability import explainability_service

        result = explainability_service.run_daily_job()
        
        if result.get("status") == "deferred":
            logger.warning(
                f"Explainability job deferred: {result.get('reason')}. "
                f"Prices not ready yet. Will retry next hour."
            )
        elif result.get("status") == "success":
            logger.info(f"Daily explainability job completed: {result}")
        else:
            logger.warning(f"Daily explainability job result: {result}")

    except Exception as e:
        logger.error(f"Daily explainability job failed: {e}", exc_info=True)


def trigger_explainability_job_now() -> dict:
    """
    Manually trigger the explainability job (for testing/admin).

    Returns:
        Result dict from run_daily_job.
    """
    logger.info("Manual explainability job trigger requested")

    try:
        from app.services.explainability import explainability_service

        result = explainability_service.run_daily_job()
        return {"status": "success", "result": result}

    except Exception as e:
        logger.error(f"Manual trigger failed: {e}", exc_info=True)
        return {"status": "failed", "error": str(e)}
