"""
FastAPI application for Brent Oil Price Prediction.

Endpoints:
- GET  /predict     - Auto-fetch prices and predict
- POST /predict     - Predict with custom prices
- GET  /prices      - View fetched price data
- GET  /health      - Health check
- POST /sentiment/add   - Add daily sentiment
- POST /sentiment/bulk  - Bulk upload sentiment
- GET  /sentiment       - View sentiment history
"""

import logging
import os
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, date, time, timedelta
from functools import partial
from threading import RLock
from time import monotonic
from typing import Annotated
from zoneinfo import ZoneInfo

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Header, UploadFile, File, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from prometheus_fastapi_instrumentator import Instrumentator

from app.config import (
    API_TITLE,
    API_DESCRIPTION,
    API_VERSION,
    BRENT_TICKER,
    PREDICT_CACHE_TTL_SECONDS,
    PREDICTION_PRECOMPUTE_ENABLED,
    PREDICTION_PRECOMPUTE_INTERVAL_SECONDS,
    PREDICTION_LOCK_SCHEDULE_HOUR,
    PREDICTION_LOCK_SCHEDULE_MINUTE,
    PREDICTION_LOCK_SCHEDULE_TIMEZONE,
    SKIP_FINBERT_PRELOAD,
    SCRAPER_API_KEY,
)
from app.models.model_loader import model_artifacts
from app.database import (
    add_prediction,
    init_database,
    get_actual_vs_predicted_until,
    get_latest_locked_prediction,
    get_latest_prediction_fan_chart,
    get_prediction_for_date,
    get_news_articles,
    get_recent_news_articles,
)
from app.services.price_fetcher import (
    fetch_latest_prices,
    fetch_live_price_snapshot,
    get_market_status,
)
from app.services.prediction import prediction_service
from app.services.prediction_scheduler import (
    init_prediction_scheduler,
    shutdown_prediction_scheduler,
    trigger_prediction_job_now,
)
from app.services.upload_prediction import (
    run_prediction_from_uploaded_excel,
    build_upload_excel_template_bytes,
)
from app.services.sentiment_service import sentiment_service
from app.services.news_image_backfill import (
    backfill_news_image_urls,
    validate_backfill_date,
)
from app.services.scraper_scheduler import (
    get_scheduler_status,
    run_scraper_now,
    backfill_history,
)
from app.schemas.prediction import (
    PredictionRequest,
    PredictionResponse,
    UploadPredictionResponse,
    PredictionComparisonResponse,
    PriceDataResponse,
    HealthResponse,
    ErrorResponse,
    SentimentInput,
    BulkSentimentRequest,
    SentimentAddResponse,
    SentimentHistoryResponse,
    HistoricalPricesResponse,
    HistoricalCombinedFeaturesResponse,
    PredictionFanResponse,
    NewsArticlesResponse,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

INVALID_DATE_DETAIL = "Invalid date format. Expected YYYY-MM-DD"

_PRICE_SYNC_CACHE_TTL_SECONDS = 300.0
_price_sync_cache_lock = RLock()
_price_sync_cache: dict[int, tuple[float, pd.DataFrame]] = {}
_PREDICT_CACHE_TTL_SECONDS = PREDICT_CACHE_TTL_SECONDS
_PREDICT_CACHE_MAX_STALE_SECONDS = max(
    float(os.getenv("PREDICT_CACHE_MAX_STALE_SECONDS", "600")),
    _PREDICT_CACHE_TTL_SECONDS,
)
_predict_cache_lock = RLock()
_predict_cache: tuple[float, PredictionResponse] | None = None
_prediction_refresh_lock = asyncio.Lock()
_prediction_background_refresh_task: asyncio.Task | None = None

# Market status caching (reduces Yahoo Finance API calls)
_MARKET_STATUS_CACHE_TTL_SECONDS = 60.0  # Cache for 60 seconds
_market_status_cache_lock = RLock()
_market_status_cache: tuple[float, dict] | None = None


def _next_business_day(date_value: pd.Timestamp) -> pd.Timestamp:
    """Advance to the next weekday, skipping weekends."""
    next_date = pd.to_datetime(date_value) + pd.Timedelta(days=1)
    while next_date.weekday() >= 5:
        next_date += pd.Timedelta(days=1)
    return next_date


def _align_forecast_dates_to_last_price(
    forecasts: list[dict],
    last_price_date: str,
) -> list[dict]:
    """Ensure forecast dates start strictly after the exposed last known price date."""
    current_date = pd.to_datetime(last_price_date)
    aligned_forecasts: list[dict] = []

    for forecast in forecasts:
        current_date = _next_business_day(current_date)
        aligned_forecasts.append(
            {
                **forecast,
                "date": current_date.strftime("%Y-%m-%d"),
            }
        )

    return aligned_forecasts


def _ensure_prediction_background_refresh(persist_forecast: bool) -> None:
    """Kick off one cache refresh task if no refresh is currently running."""
    global _prediction_background_refresh_task

    if _prediction_background_refresh_task and not _prediction_background_refresh_task.done():
        return

    async def _refresh_task_runner():
        global _prediction_background_refresh_task
        try:
            await _refresh_prediction_cache(force_refresh=True, persist_forecast=persist_forecast)
        except Exception as exc:
            logger.warning("Background prediction cache refresh failed: %s", exc, exc_info=True)
        finally:
            _prediction_background_refresh_task = None

    _prediction_background_refresh_task = asyncio.create_task(_refresh_task_runner())


async def _build_prediction_response(persist_forecast: bool = True) -> PredictionResponse:
    """Run full prediction pipeline and build API response payload."""
    latest_prices = await run_in_threadpool(_sync_latest_prices_cached, 120)
    latest_prices["date"] = pd.to_datetime(latest_prices["date"])
    latest_prices = latest_prices.sort_values("date").reset_index(drop=True)

    # Generate predictions using the refreshed price history.
    forecasts = await run_in_threadpool(partial(prediction_service.predict, prices=latest_prices))

    close_price = float(latest_prices["price"].iloc[-1])
    close_date = pd.to_datetime(latest_prices["date"].iloc[-1]).strftime("%Y-%m-%d")

    # Use intraday quote as current last known price when available,
    # but do not persist intraday rows into prices table.
    live_snapshot = await run_in_threadpool(fetch_live_price_snapshot)
    if live_snapshot and float(live_snapshot["price"]) > 0:
        last_price = float(live_snapshot["price"])
        last_date = str(live_snapshot["as_of_date"])
        if close_price > 0:
            scale = last_price / close_price
            forecasts = [
                {
                    **f,
                    "forecasted_price": round(float(f["forecasted_price"]) * scale, 2),
                }
                for f in forecasts
            ]
    else:
        last_price = close_price
        last_date = close_date

    forecasts = _align_forecast_dates_to_last_price(forecasts, last_date)

    if persist_forecast:
        try:
            await run_in_threadpool(
                add_prediction,
                datetime.now().isoformat(),
                last_date,
                round(last_price, 2),
                [f.model_dump() if hasattr(f, "model_dump") else f for f in forecasts],
            )
        except Exception as db_err:
            logger.warning(f"Failed to persist prediction: {db_err}")

    # Try cached market status first to avoid hitting Yahoo Finance API repeatedly
    market = _get_cached_market_status()
    if market is None:
        market = await run_in_threadpool(get_market_status)
        _cache_market_status(market)

    return PredictionResponse(
        success=True,
        data_source=f"Yahoo Finance ({BRENT_TICKER})",
        last_price_date=last_date,
        last_price=round(last_price, 2),
        forecasts=forecasts,
        is_market_open=market["is_open"],
        market_open_time=market["market_open_time"],
        market_close_time=market["market_close_time"],
        timezone_info=market["timezone_info"],
    )


async def _refresh_prediction_cache(
    *,
    force_refresh: bool,
    persist_forecast: bool,
) -> PredictionResponse:
    """Return warm prediction data and serialize expensive refresh work."""
    global _predict_cache

    if not force_refresh:
        cached_response = _get_cached_prediction(persist_forecast=persist_forecast)
        if cached_response is not None:
            return cached_response

    async with _prediction_refresh_lock:
        if not force_refresh:
            cached_response = _get_cached_prediction(persist_forecast=persist_forecast)
            if cached_response is not None:
                return cached_response

        response = await _build_prediction_response(persist_forecast=persist_forecast)

        if _cache_enabled():
            with _predict_cache_lock:
                _predict_cache = (monotonic(), response)

        return response


def _get_cached_prediction(*, persist_forecast: bool) -> PredictionResponse | None:
    """Return valid/stale cache entry when available and trigger background refresh if needed."""
    if not _cache_enabled():
        return None

    with _predict_cache_lock:
        if not _predict_cache:
            return None

        cache_age_seconds = monotonic() - _predict_cache[0]
        if cache_age_seconds < _PREDICT_CACHE_TTL_SECONDS:
            return _predict_cache[1]

        if cache_age_seconds < _PREDICT_CACHE_MAX_STALE_SECONDS:
            _ensure_prediction_background_refresh(persist_forecast=persist_forecast)
            return _predict_cache[1]

    return None


def _get_cached_market_status() -> dict | None:
    """Return cached market status if available and fresh (within TTL)."""
    if not _cache_enabled():
        return None

    with _market_status_cache_lock:
        if not _market_status_cache:
            return None

        cache_age_seconds = monotonic() - _market_status_cache[0]
        if cache_age_seconds < _MARKET_STATUS_CACHE_TTL_SECONDS:
            return _market_status_cache[1]

    return None


def _cache_market_status(market_status: dict) -> None:
    """Store market status in cache with current timestamp."""
    if not _cache_enabled():
        return

    with _market_status_cache_lock:
        global _market_status_cache
        _market_status_cache = (monotonic(), market_status)


async def _prediction_precompute_loop(stop_event: asyncio.Event):
    """Refresh prediction cache on a fixed interval to reduce request-time latency."""
    interval_seconds = max(PREDICTION_PRECOMPUTE_INTERVAL_SECONDS, 60)
    logger.info(
        "Prediction precompute loop started (interval=%ss)",
        interval_seconds,
    )

    while not stop_event.is_set():
        try:
            await _refresh_prediction_cache(force_refresh=True, persist_forecast=True)
        except Exception as exc:
            logger.warning("Background prediction precompute failed: %s", exc, exc_info=True)

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_seconds)
        except asyncio.TimeoutError:
            continue

    logger.info("Prediction precompute loop stopped")


def _cache_enabled() -> bool:
    """Disable in-memory caching during pytest to preserve deterministic mocks."""
    return "PYTEST_CURRENT_TEST" not in os.environ


def _current_prediction_date_local() -> str:
    """Return prediction date key using the configured scheduler timezone."""
    tz = ZoneInfo(PREDICTION_LOCK_SCHEDULE_TIMEZONE)
    return datetime.now(tz).strftime("%Y-%m-%d")


def _next_locked_update_iso(now_local: datetime | None = None) -> str:
    """Return next scheduled lock refresh timestamp in configured local timezone."""
    tz = ZoneInfo(PREDICTION_LOCK_SCHEDULE_TIMEZONE)
    local_now = now_local.astimezone(tz) if now_local is not None else datetime.now(tz)

    next_update = datetime.combine(
        local_now.date(),
        time(hour=PREDICTION_LOCK_SCHEDULE_HOUR, minute=PREDICTION_LOCK_SCHEDULE_MINUTE),
        tzinfo=tz,
    )
    if local_now >= next_update:
        next_update = next_update + timedelta(days=1)

    return next_update.isoformat()


def _prediction_response_from_record(record: dict, market: dict) -> PredictionResponse:
    """Map a stored locked prediction record to PredictionResponse."""
    based_on_price_date = record.get("based_on_price_date") or record.get("last_price_date")
    based_on_price = record.get("based_on_price")
    if based_on_price is None:
        based_on_price = record.get("last_price", 0.0)

    forecasts = record.get("forecasts") or []

    return PredictionResponse(
        success=True,
        data_source=f"Yahoo Finance ({BRENT_TICKER})",
        last_price_date=str(based_on_price_date),
        last_price=round(float(based_on_price), 2),
        forecasts=forecasts,
        is_market_open=market["is_open"],
        market_open_time=market["market_open_time"],
        market_close_time=market["market_close_time"],
        timezone_info=market["timezone_info"],
        generated_at=str(record.get("locked_at") or record.get("generated_at") or ""),
        prediction_date=str(record.get("prediction_date") or ""),
        based_on_price_date=str(based_on_price_date),
        based_on_price=round(float(based_on_price), 2),
        next_update_at=_next_locked_update_iso(),
    )


def _sync_latest_prices(lookback_days: int = 120) -> pd.DataFrame:
    """Fetch the latest available market prices and upsert them into the database."""
    from app.database import add_bulk_prices, get_prices as get_prices_db

    try:
        latest_prices = fetch_latest_prices(lookback_days=lookback_days)
        records = [
            {
                "date": pd.to_datetime(row["date"]).strftime("%Y-%m-%d"),
                "price": float(row["price"]),
                "source": "yahoo_finance",
            }
            for row in latest_prices[["date", "price"]].to_dict(orient="records")
        ]
        if records:
            add_bulk_prices(records)
            normalized = latest_prices[["date", "price"]].copy()
            normalized["date"] = pd.to_datetime(normalized["date"]).dt.tz_localize(None)
            normalized["source"] = "yahoo_finance"
            return normalized.sort_values("date").reset_index(drop=True)
    except Exception as exc:
        logger.warning("Failed to refresh live prices, falling back to stored data: %s", exc)
        if not _cache_enabled():
            raise ValueError(f"Failed to refresh live prices: {exc}")

    stored_prices = get_prices_db(days=lookback_days)
    if stored_prices.empty:
        raise ValueError("No price data available for prediction")
    return stored_prices


def _sync_latest_prices_cached(lookback_days: int = 120) -> pd.DataFrame:
    """Return recent synced prices, reusing a short-lived in-memory cache."""
    if not _cache_enabled():
        return _sync_latest_prices(lookback_days=lookback_days)

    now_ts = monotonic()

    with _price_sync_cache_lock:
        cached = _price_sync_cache.get(lookback_days)
        if cached and (now_ts - cached[0]) < _PRICE_SYNC_CACHE_TTL_SECONDS:
            return cached[1].copy()

    fresh = _sync_latest_prices(lookback_days=lookback_days)

    with _price_sync_cache_lock:
        _price_sync_cache[lookback_days] = (now_ts, fresh.copy())

    return fresh


def _aggregate_historical_prices(df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    """Aggregate historical price series for chart-friendly granularity."""
    if df.empty or granularity == "daily":
        return df

    working = df.copy().sort_values("date")
    working["date"] = pd.to_datetime(working["date"])
    working = working.set_index("date")

    rule = "W" if granularity == "weekly" else "MS"

    aggregated = pd.DataFrame(
        {
            "open": working["open"].resample(rule).first(),
            "high": working["high"].resample(rule).max(),
            "low": working["low"].resample(rule).min(),
            "price": working["price"].resample(rule).last(),
            "volume": working["volume"].resample(rule).sum(),
            "change_pct": working["change_pct"].resample(rule).mean(),
            "source": working["source"].resample(rule).first(),
        }
    )
    aggregated = aggregated.dropna(subset=["price"]).reset_index()
    return aggregated


def _aggregate_historical_features(df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    """Aggregate combined historical price + news features by period."""
    if df.empty or granularity == "daily":
        return df

    working = df.copy().sort_values("date")
    working["date"] = pd.to_datetime(working["date"])
    working = working.set_index("date")

    rule = "W" if granularity == "weekly" else "MS"

    aggregated = pd.DataFrame(
        {
            "open": working["open"].resample(rule).first(),
            "high": working["high"].resample(rule).max(),
            "low": working["low"].resample(rule).min(),
            "price": working["price"].resample(rule).last(),
            "volume": working["volume"].resample(rule).sum(),
            "change_pct": working["change_pct"].resample(rule).mean(),
            "daily_sentiment_decay": working["daily_sentiment_decay"].resample(rule).mean(),
            "news_volume": working["news_volume"].resample(rule).sum(),
            "log_news_volume": working["log_news_volume"].resample(rule).mean(),
            "decayed_news_volume": working["decayed_news_volume"].resample(rule).mean(),
            "high_news_regime": working["high_news_regime"].resample(rule).max(),
        }
    )
    aggregated = aggregated.dropna(subset=["price", "daily_sentiment_decay"]).reset_index()
    return aggregated


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and initialize database on startup."""
    logger.info("Starting application...")
    
    # Initialize sentiment database
    try:
        await run_in_threadpool(init_database)
        logger.info("Sentiment database initialized!")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}", exc_info=True)
        # Continue - database can be created on first request

    # Load ML models
    try:
        await run_in_threadpool(model_artifacts.load_all)
        logger.info("Models loaded successfully!")
    except Exception as e:
        logger.error(f"Model loading failed: {e}", exc_info=True)
        # Continue - predictions will fail but app can still start

    # Pre-load FinBERT sentiment model (eliminates cold-start on first request)
    if not SKIP_FINBERT_PRELOAD:
        try:
            from app.services.finbert_analyzer import preload_model

            await run_in_threadpool(preload_model)
            logger.info("FinBERT model pre-loaded successfully")
        except Exception as e:
            logger.warning(f"FinBERT preload failed: {e}", exc_info=True)
            logger.info("FinBERT model will load on first request")
    else:
        logger.info("FinBERT preload skipped (SKIP_FINBERT_PRELOAD=true)")

    logger.info(
        "Prediction precompute loop disabled: serving locked daily forecasts from database"
    )

    try:
        init_prediction_scheduler()
        logger.info("Daily locked prediction scheduler initialized")
    except Exception as e:
        logger.warning(f"Prediction scheduler initialization failed: {e}", exc_info=True)

    # Initialize explainability scheduler (must run in async context, not threadpool)
    try:
        from app.services.explainability_scheduler import init_scheduler

        init_scheduler()
        logger.info("Explainability scheduler initialized")
    except Exception as e:
        logger.warning(f"Explainability scheduler initialization failed: {e}", exc_info=True)

    logger.info("Application startup completed successfully")

    yield
    
    # Shutdown explainability scheduler
    try:
        from app.services.explainability_scheduler import shutdown_scheduler

        shutdown_scheduler()
    except Exception as e:
        logger.warning(f"Explainability scheduler shutdown failed: {e}")

    try:
        shutdown_prediction_scheduler()
    except Exception as e:
        logger.warning(f"Prediction scheduler shutdown failed: {e}")

    logger.info("Application shutting down...")


# Create FastAPI app
app = FastAPI(
    title=API_TITLE, description=API_DESCRIPTION, version=API_VERSION, lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Expose Prometheus /metrics endpoint
Instrumentator().instrument(app).expose(app, include_in_schema=False, tags=["observability"])


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint - redirect info."""
    return {
        "message": "Oil Price Prediction API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with market status."""
    # Try cached market status first to avoid hitting Yahoo Finance API repeatedly
    market = _get_cached_market_status()
    if market is None:
        market = await run_in_threadpool(get_market_status)
        _cache_market_status(market)
    
    return HealthResponse(
        status="healthy",
        models_loaded=model_artifacts._loaded,
        timestamp=datetime.now().isoformat(),
        version=API_VERSION,
        is_market_open=market["is_open"],
        market_open_time=market["market_open_time"],
        market_close_time=market["market_close_time"],
        timezone_info=market["timezone_info"],
    )


@app.get(
    "/prices",
    response_model=PriceDataResponse,
    responses={
        500: {"model": ErrorResponse, "description": "Server error fetching prices"}
    },
)
async def get_prices():
    """
    Fetch and display the most recent Brent oil price data.

    Refreshes live prices first, then returns the last 30 trading days.
    Falls back to stored prices if the live refresh fails.
    """
    try:
        all_prices = await run_in_threadpool(partial(_sync_latest_prices_cached, lookback_days=60))
        all_prices["date"] = pd.to_datetime(all_prices["date"])
        all_prices = all_prices.sort_values("date").reset_index(drop=True)
        prices = all_prices.tail(30)  # Get last 30 days

        return PriceDataResponse(
            success=True,
            ticker=BRENT_TICKER,
            data_points=len(prices),
            date_range={
                "start": str(prices["date"].iloc[0].date()),
                "end": str(prices["date"].iloc[-1].date()),
            },
            prices=prices.to_dict(orient="records"),
        )

    except Exception as e:
        logger.error(f"Error fetching prices: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/news",
    response_model=NewsArticlesResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid date format"},
        500: {"model": ErrorResponse, "description": "Server error fetching news articles"},
    },
)
async def get_news(
    article_date: Annotated[str | None, Query(pattern=r"^\d{4}-\d{2}-\d{2}$")] = None,
    days: Annotated[int, Query(ge=1)] = 7,
):
    """
    Return stored news articles for the frontend.

    If article_date is provided, returns all articles for that exact date.
    Otherwise returns articles from the most recent N distinct dates.
    """
    if article_date is not None:
        try:
            datetime.strptime(article_date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=INVALID_DATE_DETAIL,
            )

    try:
        if article_date is not None:
            articles = await run_in_threadpool(get_news_articles, article_date)
        else:
            articles = await run_in_threadpool(partial(get_recent_news_articles, days=days))
        latest_article_date = articles[0]["article_date"] if articles else None

        return NewsArticlesResponse(
            success=True,
            total_records=len(articles),
            requested_date=article_date,
            days=1 if article_date is not None else days,
            latest_article_date=latest_article_date,
            articles=articles,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching news articles: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get(
    "/historical/prices",
    response_model=HistoricalPricesResponse,
    responses={
        500: {"model": ErrorResponse, "description": "Server error fetching historical prices"}
    },
)
async def get_historical_prices(
    start_date: Annotated[str | None, Query(pattern=r"^\d{4}-\d{2}-\d{2}$")] = None,
    end_date: Annotated[str | None, Query(pattern=r"^\d{4}-\d{2}-\d{2}$")] = None,
    granularity: Annotated[str, Query(pattern=r"^(daily|weekly|monthly)$")] = "daily",
    limit: Annotated[int, Query(ge=1, le=5000)] = 500,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    """Return imported historical price records from historical_prices table."""
    try:
        from app.database import (
            get_historical_prices_paginated,
            get_historical_prices_aggregated,
        )

        if granularity == "daily":
            # Single Turso round-trip: data + total count via window function
            df, total_available = await run_in_threadpool(
                get_historical_prices_paginated,
                start_date,
                end_date,
                limit,
                offset,
            )
        else:
            # DB-level aggregation + pagination: no full-table fetch into Python
            df, total_available = await run_in_threadpool(
                get_historical_prices_aggregated,
                granularity,
                start_date,
                end_date,
                limit,
                offset,
            )

        if df.empty:
            return HistoricalPricesResponse(
                success=True,
                granularity=granularity,
                total_available=total_available,
                total_records=0,
                limit=limit,
                offset=offset,
                date_range={"start": None, "end": None},
                data=[],
            )

        return HistoricalPricesResponse(
            success=True,
            granularity=granularity,
            total_available=total_available,
            total_records=len(df),
            limit=limit,
            offset=offset,
            date_range={
                "start": str(df["date"].iloc[0].date()),
                "end": str(df["date"].iloc[-1].date()),
            },
            data=df.to_dict(orient="records"),
        )
    except Exception as e:
        logger.error(f"Error fetching historical prices: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/historical/features/combined",
    response_model=HistoricalCombinedFeaturesResponse,
    responses={
        500: {
            "model": ErrorResponse,
            "description": "Server error fetching combined historical features",
        }
    },
)
async def get_historical_features_combined(
    start_date: Annotated[str | None, Query(pattern=r"^\d{4}-\d{2}-\d{2}$")] = None,
    end_date: Annotated[str | None, Query(pattern=r"^\d{4}-\d{2}-\d{2}$")] = None,
    granularity: Annotated[str, Query(pattern=r"^(daily|weekly|monthly)$")] = "daily",
    limit: Annotated[int, Query(ge=1, le=5000)] = 500,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    """Return combined historical price + news features joined by date."""
    try:
        from app.database import (
            get_historical_features_combined as get_historical_features_combined_db,
            get_historical_features_combined_count,
        )

        if granularity == "daily":
            total_available = await run_in_threadpool(
                get_historical_features_combined_count,
                start_date,
                end_date,
            )
            df = await run_in_threadpool(
                get_historical_features_combined_db,
                start_date,
                end_date,
                limit,
                offset,
            )
        else:
            raw_df = await run_in_threadpool(
                get_historical_features_combined_db,
                start_date,
                end_date,
            )
            aggregated_df = _aggregate_historical_features(raw_df, granularity)
            total_available = len(aggregated_df)
            df = aggregated_df.iloc[offset : offset + limit].reset_index(drop=True)

        if df.empty:
            return HistoricalCombinedFeaturesResponse(
                success=True,
                granularity=granularity,
                total_available=total_available,
                total_records=0,
                limit=limit,
                offset=offset,
                date_range={"start": None, "end": None},
                data=[],
            )

        return HistoricalCombinedFeaturesResponse(
            success=True,
            granularity=granularity,
            total_available=total_available,
            total_records=len(df),
            limit=limit,
            offset=offset,
            date_range={
                "start": str(df["date"].iloc[0].date()),
                "end": str(df["date"].iloc[-1].date()),
            },
            data=df.to_dict(orient="records"),
        )
    except Exception as e:
        logger.error(f"Error fetching combined historical features: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {
            "model": ErrorResponse,
            "description": "Invalid input or validation error",
        },
        503: {
            "model": ErrorResponse,
            "description": "No locked forecast is currently available",
        },
        500: {"model": ErrorResponse, "description": "Server error during prediction"},
    },
)
async def predict_now():
    """
    Return the locked daily forecast from database.

    Request path is read-only:
    - Does not fetch Yahoo prices
    - Does not run model inference
    - Does not write new prediction rows

    Selection logic:
    1. Try today's locked prediction_date in scheduler timezone.
    2. If missing, fallback to latest available locked record.
    """
    try:
        # Try cached market status first to avoid hitting Yahoo Finance API on every request
        market = _get_cached_market_status()
        if market is None:
            market = await run_in_threadpool(get_market_status)
            _cache_market_status(market)
        
        today_key = _current_prediction_date_local()

        todays_record = None
        latest_record = None
        try:
            todays_record = await run_in_threadpool(get_prediction_for_date, today_key)
        except Exception as db_err:
            logger.warning("Failed reading today's locked prediction: %s", db_err)

        if todays_record:
            return _prediction_response_from_record(todays_record, market)

        try:
            latest_record = await run_in_threadpool(get_latest_locked_prediction)
        except Exception as db_err:
            logger.warning("Failed reading latest locked prediction: %s", db_err)

        if latest_record:
            return _prediction_response_from_record(latest_record, market)

        raise HTTPException(
            status_code=503,
            detail=(
                "No locked daily forecast available yet. "
                "Wait for the scheduled prediction job or trigger it manually."
            ),
        )

    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/predictions/lock/run",
    responses={
        500: {
            "model": ErrorResponse,
            "description": "Server error during locked prediction generation",
        },
    },
)
async def run_locked_prediction_now():
    """Manually trigger today's locked prediction generation job."""
    try:
        return await run_in_threadpool(trigger_prediction_job_now)
    except Exception as e:
        logger.error("Manual locked prediction trigger failed", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/predict/upload-excel/template",
    responses={
        200: {
            "description": "Excel template file",
            "content": {
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": {}
            },
        }
    },
)
async def get_upload_excel_template():
    """Download strict Excel template for upload-based prediction."""
    template_bytes = await run_in_threadpool(
        build_upload_excel_template_bytes,
        model_artifacts.lookback,
    )

    return Response(
        content=template_bytes,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": "attachment; filename=oil_price_upload_template.xlsx"
        },
    )


@app.post(
    "/predict/upload-excel",
    response_model=UploadPredictionResponse,
    responses={
        400: {
            "model": ErrorResponse,
            "description": "Invalid or insufficient uploaded data",
        },
        500: {
            "model": ErrorResponse,
            "description": "Server error during uploaded prediction",
        },
    },
)
async def predict_from_uploaded_excel(
    file: Annotated[
        UploadFile,
        File(..., description="Excel file containing date/price rows"),
    ],
):
    """
    Run prediction using an uploaded Excel lookback window without storing upload rows.

    Workflow:
    1. Parse uploaded date/price data.
    2. Build the required lookback window from the active model config.
    3. Fill missing dates using existing database price tables (read-only).
    4. Align sentiment using existing sentiment table (read-only).
    5. Run model prediction and return response payload to frontend.
    """
    try:
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        payload = await run_in_threadpool(
            run_prediction_from_uploaded_excel,
            file_bytes,
            file.filename,
        )
        
        # Try cached market status first to avoid hitting Yahoo Finance API repeatedly
        market = _get_cached_market_status()
        if market is None:
            market = await run_in_threadpool(get_market_status)
            _cache_market_status(market)

        return UploadPredictionResponse(
            success=True,
            data_source=payload["data_source"],
            last_price_date=payload["last_price_date"],
            last_price=payload["last_price"],
            forecasts=payload["forecasts"],
            is_market_open=market["is_open"],
            market_open_time=market["market_open_time"],
            market_close_time=market["market_close_time"],
            timezone_info=market["timezone_info"],
            upload_window=payload["upload_window"],
            resolved_price_window=payload["resolved_price_window"],
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Upload Excel prediction error", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/predictions/compare",
    response_model=PredictionComparisonResponse,
    responses={
        400: {
            "model": ErrorResponse,
            "description": "Invalid date format",
        },
        500: {
            "model": ErrorResponse,
            "description": "Server error during comparison",
        },
    },
)
async def compare_predictions_with_actuals(
    start_date: Annotated[str | None, Query(pattern=r"^\d{4}-\d{2}-\d{2}$")] = None,
    end_date: Annotated[str | None, Query(pattern=r"^\d{4}-\d{2}-\d{2}$")] = None,
):
    """
    Compare actual stored prices with stored predictions up to a cutoff date.

    If multiple prediction runs include the same target date, predictions are
    aggregated using a horizon-weighted mean (weight = 1 / horizon), so
    shorter-horizon forecasts have higher influence.

    Args:
        start_date: Optional YYYY-MM-DD start date. Defaults to earliest available.
        end_date: Optional YYYY-MM-DD cutoff date. Defaults to today.
    """
    if start_date is not None:
        try:
            datetime.strptime(start_date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid start_date format. Expected YYYY-MM-DD",
            )

    if end_date is not None:
        try:
            datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=INVALID_DATE_DETAIL,
            )

    try:
        comparison_data = await run_in_threadpool(
            partial(
                get_actual_vs_predicted_until,
                end_date=end_date,
                start_date=start_date,
            )
        )

        return PredictionComparisonResponse(
            success=True,
            end_date=comparison_data["end_date"],
            total_days_returned=len(comparison_data["rows"]),
            aggregation_strategy=(
                "For each date, combine all available predictions using "
                "horizon-weighted mean (weight = 1/horizon)."
            ),
            metrics=comparison_data["metrics"],
            comparison=comparison_data["rows"],
        )
    except Exception as e:
        logger.error(f"Prediction comparison error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/predictions/fan",
    response_model=PredictionFanResponse,
    responses={
        404: {
            "model": ErrorResponse,
            "description": "No stored prediction runs available",
        },
        500: {
            "model": ErrorResponse,
            "description": "Server error while building fan chart data",
        },
    },
)
async def prediction_fan_chart(
    min_samples_per_horizon: Annotated[int, Query(ge=1, le=100)] = 20,
):
    """
    Return fan chart quantile bands for the latest stored forecast run.

    Bands are calibrated from historical forecast errors by horizon.
    """
    try:
        fan_payload = await run_in_threadpool(
            get_latest_prediction_fan_chart,
            min_samples_per_horizon,
        )

        return PredictionFanResponse(
            success=True,
            generated_at=fan_payload["generated_at"],
            last_price_date=fan_payload["last_price_date"],
            last_price=fan_payload["last_price"],
            calibration_method=fan_payload["calibration_method"],
            fan=fan_payload["fan"],
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Fan chart endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Removed /predict [POST] as it adds complexity and deviates from the automated clean flow.
# The GET /predict endpoint is now the definitive way to get forecasts.


@app.get("/model-info")
async def model_info():
    """Get information about the loaded models and system status."""
    sent_info = await run_in_threadpool(sentiment_service.get_latest_info)

    return {
        "lookback": model_artifacts.lookback,
        "horizon": model_artifacts.horizon,
        "arima_order": model_artifacts.arima_order,
        "device": str(model_artifacts.device),
        "models_loaded": model_artifacts._loaded,
        "sentiment_data": {
            "total_records": sent_info["total_records"],
            "latest_date": sent_info["latest_date"],
            "integration_status": "DISABLED (Price-Only Mode)",
        },
        "components": {
            "arima": "Trend forecasting",
            "mid_gru": "Mid-frequency pattern recognition",
            "sent_gru": "Sentiment-aware prediction",
            "xgb_hf": "High-frequency noise modeling",
            "meta_ensemble": "Ridge regression combination",
        },
    }


@app.get("/health")
async def health_check():
    """Health status check."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": API_VERSION,
    }


@app.get("/scraper/status")
async def scraper_status():
    """Get the news scraper scheduler status and last run info."""
    return await run_in_threadpool(get_scheduler_status)


@app.post(
    "/scraper/run",
    responses={
        400: {
            "model": ErrorResponse,
            "description": "Invalid date format",
        },
        401: {
            "model": ErrorResponse,
            "description": "Missing or invalid X-Scraper-Key header",
        },
        500: {
            "model": ErrorResponse,
            "description": "Server error during scraper execution",
        },
    },
)
async def scraper_run(
    target_date: Annotated[str | None, Query(pattern=r"^\d{4}-\d{2}-\d{2}$")] = None,
    x_scraper_key: Annotated[str | None, Header()] = None,
):
    """
    Manually trigger a news scraping run.

    Protected by X-Scraper-Key header when SCRAPER_API_KEY env var is set.

    Args:
        target_date: Optional YYYY-MM-DD date to scrape. Defaults to yesterday.
    """
    if SCRAPER_API_KEY and x_scraper_key != SCRAPER_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-Scraper-Key header")
    # Validate date format and value if provided
    validated_date = None
    if target_date is not None:
        try:
            datetime.strptime(target_date, "%Y-%m-%d")
            validated_date = target_date
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=INVALID_DATE_DETAIL
            )
    
    try:
        result = await run_in_threadpool(partial(run_scraper_now, target_date=validated_date))
        return result
    except Exception as e:
        logger.error("Manual scraper run failed", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/scraper/backfill",
    responses={
        400: {
            "model": ErrorResponse,
            "description": "Invalid input parameters",
        },
        500: {
            "model": ErrorResponse,
            "description": "Server error during backfill operation",
        },
    },
)
async def scraper_backfill(
    days_back: Annotated[int, Query(ge=1, le=365)] = 30,
    max_pages: Annotated[int, Query(ge=1, le=50)] = 15,
):
    """
    Backfill the sentiment database for the last N days.

    Crawls paginated archives of all news sources, computes sentiment,
    and applies decay for days with no articles. Call once after fresh deployment
    to fill sentiment history for the active model's extended lookback window.

    Args:
        days_back: Number of days to backfill (default 30).
        max_pages: Max pages to crawl per site (default 15).
    """
    capped_days_back = min(days_back, 365)
    capped_max_pages = min(max_pages, 50)

    try:
        result = await run_in_threadpool(
            partial(
                backfill_history,
                days_back=capped_days_back,
                max_pages_per_site=capped_max_pages,
            )
        )
        return result
    except Exception as e:
        logger.error("Backfill failed", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/news/backfill-images",
    responses={
        400: {
            "model": ErrorResponse,
            "description": "Invalid date format or input parameters",
        },
        401: {
            "model": ErrorResponse,
            "description": "Missing or invalid X-Scraper-Key header",
        },
        500: {
            "model": ErrorResponse,
            "description": "Server error during image backfill operation",
        },
    },
)
async def backfill_news_images(
    start_date: Annotated[str | None, Query(pattern=r"^\d{4}-\d{2}-\d{2}$")] = None,
    end_date: Annotated[str | None, Query(pattern=r"^\d{4}-\d{2}-\d{2}$")] = None,
    limit: Annotated[int | None, Query(ge=1, le=5000)] = None,
    reset: bool = False,
    x_scraper_key: Annotated[str | None, Header()] = None,
):
    """Backfill missing image_url values for stored news articles using Pexels."""
    if SCRAPER_API_KEY and x_scraper_key != SCRAPER_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-Scraper-Key header")

    try:
        validated_start = validate_backfill_date(start_date) if start_date else None
        validated_end = validate_backfill_date(end_date) if end_date else None
    except ValueError:
        raise HTTPException(status_code=400, detail=INVALID_DATE_DETAIL)

    if validated_start and validated_end and validated_start > validated_end:
        raise HTTPException(status_code=400, detail="start_date must be less than or equal to end_date")

    try:
        return await run_in_threadpool(
            partial(
                backfill_news_image_urls,
                start_date=validated_start,
                end_date=validated_end,
                limit=limit,
                reset=reset,
            )
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Image backfill failed", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# EXPLAINABILITY ENDPOINTS
# ============================================================================


@app.get(
    "/explain",
    response_model=None,  # Manually serialize to avoid import ordering issues
    responses={
        200: {
            "model": dict,
            "description": "Explanation result"
        },
        400: {"model": ErrorResponse, "description": "Invalid date format"},
        503: {
            "model": ErrorResponse,
            "description": "Explanation not yet computed for today (job hasn't run)",
        },
        500: {"model": ErrorResponse, "description": "Server error retrieving explanation"},
    },
)
async def get_explanation(
    explanation_date: Annotated[str | None, Query(pattern=r"^\d{4}-\d{2}-\d{2}$")] = None,
):
    """
    Retrieve stored explainability result for a given date.

    If no explanation_date is provided, defaults to today.
    The explanation is pre-computed once per day at 06:00 UTC and cached in the database.
    
    If the job hasn't run yet for today, returns 503 with a retry message.

    Args:
        explanation_date: Optional YYYY-MM-DD date. Defaults to today.

    Returns:
        Explainability result with SHAP features, sentiment analysis, and LLM narrative.
    """
    from app.database import get_explanation_for_date

    if explanation_date is None:
        explanation_date = datetime.now().strftime("%Y-%m-%d")

    # Validate date format
    try:
        datetime.strptime(explanation_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid date format. Expected YYYY-MM-DD",
        )

    try:
        # Retrieve from database (fast, <200ms guaranteed)
        explanation = await run_in_threadpool(
            get_explanation_for_date, explanation_date
        )

        if explanation is None:
            # Explanation hasn't been computed yet
            raise HTTPException(
                status_code=503,
                detail=f"Explanation not yet available for {explanation_date}. "
                "The daily job runs at 06:00 UTC. Please retry in a few moments.",
            )

        # Format response — prefer full xai_payload (dashboard-ready) if stored
        xai_payload = explanation.get("xai_payload")
        if xai_payload:
            # New format: full dashboard payload — return directly
            return xai_payload

        # Legacy format: build from individual DB columns (explanations without xai_payload)
        return {
            "success": True,
            "explanation_date": explanation_date,
            "prediction": explanation["prediction"],
            "confidence_interval_lower": explanation["confidence_interval_lower"],
            "confidence_interval_upper": explanation["confidence_interval_upper"],
            "confidence_level": explanation["confidence_level"],
            "agreement_score": explanation["agreement_score"],
            "model_contributions": {
                "arima": explanation["arima_contribution"],
                "gru_mid": explanation["gru_mid_contribution"],
                "gru_sent": explanation["gru_sent_contribution"],
                "xgb_hf": explanation["xgb_hf_contribution"],
            },
            "top_features": [
                {
                    "feature_name": f["feature_name"],
                    "shap_value": f["shap_value"],
                    "feature_value": f.get("feature_value", 0.0),
                }
                for f in explanation["top_shap_features"]
            ],
            "sentiment_headlines": [
                {
                    "headline": h["headline"],
                    "sentiment_score": h["sentiment_score"],
                    "sentiment_label": h["sentiment_label"],
                    "top_keywords": h.get("top_keywords", []),
                }
                for h in explanation["sentiment_headlines"]
            ],
            "explanation_text": explanation["explanation_text"],
            "generated_at": explanation["generated_at"],
            "computation_time_seconds": explanation["computation_time_seconds"],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving explanation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/explain/regenerate",
    response_model=None,
    responses={
        200: {"description": "Regenerated explanation"},
        400: {"model": ErrorResponse, "description": "Invalid date"},
        503: {"model": ErrorResponse, "description": "Prices not available"},
    },
)
async def regenerate_explanation(
    explanation_date: Annotated[str | None, Query(pattern=r"^\d{4}-\d{2}-\d{2}$")] = None,
):
    """
    Force-regenerate the xai_payload for a given date (default: today).

    Runs the full XAI pipeline and overwrites the stored xai_payload with
    the new dashboard-compatible format. Use this to backfill old rows that
    were computed before the new pipeline was deployed.
    """
    from app.database import get_explanation_for_date, get_prices as get_prices_db
    from app.services.explainability import ExplainabilityService
    import pandas as pd, time as _time

    if explanation_date is None:
        explanation_date = datetime.now().strftime("%Y-%m-%d")

    try:
        datetime.strptime(explanation_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Expected YYYY-MM-DD")

    async def _run():
        import warnings
        warnings.filterwarnings("ignore")
        t0 = _time.time()
        svc = ExplainabilityService()

        # Load prices from DB (bypass stale guard — user explicitly requested regeneration)
        prices_df = get_prices_db(days=120)
        prices_df["date"] = pd.to_datetime(prices_df["date"])
        prices_df = prices_df.sort_values("date").reset_index(drop=True)
        if prices_df.empty:
            raise HTTPException(status_code=503, detail="No price data available")

        last_price = float(prices_df["price"].iloc[-1])
        last_date = prices_df["date"].iloc[-1].strftime("%Y-%m-%d")

        from app.services.prediction import prediction_service
        forecasts = prediction_service.predict(prices=prices_df)
        pred = {"last_price": last_price, "last_date": last_date, "forecasts": forecasts}

        ridge_exp = svc._explain_ridge(prices_df)
        gru_exp   = svc._explain_gru_attention(prices_df)
        xgb_exp   = svc._explain_xgboost(prices_df)
        arima_exp = svc._explain_arima(prices_df)
        sent_exp  = svc._explain_sentiment()

        agg        = svc._aggregate_explanations(arima_exp, ridge_exp, gru_exp, xgb_exp, sent_exp, pred)
        prompt     = svc._build_explanation_prompt(agg)
        llm_result = svc._generate_llm_narrative(prompt, agg)
        xai_payload = svc._build_xai_payload(explanation_date, agg, llm_result)

        # Persist to DB using the proper DB layer function
        from app.database import update_explanation_xai_payload, explanation_exists_for_date
        if explanation_exists_for_date(explanation_date):
            update_explanation_xai_payload(explanation_date, xai_payload)
        else:
            explanation_text = llm_result.get("narrative", "")
            computation_time = _time.time() - t0
            svc._store_explanation(explanation_date, agg, explanation_text, computation_time, xai_payload)

        xai_payload["computation_time_seconds"] = round(_time.time() - t0, 2)
        return xai_payload

    try:
        return await run_in_threadpool(_run)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Explanation regeneration failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
