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
from datetime import datetime
from functools import partial
from threading import RLock
from time import monotonic
from typing import Annotated

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Header
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
    SKIP_FINBERT_PRELOAD,
    SCRAPER_API_KEY,
)
from app.models.model_loader import model_artifacts
from app.database import (
    init_database,
    add_prediction,
    get_actual_vs_predicted_until,
    get_latest_prediction_fan_chart,
    get_news_articles,
    get_recent_news_articles,
)
from app.services.price_fetcher import (
    fetch_latest_prices,
    fetch_live_price_snapshot,
    get_last_n_trading_days,
    get_market_status,
)
from app.services.prediction import prediction_service
from app.services.sentiment_service import sentiment_service
from app.services.scraper_scheduler import (
    get_scheduler_status,
    run_scraper_now,
    backfill_history,
)
from app.schemas.prediction import (
    PredictionRequest,
    PredictionResponse,
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
_predict_cache_lock = RLock()
_predict_cache: tuple[float, PredictionResponse] | None = None
_prediction_refresh_lock = asyncio.Lock()


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

    market = await run_in_threadpool(get_market_status)

    return PredictionResponse(
        success=True,
        data_source=f"Yahoo Finance ({BRENT_TICKER})",
        last_price_date=last_date,
        last_price=round(last_price, 2),
        forecasts=forecasts,
        is_market_open=market["is_open"],
        market_state=market["market_state"],
        market_status_message=market["message"],
    )


async def _refresh_prediction_cache(
    *,
    force_refresh: bool,
    persist_forecast: bool,
) -> PredictionResponse:
    """Return warm prediction data and serialize expensive refresh work."""
    global _predict_cache

    if _cache_enabled() and not force_refresh:
        with _predict_cache_lock:
            if _predict_cache and (monotonic() - _predict_cache[0]) < _PREDICT_CACHE_TTL_SECONDS:
                return _predict_cache[1]

    async with _prediction_refresh_lock:
        if _cache_enabled() and not force_refresh:
            with _predict_cache_lock:
                if _predict_cache and (monotonic() - _predict_cache[0]) < _PREDICT_CACHE_TTL_SECONDS:
                    return _predict_cache[1]

        response = await _build_prediction_response(persist_forecast=persist_forecast)

        if _cache_enabled():
            with _predict_cache_lock:
                _predict_cache = (monotonic(), response)

        return response


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
    precompute_task: asyncio.Task | None = None
    precompute_stop_event = asyncio.Event()
    
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

    if PREDICTION_PRECOMPUTE_ENABLED and _cache_enabled():
        precompute_task = asyncio.create_task(_prediction_precompute_loop(precompute_stop_event))
        logger.info("Prediction precompute enabled")
    else:
        logger.info("Prediction precompute disabled")

    logger.info("Application startup completed successfully")

    yield
    precompute_stop_event.set()
    if precompute_task is not None:
        try:
            await precompute_task
        except Exception as exc:
            logger.warning("Prediction precompute task shutdown failed: %s", exc)
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
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        models_loaded=model_artifacts._loaded,
        timestamp=datetime.now().isoformat(),
        version=API_VERSION,
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
    days: Annotated[int, Query(ge=1, le=30)] = 7,
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
            get_historical_prices as get_historical_prices_db,
            get_historical_prices_count,
        )

        if granularity == "daily":
            total_available = await run_in_threadpool(
                get_historical_prices_count,
                start_date,
                end_date,
            )
            df = await run_in_threadpool(
                get_historical_prices_db,
                start_date,
                end_date,
                limit,
                offset,
            )
        else:
            raw_df = await run_in_threadpool(get_historical_prices_db, start_date, end_date)
            aggregated_df = _aggregate_historical_prices(raw_df, granularity)
            total_available = len(aggregated_df)
            df = aggregated_df.iloc[offset : offset + limit].reset_index(drop=True)

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
        500: {"model": ErrorResponse, "description": "Server error during prediction"},
    },
)
async def predict_now():
    """
    Generate a 14-day forecast based on database data.

    This endpoint:
    1. Refreshes up to 120 days of historical prices from Yahoo Finance into the database.
    2. Fetches 120 days of sentiment history from database (Turso).
    3. Generates a multi-step forecast using the ensemble model.
    4. Persists the forecast to database.

    If the live price refresh fails, prediction falls back to the latest stored prices."""
    try:
        return await _refresh_prediction_cache(force_refresh=False, persist_forecast=True)

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
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
    to fill the 30-day rolling window.

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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
