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
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Header
from fastapi.middleware.cors import CORSMiddleware

from app.config import (
    API_TITLE,
    API_DESCRIPTION,
    API_VERSION,
    BRENT_TICKER,
    SKIP_FINBERT_PRELOAD,
    SCRAPER_API_KEY,
)
from app.models.model_loader import model_artifacts
from app.database import init_database, add_prediction, get_actual_vs_predicted_until
from app.services.price_fetcher import get_market_status
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
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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
        init_database()
        logger.info("Sentiment database initialized!")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}", exc_info=True)
        # Continue - database can be created on first request

    # Load ML models
    try:
        model_artifacts.load_all()
        logger.info("Models loaded successfully!")
    except Exception as e:
        logger.error(f"Model loading failed: {e}", exc_info=True)
        # Continue - predictions will fail but app can still start

    # Pre-load FinBERT sentiment model (eliminates cold-start on first request)
    if not SKIP_FINBERT_PRELOAD:
        try:
            from app.services.finbert_analyzer import preload_model

            preload_model()
            logger.info("FinBERT model pre-loaded successfully")
        except Exception as e:
            logger.warning(f"FinBERT preload failed: {e}", exc_info=True)
            logger.info("FinBERT model will load on first request")
    else:
        logger.info("FinBERT preload skipped (SKIP_FINBERT_PRELOAD=true)")

    logger.info("Application startup completed successfully")

    yield
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
    Fetch and display current Brent oil price data from database.

    Returns the last 30 trading days of prices stored in Turso.
    """
    try:
        # Fetch prices from database
        from app.database import get_prices as get_prices_db
        
        all_prices = get_prices_db(days=60)
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
            total_available = get_historical_prices_count(
                start_date=start_date,
                end_date=end_date,
            )
            df = get_historical_prices_db(
                start_date=start_date,
                end_date=end_date,
                limit=limit,
                offset=offset,
            )
        else:
            raw_df = get_historical_prices_db(start_date=start_date, end_date=end_date)
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
            total_available = get_historical_features_combined_count(
                start_date=start_date,
                end_date=end_date,
            )
            df = get_historical_features_combined_db(
                start_date=start_date,
                end_date=end_date,
                limit=limit,
                offset=offset,
            )
        else:
            raw_df = get_historical_features_combined_db(
                start_date=start_date,
                end_date=end_date,
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
    1. Fetches 120 days of historical prices from database (Turso).
    2. Fetches 120 days of sentiment history from database (Turso).
    3. Generates a multi-step forecast using the ensemble model.
    4. Persists the forecast to database.
    
    Note: Daily prices and news articles are stored by the scraper/scheduler.
    No external API calls needed - all data comes from database."""
    try:
        # Generate predictions using the automated end-to-end service
        forecasts = prediction_service.predict()

        # Get latest price for response metadata (from database)
        from app.database import get_prices

        latest_prices = get_prices(days=5)
        last_price = float(latest_prices["price"].iloc[-1])
        last_date = str(latest_prices["date"].iloc[-1].date())

        # Note: Prices are already in database from daily scraper/scheduler
        # No need to re-persist them here

        # Persist the forecast run to DB
        try:
            add_prediction(
                generated_at=datetime.now().isoformat(),
                last_price_date=last_date,
                last_price=round(last_price, 2),
                forecasts=[f.model_dump() if hasattr(f, "model_dump") else f for f in forecasts],
            )
        except Exception as db_err:
            logger.warning(f"Failed to persist prediction: {db_err}")

        market = get_market_status()

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
                detail="Invalid date format. Expected YYYY-MM-DD",
            )

    try:
        comparison_data = get_actual_vs_predicted_until(
            start_date=start_date,
            end_date=end_date,
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


# Removed /predict [POST] as it adds complexity and deviates from the automated clean flow.
# The GET /predict endpoint is now the definitive way to get forecasts.


@app.get("/model-info")
async def model_info():
    """Get information about the loaded models and system status."""
    sent_info = sentiment_service.get_latest_info()

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
    return get_scheduler_status()


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
                detail="Invalid date format. Expected YYYY-MM-DD"
            )
    
    try:
        result = run_scraper_now(target_date=validated_date)
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
        result = backfill_history(
            days_back=capped_days_back, max_pages_per_site=capped_max_pages
        )
        return result
    except Exception as e:
        logger.error("Backfill failed", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
