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
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.config import (
    API_TITLE,
    API_DESCRIPTION,
    API_VERSION,
    BRENT_TICKER,
    SCRAPER_ENABLED,
    SCRAPER_SCHEDULE_HOUR,
    SCRAPER_SCHEDULE_MINUTE,
    SKIP_FINBERT_PRELOAD,
)
from app.models.model_loader import model_artifacts
from app.database import init_database
from app.services.price_fetcher import fetch_latest_prices, get_last_n_trading_days
from app.services.prediction import prediction_service
from app.services.sentiment_service import sentiment_service
from app.services.scraper_scheduler import (
    start_scheduler,
    stop_scheduler,
    get_scheduler_status,
    run_scraper_now,
    backfill_history,
)
from app.schemas.prediction import (
    PredictionRequest,
    PredictionResponse,
    PriceDataResponse,
    HealthResponse,
    ErrorResponse,
    SentimentInput,
    BulkSentimentRequest,
    SentimentAddResponse,
    SentimentHistoryResponse,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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

    # Start daily news scraper scheduler
    if SCRAPER_ENABLED:
        try:
            start_scheduler(hour=SCRAPER_SCHEDULE_HOUR, minute=SCRAPER_SCHEDULE_MINUTE)
            logger.info("News scraper scheduler started!")
        except Exception as e:
            logger.warning(f"Scraper startup failed: {e}", exc_info=True)
            logger.info("Scraper will not run - app continues")
    else:
        logger.info("News scraper disabled")
        
    logger.info("Application startup completed successfully")
    
    yield
    # Shutdown
    if SCRAPER_ENABLED:
        try:
            stop_scheduler()
        except Exception:
            pass
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
    Fetch and display current Brent oil price data.

    Returns the last 30 trading days of prices from Yahoo Finance.
    """
    try:
        # Fetch prices
        all_prices = fetch_latest_prices(lookback_days=60)
        prices = get_last_n_trading_days(all_prices, n=30)

        return PriceDataResponse(
            success=True,
            ticker=BRENT_TICKER,
            data_points=len(prices),
            date_range={
                "start": str(prices["date"].iloc[0]),
                "end": str(prices["date"].iloc[-1]),
            },
            prices=prices.to_dict(orient="records"),
        )

    except Exception as e:
        logger.error(f"Error fetching prices: {e}")
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
    Generate a 14-day forecast based on real-time data.

    This endpoint:
    1. Fetches current Brent oil prices from Yahoo Finance.
    2. Automatically gathers missing news sentiments for the last 30 days.
    3. Analyzes news articles using the custom FinBERT model.
    4. Generates a multi-step forecast using the ensemble model.
    """
    try:
        # Generate predictions using the automated end-to-end service
        forecasts = prediction_service.predict()

        # Get latest price for response metadata
        from app.services.price_fetcher import fetch_latest_prices

        latest_prices = fetch_latest_prices(lookback_days=5)
        last_price = float(latest_prices["price"].iloc[-1])
        last_date = str(latest_prices["date"].iloc[-1].date())

        return PredictionResponse(
            success=True,
            data_source=f"Yahoo Finance ({BRENT_TICKER})",
            last_price_date=last_date,
            last_price=round(last_price, 2),
            forecasts=forecasts,
        )

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
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
        500: {
            "model": ErrorResponse,
            "description": "Server error during scraper execution",
        },
    },
)
async def scraper_run(
    target_date: Annotated[str | None, Query(pattern=r"^\d{4}-\d{2}-\d{2}$")] = None
):
    """
    Manually trigger a news scraping run.

    Args:
        target_date: Optional YYYY-MM-DD date to scrape. Defaults to yesterday.
    """
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
