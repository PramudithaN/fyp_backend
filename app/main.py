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

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import API_TITLE, API_DESCRIPTION, API_VERSION, BRENT_TICKER
from app.models.model_loader import model_artifacts
from app.database import init_database
from app.services.price_fetcher import fetch_latest_prices, get_last_n_trading_days
from app.services.prediction import prediction_service
from app.services.sentiment_service import sentiment_service
from app.schemas.prediction import (
    PredictionRequest,
    PredictionResponse,
    PriceDataResponse,
    HealthResponse,
    ErrorResponse,
    SentimentInput,
    BulkSentimentRequest,
    SentimentAddResponse,
    SentimentHistoryResponse
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and initialize database on startup."""
    logger.info("Starting application...")
    try:
        # Initialize sentiment database
        init_database()
        logger.info("Sentiment database initialized!")
        
        # Load ML models
        model_artifacts.load_all()
        logger.info("Models loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to start: {e}")
        raise
    yield
    logger.info("Shutting down application...")


# Create FastAPI app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan
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
        "predict": "/predict"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        models_loaded=model_artifacts._loaded,
        timestamp=datetime.now().isoformat(),
        version=API_VERSION
    )


@app.get("/prices", response_model=PriceDataResponse)
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
                "start": str(prices['date'].iloc[0]),
                "end": str(prices['date'].iloc[-1])
            },
            prices=prices.to_dict(orient='records')
        )
    
    except Exception as e:
        logger.error(f"Error fetching prices: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predict", response_model=PredictionResponse)
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
        forecasts = prediction_service.predict(days_of_history=30)
        
        # Get latest price for response metadata
        from app.services.price_fetcher import fetch_latest_prices
        latest_prices = fetch_latest_prices(lookback_days=5)
        last_price = float(latest_prices['price'].iloc[-1])
        last_date = str(latest_prices['date'].iloc[-1].date())
        
        return PredictionResponse(
            success=True,
            data_source=f"Yahoo Finance ({BRENT_TICKER})",
            last_price_date=last_date,
            last_price=round(last_price, 2),
            forecasts=forecasts
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
            "integration_status": "DISABLED (Price-Only Mode)"
        },
        "components": {
            "arima": "Trend forecasting",
            "mid_gru": "Mid-frequency pattern recognition",
            "sent_gru": "Sentiment-aware prediction",
            "xgb_hf": "High-frequency noise modeling",
            "meta_ensemble": "Ridge regression combination"
        }
    }


@app.get("/health")
async def health_check():
    """Health status check."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": API_VERSION
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


