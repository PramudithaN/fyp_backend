"""
FastAPI application for Brent Oil Price Prediction.

Endpoints:
- GET  /predict  - Auto-fetch prices and predict
- POST /predict  - Predict with custom prices
- GET  /prices   - View fetched price data
- GET  /health   - Health check
"""
import logging
from contextlib import asynccontextmanager
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import API_TITLE, API_DESCRIPTION, API_VERSION, BRENT_TICKER
from app.models.model_loader import model_artifacts
from app.services.price_fetcher import fetch_latest_prices, get_last_n_trading_days
from app.services.prediction import prediction_service
from app.schemas.prediction import (
    PredictionRequest,
    PredictionResponse,
    PriceDataResponse,
    HealthResponse,
    ErrorResponse
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    logger.info("Starting application...")
    try:
        model_artifacts.load_all()
        logger.info("Models loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
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
async def predict_auto():
    """
    Auto-predict: Fetch latest prices from Yahoo Finance and generate forecast.
    
    This endpoint automatically:
    1. Fetches historical Brent oil prices (enough for feature engineering)
    2. Computes technical features
    3. Runs the ensemble model
    4. Returns 14-day price forecast
    
    No input required - prices are fetched automatically.
    """
    try:
        # Fetch latest prices (need ~60 days for feature engineering)
        logger.info("Fetching latest prices from Yahoo Finance...")
        all_prices = fetch_latest_prices(lookback_days=90)
        
        # Pass all prices to prediction service
        # (it needs extra data for feature engineering like lags and rolling windows)
        logger.info(f"Fetched {len(all_prices)} days of price data")
        
        # Generate predictions
        logger.info("Running prediction pipeline...")
        forecasts = prediction_service.predict(all_prices)
        
        return PredictionResponse(
            success=True,
            data_source=f"Yahoo Finance ({BRENT_TICKER})",
            last_price_date=str(all_prices['date'].iloc[-1]),
            last_price=round(float(all_prices['price'].iloc[-1]), 2),
            forecasts=forecasts
        )
    
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResponse)
async def predict_manual(request: PredictionRequest):
    """
    Manual predict: Use user-provided prices for prediction.
    
    Useful for:
    - Backtesting with historical data
    - Testing with custom scenarios
    - When Yahoo Finance is unavailable
    
    Request body must include at least 30 days of price data.
    """
    try:
        # Convert request to DataFrame
        prices = pd.DataFrame([
            {'date': p.date, 'price': p.price}
            for p in request.prices
        ])
        prices['date'] = pd.to_datetime(prices['date'])
        prices = prices.sort_values('date').reset_index(drop=True)
        
        # Validate
        if len(prices) < model_artifacts.lookback:
            raise ValueError(
                f"Need at least {model_artifacts.lookback} days of data, "
                f"got {len(prices)}"
            )
        
        # Generate predictions
        logger.info("Running prediction pipeline with custom data...")
        forecasts = prediction_service.predict(prices)
        
        return PredictionResponse(
            success=True,
            data_source="User provided",
            last_price_date=str(prices['date'].iloc[-1].date()),
            last_price=round(float(prices['price'].iloc[-1]), 2),
            forecasts=forecasts
        )
    
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-info")
async def model_info():
    """Get information about the loaded model."""
    return {
        "lookback": model_artifacts.lookback,
        "horizon": model_artifacts.horizon,
        "arima_order": model_artifacts.arima_order,
        "device": str(model_artifacts.device),
        "models_loaded": model_artifacts._loaded,
        "components": {
            "arima": "Trend forecasting",
            "mid_gru": "Mid-frequency pattern recognition",
            "sent_gru": "Sentiment-aware prediction (Phase 1: zero sentiment)",
            "xgb_hf": "High-frequency noise modeling",
            "meta_ensemble": "Ridge regression combination"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
