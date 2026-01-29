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
            "status": "active" if sent_info["total_records"] > 30 else "needs_more_data"
        },
        "components": {
            "arima": "Trend forecasting",
            "mid_gru": "Mid-frequency pattern recognition",
            "sent_gru": "Sentiment-aware prediction",
            "xgb_hf": "High-frequency noise modeling",
            "meta_ensemble": "Ridge regression combination"
        }
    }


# ============== Sentiment Endpoints ==============

@app.post("/sentiment/add", response_model=SentimentAddResponse)
async def add_sentiment(sentiment: SentimentInput):
    """
    Add sentiment data for a specific date.
    
    This data will be used for predictions made on the following day
    (sentiment is automatically lagged by 1 day).
    
    Example:
    - Add sentiment for Jan 29
    - When predicting on Jan 30, the model uses Jan 29's sentiment
    """
    try:
        result = sentiment_service.add_daily_sentiment(
            date_str=sentiment.date,
            daily_sentiment_decay=sentiment.daily_sentiment_decay,
            news_volume=sentiment.news_volume,
            log_news_volume=sentiment.log_news_volume,
            decayed_news_volume=sentiment.decayed_news_volume,
            high_news_regime=sentiment.high_news_regime
        )
        
        return SentimentAddResponse(**result)
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error adding sentiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sentiment/bulk", response_model=SentimentAddResponse)
async def add_bulk_sentiment(request: BulkSentimentRequest):
    """
    Bulk upload sentiment data.
    
    Useful for initializing the system with historical sentiment data.
    Existing records for the same dates will be updated.
    """
    try:
        sentiment_list = [s.model_dump() for s in request.sentiment_data]
        result = sentiment_service.add_bulk_sentiment(sentiment_list)
        
        return SentimentAddResponse(
            success=True,
            message=f"Added {result['records_added']} sentiment records",
            total_records=result["total_records"]
        )
    
    except Exception as e:
        logger.error(f"Error in bulk upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sentiment", response_model=SentimentHistoryResponse)
async def get_sentiment_history(days: int = 30):
    """
    View stored sentiment history.
    
    Args:
        days: Number of days of history to retrieve (default: 30)
    
    Returns the most recent sentiment records.
    """
    try:
        info = sentiment_service.get_latest_info()
        df = sentiment_service.get_sentiment_window(days=days)
        
        data = []
        if not df.empty:
            df['date'] = df['date'].dt.strftime('%Y-%m-%d')
            data = df.to_dict(orient='records')
        
        return SentimentHistoryResponse(
            success=True,
            total_records=info["total_records"],
            latest_date=info["latest_date"],
            data=data
        )
    
    except Exception as e:
        logger.error(f"Error fetching sentiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sentiment/fetch")
async def fetch_news_sentiment(
    date: str = None,
    api_key: str = None,
    save: bool = True
):
    """
    Auto-fetch oil news from NewsAPI and compute sentiment features.
    
    This endpoint:
    1. Fetches oil-related news articles from NewsAPI
    2. Computes sentiment features using decay-weighted averaging
    3. Optionally saves to database
    
    Args:
        date: Date to fetch news for (YYYY-MM-DD). Default: yesterday.
        api_key: NewsAPI key. Can also be set via NEWSAPI_KEY env var.
        save: If True (default), automatically saves to database.
    
    Note: You need a free NewsAPI key from https://newsapi.org
    """
    try:
        from app.services.news_fetcher import fetch_and_compute_sentiment
        
        # Compute sentiment from news
        result = fetch_and_compute_sentiment(date=date, api_key=api_key)
        
        # Save to database if requested
        if save:
            sentiment_service.add_daily_sentiment(
                date_str=result["date"],
                daily_sentiment_decay=result["daily_sentiment_decay"],
                news_volume=result["news_volume"],
                log_news_volume=result["log_news_volume"],
                decayed_news_volume=result["decayed_news_volume"],
                high_news_regime=result["high_news_regime"]
            )
            result["saved_to_db"] = True
        else:
            result["saved_to_db"] = False
        
        result["success"] = True
        return result
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error fetching news sentiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sentiment/auto-update")
async def auto_update_sentiment(api_key: str = None):
    """
    Convenience endpoint: Fetch yesterday's news, compute sentiment, and save.
    
    Call this daily (e.g., via cron) to keep sentiment data up to date.
    
    Args:
        api_key: NewsAPI key. Can also be set via NEWSAPI_KEY env var.
    
    Returns:
        The computed and saved sentiment features.
    """
    from datetime import datetime, timedelta
    
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    try:
        from app.services.news_fetcher import fetch_and_compute_sentiment
        
        result = fetch_and_compute_sentiment(date=yesterday, api_key=api_key)
        
        # Save to database
        sentiment_service.add_daily_sentiment(
            date_str=result["date"],
            daily_sentiment_decay=result["daily_sentiment_decay"],
            news_volume=result["news_volume"],
            log_news_volume=result["log_news_volume"],
            decayed_news_volume=result["decayed_news_volume"],
            high_news_regime=result["high_news_regime"]
        )
        
        return {
            "success": True,
            "message": f"Sentiment for {yesterday} fetched and saved",
            "data": result
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in auto-update: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


