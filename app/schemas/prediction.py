"""
Pydantic schemas for API request/response models.
"""
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from datetime import datetime


class PriceInput(BaseModel):
    """Single price data point."""
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    price: float = Field(..., gt=0, description="Brent oil price (USD)")
    
    @field_validator('date')
    @classmethod
    def validate_date(cls, v):
        try:
            datetime.strptime(v, '%Y-%m-%d')
        except ValueError:
            raise ValueError('Date must be in YYYY-MM-DD format')
        return v


class PredictionRequest(BaseModel):
    """Request body for manual prediction endpoint."""
    prices: List[PriceInput] = Field(
        ..., 
        min_length=30,
        description="List of price data points (at least 30 days)"
    )


class ForecastDay(BaseModel):
    """Single day forecast."""
    date: str = Field(..., description="Forecast date")
    forecasted_price: float = Field(..., description="Predicted price (USD)")
    forecasted_return: float = Field(..., description="Predicted log return")
    horizon: int = Field(..., ge=1, le=14, description="Days ahead (1-14)")


class PredictionResponse(BaseModel):
    """Response for prediction endpoint."""
    success: bool = Field(..., description="Whether prediction succeeded")
    data_source: str = Field(..., description="Source of price data")
    last_price_date: str = Field(..., description="Date of last known price")
    last_price: float = Field(..., description="Last known price (USD)")
    forecasts: List[ForecastDay] = Field(..., description="14-day price forecasts")


class PriceDataResponse(BaseModel):
    """Response for price data endpoint."""
    success: bool
    ticker: str
    data_points: int
    date_range: dict
    prices: List[dict]


class HealthResponse(BaseModel):
    """Response for health check endpoint."""
    status: str
    models_loaded: bool
    timestamp: str
    version: str


class ErrorResponse(BaseModel):
    """Error response."""
    success: bool = False
    error: str
    detail: Optional[str] = None


# ============== Sentiment Schemas ==============

class SentimentInput(BaseModel):
    """Single sentiment data point."""
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    daily_sentiment_decay: float = Field(..., description="Decay-weighted sentiment score")
    news_volume: int = Field(..., ge=0, description="Number of news articles")
    log_news_volume: float = Field(..., description="Log-transformed volume")
    decayed_news_volume: float = Field(..., ge=0, description="Decay-weighted volume")
    high_news_regime: int = Field(..., ge=0, le=1, description="High news flag (0 or 1)")
    
    @field_validator('date')
    @classmethod
    def validate_date(cls, v):
        try:
            datetime.strptime(v, '%Y-%m-%d')
        except ValueError:
            raise ValueError('Date must be in YYYY-MM-DD format')
        return v


class BulkSentimentRequest(BaseModel):
    """Request for bulk sentiment upload."""
    sentiment_data: List[SentimentInput] = Field(
        ...,
        min_length=1,
        description="List of sentiment data points"
    )


class SentimentAddResponse(BaseModel):
    """Response for adding sentiment."""
    success: bool
    message: str
    total_records: int


class SentimentHistoryResponse(BaseModel):
    """Response for sentiment history."""
    success: bool
    total_records: int
    latest_date: Optional[str]
    data: List[dict]

