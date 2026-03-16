"""
Pydantic schemas for API request/response models.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from datetime import datetime


DATE_FMT_DESC = "Date in YYYY-MM-DD format"


class PriceInput(BaseModel):
    """Single price data point."""

    date: str = Field(..., description=DATE_FMT_DESC)
    price: float = Field(..., gt=0, description="Brent oil price (USD)")

    @field_validator("date")
    @classmethod
    def validate_date(cls, v):
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format")
        return v


class PredictionRequest(BaseModel):
    """Request body for manual prediction endpoint."""

    prices: List[PriceInput] = Field(
        ..., min_length=30, description="List of price data points (at least 30 days)"
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
    is_market_open: bool = Field(..., description="Whether the oil market is currently open")
    market_state: str = Field(..., description="Raw market state from Yahoo Finance (REGULAR, CLOSED, PRE, POST, etc.)")
    market_status_message: str = Field(..., description="Human-readable market status")


class PriceDataResponse(BaseModel):
    """Response for price data endpoint."""

    success: bool
    ticker: str
    data_points: int
    date_range: dict
    prices: List[dict]


class NewsArticle(BaseModel):
    """Single stored news article."""

    id: int
    article_date: str = Field(..., description=DATE_FMT_DESC)
    title: str = Field(..., description="Article headline")
    description: Optional[str] = Field(None, description="Article summary or snippet")
    url: Optional[str] = Field(None, description="Canonical article URL")
    source: Optional[str] = Field(None, description="News source name")
    published_at: Optional[str] = Field(None, description="Original publication timestamp")
    sentiment_score: Optional[float] = Field(None, description="Per-article sentiment score")
    created_at: Optional[str] = Field(None, description="Database insertion timestamp")


class NewsArticlesResponse(BaseModel):
    """Response for news articles endpoint."""

    success: bool
    total_records: int
    requested_date: Optional[str] = Field(None, description="Exact article date requested")
    days: int = Field(..., ge=1, description="Number of recent distinct days searched")
    latest_article_date: Optional[str] = Field(None, description="Latest article date in response")
    articles: List[NewsArticle]


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

    date: str = Field(..., description=DATE_FMT_DESC)
    daily_sentiment_decay: float = Field(
        ..., description="Decay-weighted sentiment score"
    )
    news_volume: int = Field(..., ge=0, description="Number of news articles")
    log_news_volume: float = Field(..., description="Log-transformed volume")
    decayed_news_volume: float = Field(..., ge=0, description="Decay-weighted volume")
    high_news_regime: int = Field(
        ..., ge=0, le=1, description="High news flag (0 or 1)"
    )

    @field_validator("date")
    @classmethod
    def validate_date(cls, v):
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format")
        return v


class BulkSentimentRequest(BaseModel):
    """Request for bulk sentiment upload."""

    sentiment_data: List[SentimentInput] = Field(
        ..., min_length=1, description="List of sentiment data points"
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
    latest_date: Optional[str] = None
    data: List[dict]


class PredictionComparisonDay(BaseModel):
    """Daily actual vs aggregated predicted values."""

    date: str = Field(..., description=DATE_FMT_DESC)
    actual_price: float = Field(..., description="Actual stored price (USD)")
    predicted_price: float = Field(
        ..., description="Aggregated predicted price (horizon-weighted mean)"
    )
    predicted_price_median: float = Field(
        ..., description="Median of all predictions for the day"
    )
    predicted_price_latest: float = Field(
        ..., description="Latest prediction made for the day"
    )
    prediction_count: int = Field(
        ..., ge=1, description="Number of prediction runs contributing to this day"
    )
    error: float = Field(..., description="actual_price - predicted_price")
    abs_error: float = Field(..., description="Absolute prediction error")
    abs_pct_error: Optional[float] = Field(
        None, description="Absolute percentage error (%)"
    )


class PredictionComparisonMetrics(BaseModel):
    """Aggregate performance metrics across compared days."""

    compared_days: int = Field(..., ge=0)
    mae: Optional[float] = Field(None, description="Mean absolute error")
    rmse: Optional[float] = Field(None, description="Root mean squared error")
    mape: Optional[float] = Field(None, description="Mean absolute percentage error")


class PredictionComparisonResponse(BaseModel):
    """Response for actual vs predicted comparison endpoint."""

    success: bool = Field(..., description="Whether comparison succeeded")
    end_date: str = Field(..., description="Comparison cutoff date")
    total_days_returned: int = Field(..., ge=0)
    aggregation_strategy: str = Field(
        ..., description="How multiple predictions for same date were combined"
    )
    metrics: PredictionComparisonMetrics
    comparison: List[PredictionComparisonDay]


class FanChartPoint(BaseModel):
    """Single forecast point with fan chart quantile bands."""

    date: str = Field(..., description="Forecast date")
    horizon: int = Field(..., ge=1, le=14, description="Days ahead (1-14)")
    point_forecast: float = Field(..., description="Point forecast price (USD)")
    p10: float = Field(..., description="10th percentile forecast price (USD)")
    p25: float = Field(..., description="25th percentile forecast price (USD)")
    p50: float = Field(..., description="50th percentile forecast price (USD)")
    p75: float = Field(..., description="75th percentile forecast price (USD)")
    p90: float = Field(..., description="90th percentile forecast price (USD)")
    sample_count: int = Field(..., ge=0, description="Calibration samples used")


class PredictionFanResponse(BaseModel):
    """Response for fan chart forecast endpoint."""

    success: bool = Field(..., description="Whether fan chart generation succeeded")
    generated_at: str = Field(..., description="Timestamp of latest forecast run")
    last_price_date: str = Field(..., description="Reference date for latest forecast")
    last_price: float = Field(..., description="Last known price used by latest forecast")
    calibration_method: str = Field(
        ..., description="How uncertainty bands were calibrated"
    )
    fan: List[FanChartPoint] = Field(
        ..., description="Forecast points with fan chart quantile bands"
    )


class HistoricalPricesResponse(BaseModel):
    """Response for historical prices endpoint."""

    success: bool
    granularity: str
    total_available: int
    total_records: int
    limit: int
    offset: int
    date_range: dict
    data: List[dict]


class HistoricalCombinedFeaturesResponse(BaseModel):
    """Response for combined historical features endpoint."""

    success: bool
    granularity: str
    total_available: int
    total_records: int
    limit: int
    offset: int
    date_range: dict
    data: List[dict]
