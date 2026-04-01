"""
Pydantic schemas for API request/response models.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from datetime import datetime

from app.models.model_loader import model_artifacts

DATE_FMT_DESC = "Date in YYYY-MM-DD format"
IS_MARKET_OPEN_DESC = "Whether the oil market is currently open"
MARKET_OPEN_TIME_DESC = "Market open time (via Yahoo Finance market status)"
MARKET_CLOSE_TIME_DESC = "Market close time (via Yahoo Finance market status)"
TIMEZONE_INFO_DESC = "Timezone reference for market hours"


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
        ...,
        description="List of price data points covering at least the model lookback window",
    )

    @field_validator("prices")
    @classmethod
    def validate_lookback_length(cls, prices):
        required = int(model_artifacts.lookback)
        if len(prices) < required:
            raise ValueError(
                f"At least {required} price data points are required for the active model"
            )
        return prices


class ForecastDay(BaseModel):
    """Single day forecast."""

    date: str = Field(..., description="Forecast date")
    forecasted_price: float = Field(..., description="Predicted price (USD)")
    forecasted_return: float = Field(..., description="Predicted log return")
    lower_bound: Optional[float] = Field(
        None,
        description="Lower 95% forecast bound (USD)",
    )
    upper_bound: Optional[float] = Field(
        None,
        description="Upper 95% forecast bound (USD)",
    )
    horizon: int = Field(
        ..., ge=1, description="Forecast step index for the active model horizon"
    )

    @field_validator("horizon")
    @classmethod
    def validate_horizon(cls, horizon):
        max_horizon = int(model_artifacts.horizon)
        if horizon > max_horizon:
            raise ValueError(
                f"Horizon must be between 1 and {max_horizon} for the active model"
            )
        return horizon


class PredictionResponse(BaseModel):
    """Response for prediction endpoint."""

    success: bool = Field(..., description="Whether prediction succeeded")
    data_source: str = Field(..., description="Source of price data")
    last_price_date: str = Field(..., description="Date of last known price")
    last_price: float = Field(..., description="Last known price (USD)")
    forecasts: List[ForecastDay] = Field(
        ..., description="Multi-step price forecasts for the active model horizon"
    )
    is_market_open: bool = Field(..., description=IS_MARKET_OPEN_DESC)
    market_open_time: str = Field(..., description=MARKET_OPEN_TIME_DESC)
    market_close_time: str = Field(..., description=MARKET_CLOSE_TIME_DESC)
    timezone_info: str = Field(..., description=TIMEZONE_INFO_DESC)
    generated_at: Optional[str] = Field(
        None,
        description="Timestamp when the locked daily forecast record was generated",
    )
    prediction_date: Optional[str] = Field(
        None,
        description="Daily forecast key date (YYYY-MM-DD)",
    )
    based_on_price_date: Optional[str] = Field(
        None,
        description="Official close date used as model input",
    )
    based_on_price: Optional[float] = Field(
        None,
        description="Official close price used as model input",
    )
    next_update_at: Optional[str] = Field(
        None,
        description="Next scheduled locked forecast refresh timestamp",
    )


class UploadWindowStats(BaseModel):
    """Stats describing how the uploaded lookback window was resolved."""

    lookback_days: int = Field(..., ge=1)
    window_start: str = Field(..., description=DATE_FMT_DESC)
    window_end: str = Field(..., description=DATE_FMT_DESC)
    uploaded_rows_used: int = Field(..., ge=0)
    filled_from_database: int = Field(..., ge=0)
    filled_by_carry: int = Field(..., ge=0)


class ResolvedPricePoint(BaseModel):
    """Single row in resolved lookback prices after fallback filling."""

    date: str = Field(..., description=DATE_FMT_DESC)
    price: float = Field(..., gt=0)
    source: str = Field(..., description="uploaded | database | carry_fill")


class UploadPredictionResponse(BaseModel):
    """Response for Excel upload prediction endpoint."""

    success: bool = Field(..., description="Whether prediction succeeded")
    data_source: str = Field(..., description="Composed data source description")
    last_price_date: str = Field(..., description="Date of last known price")
    last_price: float = Field(..., description="Last known price (USD)")
    forecasts: List[ForecastDay] = Field(
        ..., description="Multi-step price forecasts for the active model horizon"
    )
    is_market_open: bool = Field(..., description=IS_MARKET_OPEN_DESC)
    market_open_time: str = Field(..., description=MARKET_OPEN_TIME_DESC)
    market_close_time: str = Field(..., description=MARKET_CLOSE_TIME_DESC)
    timezone_info: str = Field(..., description=TIMEZONE_INFO_DESC)
    upload_window: UploadWindowStats
    resolved_price_window: List[ResolvedPricePoint]


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
    image_url: Optional[str] = Field(None, description="Article image URL")
    source: Optional[str] = Field(None, description="News source name")
    published_at: Optional[str] = Field(
        None, description="Original publication timestamp"
    )
    sentiment_score: Optional[float] = Field(
        None, description="Per-article sentiment score"
    )
    created_at: Optional[str] = Field(None, description="Database insertion timestamp")


class NewsArticlesResponse(BaseModel):
    """Response for news articles endpoint."""

    success: bool
    total_records: int
    requested_date: Optional[str] = Field(
        None, description="Exact article date requested"
    )
    days: int = Field(..., ge=1, description="Number of recent distinct days searched")
    latest_article_date: Optional[str] = Field(
        None, description="Latest article date in response"
    )
    articles: List[NewsArticle]


class HealthResponse(BaseModel):
    """Response for health check endpoint."""

    status: str
    models_loaded: bool
    timestamp: str
    version: str
    is_market_open: bool = Field(..., description=IS_MARKET_OPEN_DESC)
    market_open_time: str = Field(..., description=MARKET_OPEN_TIME_DESC)
    market_close_time: str = Field(..., description=MARKET_CLOSE_TIME_DESC)
    timezone_info: str = Field(..., description=TIMEZONE_INFO_DESC)


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
    lower_bound: Optional[float] = Field(
        None,
        description="Aggregated lower forecast bound (horizon-weighted mean, USD)",
    )
    upper_bound: Optional[float] = Field(
        None,
        description="Aggregated upper forecast bound (horizon-weighted mean, USD)",
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
    horizon: int = Field(
        ..., ge=1, description="Forecast step index for the active model horizon"
    )
    point_forecast: float = Field(..., description="Point forecast price (USD)")

    @field_validator("horizon")
    @classmethod
    def validate_horizon(cls, horizon):
        max_horizon = int(model_artifacts.horizon)
        if horizon > max_horizon:
            raise ValueError(
                f"Horizon must be between 1 and {max_horizon} for the active model"
            )
        return horizon

    p10: float = Field(..., description="10th percentile forecast price (USD)")
    p25: float = Field(..., description="25th percentile forecast price (USD)")
    p50: float = Field(..., description="50th percentile forecast price (USD)")
    p75: float = Field(..., description="75th percentile forecast price (USD)")
    p90: float = Field(..., description="90th percentile forecast price (USD)")
    lower_bound: Optional[float] = Field(
        None,
        description="Lower 95% forecast bound from model error stds (USD)",
    )
    upper_bound: Optional[float] = Field(
        None,
        description="Upper 95% forecast bound from model error stds (USD)",
    )
    sample_count: int = Field(..., ge=0, description="Calibration samples used")


class PredictionFanResponse(BaseModel):
    """Response for fan chart forecast endpoint."""

    success: bool = Field(..., description="Whether fan chart generation succeeded")
    generated_at: str = Field(..., description="Timestamp of latest forecast run")
    last_price_date: str = Field(..., description="Reference date for latest forecast")
    last_price: float = Field(
        ..., description="Last known price used by latest forecast"
    )
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


class SHAPFeature(BaseModel):
    """Single feature in SHAP explanation."""

    feature_name: str = Field(..., description="Name of the feature")
    shap_value: float = Field(..., description="SHAP value contribution")
    feature_value: float = Field(
        ..., description="Actual value of the feature in the data"
    )


class SentimentHeadline(BaseModel):
    """Sentiment headline in explanation."""

    headline: str = Field(...)
    sentiment_score: float = Field(..., ge=-1.0, le=1.0)
    sentiment_label: str = Field(..., description="bullish|bearish|neutral")
    top_keywords: List[str] = Field(...)


class ExplanationResponse(BaseModel):
    """Response for /explain endpoint."""

    success: bool = Field(..., description="Whether explanation was retrieved")
    explanation_date: str = Field(..., description=DATE_FMT_DESC)
    prediction: float = Field(..., description="Predicted price (USD/barrel)")
    confidence_interval_lower: float = Field(...)
    confidence_interval_upper: float = Field(...)
    confidence_level: str = Field(..., description="high|moderate")
    agreement_score: float = Field(
        ..., description="Model agreement (0=perfect, 1+=high disagreement)"
    )
    model_contributions: dict = Field(
        ..., description="USD contribution from each model"
    )
    top_features: List[SHAPFeature] = Field(
        ..., description="Top 7 global SHAP features"
    )
    sentiment_headlines: List[SentimentHeadline] = Field(
        ..., description="Top 3 sentiment headlines"
    )
    explanation_text: str = Field(..., description="3-sentence plain English narrative")
    generated_at: str = Field(..., description="ISO timestamp of generation")
    computation_time_seconds: float = Field(
        ..., description="How long computation took"
    )


class SentimentDayPoint(BaseModel):
    """Daily sentiment point with raw and decayed sentiment details."""

    date: str = Field(..., description=DATE_FMT_DESC)
    raw_daily_sentiment: float = Field(
        ..., description="Raw daily sentiment mean stored for the date"
    )
    cross_day_decayed_sentiment: float = Field(
        ..., description="Cross-day decayed sentiment using lambda-based recurrence"
    )
    sentiment_change_vs_prev_day: float = Field(
        ..., description="Delta of raw daily sentiment versus previous day"
    )
    decayed_sentiment_change_vs_prev_day: float = Field(
        ..., description="Delta of decayed sentiment versus previous day"
    )
    news_volume: int = Field(..., ge=0, description="Number of news articles")
    log_news_volume: float = Field(..., description="Log-transformed news volume")
    decayed_news_volume: float = Field(..., ge=0)
    high_news_regime: bool = Field(
        ..., description="Whether the day is classified as high news regime"
    )
    ema: dict = Field(..., description="EMA values for sentiment and volume features")


class SentimentOverviewMeta(BaseModel):
    """Metadata and model assumptions used for sentiment calculations."""

    requested_days: Optional[int] = Field(None, ge=1)
    actual_records: int = Field(..., ge=0)
    start_date: Optional[str] = Field(None, description="First date in returned data")
    end_date: Optional[str] = Field(None, description="Last date in returned data")
    decay_lambda: float = Field(..., description="Lambda used in decay recurrence")
    decay_factor: float = Field(..., description="exp(-lambda)")
    decay_formula: str = Field(..., description="Formula used for cross-day decay")
    ema_windows: List[int] = Field(..., description="EMA windows included in payload")


class SentimentOverviewSummary(BaseModel):
    """High-level sentiment summary stats for dashboard cards."""

    latest_raw_sentiment: Optional[float] = None
    latest_decayed_sentiment: Optional[float] = None
    average_raw_sentiment: Optional[float] = None
    average_decayed_sentiment: Optional[float] = None
    average_news_volume: Optional[float] = None
    high_news_regime_days: int = Field(..., ge=0)
    positive_days: int = Field(..., ge=0)
    negative_days: int = Field(..., ge=0)
    neutral_days: int = Field(..., ge=0)
    latest_trend: str = Field(
        ..., description="bullish|bearish|neutral trend based on recent decayed slope"
    )


class SentimentOverviewResponse(BaseModel):
    """Response for frontend sentiment overview endpoint."""

    success: bool
    meta: SentimentOverviewMeta
    summary: SentimentOverviewSummary
    timeline: List[SentimentDayPoint]
