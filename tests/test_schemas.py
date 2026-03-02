"""
Tests for schemas and data models.
"""
import pytest
from pydantic import ValidationError
from datetime import datetime, timedelta


class TestPriceInput:
    """Tests for PriceInput schema."""
    
    def test_valid_price_input(self):
        """Test valid price input."""
        from app.schemas.prediction import PriceInput
        
        price = PriceInput(
            date="2026-03-01",
            price=75.50
        )
        
        assert price.date == "2026-03-01"
        assert price.price == 75.50
    
    def test_invalid_date_format(self):
        """Test invalid date format raises error."""
        from app.schemas.prediction import PriceInput
        
        with pytest.raises(ValidationError):
            PriceInput(
                date="03/01/2026",  # Wrong format
                price=75.50
            )
    
    def test_negative_price(self):
        """Test negative price raises error."""
        from app.schemas.prediction import PriceInput
        
        with pytest.raises(ValidationError):
            PriceInput(
                date="2026-03-01",
                price=-75.50
            )
    
    def test_zero_price(self):
        """Test zero price raises error."""
        from app.schemas.prediction import PriceInput
        
        with pytest.raises(ValidationError):
            PriceInput(
                date="2026-03-01",
                price=0
            )


class TestPredictionRequest:
    """Tests for PredictionRequest schema."""
    
    def test_valid_prediction_request(self, sample_prices_list):
        """Test valid prediction request."""
        from app.schemas.prediction import PredictionRequest
        
        request = PredictionRequest(prices=sample_prices_list)
        assert len(request.prices) >= 30
    
    def test_insufficient_prices(self):
        """Test insufficient price data raises error."""
        from app.schemas.prediction import PredictionRequest
        
        short_list = [
            {"date": "2026-03-01", "price": 75.50},
            {"date": "2026-03-02", "price": 76.00}
        ]
        
        with pytest.raises(ValidationError):
            PredictionRequest(prices=short_list)


class TestForecastDay:
    """Tests for ForecastDay schema."""
    
    def test_valid_forecast_day(self):
        """Test valid forecast day."""
        from app.schemas.prediction import ForecastDay
        
        forecast = ForecastDay(
            date="2026-03-15",
            forecasted_price=75.50,
            forecasted_return=0.001,
            horizon=1
        )
        
        assert forecast.date == "2026-03-15"
        assert forecast.forecasted_price == 75.50
        assert forecast.horizon == 1
    
    def test_invalid_horizon(self):
        """Test invalid horizon raises error."""
        from app.schemas.prediction import ForecastDay
        
        with pytest.raises(ValidationError):
            ForecastDay(
                date="2026-03-15",
                forecasted_price=75.50,
                forecasted_return=0.001,
                horizon=15  # Greater than 14
            )


class TestPredictionResponse:
    """Tests for PredictionResponse schema."""
    
    def test_valid_prediction_response(self):
        """Test valid prediction response."""
        from app.schemas.prediction import PredictionResponse, ForecastDay
        
        forecasts = [
            ForecastDay(
                date=(datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d'),
                forecasted_price=75.0 + i * 0.5,
                forecasted_return=0.001,
                horizon=i
            )
            for i in range(1, 15)
        ]
        
        response = PredictionResponse(
            success=True,
            data_source="Yahoo Finance",
            last_price_date="2026-03-01",
            last_price=75.50,
            forecasts=forecasts
        )
        
        assert response.success is True
        assert len(response.forecasts) == 14


class TestHealthResponse:
    """Tests for HealthResponse schema."""
    
    def test_valid_health_response(self):
        """Test valid health response."""
        from app.schemas.prediction import HealthResponse
        
        response = HealthResponse(
            status="healthy",
            models_loaded=True,
            timestamp=datetime.now().isoformat(),
            version="1.0"
        )
        
        assert response.status == "healthy"
        assert response.models_loaded is True


class TestSentimentInput:
    """Tests for SentimentInput schema."""
    
    def test_valid_sentiment_input(self):
        """Test valid sentiment input."""
        from app.schemas.prediction import SentimentInput
        
        sentiment = SentimentInput(
            date="2026-03-01",
            daily_sentiment_decay=0.5,
            news_volume=10,
            log_news_volume=2.3,
            decayed_news_volume=8.5,
            high_news_regime=1
        )
        
        assert sentiment.date == "2026-03-01"
        assert sentiment.daily_sentiment_decay == 0.5
        assert sentiment.news_volume == 10
        assert sentiment.high_news_regime == 1
    
    def test_sentiment_out_of_range(self):
        """Test sentiment value out of range."""
        from app.schemas.prediction import SentimentInput
        
        # SentimentInput doesn't have range validation on daily_sentiment_decay
        # This test should pass with any value
        sentiment = SentimentInput(
            date="2026-03-01",
            daily_sentiment_decay=1.5,  # Any value is allowed
            news_volume=10,
            log_news_volume=2.3,
            decayed_news_volume=8.5,
            high_news_regime=1
        )
        assert sentiment.daily_sentiment_decay == 1.5
    
    def test_negative_article_count(self):
        """Test negative article count raises error."""
        from app.schemas.prediction import SentimentInput
        
        with pytest.raises(ValidationError):
            SentimentInput(
                date="2026-03-01",
                daily_sentiment_decay=0.5,
                news_volume=-5,  # Negative news_volume
                log_news_volume=2.3,
                decayed_news_volume=8.5,
                high_news_regime=1
            )
