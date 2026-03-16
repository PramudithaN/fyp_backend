"""
Tests for API endpoints.
"""

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check(self, test_client):
        """Test health check returns correct status."""
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root_endpoint(self, test_client):
        """Test root endpoint returns API info."""
        response = test_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "docs" in data
        assert "health" in data
        assert "predict" in data


class TestPricesEndpoint:
    """Tests for prices endpoint."""

    @patch("app.main._sync_latest_prices")
    def test_get_prices_success(
        self, mock_sync_prices, test_client, sample_prices_df
    ):
        """Test successful price fetch."""
        mock_sync_prices.return_value = sample_prices_df

        response = test_client.get("/prices")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "ticker" in data
        assert "data_points" in data
        assert "prices" in data
        assert len(data["prices"]) > 0
        mock_sync_prices.assert_called_once_with(lookback_days=60)

    @patch("app.main._sync_latest_prices")
    def test_get_prices_error(self, mock_sync_prices, test_client):
        """Test price fetch handles errors."""
        mock_sync_prices.side_effect = Exception("API Error")

        response = test_client.get("/prices")
        assert response.status_code == 500


class TestNewsEndpoint:
    """Tests for news endpoint."""

    @patch("app.main.get_recent_news_articles")
    def test_get_recent_news_success(self, mock_get_recent_news, test_client):
        """Test recent news retrieval for frontend consumption."""
        mock_get_recent_news.return_value = [
            {
                "id": 1,
                "article_date": "2026-03-16",
                "title": "Oil prices edge higher",
                "description": "Brent crude rose on supply concerns.",
                "url": "https://example.com/oil-1",
                "source": "Reuters",
                "published_at": "2026-03-16T09:30:00",
                "sentiment_score": 0.27,
            }
        ]

        response = test_client.get("/news?days=3")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["total_records"] == 1
        assert data["days"] == 3
        assert data["latest_article_date"] == "2026-03-16"
        assert data["articles"][0]["title"] == "Oil prices edge higher"
        mock_get_recent_news.assert_called_once_with(days=3)

    @patch("app.main.get_news_articles")
    def test_get_news_by_date_success(self, mock_get_news_articles, test_client):
        """Test exact-date article retrieval."""
        mock_get_news_articles.return_value = [
            {
                "id": 2,
                "article_date": "2026-03-15",
                "title": "OPEC output steady",
                "description": "Production remained flat this week.",
                "url": "https://example.com/oil-2",
                "source": "Bloomberg",
                "published_at": "2026-03-15T07:00:00",
                "sentiment_score": -0.05,
                "created_at": "2026-03-15T08:00:00",
            }
        ]

        response = test_client.get("/news?article_date=2026-03-15")
        assert response.status_code == 200
        data = response.json()
        assert data["requested_date"] == "2026-03-15"
        assert data["days"] == 1
        assert data["articles"][0]["source"] == "Bloomberg"
        mock_get_news_articles.assert_called_once_with("2026-03-15")

    def test_get_news_invalid_date(self, test_client):
        """Test invalid article date validation."""
        response = test_client.get("/news?article_date=2026-02-30")
        assert response.status_code == 400

    @patch("app.main.get_recent_news_articles")
    def test_get_news_server_error(self, mock_get_recent_news, test_client):
        """Test news endpoint handles storage errors."""
        mock_get_recent_news.side_effect = Exception("DB Error")

        response = test_client.get("/news")
        assert response.status_code == 500


class TestPredictEndpoint:
    """Tests for prediction endpoint."""

    @patch("app.main.fetch_live_price_snapshot")
    @patch("app.main.prediction_service.predict")
    @patch("app.main._sync_latest_prices")
    def test_predict_success(
        self,
        mock_sync_prices,
        mock_predict,
        mock_live_snapshot,
        test_client,
        sample_prices_df,
    ):
        """Test successful prediction."""
        mock_sync_prices.return_value = sample_prices_df
        mock_live_snapshot.return_value = {
            "price": 92.0,
            "as_of": "2026-03-17T10:00:00",
            "as_of_date": "2026-03-17",
            "source": "yahoo_finance_intraday",
        }

        # Mock prediction result
        mock_forecasts = [
            {
                "date": (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d"),
                "forecasted_price": 75.0 + i * 0.5,
                "forecasted_return": 0.001 * i,
                "horizon": i,
            }
            for i in range(1, 15)
        ]
        mock_predict.return_value = mock_forecasts

        response = test_client.get("/predict")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "last_price" in data
        assert "forecasts" in data
        assert len(data["forecasts"]) == 14
        assert data["last_price"] == pytest.approx(92.0)
        assert data["last_price_date"] == "2026-03-17"
        mock_sync_prices.assert_called_once_with(lookback_days=120)
        synced_prices = mock_predict.call_args.kwargs["prices"]
        pd.testing.assert_frame_equal(synced_prices, sample_prices_df)

    @patch("app.main.prediction_service.predict")
    def test_predict_validation_error(self, mock_predict, test_client):
        """Test prediction handles validation errors."""
        mock_predict.side_effect = ValueError("Invalid data")

        response = test_client.get("/predict")
        assert response.status_code == 400

    @patch("app.main.prediction_service.predict")
    def test_predict_server_error(self, mock_predict, test_client):
        """Test prediction handles server errors."""
        mock_predict.side_effect = Exception("Model error")

        response = test_client.get("/predict")
        assert response.status_code == 500


class TestModelInfoEndpoint:
    """Tests for model info endpoint."""

    @patch("app.main.sentiment_service.get_latest_info")
    def test_model_info(self, mock_sentiment, test_client):
        """Test model info endpoint."""
        mock_sentiment.return_value = {
            "total_records": 100,
            "latest_date": "2026-03-01",
        }

        response = test_client.get("/model-info")
        assert response.status_code == 200
        data = response.json()
        assert "lookback" in data
        assert "horizon" in data
        assert "device" in data
        assert "models_loaded" in data
        assert "sentiment_data" in data
        assert "components" in data


class TestScraperEndpoints:
    """Tests for scraper endpoints."""

    @patch("app.main.get_scheduler_status")
    def test_scraper_status(self, mock_status, test_client):
        """Test scraper status endpoint."""
        mock_status.return_value = {"enabled": False, "running": False}

        response = test_client.get("/scraper/status")
        assert response.status_code == 200
        data = response.json()
        assert "enabled" in data

    @patch("app.main.run_scraper_now")
    def test_scraper_run_success(self, mock_run, test_client):
        """Test manual scraper run."""
        mock_run.return_value = {"status": "success", "articles_scraped": 10}

        response = test_client.post("/scraper/run")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    @patch("app.main.run_scraper_now")
    def test_scraper_run_error(self, mock_run, test_client):
        """Test scraper run handles errors."""
        mock_run.side_effect = Exception("Scraper error")

        response = test_client.post("/scraper/run")
        assert response.status_code == 500

    @patch("app.main.backfill_history")
    def test_scraper_backfill_success(self, mock_backfill, test_client):
        """Test scraper backfill."""
        mock_backfill.return_value = {"status": "success", "days_filled": 30}

        response = test_client.post("/scraper/backfill?days_back=30&max_pages=15")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    @patch("app.main.backfill_history")
    def test_scraper_backfill_error(self, mock_backfill, test_client):
        """Test scraper backfill handles errors."""
        mock_backfill.side_effect = Exception("Backfill error")

        response = test_client.post("/scraper/backfill")
        assert response.status_code == 500
