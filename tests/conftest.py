"""
Pytest configuration and fixtures.
"""
import os
import sys
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from fastapi.testclient import TestClient

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.main import app


@pytest.fixture(scope="session")
def test_client():
    """Create a test client for the FastAPI app."""
    # Disable scheduler during tests
    os.environ["SCRAPER_ENABLED"] = "false"
    return TestClient(app)


@pytest.fixture
def sample_prices_df():
    """Generate sample price data for testing."""
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    prices = np.random.uniform(70, 90, size=30)
    
    return pd.DataFrame({
        'date': dates,
        'price': prices
    })


@pytest.fixture
def sample_prices_list():
    """Generate sample price list for POST requests."""
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    return [
        {
            "date": date.strftime('%Y-%m-%d'),
            "price": float(np.random.uniform(70, 90))
        }
        for date in dates
    ]


@pytest.fixture
def sample_sentiment_df():
    """Generate sample sentiment data for testing."""
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    sentiments = np.random.uniform(-0.5, 0.5, size=30)
    
    return pd.DataFrame({
        'date': dates,
        'sentiment': sentiments,
        'article_count': np.random.randint(5, 20, size=30)
    })


@pytest.fixture
def mock_model_artifacts(monkeypatch):
    """Mock model artifacts for testing."""
    class MockModelArtifacts:
        _loaded = True
        lookback = 30
        horizon = 14
        arima_order = (2, 1, 2)
        device = "cpu"
        
        def load_all(self):
            pass
    
    from app.models import model_loader
    monkeypatch.setattr(model_loader, 'model_artifacts', MockModelArtifacts())
    return MockModelArtifacts()
