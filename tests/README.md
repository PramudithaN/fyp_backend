# Test Suite for Oil Price Prediction API

This directory contains comprehensive test coverage for the Brent Oil Price Prediction API.

## Test Structure

```
tests/
├── __init__.py              # Test package initialization
├── conftest.py              # Pytest configuration and fixtures
├── test_api_endpoints.py    # API endpoint tests
├── test_services.py         # Service layer tests
├── test_schemas.py          # Schema validation tests
├── test_models.py           # Model loading and inference tests
├── test_database.py         # Database operation tests
└── test_integration.py      # Integration and end-to-end tests
```

## Running Tests

### Run all tests
```bash
pytest
```

### Run specific test file
```bash
pytest tests/test_api_endpoints.py
```

### Run specific test class
```bash
pytest tests/test_api_endpoints.py::TestHealthEndpoint
```

### Run specific test
```bash
pytest tests/test_api_endpoints.py::TestHealthEndpoint::test_health_check
```

### Run with coverage
```bash
pytest --cov=app --cov-report=html --cov-report=term
```

### Run with verbose output
```bash
pytest -v
```

### Run tests in parallel
```bash
pytest -n auto
```

## Test Categories

### 1. API Endpoint Tests (`test_api_endpoints.py`)
- Health check endpoint
- Root endpoint
- Price fetching endpoint
- Prediction endpoint
- Model info endpoint
- Scraper control endpoints

### 2. Service Layer Tests (`test_services.py`)
- Sentiment service operations
- Prediction service pipeline
- Price fetcher functionality
- Feature engineering
- News fetching and scraping
- FinBERT sentiment analysis

### 3. Schema Tests (`test_schemas.py`)
- Input validation (PriceInput, PredictionRequest)
- Output validation (PredictionResponse, ForecastDay)
- Error handling for invalid data
- Date format validation
- Range validation

### 4. Model Tests (`test_models.py`)
- Model loader functionality
- GRU model architecture
- Forward pass tests
- Device selection (CPU/GPU)
- Model parameter validation

### 5. Database Tests (`test_database.py`)
- Database initialization
- CRUD operations for sentiment data
- Bulk operations
- Error handling
- Connection management

### 6. Integration Tests (`test_integration.py`)
- End-to-end prediction pipeline
- API integration workflows
- Data flow through system
- Error handling across layers
- Concurrent request handling

## Test Fixtures

Common fixtures available in `conftest.py`:

- `test_client`: FastAPI test client
- `sample_prices_df`: Sample price DataFrame
- `sample_prices_list`: Sample price list for POST requests
- `sample_sentiment_df`: Sample sentiment DataFrame
- `mock_model_artifacts`: Mocked model artifacts

## Coverage Goals

Target coverage: **>80%**

Generate coverage report:
```bash
pytest --cov=app --cov-report=html
open htmlcov/index.html  # View in browser
```

## Writing New Tests

### Test Naming Convention
- Test files: `test_*.py`
- Test classes: `Test*`
- Test methods: `test_*`

### Example Test Structure
```python
class TestFeatureName:
    """Tests for specific feature."""
    
    def test_success_case(self, fixture):
        """Test successful operation."""
        # Arrange
        input_data = {...}
        
        # Act
        result = function(input_data)
        
        # Assert
        assert result.success is True
    
    def test_error_case(self):
        """Test error handling."""
        with pytest.raises(ValueError):
            function(invalid_input)
```

## Mocking External Dependencies

Tests mock external dependencies to ensure:
- Fast execution
- Deterministic results
- No external API calls
- No actual model loading (when not needed)

Example:
```python
@patch('app.services.price_fetcher.fetch_latest_prices')
def test_with_mock(mock_fetch):
    mock_fetch.return_value = sample_data
    # Test code here
```

## Continuous Integration

Tests are designed to run in CI/CD pipelines:
- No external dependencies required
- Mock all external services
- Fast execution (< 60 seconds)
- Deterministic results

## Troubleshooting

### Tests fail with import errors
```bash
# Ensure app is in PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Tests fail with model loading errors
- Tests mock model loading by default
- If you need real models, disable mocking in specific tests

### Database tests fail
- Tests use mocked database connections
- No actual database required for tests

## Dependencies

Required packages:
- `pytest>=7.4.0`
- `pytest-cov>=4.1.0`
- `pytest-mock>=3.11.0`
- `pytest-asyncio>=0.21.0`
- `httpx>=0.24.0`

Install test dependencies:
```bash
pip install -r requirements-dev.txt
```
