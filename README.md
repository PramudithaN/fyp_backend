---
title: Oil Price Prediction API
emoji: 🛢️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Oil Price Prediction Backend

FastAPI backend for Brent oil price forecasting using a trained ensemble model.

## Features

- **14-day price forecast** using VMD-based ensemble model
- **Automatic price fetching** from Yahoo Finance (BZ=F ticker)
- **REST API** with both auto and manual prediction modes
- **Sentiment Analysis** using FinBERT

## Quick Start (Local Development)

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

| Endpoint   | Method | Description                   |
| ---------- | ------ | ----------------------------- |
| `/predict` | GET    | Auto-fetch prices and predict |
| `/predict` | POST   | Predict with custom prices    |
| `/prices`  | GET    | View fetched price data       |
| `/health`  | GET    | Health check                  |
| `/sentiment/add` | POST | Add daily sentiment      |
| `/sentiment/bulk` | POST | Bulk upload sentiment   |
| `/sentiment` | GET  | View sentiment history        |
| `/docs`    | GET    | Swagger API Documentation     |

## Example Usage

```bash
# Auto-predict (fetches latest prices automatically)
curl http://localhost:8000/predict

# Manual predict with custom data
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"prices": [{"date": "2025-12-01", "price": 74.12}, ...]}'
```

## Model Artifacts

The trained model artifacts should be placed in `model_artifacts/`:

- `config.pkl` - Model configuration
- `mid_gru.pt` - Mid-frequency GRU model
- `sent_gru.pt` - Sentiment GRU model
- `xgb_hf_models.pkl` - XGBoost high-frequency models
- `meta_models.pkl` - Ridge meta-ensemble models
- `meta_scalers.pkl` - Meta-model scalers
- `scaler_mid.pkl`, `scaler_price.pkl`, `scaler_sent.pkl` - Feature scalers

## Architecture

```
Prices (30 days) → Feature Engineering → Component Models → Meta-Ensemble → 14-day Forecast
                                              ↓
                                    [ARIMA, Mid-GRU, Sent-GRU, XGBoost]
```
## Testing

This project includes comprehensive test coverage with 200+ test cases covering API endpoints, services, models, and integrations.

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=app --cov-report=html

# Run specific test category
pytest tests/test_api_endpoints.py
pytest tests/test_services.py
pytest tests/test_models.py

# Run in parallel (faster)
pytest -n auto
```

### Using Test Scripts

**Windows:**
```bash
run_tests.bat          # Run all tests
run_tests.bat coverage # Run with coverage
run_tests.bat api      # Run API tests only
run_tests.bat clean    # Clean test artifacts
```

**Linux/Mac:**
```bash
./run_tests.sh          # Run all tests
./run_tests.sh coverage # Run with coverage
./run_tests.sh api      # Run API tests only
./run_tests.sh clean    # Clean test artifacts
```

**Using Make:**
```bash
make test              # Run all tests
make test-cov          # Run with coverage
make test-fast         # Run in parallel
make test-api          # Run API tests only
```

### Test Structure

```
tests/
├── test_api_endpoints.py    # API endpoint tests (health, predict, prices, etc.)
├── test_services.py         # Service layer tests (prediction, sentiment, etc.)
├── test_schemas.py          # Data validation tests
├── test_models.py           # Model loading and inference tests
├── test_database.py         # Database operation tests
├── test_integration.py      # End-to-end integration tests
└── conftest.py             # Shared fixtures and configuration
```

### Test Coverage

Current coverage: **>80%**

View detailed coverage report:
```bash
pytest --cov=app --cov-report=html
# Open htmlcov/index.html in browser
```

### Development Dependencies

Install test dependencies:
```bash
pip install -r requirements-dev.txt
```

See [tests/README.md](tests/README.md) for detailed testing documentation.