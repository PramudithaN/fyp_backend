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

[![Tests](https://github.com/PramudithaN/fyp_backend/actions/workflows/tests.yml/badge.svg)](https://github.com/PramudithaN/fyp_backend/actions/workflows/tests.yml)
[![SonarCloud](https://github.com/PramudithaN/fyp_backend/actions/workflows/sonarcloud.yml/badge.svg)](https://github.com/PramudithaN/fyp_backend/actions/workflows/sonarcloud.yml)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=PramudithaN_fyp_backend&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=PramudithaN_fyp_backend)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=PramudithaN_fyp_backend&metric=coverage)](https://sonarcloud.io/summary/new_code?id=PramudithaN_fyp_backend)
[![Security Rating](https://sonarcloud.io/api/project_badges/measure?project=PramudithaN_fyp_backend&metric=security_rating)](https://sonarcloud.io/summary/new_code?id=PramudithaN_fyp_backend)
[![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=PramudithaN_fyp_backend&metric=sqale_rating)](https://sonarcloud.io/summary/new_code?id=PramudithaN_fyp_backend)

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

## Environment Variables

Configure the application using environment variables (create a `.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `NEWSAPI_KEY` | - | NewsAPI.org API key for news fetching |
| `NEWSDATA_KEY` | - | NewsData.io API key for news fetching |
| `SENTIMENT_MODE` | `finbert` | Sentiment analysis mode: `simple` or `finbert` |
| `SKIP_FINBERT_PRELOAD` | `false` | Skip FinBERT model preloading at startup (set to `true` for deployments without HuggingFace access) |
| `SCRAPER_ENABLED` | `true` | Enable/disable the daily news scraper |
| `SCRAPER_SCHEDULE_HOUR` | `6` | Hour (0-23) to run daily news scraper |
| `SCRAPER_SCHEDULE_MINUTE` | `0` | Minute (0-59) to run daily news scraper |

**Example `.env` file:**
```env
NEWSAPI_KEY=your_api_key_here
NEWSDATA_KEY=your_api_key_here
SENTIMENT_MODE=finbert
SKIP_FINBERT_PRELOAD=false
SCRAPER_ENABLED=true
```

### FinBERT Model Loading

The FinBERT sentiment model is loaded from Hugging Face on first use. For deployments in restricted network environments:

- Set `SKIP_FINBERT_PRELOAD=true` to skip model preloading during startup
- The model will be loaded on the first sentiment analysis request instead
- Ensure the deployment environment can access `huggingface.co` for model downloads

## API Endpoints

| Endpoint   | Method | Description                   |
| ---------- | ------ | ----------------------------- |
| `/predict` | GET    | Auto-fetch prices and predict |
| `/predictions/fan` | GET | Fan chart quantile bands for latest forecast |
| `/predictions/compare` | GET | Compare stored forecasts vs actual prices |
| `/prices`  | GET    | View fetched price data       |
| `/historical/prices` | GET | Historical imported prices (daily/weekly/monthly) |
| `/historical/features/combined` | GET | Historical joined price + news features |
| `/health`  | GET    | Health check                  |
| `/sentiment/add` | POST | Add daily sentiment      |
| `/sentiment/bulk` | POST | Bulk upload sentiment   |
| `/docs`    | GET    | Swagger API Documentation     |

## Example Usage

```bash
# Generate latest 14-day forecast
curl http://localhost:8000/predict

# Fan chart bands for frontend visualization
curl "http://localhost:8000/predictions/fan?min_samples_per_horizon=20"

# Historical actual vs predicted comparison
curl "http://localhost:8000/predictions/compare?start_date=2025-01-01&end_date=2025-12-31"
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

## Code Quality & Analysis

### SonarCloud

This project uses SonarCloud for continuous code quality and security analysis.

[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=PramudithaN_fyp_backend&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=PramudithaN_fyp_backend)

**Key Metrics:**
- **Coverage:** Tracks test coverage across the codebase
- **Security:** Identifies security vulnerabilities and hotspots
- **Maintainability:** Measures code smells and technical debt
- **Reliability:** Detects bugs and code issues
- **Duplications:** Identifies duplicate code blocks

**View Full Report:**  
[https://sonarcloud.io/dashboard?id=PramudithaN_fyp_backend](https://sonarcloud.io/dashboard?id=PramudithaN_fyp_backend)

### Local Analysis

Run SonarScanner locally (requires SonarCloud token):

```bash
# Install SonarScanner (one-time setup)
# macOS: brew install sonar-scanner
# Windows: Download from https://docs.sonarcloud.io/advanced-setup/ci-based-analysis/sonarscanner-cli/

# Run analysis
sonar-scanner \
  -Dsonar.organization=pramudithan \
  -Dsonar.projectKey=PramudithaN_fyp_backend \
  -Dsonar.sources=app \
  -Dsonar.host.url=https://sonarcloud.io \
  -Dsonar.login=YOUR_SONAR_TOKEN
```

### Setting Up SonarCloud (For Maintainers)

1. Go to [SonarCloud](https://sonarcloud.io)
2. Import the GitHub repository
3. Add `SONAR_TOKEN` secret to GitHub repository settings
4. The workflow will automatically run on push and pull requests
