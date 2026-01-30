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
