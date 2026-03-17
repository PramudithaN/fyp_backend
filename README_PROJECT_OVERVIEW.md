# Project Overview: Brent Oil Price Prediction Backend

## What this project does
This backend predicts Brent crude oil prices for the next 14 days and exposes the results through a FastAPI API.

At a high level, it:
- fetches and stores recent Brent prices from Yahoo Finance (`BZ=F`)
- computes sentiment from oil-related news
- engineers price + sentiment features
- runs a multi-model ensemble forecast pipeline
- stores prediction runs for comparison and fan-chart uncertainty bands
- exposes forecast, explainability, historical, and scraper-control endpoints

## Core idea in one line
The system combines statistical, deep learning, tree-based, and meta-ensemble models to forecast future returns, then converts those returns into future prices.

## How it works (end-to-end)

### 1. App startup
When the API starts (`app/main.py`):
- database schema is initialized (Turso/libsql)
- model artifacts are loaded from `model_artifacts/`
- FinBERT can be preloaded unless disabled with `SKIP_FINBERT_PRELOAD=true`

### 2. Data ingestion
- Prices:
  - fetched from Yahoo Finance by `app/services/price_fetcher.py`
  - upserted into `prices` table
- News + sentiment:
  - scraper collects articles from configured sources
  - sentiment is computed per article (FinBERT or fallback)
  - daily aggregates are stored in `sentiment_history`
  - article rows are stored in `news_articles`

### 3. Feature engineering
`app/services/feature_engineering.py` builds training-compatible features:
- log returns
- lagged returns (`ret_lag_*`)
- rolling volatility (`vol_*`)
- RSI and momentum
- sentiment + EMA-smoothed sentiment/volume signals

### 4. Ensemble prediction pipeline
`app/services/prediction.py` orchestrates:
- ARIMA trend model
- Mid-frequency GRU
- Sentiment-aware GRU
- Horizon-wise XGBoost models
- Ridge meta-ensemble to combine all component outputs
- optional sentiment-based adjustment for reversal signals

Then the service converts predicted returns into forecasted prices for days 1..14.

### 5. Persistence and analytics
Each `/predict` run is stored in `predictions` as a full 14-day forecast payload. This enables:
- `/predictions/compare` for actual vs predicted analysis
- `/predictions/fan` for quantile fan-chart bands calibrated from historical forecast errors

### 6. Explainability mode
`/explain` returns forecast + explainability:
- feature attributions for XGBoost component (SHAP)
- per-model attribution from the meta-ensemble
- sentiment signal diagnostics and impactful news summary

## Main API endpoints
- `GET /predict`: generate latest 14-day forecast
- `GET /explain`: forecast with explainability payload
- `GET /predictions/compare`: compare stored predictions with realized prices
- `GET /predictions/fan`: fan-chart quantile bands
- `GET /prices`: latest stored/fetched price series
- `GET /news`: stored news article feed
- `GET /historical/prices`: imported historical OHLCV data
- `GET /historical/features/combined`: joined historical price + news features
- `GET /model-info`: model and feature pipeline status
- `GET /scraper/status`: last scraper status
- `POST /scraper/run`: run scraper for a target date (header-protected)
- `POST /scraper/backfill`: backfill multi-day sentiment history

## Data storage model
Database layer is in `app/database.py`.

Primary tables:
- `prices`: market prices from Yahoo
- `sentiment_history`: daily sentiment feature rows
- `news_articles`: per-article metadata + sentiment score
- `predictions`: serialized forecast runs
- `historical_prices`: imported historical market dataset
- `historical_news_features`: imported historical news features

View:
- `historical_features_combined`: date-level combined historical view

## Operational scripts
Under `scripts/`:
- `run_scraper.py`: run one daily scraping/sentiment cycle
- `backfill_prices_and_news.py`: backfill prices + articles to Turso
- `import_historical_data.py`: import historical price/news datasets
- `migrate_to_turso.py`: migrate local SQLite data to Turso
- `recompute_sentiment_history.py`: recompute sentiment history from stored articles

## Configuration and environment
Important environment variables:
- `TURSO_DATABASE_URL`, `TURSO_AUTH_TOKEN`
- `SENTIMENT_MODE` (`finbert` or `simple`)
- `SKIP_FINBERT_PRELOAD`
- `NEWSAPI_KEY`, `NEWSDATA_KEY`
- `SCRAPER_API_KEY`

## Run and test
Run locally:
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Run tests:
```bash
pytest tests/ -v
```

With coverage:
```bash
pytest tests/ --cov=app --cov-report=html --cov-report=term-missing
```

## Notes from source analysis
- There are two `GET /health` route handlers in `app/main.py`; the later definition takes effect at runtime.
- Some older docs mention sentiment insert endpoints, but the current implementation centers on scraper-driven sentiment ingestion (`/scraper/run`, `/scraper/backfill`).

## Recommended reading order in code
1. `app/main.py` (API and request flow)
2. `app/services/prediction.py` (forecast orchestration)
3. `app/services/feature_engineering.py` (feature definitions)
4. `app/models/model_loader.py` (artifact loading)
5. `app/database.py` (persistence schema and queries)
6. `app/services/scraper_scheduler.py` and `app/services/news_fetcher.py` (news/sentiment pipeline)
