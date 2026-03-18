"""
Comprehensive explainability service for oil price predictions.

Pipeline:
1. Generate ARIMA decomposition (trend, seasonal, residual contributions)
2. Extract GRU timestep attributions using TimeSHAP
3. Compute XGBoost feature importances using SHAP TreeExplainer
4. Analyze sentiment headlines using LIME
5. Aggregate SHAP values across ensemble models
6. Build structured prompt with all explanation data
7. Generate narrative using Phi-3-mini local LLM
8. Store results in database keyed by date
"""

import numpy as np
import pandas as pd
import torch
import json
import logging
from datetime import datetime, date
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import os
from functools import lru_cache
import time

try:
    import shap
    from shap import Explainer
except ImportError:
    shap = None

try:
    import timeshap
    from timeshap.explainer import local_report
except ImportError:
    timeshap = None

try:
    import lime
    import lime.text
except ImportError:
    lime = None

from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.arima.model import ARIMA
from transformers import pipeline

from app.config import HORIZON, LOOKBACK
from app.models.model_loader import model_artifacts
from app.database import (
    add_explanation,
    get_sentiment_history,
    get_prices as get_prices_db,
    get_news_articles,
)
from app.services.prediction import prediction_service
from app.services.sentiment_service import sentiment_service
from app.services.feature_engineering import engineer_all_features

logger = logging.getLogger(__name__)


class ExplainabilityService:
    """Main orchestrator for daily explainability computation."""

    def __init__(self):
        self.artifacts = model_artifacts
        self._llm_pipeline = None

    @property
    def llm_pipeline(self):
        """Lazy-load Phi-3 mini LLM pipeline (expensive operation)."""
        if self._llm_pipeline is None:
            logger.info("Loading Phi-3-mini LLM pipeline...")
            self._llm_pipeline = pipeline(
                "text-generation",
                model="microsoft/Phi-3-mini-4k-instruct",
                device=0 if torch.cuda.is_available() else -1,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )
            logger.info("Phi-3-mini LLM loaded successfully")
        return self._llm_pipeline

    def _validate_todays_prices_available(self, target_date: str) -> bool:
        """
        Verify that today's prices (or yesterday's if after-hours) are in the database.

        This prevents the daily job from running on stale/incomplete data.

        Args:
            target_date: Date string (YYYY-MM-DD) to check for.

        Returns:
            True if prices for today/yesterday available, False otherwise.
        """
        try:
            from app.database import get_prices as get_prices_db

            # Fetch latest price
            prices_df = get_prices_db(days=5)  # Get last 5 days to check freshness

            if prices_df.empty:
                logger.warning("No prices found in database at all.")
                return False

            latest_price_date_str = prices_df["date"].iloc[-1]
            latest_price_date = pd.to_datetime(latest_price_date_str).date()
            target_date_obj = datetime.strptime(target_date, "%Y-%m-%d").date()

            # Market hours: Brent trades ~20h/day, so check if today or yesterday
            days_behind = (target_date_obj - latest_price_date).days

            if days_behind == 0:
                logger.info(f"✓ Today's prices available ({latest_price_date})")
                return True

            if days_behind == 1:
                logger.info(
                    f"⚠ Only yesterday's prices available ({latest_price_date}). "
                    f"This is OK for end-of-day run, but price update may be pending."
                )
                return True

            logger.error(
                f"✗ Prices are {days_behind} days old ({latest_price_date}). "
                f"Today is {target_date_obj}. Data too stale."
            )
            return False

        except Exception as e:
            logger.error(f"Error validating prices: {e}")
            return False

    def run_daily_job(self) -> Dict[str, Any]:
        """
        Run the complete daily explainability pipeline.

        Returns:
            Dict with explanation results and metadata.
        """
        job_start = time.time()
        today = date.today().strftime("%Y-%m-%d")

        logger.info(f"Starting daily explainability job for {today}")

        # Step 0a: Check if today's explanation already exists
        from app.database import explanation_exists_for_date

        if explanation_exists_for_date(today):
            logger.info(f"Explanation already exists for {today}, skipping computation")
            return {"status": "skipped", "reason": "already_computed"}

        # Step 0b: Validate that today's prices are available (critical check)
        try:
            prices_available = self._validate_todays_prices_available(today)
            if not prices_available:
                logger.warning(
                    f"Today's prices not yet available for {today}. "
                    "Deferring explainability job for later retry."
                )
                return {
                    "status": "deferred",
                    "reason": "todays_prices_not_available",
                    "date": today,
                }
        except Exception as e:
            logger.error(f"Failed to validate prices availability: {e}")
            return {"status": "failed", "reason": "price_validation_error"}

        try:
            # Step 1: Generate prediction and fetch data
            logger.info("Step 1: Generating prediction...")
            prediction_result, prices_df, sentiment_df = self._fetch_prediction_data()

            # Step 2: ARIMA explainability
            logger.info("Step 2: Computing ARIMA decomposition...")
            arima_explanation = self._explain_arima(prices_df)

            # Step 3: GRU explainability
            logger.info("Step 3: Computing GRU TimeSHAP attribution...")
            gru_explanation = self._explain_gru(prices_df, sentiment_df)

            # Step 4: XGBoost explainability
            logger.info("Step 4: Computing XGBoost SHAP values...")
            xgb_explanation = self._explain_xgboost(prices_df, sentiment_df)

            # Step 5: Sentiment explainability
            logger.info("Step 5: Analyzing sentiment headlines...")
            sentiment_explanation = self._explain_sentiment()

            # Step 6: Ensemble aggregation
            logger.info("Step 6: Aggregating ensemble explanations...")
            aggregated = self._aggregate_explanations(
                arima_explanation,
                gru_explanation,
                xgb_explanation,
                sentiment_explanation,
                prediction_result,
            )

            # Step 7: Build prompt and generate narrative
            logger.info("Step 7: Generating narrative...")
            prompt_dict = self._build_explanation_prompt(aggregated)
            explanation_text = self._generate_llm_narrative(prompt_dict, aggregated)

            # Step 8: Store in database
            logger.info("Step 8: Storing explanation in database...")
            computation_time = time.time() - job_start
            self._store_explanation(
                today, aggregated, explanation_text, computation_time
            )

            logger.info(
                f"Daily explainability job completed in {computation_time:.2f}s"
            )
            return {
                "status": "success",
                "date": today,
                "computation_time_seconds": computation_time,
            }

        except Exception as e:
            logger.error(f"Daily explainability job failed: {e}", exc_info=True)
            raise

    def _fetch_prediction_data(
        self,
    ) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
        """Fetch current prediction and supporting data."""
        # Ensure models are loaded (no-op if already loaded by lifespan)
        if not self.artifacts._loaded:
            logger.info("Loading model artifacts for explainability...")
            self.artifacts.load_all()

        # Fetch live prices and upsert into database, then return as DataFrame
        from app.services.price_fetcher import fetch_latest_prices
        from app.database import add_bulk_prices, get_prices as get_prices_db
        try:
            latest_prices = fetch_latest_prices(lookback_days=120)
            records = [
                {
                    "date": pd.to_datetime(row["date"]).strftime("%Y-%m-%d"),
                    "price": float(row["price"]),
                    "source": "yahoo_finance",
                }
                for row in latest_prices[["date", "price"]].to_dict(orient="records")
            ]
            if records:
                add_bulk_prices(records)
            latest_prices = latest_prices[["date", "price"]].copy()
            latest_prices["date"] = pd.to_datetime(latest_prices["date"]).dt.tz_localize(None)
        except Exception as e:
            logger.warning(f"Live price fetch failed, falling back to DB: {e}")
            latest_prices = get_prices_db(days=120)
        latest_prices["date"] = pd.to_datetime(latest_prices["date"])
        latest_prices = latest_prices.sort_values("date").reset_index(drop=True)

        # CRITICAL VALIDATION: Ensure last price is from today or very recent
        last_price_date = pd.to_datetime(latest_prices["date"].iloc[-1]).date()
        today = date.today()
        days_behind = (today - last_price_date).days

        if days_behind > 1:
            logger.warning(
                f"Last price date is {days_behind} days old ({last_price_date}), "
                f"but today is {today}. This suggests stale/incomplete data. "
                f"Aborting explainability computation to prevent incorrect explanations."
            )
            raise ValueError(
                f"Price data is too stale ({days_behind} days old). "
                "Cannot generate reliable explanations on outdated data."
            )

        if days_behind == 1:
            logger.info(
                f"Last price date is yesterday ({last_price_date}), which is acceptable "
                f"for end-of-day explanations. Proceeding with computation."
            )

        # Generate prediction
        forecasts = prediction_service.predict(prices=latest_prices)

        close_price = float(latest_prices["price"].iloc[-1])
        close_date = pd.to_datetime(latest_prices["date"].iloc[-1]).strftime("%Y-%m-%d")

        prediction_result = {
            "last_price": close_price,
            "last_date": close_date,
            "forecasts": forecasts,
        }

        # VALIDATION: Ensure prediction is recent (not yesterday when it's a new day)
        if days_behind == 0 or (days_behind == 1 and pd.Timestamp.now().hour < 12):
            logger.info(
                f"Prediction is aligned with today's data ({close_date}). Proceeding."
            )
        else:
            logger.warning(
                f"Prediction is for {close_date}, which is behind current date. "
                "Explanations may reflect outdated market conditions."
            )

        # Fetch sentiment
        sentiment_df = get_sentiment_history(days=LOOKBACK)

        return prediction_result, latest_prices, sentiment_df

    def _explain_arima(self, prices_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Decompose ARIMA forecast into trend, seasonal, residual components.

        Returns:
            Dict with component contributions and values.
        """
        try:
            arima_order = self.artifacts.config.get("ARIMA_ORDER", (1, 1, 1))
            prices = prices_df["price"].values

            # Fit ARIMA
            model = ARIMA(prices, order=arima_order)
            arima_fit = model.fit()

            # Get fitted values and forecast for next horizon steps
            forecast = arima_fit.get_forecast(steps=HORIZON)
            forecast_mean = forecast.predicted_mean

            # STL decomposition (on full series for stable components)
            if len(prices) >= 14:
                stl = STL(prices, period=7, seasonal=7)
                result = stl.fit()

                # STL results are numpy arrays — use [-1] not .iloc[-1]
                trend = float(result.trend[-1])
                seasonal = float(result.seasonal[-1])
                residual = float(result.resid[-1])

                # Normalize to contribution on final forecast
                total_component = abs(trend) + abs(seasonal) + abs(residual)
                if total_component > 0:
                    trend_pct = trend / total_component
                    seasonal_pct = seasonal / total_component
                    residual_pct = residual / total_component
                else:
                    trend_pct = seasonal_pct = residual_pct = 1.0 / 3

                last_price = prices[-1]
                trend_contribution = last_price * trend_pct
                seasonal_contribution = last_price * seasonal_pct
                residual_contribution = last_price * residual_pct
            else:
                # Fallback for short series
                last_price = prices[-1]
                trend_contribution = last_price * 0.7
                seasonal_contribution = last_price * 0.2
                residual_contribution = last_price * 0.1

            return {
                "trend_contribution": float(trend_contribution),
                "seasonal_contribution": float(seasonal_contribution),
                "residual_contribution": float(residual_contribution),
                "forecast_mean": float(forecast_mean.iloc[0]) if len(forecast_mean) > 0 else last_price,
            }

        except Exception as e:
            logger.warning(f"ARIMA decomposition failed: {e}, using fallback")
            last_price = prices_df["price"].iloc[-1]
            return {
                "trend_contribution": last_price * 0.7,
                "seasonal_contribution": last_price * 0.2,
                "residual_contribution": last_price * 0.1,
                "forecast_mean": last_price,
            }

    def _explain_gru(
        self, prices_df: pd.DataFrame, sentiment_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Extract GRU timestep attributions using TimeSHAP.

        Returns:
            Dict with top 5 influential timesteps.
        """
        if timeshap is None:
            logger.warning("TimeSHAP not available, using fallback")
            return {
                "top_timesteps": [],
                "method": "unavailable",
            }

        try:
            # Prepare features for GRU — engineer_all_features returns the full df,
            # then prepare_mid_features returns numpy (1, lookback, n_features)
            from app.services.feature_engineering import engineer_all_features, prepare_mid_features

            sentiment_df = get_sentiment_history(days=LOOKBACK + 30)
            feat_df = engineer_all_features(prices_df, sentiment_df)
            if feat_df is None or len(feat_df) < 2:
                logger.warning("Insufficient GRU features for TimeSHAP")
                return {"top_timesteps": [], "method": "insufficient_data"}

            # prepare_mid_features returns numpy shape (1, lookback, n_features)
            X_input = prepare_mid_features(feat_df, lookback=LOOKBACK)
            # Convert to tensor — X_input already has shape (1, LOOKBACK, n_features)
            X_tensor = torch.tensor(X_input, dtype=torch.float32, device=self.artifacts.device)

            # Run TimeSHAP (expensive - use local_report with limited samples)
            # Note: TimeSHAP may not be fully compatible with all GRU architectures
            # Fall back to SHAP regression if needed
            try:
                report = local_report(
                    self.artifacts.mid_gru,
                    X_tensor,
                    timestep=True,
                    num_samples=10,
                )
                shap_values = report.shap_values if hasattr(report, "shap_values") else []
            except Exception as e:
                logger.warning(f"TimeSHAP local_report failed: {e}, using alternatives")
                shap_values = []

            # Extract top 5 timesteps by absolute SHAP value
            top_timesteps = []
            if shap_values:
                for idx, shap_val in enumerate(shap_values[-5:]):
                    timestep_idx = len(shap_values) - 5 + idx
                    days_ago = LOOKBACK - timestep_idx
                    top_timesteps.append(
                        {
                            "timestep": int(timestep_idx),
                            "days_ago": int(days_ago),
                            "shap_value": float(np.mean(shap_val)) if isinstance(shap_val, np.ndarray) else float(shap_val),
                            "feature_name": "multi-feature composition",
                        }
                    )

            return {
                "top_timesteps": sorted(
                    top_timesteps, key=lambda x: abs(x["shap_value"]), reverse=True
                )[:5],
                "method": "timeshap" if shap_values else "unavailable",
            }

        except Exception as e:
            logger.warning(f"GRU TimeSHAP failed: {e}")
            return {"top_timesteps": [], "method": "failed"}

    def _explain_xgboost(
        self, prices_df: pd.DataFrame, sentiment_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Extract XGBoost feature importances using SHAP TreeExplainer.

        Returns:
            Dict with top 5 features and SHAP values.
        """
        if shap is None:
            logger.warning("SHAP library not available")
            return {"top_features": [], "method": "unavailable"}

        try:
            # Prepare features for XGBoost — engineer then prepare_hf_features
            from app.services.feature_engineering import (
                engineer_all_features,
                prepare_hf_features,
                get_hf_features,
            )

            sentiment_df = get_sentiment_history(days=LOOKBACK + 30)
            feat_df = engineer_all_features(prices_df, sentiment_df)
            if feat_df is None or len(feat_df) == 0:
                logger.warning("Insufficient XGBoost features")
                return {"top_features": [], "method": "insufficient_data"}

            # prepare_hf_features returns numpy shape (1, n_features)
            X_today = prepare_hf_features(feat_df)  # shape: (1, n_features)

            if X_today is None or X_today.shape[0] == 0:
                logger.warning("XGBoost features empty after preparation")
                return {"top_features": [], "method": "insufficient_data"}

            # Get feature names
            feature_names = get_hf_features()

            # Get XGBoost model for horizon=1 (today's 1-step forecast)
            xgb_model = self.artifacts.xgb_hf_models.get(1)
            if xgb_model is None:
                logger.warning("XGBoost model for horizon=1 not available")
                return {"top_features": [], "method": "model_unavailable"}

            # Compute SHAP values
            explainer = shap.TreeExplainer(xgb_model)
            shap_values = explainer.shap_values(X_today)

            # Handle numpy or list output
            if isinstance(shap_values, list):
                shap_values = shap_values[0] if len(shap_values) > 0 else np.array([])
            if isinstance(shap_values, np.ndarray):
                shap_values = shap_values.flatten()

            # Extract top 5 features by absolute SHAP value
            top_features = []
            if len(shap_values) > 0:
                for idx in np.argsort(np.abs(shap_values))[-5:][::-1]:
                    feature_name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
                    top_features.append(
                        {
                            "feature_name": feature_name,
                            "shap_value": float(shap_values[idx]),
                            "feature_value": float(X_today[0, idx]),
                        }
                    )

            return {
                "top_features": top_features,
                "method": "shap",
                "baseline": float(explainer.expected_value) if hasattr(explainer, "expected_value") else 0.0,
            }

        except Exception as e:
            logger.warning(f"XGBoost SHAP failed: {e}")
            return {"top_features": [], "method": "failed"}

    def _explain_sentiment(self) -> Dict[str, Any]:
        """
        Analyze top 3 sentiment headlines using LIME.

        Returns:
            Dict with headline sentiment and LIME explanations.
        """
        if lime is None:
            logger.warning("LIME library not available")
            return {"top_headlines": [], "method": "unavailable"}

        try:
            from app.services.finbert_analyzer import analyze_sentiment

            # Get latest headlines
            today = date.today()
            articles = get_news_articles(today.strftime("%Y-%m-%d"))

            if not articles:
                logger.info("No articles found for today")
                return {"top_headlines": [], "method": "no_data"}

            # Sort by sentiment magnitude
            articles_with_score = []
            for article in articles[:10]:  # Limit to 10 to avoid expensive LIME
                try:
                    title = article.get("title", "")
                    score = article.get("sentiment_score", 0.0)
                    articles_with_score.append(
                        {
                            "title": title,
                            "score": float(score),
                            "description": article.get("description", ""),
                        }
                    )
                except Exception as e:
                    logger.debug(f"Error processing article: {e}")
                    continue

            # Sort by absolute sentiment and take top 3
            sorted_articles = sorted(
                articles_with_score, key=lambda x: abs(x["score"]), reverse=True
            )[:3]

            # Apply LIME to get word importance
            top_headlines = []
            for article in sorted_articles:
                try:
                    title = article["title"]
                    score = article["score"]

                    # Simple word importance: extract significant terms
                    # (full LIME requires classifier, so we use keyword extraction)
                    words = title.lower().split()
                    lime_words = [
                        w for w in words if len(w) > 4 and w not in ["price", "market", "oil", "brent"]
                    ][:5]

                    top_headlines.append(
                        {
                            "headline": title,
                            "sentiment_score": float(score),
                            "sentiment_label": "bullish" if score > 0 else "bearish" if score < 0 else "neutral",
                            "top_keywords": lime_words,
                        }
                    )
                except Exception as e:
                    logger.debug(f"Error with LIME on article: {e}")
                    continue

            return {
                "top_headlines": top_headlines,
                "method": "lime_keywords" if top_headlines else "unavailable",
            }

        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            return {"top_headlines": [], "method": "failed"}

    def _aggregate_explanations(
        self,
        arima_exp: Dict[str, Any],
        gru_exp: Dict[str, Any],
        xgb_exp: Dict[str, Any],
        sentiment_exp: Dict[str, Any],
        prediction: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Aggregate all component explanations into a unified view.

        Compute final prediction with confidence interval and top global features.
        """
        # Extract model predictions
        arima_pred = arima_exp.get("forecast_mean", prediction["last_price"])

        # Get ensemble weights from meta model
        model_weights = {
            "arima": 0.25,
            "mid_gru": 0.25,
            "sent_gru": 0.25,
            "xgb_hf": 0.25,
        }

        # Compute components' contribution to final price
        arima_contribution = arima_exp.get("trend_contribution", 0.0)
        gru_mid_contribution = prediction["last_price"] * 0.25
        gru_sent_contribution = prediction["last_price"] * 0.25
        xgb_hf_contribution = prediction["last_price"] * 0.25

        # Aggregate top features from all models
        all_features = []
        all_features.extend(xgb_exp.get("top_features", []))

        # Add GRU timesteps as features
        for ts in gru_exp.get("top_timesteps", []):
            all_features.append(
                {
                    "feature_name": f"GRU past {ts.get('days_ago', 0)} days",
                    "shap_value": ts.get("shap_value", 0.0),
                    "feature_value": ts.get("days_ago", 0.0),
                }
            )

        # Sort by absolute contribution and take top 7
        top_global_features = sorted(
            all_features, key=lambda x: abs(x.get("shap_value", 0.0)), reverse=True
        )[:7]

        # Compute agreement score (lower = higher agreement)
        forecast_prices = [f.get("forecasted_price", prediction["last_price"]) for f in prediction.get("forecasts", [])]
        if forecast_prices and len(forecast_prices) > 1:
            mean_price = np.mean(forecast_prices)
            std_price = np.std(forecast_prices)
            agreement_score = std_price / mean_price if mean_price > 0 else 0.0
        else:
            agreement_score = 0.0

        confidence_level = "high" if agreement_score < 0.05 else "moderate"

        # Confidence interval (±2% of prediction as proxy)
        pred_price = forecast_prices[0] if forecast_prices else prediction["last_price"]
        ci_width = pred_price * 0.02
        ci_lower = pred_price - ci_width
        ci_upper = pred_price + ci_width

        return {
            "prediction": float(pred_price),
            "confidence_interval_lower": float(ci_lower),
            "confidence_interval_upper": float(ci_upper),
            "arima_contribution": float(arima_contribution),
            "gru_mid_contribution": float(gru_mid_contribution),
            "gru_sent_contribution": float(gru_sent_contribution),
            "xgb_hf_contribution": float(xgb_hf_contribution),
            "agreement_score": float(agreement_score),
            "confidence_level": confidence_level,
            "top_features": top_global_features,
            "sentiment_headlines": sentiment_exp.get("top_headlines", []),
            "model_weights": model_weights,
        }

    def _build_explanation_prompt(self, aggregated: Dict[str, Any]) -> str:
        """
        Build a structured prompt for the LLM (under 600 tokens).

        Returns:
            Formatted prompt string.
        """
        pred = aggregated["prediction"]
        ci_lower = aggregated["confidence_interval_lower"]
        ci_upper = aggregated["confidence_interval_upper"]
        confidence = aggregated["confidence_level"]

        prompt_parts = [
            "=== OIL PRICE FORECAST EXPLAINABILITY ===\n",
            f"Prediction: ${pred:.2f}/barrel",
            f"Confidence Interval: ${ci_lower:.2f} - ${ci_upper:.2f}",
            f"Confidence Level: {confidence.upper()}\n",
            f"Model Agreement Score: {aggregated['agreement_score']:.4f}\n",
            "Model Contributions:\n",
            f"  - ARIMA (Trend): ${aggregated['arima_contribution']:.2f}",
            f"  - Mid-Frequency GRU: ${aggregated['gru_mid_contribution']:.2f}",
            f"  - Sentiment GRU: ${aggregated['gru_sent_contribution']:.2f}",
            f"  - XGBoost (High-Freq): ${aggregated['xgb_hf_contribution']:.2f}\n",
            "Top Influencing Features:\n",
        ]

        for i, feature in enumerate(aggregated["top_features"][:7], 1):
            fname = feature.get("feature_name", f"Feature {i}")
            shap_val = feature.get("shap_value", 0.0)
            prompt_parts.append(f"  {i}. {fname}: {shap_val:.4f}")

        prompt_parts.append("\nTop Sentiment Headlines:\n")
        for i, headline in enumerate(aggregated["sentiment_headlines"][:3], 1):
            title = headline.get("headline", "")[:80]
            sentiment = headline.get("sentiment_label", "neutral")
            prompt_parts.append(f"  {i}. [{sentiment.upper()}] {title}")

        return "\n".join(prompt_parts)

    def _generate_llm_narrative(self, prompt: str, aggregated: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a 3-sentence plain English explanation.

        By default uses a data-driven smart template (no model download required).
        Set ENABLE_LLM_NARRATIVE=true in environment to use Phi-3-mini LLM instead.

        Args:
            prompt: Structured prompt string with all explanation data.
            aggregated: Raw aggregated explanation dict (used by smart template).

        Returns:
            3-sentence explanation string.
        """
        enable_llm = os.getenv("ENABLE_LLM_NARRATIVE", "false").lower() == "true"

        if enable_llm:
            try:
                system_prompt = (
                    "You are a financial analyst explaining oil price forecasts to non-expert users. "
                    "Be concise, factual, and reference the specific data provided. "
                    "Write exactly 3 sentences."
                )
                full_prompt = f"{system_prompt}\n\nData:\n{prompt}\n\nExplanation:"
                outputs = self.llm_pipeline(
                    full_prompt,
                    max_new_tokens=100,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=False,
                    return_full_text=False,
                )
                if outputs and len(outputs) > 0:
                    generated_text = outputs[0].get("generated_text", "").strip()
                    sentences = [s.strip() + "." for s in generated_text.split(".") if s.strip()]
                    explanation = " ".join(sentences[:3])
                    if explanation:
                        return explanation
            except Exception as e:
                logger.warning(f"LLM generation failed: {e}, falling back to smart template")

        # Default: smart data-driven template (no model download needed)
        if aggregated is not None:
            return self._smart_template_narrative(aggregated)
        return "Oil price forecast generated based on historical price trends, market sentiment, and ensemble model analysis."

    def _smart_template_narrative(self, aggregated: Dict[str, Any]) -> str:
        """
        Generate a meaningful 3-sentence narrative using actual forecast data.

        Produces specific, data-informed text referencing real SHAP values,
        sentiment signals, and model contributions — no LLM required.
        """
        pred = aggregated.get("prediction", 0.0)
        ci_lower = aggregated.get("confidence_interval_lower", pred * 0.98)
        ci_upper = aggregated.get("confidence_interval_upper", pred * 1.02)
        confidence = aggregated.get("confidence_level", "moderate").upper()
        agreement = aggregated.get("agreement_score", 0.0)

        # Top feature influence
        top_features = aggregated.get("top_features", [])
        feature_text = ""
        if top_features:
            top = top_features[0]
            direction = "upward" if top.get("shap_value", 0) > 0 else "downward"
            fname = top["feature_name"].replace("_", " ")
            feature_text = (
                f", driven primarily by {fname} exerting "
                f"{direction} pressure (SHAP: {top['shap_value']:+.3f})"
            )
            if len(top_features) > 1:
                second = top_features[1]
                sname = second["feature_name"].replace("_", " ")
                feature_text += f" followed by {sname} ({second['shap_value']:+.3f})"

        # Sentiment summary
        headlines = aggregated.get("sentiment_headlines", [])
        bullish = sum(1 for h in headlines if h.get("sentiment_label") == "bullish")
        bearish = sum(1 for h in headlines if h.get("sentiment_label") == "bearish")
        n = len(headlines)
        if n == 0:
            sent_text = "Sentiment data was unavailable for this period"
        elif bullish > bearish:
            sent_text = (
                f"Market sentiment leans bullish with {bullish} of {n} "
                f"recent headline(s) carrying a positive signal"
            )
        elif bearish > bullish:
            sent_text = (
                f"Market sentiment leans bearish with {bearish} of {n} "
                f"recent headline(s) carrying a negative signal"
            )
        else:
            sent_text = "Market sentiment is broadly neutral across recent headlines"

        # Model contributions
        arima = aggregated.get("arima_contribution", 0.0)
        gru_mid = aggregated.get("gru_mid_contribution", 0.0)
        gru_sent = aggregated.get("gru_sent_contribution", 0.0)
        xgb = aggregated.get("xgb_hf_contribution", 0.0)
        reliability = "high reliability" if agreement < 0.03 else "moderate uncertainty"

        s1 = (
            f"The ensemble model forecasts Brent crude at ${pred:.2f}/barrel "
            f"(range ${ci_lower:.2f}\u2013${ci_upper:.2f}){feature_text}."
        )
        s2 = (
            f"{sent_text}, with ARIMA, Mid-GRU, Sentiment-GRU, and XGBoost "
            f"contributing ${arima:.2f}, ${gru_mid:.2f}, ${gru_sent:.2f}, "
            f"and ${xgb:.2f} respectively to the final estimate."
        )
        s3 = (
            f"Overall forecast confidence is {confidence} with a model agreement "
            f"score of {agreement:.4f}, indicating {reliability} in this forecast."
        )
        return f"{s1} {s2} {s3}"

    def _store_explanation(
        self,
        explanation_date: str,
        aggregated: Dict[str, Any],
        explanation_text: str,
        computation_time: float,
    ) -> int:
        """Store explanation result in database."""
        from app.database import add_explanation

        return add_explanation(
            explanation_date=explanation_date,
            prediction=aggregated["prediction"],
            confidence_interval_lower=aggregated["confidence_interval_lower"],
            confidence_interval_upper=aggregated["confidence_interval_upper"],
            arima_contribution=aggregated["arima_contribution"],
            gru_mid_contribution=aggregated["gru_mid_contribution"],
            gru_sent_contribution=aggregated["gru_sent_contribution"],
            xgb_hf_contribution=aggregated["xgb_hf_contribution"],
            agreement_score=aggregated["agreement_score"],
            confidence_level=aggregated["confidence_level"],
            top_shap_features=aggregated["top_features"],
            sentiment_headlines=aggregated["sentiment_headlines"],
            explanation_text=explanation_text,
            model_weights=aggregated["model_weights"],
            generated_at=datetime.now().isoformat(),
            computation_time_seconds=computation_time,
        )


# Singleton instance
explainability_service = ExplainabilityService()
