"""
Prediction service - orchestrates the ensemble prediction pipeline.
"""

import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging

from statsmodels.tsa.arima.model import ARIMA

from app.models.model_loader import model_artifacts
from app.services.feature_engineering import (
    engineer_all_features,
    prepare_mid_features,
    prepare_sentiment_features,
    prepare_hf_features,
)
from app.database import get_sentiment_history
from app.services.sentiment_service import sentiment_service

logger = logging.getLogger(__name__)


class PredictionService:
    """
    Main prediction service that orchestrates the ensemble model.

    Pipeline:
    1. Fetch sentiment from database
    2. Feature engineering (price + sentiment)
    3. Component forecasts (ARIMA, Mid-GRU, Sent-GRU, XGBoost)
    4. Meta-ensemble (Ridge)
    5. Convert returns to prices
    """

    def __init__(self):
        self.artifacts = model_artifacts

    def predict(
        self,
        prices: pd.DataFrame = None,
        sentiment_df: pd.DataFrame = None,
    ) -> List[Dict[str, Any]]:
        """
        End-to-end prediction: Fetch history -> Engineer features -> Forecast.

        Args:
            prices: Optional historical prices (if None, fetches from Yahoo Finance)
            sentiment_df: Optional sentiment history (if None, fetches from database)

        Returns:
            List of 14-day forecast dictionaries
        """
        logger.info("Starting prediction pipeline...")

        # 1. Handle Data
        prices = self._handle_price_data(prices)
        sentiment_df = self._handle_sentiment_data(sentiment_df, prices)

        # 2. Feature Engineering
        logger.info("Step 1: Feature engineering")
        df = self._prepare_features(prices, sentiment_df)

        # 3. Compute sentiment signals for trend reversal detection
        sentiment_signals = self._compute_sentiment_signals(sentiment_df, prices)

        # 4. Generate forecasts (ARIMA, Mid-GRU, Sent-GRU, XGBoost)
        logger.info("Step 2: Generating component forecasts")
        component_forecasts = self._generate_component_forecasts(df)

        # 5. Meta-ensemble and Format
        logger.info("Step 3: Meta-ensemble combination")
        ensemble_returns = self._meta_ensemble(
            *component_forecasts, self.artifacts.horizon
        )

        # 6. Apply sentiment-based adjustments for trend reversal detection
        ensemble_returns = self._apply_sentiment_adjustment(
            ensemble_returns, sentiment_signals
        )

        # 7. Convert returns to prices
        logger.info("Step 4: Converting returns to prices")
        recent_volatility = self._calculate_recent_volatility(prices)
        forecasts = self._returns_to_prices(
            ensemble_returns,
            float(prices["price"].iloc[-1]),
            pd.to_datetime(prices["date"].iloc[-1]),
            self.artifacts.horizon,
            recent_volatility,
        )

        logger.info("Prediction pipeline complete!")
        return forecasts

    def _handle_price_data(self, prices: pd.DataFrame = None) -> pd.DataFrame:
        """Handle price data loading and validation."""
        if prices is None:
            logger.info("Auto-fetching 120 days of price history...")
            from app.services.price_fetcher import fetch_latest_prices
            return fetch_latest_prices(lookback_days=120)
        
        logger.info(f"Using provided price data ({len(prices)} points)")
        if "date" not in prices.columns or "price" not in prices.columns:
            raise ValueError("Price DataFrame must have 'date' and 'price' columns")
        
        prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None)
        return prices.sort_values("date").reset_index(drop=True)

    def _handle_sentiment_data(
        self, sentiment_df: pd.DataFrame = None, prices: pd.DataFrame = None
    ) -> Optional[pd.DataFrame]:
        """Handle sentiment data loading with fallback to price-only mode."""
        if sentiment_df is not None:
            logger.info(f"Using provided sentiment data ({len(sentiment_df)} days)")
            return sentiment_df

        logger.info("Fetching sentiment data from database...")
        try:
            price_end_date = pd.to_datetime(prices["date"].iloc[-1])
            sentiment_df = sentiment_service.get_sentiment_window(
                days=120, end_date=price_end_date
            )

            if sentiment_df is not None and not sentiment_df.empty:
                logger.info(
                    f"Loaded {len(sentiment_df)} days of sentiment data (up to {price_end_date.date()})"
                )
                return sentiment_service.apply_cross_day_decay(sentiment_df)
            
            logger.warning("No sentiment data for this date range - using price-only mode")
            return None
        except Exception as e:
            logger.warning(f"Failed to load sentiment: {e}. Using price-only mode.")
            return None

    def _prepare_features(
        self, prices: pd.DataFrame, sentiment_df: pd.DataFrame = None
    ) -> pd.DataFrame:
        """Engineer features and validate data quality."""
        df = engineer_all_features(prices, sentiment_df=sentiment_df)
        df = df.dropna().tail(self.artifacts.lookback)

        if len(df) < self.artifacts.lookback:
            raise ValueError(
                f"Insufficient valid data points after feature engineering. "
                f"Got {len(df)}, need {self.artifacts.lookback}."
            )
        return df

    def _generate_component_forecasts(self, df: pd.DataFrame) -> tuple:
        """Generate all component forecasts and return as tuple."""
        trend_fc = self._arima_forecast(df, self.artifacts.horizon)
        mid_fc = self._mid_gru_forecast(df, self.artifacts.lookback)
        sent_fc = self._sent_gru_forecast(df, self.artifacts.lookback)
        hf_fc = self._xgb_hf_forecast(df, self.artifacts.horizon)
        return trend_fc, mid_fc, sent_fc, hf_fc

    def _calculate_recent_volatility(self, prices: pd.DataFrame) -> float:
        """Calculate recent 7-day volatility from price data."""
        if len(prices) < 8:
            return 0.015  # Default 1.5%

        recent_prices = prices["price"].tail(8).values
        recent_momentum = np.log(recent_prices[-1] / recent_prices[0])
        logger.info(
            f"Recent 7-day momentum: {recent_momentum:.4f} ({recent_momentum*100:.2f}%)"
        )

        recent_returns = np.diff(np.log(recent_prices))
        recent_volatility = np.std(recent_returns)
        recent_vol_pct = recent_volatility * 100
        logger.info(
            "Recent 7-day volatility: %.4f (%.2f%% daily)",
            recent_volatility,
            recent_vol_pct,
        )
        return recent_volatility

    def _compute_sentiment_signals(
        self, sentiment_df: pd.DataFrame, prices: pd.DataFrame
    ) -> dict:
        """
        Compute sentiment-based signals for trend reversal detection.

        Returns:
            Dictionary with sentiment signals
        """
        signals = self._init_sentiment_signals()

        if sentiment_df is None or sentiment_df.empty:
            return signals

        try:
            sent_col = self._get_sentiment_column(sentiment_df)
            if sent_col is None:
                return signals

            sentiment_df = self._prepare_sentiment_data(sentiment_df)
            prices = self._prepare_price_data(prices)

            # Calculate individual signal components
            self._calculate_sentiment_momentum(sentiment_df, sent_col, signals)
            self._calculate_price_momentum(prices, signals)
            self._calculate_divergence(signals)
            self._detect_extreme_sentiment(sentiment_df, sent_col, signals)

            # Compute reversal probability and adjustment
            self._calculate_reversal_probability(signals)

        except Exception as e:
            logger.warning(f"Failed to compute sentiment signals: {e}")

        return signals

    def _init_sentiment_signals(self) -> dict:
        """Initialize sentiment signals dictionary."""
        return {
            "sentiment_momentum": 0.0,
            "divergence": 0.0,
            "extreme_sentiment": 0.0,
            "reversal_probability": 0.0,
            "adjustment_factor": 0.0,
            "momentum_persistence": 0.0,
            "trend_strength": 0.0,
            "price_drop_7d": 0.0,
        }

    def _get_sentiment_column(self, sentiment_df: pd.DataFrame) -> Optional[str]:
        """Get the appropriate sentiment column name."""
        if "daily_sentiment" in sentiment_df.columns:
            return "daily_sentiment"
        if "daily_sentiment_decay" in sentiment_df.columns:
            return "daily_sentiment_decay"
        return None

    def _prepare_sentiment_data(self, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare sentiment data for analysis."""
        sentiment_df = sentiment_df.copy()
        sentiment_df["date"] = pd.to_datetime(sentiment_df["date"])
        return sentiment_df.sort_values("date")

    def _prepare_price_data(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Prepare price data for analysis."""
        prices = prices.copy()
        prices["date"] = pd.to_datetime(prices["date"])
        return prices

    def _calculate_sentiment_momentum(
        self, sentiment_df: pd.DataFrame, sent_col: str, signals: dict
    ) -> None:
        """Calculate sentiment momentum and persistence."""
        if len(sentiment_df) < 7:
            return

        recent_sentiment = sentiment_df[sent_col].tail(7).values
        if len(recent_sentiment) < 2:
            return

        sentiment_momentum = recent_sentiment[-1] - recent_sentiment[0]
        signals["sentiment_momentum"] = sentiment_momentum
        logger.info(f"Sentiment momentum (7d): {sentiment_momentum:.4f}")

        # Momentum persistence
        daily_changes = np.diff(recent_sentiment)
        positive_days = np.sum(daily_changes > 0)
        negative_days = np.sum(daily_changes < 0)

        signals["momentum_persistence"] = self._determine_persistence(
            positive_days, negative_days
        )

    def _determine_persistence(self, positive_days: int, negative_days: int) -> float:
        """Determine momentum persistence score."""
        if positive_days >= 5:
            logger.info(f"STRONG momentum persistence: {positive_days}/6 days positive")
            return 1.0
        if positive_days >= 4:
            return 0.5
        if negative_days >= 5:
            logger.info(f"STRONG negative persistence: {negative_days}/6 days negative")
            return -1.0
        if negative_days >= 4:
            return -0.5
        return 0.0

    def _calculate_price_momentum(self, prices: pd.DataFrame, signals: dict) -> None:
        """Calculate price momentum and trend strength."""
        if len(prices) < 7:
            return

        price_momentum = np.log(prices["price"].iloc[-1] / prices["price"].iloc[-7])
        signals["price_drop_7d"] = price_momentum
        signals["trend_strength"] = self._determine_trend_strength(price_momentum)

    def _determine_trend_strength(self, price_momentum: float) -> float:
        """Determine trend strength from price momentum."""
        if price_momentum < -0.05:
            logger.info("STRONG DOWNTREND detected: %.1f%% in 7 days", price_momentum * 100)
            return -2.0
        if price_momentum < -0.03:
            return -1.0
        if price_momentum > 0.05:
            return 2.0
        if price_momentum > 0.03:
            return 1.0
        return 0.0

    def _calculate_divergence(self, signals: dict) -> None:
        """Calculate sentiment-price divergence."""
        if signals["sentiment_momentum"] == 0 or signals["price_drop_7d"] == 0:
            return

        norm_sent = np.sign(signals["sentiment_momentum"]) * min(
            abs(signals["sentiment_momentum"]), 1.0
        )
        norm_price = np.sign(signals["price_drop_7d"]) * min(abs(signals["price_drop_7d"]), 0.1) * 10

        divergence = norm_sent - norm_price

        # Amplify divergence when price moved significantly
        if abs(signals["price_drop_7d"]) > 0.03:
            divergence *= 1.5

        signals["divergence"] = divergence
        logger.info(
            f"Sentiment-Price divergence: {divergence:.4f} (sent={norm_sent:.2f}, price={norm_price:.2f})"
        )

    def _detect_extreme_sentiment(
        self, sentiment_df: pd.DataFrame, sent_col: str, signals: dict
    ) -> None:
        """Detect extreme sentiment using z-score."""
        if len(sentiment_df) < 30:
            return

        current_sentiment = sentiment_df[sent_col].iloc[-1]
        sent_mean = sentiment_df[sent_col].tail(30).mean()
        sent_std = sentiment_df[sent_col].tail(30).std()

        if sent_std <= 0:
            return

        z_score = (current_sentiment - sent_mean) / sent_std
        signals["extreme_sentiment"] = z_score

        if z_score < -1.5:
            logger.info(f"CONTRARIAN SIGNAL: Extreme negative sentiment (z={z_score:.2f})")
        elif z_score > 1.5:
            logger.info(f"CONTRARIAN SIGNAL: Extreme positive sentiment (z={z_score:.2f})")

    def _calculate_reversal_probability(self, signals: dict) -> None:
        """Calculate reversal probability and adjustment factor."""
        bullish_score = self._calculate_bullish_signals(signals)
        bearish_score = self._calculate_bearish_signals(signals)

        signals["reversal_probability"] = bullish_score - bearish_score

        # Determine adjustment based on signal confidence
        confirming_signals = self._count_confirming_signals(signals)
        adj_per_unit = self._determine_adjustment_rate(confirming_signals, signals["reversal_probability"])

        signals["adjustment_factor"] = np.clip(
            signals["reversal_probability"] * adj_per_unit, -0.010, 0.010
        )

        logger.info(
            f"Reversal signals: bull={bullish_score:.2f}, bear={bearish_score:.2f}, "
            f"prob={signals['reversal_probability']:.2f}, adj={signals['adjustment_factor']*100:.3f}%"
        )

    def _calculate_bullish_signals(self, signals: dict) -> float:
        """Calculate bullish reversal score."""
        score = 0.0

        # Sentiment improving rapidly
        if signals["sentiment_momentum"] > 0.15:
            score += 0.4
        elif signals["sentiment_momentum"] > 0.1:
            score += 0.3
        elif signals["sentiment_momentum"] > 0.05:
            score += 0.15

        # Positive divergence
        if signals["divergence"] > 1.0:
            score += 0.4
        elif signals["divergence"] > 0.5:
            score += 0.3
        elif signals["divergence"] > 0.2:
            score += 0.15

        # Extreme negative sentiment (contrarian)
        if signals["extreme_sentiment"] < -2.0:
            score += 0.3
        elif signals["extreme_sentiment"] < -1.5:
            score += 0.2
        elif signals["extreme_sentiment"] < -1.0:
            score += 0.1

        # Momentum persistence bonus
        if signals["momentum_persistence"] > 0:
            score += 0.15 * signals["momentum_persistence"]

        # Reversal pattern: downtrend + improving sentiment
        if signals["trend_strength"] < -1 and signals["sentiment_momentum"] > 0.10:
            score += 0.25
            logger.info("REVERSAL PATTERN: Strong downtrend + improving sentiment")

        return score

    def _calculate_bearish_signals(self, signals: dict) -> float:
        """Calculate bearish reversal score."""
        score = 0.0

        # Sentiment declining rapidly
        if signals["sentiment_momentum"] < -0.15:
            score += 0.4
        elif signals["sentiment_momentum"] < -0.1:
            score += 0.3
        elif signals["sentiment_momentum"] < -0.05:
            score += 0.15

        # Negative divergence
        if signals["divergence"] < -1.0:
            score += 0.4
        elif signals["divergence"] < -0.5:
            score += 0.3
        elif signals["divergence"] < -0.2:
            score += 0.15

        # Extreme positive sentiment (contrarian)
        if signals["extreme_sentiment"] > 2.0:
            score += 0.3
        elif signals["extreme_sentiment"] > 1.5:
            score += 0.2
        elif signals["extreme_sentiment"] > 1.0:
            score += 0.1

        # Negative momentum persistence
        if signals["momentum_persistence"] < 0:
            score += 0.15 * abs(signals["momentum_persistence"])

        # Reversal pattern: uptrend + declining sentiment
        if signals["trend_strength"] > 1 and signals["sentiment_momentum"] < -0.05:
            score += 0.25
            logger.info("REVERSAL PATTERN: Strong uptrend + declining sentiment")

        return score

    def _count_confirming_signals(self, signals: dict) -> int:
        """Count how many indicators are triggering."""
        count = 0
        if abs(signals["sentiment_momentum"]) > 0.08:
            count += 1
        if abs(signals["divergence"]) > 0.4:
            count += 1
        if abs(signals["extreme_sentiment"]) > 1.2:
            count += 1
        if abs(signals["momentum_persistence"]) > 0.4:
            count += 1
        if abs(signals["trend_strength"]) > 0.5:
            count += 1
        return count

    def _determine_adjustment_rate(self, confirming_signals: int, prob: float) -> float:
        """Determine adjustment rate based on signal confidence."""
        if confirming_signals >= 3 and abs(prob) > 0.5:
            logger.info(f"HIGH CONFIDENCE: {confirming_signals} signals confirming")
            return 0.006
        if confirming_signals >= 2 and abs(prob) > 0.4:
            logger.info(f"MEDIUM CONFIDENCE: {confirming_signals} signals confirming")
            return 0.004
        if abs(prob) > 0.2:
            logger.info(f"LOW CONFIDENCE: Only {confirming_signals} signals, skipping adjustment")
        return 0.0

    def _apply_sentiment_adjustment(
        self, returns: np.ndarray, signals: dict
    ) -> np.ndarray:
        """
        Apply sentiment-based adjustment to predicted returns with decay.
        """
        adjustment = signals.get("adjustment_factor", 0.0)

        if abs(adjustment) < 0.0001:
            return returns

        adjusted_returns = returns.copy()
        decay_rate = self._determine_decay_rate(signals)

        for i in range(len(adjusted_returns)):
            decay = np.exp(-decay_rate * i)
            adjusted_returns[i] += adjustment * decay

        total_adj = sum(adjustment * np.exp(-decay_rate * i) for i in range(len(adjusted_returns)))
        logger.info(
            f"Applied sentiment adjustment: {adjustment*100:.3f}%/day (decay={decay_rate}, total={total_adj*100:.2f}%)"
        )
        return adjusted_returns

    def _determine_decay_rate(self, signals: dict) -> float:
        """Determine decay rate based on signal strength and persistence."""
        prob = abs(signals.get("reversal_probability", 0))
        persistence = abs(signals.get("momentum_persistence", 0))

        if prob > 0.6 and persistence > 0.5:
            return 0.05  # Very slow decay for high-confidence persistent signals
        if prob > 0.4:
            return 0.08  # Slower decay for strong signals
        return 0.12  # Normal decay

    def _arima_forecast(self, df: pd.DataFrame, horizon: int) -> np.ndarray:
        """
        Generate ARIMA forecast for trend component.

        Note: In production, we simplify by fitting ARIMA on log returns directly
        since we don't have the full VMD decomposition available.
        """
        try:
            # Use log returns as proxy for trend
            returns = df["log_return"].values

            # Fit ARIMA with saved order
            order = self.artifacts.arima_order
            model = ARIMA(returns, order=order)
            result = model.fit()

            # Forecast
            forecast = result.forecast(steps=horizon)

            logger.info(f"ARIMA forecast generated with order {order}")
            return forecast

        except Exception as e:
            logger.warning(f"ARIMA failed: {e}. Using zero forecast.")
            return np.zeros(horizon)

    def _mid_gru_forecast(self, df: pd.DataFrame, lookback: int) -> np.ndarray:
        """Generate mid-frequency GRU forecast."""
        try:
            # Prepare features
            x_mid = prepare_mid_features(df, lookback)

            # Scale features
            x_mid_flat = x_mid.reshape(-1, x_mid.shape[-1])
            x_mid_scaled = self.artifacts.scaler_mid.transform(x_mid_flat)
            x_mid_scaled = x_mid_scaled.reshape(1, lookback, -1)

            # Convert to tensor
            x_tensor = torch.tensor(x_mid_scaled, dtype=torch.float32)
            x_tensor = x_tensor.to(self.artifacts.device)

            # Predict
            with torch.no_grad():
                forecast = self.artifacts.mid_gru(x_tensor).cpu().numpy().flatten()

            logger.info("Mid-GRU forecast generated")
            return forecast

        except Exception as e:
            logger.error(f"Mid-GRU failed: {e}")
            raise

    def _sent_gru_forecast(self, df: pd.DataFrame, lookback: int) -> np.ndarray:
        """
        Generate sentiment-aware GRU forecast.

        Note: In Phase 1, sentiment features are zeros.
        """
        try:
            # Prepare features
            x_price, x_sent = prepare_sentiment_features(df, lookback)

            # Scale features
            x_price_flat = x_price.reshape(-1, x_price.shape[-1])
            x_sent_flat = x_sent.reshape(-1, x_sent.shape[-1])

            x_price_scaled = self.artifacts.scaler_price.transform(x_price_flat)
            x_sent_scaled = self.artifacts.scaler_sent.transform(x_sent_flat)

            x_price_scaled = x_price_scaled.reshape(1, lookback, -1)
            x_sent_scaled = x_sent_scaled.reshape(1, lookback, -1)

            # Convert to tensors
            x_price_tensor = torch.tensor(x_price_scaled, dtype=torch.float32).to(
                self.artifacts.device
            )
            x_sent_tensor = torch.tensor(x_sent_scaled, dtype=torch.float32).to(
                self.artifacts.device
            )

            # Predict
            with torch.no_grad():
                forecast = (
                    self.artifacts.sent_gru(x_price_tensor, x_sent_tensor)
                    .cpu()
                    .numpy()
                    .flatten()
                )

            logger.info("Sentiment-GRU forecast generated")
            return forecast

        except Exception as e:
            logger.error(f"Sentiment-GRU failed: {e}")
            raise

    def _xgb_hf_forecast(self, df: pd.DataFrame, horizon: int) -> np.ndarray:
        """Generate XGBoost high-frequency forecasts for each horizon."""
        try:
            # Prepare features (only last row)
            x_hf = prepare_hf_features(df)

            # Predict for each horizon
            forecasts = []
            for h in range(1, horizon + 1):
                pred = self.artifacts.xgb_hf_models[h].predict(x_hf)[0]
                forecasts.append(pred)

            logger.info("XGBoost HF forecasts generated")
            return np.array(forecasts)

        except Exception as e:
            logger.error(f"XGBoost HF failed: {e}")
            raise

    def _meta_ensemble(
        self,
        trend_fc: np.ndarray,
        mid_fc: np.ndarray,
        sent_fc: np.ndarray,
        hf_fc: np.ndarray,
        horizon: int,
    ) -> np.ndarray:
        """
        Combine component forecasts using Ridge meta-models.

        For each horizon h:
        1. Stack [trend, mid, sent, hf] predictions
        2. Scale with meta_scalers[h]
        3. Predict with meta_models[h]
        """
        ensemble_fc = []

        for h in range(1, horizon + 1):
            # Stack component forecasts
            x_meta = np.array(
                [[trend_fc[h - 1], mid_fc[h - 1], sent_fc[h - 1], hf_fc[h - 1]]]
            )

            # Scale
            x_meta_scaled = self.artifacts.meta_scalers[h].transform(x_meta)

            # Predict
            pred = self.artifacts.meta_models[h].predict(x_meta_scaled)[0]
            ensemble_fc.append(pred)

        logger.info("Meta-ensemble combination complete")
        return np.array(ensemble_fc)

    def _returns_to_prices(
        self,
        returns: np.ndarray,
        last_price: float,
        last_date: pd.Timestamp,
        horizon: int,
        recent_volatility: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Convert log returns to price forecast with volatility-adjusted scaling.

        Args:
            returns: Predicted log returns from ensemble
            last_price: Last known price
            last_date: Last known date
            horizon: Forecast horizon
            recent_volatility: Recent volatility (7-day std of returns)
        """
        vol_multiplier = self._calculate_volatility_multiplier(recent_volatility)
        logger.info(f"Volatility multiplier: {vol_multiplier:.2f}x")

        forecasts = []
        current_price = last_price
        current_date = last_date

        for i in range(horizon):
            current_date = self._next_business_day_simple(current_date)
            ret = self._apply_return_adjustments(returns[i], vol_multiplier, i)
            next_price = current_price * np.exp(ret)

            forecasts.append({
                "horizon": i + 1,
                "date": current_date.strftime("%Y-%m-%d"),
                "forecasted_price": round(next_price, 2),
                "forecasted_return": round(float(ret), 4),
            })

            current_price = next_price

        return forecasts

    def _calculate_volatility_multiplier(self, recent_volatility: float) -> float:
        """Calculate volatility multiplier with caps."""
        BASELINE_VOL = 0.018
        if recent_volatility <= 0:
            return 1.0
        
        vol_multiplier = max(1.0, recent_volatility / BASELINE_VOL)
        return min(vol_multiplier, 1.3)

    def _next_business_day_simple(self, date: pd.Timestamp) -> pd.Timestamp:
        """Get next business day (skip weekends)."""
        date += pd.Timedelta(days=1)
        while date.weekday() >= 5:
            date += pd.Timedelta(days=1)
        return date

    def _apply_return_adjustments(self, ret: float, vol_multiplier: float, day_index: int) -> float:
        """Apply volatility scaling and caps to return."""
        ret = ret * vol_multiplier

        # Soft cap at 2%
        SOFT_CAP = 0.02
        if abs(ret) > SOFT_CAP:
            sign = 1 if ret > 0 else -1
            excess = abs(ret) - SOFT_CAP
            ret = sign * (SOFT_CAP + excess * 0.5)

        # Hard cap at 3%
        HARD_CAP = 0.03
        if abs(ret) > HARD_CAP:
            sign = 1 if ret > 0 else -1
            logger.warning(f"Capping prediction day {day_index+1}: {ret:.2%} -> {sign*HARD_CAP:.2%}")
            ret = sign * HARD_CAP

        return ret

    def _next_business_day(
        self, start_date: pd.Timestamp, days_ahead: int
    ) -> pd.Timestamp:
        """Get the next business day N days ahead."""
        business_days = pd.bdate_range(start=start_date, periods=days_ahead + 1)
        return business_days[-1]


# Global singleton
prediction_service = PredictionService()
