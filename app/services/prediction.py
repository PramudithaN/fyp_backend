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
    prepare_hf_features
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
    
    def predict(self, 
                prices: pd.DataFrame = None,
                sentiment_df: pd.DataFrame = None,
                days_of_history: int = 30) -> List[Dict[str, Any]]:
        """
        End-to-end prediction: Fetch history -> Engineer features -> Forecast.
        
        Args:
            prices: Optional historical prices (if None, fetches from Yahoo Finance)
            sentiment_df: Optional sentiment history (if None, fetches from database)
            days_of_history: Minimum trading days required for features (if auto-fetching)
        
        Returns:
            List of 14-day forecast dictionaries
        """
        logger.info("Starting prediction pipeline...")
        
        # 1. Handle Price Data
        if prices is None:
            logger.info(f"Auto-fetching {days_of_history} days of price history...")
            from app.services.price_fetcher import fetch_latest_prices
            prices = fetch_latest_prices(lookback_days=90)
        else:
            logger.info(f"Using provided price data ({len(prices)} points)")
            # Ensure required columns
            if 'date' not in prices.columns or 'price' not in prices.columns:
                raise ValueError("Price DataFrame must have 'date' and 'price' columns")
            prices['date'] = pd.to_datetime(prices['date']).dt.tz_localize(None)
            prices = prices.sort_values('date').reset_index(drop=True)

        # 2. Handle Sentiment Data - ENABLED for news-driven momentum capture
        if sentiment_df is None:
            logger.info("Fetching sentiment data from database...")
            try:
                # Get the end date from prices for proper alignment
                price_end_date = pd.to_datetime(prices['date'].iloc[-1])
                
                # Get sentiment history aligned with price data
                sentiment_df = sentiment_service.get_sentiment_window(days=90, end_date=price_end_date)
                
                if sentiment_df is not None and not sentiment_df.empty:
                    logger.info(f"Loaded {len(sentiment_df)} days of sentiment data (up to {price_end_date.date()})")
                    # Apply cross-day decay
                    sentiment_df = sentiment_service.apply_cross_day_decay(sentiment_df)
                else:
                    logger.warning("No sentiment data for this date range - using price-only mode")
                    sentiment_df = None
            except Exception as e:
                logger.warning(f"Failed to load sentiment: {e}. Using price-only mode.")
                sentiment_df = None
        else:
            logger.info(f"Using provided sentiment data ({len(sentiment_df)} days)")
        
        # 3. Feature Engineering
        logger.info("Step 1: Feature engineering")
        df = engineer_all_features(prices, sentiment_df=sentiment_df)
        
        # Drop initial NaN rows used for rolling features
        # We need at least 'lookback' valid rows for the GRUs
        df = df.dropna().tail(self.artifacts.lookback)
        
        if len(df) < self.artifacts.lookback:
            raise ValueError(f"Insufficient valid data points after feature engineering. Got {len(df)}, need {self.artifacts.lookback}.")
        
        # 4. Compute sentiment signals for trend reversal detection
        sentiment_signals = self._compute_sentiment_signals(sentiment_df, prices)
        
        # 5. Generate forecasts (ARIMA, Mid-GRU, Sent-GRU, XGBoost)
        logger.info("Step 2: Generating component forecasts")
        trend_fc = self._arima_forecast(df, self.artifacts.horizon)
        mid_fc = self._mid_gru_forecast(df, self.artifacts.lookback)
        sent_fc = self._sent_gru_forecast(df, self.artifacts.lookback)
        hf_fc = self._xgb_hf_forecast(df, self.artifacts.horizon)
        
        # 6. Meta-ensemble and Format
        logger.info("Step 3: Meta-ensemble combination")
        ensemble_returns = self._meta_ensemble(trend_fc, mid_fc, sent_fc, hf_fc, self.artifacts.horizon)
        
        # 7. Apply sentiment-based adjustments for trend reversal detection
        ensemble_returns = self._apply_sentiment_adjustment(ensemble_returns, sentiment_signals)
        
        logger.info("Step 4: Converting returns to prices")
        last_price = float(prices['price'].iloc[-1])
        last_date = pd.to_datetime(prices['date'].iloc[-1])
        
        # Calculate recent momentum (7-day cumulative return)
        recent_prices = prices['price'].tail(8).values  # Need 8 to compute 7-day return
        if len(recent_prices) >= 8:
            recent_momentum = np.log(recent_prices[-1] / recent_prices[0])
            logger.info(f"Recent 7-day momentum: {recent_momentum:.4f} ({recent_momentum*100:.2f}%)")
        else:
            recent_momentum = 0.0
        
        # Calculate recent volatility (7-day std of daily returns)
        if len(prices) >= 8:
            recent_returns = np.diff(np.log(prices['price'].tail(8).values))
            recent_volatility = np.std(recent_returns)
            logger.info(f"Recent 7-day volatility: {recent_volatility:.4f} ({recent_volatility*100:.2f}% daily)")
        else:
            recent_volatility = 0.015  # Default 1.5%
        
        forecasts = self._returns_to_prices(
            ensemble_returns, last_price, last_date, self.artifacts.horizon, 
            recent_momentum, recent_volatility
        )
        
        logger.info("Prediction pipeline complete!")
        return forecasts
    
    def _compute_sentiment_signals(self, sentiment_df: pd.DataFrame, prices: pd.DataFrame) -> dict:
        """
        Compute sentiment-based signals for trend reversal detection.
        
        IMPROVED VERSION with:
        1. Sentiment momentum (rate of change)
        2. Sentiment-price divergence with weighting
        3. Extreme sentiment (contrarian indicator)
        4. Momentum persistence (consistency bonus)
        5. Trend strength indicator (price drop + sentiment rise)
        6. Adaptive adjustment based on signal strength
        
        Returns:
            Dictionary with sentiment signals
        """
        signals = {
            'sentiment_momentum': 0.0,
            'divergence': 0.0,
            'extreme_sentiment': 0.0,
            'reversal_probability': 0.0,
            'adjustment_factor': 0.0,
            'momentum_persistence': 0.0,
            'trend_strength': 0.0,
            'price_drop_7d': 0.0
        }
        
        if sentiment_df is None or sentiment_df.empty:
            return signals
        
        try:
            # Get sentiment column name
            sent_col = 'daily_sentiment' if 'daily_sentiment' in sentiment_df.columns else 'daily_sentiment_decay'
            if sent_col not in sentiment_df.columns:
                return signals
            
            sentiment_df = sentiment_df.copy()
            sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
            sentiment_df = sentiment_df.sort_values('date')
            
            # 1. Sentiment Momentum (7-day change in sentiment)
            if len(sentiment_df) >= 7:
                recent_sentiment = sentiment_df[sent_col].tail(7).values
                if len(recent_sentiment) >= 2:
                    sentiment_momentum = recent_sentiment[-1] - recent_sentiment[0]
                    signals['sentiment_momentum'] = sentiment_momentum
                    logger.info(f"Sentiment momentum (7d): {sentiment_momentum:.4f}")
                    
                    # NEW: Momentum Persistence - check if sentiment consistently improving
                    daily_changes = np.diff(recent_sentiment)
                    positive_days = np.sum(daily_changes > 0)
                    negative_days = np.sum(daily_changes < 0)
                    
                    if positive_days >= 5:  # 5+ of 6 days positive
                        signals['momentum_persistence'] = 1.0
                        logger.info(f"STRONG momentum persistence: {positive_days}/6 days positive")
                    elif positive_days >= 4:
                        signals['momentum_persistence'] = 0.5
                    elif negative_days >= 5:
                        signals['momentum_persistence'] = -1.0
                        logger.info(f"STRONG negative persistence: {negative_days}/6 days negative")
                    elif negative_days >= 4:
                        signals['momentum_persistence'] = -0.5
            
            # 2. Price momentum for divergence detection
            prices = prices.copy()
            prices['date'] = pd.to_datetime(prices['date'])
            price_momentum = 0.0
            if len(prices) >= 7:
                price_momentum = np.log(prices['price'].iloc[-1] / prices['price'].iloc[-7])
                signals['price_drop_7d'] = price_momentum
                
                # NEW: Trend Strength - significant price drop
                if price_momentum < -0.05:  # >5% drop in 7 days
                    signals['trend_strength'] = -2.0  # Strong downtrend
                    logger.info(f"STRONG DOWNTREND detected: {price_momentum*100:.1f}% in 7 days")
                elif price_momentum < -0.03:  # >3% drop
                    signals['trend_strength'] = -1.0
                elif price_momentum > 0.05:  # >5% rise
                    signals['trend_strength'] = 2.0
                elif price_momentum > 0.03:
                    signals['trend_strength'] = 1.0
            
            # 3. Sentiment-Price Divergence (IMPROVED with weighting)
            if signals['sentiment_momentum'] != 0 and price_momentum != 0:
                # Normalize both to similar scale
                norm_sent = np.sign(signals['sentiment_momentum']) * min(abs(signals['sentiment_momentum']), 1.0)
                norm_price = np.sign(price_momentum) * min(abs(price_momentum), 0.1) * 10  # Scale to ~1
                
                # Divergence is positive when sentiment up but price down (or vice versa)
                divergence = norm_sent - norm_price
                
                # NEW: Weight divergence higher when price move is large
                if abs(price_momentum) > 0.03:
                    divergence *= 1.5  # Amplify divergence when price moved significantly
                
                signals['divergence'] = divergence
                logger.info(f"Sentiment-Price divergence: {divergence:.4f} (sent={norm_sent:.2f}, price={norm_price:.2f})")
            
            # 4. Extreme Sentiment Detection (contrarian signal)
            current_sentiment = sentiment_df[sent_col].iloc[-1] if len(sentiment_df) > 0 else 0
            
            # Calculate sentiment z-score (how extreme is current sentiment)
            if len(sentiment_df) >= 30:
                sent_mean = sentiment_df[sent_col].tail(30).mean()
                sent_std = sentiment_df[sent_col].tail(30).std()
                if sent_std > 0:
                    z_score = (current_sentiment - sent_mean) / sent_std
                    signals['extreme_sentiment'] = z_score
                    
                    if z_score < -1.5:
                        logger.info(f"CONTRARIAN SIGNAL: Extreme negative sentiment (z={z_score:.2f})")
                    elif z_score > 1.5:
                        logger.info(f"CONTRARIAN SIGNAL: Extreme positive sentiment (z={z_score:.2f})")
            
            # 5. Compute Reversal Probability and Adjustment Factor (IMPROVED)
            bullish_reversal_score = 0.0
            bearish_reversal_score = 0.0
            
            # --- BULLISH SIGNALS ---
            # A. Sentiment improving rapidly
            if signals['sentiment_momentum'] > 0.15:
                bullish_reversal_score += 0.4  # Increased from 0.3
            elif signals['sentiment_momentum'] > 0.1:
                bullish_reversal_score += 0.3
            elif signals['sentiment_momentum'] > 0.05:
                bullish_reversal_score += 0.15
            
            # B. Positive divergence (sentiment up, price down)
            if signals['divergence'] > 1.0:
                bullish_reversal_score += 0.4  # Strong divergence
            elif signals['divergence'] > 0.5:
                bullish_reversal_score += 0.3
            elif signals['divergence'] > 0.2:
                bullish_reversal_score += 0.15
            
            # C. Extreme negative sentiment (contrarian)
            if signals['extreme_sentiment'] < -2.0:
                bullish_reversal_score += 0.3  # Very extreme
            elif signals['extreme_sentiment'] < -1.5:
                bullish_reversal_score += 0.2
            elif signals['extreme_sentiment'] < -1.0:
                bullish_reversal_score += 0.1
            
            # D. NEW: Momentum persistence bonus
            if signals['momentum_persistence'] > 0:
                bullish_reversal_score += 0.15 * signals['momentum_persistence']
            
            # E. NEW: Strong downtrend + improving sentiment = reversal
            # BUT require stronger sentiment improvement to confirm
            if signals['trend_strength'] < -1 and signals['sentiment_momentum'] > 0.10:  # Increased threshold
                bullish_reversal_score += 0.25
                logger.info("REVERSAL PATTERN: Strong downtrend + improving sentiment")
            
            # --- BEARISH SIGNALS ---
            if signals['sentiment_momentum'] < -0.15:
                bearish_reversal_score += 0.4
            elif signals['sentiment_momentum'] < -0.1:
                bearish_reversal_score += 0.3
            elif signals['sentiment_momentum'] < -0.05:
                bearish_reversal_score += 0.15
            
            if signals['divergence'] < -1.0:
                bearish_reversal_score += 0.4
            elif signals['divergence'] < -0.5:
                bearish_reversal_score += 0.3
            elif signals['divergence'] < -0.2:
                bearish_reversal_score += 0.15
            
            if signals['extreme_sentiment'] > 2.0:
                bearish_reversal_score += 0.3
            elif signals['extreme_sentiment'] > 1.5:
                bearish_reversal_score += 0.2
            elif signals['extreme_sentiment'] > 1.0:
                bearish_reversal_score += 0.1
            
            if signals['momentum_persistence'] < 0:
                bearish_reversal_score += 0.15 * abs(signals['momentum_persistence'])
            
            if signals['trend_strength'] > 1 and signals['sentiment_momentum'] < -0.05:
                bearish_reversal_score += 0.25
                logger.info("REVERSAL PATTERN: Strong uptrend + declining sentiment")
            
            # Net reversal signal (-1 to +1, positive = bullish reversal expected)
            signals['reversal_probability'] = bullish_reversal_score - bearish_reversal_score
            
            # CONFIDENCE-BASED ADJUSTMENT: Only adjust when multiple signals confirm
            # Count how many indicators are triggering
            confirming_signals = 0
            if abs(signals['sentiment_momentum']) > 0.08:
                confirming_signals += 1
            if abs(signals['divergence']) > 0.4:
                confirming_signals += 1
            if abs(signals['extreme_sentiment']) > 1.2:
                confirming_signals += 1
            if abs(signals['momentum_persistence']) > 0.4:
                confirming_signals += 1
            if abs(signals['trend_strength']) > 0.5:
                confirming_signals += 1
            
            prob = signals['reversal_probability']
            
            # REQUIRE MULTIPLE CONFIRMATIONS for adjustment
            if confirming_signals >= 3 and abs(prob) > 0.5:
                # High confidence: 3+ signals confirming
                adj_per_unit = 0.006  # 0.6% per unit
                logger.info(f"HIGH CONFIDENCE: {confirming_signals} signals confirming")
            elif confirming_signals >= 2 and abs(prob) > 0.4:
                # Medium confidence: 2 signals confirming
                adj_per_unit = 0.004  # 0.4% per unit
                logger.info(f"MEDIUM CONFIDENCE: {confirming_signals} signals confirming")
            else:
                # Low confidence: don't adjust
                adj_per_unit = 0.0
                if abs(prob) > 0.2:
                    logger.info(f"LOW CONFIDENCE: Only {confirming_signals} signals, skipping adjustment")
            
            # Cap at ±1.0% daily adjustment
            signals['adjustment_factor'] = np.clip(prob * adj_per_unit, -0.010, 0.010)
            
            logger.info(f"Reversal signals: bull={bullish_reversal_score:.2f}, bear={bearish_reversal_score:.2f}, "
                       f"prob={prob:.2f}, adj={signals['adjustment_factor']*100:.3f}%")
            
        except Exception as e:
            logger.warning(f"Failed to compute sentiment signals: {e}")
        
        return signals
    
    def _apply_sentiment_adjustment(self, returns: np.ndarray, signals: dict) -> np.ndarray:
        """
        Apply sentiment-based adjustment to predicted returns.
        
        IMPROVED VERSION:
        - Stronger adjustments for high-confidence signals
        - Slower decay for persistent signals
        - Additional boost when multiple signals align
        """
        adjustment = signals.get('adjustment_factor', 0.0)
        
        if abs(adjustment) < 0.0001:
            return returns
        
        # Apply decaying adjustment over horizon
        # IMPROVED: Slower decay for persistent/strong signals
        adjusted_returns = returns.copy()
        
        # Determine decay rate based on signal strength
        prob = abs(signals.get('reversal_probability', 0))
        persistence = abs(signals.get('momentum_persistence', 0))
        
        if prob > 0.6 and persistence > 0.5:
            decay_rate = 0.05  # Very slow decay for high-confidence persistent signals
        elif prob > 0.4:
            decay_rate = 0.08  # Slower decay for strong signals
        else:
            decay_rate = 0.12  # Normal decay
        
        for i in range(len(adjusted_returns)):
            decay = np.exp(-decay_rate * i)  # Decay factor
            adjusted_returns[i] += adjustment * decay
        
        total_adj = sum(adjustment * np.exp(-decay_rate * i) for i in range(len(adjusted_returns)))
        logger.info(f"Applied sentiment adjustment: {adjustment*100:.3f}%/day (decay={decay_rate}, total={total_adj*100:.2f}%)")
        return adjusted_returns
    
    def _arima_forecast(self, df: pd.DataFrame, horizon: int) -> np.ndarray:
        """
        Generate ARIMA forecast for trend component.
        
        Note: In production, we simplify by fitting ARIMA on log returns directly
        since we don't have the full VMD decomposition available.
        """
        try:
            # Use log returns as proxy for trend
            returns = df['log_return'].values
            
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
            X_mid = prepare_mid_features(df, lookback)
            
            # Scale features
            X_mid_flat = X_mid.reshape(-1, X_mid.shape[-1])
            X_mid_scaled = self.artifacts.scaler_mid.transform(X_mid_flat)
            X_mid_scaled = X_mid_scaled.reshape(1, lookback, -1)
            
            # Convert to tensor
            X_tensor = torch.tensor(X_mid_scaled, dtype=torch.float32)
            X_tensor = X_tensor.to(self.artifacts.device)
            
            # Predict
            with torch.no_grad():
                forecast = self.artifacts.mid_gru(X_tensor).cpu().numpy().flatten()
            
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
            Xp, Xs = prepare_sentiment_features(df, lookback)
            
            # Scale features
            Xp_flat = Xp.reshape(-1, Xp.shape[-1])
            Xs_flat = Xs.reshape(-1, Xs.shape[-1])
            
            Xp_scaled = self.artifacts.scaler_price.transform(Xp_flat)
            Xs_scaled = self.artifacts.scaler_sent.transform(Xs_flat)
            
            Xp_scaled = Xp_scaled.reshape(1, lookback, -1)
            Xs_scaled = Xs_scaled.reshape(1, lookback, -1)
            
            # Convert to tensors
            Xp_tensor = torch.tensor(Xp_scaled, dtype=torch.float32).to(self.artifacts.device)
            Xs_tensor = torch.tensor(Xs_scaled, dtype=torch.float32).to(self.artifacts.device)
            
            # Predict
            with torch.no_grad():
                forecast = self.artifacts.sent_gru(Xp_tensor, Xs_tensor).cpu().numpy().flatten()
            
            logger.info("Sentiment-GRU forecast generated")
            return forecast
        
        except Exception as e:
            logger.error(f"Sentiment-GRU failed: {e}")
            raise
    
    def _xgb_hf_forecast(self, df: pd.DataFrame, horizon: int) -> np.ndarray:
        """Generate XGBoost high-frequency forecasts for each horizon."""
        try:
            # Prepare features (only last row)
            X_hf = prepare_hf_features(df)
            
            # Predict for each horizon
            forecasts = []
            for h in range(1, horizon + 1):
                pred = self.artifacts.xgb_hf_models[h].predict(X_hf)[0]
                forecasts.append(pred)
            
            logger.info("XGBoost HF forecasts generated")
            return np.array(forecasts)
        
        except Exception as e:
            logger.error(f"XGBoost HF failed: {e}")
            raise
    
    def _meta_ensemble(self, trend_fc: np.ndarray, mid_fc: np.ndarray,
                       sent_fc: np.ndarray, hf_fc: np.ndarray,
                       horizon: int) -> np.ndarray:
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
            X_meta = np.array([[
                trend_fc[h-1],
                mid_fc[h-1],
                sent_fc[h-1],
                hf_fc[h-1]
            ]])
            
            # Scale
            X_meta_scaled = self.artifacts.meta_scalers[h].transform(X_meta)
            
            # Predict
            pred = self.artifacts.meta_models[h].predict(X_meta_scaled)[0]
            ensemble_fc.append(pred)
        
        logger.info("Meta-ensemble combination complete")
        return np.array(ensemble_fc)
    
    def _returns_to_prices(self, returns: np.ndarray, 
                          last_price: float, 
                          last_date: pd.Timestamp,
                          horizon: int,
                          recent_momentum: float = 0.0,
                          recent_volatility: float = 0.0) -> List[Dict[str, Any]]:
        """
        Convert log returns to price forecast (with volatility-adjusted scaling).
        
        Args:
            returns: Predicted log returns from ensemble
            last_price: Last known price
            last_date: Last known date
            horizon: Forecast horizon
            recent_momentum: Recent price momentum (7-day return)
            recent_volatility: Recent volatility (7-day std of returns)
        """
        forecasts = []
        current_price = last_price
        current_date = last_date
        
        # Volatility scaling: REDUCED impact to prevent extreme predictions
        # The base model already has issues with scaler_mid, so we minimize amplification
        # Typical daily volatility for oil is ~1.5-2%
        BASELINE_VOL = 0.018  # Increased baseline to reduce multiplier
        vol_multiplier = max(1.0, recent_volatility / BASELINE_VOL) if recent_volatility > 0 else 1.0
        vol_multiplier = min(vol_multiplier, 1.3)  # Reduced cap from 2.0 to 1.3
        
        logger.info(f"Volatility multiplier: {vol_multiplier:.2f}x")
        
        for i in range(horizon):
            # Calculate next date (skipping weekends simple version)
            current_date += pd.Timedelta(days=1)
            while current_date.weekday() >= 5:  # Skip Sat/Sun
                current_date += pd.Timedelta(days=1)
                
            ret = returns[i]
            
            # Scale returns by volatility (reduced impact for stability)
            ret = ret * vol_multiplier
            
            # Apply graduated smoothing for extreme returns
            # First soft cap at 2%, then hard cap at 3%
            SOFT_CAP = 0.02  # 2% soft cap
            HARD_CAP = 0.03  # 3% hard cap (reduced from 5%)
            
            if abs(ret) > SOFT_CAP:
                sign = 1 if ret > 0 else -1
                excess = abs(ret) - SOFT_CAP
                # Dampen excess by 50%
                ret = sign * (SOFT_CAP + excess * 0.5)
            
            if abs(ret) > HARD_CAP:
                sign = 1 if ret > 0 else -1
                logger.warning(f"Capping prediction day {i+1}: {ret:.2%} -> {sign*HARD_CAP:.2%}")
                ret = sign * HARD_CAP
            
            # Convert log return to price: P_t = P_{t-1} * exp(r_t)
            next_price = current_price * np.exp(ret)
            
            forecasts.append({
                "horizon": i + 1,
                "date": current_date.strftime("%Y-%m-%d"),
                "forecasted_price": round(next_price, 2),
                "forecasted_return": round(float(ret), 4)
            })
            
            current_price = next_price
            
        return forecasts
    
    def _next_business_day(self, start_date: pd.Timestamp, days_ahead: int) -> pd.Timestamp:
        """Get the next business day N days ahead."""
        business_days = pd.bdate_range(start=start_date, periods=days_ahead + 1)
        return business_days[-1]


# Global singleton
prediction_service = PredictionService()
