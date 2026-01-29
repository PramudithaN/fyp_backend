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
    
    def predict(self, prices: pd.DataFrame, 
                sentiment_df: pd.DataFrame = None) -> List[Dict[str, Any]]:
        """
        Generate 14-day price forecast from price data.
        
        Args:
            prices: DataFrame with 'date' and 'price' columns (at least 30 days)
            sentiment_df: Optional DataFrame with sentiment data.
                         If None, fetches from database automatically.
        
        Returns:
            List of forecast dictionaries with date, price, return, horizon
        """
        logger.info("Starting prediction pipeline...")
        
        # Ensure we have enough data
        lookback = self.artifacts.lookback
        horizon = self.artifacts.horizon
        
        if len(prices) < lookback:
            raise ValueError(f"Need at least {lookback} days of data, got {len(prices)}")
        
        # Step 0: Fetch sentiment from database if not provided
        if sentiment_df is None:
            logger.info("Fetching sentiment from database...")
            sentiment_df = get_sentiment_history(days=60)
            if sentiment_df.empty:
                logger.warning("No sentiment data in database, using zeros")
            else:
                logger.info(f"Loaded {len(sentiment_df)} sentiment records")
        
        # Step 1: Feature engineering
        logger.info("Step 1: Feature engineering")
        df = engineer_all_features(prices, sentiment_df=sentiment_df)
        
        # Drop NaN rows from feature computation (first few rows)
        df = df.dropna()
        
        if len(df) < lookback:
            raise ValueError(f"After feature engineering, only {len(df)} valid rows. Need {lookback}.")
        
        # Step 2: Get component forecasts
        logger.info("Step 2: Generating component forecasts")
        
        # ARIMA forecast (simplified for production)
        trend_fc = self._arima_forecast(df, horizon)
        
        # Mid-frequency GRU forecast
        mid_fc = self._mid_gru_forecast(df, lookback)
        
        # Sentiment GRU forecast (with zero sentiment in Phase 1)
        sent_fc = self._sent_gru_forecast(df, lookback)
        
        # High-frequency XGBoost forecast
        hf_fc = self._xgb_hf_forecast(df, horizon)
        
        # Step 3: Meta-ensemble
        logger.info("Step 3: Meta-ensemble combination")
        ensemble_returns = self._meta_ensemble(trend_fc, mid_fc, sent_fc, hf_fc, horizon)
        
        # Step 4: Convert to prices
        logger.info("Step 4: Converting returns to prices")
        last_price = prices['price'].iloc[-1]
        last_date = pd.to_datetime(prices['date'].iloc[-1])
        
        forecasts = self._returns_to_prices(
            ensemble_returns, last_price, last_date, horizon
        )
        
        logger.info("Prediction pipeline complete!")
        return forecasts
    
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
    
    def _returns_to_prices(self, returns: np.ndarray, last_price: float,
                           last_date: pd.Timestamp, horizon: int) -> List[Dict[str, Any]]:
        """
        Convert log returns to price forecasts.
        
        price[t+k] = last_price * exp(sum(returns[0:k]))
        """
        forecasts = []
        cumulative_return = 0.0
        
        for h in range(horizon):
            cumulative_return += returns[h]
            forecast_price = last_price * np.exp(cumulative_return)
            
            # Calculate next business day
            forecast_date = self._next_business_day(last_date, h + 1)
            
            forecasts.append({
                'date': forecast_date.strftime('%Y-%m-%d'),
                'forecasted_price': round(float(forecast_price), 2),
                'forecasted_return': round(float(returns[h]), 6),
                'horizon': h + 1
            })
        
        return forecasts
    
    def _next_business_day(self, start_date: pd.Timestamp, days_ahead: int) -> pd.Timestamp:
        """Get the next business day N days ahead."""
        business_days = pd.bdate_range(start=start_date, periods=days_ahead + 1)
        return business_days[-1]


# Global singleton
prediction_service = PredictionService()
