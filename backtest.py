
import sys
import os
import joblib
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.price_fetcher import fetch_latest_prices
from app.services.feature_engineering import engineer_all_features, get_mid_freq_features
from app.models.model_loader import model_artifacts
from app.config import MODEL_ARTIFACTS_DIR

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_backtest(target_date_str: str, override_price: float = None):
    """
    Run a backtest with robust scaler checking.
    If the scaler is mismatched (unfitted), fit it on history.
    """
    logger.info("Initializing backtest...")
    model_artifacts.load_all()
    
    # CHECK SCALER STATUS
    mid_features = get_mid_freq_features()
    expected_features = len(mid_features)
    
    # Check if scaler is identity (unfitted due to mismatch)
    # The loader sets mean_=0, scale_=1 for mismatch fallback
    is_mismatched = False
    if model_artifacts.scaler_mid.n_features_in_ != expected_features:
        # This shouldn't happen if loader did its fallback, because fallback sets n_features_in_ correctly
        # But let's check the values
        pass
    
    # Heuristic: if scale is all 1s and mean is all 0s, it's likely the fallback identity scaler
    if np.allclose(model_artifacts.scaler_mid.mean_, 0) and np.allclose(model_artifacts.scaler_mid.scale_, 1):
        logger.warning("DETECTED IDENTITY SCALER! The scaler_mid.pkl is missing or mismatched.")
        is_mismatched = True
        
    if is_mismatched:
        logger.info(">>> ATTEMPTING AUTO-FIX: Fitting scaler on historical data... <<<")
        target_date = datetime.strptime(target_date_str, "%Y-%m-%d")
        
        # Fetch long history for statistics
        history_days = 730
        prices = fetch_latest_prices(lookback_days=history_days, end_date=target_date)
        
        # Trim to target
        if prices['date'].iloc[-1] > target_date:
            prices = prices[prices['date'] <= target_date]
            
        # Feature Engineering
        df = engineer_all_features(prices, sentiment_df=None)
        df = df.dropna()
        
        X_mid = df[mid_features].values
        
        # Fit new scaler
        new_scaler = StandardScaler()
        new_scaler.fit(X_mid)
        
        # Overwrite artifact
        model_artifacts.scaler_mid = new_scaler
        logger.info(f"Scaler fixed using {len(df)} historical days.")
        
    # Proceed with standard prediction
    target_date = datetime.strptime(target_date_str, "%Y-%m-%d")
    prices = fetch_latest_prices(lookback_days=120, end_date=target_date)
    
    if prices['date'].iloc[-1] > target_date:
        prices = prices[prices['date'] <= target_date]
        
    if override_price is not None:
        logger.info(f"Overriding price: {prices['price'].iloc[-1]:.2f} -> {override_price}")
        prices.iloc[-1, prices.columns.get_loc('price')] = override_price
        
    logger.info(f"Running prediction from {prices['date'].iloc[-1].date()}")
    
    from app.services.prediction import prediction_service
    try:
        forecasts = prediction_service.predict(prices=prices, days_of_history=30)
        
        print("\n--- Forecast Results ---")
        print(f"{'Date':<15} | {'Forecast':<10} | {'Return':<10}")
        print("-" * 50)
        for f in forecasts:
            print(f"{f['date']:<15} | ${f['forecasted_price']:<9.2f} | {f['forecasted_return']:+.4f}")
            
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    target = "2026-01-15"
    price = None
    
    if len(sys.argv) > 1:
        target = sys.argv[1]
    if len(sys.argv) > 2:
        try:
            price = float(sys.argv[2])
        except:
            pass
            
    run_backtest(target, price)
