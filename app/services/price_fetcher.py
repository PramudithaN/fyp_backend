"""
Price fetcher service - fetches Brent oil prices from Yahoo Finance.
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
import logging

from app.config import BRENT_TICKER

logger = logging.getLogger(__name__)


def fetch_latest_prices(lookback_days: int = 90) -> pd.DataFrame:
    """
    Fetch the latest Brent oil prices from Yahoo Finance.
    
    Args:
        lookback_days: Number of calendar days to fetch (extra buffer for weekends/holidays)
                      Default is 90 to ensure we get at least 30 valid trading days
                      after feature engineering drops initial NaN rows.
    
    Returns:
        DataFrame with columns: date, price (at least 30 trading days)
    
    Raises:
        ValueError: If unable to fetch sufficient price data
    """
    logger.info(f"Fetching Brent oil prices (ticker: {BRENT_TICKER})")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    
    try:
        ticker = yf.Ticker(BRENT_TICKER)
        df = ticker.history(start=start_date, end=end_date)
        
        if df.empty:
            raise ValueError(f"No data returned for ticker {BRENT_TICKER}")
        
        # Use 'Close' price
        prices = df[['Close']].reset_index()
        prices.columns = ['date', 'price']
        
        # Ensure date is just date (not datetime)
        prices['date'] = pd.to_datetime(prices['date']).dt.tz_localize(None)
        
        # Sort by date
        prices = prices.sort_values('date').reset_index(drop=True)
        
        logger.info(f"Fetched {len(prices)} days of price data")
        logger.info(f"Date range: {prices['date'].min()} to {prices['date'].max()}")
        logger.info(f"Latest price: ${prices['price'].iloc[-1]:.2f}")
        
        if len(prices) < 30:
            raise ValueError(f"Insufficient data: got {len(prices)} days, need at least 30")
        
        return prices
    
    except Exception as e:
        logger.error(f"Error fetching prices: {e}")
        raise ValueError(f"Failed to fetch Brent oil prices: {e}")


def get_last_n_trading_days(prices: pd.DataFrame, n: int = 30) -> pd.DataFrame:
    """
    Get the last N trading days from a price DataFrame.
    
    Args:
        prices: DataFrame with 'date' and 'price' columns
        n: Number of trading days to return
    
    Returns:
        DataFrame with last N trading days
    """
    return prices.tail(n).reset_index(drop=True)


def validate_price_data(prices: pd.DataFrame, min_days: int = 30) -> bool:
    """
    Validate that price data meets requirements.
    
    Args:
        prices: DataFrame with 'date' and 'price' columns
        min_days: Minimum number of trading days required
    
    Returns:
        True if valid, False otherwise
    """
    if prices is None or prices.empty:
        return False
    
    if len(prices) < min_days:
        return False
    
    if 'date' not in prices.columns or 'price' not in prices.columns:
        return False
    
    if prices['price'].isna().any():
        return False
    
    return True
