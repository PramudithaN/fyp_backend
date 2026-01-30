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


def fetch_latest_prices(lookback_days: int = 90, end_date: datetime = None) -> pd.DataFrame:
    """
    Fetch the latest Brent oil prices from Yahoo Finance.
    
    Args:
        lookback_days: Number of calendar days to fetch (extra buffer for weekends/holidays)
                      Default is 90 to ensure we get at least 30 valid trading days
                      after feature engineering drops initial NaN rows.
        end_date: Optional end date for fetching prices (exclusive in yfinance, so we add 1 day if needed).
                  If None, uses current date.
    
    Returns:
        DataFrame with columns: date, price (at least 30 trading days)
    
    Raises:
        ValueError: If unable to fetch sufficient price data
    """
    logger.info(f"Fetching Brent oil prices (ticker: {BRENT_TICKER})")
    
    if end_date is None:
        end_date = datetime.now()
    
    start_date = end_date - timedelta(days=lookback_days)
    
    try:
        ticker = yf.Ticker(BRENT_TICKER)
        # yfinance end date is exclusive, so if we want to include end_date, we might need to adjust
        # However, usually 'history' fetches up to the end_date. 
        # If end_date is today, it gets today. 
        # Let's ensure consistency: if explicit end_date is passed (e.g. 2026-01-15), 
        # we usually want that to be the LAST day of data.
        # yfinance download(end=...) is exclusive. history(end=...) is also exclusive.
        # So if we want data FROM 2026-01-15, we should pass 2026-01-16 as end.

        # If end_date was passed explicitly, add 1 day to include it
        # If it's datetime.now(), it's fine.
        
        target_end = end_date
        if end_date.hour == 0 and end_date.minute == 0: # heuristic for "date only"
             target_end = end_date + timedelta(days=1)

        df = ticker.history(start=start_date, end=target_end)
        
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
        
        # Validate based on requested window (allowing for weekends/holidays)
        # We expect roughly 5 trading days for every 7 calendar days
        min_expected = max(1, lookback_days // 2) 
        if len(prices) < min_expected:
            raise ValueError(f"Insufficient data: got {len(prices)} days, need at least {min_expected} for a {lookback_days} day window")
        
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
