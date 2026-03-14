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


def fetch_latest_prices(
    lookback_days: int = 90, end_date: datetime = None
) -> pd.DataFrame:
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
        # Use UTC to avoid local DST gaps (e.g., 02:xx on DST start)
        end_date = datetime.now()

    start_date = end_date - timedelta(days=lookback_days)

    try:
        ticker = yf.Ticker(BRENT_TICKER)
        # yfinance history(end=...) is exclusive, so always pass end+1 day to
        # include today's (or the requested end date's) closing price.
        # Passing end=tomorrow also ensures yfinance never serves this request
        # from its internal LRU cache (cache_get is only triggered when
        # end_dt + 30min <= now, i.e. end is already in the past).
        start_arg = start_date.date()
        end_arg = end_date.date() + timedelta(days=1)

        df = ticker.history(start=start_arg, end=end_arg)

        if df.empty:
            raise ValueError(f"No data returned for ticker {BRENT_TICKER}")

        # Use 'Close' price
        prices = df[["Close"]].reset_index()
        prices.columns = ["date", "price"]

        # Ensure date is just date (not datetime)
        prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None)

        # Sort by date
        prices = prices.sort_values("date").reset_index(drop=True)

        logger.info(f"Fetched {len(prices)} days of price data")
        logger.info(f"Date range: {prices['date'].min()} to {prices['date'].max()}")
        logger.info(f"Latest price: ${prices['price'].iloc[-1]:.2f}")

        # Validate based on requested window (allowing for weekends/holidays)
        # We expect roughly 5 trading days for every 7 calendar days
        min_expected = max(1, lookback_days // 2)
        if len(prices) < min_expected:
            raise ValueError(
                f"Insufficient data: got {len(prices)} days, need at least {min_expected} for a {lookback_days} day window"
            )

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


def get_market_status() -> dict:
    """
    Determine whether the Brent oil market is currently open using the live
    ``marketState`` field returned by Yahoo Finance for the BZ=F ticker.

    Yahoo Finance ``marketState`` values:
        - ``"REGULAR"``  — regular trading session (market open)
        - ``"PRE"``      — pre-market session
        - ``"POST"``     — post-market / after-hours session
        - ``"CLOSED"``   — market closed (weekend, holiday, etc.)
        - ``"PREPRE"`` / ``"POSTPOST"`` — extended hours

    Falls back to a weekend/date check if the API call fails.

    Returns:
        dict with keys:
            - is_open (bool): True only when ``marketState == "REGULAR"``.
            - market_state (str): Raw value from Yahoo Finance (or "UNKNOWN").
            - message (str): Human-readable status string.
    """
    try:
        ticker = yf.Ticker(BRENT_TICKER)
        market_state = ticker.info.get("marketState", "UNKNOWN")
        is_open = market_state == "REGULAR"

        state_messages = {
            "REGULAR": "Market open",
            "PRE": "Market closed (pre-market session)",
            "POST": "Market closed (post-market session)",
            "PREPRE": "Market closed (extended pre-market)",
            "POSTPOST": "Market closed (extended post-market)",
            "CLOSED": "Market closed",
        }
        message = state_messages.get(market_state, f"Market closed (state: {market_state})")

        logger.info(f"Yahoo Finance marketState for {BRENT_TICKER}: {market_state}")
        return {"is_open": is_open, "market_state": market_state, "message": message}

    except Exception as e:
        logger.warning(f"Could not fetch market state from Yahoo Finance: {e}. Falling back to date check.")

        # Fallback: weekend check
        from datetime import date
        today = date.today()
        if today.weekday() == 5:  # Saturday
            return {"is_open": False, "market_state": "UNKNOWN", "message": "Market closed (Saturday)"}
        return {"is_open": True, "market_state": "UNKNOWN", "message": "Market likely open (fallback)"}


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

    if "date" not in prices.columns or "price" not in prices.columns:
        return False

    if prices["price"].isna().any():
        return False

    return True
