"""Benchmark quote and Brent-derived forecast helpers.

This service keeps Brent as the only model-backed forecast and provides
front-end friendly benchmark views for WTI/OPEC/Dubai with explicit
quality labels.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
import yfinance as yf

from app.config import BRENT_TICKER
from app.services.prediction_snapshot import get_required_locked_prediction_snapshot

logger = logging.getLogger(__name__)

_BENCHMARK_LABELS: dict[str, str] = {
    "brent": "Brent Crude",
    "wti": "WTI Crude",
    "opec": "OPEC Basket",
    "dubai": "Dubai Fateh",
}

_BENCHMARK_TICKERS: dict[str, Optional[str]] = {
    "brent": BRENT_TICKER,
    "wti": "CL=F",
    "opec": os.getenv("OPEC_BASKET_TICKER", "" ).strip() or None,
    "dubai": os.getenv("DUBAI_FATEH_TICKER", "" ).strip() or None,
}

# Fallback calibration values used only when no direct/paired market data is available.
_DEFAULT_SPREADS: dict[str, float] = {
    "wti": float(os.getenv("WTI_DERIVED_SPREAD_USD", "-4.0")),
    "opec": float(os.getenv("OPEC_DERIVED_SPREAD_USD", "-1.4")),
    "dubai": float(os.getenv("DUBAI_DERIVED_SPREAD_USD", "-2.1")),
}

_DEFAULT_RATIOS: dict[str, float] = {
    "wti": float(os.getenv("WTI_DERIVED_RATIO", "0.95")),
    "opec": float(os.getenv("OPEC_DERIVED_RATIO", "0.982")),
    "dubai": float(os.getenv("DUBAI_DERIVED_RATIO", "0.973")),
}


def _safe_iso_date(value: Any) -> str:
    ts = pd.to_datetime(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.strftime("%Y-%m-%d")


def _fetch_latest_close(ticker: str) -> Optional[dict[str, Any]]:
    """Fetch latest daily close for a ticker."""
    try:
        df = yf.Ticker(ticker).history(period="10d", interval="1d")
        if df is None or df.empty or "Close" not in df.columns:
            return None

        close_series = df["Close"].dropna()
        if close_series.empty:
            return None

        last_idx = close_series.index[-1]
        return {
            "price": float(close_series.iloc[-1]),
            "as_of": _safe_iso_date(last_idx),
        }
    except Exception as exc:
        logger.warning("Failed to fetch latest close for %s: %s", ticker, exc)
        return None


def _fetch_pair_stats(
    base_ticker: str,
    target_ticker: str,
    lookback_days: int,
) -> Optional[dict[str, float]]:
    """Compute median spread/ratio from overlapping recent closes."""
    try:
        base_df = yf.Ticker(base_ticker).history(
            period=f"{max(lookback_days, 30)}d", interval="1d"
        )
        target_df = yf.Ticker(target_ticker).history(
            period=f"{max(lookback_days, 30)}d", interval="1d"
        )

        if (
            base_df is None
            or target_df is None
            or base_df.empty
            or target_df.empty
            or "Close" not in base_df.columns
            or "Close" not in target_df.columns
        ):
            return None

        base_close = base_df[["Close"]].rename(columns={"Close": "base_close"})
        target_close = target_df[["Close"]].rename(columns={"Close": "target_close"})
        joined = base_close.join(target_close, how="inner").dropna()

        if len(joined) < 10:
            return None

        spread = (joined["target_close"] - joined["base_close"]).median()
        ratio = (joined["target_close"] / joined["base_close"]).median()

        if not pd.notna(spread) or not pd.notna(ratio):
            return None

        return {
            "spread": float(spread),
            "ratio": float(ratio),
            "sample_days": int(len(joined)),
        }
    except Exception as exc:
        logger.warning(
            "Failed to compute pair stats %s -> %s: %s",
            base_ticker,
            target_ticker,
            exc,
        )
        return None


def _resolve_transform_params(target: str, lookback_days: int) -> dict[str, Any]:
    """Resolve spread/ratio transformation parameters for a benchmark target."""
    if target == "brent":
        return {
            "spread": 0.0,
            "ratio": 1.0,
            "sample_days": 0,
            "source": "identity",
            "fallback_used": False,
        }

    if target == "wti":
        stats = _fetch_pair_stats(BRENT_TICKER, "CL=F", lookback_days)
        if stats is not None:
            return {
                **stats,
                "source": "yahoo_pair_history",
                "fallback_used": False,
            }

    return {
        "spread": float(_DEFAULT_SPREADS[target]),
        "ratio": float(_DEFAULT_RATIOS[target]),
        "sample_days": 0,
        "source": "configured_fallback",
        "fallback_used": True,
    }


def get_benchmark_quotes(lookback_days: int = 60) -> dict[str, Any]:
    """Return benchmark quote cards for frontend display.

    Brent/WTI use direct Yahoo closes where possible.
    OPEC/Dubai are derived from Brent unless explicit Yahoo tickers are configured.
    """
    lookback_days = max(30, min(int(lookback_days), 365))

    brent_quote = _fetch_latest_close(BRENT_TICKER)
    if brent_quote is None:
        raise ValueError("Unable to fetch Brent quote from Yahoo Finance")

    quotes: list[dict[str, Any]] = []

    for benchmark in ("brent", "wti", "opec", "dubai"):
        label = _BENCHMARK_LABELS[benchmark]
        ticker = _BENCHMARK_TICKERS.get(benchmark)

        if benchmark == "brent":
            quotes.append(
                {
                    "benchmark": benchmark,
                    "display_name": label,
                    "ticker": BRENT_TICKER,
                    "price": round(float(brent_quote["price"]), 2),
                    "as_of": brent_quote["as_of"],
                    "quote_type": "direct",
                    "source": "yahoo_finance",
                    "quality": "model_target",
                    "status": "ok",
                    "note": None,
                }
            )
            continue

        direct_quote = _fetch_latest_close(ticker) if ticker else None
        if direct_quote is not None:
            quotes.append(
                {
                    "benchmark": benchmark,
                    "display_name": label,
                    "ticker": ticker,
                    "price": round(float(direct_quote["price"]), 2),
                    "as_of": direct_quote["as_of"],
                    "quote_type": "direct",
                    "source": "yahoo_finance",
                    "quality": "observed",
                    "status": "ok",
                    "note": None,
                }
            )
            continue

        params = _resolve_transform_params(benchmark, lookback_days)
        derived_price = (
            float(brent_quote["price"]) + float(params["spread"])
            if benchmark in {"opec", "dubai"} or params["source"] != "configured_fallback"
            else float(brent_quote["price"]) * float(params["ratio"])
        )

        quotes.append(
            {
                "benchmark": benchmark,
                "display_name": label,
                "ticker": ticker,
                "price": round(float(derived_price), 2),
                "as_of": brent_quote["as_of"],
                "quote_type": "derived",
                "source": f"brent_transform:{params['source']}",
                "quality": "indicative",
                "status": "estimated",
                "note": "Estimated from Brent quote (not a separately trained model target)",
            }
        )

    return {
        "success": True,
        "currency": "USD",
        "unit": "bbl",
        "base_benchmark": "brent",
        "generated_at": datetime.utcnow().isoformat(),
        "quotes": quotes,
    }


def get_derived_forecast(
    target: str,
    method: str = "spread",
    lookback_days: int = 60,
) -> dict[str, Any]:
    """Return benchmark forecast by transforming the locked Brent forecast."""
    target = str(target).lower().strip()
    method = str(method).lower().strip()
    if target not in _BENCHMARK_LABELS:
        raise ValueError("target must be one of: brent, wti, opec, dubai")
    if method not in {"spread", "ratio"}:
        raise ValueError("method must be one of: spread, ratio")

    snapshot = get_required_locked_prediction_snapshot()
    forecasts = snapshot.get("forecasts") or []
    if not forecasts:
        raise ValueError("Locked Brent forecast is empty")

    params = _resolve_transform_params(target, lookback_days=max(30, min(lookback_days, 365)))
    spread = float(params["spread"])
    ratio = float(params["ratio"])

    transformed: list[dict[str, Any]] = []
    for row in forecasts:
        brent_price = float(row.get("forecasted_price", 0.0))
        brent_lb = row.get("lower_bound")
        brent_ub = row.get("upper_bound")

        if target == "brent":
            est_price = brent_price
            est_lb = float(brent_lb) if brent_lb is not None else None
            est_ub = float(brent_ub) if brent_ub is not None else None
        elif method == "spread":
            est_price = brent_price + spread
            est_lb = (float(brent_lb) + spread) if brent_lb is not None else None
            est_ub = (float(brent_ub) + spread) if brent_ub is not None else None
        else:
            est_price = brent_price * ratio
            est_lb = (float(brent_lb) * ratio) if brent_lb is not None else None
            est_ub = (float(brent_ub) * ratio) if brent_ub is not None else None

        transformed.append(
            {
                "date": str(row.get("date", "")),
                "horizon": int(row.get("horizon", 0)),
                "benchmark": target,
                "benchmark_label": _BENCHMARK_LABELS[target],
                "forecast_type": "model" if target == "brent" else "derived_from_brent",
                "brent_forecasted_price": round(brent_price, 2),
                "forecasted_price": round(float(est_price), 2),
                "lower_bound": round(float(est_lb), 2) if est_lb is not None else None,
                "upper_bound": round(float(est_ub), 2) if est_ub is not None else None,
                "forecasted_return": row.get("forecasted_return"),
            }
        )

    return {
        "success": True,
        "target": target,
        "target_label": _BENCHMARK_LABELS[target],
        "method": method,
        "currency": "USD",
        "unit": "bbl",
        "quality": "model" if target == "brent" else "indicative",
        "disclaimer": (
            "Brent is model-backed. Non-Brent forecasts are transformed estimates, "
            "not separately trained model outputs."
        ),
        "transform": {
            "spread": spread,
            "ratio": ratio,
            "lookback_days": int(max(30, min(lookback_days, 365))),
            "sample_days": int(params.get("sample_days", 0)),
            "source": str(params.get("source", "unknown")),
            "fallback_used": bool(params.get("fallback_used", False)),
        },
        "prediction_date": snapshot.get("prediction_date"),
        "based_on_price_date": snapshot.get("based_on_price_date"),
        "generated_at": snapshot.get("generated_at"),
        "forecasts": transformed,
    }
