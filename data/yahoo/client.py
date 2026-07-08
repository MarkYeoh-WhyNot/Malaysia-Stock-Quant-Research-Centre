"""
Yahoo Finance price client — wraps yfinance for Bursa Malaysia (.KL) stocks.
All DataFrames normalise to lowercase OHLCV columns with a DatetimeIndex.
"""
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Union
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Annualisation constants for Bursa Malaysia
# Trading hours: 9:00–12:30, 14:30–17:00 (~6h/day), ~252 days/year
BARS_PER_YEAR = {
    "1m":  252 * 360,
    "5m":  252 * 72,
    "15m": 252 * 24,
    "30m": 252 * 12,
    "1h":  252 * 6,
    "1d":  252,
    "1wk": 52,
    "1mo": 12,
    # Legacy OANDA notation — map to nearest equivalent
    "H1":  252 * 6,
    "H4":  252 * 2,
    "D":   252,
    "W":   52,
}


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase columns, ensure DatetimeIndex, drop timezone.

    Keeps the `dividends` column: with auto_adjust=True the price series hides
    ex-date drops, so dividend-aware strategies (dividend capture is a promoted
    idea class) need the raw dividend cash amounts per bar.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "time"
    for col in ("stock splits", "capital gains"):
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    if "dividends" in df.columns:
        df["dividends"] = df["dividends"].fillna(0.0)
    return df


# ---------------------------------------------------------------------------
# Single-ticker helpers
# ---------------------------------------------------------------------------

def get_prices(
    symbol: str,
    period: str = "2y",
    interval: str = "1d",
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch OHLCV for one symbol.

    Args:
        symbol:   Yahoo Finance ticker, e.g. "1155.KL"
        period:   "1y", "2y", "5y" etc. (ignored if start/end given)
        interval: "1d", "1h", "1wk", "1mo"
        start/end: ISO date strings for exact range

    Returns:
        DataFrame with lowercase columns (open, high, low, close, volume).
    """
    try:
        ticker = yf.Ticker(symbol)
        if start and end:
            df = ticker.history(start=start, end=end, interval=interval, auto_adjust=True)
        else:
            df = ticker.history(period=period, interval=interval, auto_adjust=True)
        df = _normalise(df)
        if df.empty:
            logger.warning(f"No price data returned for {symbol}")
        else:
            logger.debug(f"Fetched {len(df)} bars for {symbol} ({interval})")
        return df
    except Exception as e:
        logger.error(f"yfinance error for {symbol}: {e}")
        return pd.DataFrame()


def get_info(symbol: str) -> dict:
    """Return yfinance .info dict for a symbol (fundamentals, metadata)."""
    try:
        info = yf.Ticker(symbol).info
        return {
            "symbol":        symbol,
            "name":          info.get("longName", info.get("shortName", "")),
            "sector":        info.get("sector", ""),
            "industry":      info.get("industry", ""),
            "market_cap":    info.get("marketCap"),
            "pe_trailing":   info.get("trailingPE"),
            "pe_forward":    info.get("forwardPE"),
            "pb_ratio":      info.get("priceToBook"),
            "ps_ratio":      info.get("priceToSalesTrailing12Months"),
            "eps_trailing":  info.get("trailingEps"),
            "dividend_yield": info.get("dividendYield"),
            "roe":           info.get("returnOnEquity"),
            "roa":           info.get("returnOnAssets"),
            "debt_equity":   info.get("debtToEquity"),
            "current_ratio": info.get("currentRatio"),
            "revenue_growth": info.get("revenueGrowth"),
            "earnings_growth": info.get("earningsGrowth"),
            "52w_high":      info.get("fiftyTwoWeekHigh"),
            "52w_low":       info.get("fiftyTwoWeekLow"),
            "beta":          info.get("beta"),
            "avg_volume":    info.get("averageVolume"),
            "currency":      info.get("currency", "MYR"),
            "exchange":      info.get("exchange", "KLS"),
        }
    except Exception as e:
        logger.error(f"get_info failed for {symbol}: {e}")
        return {"symbol": symbol, "error": str(e)}


def get_latest_price(symbol: str) -> dict:
    """Return the most recent close price and basic stats."""
    try:
        fast = yf.Ticker(symbol).fast_info
        return {
            "symbol":        symbol,
            "price":         getattr(fast, "last_price", None),
            "prev_close":    getattr(fast, "previous_close", None),
            "open":          getattr(fast, "open", None),
            "day_high":      getattr(fast, "day_high", None),
            "day_low":       getattr(fast, "day_low", None),
            "volume":        getattr(fast, "last_volume", None),
            "market_cap":    getattr(fast, "market_cap", None),
            "currency":      getattr(fast, "currency", "MYR"),
        }
    except Exception as e:
        logger.warning(f"get_latest_price failed for {symbol}: {e}")
        return {"symbol": symbol, "price": None, "error": str(e)}


# ---------------------------------------------------------------------------
# Multi-ticker helpers
# ---------------------------------------------------------------------------

def get_multi_prices(
    symbols: list,
    period: str = "2y",
    interval: str = "1d",
) -> dict[str, pd.DataFrame]:
    """
    Fetch OHLCV for a list of symbols.
    Returns dict: {symbol: DataFrame}.
    Uses yf.download for efficiency, falls back to per-ticker if needed.
    """
    if not symbols:
        return {}

    try:
        raw = yf.download(
            symbols,
            period=period,
            interval=interval,
            auto_adjust=True,
            group_by="ticker",
            progress=False,
            threads=True,
        )
    except Exception as e:
        logger.error(f"yf.download failed: {e}")
        raw = None

    result = {}

    if raw is not None and not raw.empty:
        if len(symbols) == 1:
            df = _normalise(raw)
            if not df.empty:
                result[symbols[0]] = df
        else:
            for sym in symbols:
                try:
                    df = _normalise(raw[sym].dropna(how="all"))
                    if not df.empty:
                        result[sym] = df
                except Exception:
                    pass

    # Fill any gaps with individual fetches
    missing = [s for s in symbols if s not in result]
    for sym in missing:
        df = get_prices(sym, period=period, interval=interval)
        if not df.empty:
            result[sym] = df
        time.sleep(0.15)

    logger.info(f"Fetched prices for {len(result)}/{len(symbols)} symbols")
    return result


def get_latest_prices(symbols: list) -> list:
    """Return list of latest price dicts for all symbols."""
    results = []
    for sym in symbols:
        p = get_latest_price(sym)
        results.append(p)
        time.sleep(0.05)
    return results


def get_multi_info(symbols: list) -> list:
    """Return fundamental info for a list of symbols."""
    results = []
    for sym in symbols:
        results.append(get_info(sym))
        time.sleep(0.1)
    return results


# ---------------------------------------------------------------------------
# Historical data with days parameter (OANDA-compatible interface)
# ---------------------------------------------------------------------------

def extract_tickers(raw: str) -> list:
    """Extract valid Bursa ticker codes from any string.

    Handles inputs like:
      - "5225.KL"                                       → ["5225.KL"]
      - "Healthcare sector (e.g., 5225.KL, 5878.KL)"   → ["5225.KL", "5878.KL"]
      - "IHH Healthcare 5225.KL"                        → ["5225.KL"]
      - "5225.KL,5878.KL,7081.KL"                       → ["5225.KL", "5878.KL", "7081.KL"]

    Falls back to [raw.strip()] when no .KL codes are found (preserves caller's
    ability to detect failure and fall through to a default).
    """
    import re as _re
    tickers = _re.findall(r'\b\d{4,5}\.KL\b', raw)
    if tickers:
        # Deduplicate while preserving order
        seen: list = []
        for t in tickers:
            if t not in seen:
                seen.append(t)
        return seen
    return [raw.strip()]


def get_historical_data(symbol: str, interval: str = "1d", days: int = 730) -> pd.DataFrame:
    """
    Fetch historical OHLCV matching the OANDA client interface signature.
    interval: "1d", "1h", "1wk" or legacy OANDA "H1", "H4", "D"
    """
    # Map legacy OANDA notation
    _map = {"H1": "1h", "H4": "4h", "D": "1d", "W": "1wk", "M": "1mo"}
    interval = _map.get(interval, interval)

    # yfinance period strings
    if days <= 7:       period = "7d"
    elif days <= 30:    period = "1mo"
    elif days <= 60:    period = "2mo"
    elif days <= 90:    period = "3mo"
    elif days <= 180:   period = "6mo"
    elif days <= 365:   period = "1y"
    elif days <= 730:   period = "2y"
    elif days <= 1095:  period = "3y"
    elif days <= 1825:  period = "5y"
    else:               period = "max"

    # For intraday, yfinance limits history to 60 days (1h) or 730 days (1h with extended)
    if interval in ("1h", "4h") and days > 730:
        period = "2y"

    return get_prices(symbol, period=period, interval=interval)
