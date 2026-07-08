"""Crypto exchange price client via ccxt — mirrors data/yahoo/client.py's contract.

Public OHLCV endpoints only (no API key needed). The exchange id is
configurable (CRYPTO_EXCHANGE_ID env, default "binance") because binance.com
geo-restricts some regions — okx / kraken are drop-in fallbacks for public
market data through ccxt's unified interface.

DataFrames normalise to lowercase OHLCV columns on a naive DatetimeIndex, plus
a zero-filled `dividends` column for schema parity with the yahoo client (the
backtester carries that column; crypto simply has none).
"""
from __future__ import annotations

import logging
import os
import re
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

logger = logging.getLogger(__name__)

# Annualisation constants for a 24/7 market (365 trading days).
BARS_PER_YEAR = {
    "1m":  365 * 1440,
    "5m":  365 * 288,
    "15m": 365 * 96,
    "30m": 365 * 48,
    "1h":  365 * 24,
    "4h":  365 * 6,
    "1d":  365,
    "1wk": 52,
    "1mo": 12,
    # Legacy notation — map to nearest equivalent
    "H1":  365 * 24,
    "H4":  365 * 6,
    "D":   365,
    "W":   52,
}

# ccxt timeframe naming differs slightly from yfinance's
_TIMEFRAME_MAP = {
    "1d": "1d", "1wk": "1w", "1mo": "1M", "1h": "1h", "4h": "4h",
    "30m": "30m", "15m": "15m", "5m": "5m", "1m": "1m",
    "H1": "1h", "H4": "4h", "D": "1d", "W": "1w", "M": "1M",
}

_PAIR_RE = re.compile(r"\b[A-Z0-9]{2,10}/USDT\b")

_exchange = None  # lazy singleton — ccxt markets load is slow, do it once


def _get_exchange():
    global _exchange
    if _exchange is None:
        import ccxt
        exchange_id = os.getenv("CRYPTO_EXCHANGE_ID", "binance")
        cls = getattr(ccxt, exchange_id)
        _exchange = cls({"enableRateLimit": True})
    return _exchange


def extract_tickers(raw: str) -> list:
    """Extract crypto spot pairs from any string.

    Handles "BTC/USDT", "majors (BTC/USDT, ETH/USDT)", comma lists, etc.
    Falls back to [raw.strip()] when no pairs are found (same contract as the
    yahoo client — callers detect failure and fall through to a default).
    """
    pairs = _PAIR_RE.findall(raw or "")
    if pairs:
        seen: list = []
        for p in pairs:
            if p not in seen:
                seen.append(p)
        return seen
    return [(raw or "").strip()]


def get_historical_data(symbol: str, interval: str = "1d",
                        days: int = 730) -> pd.DataFrame:
    """Fetch OHLCV for one spot pair, paginating past the per-call limit.

    Returns lowercase OHLCV + zero `dividends`, naive DatetimeIndex named
    "time" — identical shape to the yahoo client so every downstream consumer
    (cache, feature engineering, backtester, DSL) works unchanged.
    """
    timeframe = _TIMEFRAME_MAP.get(interval, "1d")
    ex = _get_exchange()

    since = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    all_rows: list = []
    limit = 1000  # binance max per fetch_ohlcv call

    try:
        while True:
            batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
            if not batch:
                break
            all_rows.extend(batch)
            if len(batch) < limit:
                break
            # next page starts one bar after the last received bar
            since = batch[-1][0] + 1
            time.sleep(getattr(ex, "rateLimit", 200) / 1000.0)
    except Exception as e:
        logger.warning(f"binance client: fetch failed for {symbol} {timeframe}: {e}")
        return pd.DataFrame()

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows, columns=["time", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    df = df.drop_duplicates(subset="time").set_index("time").sort_index()
    df.index.name = "time"
    df["dividends"] = 0.0   # schema parity with the yahoo client
    return df
