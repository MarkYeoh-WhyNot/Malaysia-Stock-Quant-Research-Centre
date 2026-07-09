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

_exchange = None       # lazy singleton — ccxt markets load is slow, do it once
_perp_exchange = None  # separate handle: futures/swap market (funding, OI)


def _get_exchange():
    global _exchange
    if _exchange is None:
        import ccxt
        exchange_id = os.getenv("CRYPTO_EXCHANGE_ID", "binance")
        cls = getattr(ccxt, exchange_id)
        _exchange = cls({"enableRateLimit": True})
    return _exchange


def _get_perp_exchange():
    """Separate ccxt handle configured for USDT-margined perpetuals (defaultType
    'future'). Funding rate / open interest are futures-market concepts — the
    spot handle above never sees them. Public endpoints only, no API key."""
    global _perp_exchange
    if _perp_exchange is None:
        import ccxt
        exchange_id = os.getenv("CRYPTO_EXCHANGE_ID", "binance")
        cls = getattr(ccxt, exchange_id)
        _perp_exchange = cls({"enableRateLimit": True, "options": {"defaultType": "future"}})
    return _perp_exchange


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


def get_funding_rate_history(symbol: str, days: int = 1825) -> pd.DataFrame:
    """Historical perp funding settlements for one pair, paginated.

    Binance keeps the FULL funding history (unlike open interest, capped at
    ~30 days) — ~3 settlements/day → 5yr ≈ 5,475 rows ≈ 6 pages. History
    starts at each perp's listing date (ARB ~2023, OP ~2022, ...), so callers
    must handle series shorter than `days` honestly.

    Returns a DataFrame with one column `funding_rate` (per-8h decimal, e.g.
    0.0001 = 0.01%) on a naive UTC DatetimeIndex at settlement timestamps —
    same index conventions as OHLCV. Empty DataFrame on any failure (this
    client never raises).
    """
    ex = _get_perp_exchange()
    since = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    all_rows: list = []
    limit = 1000  # binance max per fundingRate call

    try:
        while True:
            batch = ex.fetch_funding_rate_history(symbol, since=since, limit=limit)
            if not batch:
                break
            all_rows.extend(batch)
            if len(batch) < limit:
                break
            since = batch[-1]["timestamp"] + 1
            time.sleep(getattr(ex, "rateLimit", 200) / 1000.0)
    except Exception as e:
        logger.warning(f"binance client: funding history fetch failed for {symbol}: {e}")
        return pd.DataFrame()

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        {"time": [r["timestamp"] for r in all_rows],
         "funding_rate": [float(r.get("fundingRate") or 0.0) for r in all_rows]})
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    df = df.drop_duplicates(subset="time").set_index("time").sort_index()
    df.index.name = "time"
    return df


def fetch_live_prices(symbols: list | None = None) -> dict:
    """Live spot quotes for a set of pairs in ONE bulk request.

    Display-only (never persisted, never fed to the pipeline) — powers the
    dashboard Live Prices panel. Resilient like the rest of this client: never
    raises. Returns:

        {"prices": [{symbol, last, change_pct_24h, high_24h, low_24h,
                     quote_volume_24h}, ...],
         "errors": [str, ...]}

    A pair the exchange doesn't return is skipped (not an error entry); a total
    fetch failure (e.g. geo-block) yields empty prices + one error string so the
    caller can show a message instead of a blank table.
    """
    from config.settings import DEFAULT_SYMBOLS
    symbols = list(symbols) if symbols else list(DEFAULT_SYMBOLS)
    ex = _get_exchange()

    try:
        # One call for everything; filter to our universe afterwards. fetch_tickers
        # with an explicit symbol list is supported by binance/okx/kraken and is
        # far cheaper than one request per pair.
        tickers = ex.fetch_tickers(symbols)
    except Exception as e:
        logger.warning(f"binance client: fetch_tickers failed: {e}")
        return {"prices": [], "errors": [f"live price fetch failed: {e}"]}

    prices: list = []
    errors: list = []
    for sym in symbols:
        t = tickers.get(sym)
        if not t or t.get("last") is None:
            continue   # pair not returned this cycle — skip, don't error
        prices.append({
            "symbol":            sym,
            "last":              t.get("last"),
            "change_pct_24h":    t.get("percentage"),
            "high_24h":          t.get("high"),
            "low_24h":           t.get("low"),
            "quote_volume_24h":  t.get("quoteVolume"),
        })
    return {"prices": prices, "errors": errors}


def get_funding_rate(symbol: str) -> dict | None:
    """Current funding rate for one perp (e.g. "BTC/USDT:USDT" or "BTC/USDT" —
    ccxt resolves the perp contract for the pair on most exchanges). Returns
    None on any failure (unsupported pair, geo-block, no perp market) — never
    raises; callers treat None as "skip this symbol", same as fetch_live_prices.
    """
    ex = _get_perp_exchange()
    try:
        fr = ex.fetch_funding_rate(symbol)
    except Exception as e:
        logger.debug(f"binance client: fetch_funding_rate failed for {symbol}: {e}")
        return None
    rate = fr.get("fundingRate")
    if rate is None:
        return None
    return {
        "symbol": symbol,
        "funding_rate": rate,
        "funding_rate_pct": rate * 100.0,
        "mark_price": fr.get("markPrice"),
        "index_price": fr.get("indexPrice"),
        "next_funding_time": fr.get("fundingDatetime"),
    }


def fetch_live_funding(symbols: list | None = None) -> dict:
    """Current funding rate for a set of perps in ONE bulk request — the
    funding-rate analog of fetch_live_prices(). Display-only, never persisted,
    never fed to the pipeline. Resilient: never raises.

    Returns {"funding": {symbol: {funding_rate_pct, next_funding_time}, ...},
             "errors": [str, ...]}.

    ccxt returns bulk funding keyed by the exchange's full contract symbol
    (e.g. "BTC/USDT:USDT"), not the plain pair — normalise back to the plain
    "BTC/USDT" symbol so callers can look up by the same key fetch_live_prices
    uses.
    """
    from config.settings import DEFAULT_SYMBOLS
    symbols = list(symbols) if symbols else list(DEFAULT_SYMBOLS)
    ex = _get_perp_exchange()

    try:
        rates = ex.fetch_funding_rates(symbols)
    except Exception as e:
        logger.warning(f"binance client: fetch_funding_rates failed: {e}")
        return {"funding": {}, "errors": [f"live funding fetch failed: {e}"]}

    funding: dict = {}
    for _, r in rates.items():
        plain_symbol = (r.get("symbol") or "").split(":")[0]
        rate = r.get("fundingRate")
        if not plain_symbol or rate is None:
            continue
        funding[plain_symbol] = {
            "funding_rate_pct": rate * 100.0,
            "next_funding_time": r.get("fundingDatetime"),
        }
    return {"funding": funding, "errors": []}


def get_open_interest(symbol: str) -> dict | None:
    """Current open interest (contracts + notional value) for one perp.
    Returns None on any failure — same resilient contract as get_funding_rate."""
    ex = _get_perp_exchange()
    try:
        oi = ex.fetch_open_interest(symbol)
    except Exception as e:
        logger.debug(f"binance client: fetch_open_interest failed for {symbol}: {e}")
        return None
    value = oi.get("openInterestAmount") or oi.get("openInterestValue") or oi.get("openInterest")
    if value is None:
        return None
    return {
        "symbol": symbol,
        "open_interest": value,
        "open_interest_value": oi.get("openInterestValue"),
        "timestamp": oi.get("datetime"),
    }
