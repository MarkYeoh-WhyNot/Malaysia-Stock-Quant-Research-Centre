"""Binance/ccxt client contract tests (dual-market Phase B). No network —
ccxt is mocked; the client must produce yahoo-client-shaped DataFrames."""
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd

import data.binance.client as bc


def _fake_ohlcv(n=30, start_ms=None):
    start_ms = start_ms or int(datetime(2026, 5, 1, tzinfo=timezone.utc).timestamp() * 1000)
    day = 86_400_000
    return [
        [start_ms + i * day, 100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i, 1_000_000 + i]
        for i in range(n)
    ]


def _mock_exchange(batches):
    ex = MagicMock()
    ex.rateLimit = 0
    ex.fetch_ohlcv = MagicMock(side_effect=batches)
    return ex


def test_dataframe_matches_yahoo_contract():
    ex = _mock_exchange([_fake_ohlcv(30)])
    with patch.object(bc, "_get_exchange", return_value=ex):
        df = bc.get_historical_data("BTC/USDT", "1d", days=60)
    assert list(df.columns) == ["open", "high", "low", "close", "volume", "dividends"]
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.tz is None
    assert df.index.name == "time"
    assert len(df) == 30
    assert (df["dividends"] == 0.0).all()
    assert df["close"].iloc[0] == 100.5


def test_pagination_stitches_batches():
    first = _fake_ohlcv(1000)
    next_start = first[-1][0] + 86_400_000
    second = _fake_ohlcv(50, start_ms=next_start)
    ex = _mock_exchange([first, second])
    with patch.object(bc, "_get_exchange", return_value=ex):
        df = bc.get_historical_data("ETH/USDT", "1d", days=1825)
    assert len(df) == 1050
    assert ex.fetch_ohlcv.call_count == 2
    assert df.index.is_monotonic_increasing


def test_fetch_failure_returns_empty_frame():
    ex = MagicMock()
    ex.rateLimit = 0
    ex.fetch_ohlcv = MagicMock(side_effect=RuntimeError("geo blocked"))
    with patch.object(bc, "_get_exchange", return_value=ex):
        df = bc.get_historical_data("BTC/USDT", "1d")
    assert df.empty


def test_timeframe_mapping_used():
    ex = _mock_exchange([_fake_ohlcv(5)])
    with patch.object(bc, "_get_exchange", return_value=ex):
        bc.get_historical_data("BTC/USDT", "1wk", days=30)
    assert ex.fetch_ohlcv.call_args.kwargs["timeframe"] == "1w"


def test_extract_tickers_crypto():
    assert bc.extract_tickers("BTC/USDT") == ["BTC/USDT"]
    assert bc.extract_tickers("majors (BTC/USDT, ETH/USDT) rotation") == \
        ["BTC/USDT", "ETH/USDT"]
    assert bc.extract_tickers("BTC/USDT,BTC/USDT,ETH/USDT") == ["BTC/USDT", "ETH/USDT"]
    # fallback contract identical to the yahoo client
    assert bc.extract_tickers("no pairs here") == ["no pairs here"]


def test_bars_per_year_is_365_daily():
    assert bc.BARS_PER_YEAR["1d"] == 365
    assert bc.BARS_PER_YEAR["1wk"] == 52


# ── Live prices (dashboard Live Prices panel) ────────────────────────────────

_FAKE_TICKERS = {
    "BTC/USDT": {"last": 62000.0, "percentage": -1.5, "high": 63000.0,
                 "low": 61000.0, "quoteVolume": 1.2e9},
    "ETH/USDT": {"last": 3400.0, "percentage": 2.1, "high": 3500.0,
                 "low": 3300.0, "quoteVolume": 8.0e8},
}


def test_fetch_live_prices_maps_fields_and_filters():
    ex = MagicMock()
    ex.fetch_tickers = MagicMock(return_value=_FAKE_TICKERS)
    with patch.object(bc, "_get_exchange", return_value=ex):
        # request a third pair the exchange didn't return → silently skipped
        r = bc.fetch_live_prices(["BTC/USDT", "ETH/USDT", "ZZZ/USDT"])
    assert r["errors"] == []
    assert [p["symbol"] for p in r["prices"]] == ["BTC/USDT", "ETH/USDT"]
    btc = r["prices"][0]
    assert btc["last"] == 62000.0
    assert btc["change_pct_24h"] == -1.5
    assert btc["high_24h"] == 63000.0
    assert btc["low_24h"] == 61000.0
    assert btc["quote_volume_24h"] == 1.2e9


def test_fetch_live_prices_skips_symbols_missing_last():
    ex = MagicMock()
    ex.fetch_tickers = MagicMock(return_value={
        "BTC/USDT": {"last": 62000.0, "percentage": 1.0},
        "ETH/USDT": {"last": None},   # present but no price → skip, not error
    })
    with patch.object(bc, "_get_exchange", return_value=ex):
        r = bc.fetch_live_prices(["BTC/USDT", "ETH/USDT"])
    assert [p["symbol"] for p in r["prices"]] == ["BTC/USDT"]
    assert r["errors"] == []


def test_fetch_live_prices_total_failure_is_graceful():
    ex = MagicMock()
    ex.fetch_tickers = MagicMock(side_effect=RuntimeError("geo blocked"))
    with patch.object(bc, "_get_exchange", return_value=ex):
        r = bc.fetch_live_prices(["BTC/USDT"])
    assert r["prices"] == []
    assert len(r["errors"]) == 1 and "geo blocked" in r["errors"][0]


def test_fetch_live_prices_defaults_to_active_universe():
    ex = MagicMock()
    ex.fetch_tickers = MagicMock(return_value=_FAKE_TICKERS)
    with patch.object(bc, "_get_exchange", return_value=ex):
        bc.fetch_live_prices()   # no symbols → uses settings.DEFAULT_SYMBOLS
    # in default (bursa) test process DEFAULT_SYMBOLS is the KLCI list; the point
    # is simply that a symbol list was passed to fetch_tickers, not None
    assert ex.fetch_tickers.call_args.args[0]  # non-empty list
