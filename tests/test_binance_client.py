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
