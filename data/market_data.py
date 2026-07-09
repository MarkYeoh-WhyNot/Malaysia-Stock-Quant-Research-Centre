"""Market-data facade — dispatches to the active profile's data backend.

Consumers (DataEngineer, BacktestEngineer) import `get_historical_data`,
`extract_tickers`, and `BARS_PER_YEAR` from HERE instead of a concrete client.
The active market profile's DATA_BACKEND ("yahoo" for Bursa, "binance" for
crypto) decides which implementation serves them — same contract either way:
lowercase-OHLCV DataFrames on a naive DatetimeIndex.

`get_funding_rate_history` is perp-only: real implementation on the binance
backend, empty-DataFrame stub on yahoo (funding does not exist for equities —
Bursa code paths never call it, but the import contract stays uniform).
"""
from __future__ import annotations

import pandas as pd

from config.settings import DATA_BACKEND

if DATA_BACKEND == "binance":
    from data.binance.client import (          # noqa: F401
        get_historical_data, extract_tickers, BARS_PER_YEAR,
        get_funding_rate_history,
    )
else:
    from data.yahoo.client import (            # noqa: F401
        get_historical_data, extract_tickers, BARS_PER_YEAR,
    )

    def get_funding_rate_history(symbol: str, days: int = 1825) -> pd.DataFrame:
        """Equities have no funding settlements — uniform-contract stub."""
        return pd.DataFrame()
