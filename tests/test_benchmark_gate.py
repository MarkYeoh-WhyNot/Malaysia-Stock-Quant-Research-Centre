"""Phase 3.2 benchmark gate — equal-weight KLCI baseline helper.

The gate itself lives inside BacktestEngineer.backtest_idea (needs full price
data), but its baseline input — the equal-weight KLCI return series — is a pure,
network-free helper we can pin down here.
"""
import numpy as np
import pandas as pd

from agents.backtest_engineer.backtest_engineer import BacktestEngineer
from config.settings import DEFAULT_SYMBOLS, GATE_CONFIG


def _frame(base, n=300):
    idx = pd.date_range("2021-01-01", periods=n, freq="B")
    return pd.DataFrame({"close": np.linspace(base, base * 1.2, n)}, index=idx)


def test_equal_weight_is_cross_sectional_mean_and_memoised():
    be = BacktestEngineer()
    calls = {"n": 0}

    def fake_fetch(symbol, interval="1d", days=1825):
        calls["n"] += 1
        # each stock rises at a different rate so the mean is a genuine blend
        return _frame(10.0 + hash(symbol) % 5)

    be._fetch_prices = fake_fetch

    ew = be._equal_weight_klci_returns("1d")
    assert not ew.empty
    # one fetch per constituent on first call
    assert calls["n"] == len(DEFAULT_SYMBOLS)

    # second call is served from the instance cache — no additional fetches
    ew2 = be._equal_weight_klci_returns("1d")
    assert calls["n"] == len(DEFAULT_SYMBOLS)
    pd.testing.assert_series_equal(ew, ew2)

    # equal-weight daily return equals the cross-sectional mean of constituent
    # daily returns on any given day
    rets = [fake_fetch(s)["close"].pct_change() for s in DEFAULT_SYMBOLS]
    # reset the counter's side effects don't matter; compare values
    expected = pd.concat(rets, axis=1).mean(axis=1).dropna()
    pd.testing.assert_series_equal(
        ew.reindex(expected.index).dropna(), expected, check_names=False
    )


def test_benchmark_gate_config_defaults():
    # gate is on by default and requires beating the baseline (>= 0 excess)
    assert GATE_CONFIG.benchmark_gate_enabled is True
    assert GATE_CONFIG.benchmark_min_excess_ann == 0.0
