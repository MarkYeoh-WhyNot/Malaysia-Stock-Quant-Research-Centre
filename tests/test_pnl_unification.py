"""PnL fidelity: the single-source net-return engine, CAGR, drawdown quality,
reconstructed trade blotter, and the report-only fill/capacity variants.

Guards the keystone invariant — everything that needs "what did this strategy
earn each bar" goes through _net_return_series — plus that the reconstructed
trade blotter reconciles exactly to that series.
"""
import logging

import numpy as np
import pandas as pd

from agents.backtest_engineer.backtest_engineer import BacktestEngineer
from data.market_data import BARS_PER_YEAR

_ANN = BARS_PER_YEAR.get("1d", 252)   # 252 Bursa / 365 crypto


def _engine():
    be = BacktestEngineer.__new__(BacktestEngineer)
    be.logger = logging.getLogger("test")
    be.name = "BacktestEngineer"
    return be


def _synth(n=400, seed=1):
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="D")
    close = pd.Series(100 * np.cumprod(1 + rng.randn(n) * 0.01), index=idx)
    return pd.DataFrame({
        "close": close,
        "open": close.shift(1).fillna(close.iloc[0]),
        "volume": rng.randint(1_000_000, 3_000_000, n).astype(float),
    })


def _trend_signal(df):
    return (df["close"] > df["close"].rolling(20).mean()).astype(float).fillna(0)


def test_net_series_indexed_and_bar0_zero():
    be, df = _engine(), _synth()
    r = be._net_return_series(df, _trend_signal(df), "1d")
    assert list(r["net"].index) == list(df.index)          # date-aligned
    assert float(r["net"].iloc[0]) == 0.0                  # no position at start
    assert float(r["signal_shifted"].iloc[0]) == 0.0


def test_reconstructed_trades_reconcile_to_net_series():
    """Every held bar belongs to exactly one trade → summed net_pct must equal
    the total of the net-return series that produces the gated Sharpe."""
    be, df = _engine(), _synth()
    sig = _trend_signal(df)
    r = be._net_return_series(df, sig, "1d")
    trades = be._reconstruct_trades(df, sig, "1d")
    assert len(trades) > 3
    summed_net = sum(t["net_pct"] for t in trades) / 100.0
    # residual is only 4-decimal per-trade rounding across N trades
    assert abs(summed_net - float(r["net"].sum())) < 1e-4
    for t in trades:
        # net = gross - cost - funding, so net <= gross - cost (funding >= 0
        # drag on crypto, exactly 0 on Bursa)
        assert t["net_pct"] <= t["gross_pct"] - t["cost_pct"] + 1e-9
        assert t["bars_held"] >= 1
        assert t["direction"] in ("long", "short")


def test_cagr_is_geometric():
    be, df = _engine(), _synth()
    perf = be._compute_performance(df, _trend_signal(df), "1d")
    ann = _ANN
    n = perf["n_obs"]
    # rebuild the equity endpoint the same way and confirm CAGR matches
    r = be._net_return_series(df, _trend_signal(df), "1d")
    net = r["net"].values[1:]
    cum_end = float(np.cumprod(1 + np.clip(net, -0.5, 0.5))[-1])
    expected = cum_end ** (ann / n) - 1.0
    assert abs(perf["cagr"] - round(expected, 4)) < 1e-3
    # CAGR (geometric) generally differs from ann_return (arithmetic)
    assert "ann_return" in perf and "cagr" in perf


def test_drawdown_quality_fields_present_and_sane():
    be, df = _engine(), _synth()
    perf = be._compute_performance(df, _trend_signal(df), "1d")
    assert 0.0 <= perf["ulcer_index"] <= 1.0
    assert perf["ulcer_index"] <= perf["max_dd"] + 1e-9   # RMS DD <= max DD
    assert perf["dd_duration_bars"] >= 0
    assert 0.0 <= perf["avg_drawdown"] <= perf["max_dd"] + 1e-9


def test_conservative_fill_and_capacity_are_reports_not_gate_changes():
    """lag=2 and extra_cost>0 must NOT alter the baseline (lag=1, extra=0)."""
    be, df = _engine(), _synth()
    sig = _trend_signal(df)
    base = be._compute_performance(df, sig, "1d")
    same = be._compute_performance(df, sig, "1d", lag=1, extra_cost_per_side=0.0)
    assert base["sharpe_net"] == same["sharpe_net"]
    # a 2-bar delayed fill is a different (usually worse or equal power) number
    cons = be._compute_performance(df, sig, "1d", lag=2)
    assert cons["sharpe_net"] != base["sharpe_net"] or base["total_trades"] == 0
    # capacity haircut only ever reduces (or keeps) net Sharpe
    cap = be._compute_performance(df, sig, "1d", extra_cost_per_side=0.005)
    assert cap["sharpe_net"] <= base["sharpe_net"] + 1e-9
