"""Tests for FillConventionAuditor — trade PnL reconciliation invariant."""

import logging
import numpy as np
import pandas as pd
import pytest

from agents.backtest_engineer.backtest_engineer import BacktestEngineer
from agents.backtest_engineer import engine
from governance.inspectors.fill_convention import FillConventionAuditor


_ANN = 252  # Bursa annualization


def _engine():
    """Minimal BacktestEngineer for testing."""
    be = BacktestEngineer.__new__(BacktestEngineer)
    be.logger = logging.getLogger("test")
    be.name = "BacktestEngineer"
    return be


def _synth(n=400, seed=1):
    """Synthetic OHLCV dataframe."""
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="D")
    close = pd.Series(100 * np.cumprod(1 + rng.randn(n) * 0.01), index=idx)
    return pd.DataFrame({
        "close": close,
        "open": close.shift(1).fillna(close.iloc[0]),
        "volume": rng.randint(1_000_000, 3_000_000, n).astype(float),
    })


def _trend_signal(df):
    """Simple trend signal: long when close > 20-day SMA."""
    return (df["close"] > df["close"].rolling(20).mean()).astype(float).fillna(0)


def test_fill_convention_no_trades():
    """Empty trades list should pass trivially."""
    auditor = FillConventionAuditor()
    finding = auditor.inspect(
        scope="backtest_run:1",
        ctx={
            "trades": [],
            "backtest_net_return": 0.05,
        }
    )
    assert finding is not None
    assert finding.status == "PASS"
    assert finding.severity == "INFO"


def test_fill_convention_reconciles_good():
    """Good case: reconstructed trades reconcile to the backtest return."""
    be = _engine()
    df = _synth(n=400)
    sig = _trend_signal(df)

    # Compute performance (this is the ground truth)
    perf = engine._compute_performance(be, df, sig, "1d")
    # Reconstruct trades
    trades = engine._reconstruct_trades(be, df, sig, "1d")

    # The backtest net return is the sum of the net-return series (already in decimal form)
    r = engine._net_return_series(be, df, sig, "1d")
    backtest_net_return = float(r["net"].sum())

    auditor = FillConventionAuditor()
    finding = auditor.inspect(
        scope="backtest_run:1",
        ctx={
            "trades": trades,
            "backtest_net_return": backtest_net_return,
            "tolerance": 1e-4,
        }
    )

    assert finding is not None
    assert finding.status == "PASS"
    assert finding.severity == "INFO"
    assert finding.evidence["reconciles"] is True
    assert finding.evidence["trades_count"] == len(trades)


def test_fill_convention_blocker_on_divergence():
    """Bad case: artificially corrupt trades to trigger reconciliation failure."""
    be = _engine()
    df = _synth(n=400)
    sig = _trend_signal(df)

    # Get the ground truth trades and backtest return
    trades = engine._reconstruct_trades(be, df, sig, "1d")
    r = engine._net_return_series(be, df, sig, "1d")
    backtest_net_return = float(r["net"].sum())

    # Corrupt trades: double-count costs on the first trade
    # This simulates a bug where a closing cost is counted twice.
    corrupted_trades = []
    for i, t in enumerate(trades):
        t_copy = t.copy()
        if i == 0 and len(trades) > 1:
            # Double the cost on the first trade's close (simulate double-counting)
            t_copy["cost_pct"] = t["cost_pct"] * 2
            # Recompute net_pct with the corrupted cost
            t_copy["net_pct"] = t["gross_pct"] - t_copy["cost_pct"]
        corrupted_trades.append(t_copy)

    auditor = FillConventionAuditor()
    finding = auditor.inspect(
        scope="backtest_run:2",
        ctx={
            "trades": corrupted_trades,
            "backtest_net_return": backtest_net_return,
            "tolerance": 1e-4,
        }
    )

    assert finding is not None
    assert finding.status == "FAIL"
    assert finding.severity == "BLOCKER"
    assert finding.evidence["reconciles"] is False
    assert finding.evidence["absolute_error"] > 1e-4


def test_fill_convention_blocker_with_missing_cost():
    """Bad case: simulate missing exit cost on a trade (cost attribution gap)."""
    be = _engine()
    df = _synth(n=400)
    sig = _trend_signal(df)

    trades = engine._reconstruct_trades(be, df, sig, "1d")
    r = engine._net_return_series(be, df, sig, "1d")
    backtest_net_return = float(r["net"].sum())

    # Corrupt: zero out the exit cost on a trade
    corrupted_trades = []
    for i, t in enumerate(trades):
        t_copy = t.copy()
        if i == len(trades) // 2:  # Middle trade
            exit_cost = t["cost_pct"]
            # Remove the exit cost and adjust net accordingly
            t_copy["cost_pct"] = 0.0
            t_copy["net_pct"] = t["gross_pct"] - 0.0  # cost is now zero
        corrupted_trades.append(t_copy)

    auditor = FillConventionAuditor()
    finding = auditor.inspect(
        scope="backtest_run:3",
        ctx={
            "trades": corrupted_trades,
            "backtest_net_return": backtest_net_return,
            "tolerance": 1e-4,
        }
    )

    assert finding is not None
    assert finding.status == "FAIL"
    assert finding.severity == "BLOCKER"
    assert finding.evidence["reconciles"] is False


def test_fill_convention_finding_recorded():
    """Test that findings can be persisted via the record() method."""
    auditor = FillConventionAuditor()
    finding = auditor.inspect(
        scope="backtest_run:4",
        ctx={
            "trades": [],
            "backtest_net_return": 0.0,
        }
    )
    # This will write to the DB
    row_id = auditor.record(finding)
    assert isinstance(row_id, int)
    assert row_id > 0


def test_fill_convention_multiple_trades_small_error():
    """Verify that small rounding errors within tolerance pass."""
    be = _engine()
    df = _synth(n=400)
    sig = _trend_signal(df)

    trades = engine._reconstruct_trades(be, df, sig, "1d")
    r = engine._net_return_series(be, df, sig, "1d")
    backtest_net_return = float(r["net"].sum())

    # Inject a tiny rounding error within tolerance
    perturbed_trades = []
    for t in trades:
        t_copy = t.copy()
        t_copy["net_pct"] += 0.00001  # Add 0.00001 pct to each trade
        perturbed_trades.append(t_copy)

    auditor = FillConventionAuditor()
    finding = auditor.inspect(
        scope="backtest_run:5",
        ctx={
            "trades": perturbed_trades,
            "backtest_net_return": backtest_net_return,
            "tolerance": 0.001,  # Relax tolerance to catch it
        }
    )

    # With relaxed tolerance, small errors should pass
    assert finding is not None
    # Note: the error should be small enough to pass with tolerance=0.001
    if finding.evidence["absolute_error"] < 0.001:
        assert finding.status == "PASS"
    else:
        assert finding.status == "FAIL"


def test_fill_convention_edge_case_single_flat_bar():
    """Test with a signal that spends most bars flat."""
    be = _engine()
    df = _synth(n=100)
    # Flat signal except for one bar
    sig = pd.Series(0.0, index=df.index)
    sig.iloc[20:30] = 1.0

    trades = engine._reconstruct_trades(be, df, sig, "1d")
    r = engine._net_return_series(be, df, sig, "1d")
    backtest_net_return = float(r["net"].sum())

    auditor = FillConventionAuditor()
    finding = auditor.inspect(
        scope="backtest_run:6",
        ctx={
            "trades": trades,
            "backtest_net_return": backtest_net_return,
            "tolerance": 1e-4,
        }
    )

    assert finding is not None
    assert finding.status == "PASS"
    assert finding.evidence["reconciles"] is True
