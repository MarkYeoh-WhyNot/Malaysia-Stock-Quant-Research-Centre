"""PnLConsistencyInspector — every consumer of per-bar net returns for a
backtest run (persisted equity curve, regime attribution, gated metrics)
MUST route through agents.backtest_engineer.engine._net_return_series, the
single source of truth. This test plants a known-bad case (a consumer
recomputed with a DIFFERENT cost rate than the canonical series) and a
known-good case (all consumers derived from the identical canonical output).
"""
import numpy as np
import pandas as pd
import pytest

from data.database import db_session, init_db
from governance.schemas import Finding
from governance.inspectors.pnl_consistency import PnLConsistencyInspector
from agents.backtest_engineer.backtest_engineer import BacktestEngineer
from agents.backtest_engineer.engine import _net_return_series


@pytest.fixture(autouse=True)
def _setup():
    init_db()
    yield


def _make_df(n=300, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=n, freq="B")
    steps = rng.normal(0.0005, 0.01, n)
    close = 10.0 * np.cumprod(1 + steps)
    return pd.DataFrame({"close": close, "volume": 3_000_000}, index=idx)


def _make_signals(df):
    n = len(df)
    # Alternate flat/long blocks of 10 bars so there are real position
    # changes (and therefore real transaction costs) to diverge on.
    pattern = (np.arange(n) % 20 < 10).astype(float)
    return pd.Series(pattern, index=df.index)


def test_known_good_all_consumers_match_canonical():
    """All three consumers derive from the SAME canonical series → PASS."""
    engine = BacktestEngineer()
    df = _make_df()
    signals = _make_signals(df)

    canonical = _net_return_series(engine, df, signals, "1d")["net"]

    inspector = PnLConsistencyInspector()
    ctx = {
        "engine": engine,
        "df": df,
        "signals": signals,
        "interval": "1d",
        "reported": {
            "equity_curve": canonical,
            "regime": canonical,
            "gate_metrics": canonical,
        },
    }

    finding = inspector.inspect("backtest_run:good-1", ctx)

    assert finding is not None
    assert isinstance(finding, Finding)
    assert finding.status == "PASS"
    assert finding.severity == "INFO"
    assert finding.evidence["mismatches"] == []
    assert set(finding.evidence["checked_consumers"]) == {
        "equity_curve", "regime", "gate_metrics",
    }


def test_known_bad_divergent_cost_rate_flags_blocker():
    """Plant the exact bug class this inspector exists to catch: one
    consumer (the persisted equity curve) is built from a net-return series
    computed with a DIFFERENT cost rate (extra_cost_per_side) than the
    canonical series behind the gate. Must return a BLOCKER Finding."""
    engine = BacktestEngineer()
    df = _make_df()
    signals = _make_signals(df)

    # The "reported" equity curve was (incorrectly) built with an extra
    # market-impact haircut baked in...
    divergent_equity_curve = _net_return_series(
        engine, df, signals, "1d", extra_cost_per_side=0.01,
    )["net"]
    # ...while regime/gate consumers correctly used the canonical (default
    # cost) series.
    canonical = _net_return_series(engine, df, signals, "1d")["net"]

    inspector = PnLConsistencyInspector()
    ctx = {
        "engine": engine,
        "df": df,
        "signals": signals,
        "interval": "1d",
        # extra_cost_per_side omitted → canonical recompute uses the
        # default (0.0), matching the gate/regime consumers, NOT the
        # divergent equity curve.
        "reported": {
            "equity_curve": divergent_equity_curve,
            "regime": canonical,
            "gate_metrics": canonical,
        },
    }

    finding = inspector.inspect("backtest_run:bad-1", ctx)

    assert finding is not None
    assert finding.status == "FAIL"
    assert finding.severity == "BLOCKER"
    mismatched = {m["consumer"] for m in finding.evidence["mismatches"]}
    assert mismatched == {"equity_curve"}
    assert "regime" not in mismatched
    assert "gate_metrics" not in mismatched
    assert finding.escalate_to == "BacktestEngineer"


def test_known_bad_divergent_lag_flags_blocker():
    """A second independent divergence path: a consumer recomputed with a
    different signal lag (lag=2, the fill-robustness variant) instead of
    the run's actual lag=1 canonical series."""
    engine = BacktestEngineer()
    df = _make_df()
    signals = _make_signals(df)

    canonical = _net_return_series(engine, df, signals, "1d", lag=1)["net"]
    divergent_regime = _net_return_series(engine, df, signals, "1d", lag=2)["net"]

    inspector = PnLConsistencyInspector()
    ctx = {
        "engine": engine,
        "df": df,
        "signals": signals,
        "interval": "1d",
        "lag": 1,
        "reported": {
            "equity_curve": canonical,
            "regime": divergent_regime,
            "gate_metrics": canonical,
        },
    }

    finding = inspector.inspect("backtest_run:bad-2", ctx)

    assert finding.status == "FAIL"
    assert finding.severity == "BLOCKER"
    mismatched = {m["consumer"] for m in finding.evidence["mismatches"]}
    assert mismatched == {"regime"}


def test_missing_context_returns_none():
    """No `reported` consumers to check → not applicable, returns None."""
    engine = BacktestEngineer()
    df = _make_df()
    signals = _make_signals(df)

    inspector = PnLConsistencyInspector()
    finding = inspector.inspect("backtest_run:na-1", {
        "engine": engine, "df": df, "signals": signals,
    })
    assert finding is None


def test_finding_can_be_recorded():
    """Sanity check: the Finding this inspector produces persists through
    the shared Inspector.record() path (governance_findings table)."""
    engine = BacktestEngineer()
    df = _make_df()
    signals = _make_signals(df)
    canonical = _net_return_series(engine, df, signals, "1d")["net"]

    inspector = PnLConsistencyInspector()
    finding = inspector.inspect("backtest_run:record-1", {
        "engine": engine,
        "df": df,
        "signals": signals,
        "interval": "1d",
        "reported": {"gate_metrics": canonical},
    })
    assert finding.status == "PASS"

    row_id = inspector.record(finding)
    assert row_id > 0

    with db_session() as conn:
        cursor = conn.execute(
            "SELECT agent, level, scope, status, severity FROM governance_findings WHERE id = ?",
            (row_id,),
        )
        row = cursor.fetchone()

    assert row is not None
    assert row["agent"] == "PnLConsistencyInspector"
    assert row["level"] == "L0"
    assert row["scope"] == "backtest_run:record-1"
    assert row["status"] == "PASS"
    assert row["severity"] == "INFO"
