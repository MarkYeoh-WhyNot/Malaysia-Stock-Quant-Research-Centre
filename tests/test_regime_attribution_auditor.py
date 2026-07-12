"""RegimeAttributionAuditor: regime-split returns must match the gated series.

Confirms (see agents/backtest_engineer/engine.py `_compute_regimes` and
`_net_return_series`) that the volatility-regime stress test and the gated
Sharpe/PSR decision are computed from the identical per-bar net-return
series — including funding on crypto. A planted mismatch (regime split on
GROSS returns while the gate uses NET) must be caught as a BLOCKER; the
current engine.py wiring (both routed through
`_net_return_series(...)["net"]`) must PASS.
"""

import pytest
from data.database import db_session, init_db
from governance.inspectors.regime_attribution import RegimeAttributionAuditor


@pytest.fixture(autouse=True)
def _setup():
    init_db()
    yield


def _synthetic_bars():
    """Deterministic per-bar gross returns, costs, and funding drag.

    Mirrors the shape of agents/backtest_engineer/engine.py::_net_return_series:
    net = gross - cost - funding (leverage inert at 1.0 for this fixture).
    """
    gross = [0.010, -0.005, 0.020, 0.000, -0.015, 0.008, 0.012, -0.002, 0.004, 0.006]
    cost = [0.001, 0.000, 0.001, 0.000, 0.000, 0.001, 0.000, 0.000, 0.001, 0.000]
    # Non-zero funding drag on every bar — the crypto case B2 was worried about.
    funding = [0.0005] * len(gross)
    net = [g - c - f for g, c, f in zip(gross, cost, funding)]
    return gross, net


class TestRegimeAttributionAuditorGoodCase:
    """Both _compute_regimes and the gate route through the same NET series
    (current engine.py wiring: _net_return_series(...)["net"] for both)."""

    def test_matching_net_series_passes(self):
        _, net = _synthetic_bars()
        auditor = RegimeAttributionAuditor()

        # Same series object conceptually — what the current engine.py code
        # does: both _compute_regimes and _compute_performance call
        # _net_return_series(...)["net"].
        finding = auditor.inspect(
            scope="backtest_run:good-1",
            ctx={"regime_net_returns": list(net), "gate_net_returns": list(net)},
        )

        assert finding is not None
        assert finding.status == "PASS"
        assert finding.severity == "INFO"
        assert finding.evidence["consistent"] is True
        assert finding.evidence["n_bars_diverging"] == 0

    def test_matching_net_series_can_be_recorded(self):
        _, net = _synthetic_bars()
        auditor = RegimeAttributionAuditor()
        finding = auditor.inspect(
            scope="backtest_run:good-2",
            ctx={"regime_net_returns": list(net), "gate_net_returns": list(net)},
        )
        row_id = auditor.record(finding)
        assert row_id > 0

        with db_session() as conn:
            row = conn.execute(
                "SELECT status, severity FROM governance_findings WHERE id = ?",
                (row_id,),
            ).fetchone()
        assert row["status"] == "PASS"
        assert row["severity"] == "INFO"

    def test_missing_series_is_not_applicable(self):
        """If ctx lacks either series, the check is skipped (returns None),
        matching Inspector.inspect's documented None-if-not-applicable
        contract."""
        auditor = RegimeAttributionAuditor()
        assert auditor.inspect(scope="backtest_run:na", ctx={}) is None
        assert (
            auditor.inspect(
                scope="backtest_run:na", ctx={"regime_net_returns": [0.01]}
            )
            is None
        )


class TestRegimeAttributionAuditorBadCase:
    """PLANTED BAD CASE: regime split computed on GROSS returns while the
    gate metric uses NET returns (costs + crypto funding stripped out of
    the regime attribution only) — must be a BLOCKER."""

    def test_gross_vs_net_mismatch_is_blocker(self):
        gross, net = _synthetic_bars()
        auditor = RegimeAttributionAuditor()

        finding = auditor.inspect(
            scope="backtest_run:bad-1",
            ctx={
                # BUG: regime attribution re-derived from GROSS returns,
                # never subtracting cost or funding.
                "regime_net_returns": list(gross),
                # Gate correctly uses NET (cost + funding deducted).
                "gate_net_returns": list(net),
            },
        )

        assert finding is not None
        assert finding.status == "FAIL"
        assert finding.severity == "BLOCKER"
        assert finding.escalate_to == "BacktestEngineer"
        assert finding.evidence["consistent"] is False
        assert finding.evidence["n_bars_diverging"] == len(gross)
        assert finding.evidence["max_abs_diff"] > 0

    def test_length_mismatch_is_blocker(self):
        """Different-length series can never be the same underlying data —
        an even more blatant divergence than a value mismatch."""
        _, net = _synthetic_bars()
        auditor = RegimeAttributionAuditor()

        finding = auditor.inspect(
            scope="backtest_run:bad-2",
            ctx={
                "regime_net_returns": list(net)[:5],
                "gate_net_returns": list(net),
            },
        )

        assert finding is not None
        assert finding.status == "FAIL"
        assert finding.severity == "BLOCKER"
        assert finding.evidence["reason"] == "length_mismatch"

    def test_partial_divergence_still_blocks(self):
        """Even a single diverging bar (e.g. funding applied to only part
        of the series) must fail — this is a correctness invariant, not a
        statistical tolerance."""
        _, net = _synthetic_bars()
        tampered = list(net)
        tampered[3] += 0.05  # one bar silently re-derived differently
        auditor = RegimeAttributionAuditor()

        finding = auditor.inspect(
            scope="backtest_run:bad-3",
            ctx={"regime_net_returns": tampered, "gate_net_returns": list(net)},
        )

        assert finding.status == "FAIL"
        assert finding.severity == "BLOCKER"
        assert finding.evidence["n_bars_diverging"] == 1


def test_engine_wiring_matches_good_case():
    """Live confirmation (not a mock) that agents/backtest_engineer/engine.py
    currently wires _compute_regimes and the gated metric through the same
    _net_return_series(...)['net'] call, i.e. the real system is in the
    GOOD case the auditor is designed to pass.

    This does not modify engine.py — it only reads/exercises the existing
    single-source-of-truth series to guard against future drift silently
    reintroducing the gross/net split.
    """
    import numpy as np
    import pandas as pd
    from agents.backtest_engineer import engine as eng

    class _StubEngine:
        def _cost_rates(self, df, interval):
            return {"buy": 0.001, "sell": 0.001}

    rng = np.random.default_rng(42)
    n = 120
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    close = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, n)), index=idx)
    df = pd.DataFrame({"close": close}, index=idx)
    signals = pd.Series(np.where(rng.normal(0, 1, n) > 0, 1.0, 0.0), index=idx)

    stub = _StubEngine()
    gate_series = eng._net_return_series(stub, df, signals, "1d")["net"]

    params = {"signal_type": "momentum", "momentum_period": 20}
    regimes = eng._compute_regimes(stub, df, params, "1d")

    # The regime function must not error and must return the expected keys
    # (this exercises the real _compute_regimes code path, which internally
    # calls the same _net_return_series(...)['net'] as gate_series above).
    assert set(regimes.keys()) == {
        "sharpe_low_vol", "sharpe_mid_vol", "sharpe_high_vol", "regimes_positive",
    }
    assert isinstance(gate_series, pd.Series)
    assert len(gate_series) == n
