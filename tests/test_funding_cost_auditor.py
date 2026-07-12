"""B2 — FundingCostAuditor (governance L0, crypto only).

Pins: (1) the auditor is a no-op on Bursa (no perp funding exists there),
(2) it PASSes when funding genuinely moves the net-return series, and
(3) it BLOCKERs a planted "skips funding" call path — one that computes net
returns from a frame stripped of `funding_bar_sum` while real funding data
exists for the run, which would otherwise silently overstate net Sharpe.
"""
import numpy as np
import pandas as pd
import pytest

from governance.inspectors.funding_cost import FundingCostAuditor


def _crypto_case(seed=5, funding_rate=0.0009):
    from agents.backtest_engineer.backtest_engineer import BacktestEngineer

    n = 300
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(0.01 * rng.standard_normal(n)))
    df = pd.DataFrame({
        "open": close, "high": close * 1.001, "low": close * 0.999,
        "close": close, "volume": np.full(n, 1e9),
    }, index=idx)
    df["funding_bar_sum"] = funding_rate
    signals = pd.Series(1.0, index=idx)  # permanently long
    return BacktestEngineer(), df, signals, "1d"


@pytest.fixture
def crypto_mode(monkeypatch):
    """Flip the process into crypto mode for the auditor's own MARKET_MODE
    gate, and give the engine module real crypto funding constants (mirrors
    the monkeypatch pattern already used in tests/test_perps_long_short.py —
    `_net_return_series` reads these as engine-module-bound names)."""
    import config.settings as settings
    import agents.backtest_engineer.engine as engine_mod
    monkeypatch.setattr(settings, "MARKET_MODE", "crypto")
    monkeypatch.setattr(engine_mod, "FUNDING_INTERVAL_HOURS", 8)
    monkeypatch.setattr(engine_mod, "AVG_FUNDING_RATE_PER_INTERVAL", 0.0001)
    return engine_mod


def test_skips_on_bursa(monkeypatch):
    """Bursa has no perp funding — the auditor must be a documented no-op."""
    import config.settings as settings
    monkeypatch.setattr(settings, "MARKET_MODE", "bursa")
    inspector = FundingCostAuditor()
    assert inspector.inspect("backtest_run:bursa-1", {}) is None


def test_pass_when_funding_genuinely_applied(crypto_mode):
    """GOOD case: the real _net_return_series call path is used unmodified —
    funding must move net returns away from the funding-disabled baseline."""
    engine, df, signals, interval = _crypto_case()
    inspector = FundingCostAuditor()
    finding = inspector.inspect("backtest_run:good-1", {
        "engine": engine, "df": df, "signals": signals, "interval": interval,
    })
    assert finding is not None
    assert finding.status == "PASS"
    assert finding.severity == "INFO"
    assert finding.evidence["funding_present"] is True
    assert finding.evidence["identical_with_without"] is False
    assert finding.evidence["max_abs_diff"] > 0


def test_blocker_when_funding_skipped(crypto_mode):
    """BAD case: plant a call path that drops funding_bar_sum before
    computing net returns — despite nonzero funding data existing for the
    run, this path's result must be identical to the funding-disabled
    baseline, which the auditor must catch as a BLOCKER."""
    engine, df, signals, interval = _crypto_case()

    def buggy_with_funding_fn(engine, df, signals, interval):
        # BUG: strips the real per-bar funding settlements AND disables the
        # modeled fallback before computing net returns — the funding term
        # drops out of net returns entirely despite genuine nonzero funding
        # data existing for this run. Exactly the "skips funding" regression
        # under audit (e.g. a reporting code path that re-derives net
        # returns without ever wiring funding in).
        import agents.backtest_engineer.engine as engine_mod
        from agents.backtest_engineer.engine import _net_return_series
        orig_hours = engine_mod.FUNDING_INTERVAL_HOURS
        orig_rate = engine_mod.AVG_FUNDING_RATE_PER_INTERVAL
        engine_mod.FUNDING_INTERVAL_HOURS = None
        engine_mod.AVG_FUNDING_RATE_PER_INTERVAL = 0.0
        try:
            return _net_return_series(
                engine, df.drop(columns=["funding_bar_sum"]), signals, interval)
        finally:
            engine_mod.FUNDING_INTERVAL_HOURS = orig_hours
            engine_mod.AVG_FUNDING_RATE_PER_INTERVAL = orig_rate

    inspector = FundingCostAuditor()
    finding = inspector.inspect("backtest_run:bad-1", {
        "engine": engine, "df": df, "signals": signals, "interval": interval,
        "net_return_with_funding_fn": buggy_with_funding_fn,
    })
    assert finding is not None
    assert finding.status == "FAIL"
    assert finding.severity == "BLOCKER"
    assert finding.evidence["funding_present"] is True
    assert finding.evidence["identical_with_without"] is True
    assert finding.escalate_to == "BacktestEngineer"


def test_standalone_synthetic_case_when_ctx_empty(crypto_mode):
    """With no engine/df/signals supplied, the auditor falls back to its own
    synthetic crypto run — it must still work as a standalone regression
    guard (e.g. run on a schedule, not only per backtest run)."""
    inspector = FundingCostAuditor()
    finding = inspector.inspect("backtest_run:standalone-1", {})
    assert finding is not None
    assert finding.status == "PASS"
    assert finding.evidence["funding_present"] is True


def test_record_persists_finding(crypto_mode):
    """Findings from this inspector must persist via the shared Inspector.record()
    path (governance_findings table) like any other inspector."""
    from data.database import db_session, init_db
    init_db()
    engine, df, signals, interval = _crypto_case()
    inspector = FundingCostAuditor()
    finding = inspector.inspect("backtest_run:persist-1", {
        "engine": engine, "df": df, "signals": signals, "interval": interval,
    })
    row_id = inspector.record(finding)
    assert row_id > 0
    with db_session() as conn:
        row = conn.execute(
            "SELECT agent, level, status, severity FROM governance_findings WHERE id = ?",
            (row_id,),
        ).fetchone()
    assert row["agent"] == "FundingCostAuditor"
    assert row["level"] == "L0"
    assert row["status"] == "PASS"
    assert row["severity"] == "INFO"
