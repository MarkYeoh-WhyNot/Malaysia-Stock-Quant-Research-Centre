"""Translation-integrity gates in _run_backtest: unrepresentable ideas are
rejected with reasons (never genericized), DSL verify blocks, robustness
gate math. All LLM and network calls mocked."""
import numpy as np
import pandas as pd
import pytest

from data.database import db_session, init_db
from agents.backtest_engineer.backtest_engineer import BacktestEngineer

TEST_IDEA_ID = 987_654_320


@pytest.fixture()
def idea(monkeypatch):
    init_db()
    _cleanup()
    with db_session() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO alpha_ideas (id, slug, title, hypothesis, ticker, "
            "factor_formula, timeframe, stage, status) "
            "VALUES (?, 'test-integrity-idea', 'Integrity test idea', 'test', "
            "'1155.KL', 'volume spike with gap up', '1d', 'stage2', 'active')",
            (TEST_IDEA_ID,))

    be = BacktestEngineer.__new__(BacktestEngineer)
    import logging
    be.logger = logging.getLogger("test")
    be.name = "BacktestEngineer"

    # synthetic 5y price data with volume
    rng = np.random.RandomState(7)
    n = 1300
    idx = pd.date_range("2021-01-01", periods=n, freq="B")
    close = pd.Series(10 * np.cumprod(1 + rng.randn(n) * 0.01 + 0.0004), index=idx)
    df = pd.DataFrame({
        "close": close,
        "open": close.shift(1).fillna(close.iloc[0]),
        "volume": rng.randint(2_000_000, 4_000_000, n).astype(float),
    })
    monkeypatch.setattr(BacktestEngineer, "_fetch_prices",
                        lambda self, *a, **k: df)
    monkeypatch.setattr(BacktestEngineer, "_log_progress", lambda self, *a, **k: None)
    monkeypatch.setattr(BacktestEngineer, "_clear_progress", lambda self, *a, **k: None)
    yield be, df
    _cleanup()


def _cleanup():
    with db_session() as conn:
        conn.execute("DELETE FROM backtest_runs WHERE idea_id=?", (TEST_IDEA_ID,))
        conn.execute("DELETE FROM pipeline_events WHERE idea_id=?", (TEST_IDEA_ID,))
        conn.execute("DELETE FROM gate_decisions WHERE idea_id=?", (TEST_IDEA_ID,))
        conn.execute("DELETE FROM backtest_series WHERE idea_id=?", (TEST_IDEA_ID,))
        conn.execute("DELETE FROM alpha_ideas WHERE id=?", (TEST_IDEA_ID,))


def _idea_status():
    with db_session() as conn:
        row = conn.execute(
            "SELECT status, rejection_reason FROM alpha_ideas WHERE id=?",
            (TEST_IDEA_ID,)).fetchone()
    return row["status"], row["rejection_reason"]


def test_unrepresentable_rejected_with_reason_not_genericized(idea, monkeypatch):
    be, df = idea
    monkeypatch.setattr(BacktestEngineer, "_parse_factor",
                        lambda self, *a: {"representable": False,
                                          "reason": "requires analyst sentiment data"})
    result = be._run_backtest(TEST_IDEA_ID)
    assert result["verdict"] == "REJECTED"
    assert result["error"] == "dsl_unrepresentable"
    status, reason = _idea_status()
    assert status == "rejected"
    assert "analyst sentiment" in reason
    with db_session() as conn:
        run = conn.execute(
            "SELECT run_type FROM backtest_runs WHERE idea_id=?",
            (TEST_IDEA_ID,)).fetchone()
    assert run["run_type"] == "dsl_unrepresentable"


def test_never_firing_signal_blocked_by_verify(idea, monkeypatch):
    be, df = idea
    tree = {"entry": {"leaf": "gap", "direction": "up", "min_pct": 0.19},  # ~unreachable
            "exit": None}
    monkeypatch.setattr(BacktestEngineer, "_parse_factor",
                        lambda self, *a: {"signal_type": "dsl", "representable": True,
                                          "dsl": tree, "long_only": True})
    result = be._run_backtest(TEST_IDEA_ID)
    assert result["verdict"] == "REJECTED"
    assert result["error"] == "dsl_verify"
    _, reason = _idea_status()
    assert "never fires" in reason


def test_always_on_signal_blocked_by_verify(idea, monkeypatch):
    be, df = idea
    # momentum over 2 bars with threshold -0.5: essentially always true
    tree = {"entry": {"leaf": "momentum", "period": 2, "min_return": -0.5}, "exit": None}
    monkeypatch.setattr(BacktestEngineer, "_parse_factor",
                        lambda self, *a: {"signal_type": "dsl", "representable": True,
                                          "dsl": tree, "long_only": True})
    result = be._run_backtest(TEST_IDEA_ID)
    assert result["verdict"] == "REJECTED"
    assert result["error"] == "dsl_verify"
    _, reason = _idea_status()
    assert "buy-and-hold" in reason


def test_robustness_check_math():
    """Robust regime signal keeps most perturbations; the score is a valid
    fraction and reproducible (seeded)."""
    be = BacktestEngineer.__new__(BacktestEngineer)
    import logging
    be.logger = logging.getLogger("test")
    be.name = "BacktestEngineer"
    rng = np.random.RandomState(3)
    n = 400
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    # steady uptrend: momentum > 0 is robust to period jitter
    close = pd.Series(10 * np.cumprod(1 + np.abs(rng.randn(n)) * 0.002 + 0.001), index=idx)
    df = pd.DataFrame({"close": close,
                       "volume": np.full(n, 3_000_000.0)})
    tree = {"entry": {"leaf": "momentum", "period": 20, "min_return": 0.0}, "exit": None}
    base_sig = be._compute_signals(df, {"signal_type": "dsl", "dsl": tree})
    base = be._compute_performance(df, base_sig, "1d")
    score1 = be._robustness_check(df, tree, base["sharpe_net"], "1d")
    score2 = be._robustness_check(df, tree, base["sharpe_net"], "1d")
    assert score1 == score2  # seeded → reproducible
    assert score1 >= 0.6, f"steady-trend momentum should be robust, got {score1}"


def test_legacy_params_still_work(idea):
    """Paper trading replays stored pre-DSL params — signals must be unchanged."""
    be, df = idea
    legacy = {"signal_type": "sma_crossover", "fast_period": 20,
              "slow_period": 50, "long_only": True}
    sig = be._compute_signals(df, legacy)
    fast = df["close"].rolling(20).mean()
    slow = df["close"].rolling(50).mean()
    want = pd.Series(np.where(fast > slow, 1.0, 0.0), index=df.index)
    assert (sig == want).all()
