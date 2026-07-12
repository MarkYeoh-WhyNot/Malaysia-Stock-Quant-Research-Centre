"""Translation-integrity gates in _run_backtest: unrepresentable ideas are
rejected with reasons (never genericized), DSL verify blocks, robustness
gate math. All LLM and network calls mocked."""
import numpy as np
import pandas as pd
import pytest

from config.settings import GATE_CONFIG
from data.database import db_session, init_db
from agents.backtest_engineer.backtest_engineer import BacktestEngineer
from agents.backtest_engineer import stats
from agents.backtest_engineer import engine
from agents.leaf_synthesizer.leaf_synthesizer import LeafSynthesizer

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
        # Unrepresentable-rejection tests trigger a real (unmocked)
        # LeafSynthesizer attempt, which fails fast on the missing API key
        # but still logs its audit row — delete it before the parent idea.
        conn.execute("DELETE FROM leaf_synthesis_attempts WHERE idea_id=?", (TEST_IDEA_ID,))
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


def test_unrepresentable_reason_category_wins_over_crypto_keyword_guess(idea, monkeypatch):
    """2026-07-13 fix: even when the rejection text contains 'crypto' (which
    used to force-classify as 'irrelevant'), the explicit reason_category
    passed by _reject_idea must land in rejection_patterns as
    'unrepresentable', not be re-guessed from the text."""
    be, df = idea
    monkeypatch.setattr(
        BacktestEngineer, "_parse_factor",
        lambda self, *a: {"representable": False,
                          "reason": "requires a custom crypto BTC dominance index ratio"})
    monkeypatch.setattr(
        "agents.leaf_synthesizer.leaf_synthesizer.LeafSynthesizer.synthesize",
        lambda self, *a, **kw: None)
    be._run_backtest(TEST_IDEA_ID)

    with db_session() as conn:
        cemetery = conn.execute(
            "SELECT factor_type, sector FROM strategy_cemetery WHERE idea_id=?",
            (TEST_IDEA_ID,)).fetchone()
        pattern = conn.execute(
            "SELECT reason_category FROM rejection_patterns WHERE factor_type=? "
            "AND sector=? AND reason_category='unrepresentable'",
            (cemetery["factor_type"], cemetery["sector"])).fetchone()
    assert pattern is not None, "explicit reason_category must beat the 'crypto' keyword guess"


def test_unrepresentable_triggers_leaf_synthesis_attempt(idea, monkeypatch):
    be, df = idea
    monkeypatch.setattr(BacktestEngineer, "_parse_factor",
                        lambda self, *a: {"representable": False,
                                          "reason": "requires analyst sentiment data"})
    calls = []
    monkeypatch.setattr(LeafSynthesizer, "synthesize",
                        lambda self, idea_id, hyp, formula, reason:
                        calls.append((idea_id, hyp, formula, reason)))
    result = be._run_backtest(TEST_IDEA_ID)
    assert result["verdict"] == "REJECTED"
    assert len(calls) == 1
    assert calls[0][0] == TEST_IDEA_ID
    assert calls[0][3] == "requires analyst sentiment data"


def test_leaf_synthesis_failure_never_blocks_the_rejection(idea, monkeypatch):
    be, df = idea
    monkeypatch.setattr(BacktestEngineer, "_parse_factor",
                        lambda self, *a: {"representable": False,
                                          "reason": "requires analyst sentiment data"})

    def boom(self, *a, **kw):
        raise RuntimeError("synthesis pipeline exploded")
    monkeypatch.setattr(LeafSynthesizer, "synthesize", boom)

    result = be._run_backtest(TEST_IDEA_ID)
    assert result["verdict"] == "REJECTED"
    assert result["error"] == "dsl_unrepresentable"
    status, reason = _idea_status()
    assert status == "rejected"
    assert "analyst sentiment" in reason


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
    base_sig = engine._compute_signals(be, df, {"signal_type": "dsl", "dsl": tree})
    base = engine._compute_performance(be, df, base_sig, "1d")
    score1 = stats.robustness_check(be, df, tree, base["sharpe_net"], "1d", GATE_CONFIG)
    score2 = stats.robustness_check(be, df, tree, base["sharpe_net"], "1d", GATE_CONFIG)
    assert score1 == score2  # seeded → reproducible
    assert score1 >= 0.6, f"steady-trend momentum should be robust, got {score1}"


def test_legacy_params_still_work(idea):
    """Paper trading replays stored pre-DSL params — signals must be unchanged."""
    be, df = idea
    legacy = {"signal_type": "sma_crossover", "fast_period": 20,
              "slow_period": 50, "long_only": True}
    sig = engine._compute_signals(be, df, legacy)
    fast = df["close"].rolling(20).mean()
    slow = df["close"].rolling(50).mean()
    want = pd.Series(np.where(fast > slow, 1.0, 0.0), index=df.index)
    assert (sig == want).all()


def test_parse_factor_prompt_carries_shape_guide(monkeypatch):
    """Prompt-pin: the parser prompt must carry the leaf catalog, the
    structure-only shape cards, and the idea-#73 WRONG-vs-RIGHT negative
    example — and still never contain the word 'default' (anchoring)."""
    captured = {}

    def fake_call(self, system, messages, **kw):
        captured["prompt"] = messages[0]["content"]
        return {"representable": False, "reason": "capture only"}

    monkeypatch.setattr(BacktestEngineer, "call_claude_json", fake_call)
    be = BacktestEngineer.__new__(BacktestEngineer)
    out = be._parse_factor("close above its 50-day EMA", "EMA level", "uptrend filter")
    assert out["representable"] is False
    p = captured["prompt"]
    assert "CONDITION SHAPE GUIDE" in p
    assert "ma_level" in p
    assert '"leaf": "ema_cross", "fast": 2, "slow": 50' in p     # BAD example
    assert '"leaf": "ma_level", "ma_type": "ema", "period": 50' in p  # CORRECT
    assert "NEVER approximate" in p
    assert "default" not in p.lower()
