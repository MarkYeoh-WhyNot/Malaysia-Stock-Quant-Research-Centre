"""Parameter-sweep optimizer: zscore leaf, seeded config generation, honest
test-slice protocol, deflated-hurdle integration, queue wiring."""
import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from agents.backtest_engineer import signal_dsl
from agents.backtest_engineer.optimizer import (
    randomize_tree, generate_configs, DEFAULT_N_CONFIGS)


ZTREE = {"entry": {"leaf": "zscore", "period": 20, "below": -2.0},
         "exit": {"leaf": "zscore", "period": 20, "above": 0.0}}


# ── zscore leaf ───────────────────────────────────────────────────────────────

def test_zscore_leaf_validates_and_fires_on_synthetic_reversion():
    assert signal_dsl.validate(ZTREE) == []
    # strongly mean-reverting series: sine wave + noise
    idx = pd.date_range("2024-01-01", periods=500, freq="D")
    rng = np.random.RandomState(7)
    close = pd.Series(100 + 10 * np.sin(np.arange(500) / 5.0) + rng.randn(500) * 0.5,
                      index=idx)
    df = pd.DataFrame({"close": close, "volume": 1e6})
    sig = signal_dsl.signal_from_dsl(df, ZTREE)
    assert (sig != 0).sum() > 20            # it actually trades
    assert set(sig.unique()) <= {0.0, 1.0}  # long-only tree in Bursa mode


def test_zscore_rejects_out_of_range():
    bad = {"entry": {"leaf": "zscore", "period": 5, "below": -2.0}}  # period < 10
    assert signal_dsl.validate(bad)


def test_zscore_in_catalog_and_perturbable():
    assert "zscore" in signal_dsl.leaf_catalog_text()
    p = signal_dsl.perturb_tree(ZTREE, np.random.RandomState(3))
    assert 10 <= p["entry"]["period"] <= 200
    assert -4.0 <= p["entry"]["below"] <= 0.0


def test_ma_level_randomize_preserves_choices():
    tree = {"entry": {"leaf": "ma_level", "ma_type": "ema", "period": 50,
                      "direction": "above"}}
    rng = np.random.RandomState(3)
    for _ in range(20):
        r = randomize_tree(tree, rng)
        assert signal_dsl.validate(r) == []
        node = r["entry"]
        assert node["ma_type"] == "ema" and node["direction"] == "above"
        assert 2 <= node["period"] <= 300


# ── config generation ─────────────────────────────────────────────────────────

def test_generate_configs_deterministic_and_in_range():
    a = generate_configs(ZTREE, "1155.KL", seed=99, n_total=40)
    b = generate_configs(ZTREE, "1155.KL", seed=99, n_total=40)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    c = generate_configs(ZTREE, "1155.KL", seed=100, n_total=40)
    assert json.dumps(a, sort_keys=True) != json.dumps(c, sort_keys=True)
    for cfg in a:
        assert signal_dsl.validate(cfg["dsl"]) == []
        assert cfg["instrument"] == "1155.KL"
        assert cfg["timeframe"] == "1d"     # Bursa: SWEEP_TIMEFRAMES == ["1d"]


def test_generate_configs_uses_universe_when_no_ticker():
    cfgs = generate_configs(ZTREE, "", seed=1, n_total=40)
    instruments = {c["instrument"] for c in cfgs}
    assert len(instruments) == 5
    assert "1155.KL" in instruments


# ── run_sweep protocol (synthetic, no network/LLM) ────────────────────────────

def _fake_engine_env(monkeypatch, idea_id):
    """Insert a fake idea and patch data/parse so run_sweep is hermetic."""
    from data.database import init_db, db_session
    init_db()
    with db_session() as conn:
        conn.execute("DELETE FROM alpha_ideas WHERE id=?", (idea_id,))
        conn.execute(
            "INSERT INTO alpha_ideas (id, slug, title, hypothesis, ticker, timeframe, "
            "factor_formula, stage, status) VALUES (?,?,?,?,?,?,?,?,?)",
            (idea_id, f"opt-test-{idea_id}", "OPT zscore test", "mean reversion",
             "1155.KL", "1d", "z-score < -2 on 20 bars", "stage2", "optimizing"))

    idx = pd.date_range("2019-01-01", periods=1200, freq="D")
    rng = np.random.RandomState(11)
    close = pd.Series(100 + 15 * np.sin(np.arange(1200) / 7.0) + rng.randn(1200),
                      index=idx)
    df = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99,
                       "close": close, "volume": 2e6})

    from agents.backtest_engineer.backtest_engineer import BacktestEngineer
    monkeypatch.setattr(BacktestEngineer, "_parse_factor",
                        lambda self, f, t, h: {"representable": True, "dsl": ZTREE,
                                               "signal_type": "dsl"})
    monkeypatch.setattr(BacktestEngineer, "_fetch_prices",
                        lambda self, sym, interval, days=1825: df)
    return df


def test_run_sweep_selects_winner_and_touches_test_once(monkeypatch):
    from agents.backtest_engineer.optimizer import run_sweep
    from agents.backtest_engineer.backtest_engineer import BacktestEngineer
    from agents.backtest_engineer import engine as engine_mod
    idea_id = 999911
    df = _fake_engine_env(monkeypatch, idea_id)

    # Spy: count _compute_performance calls on the TEST slice, identified by
    # its start timestamp (val and test have equal lengths, so length alone
    # can't distinguish them).
    t = int(len(df) * 0.6)
    v = int(len(df) * 0.2)
    test_start = df.index[t + v]
    calls = {"test": 0}
    orig = engine_mod._compute_performance

    def spy(eng, frame, signals, interval, leverage=None, lag=1, extra_cost_per_side=0.0):
        if len(frame) and frame.index[0] == test_start:
            calls["test"] += 1
        return orig(eng, frame, signals, interval, leverage, lag, extra_cost_per_side)

    monkeypatch.setattr(engine_mod, "_compute_performance", spy)

    result = run_sweep(idea_id, seed=7, n_configs=30)
    assert not result.get("error")
    assert result["n_configs"] == 30
    assert result["n_evaluated"] == 30
    assert calls["test"] <= 1          # test slice evaluated at most ONCE
    if result["winner"]:
        assert calls["test"] == 1
        assert "test_sharpe" in result["winner"]
        assert result["winner"]["dsl"]  # tree carried for promotion
        # summary top rows are slim (no trees)
        assert all("dsl" not in t for t in result["top"])

    from data.database import db_session
    with db_session() as conn:
        conn.execute("DELETE FROM alpha_ideas WHERE id=?", (idea_id,))


def test_sweep_trials_raise_deflated_hurdle():
    """The QC6 n_trials query must include done optimizer_runs configs."""
    from data.database import init_db, db_session
    init_db()
    idea_id = 999912
    with db_session() as conn:
        conn.execute("DELETE FROM optimizer_runs WHERE idea_id=?", (idea_id,))
        conn.execute("DELETE FROM alpha_ideas WHERE id=?", (idea_id,))
        conn.execute(
            "INSERT INTO alpha_ideas (id, slug, title, ticker, stage, status) "
            "VALUES (?, 'opt-hurdle-test', 'OPT hurdle', '1155.KL', 'stage2', 'optimizing')",
            (idea_id,))
        conn.execute(
            "INSERT INTO optimizer_runs (idea_id, status, seed, n_configs) "
            "VALUES (?, 'done', 42, 300)", (idea_id,))
        base = conn.execute(
            "SELECT COUNT(DISTINCT idea_id) AS n FROM backtest_runs").fetchone()["n"] + 1
        sweep = conn.execute(
            "SELECT COALESCE(SUM(n_configs),0) AS n FROM optimizer_runs "
            "WHERE idea_id=? AND status='done'", (idea_id,)).fetchone()["n"]
    assert sweep == 300
    n_bars = 1000
    hurdle_plain = np.sqrt(2 * np.log(max(base, 2)) / n_bars)
    hurdle_swept = np.sqrt(2 * np.log(max(base + sweep, 2)) / n_bars)
    assert hurdle_swept > hurdle_plain
    with db_session() as conn:
        conn.execute("DELETE FROM optimizer_runs WHERE idea_id=?", (idea_id,))
        conn.execute("DELETE FROM alpha_ideas WHERE id=?", (idea_id,))


# ── sandbox optimize flag → queue ─────────────────────────────────────────────

def test_sandbox_optimize_flag_queues_run():
    from data.database import init_db, db_session
    init_db()
    from pipeline.sandbox import submit_sandbox_idea
    r = submit_sandbox_idea({
        "title": "SBX optimize zscore", "hypothesis": "z-score reversion family search over weeks",
        "ticker": "1155.KL",
        "factor_formula": "enter long when z-score(20) < -2, exit at z-score 0",
    }, optimize=True)
    assert r["ok"] is True and r.get("optimizing") is True
    assert r["status"] == "optimizing"
    with db_session() as conn:
        q = conn.execute(
            "SELECT status, n_configs FROM optimizer_runs WHERE idea_id=?",
            (r["idea_id"],)).fetchone()
        assert q["status"] == "queued" and q["n_configs"] == 300
        conn.execute("DELETE FROM optimizer_runs WHERE idea_id=?", (r["idea_id"],))
        conn.execute("DELETE FROM alpha_ideas WHERE id=?", (r["idea_id"],))
