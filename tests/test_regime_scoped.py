"""Regime-scoped candidate type (Phase 4 of the ideation-loop wiring).

Pins:
  * the UNSCOPED regime gate stays byte-identical (regimes_positive >= 2) —
    the new candidate type is an addition, never a relaxation;
  * regime_filter validation (non-empty PROPER subset of the terciles);
  * the position mask is ex-ante (prefix-stable, 1-bar lagged) and actually
    forces flat outside the declared terciles;
  * perturb_tree preserves regime_filter (robustness gate must compare
    scoped-vs-scoped);
  * submit_regime_scoped_idea charges the >= 6-config DOF row;
  * canonical_signature distinguishes a scoped tree from its unscoped
    sibling AND from a differently-scoped variant (2026-07-12 bug: the
    signature ignored regime_filter entirely, so submit_regime_scoped_idea's
    dedup check treated every scoped candidate as a duplicate of its
    unscoped counterpart whenever one already existed).
"""
import numpy as np
import pandas as pd
import pytest

from agents.backtest_engineer.gates import regime_gate_decision
from agents.backtest_engineer.signal_dsl import (
    _regime_mask, canonical_signature, perturb_tree, signal_from_dsl, validate,
)

_BASE = {"entry": {"leaf": "momentum", "period": 20, "min_return": 0.01},
         "exit": {"leaf": "reversal", "period": 5, "max_return": -0.02}}


def _scoped(active):
    t = dict(_BASE)
    t["regime_filter"] = {"type": "vol_tercile", "active": active}
    return t


# ── validation ───────────────────────────────────────────────────────────────

def test_validate_accepts_proper_subsets_and_rejects_degenerate():
    assert validate(_scoped(["high_vol"])) == []
    assert validate(_scoped(["low_vol", "mid_vol"])) == []
    for bad in ([], ["low_vol", "mid_vol", "high_vol"], ["sideways"], "high_vol"):
        t = dict(_BASE)
        t["regime_filter"] = {"type": "vol_tercile", "active": bad}
        assert validate(t), f"should reject active={bad!r}"
    t = dict(_BASE)
    t["regime_filter"] = {"type": "macro", "active": ["high_vol"]}
    assert validate(t)
    assert validate(_BASE) == []  # unscoped trees unaffected


# ── mask semantics ───────────────────────────────────────────────────────────

def _two_regime_df(n=1200, seed=3):
    """Calm first half, violent second half — unambiguous terciles."""
    rng = np.random.RandomState(seed)
    rets = np.concatenate([rng.normal(0, 0.003, n // 2),
                           rng.normal(0, 0.03, n - n // 2)])
    close = pd.Series(100 * np.cumprod(1 + rets),
                      index=pd.date_range("2020-01-01", periods=n, freq="D"))
    return pd.DataFrame({"open": close, "high": close, "low": close,
                         "close": close, "volume": 1e6})


def test_mask_selects_declared_tercile_and_forces_flat():
    df = _two_regime_df()
    high = _regime_mask(df, ["high_vol"])
    low = _regime_mask(df, ["low_vol"])
    # Ex-ante terciles are relative to history SO FAR: inside the calm half
    # its own top-third still labels high_vol (~1/3), but once the violent
    # half arrives it dominates the high tercile and the calm past never
    # re-labels. Violent tail ≫ calm interior; low_vol vanishes in violence.
    assert high.iloc[-200:].mean() > 0.8
    assert high.iloc[300:600].mean() < 0.5
    assert high.iloc[-200:].mean() > high.iloc[300:600].mean() + 0.4
    assert low.iloc[-200:].mean() < 0.1

    # a scoped always-in-market signal is flat exactly off-mask
    t = {"entry": {"leaf": "momentum", "period": 5, "min_return": -1.0},
         "regime_filter": {"type": "vol_tercile", "active": ["high_vol"]}}
    sig = signal_from_dsl(df, t)
    assert (sig[~high] == 0).all()
    assert (sig[high] == 1).all()


def test_mask_is_prefix_stable_no_lookahead():
    df = _two_regime_df()
    full = _regime_mask(df, ["high_vol"])
    truncated = _regime_mask(df.iloc[:800], ["high_vol"])
    # expanding quantiles + shift(1) depend only on the past: the first 800
    # bars must not change when the future arrives
    pd.testing.assert_series_equal(full.iloc[:800], truncated)


def test_mask_warmup_is_flat():
    df = _two_regime_df()
    mask = _regime_mask(df, ["low_vol", "high_vol"])
    assert not mask.iloc[:252].any()


# ── perturbation keeps the scope ─────────────────────────────────────────────

def test_perturb_tree_preserves_regime_filter():
    out = perturb_tree(_scoped(["high_vol"]), np.random.RandomState(0))
    assert out["regime_filter"] == {"type": "vol_tercile", "active": ["high_vol"]}


# ── signature distinguishes scoped from unscoped and from each other ───────

def test_signature_distinguishes_regime_scoped_from_unscoped():
    sig_unscoped = canonical_signature(_BASE, "BTC/USDT")
    sig_high = canonical_signature(_scoped(["high_vol"]), "BTC/USDT")
    sig_low = canonical_signature(_scoped(["low_vol"]), "BTC/USDT")
    assert len({sig_unscoped, sig_high, sig_low}) == 3


def test_signature_unchanged_for_trees_without_regime_filter():
    # No regression: a plain tree's hash must not depend on this fix at all.
    assert (canonical_signature(_BASE, "BTC/USDT")
            == canonical_signature(dict(_BASE), "BTC/USDT"))


# ── gate decision ────────────────────────────────────────────────────────────

def test_unscoped_gate_pinned_byte_identical():
    for n_pos, expected in ((0, False), (1, False), (2, True), (3, True)):
        ok, _note, scoped = regime_gate_decision(
            {"signal_type": "dsl", "dsl": _BASE}, n_pos, None)
        assert scoped is False
        assert ok is expected
    # non-DSL params (fundamental_screen etc.) take the unscoped branch too
    ok, _n, scoped = regime_gate_decision({"signal_type": "fundamental_screen"}, 2, None)
    assert ok and not scoped


def test_scoped_gate_requires_every_declared_tercile_positive():
    params = {"signal_type": "dsl", "dsl": _scoped(["high_vol", "mid_vol"])}
    sharpes_ok = {"sharpe_low_vol": -1.0, "sharpe_mid_vol": 0.4, "sharpe_high_vol": 1.1}
    sharpes_bad = {"sharpe_low_vol": 2.0, "sharpe_mid_vol": -0.1, "sharpe_high_vol": 1.1}
    ok, _n, scoped = regime_gate_decision(params, 1, sharpes_ok)
    assert scoped and ok  # regimes_positive=1 would fail unscoped; scoped passes
    ok, note, _ = regime_gate_decision(params, 2, sharpes_bad)
    assert not ok and "declared tercile" in note
    ok, _n, _ = regime_gate_decision(params, 3, None)  # missing sharpes → fail closed
    assert not ok


# ── submission helper ────────────────────────────────────────────────────────

def test_submit_charges_dof_and_dedupes():
    from data.database import db_session, init_db
    from pipeline.regime_candidates import submit_regime_scoped_idea
    init_db()

    def _purge():
        with db_session() as conn:
            for r in conn.execute(
                    "SELECT id FROM alpha_ideas WHERE title LIKE 'test-rg-%'"):
                conn.execute("DELETE FROM optimizer_runs WHERE idea_id=?", (r["id"],))
                conn.execute("DELETE FROM alpha_ideas WHERE id=?", (r["id"],))

    _purge()
    try:
        res = submit_regime_scoped_idea(
            _BASE, ["high_vol"], title="test-rg-hv", hypothesis="test",
            ticker="BTC/USDT", timeframe="1d", extra_trials=10)
        assert res["ok"] and res["n_configs"] == 16
        with db_session() as conn:
            row = conn.execute(
                "SELECT n_configs, winner_json FROM optimizer_runs WHERE idea_id=?",
                (res["idea_id"],)).fetchone()
        assert row["n_configs"] == 16
        import json
        assert json.loads(row["winner_json"])["dsl"]["regime_filter"]["active"] == ["high_vol"]

        dup = submit_regime_scoped_idea(
            _BASE, ["high_vol"], title="test-rg-hv2", hypothesis="test",
            ticker="BTC/USDT", timeframe="1d")
        assert not dup["ok"] and "duplicate" in dup["error"]

        bad = submit_regime_scoped_idea(
            _BASE, ["low_vol", "mid_vol", "high_vol"], title="test-rg-all",
            hypothesis="test", ticker="BTC/USDT", timeframe="1d")
        assert not bad["ok"]
    finally:
        _purge()
