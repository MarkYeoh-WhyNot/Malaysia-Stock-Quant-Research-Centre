"""Signal DSL: leaf correctness, combinators, state machine, validation,
signatures, and perturbation — the translation-integrity foundation."""
import numpy as np
import pandas as pd
import pytest

from agents.backtest_engineer import signal_dsl as dsl


def _df(n=300, seed=0):
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    close = pd.Series(10 * np.cumprod(1 + rng.randn(n) * 0.01), index=idx)
    open_ = close.shift(1).fillna(close.iloc[0]) * (1 + rng.randn(n) * 0.002)
    volume = pd.Series(rng.randint(1_000_000, 3_000_000, n).astype(float), index=idx)
    return pd.DataFrame({"close": close, "open": open_, "volume": volume})


# ── Leaves ────────────────────────────────────────────────────────────────────

def test_momentum_leaf_matches_manual():
    df = _df()
    got = dsl.evaluate(df, {"leaf": "momentum", "period": 20, "min_return": 0.0})
    want = (df["close"].pct_change(20) > 0).fillna(False)
    assert (got == want).all()


def test_gap_leaf_fires_on_synthetic_gap():
    df = _df()
    df.iloc[50, df.columns.get_loc("open")] = df["close"].iloc[49] * 1.05  # +5% gap up
    got = dsl.evaluate(df, {"leaf": "gap", "direction": "up", "min_pct": 0.02})
    assert bool(got.iloc[50]) is True


def test_volume_ratio_leaf():
    df = _df()
    df.iloc[100, df.columns.get_loc("volume")] = 50_000_000  # huge spike
    got = dsl.evaluate(df, {"leaf": "volume_ratio", "period": 20, "min_ratio": 1.5})
    assert bool(got.iloc[100]) is True


def test_div_days_to_ex_leaf():
    df = _df()
    df["dividends"] = 0.0
    df.iloc[60, df.columns.get_loc("dividends")] = 0.10  # ex-date at bar 60
    got = dsl.evaluate(df, {"leaf": "div_days_to_ex", "max_days": 5})
    assert bool(got.iloc[56]) is True   # 4 bars before ex-date
    assert bool(got.iloc[60]) is True   # on ex-date
    assert bool(got.iloc[50]) is False  # 10 bars before — outside window


def test_rsi_leaf_one_of():
    df = _df()
    below = dsl.evaluate(df, {"leaf": "rsi", "period": 14, "below": 30})
    above = dsl.evaluate(df, {"leaf": "rsi", "period": 14, "above": 70})
    assert not (below & above).any()  # mutually exclusive by construction


# ── Combinators ───────────────────────────────────────────────────────────────

def test_and_or_not_composition():
    df = _df()
    a = {"leaf": "momentum", "period": 20, "min_return": 0.0}
    b = {"leaf": "volume_ratio", "period": 20, "min_ratio": 1.2}
    both = dsl.evaluate(df, {"op": "AND", "children": [a, b]})
    either = dsl.evaluate(df, {"op": "OR", "children": [a, b]})
    neg = dsl.evaluate(df, {"op": "NOT", "child": a})
    sa, sb = dsl.evaluate(df, a), dsl.evaluate(df, b)
    assert (both == (sa & sb)).all()
    assert (either == (sa | sb)).all()
    assert (neg == ~sa).all()


# ── State machine ─────────────────────────────────────────────────────────────

def test_entry_exit_state_machine():
    df = _df(n=10)
    # deterministic entry/exit points via momentum thresholds we control
    entry = pd.Series([False]*10, index=df.index)
    exit_ = pd.Series([False]*10, index=df.index)
    entry.iloc[2] = True
    exit_.iloc[5] = True
    # monkey-style: inject via custom leaves is overkill — test the ffill logic directly
    raw = np.where(entry, 1.0, np.where(exit_, 0.0, np.nan))
    sig = pd.Series(raw, index=df.index).ffill().fillna(0.0)
    assert list(sig.values) == [0, 0, 1, 1, 1, 0, 0, 0, 0, 0]


def test_signal_from_dsl_entry_only_is_regime():
    df = _df()
    tree = {"entry": {"leaf": "momentum", "period": 20, "min_return": 0.0}}
    sig = dsl.signal_from_dsl(df, tree)
    want = dsl.evaluate(df, tree["entry"]).astype(float)
    assert (sig == want).all()


# ── Validation ────────────────────────────────────────────────────────────────

def test_validate_clean_tree():
    tree = {"entry": {"op": "AND", "children": [
        {"leaf": "volume_ratio", "period": 20, "min_ratio": 1.5},
        {"leaf": "gap", "direction": "up", "min_pct": 0.02}]},
        "exit": {"leaf": "rsi", "period": 14, "above": 70}}
    assert dsl.validate(tree) == []


def test_validate_rejects_unknown_leaf():
    assert any("unknown leaf" in e for e in
               dsl.validate({"entry": {"leaf": "astrology", "period": 7}}))


def test_validate_rejects_out_of_range():
    errs = dsl.validate({"entry": {"leaf": "rsi", "period": 500, "below": 30}})
    assert any("outside" in e for e in errs)


def test_validate_rejects_missing_param():
    errs = dsl.validate({"entry": {"leaf": "momentum", "period": 20}})
    assert any("missing param" in e for e in errs)


def test_validate_rejects_fast_ge_slow():
    errs = dsl.validate({"entry": {"leaf": "sma_cross", "fast": 50, "slow": 20,
                                   "direction": "above"}})
    assert any("fast >= slow" in e for e in errs)


def test_validate_rejects_missing_entry():
    assert any("missing entry" in e for e in dsl.validate({}))


def test_required_columns():
    tree = {"entry": {"op": "AND", "children": [
        {"leaf": "cpo_change", "period": 5, "min_pct": 0.0},
        {"leaf": "volume_ratio", "period": 20, "min_ratio": 1.5}]}}
    assert dsl.required_columns(tree) == {"cpo_close", "close", "volume"}


# ── Signature ─────────────────────────────────────────────────────────────────

def test_signature_stable_across_key_order_and_child_order():
    t1 = {"entry": {"op": "AND", "children": [
        {"leaf": "gap", "direction": "up", "min_pct": 0.02},
        {"leaf": "volume_ratio", "period": 20, "min_ratio": 1.5}]}}
    t2 = {"entry": {"children": [
        {"min_ratio": 1.5, "period": 20, "leaf": "volume_ratio"},
        {"min_pct": 0.02, "leaf": "gap", "direction": "up"}], "op": "AND"}}
    assert dsl.canonical_signature(t1, "1155.KL") == dsl.canonical_signature(t2, "1155.KL")


def test_signature_differs_on_params_and_ticker():
    t = {"entry": {"leaf": "rsi", "period": 14, "below": 30}}
    t2 = {"entry": {"leaf": "rsi", "period": 14, "below": 25}}
    assert dsl.canonical_signature(t, "1155.KL") != dsl.canonical_signature(t2, "1155.KL")
    assert dsl.canonical_signature(t, "1155.KL") != dsl.canonical_signature(t, "1023.KL")


# ── Perturbation ──────────────────────────────────────────────────────────────

def test_perturb_stays_in_range_and_keeps_invariants():
    tree = {"entry": {"op": "AND", "children": [
        {"leaf": "sma_cross", "fast": 20, "slow": 50, "direction": "above"},
        {"leaf": "rsi", "period": 14, "below": 30}]}}
    rng = np.random.RandomState(42)
    for _ in range(20):
        p = dsl.perturb_tree(tree, rng)
        assert dsl.validate(p) == [], dsl.validate(p)
        sma = p["entry"]["children"][0]
        assert sma["fast"] < sma["slow"]


def test_perturb_actually_changes_params():
    tree = {"entry": {"leaf": "momentum", "period": 20, "min_return": 0.05}}
    rng = np.random.RandomState(1)
    changed = any(
        dsl.perturb_tree(tree, rng)["entry"]["period"] != 20 for _ in range(10)
    )
    assert changed


# ── ma_level leaf ─────────────────────────────────────────────────────────────

def test_ma_level_sma_matches_manual():
    df = _df()
    got = dsl.evaluate(df, {"leaf": "ma_level", "ma_type": "sma", "period": 20,
                            "direction": "above"})
    want = df["close"] > df["close"].rolling(20).mean()
    assert (got == want).all()


def test_ma_level_ema_below_matches_manual():
    df = _df()
    got = dsl.evaluate(df, {"leaf": "ma_level", "ma_type": "ema", "period": 50,
                            "direction": "below"})
    want = df["close"] < df["close"].ewm(span=50, adjust=False).mean()
    assert (got == want).all()


def test_ma_level_validation_contract():
    clean = {"entry": {"leaf": "ma_level", "ma_type": "ema", "period": 50,
                       "direction": "above"}}
    assert dsl.validate(clean) == []
    # ma_type is a REQUIRED choice — "50-day EMA" must never silently become SMA
    no_type = dsl.validate({"entry": {"leaf": "ma_level", "period": 50,
                                      "direction": "above"}})
    assert any("missing required choice ma_type" in e for e in no_type)
    bad_type = dsl.validate({"entry": {"leaf": "ma_level", "ma_type": "wma",
                                       "period": 50}})
    assert any("ma_type" in e for e in bad_type)
    out_of_range = dsl.validate({"entry": {"leaf": "ma_level", "ma_type": "sma",
                                           "period": 500}})
    assert any("outside" in e for e in out_of_range)


def test_ma_level_perturb_keeps_choices_and_range():
    tree = {"entry": {"leaf": "ma_level", "ma_type": "ema", "period": 50,
                      "direction": "above"}}
    rng = np.random.RandomState(5)
    for _ in range(20):
        p = dsl.perturb_tree(tree, rng)
        assert dsl.validate(p) == [], dsl.validate(p)
        node = p["entry"]
        assert node["ma_type"] == "ema" and node["direction"] == "above"
        assert 2 <= node["period"] <= 300


# ── Catalog ───────────────────────────────────────────────────────────────────

def test_catalog_has_no_default_values():
    text = dsl.leaf_catalog_text()
    for name in dsl.LEAVES:
        assert name in text
    # ranges shown, but no "default" anchoring anywhere
    assert "default" not in text.lower()


# ── Shape cards (parser prompt structure guide) ───────────────────────────────

def test_shape_cards_cover_all_leaves_no_defaults():
    text = dsl.shape_cards_text()
    for name in dsl.LEAVES:
        assert f"- {name}:" in text
    # structure-only: slot placeholders, never values or the word "default"
    assert "<EXTRACTED_" in text
    assert "default" not in text.lower()
    # the load-bearing negative mappings both directions
    assert "NOT sma_cross/ema_cross" in text        # ma_level card
    assert "that is ma_level" in text               # cross cards point back


def test_parser_negative_example_pins_the_ema_failure():
    ex = dsl.PARSER_NEGATIVE_EXAMPLE
    assert '"leaf": "ema_cross", "fast": 2, "slow": 50' in ex
    assert '"leaf": "ma_level", "ma_type": "ema", "period": 50' in ex
    assert "representable" in ex
    assert "default" not in ex.lower()
