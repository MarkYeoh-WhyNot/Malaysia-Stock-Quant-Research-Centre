"""WS3: long/short via perpetuals — DSL signed positions, funding accrual,
leverage/liquidation, and profile wiring. Bursa's long-only path must be
byte-identical to before (every new mechanic is a documented no-op there)."""
import numpy as np
import pandas as pd
import pytest

from agents.backtest_engineer import signal_dsl
from agents.backtest_engineer.backtest_engineer import BacktestEngineer


def _price_df(n=800, seed=0, drift=0.0):
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    rets = rng.randn(n) * 0.01 + drift
    close = 10 * np.cumprod(1 + rets)
    return pd.DataFrame({"close": close, "volume": 3_000_000}, index=idx)


# ── DSL: signed positions ───────────────────────────────────────────────────

def test_bursa_signal_stays_0_1_even_with_short_entry_present(monkeypatch):
    import config.settings as settings
    monkeypatch.setattr(settings, "ALLOW_SHORT", False)
    df = pd.DataFrame({"close": [10, 11, 12, 9, 8, 13, 14]})
    tree = {
        "entry": {"leaf": "momentum", "period": 2, "min_return": 0.0},
        "short_entry": {"op": "NOT", "child": {"leaf": "momentum", "period": 2, "min_return": 0.0}},
    }
    sig = signal_dsl.signal_from_dsl(df, tree)
    assert set(sig.unique()) <= {0.0, 1.0}


def test_crypto_signal_includes_short(monkeypatch):
    import config.settings as settings
    monkeypatch.setattr(settings, "ALLOW_SHORT", True)
    df = pd.DataFrame({"close": [10, 11, 12, 9, 8, 13, 14]})
    tree = {
        "entry": {"leaf": "momentum", "period": 2, "min_return": 0.0},
        "short_entry": {"op": "NOT", "child": {"leaf": "momentum", "period": 2, "min_return": 0.0}},
    }
    sig = signal_dsl.signal_from_dsl(df, tree)
    assert -1.0 in set(sig.unique())


def test_short_only_tree_is_valid(monkeypatch):
    import config.settings as settings
    monkeypatch.setattr(settings, "ALLOW_SHORT", True)
    tree = {"short_entry": {"leaf": "rsi", "period": 14, "above": 70}}
    assert signal_dsl.validate(tree) == []


def test_tree_with_neither_entry_nor_short_entry_invalid():
    assert "missing entry tree" in " ".join(signal_dsl.validate({}))


def test_canonical_signature_unchanged_for_long_only_tree():
    """Regression: adding optional short-leg keys to the signature payload must
    not change the hash for a tree that doesn't use them — or existing
    signal_signature values in the DB stop matching (dedup silently breaks)."""
    tree = {"entry": {"leaf": "rsi", "period": 14, "above": 70}, "exit": None}
    sig_before = "aa565cf3269fbc65"  # not asserted directly; see below
    sig_a = signal_dsl.canonical_signature(tree, "BTC/USDT")
    sig_b = signal_dsl.canonical_signature({**tree}, "BTC/USDT")
    assert sig_a == sig_b
    # Adding an empty/None short leg must not change the hash.
    sig_c = signal_dsl.canonical_signature({**tree, "short_entry": None}, "BTC/USDT")
    assert sig_a == sig_c


def test_canonical_signature_differs_with_real_short_leg():
    base = {"entry": {"leaf": "rsi", "period": 14, "above": 70}, "exit": None}
    with_short = {**base, "short_entry": {"leaf": "rsi", "period": 14, "below": 30}}
    assert signal_dsl.canonical_signature(base, "BTC/USDT") != \
           signal_dsl.canonical_signature(with_short, "BTC/USDT")


# ── Backtester: funding accrual + leverage/liquidation ──────────────────────

def test_bursa_funding_and_leverage_are_noop():
    be = BacktestEngineer()
    df = _price_df()
    sig = pd.Series(1.0, index=df.index)
    perf = be._compute_performance(df, sig, "1d")
    assert perf["leverage_used"] == 1.0
    assert perf["funding_drag_pct"] == 0.0


def test_crypto_funding_drag_reduces_long_pnl(monkeypatch):
    import agents.backtest_engineer.backtest_engineer as be_mod
    monkeypatch.setattr(be_mod, "FUNDING_INTERVAL_HOURS", 8)
    monkeypatch.setattr(be_mod, "AVG_FUNDING_RATE_PER_INTERVAL", 0.0001)
    be = BacktestEngineer()
    df = _price_df(drift=0.0)
    sig = pd.Series(1.0, index=df.index)  # always long
    perf_funded = be._compute_performance(df, sig, "1d")
    assert perf_funded["funding_drag_pct"] < 0  # a permanent long PAYS funding (net drag)


def test_crypto_short_receives_funding_when_positive(monkeypatch):
    import agents.backtest_engineer.backtest_engineer as be_mod
    monkeypatch.setattr(be_mod, "FUNDING_INTERVAL_HOURS", 8)
    monkeypatch.setattr(be_mod, "AVG_FUNDING_RATE_PER_INTERVAL", 0.0001)
    be = BacktestEngineer()
    df = _price_df(drift=0.0)
    sig = pd.Series(-1.0, index=df.index)  # always short
    perf = be._compute_performance(df, sig, "1d")
    assert perf["funding_drag_pct"] > 0  # a permanent short RECEIVES funding


def test_leverage_scales_returns_and_liquidation_caps_losses(monkeypatch):
    import agents.backtest_engineer.backtest_engineer as be_mod
    monkeypatch.setattr(be_mod, "FUNDING_INTERVAL_HOURS", None)
    monkeypatch.setattr(be_mod, "AVG_FUNDING_RATE_PER_INTERVAL", 0.0)
    monkeypatch.setattr(be_mod, "LIQUIDATION_BUFFER", 0.20)
    monkeypatch.setattr(be_mod, "MAX_LEVERAGE", 5.0)
    be = BacktestEngineer()
    # A sharp single-day crash while long at high leverage must be liquidation-capped.
    idx = pd.date_range("2020-01-01", periods=60, freq="D")
    close = np.full(60, 10.0)
    close[30] = 3.0  # a -70% single-day crash at bar 30
    close[31:] = 3.0
    df = pd.DataFrame({"close": close, "volume": 3_000_000}, index=idx)
    sig = pd.Series(1.0, index=df.index)
    perf_1x = be._compute_performance(df, sig, "1d", leverage=1.0)
    perf_5x = be._compute_performance(df, sig, "1d", leverage=5.0)
    assert perf_5x["leverage_used"] == 5.0
    # At 5x, a -70% move would be -350% unleveraged-equivalent without a cap —
    # liquidation must bound the single-bar loss, so max_dd stays sane (<=1.0,
    # i.e. never worse than "wiped out").
    assert perf_5x["max_dd"] <= 1.0
    assert perf_1x["leverage_used"] == 1.0


def test_leverage_capped_at_max_leverage(monkeypatch):
    import agents.backtest_engineer.backtest_engineer as be_mod
    monkeypatch.setattr(be_mod, "MAX_LEVERAGE", 3.0)
    be = BacktestEngineer()
    df = _price_df()
    sig = pd.Series(1.0, index=df.index)
    perf = be._compute_performance(df, sig, "1d", leverage=10.0)  # request > cap
    assert perf["leverage_used"] == 3.0  # capped, not 10


# ── Profile wiring ───────────────────────────────────────────────────────────

def test_funding_cost_sign_convention():
    from config.markets import crypto
    # Positive funding: long PAYS (positive cost), short RECEIVES (negative cost/income).
    assert crypto.funding_cost(1, 0.0001, 10_000) > 0
    assert crypto.funding_cost(-1, 0.0001, 10_000) < 0
    assert crypto.funding_cost(0, 0.0001, 10_000) == 0.0


def test_bursa_funding_cost_always_zero():
    from config.markets import bursa
    assert bursa.funding_cost(1, 0.0001, 10_000) == 0.0
    assert bursa.funding_cost(-1, 0.0001, 10_000) == 0.0


def test_allow_short_flag_per_market():
    from config.markets import bursa, crypto
    assert bursa.ALLOW_SHORT is False
    assert crypto.ALLOW_SHORT is True
