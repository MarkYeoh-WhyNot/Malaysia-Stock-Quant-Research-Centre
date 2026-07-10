"""Pins for the deterministic DSL -> Pine Script translator
(agents/backtest_engineer/pinescript_gen.py).
"""
import re

from agents.backtest_engineer.pinescript_gen import generate_pinescript


def _assigned_var(code: str, rhs_substr: str) -> str:
    """Find the var name assigned to a line whose right-hand side contains
    rhs_substr — lets tests check relationships without hardcoding the
    translator's internal numbering scheme."""
    m = re.search(r"(\w+) = [^\n]*" + re.escape(rhs_substr), code)
    assert m, f"no assignment found containing {rhs_substr!r} in:\n{code}"
    return m.group(1)


def _gen(dsl, allow_short=True, title="Test Strategy", timeframe="1d"):
    r = generate_pinescript(dsl, title, timeframe, allow_short)
    assert r["ok"] is True, r
    return r["code"]


def test_rsi_leaf():
    code = _gen({"entry": {"leaf": "rsi", "period": 14, "below": 30},
                "exit": {"leaf": "rsi", "period": 14, "above": 70}})
    assert "ta.rsi(close, 14)" in code
    assert "< 30.0" in code and "> 70.0" in code
    # same (leaf, period) reused in entry+exit -> ONE ta.rsi line
    assert code.count("ta.rsi(close, 14)") == 1


def test_sma_cross_leaf():
    code = _gen({"entry": {"leaf": "sma_cross", "fast": 20, "slow": 50,
                           "direction": "above"}})
    assert "ta.sma(close, 20)" in code and "ta.sma(close, 50)" in code
    fvar = _assigned_var(code, "ta.sma(close, 20)")
    svar = _assigned_var(code, "ta.sma(close, 50)")
    assert f"{fvar} > {svar}" in code


def test_ema_cross_leaf():
    code = _gen({"entry": {"leaf": "ema_cross", "fast": 9, "slow": 21,
                           "direction": "below"}})
    assert "ta.ema(close, 9)" in code and "ta.ema(close, 21)" in code
    fvar = _assigned_var(code, "ta.ema(close, 9)")
    svar = _assigned_var(code, "ta.ema(close, 21)")
    assert f"{fvar} < {svar}" in code


def test_momentum_and_reversal_leaves():
    code = _gen({"entry": {"leaf": "momentum", "period": 20, "min_return": 0.03},
                "exit": {"leaf": "reversal", "period": 5, "max_return": -0.02}})
    assert "close / close[20] - 1" in code
    assert "> 0.03" in code
    assert "close / close[5] - 1" in code
    assert "< -0.02" in code


def test_bollinger_leaf():
    code = _gen({"entry": {"leaf": "bollinger", "period": 20, "std": 2.0,
                           "band": "below_lower"}})
    assert "ta.sma(close, 20)" in code
    assert "ta.stdev(close, 20)" in code
    basis = _assigned_var(code, "ta.sma(close, 20)")
    lower_var = _assigned_var(code, f"{basis} -")
    assert f"close < {lower_var}" in code


def test_macd_leaf():
    code = _gen({"entry": {"leaf": "macd", "fast": 12, "slow": 26, "signal": 9,
                           "condition": "bullish"}})
    assert "ta.macd(close, 12, 26, 9)" in code
    m = re.search(r"\[(\w+), (\w+), _\] = ta\.macd", code)
    assert m, code
    macd_var, signal_var = m.group(1), m.group(2)
    assert f"{macd_var} > {signal_var}" in code


def test_volume_ratio_leaf():
    code = _gen({"entry": {"leaf": "volume_ratio", "period": 20, "min_ratio": 1.5}})
    assert "ta.sma(volume, 20)" in code
    assert "volume > 1.5 * volMA1" in code


def test_gap_leaf():
    code = _gen({"entry": {"leaf": "gap", "min_pct": 0.02, "direction": "up"}})
    assert "(open - close[1]) / close[1]" in code
    assert "> 0.02" in code


def test_rolling_rank_leaf():
    code = _gen({"entry": {"leaf": "rolling_rank", "formation": 126, "skip": 10,
                           "window": 252, "min_pct": 0.8}})
    assert "close[10] / close[136] - 1" in code
    assert "ta.percentrank(formRet1, 252)" in code
    assert ">= 80.0" in code


def test_zscore_leaf():
    code = _gen({"entry": {"leaf": "zscore", "period": 20, "below": -2.0},
                "short_entry": {"leaf": "zscore", "period": 20, "above": 2.0},
                "short_exit": {"leaf": "zscore", "period": 20, "below": 0.0}})
    assert "ta.sma(close, 20)" in code and "ta.stdev(close, 20)" in code
    assert code.count("ta.sma(close, 20)") == 1   # dedup across entry+short legs


def test_and_or_not_combinators():
    tree = {"entry": {"op": "AND", "children": [
                {"leaf": "volume_ratio", "period": 20, "min_ratio": 1.5},
                {"leaf": "gap", "direction": "up", "min_pct": 0.02}]},
            "exit": {"op": "NOT", "child":
                    {"leaf": "rsi", "period": 14, "below": 50}}}
    code = _gen(tree)
    assert " and " in code
    assert "(not (" in code


def test_or_combinator():
    tree = {"entry": {"op": "OR", "children": [
                {"leaf": "rsi", "period": 14, "below": 30},
                {"leaf": "rsi", "period": 7, "below": 20}]}}
    code = _gen(tree)
    assert " or " in code


def test_short_tie_break_and_gating():
    tree = {"entry": {"leaf": "rsi", "period": 14, "below": 30},
            "exit": {"leaf": "rsi", "period": 14, "above": 70},
            "short_entry": {"leaf": "rsi", "period": 14, "above": 70},
            "short_exit": {"leaf": "rsi", "period": 14, "below": 30}}
    code = _gen(tree, allow_short=True)
    assert "shortEntryCond = (" in code and "and not longEntryCond" in code
    assert 'strategy.entry("Short"' in code

    # Bursa (allow_short=False): short legs must be entirely absent
    code_lo = _gen(tree, allow_short=False)
    assert "shortEntryCond" not in code_lo
    assert 'strategy.entry("Short"' not in code_lo


def test_strategy_scaffold_present():
    code = _gen({"entry": {"leaf": "rsi", "period": 14, "below": 30}})
    assert code.startswith("//@version=5\n")
    assert "strategy(" in code
    assert 'strategy.entry("Long", strategy.long, when=longEntryCond)' in code


def test_unsupported_leaves_decline_with_reason():
    for leaf, extra in [
        ("div_days_to_ex", {"max_days": 5}),
        ("cpo_change", {"period": 5, "min_pct": 0.01}),
        ("funding_level", {"above": 0.0005}),
        ("funding_zscore", {"period": 30, "above": 2.0}),
    ]:
        node = {"leaf": leaf, **extra}
        r = generate_pinescript({"entry": node}, "t", "1d", True)
        assert r["ok"] is False, (leaf, r)
        assert leaf in r["reason"]


def test_entry_only_no_exit():
    """A tree with only entry (no exit) still produces a valid strategy —
    entry itself is the position regime, per signal_from_dsl."""
    code = _gen({"entry": {"leaf": "sma_cross", "fast": 20, "slow": 50}})
    assert 'strategy.entry("Long"' in code
    assert 'strategy.close("Long"' not in code
