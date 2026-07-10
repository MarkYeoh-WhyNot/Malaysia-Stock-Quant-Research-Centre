"""Arsenal v2: every technique entry (both markets) carries machine-validated
signature fields — family_id, strategy_shape, representability, example — and
every example passes the LIVE registries (signal_dsl.LEAVES, factors.FACTORS).

Validating against the live registries IS the drift detection: change a leaf
and its examples fail here; implement a leaf named in missing_leaves and the
disjointness test forces the arsenal entry to be updated the same day.
"""
import numpy as np
import pandas as pd
import pytest

from agents.backtest_engineer import signal_dsl
from agents.backtest_engineer.factors import FACTORS, validate_factor
from knowledge.ingestion.technique_library import BURSA_TECHNIQUE_LIBRARY
from knowledge.ingestion.crypto_techniques import CRYPTO_TECHNIQUE_LIBRARY

_V2_FIELDS = ("family_id", "strategy_shape", "representability", "example")
_SHAPES = {"dsl_tree", "cross_sectional_factor", "methodology",
           "unimplemented_concept"}

_ALL = ([("bursa", k, v) for k, v in BURSA_TECHNIQUE_LIBRARY.items()]
        + [("crypto", k, v) for k, v in CRYPTO_TECHNIQUE_LIBRARY.items()])
_IDS = [f"{m}:{k}" for m, k, _ in _ALL]


def _synth_df(n=400, seed=0):
    """One frame carrying EVERY column any leaf can require."""
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    close = pd.Series(100 * np.cumprod(1 + rng.randn(n) * 0.01), index=idx)
    return pd.DataFrame({
        "close": close,
        "open": close.shift(1).fillna(close.iloc[0]),
        "volume": rng.randint(1_000_000, 3_000_000, n).astype(float),
        "dividends": 0.0,
        "cpo_close": pd.Series(4000 + rng.randn(n).cumsum() * 10, index=idx),
        "funding_rate": pd.Series(rng.randn(n) * 1e-4, index=idx),
    })


@pytest.mark.parametrize("market,key,tech", _ALL, ids=_IDS)
def test_entry_has_v2_fields(market, key, tech):
    for field in _V2_FIELDS:
        assert field in tech, f"{market}:{key} missing {field}"
    assert isinstance(tech["family_id"], str) and tech["family_id"]
    assert tech["strategy_shape"] in _SHAPES
    rep = tech["representability"]
    for f in ("is_representable", "representation_type", "required_leaves",
              "required_factor", "missing_leaves"):
        assert f in rep, f"{market}:{key} representability missing {f}"


@pytest.mark.parametrize("market,key,tech", _ALL, ids=_IDS)
def test_shape_consistent_with_example_and_representability(market, key, tech):
    shape, ex, rep = tech["strategy_shape"], tech["example"], tech["representability"]
    if shape == "dsl_tree":
        assert "dsl" in ex and rep["is_representable"] is True
        assert rep["representation_type"] == "dsl_tree"
        assert rep["required_leaves"], f"{market}:{key} dsl_tree needs required_leaves"
    elif shape == "cross_sectional_factor":
        assert "factor_spec" in ex and rep["is_representable"] is True
        assert rep["representation_type"] == "cross_sectional_factor"
        assert rep["required_factor"], f"{market}:{key} needs required_factor"
    else:  # methodology / unimplemented_concept — honest no-example
        assert "none" in ex and rep["is_representable"] is False
        assert isinstance(ex["none"], str) and len(ex["none"]) > 10, \
            f"{market}:{key} needs an honest no-example reason"
        if shape == "unimplemented_concept":
            assert rep["missing_leaves"], \
                f"{market}:{key} unimplemented_concept must name whats missing"


@pytest.mark.parametrize("market,key,tech", _ALL, ids=_IDS)
def test_dsl_examples_validate_and_evaluate(market, key, tech):
    ex = tech["example"]
    if "dsl" not in ex:
        pytest.skip("no dsl example")
    tree = ex["dsl"]
    errors = signal_dsl.validate(tree)
    assert errors == [], f"{market}:{key} example fails live registry: {errors}"
    if market == "bursa":
        assert "short_entry" not in tree and "short_exit" not in tree, \
            f"{market}:{key} Bursa example must be long-only"
    df = _synth_df()
    for part in ("entry", "exit", "short_entry", "short_exit"):
        node = tree.get(part)
        if node is not None:
            out = signal_dsl.evaluate(df, node)
            assert out.dtype == bool and len(out) == len(df)


@pytest.mark.parametrize("market,key,tech", _ALL, ids=_IDS)
def test_factor_examples_validate(market, key, tech):
    ex = tech["example"]
    if "factor_spec" not in ex:
        pytest.skip("no factor example")
    spec = ex["factor_spec"]
    factor = spec["factor"]
    validate_factor(factor["name"], factor.get("params", {}))  # raises if bad
    assert spec["top_n"] >= 1
    assert spec["rebalance_bars"] >= 1
    if market == "bursa":
        assert spec["bottom_n"] == 0, \
            f"{market}:{key} Bursa factor example must be long-only (bottom_n=0)"
    assert factor["name"] == tech["representability"]["required_factor"]


@pytest.mark.parametrize("market,key,tech", _ALL, ids=_IDS)
def test_required_registrations_live_and_missing_actually_missing(market, key, tech):
    rep = tech["representability"]
    for leaf in rep["required_leaves"]:
        assert leaf in signal_dsl.LEAVES, \
            f"{market}:{key} requires unknown leaf {leaf!r}"
    if rep["required_factor"]:
        assert rep["required_factor"] in FACTORS, \
            f"{market}:{key} requires unknown factor {rep['required_factor']!r}"
    # the drift alarm: the day a "missing" leaf/factor gets implemented, this
    # fails and forces the arsenal entry (and its example) to be updated
    for missing in rep["missing_leaves"]:
        assert missing not in signal_dsl.LEAVES, \
            f"{market}:{key}: {missing!r} is now a live leaf — update the entry"
        assert missing not in FACTORS, \
            f"{market}:{key}: {missing!r} is now a live factor — update the entry"


def test_all_33_entries_present():
    assert len(BURSA_TECHNIQUE_LIBRARY) >= 21
    assert len(CRYPTO_TECHNIQUE_LIBRARY) >= 12
