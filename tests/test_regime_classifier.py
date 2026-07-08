"""Phase 5.3: macro/sector regime classifier (network-free — takes price
series as input rather than fetching)."""
import numpy as np
import pandas as pd
import pytest

from data.database import db_session, init_db
from agents.regime.regime_classifier import classify_regime, persist_macro_snapshot


def _series(trend):
    n = 40
    if trend == "up":
        vals = np.linspace(100, 130, n)
    elif trend == "down":
        vals = np.linspace(130, 100, n)
    else:
        vals = np.full(n, 100.0)
    return pd.Series(vals)


def test_cpo_uptrend_detected():
    r = classify_regime(cpo_prices=_series("up"))
    assert r["cpo_trend"] == "up"
    assert "commodity_upcycle" in r["regime_label"]


def test_cpo_downtrend_detected():
    r = classify_regime(cpo_prices=_series("down"))
    assert r["cpo_trend"] == "down"
    assert "commodity_downcycle" in r["regime_label"]


def test_flat_series_is_stable():
    r = classify_regime(cpo_prices=_series("flat"))
    assert r["cpo_trend"] == "flat"
    assert "commodity_stable" in r["regime_label"]


def test_no_data_is_unclassified():
    r = classify_regime()
    assert r["regime_label"] == "unclassified"
    assert r["opr_trend"] == "unknown"


def test_combines_opr_and_myr_when_provided():
    r = classify_regime(cpo_prices=_series("up"), opr_trend="rising", myr_trend="strong")
    assert "opr_rising" in r["regime_label"]
    assert "myr_strong" in r["regime_label"]


def test_persist_macro_snapshot_upserts():
    init_db()
    date = "2026-06-15"
    with db_session() as conn:
        conn.execute("DELETE FROM macro_features WHERE as_of_date=?", (date,))
    regime = classify_regime(cpo_prices=_series("up"))
    persist_macro_snapshot(date, cpo_price=4500.0, regime=regime)
    persist_macro_snapshot(date, cpo_price=4600.0, regime=regime)  # upsert, not duplicate
    with db_session() as conn:
        rows = conn.execute(
            "SELECT * FROM macro_features WHERE as_of_date=?", (date,)
        ).fetchall()
    assert len(rows) == 1
    assert rows[0]["cpo_price"] == 4600.0
    with db_session() as conn:
        conn.execute("DELETE FROM macro_features WHERE as_of_date=?", (date,))
