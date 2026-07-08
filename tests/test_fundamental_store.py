"""Phase 5.2: fundamental feature store."""
import pytest

from data.database import db_session, init_db
from data.klse.fundamental_store import persist_fundamentals, latest_fundamentals

TICKER = "1155.KL"


@pytest.fixture()
def clean():
    init_db()
    _cleanup()
    yield
    _cleanup()


def _cleanup():
    with db_session() as conn:
        conn.execute("DELETE FROM fundamental_features WHERE ticker=?", (TICKER,))


def test_persist_and_read_back(clean):
    row_id = persist_fundamentals(TICKER, {"roe": 12.5, "pe": 10.2, "dy": 4.1},
                                  as_of_date="2026-06-01")
    assert row_id is not None
    latest = latest_fundamentals(TICKER)
    assert latest["roe"] == 12.5
    assert latest["pe"] == 10.2
    assert latest["dividend_yield"] == 4.1  # aliased from "dy"


def test_upsert_overwrites_same_date(clean):
    persist_fundamentals(TICKER, {"roe": 10.0}, as_of_date="2026-06-01")
    persist_fundamentals(TICKER, {"roe": 15.0}, as_of_date="2026-06-01")
    with db_session() as conn:
        n = conn.execute(
            "SELECT COUNT(*) n FROM fundamental_features WHERE ticker=?", (TICKER,)
        ).fetchone()["n"]
    assert n == 1
    assert latest_fundamentals(TICKER)["roe"] == 15.0


def test_unknown_fields_ignored_no_crash(clean):
    row_id = persist_fundamentals(TICKER, {"totally_unknown_field": 99},
                                  as_of_date="2026-06-01")
    assert row_id is None  # nothing recognised → nothing written


def test_missing_returns_none():
    assert latest_fundamentals("9999.KL") is None
