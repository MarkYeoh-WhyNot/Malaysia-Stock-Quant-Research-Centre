"""Shared factor-sandbox submission (pipeline/sandbox.py)."""
import pytest

from data.database import db_session, init_db
from pipeline.sandbox import submit_sandbox_idea


@pytest.fixture(autouse=True)
def _clean():
    init_db()
    _cleanup()
    yield
    _cleanup()


def _cleanup():
    with db_session() as conn:
        conn.execute("DELETE FROM alpha_ideas WHERE title LIKE 'SBX %'")


def test_valid_idea_inserts_at_stage2_pending():
    r = submit_sandbox_idea({
        "title": "SBX MA cross", "hypothesis": "20/50 MA crossover on Maybank, hold weeks",
        "ticker": "1155.KL", "factor_formula": "sma(20) crosses above sma(50)",
    }, run_backtest=False, source="concierge")
    assert r["ok"] is True
    with db_session() as conn:
        row = conn.execute(
            "SELECT stage, status, screen_source FROM alpha_ideas WHERE id=?",
            (r["idea_id"],)).fetchone()
    assert row["stage"] == "stage2"
    assert row["status"] == "pending"       # daemon picks it up async
    assert row["screen_source"] == "concierge"


def test_short_selling_is_hard_blocked():
    r = submit_sandbox_idea({
        "title": "SBX short", "hypothesis": "short sell overvalued banks",
        "ticker": "1155.KL", "factor_formula": "go short when rsi above 70 for days",
    })
    assert r["ok"] is False
    assert "long-only" in r["error"].lower()


def test_bursa_refusal_wording_pinned():
    """Bursa refusal strings must stay byte-identical — the wording is now
    built from ALLOW_SHORT, and this pins the ALLOW_SHORT=False rendering."""
    blocked = submit_sandbox_idea({
        "title": "SBX pin blocked", "hypothesis": "short sell the weakest bank",
        "ticker": "1155.KL", "factor_formula": "short when rsi above 70 for days",
    })
    assert blocked["error"].endswith(
        "this system is long-only and trades daily bars "
        "(no short-selling, pairs, or intraday).")
    from pipeline.sandbox import _INFEASIBLE_HINT
    assert _INFEASIBLE_HINT == "short-selling, intraday, or unavailable-data reliance"


def test_intraday_is_hard_blocked():
    r = submit_sandbox_idea({
        "title": "SBX scalp", "hypothesis": "intraday scalping the open",
        "ticker": "1155.KL", "factor_formula": "buy dips and exit same day quickly",
    })
    assert r["ok"] is False
    assert "intraday" in r["error"].lower() or "long-only" in r["error"].lower()


def test_invalid_ticker_rejected():
    r = submit_sandbox_idea({
        "title": "SBX aapl", "hypothesis": "momentum", "ticker": "AAPL",
        "factor_formula": "sma(20) crosses above sma(50)",
    })
    assert r["ok"] is False
    assert ".KL" in r["error"]


def test_duplicate_not_resubmitted():
    brief = {"title": "SBX dup", "hypothesis": "weekly momentum on Tenaga, hold weeks",
             "ticker": "5347.KL", "factor_formula": "close crosses above sma(50)"}
    first = submit_sandbox_idea(brief)
    assert first["ok"] is True
    second = submit_sandbox_idea({**brief, "title": "SBX dup reworded"})
    assert second["ok"] is False
    assert second["duplicate_of"] == first["idea_id"]
