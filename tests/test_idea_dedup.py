"""save_idea semantic dedup (text-signature layer) and proxy lineage."""
import pytest

from data.database import db_session, init_db
from agents.researcher.strategy_researcher import StrategyResearcher

SLUG_PREFIX = "test-dedup-"


@pytest.fixture()
def researcher():
    init_db()
    _cleanup()
    sr = StrategyResearcher()
    # avoid daemon-log noise coupling in tests
    yield sr
    _cleanup()


def _cleanup():
    with db_session() as conn:
        conn.execute(
            "DELETE FROM alpha_ideas WHERE title LIKE 'TESTDEDUP%'")


def test_reworded_duplicate_not_saved(researcher):
    base = {
        "title": "TESTDEDUP Maybank golden cross",
        "hypothesis": "MA cross on Maybank",
        "ticker": "1155.KL",
        "factor_formula": "20-day SMA crosses above 50-day SMA",
    }
    id1 = researcher.save_idea(dict(base))
    reworded = dict(base)
    reworded["title"] = "TESTDEDUP Maybank 20/50 moving average crossover"
    id2 = researcher.save_idea(reworded)
    assert id1 == id2, "reworded duplicate must return the existing idea id"
    with db_session() as conn:
        n = conn.execute(
            "SELECT COUNT(*) AS n FROM alpha_ideas WHERE title LIKE 'TESTDEDUP%'"
        ).fetchone()["n"]
    assert n == 1


def test_different_ticker_is_not_duplicate(researcher):
    base = {
        "title": "TESTDEDUP golden cross A",
        "hypothesis": "MA cross",
        "ticker": "1155.KL",
        "factor_formula": "20-day SMA crosses above 50-day SMA",
    }
    other = dict(base)
    other["title"] = "TESTDEDUP golden cross B"
    other["ticker"] = "1023.KL"
    id1 = researcher.save_idea(base)
    id2 = researcher.save_idea(other)
    assert id1 != id2


def test_rejected_duplicate_can_be_resubmitted(researcher):
    base = {
        "title": "TESTDEDUP resubmit test",
        "hypothesis": "x",
        "ticker": "1155.KL",
        "factor_formula": "RSI below 30 with volume above average",
    }
    id1 = researcher.save_idea(dict(base))
    with db_session() as conn:
        conn.execute("UPDATE alpha_ideas SET status='rejected' WHERE id=?", (id1,))
    fresh = dict(base)
    fresh["title"] = "TESTDEDUP resubmit test v2"
    id2 = researcher.save_idea(fresh)
    assert id2 != id1, "dedup only blocks against LIVE (non-rejected) ideas"


def test_parent_idea_id_recorded(researcher):
    parent = researcher.save_idea({
        "title": "TESTDEDUP parent fundamental idea",
        "hypothesis": "ROE-based screen",
        "ticker": "1155.KL",
        "factor_formula": "high ROE with low PB",
    })
    child = researcher.save_idea({
        "title": "TESTDEDUP Price proxy: parent fundamental idea",
        "hypothesis": "price drawdown as value proxy",
        "ticker": "1155.KL",
        "factor_formula": "52-week drawdown above 30 percent",
        "parent_idea_id": parent,
    })
    with db_session() as conn:
        row = conn.execute(
            "SELECT parent_idea_id FROM alpha_ideas WHERE id=?", (child,)).fetchone()
    assert row["parent_idea_id"] == parent
