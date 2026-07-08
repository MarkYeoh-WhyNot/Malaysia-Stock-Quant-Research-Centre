"""Phase 5.5: strategy_cemetery revival conditions + similarity check."""
import pytest

from data.database import db_session, init_db
from knowledge.ingestion.rejection_memory import RejectionMemory

SENTINEL = 900_201


@pytest.fixture()
def clean():
    init_db()
    _cleanup()
    yield
    _cleanup()


def _cleanup():
    with db_session() as conn:
        conn.execute("DELETE FROM strategy_cemetery WHERE idea_id=?", (SENTINEL,))
        conn.execute("DELETE FROM rejection_patterns WHERE example_title LIKE 'Banking momentum%'")
        conn.execute("DELETE FROM alpha_ideas WHERE id=?", (SENTINEL,))


def test_record_rejection_populates_cemetery_with_revival_conditions(clean):
    with db_session() as conn:
        conn.execute(
            "INSERT INTO alpha_ideas (id, slug, title, hypothesis, ticker, stage, status) "
            "VALUES (?, 'cemetery-test', 'Banking momentum rotation', "
            "'20-day momentum on banks', '1155.KL', 'gate0', 'pending')",
            (SENTINEL,))

    RejectionMemory().record_rejection(SENTINEL, "overfitting: too many parameters", "gate0")

    with db_session() as conn:
        row = conn.execute(
            "SELECT * FROM strategy_cemetery WHERE idea_id=?", (SENTINEL,)
        ).fetchone()
    assert row is not None
    assert row["factor_type"] == "momentum"
    assert row["sector"] == "banking"
    assert "revive" in row["revival_conditions"].lower()


def test_find_similar_rejected_matches_word_overlap(clean):
    with db_session() as conn:
        conn.execute(
            "INSERT INTO alpha_ideas (id, slug, title, hypothesis, ticker, stage, status) "
            "VALUES (?, 'cemetery-test2', 'Banking momentum rotation', "
            "'20-day momentum on banks', '1155.KL', 'gate0', 'pending')",
            (SENTINEL,))
    RejectionMemory().record_rejection(SENTINEL, "overfitting", "gate0")

    hits = RejectionMemory().find_similar_rejected(
        "Banking momentum rotation strategy", "similar idea")
    assert len(hits) >= 1
    assert hits[0]["strategy_name"] == "Banking momentum rotation"
    assert hits[0]["similarity"] >= 0.5


def test_find_similar_rejected_no_match_for_unrelated_title(clean):
    hits = RejectionMemory().find_similar_rejected(
        "Completely unrelated CPO plantation lag strategy", "")
    assert hits == [] or all(h["similarity"] < 0.5 for h in hits)
