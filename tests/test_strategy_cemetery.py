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
        for pattern in ("Banking momentum%", "BTC crypto dominance%",
                       "crypto BTC strategy%", "Cross-asset ratio%"):
            conn.execute("DELETE FROM rejection_patterns WHERE example_title LIKE ?", (pattern,))
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


# ── 2026-07-13 fix: explicit reason_category bypasses keyword guessing ──────

def test_explicit_reason_category_is_used_verbatim_not_reguessed(clean):
    """The free text below would keyword-match "irrelevant" (old bug: "crypto"
    was in that bucket) if left to guess — passing reason_category="unrepresentable"
    explicitly must win regardless of what the text says."""
    with db_session() as conn:
        conn.execute(
            "INSERT INTO alpha_ideas (id, slug, title, hypothesis, ticker, stage, status) "
            "VALUES (?, 'cemetery-test3', 'BTC crypto dominance ratio', "
            "'crypto BTC dominance', 'BTC/USDT', 'stage2', 'pending')",
            (SENTINEL,))

    RejectionMemory().record_rejection(
        SENTINEL, "requires computing a custom crypto BTC index ratio",
        "unrepresentable", reason_category="unrepresentable")

    with db_session() as conn:
        row = conn.execute(
            "SELECT * FROM strategy_cemetery WHERE idea_id=?", (SENTINEL,)).fetchone()
        pattern = conn.execute(
            "SELECT reason_category FROM rejection_patterns WHERE factor_type=? "
            "AND sector=? AND reason_category='unrepresentable'",
            (row["factor_type"], row["sector"])).fetchone()
    assert pattern is not None, "explicit reason_category must land in rejection_patterns"
    assert "DSL leaf exists" in row["revival_conditions"]


def test_crypto_keyword_no_longer_triggers_irrelevant_bucket(clean):
    """Regression: 'crypto' was a Bursa-only-era keyword in the 'irrelevant'
    bucket — in the crypto daemon almost every idea mentions crypto/BTC, so
    it mislabeled on-topic rejections. It must no longer match at all."""
    with db_session() as conn:
        conn.execute(
            "INSERT INTO alpha_ideas (id, slug, title, hypothesis, ticker, stage, status) "
            "VALUES (?, 'cemetery-test4', 'crypto BTC strategy', "
            "'a crypto BTC momentum idea', 'BTC/USDT', 'stage2', 'pending')",
            (SENTINEL,))

    RejectionMemory().record_rejection(
        SENTINEL, "crypto BTC momentum showed no edge on validation", "stage2")

    with db_session() as conn:
        row = conn.execute(
            "SELECT * FROM strategy_cemetery WHERE idea_id=?", (SENTINEL,)).fetchone()
    assert row["rejection_reason"].lower().count("crypto") >= 1  # sanity: text really has it
    with db_session() as conn:
        pattern = conn.execute(
            "SELECT reason_category FROM rejection_patterns WHERE factor_type=? "
            "AND sector=? AND reason_category='irrelevant'",
            (row["factor_type"], row["sector"])).fetchone()
    assert pattern is None, "'crypto' must not classify a rejection as irrelevant"


def test_unrepresentable_keyword_fallback_classifies_correctly(clean):
    """Even without an explicit reason_category, wording matching the
    parser's own honest-rejection phrasing should fall into the
    'unrepresentable' bucket, not 'infeasible' (the old catch-all for
    "not available")."""
    with db_session() as conn:
        conn.execute(
            "INSERT INTO alpha_ideas (id, slug, title, hypothesis, ticker, stage, status) "
            "VALUES (?, 'cemetery-test5', 'Cross-asset ratio idea', "
            "'ratio z-score', 'BTC/USDT', 'stage2', 'pending')",
            (SENTINEL,))

    RejectionMemory().record_rejection(
        SENTINEL,
        "this custom derived metric is not available in the condition set",
        "stage2")

    with db_session() as conn:
        row = conn.execute(
            "SELECT * FROM strategy_cemetery WHERE idea_id=?", (SENTINEL,)).fetchone()
        pattern = conn.execute(
            "SELECT reason_category FROM rejection_patterns WHERE factor_type=? "
            "AND sector=? AND reason_category='unrepresentable'",
            (row["factor_type"], row["sector"])).fetchone()
    assert pattern is not None
