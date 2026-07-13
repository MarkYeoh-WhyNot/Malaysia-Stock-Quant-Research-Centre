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


# ── P2-5 (2026-07-13 audit): classified_by traceability + keyword fixes ─────

def test_classified_by_records_matched_keyword_for_fallback_classification(clean):
    with db_session() as conn:
        conn.execute(
            "INSERT INTO alpha_ideas (id, slug, title, hypothesis, ticker, stage, status) "
            "VALUES (?, 'cemetery-test6', 'Random idea', "
            "'some idea', '1155.KL', 'gate0', 'pending')", (SENTINEL,))

    RejectionMemory().record_rejection(SENTINEL, "overfit: too many parameters", "gate0")

    with db_session() as conn:
        row = conn.execute(
            "SELECT * FROM strategy_cemetery WHERE idea_id=?", (SENTINEL,)).fetchone()
    assert row["classified_by"] == "overfit"


def test_classified_by_records_explicit_prefix_when_caller_supplies_category(clean):
    with db_session() as conn:
        conn.execute(
            "INSERT INTO alpha_ideas (id, slug, title, hypothesis, ticker, stage, status) "
            "VALUES (?, 'cemetery-test7', 'Backtest failure idea', "
            "'some idea', '1155.KL', 'stage2', 'pending')", (SENTINEL,))

    RejectionMemory().record_rejection(
        SENTINEL, "Backtest failed G2/G3 — train_sharpe=0.10", "stage2",
        reason_category="low_sharpe")

    with db_session() as conn:
        row = conn.execute(
            "SELECT * FROM strategy_cemetery WHERE idea_id=?", (SENTINEL,)).fetchone()
    assert row["classified_by"] == "explicit:low_sharpe"


def test_data_quality_keyword_catches_not_reliably_available_phrasing(clean):
    """2026-07-13 audit finding: gate0's rationale said 'NOT reliably
    available' far more often than the plain 'not available' the old
    'infeasible' bucket matched — these were silently falling into 'other'."""
    with db_session() as conn:
        conn.execute(
            "INSERT INTO alpha_ideas (id, slug, title, hypothesis, ticker, stage, status) "
            "VALUES (?, 'cemetery-test8', 'Fundamental ratio idea', "
            "'PE and ROE screen', '1155.KL', 'gate0', 'pending')", (SENTINEL,))

    RejectionMemory().record_rejection(
        SENTINEL,
        "DATA_QUALITY FAIL: P/B and ROE fundamentals are NOT reliably "
        "available on Yahoo Finance .KL for this ticker", "gate0")

    with db_session() as conn:
        row = conn.execute(
            "SELECT * FROM strategy_cemetery WHERE idea_id=?", (SENTINEL,)).fetchone()
        pattern = conn.execute(
            "SELECT reason_category FROM rejection_patterns WHERE factor_type=? "
            "AND sector=? AND reason_category='data_quality'",
            (row["factor_type"], row["sector"])).fetchone()
    assert pattern is not None
    assert row["classified_by"] == "not reliably available"


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


# ── P2-5: ResearchDaemon._gate0_reason_category (2026-07-13 audit) ──────────
# The dominant fix — gate0's free-text "overfit" keyword match was
# mis-bucketing ~88% of Bursa strategy_cemetery rows, since Claude's
# rationale reviews all five scored dimensions even when a DIFFERENT one was
# the actual failure. These test the structured-score bypass directly.

def test_gate0_reason_category_prioritizes_data_quality_over_mentioned_overfitting():
    """The exact live pattern found in the audit: rationale text mentions
    overfitting_risk alongside a data_quality failure that's actually the
    more severe/primary issue — the structured score must win, not the
    keyword match on "overfit" that appears in the same sentence."""
    from scripts.research_daemon import ResearchDaemon
    daemon = ResearchDaemon(scan_interval=60)
    result = {
        "data_quality_score": 0.25, "logic_score": 0.40,
        "overfitting_risk": 0.72, "claude_feasibility": 0.55,
        "feasibility_score": 0.80,
        "rationale": "REJECTED on data_quality (0.25), logic (0.40), "
                    "overfitting_risk (0.72 > 0.40), and feasibility (0.55).",
    }
    assert daemon._gate0_reason_category(result) == "data_quality"


def test_gate0_reason_category_overfitting_when_that_is_the_only_failure():
    from scripts.research_daemon import ResearchDaemon
    daemon = ResearchDaemon(scan_interval=60)
    result = {
        "data_quality_score": 0.90, "overfitting_risk": 0.68,
        "claude_feasibility": 0.85, "feasibility_score": 0.75,
    }
    assert daemon._gate0_reason_category(result) == "overfitting"


def test_gate0_reason_category_infeasible_when_only_feasibility_fails():
    from scripts.research_daemon import ResearchDaemon
    daemon = ResearchDaemon(scan_interval=60)
    result = {
        "data_quality_score": 0.90, "overfitting_risk": 0.20,
        "claude_feasibility": 0.50, "feasibility_score": 0.75,
    }
    assert daemon._gate0_reason_category(result) == "infeasible"


def test_gate0_reason_category_none_for_pure_logic_failure_falls_through_to_keywords():
    """No structured bucket cleanly fits a logic-only failure — must return
    None so record_rejection falls through to keyword classification on the
    rationale text, not force a wrong structured label."""
    from scripts.research_daemon import ResearchDaemon
    daemon = ResearchDaemon(scan_interval=60)
    result = {
        "data_quality_score": 0.90, "overfitting_risk": 0.20,
        "claude_feasibility": 0.85, "feasibility_score": 0.75,
    }
    assert daemon._gate0_reason_category(result) is None


def test_gate0_rejection_uses_structured_category_not_overfit_keyword(clean):
    """End-to-end: a gate0-style rejection whose rationale mentions "overfit"
    but whose data_quality_score is the actual failing dimension must land
    in strategy_cemetery as 'data_quality', not 'overfitting'."""
    from scripts.research_daemon import ResearchDaemon
    with db_session() as conn:
        conn.execute(
            "INSERT INTO alpha_ideas (id, slug, title, hypothesis, ticker, stage, status) "
            "VALUES (?, 'cemetery-test9', 'Fundamental screen idea', "
            "'PE screen', '1155.KL', 'gate0', 'pending')", (SENTINEL,))

    daemon = ResearchDaemon(scan_interval=60)
    result = {
        "data_quality_score": 0.25, "overfitting_risk": 0.55,
        "claude_feasibility": 0.80, "feasibility_score": 0.80,
        "rationale": "Fails on data_quality and shows some overfitting risk too.",
    }
    RejectionMemory().record_rejection(
        SENTINEL, result["rationale"], "gate0",
        reason_category=daemon._gate0_reason_category(result))

    with db_session() as conn:
        row = conn.execute(
            "SELECT * FROM strategy_cemetery WHERE idea_id=?", (SENTINEL,)).fetchone()
    assert row["classified_by"] == "explicit:data_quality"
