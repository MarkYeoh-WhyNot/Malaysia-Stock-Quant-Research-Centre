"""Phase 5.1: announcement persistence (no network/Claude calls — the ingest
wrapper that calls the scraper + classifier is not exercised here)."""
import pytest

from data.database import db_session, init_db
from knowledge.ingestion.announcement_ingester import persist_classified_event

TICKER = "1155.KL"
DATE = "2026-06-01"


@pytest.fixture()
def clean():
    init_db()
    _cleanup()
    yield
    _cleanup()


def _cleanup():
    with db_session() as conn:
        conn.execute("DELETE FROM announcement_events WHERE ticker=?", (TICKER,))


def test_persists_actionable_dividend_event(clean):
    classified = {
        "event_type": "dividend_declared", "ticker": TICKER,
        "published_at": DATE, "headline": "Maybank declares 25 sen dividend",
        "sentiment": "positive", "magnitude": "medium", "confidence": 0.8,
        "is_actionable": True, "reasoning": "Pre-ex-dividend drift expected",
    }
    row_id = persist_classified_event(classified)
    assert row_id is not None

    with db_session() as conn:
        row = conn.execute(
            "SELECT * FROM announcement_events WHERE id=?", (row_id,)
        ).fetchone()
    assert row["announcement_type"] == "dividend_declared"
    assert row["is_actionable"] == 1
    assert row["sentiment_score"] == 1.0
    assert 0 < row["materiality_score"] <= 1.0


def test_missing_ticker_or_date_not_persisted(clean):
    assert persist_classified_event({"event_type": "macro_context"}) is None
    assert persist_classified_event({"ticker": TICKER}) is None


def test_duplicate_event_ignored(clean):
    classified = {
        "event_type": "earnings_beat", "ticker": TICKER, "published_at": DATE,
        "headline": "Q1 earnings beat", "sentiment": "positive",
        "magnitude": "high", "confidence": 0.9, "is_actionable": True,
    }
    first = persist_classified_event(classified)
    second = persist_classified_event(classified)
    assert first == second
    with db_session() as conn:
        n = conn.execute(
            "SELECT COUNT(*) n FROM announcement_events WHERE ticker=?", (TICKER,)
        ).fetchone()["n"]
    assert n == 1
