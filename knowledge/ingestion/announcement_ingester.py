"""Bursa announcement persistence (Phase 5.1, audit §7.6).

Deliberately split from the classification call: `persist_classified_event()` is
pure DB I/O and is what's tested (no network, no Claude). `ingest_dividend_announcements()`
wires the existing I3investorScraper + EventClassifier — both already built —
into that persistence layer; it is the network/Claude-calling path and is
exercised in tests only with mocked inputs.
"""
from __future__ import annotations

import logging

from data.database import db_session

logger = logging.getLogger(__name__)


def persist_classified_event(classified: dict, source: str = "i3investor") -> int | None:
    """Write one EventClassifier.classify() output to announcement_events.

    `classified` is the dict returned by EventClassifier.classify() (or its
    rule-based fallback): event_type, ticker, sentiment, magnitude, is_actionable,
    confidence, reasoning, etc. Returns the row id, or None if there was nothing
    tickered/dated to persist.
    """
    ticker = classified.get("ticker")
    ann_date = classified.get("published_at") or classified.get("date")
    title = classified.get("headline") or classified.get("suggested_idea_title") or ""
    if not ticker or not ann_date:
        return None

    sentiment_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
    sentiment_score = sentiment_map.get(classified.get("sentiment"), 0.0)
    magnitude_map = {"high": 1.0, "medium": 0.6, "low": 0.3}
    materiality_score = magnitude_map.get(classified.get("magnitude"), 0.3) * \
        float(classified.get("confidence", 0.5))

    try:
        with db_session() as conn:
            conn.execute("""
                INSERT OR IGNORE INTO announcement_events
                  (ticker, announcement_date, announcement_type, title, source,
                   nlp_labels, sentiment_score, materiality_score, is_actionable)
                VALUES (?,?,?,?,?,?,?,?,?)
            """, (
                ticker, ann_date, classified.get("event_type", "macro_context"),
                title[:300], source,
                classified.get("reasoning", "")[:300],
                sentiment_score, round(materiality_score, 3),
                1 if classified.get("is_actionable") else 0,
            ))
            row = conn.execute(
                "SELECT id FROM announcement_events WHERE ticker=? AND announcement_date=? AND title=?",
                (ticker, ann_date, title[:300]),
            ).fetchone()
        return row["id"] if row else None
    except Exception as e:
        logger.warning(f"persist_classified_event failed (non-blocking): {e}")
        return None


def ingest_dividend_announcements(max_items: int = 20) -> int:
    """Scrape i3investor dividend headlines, classify each with EventClassifier,
    and persist. Network + Claude calls — not exercised by unit tests directly.
    Returns the count of new rows persisted.
    """
    from data.i3investor.scraper import I3investorScraper
    from agents.event_classifier import EventClassifier

    scraper = I3investorScraper()
    classifier = EventClassifier()
    items = scraper.get_dividend_announcements(max_items=max_items)

    persisted = 0
    for item in items:
        tickers = item.get("tickers") or []
        event_raw = {
            "source": "i3investor",
            "headline": item.get("headline", ""),
            "body": item.get("dividend_amount", ""),
            "ticker": tickers[0] if tickers else None,
            "published_at": item.get("date") or item.get("ex_date"),
        }
        try:
            classified = classifier.classify(event_raw)
            classified.setdefault("ticker", event_raw["ticker"])
            classified.setdefault("published_at", event_raw["published_at"])
            classified.setdefault("headline", event_raw["headline"])
            if persist_classified_event(classified, source="i3investor"):
                persisted += 1
        except Exception as e:
            logger.warning(f"ingest_dividend_announcements: item failed: {e}")
    return persisted
