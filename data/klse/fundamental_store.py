"""Fundamental feature persistence (Phase 5.2, audit §7.4).

FundamentalScanner (data/klse/fundamental_scanner.py) already scans value,
momentum, dividend, and earnings-calendar screens but never persists per-ticker
fundamentals to a queryable store. This module adds that: a single upsert
function backtests/dashboards/red-team can read from without re-scraping.
"""
from __future__ import annotations

import logging
from datetime import datetime

from data.database import db_session

logger = logging.getLogger(__name__)

# Maps common scanner/scraper field names to the fundamental_features schema.
_FIELD_ALIASES = {
    "revenue": "revenue", "net_profit": "net_profit", "eps": "eps",
    "roe": "roe", "roa": "roa", "gross_margin": "gross_margin",
    "net_debt_equity": "net_debt_equity", "de_ratio": "net_debt_equity",
    "free_cash_flow": "free_cash_flow", "fcf": "free_cash_flow",
    "dividend_yield": "dividend_yield", "dy": "dividend_yield",
    "payout_ratio": "payout_ratio", "pe": "pe", "pb": "pb",
    "ev_ebitda": "ev_ebitda",
}


def persist_fundamentals(ticker: str, metrics: dict,
                         as_of_date: str | None = None,
                         source: str = "klse_screener") -> int | None:
    """Upsert one ticker's fundamentals for a date. `metrics` keys are matched
    against _FIELD_ALIASES (unknown keys are ignored) so callers can pass
    scanner/scraper dicts directly without reshaping them first.
    """
    as_of_date = as_of_date or datetime.utcnow().strftime("%Y-%m-%d")
    row = {col: None for col in set(_FIELD_ALIASES.values())}
    for k, v in metrics.items():
        col = _FIELD_ALIASES.get(k)
        if col and v is not None:
            try:
                row[col] = float(v)
            except (TypeError, ValueError):
                continue

    if not any(v is not None for v in row.values()):
        return None

    cols = list(row.keys())
    try:
        with db_session() as conn:
            conn.execute(f"""
                INSERT INTO fundamental_features
                  (ticker, as_of_date, {', '.join(cols)}, source)
                VALUES (?, ?, {', '.join('?' for _ in cols)}, ?)
                ON CONFLICT(ticker, as_of_date) DO UPDATE SET
                  {', '.join(f'{c}=excluded.{c}' for c in cols)}, source=excluded.source
            """, (ticker, as_of_date, *[row[c] for c in cols], source))
            r = conn.execute(
                "SELECT id FROM fundamental_features WHERE ticker=? AND as_of_date=?",
                (ticker, as_of_date),
            ).fetchone()
        return r["id"] if r else None
    except Exception as e:
        logger.warning(f"persist_fundamentals failed for {ticker} (non-blocking): {e}")
        return None


def latest_fundamentals(ticker: str) -> dict | None:
    """Most recent persisted fundamentals row for a ticker, or None."""
    with db_session() as conn:
        row = conn.execute(
            "SELECT * FROM fundamental_features WHERE ticker=? "
            "ORDER BY as_of_date DESC LIMIT 1", (ticker,)
        ).fetchone()
    return dict(row) if row else None
