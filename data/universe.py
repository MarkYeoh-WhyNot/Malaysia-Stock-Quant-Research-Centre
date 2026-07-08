"""Point-in-time index membership (Phase 2.2, audit §7.1).

Backtesting only today's KLCI constituents is survivorship-biased — it silently
drops names that were removed from the index (usually the losers). This resolver
returns the constituents as of a given date from `universe_membership`, so a
backtest can use the universe that actually existed at each point in time.

Until historical entries/exits are backfilled, the table holds only the current
constituents (effective_from = UNIVERSE_ASOF, effective_to = NULL); a backtest
whose window predates that is therefore NOT production-eligible — see
`is_production_eligible`.
"""
from __future__ import annotations

from config.settings import UNIVERSE_ASOF, DEFAULT_SYMBOLS, UNIVERSE_NAME
from data.database import db_session


def get_universe_asof(as_of: str | None = None,
                      universe_name: str = UNIVERSE_NAME) -> list[str]:
    """Tickers that were members of `universe_name` on `as_of` (YYYY-MM-DD).

    Falls back to the current DEFAULT_SYMBOLS if the table is empty. `as_of=None`
    means "today" → current members (effective_to IS NULL).
    """
    try:
        with db_session() as conn:
            if as_of is None:
                rows = conn.execute(
                    "SELECT ticker FROM universe_membership "
                    "WHERE universe_name=? AND effective_to IS NULL",
                    (universe_name,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT ticker FROM universe_membership
                    WHERE universe_name=?
                      AND effective_from <= ?
                      AND (effective_to IS NULL OR effective_to >= ?)
                    """,
                    (universe_name, as_of, as_of),
                ).fetchall()
        tickers = [r["ticker"] for r in rows]
    except Exception:
        tickers = []
    return tickers or list(DEFAULT_SYMBOLS)


def is_production_eligible(window_start: str | None,
                          universe_name: str = UNIVERSE_NAME) -> bool:
    """A backtest is production-eligible only if point-in-time membership covers
    its whole window. Today we only have membership from UNIVERSE_ASOF onward, so
    any window starting before that is research-grade only.
    """
    if not window_start:
        return False
    return window_start >= UNIVERSE_ASOF
