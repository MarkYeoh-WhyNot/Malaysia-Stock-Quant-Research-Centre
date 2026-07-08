"""Date-aware Bursa transaction-cost resolution (Phase 1.1, audit §3.2).

`config.settings.bursa_trade_cost` remains the fast, constant-based cost model
for the *current* schedule (single source of truth for "today"). This module
adds the historical dimension: given a trade date, it resolves the fee schedule
that was actually in force from the `fee_schedules` table and computes cost from
it — so a backtest spanning the 2023-07-13 stamp-duty remission boundary can use
the rate that applied at the time.

Kept out of config/settings.py deliberately: settings must not import the DB
layer (data.database imports config.settings — the reverse would cycle).
"""
from __future__ import annotations

from functools import lru_cache

from config.settings import (
    MARKET, BURSA_COMMISSION_RATE, BURSA_CLEARING_RATE, BURSA_CLEARING_CAP_MYR,
    BURSA_STAMP_DUTY_RATE, BURSA_STAMP_DUTY_CAP_MYR, BURSA_BOARD_LOT,
    BURSA_SETTLEMENT_CYCLE, BURSA_SLIPPAGE_TIERS,
)
from data.database import db_session


def _constants_schedule() -> dict:
    """Fallback schedule from the current settings constants."""
    return {
        "commission_rate": BURSA_COMMISSION_RATE,
        "clearing_rate":   BURSA_CLEARING_RATE,
        "clearing_cap":    BURSA_CLEARING_CAP_MYR,
        "stamp_duty_rate": BURSA_STAMP_DUTY_RATE,
        "stamp_duty_cap":  BURSA_STAMP_DUTY_CAP_MYR,
        "board_lot":       BURSA_BOARD_LOT,
        "settlement_cycle": BURSA_SETTLEMENT_CYCLE,
        "source":          "constants",
    }


@lru_cache(maxsize=64)
def get_fee_schedule(as_of: str | None = None,
                     instrument_type: str = "listed_equity") -> dict:
    """Return the fee schedule in force on `as_of` (YYYY-MM-DD).

    Falls back to the settings constants if the table is empty or has no row
    covering the date. `as_of=None` means "latest / today" and also falls back
    to constants (the current schedule) so the hot path stays DB-free-equivalent.
    Cached because schedules change rarely.
    """
    if as_of is None:
        return _constants_schedule()
    try:
        with db_session() as conn:
            row = conn.execute(
                """
                SELECT * FROM fee_schedules
                WHERE market=? AND instrument_type=?
                  AND effective_from <= ?
                  AND (effective_to IS NULL OR effective_to >= ?)
                ORDER BY effective_from DESC LIMIT 1
                """,
                (MARKET, instrument_type, as_of, as_of),
            ).fetchone()
    except Exception:
        row = None
    if row is None:
        return _constants_schedule()
    return {
        "commission_rate": row["commission_rate"],
        "clearing_rate":   row["clearing_rate"],
        "clearing_cap":    row["clearing_cap"],
        "stamp_duty_rate": row["stamp_duty_rate"],
        "stamp_duty_cap":  row["stamp_duty_cap"],
        "board_lot":       row["board_lot"],
        "settlement_cycle": row["settlement_cycle"],
        "source":          f"fee_schedules#{row['id']}",
    }


def bursa_trade_cost_asof(trade_value_myr: float, side: str,
                          slippage_tier: str = "BLUE_CHIP",
                          as_of: str | None = None) -> float:
    """Total MYR cost for one side of a Bursa trade, using the schedule in force
    on `as_of`. Mirrors config.settings.bursa_trade_cost but date-aware.
    """
    sched = get_fee_schedule(as_of)
    value = abs(trade_value_myr)
    cost = value * sched["commission_rate"]
    cost += min(value * sched["clearing_rate"], sched["clearing_cap"])
    if side == "buy":
        cost += min(value * sched["stamp_duty_rate"], sched["stamp_duty_cap"])
    cost += value * BURSA_SLIPPAGE_TIERS.get(
        slippage_tier, BURSA_SLIPPAGE_TIERS["MID_CAP"])
    return cost
