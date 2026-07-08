"""Broker/execution simulator + pre-trade checks (Phase 6.1/6.2, audit §11).

Pure functions, no DB/network — the caller (PortfolioExecutor) supplies already-
fetched bars and idea state. This keeps the simulator unit-testable and keeps
the higher-blast-radius paper_entry/paper_exit flow a thin wrapper around it.

Still paper-only: nothing here talks to a real broker. It models board-lot
rounding, cash constraints, and capacity-based partial fills so the pipeline's
behavior once a live account exists won't be a surprise.
"""
from __future__ import annotations

from config.settings import (
    BURSA_BOARD_LOT, BURSA_MIN_DAILY_VALUE_MYR, GATE_CONFIG,
    bursa_trade_cost, bursa_slippage_tier,
)


def simulate_fill(nav: float, price: float, adv_value_myr: float,
                  alloc_pct: float) -> dict:
    """Board-lot-rounded, capacity-aware fill simulation for one entry order.

    Two independent constraints on size: (1) NAV × alloc_pct (existing sizing),
    (2) the audit's capacity rule — no more than capacity_max_participation of
    ADV in one day. Whichever is smaller determines the fill; if the capacity
    constraint binds, the order is a PARTIAL_FILL rather than a full one.
    """
    if price <= 0:
        return {"units": 0, "status": "FAILED", "reason": "invalid price"}

    requested_value = nav * alloc_pct
    requested_units = int(requested_value / price / BURSA_BOARD_LOT) * BURSA_BOARD_LOT

    capacity_value = adv_value_myr * GATE_CONFIG.capacity_max_participation
    capacity_units = int(capacity_value / price / BURSA_BOARD_LOT) * BURSA_BOARD_LOT

    units = min(requested_units, capacity_units) if capacity_units > 0 else requested_units
    if units < BURSA_BOARD_LOT:
        return {"units": 0, "status": "FAILED",
                "reason": "insufficient NAV or capacity for one board lot",
                "requested_units": requested_units}

    status = "PARTIAL_FILL" if units < requested_units else "FILLED"
    return {"units": units, "status": status, "requested_units": requested_units}


def pre_trade_check(ticker: str, nav: float, bar: dict | None,
                    dq_confidence: float | None,
                    unresolved_corp_actions: int) -> dict:
    """Gate an order before it reaches the simulator. Returns
    {"passed": bool, "reasons": [...]} — every failing check is listed so a
    rejected order is diagnosable, not just refused.
    """
    reasons = []

    if not ticker or not ticker.upper().endswith(".KL"):
        reasons.append(f"invalid ticker '{ticker}' — must be a .KL instrument")

    if bar is None:
        reasons.append("no price data available")
    else:
        if bar.get("close", 0) <= 0:
            reasons.append("non-positive price")
        adv = bar.get("adv_value", 0.0)
        if adv < BURSA_MIN_DAILY_VALUE_MYR:
            reasons.append(
                f"liquidity floor: ADV RM{adv:,.0f} < RM{BURSA_MIN_DAILY_VALUE_MYR:,.0f}")
        if bar.get("close", 0) > 0 and nav * 0.95 / bar["close"] < BURSA_BOARD_LOT:
            reasons.append("insufficient cash for one board lot")

    if dq_confidence is not None and dq_confidence < GATE_CONFIG.dq_min_confidence:
        reasons.append(
            f"data confidence {dq_confidence}/100 < {GATE_CONFIG.dq_min_confidence}")

    if unresolved_corp_actions > 0:
        reasons.append(f"{unresolved_corp_actions} unresolved suspected corporate action(s)")

    return {"passed": not reasons, "reasons": reasons}
