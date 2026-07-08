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
    bursa_trade_cost, bursa_slippage_tier, size_units,
    TICKER_REGEX, TICKER_EXAMPLE, MARKET_CURRENCY,
)


def simulate_fill(nav: float, price: float, adv_value_myr: float,
                  alloc_pct: float) -> dict:
    """Lot-rounded, capacity-aware fill simulation for one entry order.

    Sizing uses the market profile's rule (Bursa: whole 100-share board lots;
    crypto: fractional 0.0001 steps). Two independent constraints: (1) NAV ×
    alloc_pct, (2) the capacity rule — no more than capacity_max_participation
    of ADV in one day. Whichever is smaller determines the fill; if the
    capacity constraint binds, the order is a PARTIAL_FILL.
    """
    if price <= 0:
        return {"units": 0, "status": "FAILED", "reason": "invalid price"}

    requested_units = size_units(nav, price, alloc_pct)

    capacity_value = adv_value_myr * GATE_CONFIG.capacity_max_participation
    capacity_units = size_units(capacity_value, price, 1.0)

    units = min(requested_units, capacity_units) if capacity_units > 0 else requested_units
    if units < BURSA_BOARD_LOT or units <= 0:
        return {"units": 0, "status": "FAILED",
                "reason": "insufficient NAV or capacity for one minimum lot",
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

    if not ticker or not TICKER_REGEX.fullmatch(ticker.strip()):
        reasons.append(f"invalid ticker '{ticker}' — instruments look like {TICKER_EXAMPLE}")

    if bar is None:
        reasons.append("no price data available")
    else:
        if bar.get("close", 0) <= 0:
            reasons.append("non-positive price")
        adv = bar.get("adv_value", 0.0)
        if adv < BURSA_MIN_DAILY_VALUE_MYR:
            reasons.append(
                f"liquidity floor: ADV {MARKET_CURRENCY} {adv:,.0f} < "
                f"{MARKET_CURRENCY} {BURSA_MIN_DAILY_VALUE_MYR:,.0f}")
        if bar.get("close", 0) > 0 and size_units(nav, bar["close"], 0.95) <= 0:
            reasons.append("insufficient cash for one minimum lot")

    if dq_confidence is not None and dq_confidence < GATE_CONFIG.dq_min_confidence:
        reasons.append(
            f"data confidence {dq_confidence}/100 < {GATE_CONFIG.dq_min_confidence}")

    if unresolved_corp_actions > 0:
        reasons.append(f"{unresolved_corp_actions} unresolved suspected corporate action(s)")

    return {"passed": not reasons, "reasons": reasons}
