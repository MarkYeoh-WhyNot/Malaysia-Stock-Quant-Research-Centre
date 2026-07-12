"""Capacity Aggregation Inspector — shared-liquidity capacity check across
concurrently active strategies (D3, governance work order).

`agents/portfolio_executor/execution_simulator.py::simulate_fill()` and the
per-idea capacity gate in `agents/backtest_engineer/gates.py` (fed by the
capacity haircut computed in `backtest_engineer.py::_run_backtest`, via
`_CAPACITY_IMPACT_COEF`) each reason about ONE strategy's ADV participation in
isolation — as if that strategy had the instrument's whole average daily
traded value to itself. In reality liquidity is a SHARED resource: if several
currently-active strategies all trade the same instrument at the same time,
they compete for the same ADV pool, so the market-impact haircut each one
actually experiences in practice is driven by the strategies' AGGREGATE
participation, not each one's own solo participation.

This inspector recomputes each strategy's Sharpe under that shared-capacity
assumption and flags (report-only — WARNING, never BLOCKER) any strategy whose
shared-capacity-adjusted Sharpe collapses relative to its solo estimate. It
does not touch execution_simulator.py's actual fill logic, does not touch
GATE_CONFIG/GATE_OVERRIDES (SACRED per CLAUDE.md), and does not gate anything
— it only surfaces information for a human/portfolio-level review.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional, Dict, Any, List

from governance.base import Inspector
from governance.schemas import Finding

# Reuse the existing, already-report-only market-impact coefficient from the
# per-idea capacity haircut (backtest_engineer.py) instead of inventing a
# second magic number for the same physical effect (linear impact per unit of
# ADV participation, one side of a trade).
from agents.backtest_engineer.backtest_engineer import _CAPACITY_IMPACT_COEF

# Report-only advisory floor — recorded in Finding evidence, never blocks a
# gate. Deliberately kept OUTSIDE config.settings.GATE_CONFIG / GATE_OVERRIDES
# (both SACRED, human-approval-only per CLAUDE.md); this is a brand new
# threshold for a brand new (informational) check, not an edit to an existing
# gate.
CAPACITY_SHARED_SHARPE_FLOOR: float = 0.50

# Minimum genuine Sharpe drop (attributable to crowding, not to an already-bad
# solo strategy) before we call it "degraded". Guards against flagging a
# strategy that was already below the floor on its own.
_MIN_CROWDING_DEGRADATION = 1e-6


def _group_by_instrument(strategies: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Bucket strategies by the instrument/symbol they share liquidity on."""
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in strategies:
        groups[s.get("instrument", "UNKNOWN")].append(s)
    return groups


def _shared_capacity_sharpe(strategy: Dict[str, Any], total_participation: float,
                             adv_value_myr: float) -> Optional[float]:
    """Recompute one strategy's Sharpe as if its market-impact haircut is
    driven by the AGGREGATE participation of every concurrent strategy on
    this instrument, rather than its own solo notional/ADV.

    Uses the same linear-impact-per-side model as the existing per-idea
    capacity haircut (`_CAPACITY_IMPACT_COEF * participation`), converted to
    an annualized Sharpe drag via `extra_annual_cost / ann_vol` (cost lowers
    annualized return; volatility is assumed roughly unaffected by cost
    drag — a first-order approximation, consistent with how the existing
    per-idea capacity_adjusted_sharpe is reported: disclosed, not gated).

    Returns None if the strategy doesn't carry enough info to recompute
    (missing solo_sharpe/ann_vol, or a non-positive ADV).
    """
    notional = strategy.get("notional_myr", 0.0)
    solo_sharpe = strategy.get("solo_sharpe")
    ann_vol = strategy.get("ann_vol")
    trades_per_year = strategy.get("trades_per_year") or 0.0

    if solo_sharpe is None or not ann_vol or ann_vol <= 0 or adv_value_myr <= 0:
        return None

    solo_participation = notional / adv_value_myr
    solo_impact_per_side = _CAPACITY_IMPACT_COEF * solo_participation
    shared_impact_per_side = _CAPACITY_IMPACT_COEF * total_participation

    # Only the INCREMENTAL impact from crowding matters here — the solo
    # haircut is already priced into solo_sharpe upstream (backtest_engineer.py
    # computes capacity_adjusted_sharpe per idea before this inspector ever
    # runs).
    delta_impact_per_side = max(0.0, shared_impact_per_side - solo_impact_per_side)
    extra_annual_cost = delta_impact_per_side * 2 * trades_per_year  # round trip

    return solo_sharpe - (extra_annual_cost / ann_vol)


class CapacityAggregationInspector(Inspector):
    """L0 report-only auditor: recomputes Sharpe under SHARED capacity
    participation across every currently-active strategy trading the same
    instrument, instead of each strategy's own isolated capacity haircut.

    Liquidity is a shared resource — this inspector's whole point is to catch
    the case where several strategies each individually clear the per-idea
    capacity gate, but would collectively exceed the instrument's realistic
    absorption capacity if run concurrently.
    """

    name = "CapacityAggregationInspector"
    level = "L0"

    def inspect(self, scope: str, ctx: Dict[str, Any]) -> Optional[Finding]:
        """Recompute shared-capacity Sharpe per instrument group.

        Args:
            scope: Context identifier, e.g. "capacity_aggregation:2026-07-12"
                or a portfolio/date identifier.
            ctx: Dictionary containing:
                - "strategies": list of dicts, each with:
                    - "strategy_id" (str/int) — idea/strategy identifier
                    - "instrument" (str) — symbol whose liquidity is shared
                    - "notional_myr" (float) — position notional
                    - "solo_sharpe" (float) — the strategy's own
                      capacity-adjusted Sharpe as already computed in
                      isolation (backtest_engineer.py capacity_adjusted_sharpe)
                    - "ann_vol" (float) — annualized return volatility
                    - "trades_per_year" (float) — round-trip turnover per year
                - "adv_by_instrument": dict[str, float] — average daily
                  traded value (in market currency) per instrument

        Returns:
            A single Finding rolling up every instrument group. Severity is
            always WARNING or INFO — this check is report-only and never
            returns a BLOCKER. Returns None if there's nothing to check
            (no strategies, or no group has enough data to recompute).
        """
        strategies = ctx.get("strategies") or []
        adv_by_instrument = ctx.get("adv_by_instrument") or {}

        if not strategies:
            return None

        groups = _group_by_instrument(strategies)
        results = []
        any_degraded = False

        for instrument, group in groups.items():
            adv_value_myr = adv_by_instrument.get(instrument, 0.0)
            if adv_value_myr <= 0:
                continue

            total_participation = sum(
                s.get("notional_myr", 0.0) for s in group
            ) / adv_value_myr

            for s in group:
                shared_sharpe = _shared_capacity_sharpe(s, total_participation, adv_value_myr)
                if shared_sharpe is None:
                    continue
                solo_sharpe = s["solo_sharpe"]
                crowding_degradation = solo_sharpe - shared_sharpe
                degraded = (
                    shared_sharpe < CAPACITY_SHARED_SHARPE_FLOOR
                    and crowding_degradation > _MIN_CROWDING_DEGRADATION
                )
                if degraded:
                    any_degraded = True
                results.append({
                    "strategy_id": s.get("strategy_id"),
                    "instrument": instrument,
                    "participants": len(group),
                    "solo_sharpe": round(solo_sharpe, 4),
                    "shared_capacity_sharpe": round(shared_sharpe, 4),
                    "crowding_degradation": round(crowding_degradation, 4),
                    "below_floor": bool(shared_sharpe < CAPACITY_SHARED_SHARPE_FLOOR),
                })

        if not results:
            return None

        if any_degraded:
            return Finding(
                agent=self.name,
                level=self.level,
                scope=scope,
                status="FAIL",
                severity="WARNING",  # report-only, never BLOCKER — see module docstring
                evidence={
                    "results": results,
                    "floor": CAPACITY_SHARED_SHARPE_FLOOR,
                    "note": "report-only: informs sizing/scheduling decisions, "
                            "does not gate the pipeline",
                },
                local_recommendation=(
                    "One or more strategies' shared-capacity-adjusted Sharpe falls below "
                    f"{CAPACITY_SHARED_SHARPE_FLOOR} once concurrent participants on the same "
                    "instrument are accounted for. Consider staggering entries, reducing "
                    "concurrent allocation on the crowded instrument(s), or reviewing "
                    "shared-instrument concentration across active strategies. Advisory only "
                    "— no gate threshold has changed."
                ),
                escalate_to="PortfolioExecutor",
            )

        return Finding(
            agent=self.name,
            level=self.level,
            scope=scope,
            status="PASS",
            severity="INFO",
            evidence={
                "results": results,
                "floor": CAPACITY_SHARED_SHARPE_FLOOR,
            },
            local_recommendation=None,
        )
