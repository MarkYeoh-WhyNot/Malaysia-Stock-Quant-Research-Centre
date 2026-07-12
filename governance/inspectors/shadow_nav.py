"""Shadow-NAV Inspector — validates shared-book portfolio NAV accounting.

The invariant: paper portfolio NAV is tracked against a SINGLE shared capital
allocation (PAPER_CAPITAL_MYR), not naively summed from independent sandboxes
(which would double-count capital and misrepresent leverage).

This inspector detects if someone is using a naive sum of sandbox NAVs as the
"total portfolio NAV", which would show N times the shared capital when
everything is neutral. The correct model accounts for shared capital via the
paper_capital_multiplier metric.
"""

from typing import Optional, Dict, Any
from governance.base import Inspector
from governance.schemas import Finding
from config.settings import PAPER_CAPITAL_MYR
from data.database import db_session


class ShadowNAVInspector(Inspector):
    """L0 deterministic auditor for shadow-portfolio NAV accounting.

    Validates that any reported "total paper NAV" or portfolio-level NAV
    metrics use a shared-book model (accounting for paper_capital_multiplier),
    not a naive sum of independent sandbox NAVs which would double-count
    shared capital.
    """

    name = "ShadowNAVInspector"
    level = "L0"

    def inspect(
        self, scope: str, ctx: Dict[str, Any]
    ) -> Optional[Finding]:
        """Validate that portfolio NAV accounting uses the shared-book model.

        Args:
            scope: Context identifier (e.g. "portfolio_risk_snapshot", "nav_report:2026-07-12")
            ctx: Dictionary containing:
                - "reported_total_nav": the NAV figure being validated (e.g., PAPER_CAPITAL_MYR)
                - "gross_exposure_myr": sum of absolute position values
                - "active_strategy_count": number of active ideas/strategies
                - "sandbox_navs": optional list of individual sandbox NAVs to check for naive summing

        Returns:
            Finding with status PASS if using shared-book model,
            or BLOCKER if naive sandbox summing is detected.
        """
        reported_total_nav = ctx.get("reported_total_nav")
        gross_exposure_myr = ctx.get("gross_exposure_myr", 0.0)
        active_strategy_count = ctx.get("active_strategy_count", 0)
        sandbox_navs = ctx.get("sandbox_navs", [])

        # No active strategies — trivial pass
        if active_strategy_count == 0:
            return Finding(
                agent=self.name,
                level=self.level,
                scope=scope,
                status="PASS",
                severity="INFO",
                evidence={
                    "active_strategy_count": 0,
                    "paper_capital_myr": PAPER_CAPITAL_MYR,
                    "reason": "No active strategies",
                },
                local_recommendation="Portfolio is flat; no NAV validation needed.",
            )

        # Tolerance settings
        neutral_tolerance = 0.01 * PAPER_CAPITAL_MYR  # 1% for naive sum detection
        pnl_tolerance = 0.50 * PAPER_CAPITAL_MYR  # 50% for PnL in shared-book model (covers significant drawdowns)

        # If individual sandbox NAVs are provided, check for naive summing
        if sandbox_navs and reported_total_nav is not None:
            naive_sum = sum(sandbox_navs)
            # Actual tolerance: is the reported NAV matching the naive sum?
            # This catches both neutral sandboxes and sandboxes with PnL.
            is_reported_as_naive_sum = abs(reported_total_nav - naive_sum) < neutral_tolerance

            if is_reported_as_naive_sum and len(sandbox_navs) > 1:
                # Multiple sandboxes and reported as their sum — BLOCKER
                paper_capital_multiplier = (
                    gross_exposure_myr / PAPER_CAPITAL_MYR
                    if PAPER_CAPITAL_MYR > 0
                    else 0.0
                )
                return Finding(
                    agent=self.name,
                    level=self.level,
                    scope=scope,
                    status="FAIL",
                    severity="BLOCKER",
                    evidence={
                        "reported_total_nav": reported_total_nav,
                        "naive_sum_of_sandboxes": naive_sum,
                        "active_strategy_count": active_strategy_count,
                        "paper_capital_per_sandbox": PAPER_CAPITAL_MYR,
                        "gross_exposure_myr": gross_exposure_myr,
                        "paper_capital_multiplier": paper_capital_multiplier,
                        "discrepancy": f"Reported NAV ({reported_total_nav:.2f}) matches naive sum of {len(sandbox_navs)} sandboxes",
                    },
                    local_recommendation=(
                        f"Portfolio reports total NAV = {reported_total_nav:.2f}, which equals "
                        f"the naive sum of {active_strategy_count} sandboxes. "
                        f"This double-counts shared capital. Use shared-book model: account for "
                        f"paper_capital_multiplier = {gross_exposure_myr:.2f} / {PAPER_CAPITAL_MYR:.0f} = "
                        f"{gross_exposure_myr / PAPER_CAPITAL_MYR:.2f}x. Total deployable capital is still "
                        f"{PAPER_CAPITAL_MYR:.0f}, not {reported_total_nav:.2f}."
                    ),
                    escalate_to="RiskMonitor",
                )

        # Shared-book validation: if reported_total_nav is close to PAPER_CAPITAL_MYR
        # (within PnL tolerance), it's using the correct shared-book model
        if reported_total_nav is not None:
            is_shared_book = abs(reported_total_nav - PAPER_CAPITAL_MYR) < pnl_tolerance

            if is_shared_book:
                paper_capital_multiplier = (
                    gross_exposure_myr / PAPER_CAPITAL_MYR
                    if PAPER_CAPITAL_MYR > 0
                    else 0.0
                )
                return Finding(
                    agent=self.name,
                    level=self.level,
                    scope=scope,
                    status="PASS",
                    severity="INFO",
                    evidence={
                        "reported_total_nav": reported_total_nav,
                        "paper_capital_myr": PAPER_CAPITAL_MYR,
                        "active_strategy_count": active_strategy_count,
                        "gross_exposure_myr": gross_exposure_myr,
                        "paper_capital_multiplier": paper_capital_multiplier,
                        "model": "shared-book",
                    },
                    local_recommendation=(
                        f"Portfolio NAV correctly uses shared-book model: "
                        f"total NAV ≈ {PAPER_CAPITAL_MYR:.0f}, "
                        f"paper_capital_multiplier = {paper_capital_multiplier:.2f}x. "
                        f"Portfolio is leverage-honest."
                    ),
                )
            else:
                # NAV is significantly different from PAPER_CAPITAL_MYR
                # This could be massive drawdown or some other accounting issue
                return Finding(
                    agent=self.name,
                    level=self.level,
                    scope=scope,
                    status="WARN",
                    severity="WARNING",
                    evidence={
                        "reported_total_nav": reported_total_nav,
                        "paper_capital_myr": PAPER_CAPITAL_MYR,
                        "active_strategy_count": active_strategy_count,
                        "gross_exposure_myr": gross_exposure_myr,
                        "deviation_from_capital": reported_total_nav - PAPER_CAPITAL_MYR,
                    },
                    local_recommendation=(
                        f"Portfolio NAV reported as {reported_total_nav:.2f}, "
                        f"but shared capital is {PAPER_CAPITAL_MYR:.0f}. "
                        f"This deviation ({abs(reported_total_nav - PAPER_CAPITAL_MYR):.2f}) "
                        f"exceeds tolerance ({pnl_tolerance:.2f}). "
                        f"Verify accounting model is correct."
                    ),
                    escalate_to="RiskMonitor",
                )

        # No reported_total_nav provided — pass with warning to compute it
        return Finding(
            agent=self.name,
            level=self.level,
            scope=scope,
            status="PASS",
            severity="INFO",
            evidence={
                "active_strategy_count": active_strategy_count,
                "paper_capital_myr": PAPER_CAPITAL_MYR,
                "reason": "No reported_total_nav in context; cannot validate",
            },
            local_recommendation=(
                "Provide 'reported_total_nav' in context for full validation. "
                "Ensure it uses shared-book model, not naive sandbox sum."
            ),
        )
