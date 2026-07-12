"""Metric consistency auditor for backtest performance metrics.

This inspector verifies that arithmetic annual return and compounded CAGR metrics
are present and internally consistent within a backtest run's performance data.
"""

import logging
from typing import Optional, Dict, Any
from governance.base import Inspector
from governance.schemas import Finding

logger = logging.getLogger(__name__)


class MetricConsistencyAuditor(Inspector):
    """Auditor that validates consistency between arithmetic and geometric returns.

    The backtest engine computes two return metrics:
    - ann_return: arithmetic mean daily return × trading days per year
    - cagr: compounded annual growth rate from the equity curve endpoint

    This inspector ensures both are present and that the compounded equity curve
    endpoint is consistent with the reported CAGR over the backtest period.
    """

    name = "MetricConsistencyAuditor"
    level = "L0"

    def inspect(self, scope: str, ctx: Dict[str, Any]) -> Optional[Finding]:
        """Inspect a backtest run's performance metrics for consistency.

        Args:
            scope: Identifier like "backtest_run:1234"
            ctx: Dictionary containing:
                - performance: dict with metrics from _compute_performance
                - n_obs: number of observations (bars) in the backtest
                - leverage_used: leverage factor (default 1.0)

        Returns:
            A Finding with status PASS/FAIL or None if check doesn't apply
        """
        if "performance" not in ctx or not isinstance(ctx["performance"], dict):
            return None

        perf = ctx["performance"]
        n_obs = ctx.get("n_obs", 0)

        # Check 1: Both metrics must be present
        if "ann_return" not in perf:
            return Finding(
                agent=self.name,
                level=self.level,
                scope=scope,
                status="FAIL",
                severity="BLOCKER",
                evidence={"missing_field": "ann_return"},
                local_recommendation="Backtest engine did not compute ann_return",
                escalate_to="BacktestEngineer",
            )

        if "cagr" not in perf:
            return Finding(
                agent=self.name,
                level=self.level,
                scope=scope,
                status="FAIL",
                severity="BLOCKER",
                evidence={"missing_field": "cagr"},
                local_recommendation="Backtest engine did not compute cagr",
                escalate_to="BacktestEngineer",
            )

        ann_return = float(perf["ann_return"])
        cagr = float(perf["cagr"])

        # Check 2: Both metrics should be finite
        if not (-1e6 < ann_return < 1e6):
            return Finding(
                agent=self.name,
                level=self.level,
                scope=scope,
                status="FAIL",
                severity="BLOCKER",
                evidence={
                    "ann_return": ann_return,
                    "issue": "ann_return is not finite",
                },
                local_recommendation="Check for division by zero or NaN propagation",
                escalate_to="BacktestEngineer",
            )

        if not (-1e6 < cagr < 1e6):
            return Finding(
                agent=self.name,
                level=self.level,
                scope=scope,
                status="FAIL",
                severity="BLOCKER",
                evidence={
                    "cagr": cagr,
                    "issue": "cagr is not finite",
                },
                local_recommendation="Check for non-positive equity curve endpoint",
                escalate_to="BacktestEngineer",
            )

        # Check 3: Internal consistency (advanced)
        # If we have n_obs, we can cross-check the CAGR computation.
        # The engine computes: cagr = cum[-1] ** (ann / n) - 1
        # where cum is the equity curve (cumulative product of (1 + net_returns))
        # and ann is BARS_PER_YEAR (typically 252 for daily).
        #
        # We can't reconstruct the full equity curve here, but we can check
        # basic sanity: if there are many observations and reasonable returns,
        # the CAGR should not be wildly inconsistent with ann_return.
        #
        # A rough heuristic: for a long backtest with 500+ observations,
        # CAGR and ann_return should be in similar ballparks (within 5x ratio).
        # This catches cases where the equity curve is inverted or endpoint
        # is garbage.

        if n_obs > 500:
            # Both should have the same sign and similar magnitude if reasonable
            if ann_return * cagr < 0:  # opposite signs
                return Finding(
                    agent=self.name,
                    level=self.level,
                    scope=scope,
                    status="FAIL",
                    severity="BLOCKER",
                    evidence={
                        "ann_return": ann_return,
                        "cagr": cagr,
                        "issue": "Opposite signs suggest corrupted metrics",
                        "n_obs": n_obs,
                    },
                    local_recommendation="Equity curve endpoint likely corrupted",
                    escalate_to="BacktestEngineer",
                )

            # If both are non-zero, check the ratio
            if abs(ann_return) > 0.01 and abs(cagr) > 0.01:
                ratio = abs(cagr / ann_return)
                if ratio > 5.0 or ratio < 0.2:
                    return Finding(
                        agent=self.name,
                        level=self.level,
                        scope=scope,
                        status="FAIL",
                        severity="BLOCKER",
                        evidence={
                            "ann_return": ann_return,
                            "cagr": cagr,
                            "ratio_cagr_to_ann_return": round(ratio, 3),
                            "issue": f"CAGR/AnnReturn ratio {ratio:.2f} suggests inconsistency",
                            "n_obs": n_obs,
                        },
                        local_recommendation=(
                            "Check equity curve computation: endpoint may be "
                            "inconsistent with daily return series"
                        ),
                        escalate_to="BacktestEngineer",
                    )

        # All checks passed
        return Finding(
            agent=self.name,
            level=self.level,
            scope=scope,
            status="PASS",
            severity="INFO",
            evidence={
                "ann_return": round(ann_return, 4),
                "cagr": round(cagr, 4),
                "n_obs": n_obs,
            },
            local_recommendation=None,
            escalate_to=None,
        )
