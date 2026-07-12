"""Fill Convention Auditor — validates trade PnL reconciliation.

The keystone invariant: every bar belongs to exactly one trade, so
summed trade-level net PnL must reconcile (within float tolerance) to
the overall backtest net return.
"""

from typing import Optional, Dict, Any, List
from governance.base import Inspector
from governance.schemas import Finding


class FillConventionAuditor(Inspector):
    """L0 deterministic auditor for fill convention and trade reconciliation.

    Validates that the sum of reconstructed trade net_pct equals the
    overall backtest net return (in cumulative terms). This guards the
    keystone invariant: every bar's PnL is attributed to exactly one
    trade, so the blotter reconciles to the backtest equity curve.
    """

    name = "FillConventionAuditor"
    level = "L0"

    def inspect(
        self, scope: str, ctx: Dict[str, Any]
    ) -> Optional[Finding]:
        """Validate trade-level PnL reconciles to backtest return.

        Args:
            scope: Context identifier (e.g. "backtest_run:1234")
            ctx: Dictionary containing:
                - "trades": list of trade dicts with net_pct field
                - "backtest_net_return": cumulative net return as a float (e.g. 0.153 for +15.3%)
                - "tolerance": optional float tolerance (default 1e-4, ~0.01% error)

        Returns:
            Finding with status PASS if reconciliation succeeds,
            or BLOCKER if trade sum diverges from backtest return.
        """
        trades = ctx.get("trades", [])
        backtest_net_return = ctx.get("backtest_net_return", 0.0)
        tolerance = ctx.get("tolerance", 1e-4)

        # No trades — trivial pass
        if not trades:
            return Finding(
                agent=self.name,
                level=self.level,
                scope=scope,
                status="PASS",
                severity="INFO",
                evidence={"trades_count": 0, "backtest_return": backtest_net_return},
                local_recommendation="No trades to reconcile.",
            )

        # Sum trade net PnL (as a decimal; trades store net_pct in percentage)
        summed_net_pct = sum(t.get("net_pct", 0.0) for t in trades)
        summed_net_return = summed_net_pct / 100.0

        # Compute absolute and relative error
        abs_error = abs(summed_net_return - backtest_net_return)
        rel_error = (
            abs_error / abs(backtest_net_return)
            if abs(backtest_net_return) > 1e-10
            else abs_error
        )

        # Reconciliation check
        reconciles = abs_error < tolerance

        evidence = {
            "trades_count": len(trades),
            "summed_net_return": round(summed_net_return, 6),
            "backtest_net_return": round(backtest_net_return, 6),
            "absolute_error": round(abs_error, 8),
            "relative_error": round(rel_error, 6),
            "tolerance": tolerance,
            "reconciles": reconciles,
        }

        if reconciles:
            return Finding(
                agent=self.name,
                level=self.level,
                scope=scope,
                status="PASS",
                severity="INFO",
                evidence=evidence,
                local_recommendation=f"Trade blotter reconciles to backtest return within {tolerance} tolerance.",
            )
        else:
            return Finding(
                agent=self.name,
                level=self.level,
                scope=scope,
                status="FAIL",
                severity="BLOCKER",
                evidence=evidence,
                local_recommendation=(
                    f"Trade-level net PnL diverges from backtest return by {abs_error:.8f} "
                    f"({rel_error*100:.4f}%). Likely cause: inconsistent cost attribution across trades. "
                    f"Check transition cost splits in _reconstruct_trades."
                ),
                escalate_to="BacktestEngineer",
            )
