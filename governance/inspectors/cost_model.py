"""Cost Model Auditor — validates transaction cost rates match the interval.

The critical bug class: a sub-daily (e.g. "1h", "15m") backtest run silently
applies the daily cost default instead of the interval-appropriate cost rate,
resulting in understated transaction costs and overstated Sharpe ratios.

This inspector catches the case where:
  - interval is "1h" (sub-daily)
  - but cost rates are computed as if interval were "1d"
  - yielding too-low per-bar costs and artificial alpha
"""

from typing import Optional, Dict, Any
from governance.base import Inspector
from governance.schemas import Finding


class CostModelAuditor(Inspector):
    """L0 deterministic auditor for transaction cost rate correctness.

    Validates that the cost rates applied in a backtest match the declared
    interval. Catches silent fallback to daily costs on sub-daily runs,
    which inflates Sharpe ratios by undercounting slippage/commissions
    per-bar.
    """

    name = "CostModelAuditor"
    level = "L0"

    def inspect(
        self, scope: str, ctx: Dict[str, Any]
    ) -> Optional[Finding]:
        """Validate cost rates match the backtest interval.

        Args:
            scope: Context identifier (e.g. "backtest_run:1234")
            ctx: Dictionary containing:
                - "interval": backtest timeframe (e.g. "1d", "1h", "15m")
                - "cost_rates": dict with "buy", "sell" rates applied
                - "adv_value_myr": average daily volume MYR (for tier classification)
                - "bars_per_day": expected bars per calendar day for this interval
                - "expected_cost_rates": dict with expected "buy", "sell" for comparison

        Returns:
            Finding with status PASS if costs match interval,
            or BLOCKER if cost rates appear to use wrong interval.
        """
        interval = ctx.get("interval", "1d")
        cost_rates = ctx.get("cost_rates", {})
        adv_value = ctx.get("adv_value_myr", 0.0)
        bars_per_day = ctx.get("bars_per_day", 1.0)
        expected_rates = ctx.get("expected_cost_rates", {})

        # Validation: we need both the actual and expected costs
        actual_buy = cost_rates.get("buy")
        actual_sell = cost_rates.get("sell")
        if actual_buy is None or actual_sell is None:
            return Finding(
                agent=self.name,
                level=self.level,
                scope=scope,
                status="FAIL",
                severity="BLOCKER",
                evidence={
                    "interval": interval,
                    "cost_rates": cost_rates,
                    "missing_fields": [k for k in ["buy", "sell"] if k not in cost_rates],
                },
                local_recommendation="Cost rates dict is incomplete. Check _cost_rates call.",
                escalate_to="BacktestEngineer",
            )

        expected_buy = expected_rates.get("buy")
        expected_sell = expected_rates.get("sell")
        if expected_buy is None or expected_sell is None:
            return Finding(
                agent=self.name,
                level=self.level,
                scope=scope,
                status="FAIL",
                severity="BLOCKER",
                evidence={
                    "interval": interval,
                    "expected_rates": expected_rates,
                    "missing_fields": [k for k in ["buy", "sell"] if k not in expected_rates],
                },
                local_recommendation="Expected cost rates dict is incomplete.",
                escalate_to="BacktestEngineer",
            )

        # The check: for sub-daily intervals, costs should be HIGHER per-bar
        # because the same position is held for fewer bars per day.
        # Tolerance: 0.01% (float rounding tolerance)
        tolerance = 1e-4

        is_subdaily = interval not in ("1d", "1w", "1mo")
        cost_mismatch = abs(actual_buy - expected_buy) > tolerance or abs(actual_sell - expected_sell) > tolerance

        if cost_mismatch:
            # Cost rates don't match expectations — likely interval bug
            evidence = {
                "interval": interval,
                "is_subdaily": is_subdaily,
                "bars_per_day": bars_per_day,
                "adv_value_myr": adv_value,
                "actual_buy": round(actual_buy, 8),
                "actual_sell": round(actual_sell, 8),
                "expected_buy": round(expected_buy, 8),
                "expected_sell": round(expected_sell, 8),
                "buy_delta": round(actual_buy - expected_buy, 8),
                "sell_delta": round(actual_sell - expected_sell, 8),
            }

            if is_subdaily and bars_per_day > 1.0:
                # Sub-daily backtest: costs should be SCALED DOWN per-bar
                # (divided by bars_per_day). If actual costs are MUCH HIGHER than
                # expected, it means daily costs were used instead of scaled costs.
                if actual_buy > expected_buy * 1.05 or actual_sell > expected_sell * 1.05:
                    return Finding(
                        agent=self.name,
                        level=self.level,
                        scope=scope,
                        status="FAIL",
                        severity="BLOCKER",
                        evidence=evidence,
                        local_recommendation=(
                            f"Sub-daily interval '{interval}' (bars_per_day={bars_per_day:.2f}) but "
                            f"cost rates appear to use daily defaults. Actual costs are {(actual_buy/expected_buy - 1)*100:.1f}% "
                            f"above expected — likely a silent fallback to '1d' in _cost_rates(). "
                            f"This inflates Sharpe by undercounting per-bar transaction drag."
                        ),
                        escalate_to="BacktestEngineer",
                    )

            # Cost delta for daily or other intervals
            return Finding(
                agent=self.name,
                level=self.level,
                scope=scope,
                status="FAIL",
                severity="BLOCKER",
                evidence=evidence,
                local_recommendation=(
                    f"Transaction cost rates diverge from expected for interval '{interval}'. "
                    f"Buy-side delta: {(actual_buy - expected_buy)*100:.3f}%, "
                    f"sell-side delta: {(actual_sell - expected_sell)*100:.3f}%. "
                    f"Check cost rate resolution in _cost_rates and fee_schedule."
                ),
                escalate_to="BacktestEngineer",
            )

        # Costs match expected — pass
        return Finding(
            agent=self.name,
            level=self.level,
            scope=scope,
            status="PASS",
            severity="INFO",
            evidence={
                "interval": interval,
                "bars_per_day": bars_per_day,
                "adv_value_myr": adv_value,
                "actual_buy": round(actual_buy, 8),
                "actual_sell": round(actual_sell, 8),
            },
            local_recommendation=f"Cost rates are correct for interval '{interval}' at {adv_value:.0f} MYR ADV.",
        )
