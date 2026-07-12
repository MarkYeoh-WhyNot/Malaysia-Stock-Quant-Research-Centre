"""Kill-Switch Inspector — surfaces hard-stop triggers as governance findings.

The kill-switch logic surfaces paper-trading halt conditions:
- Drawdown breach (strategy DD exceeds stage4a_max_drawdown)
- Data confidence failure (confidence_score < dq_min_confidence)
- Unresolved corporate action on the strategy's ticker

This inspector wraps the outcome of risk_monitor.check_kill_switches() and
records each trigger as a BLOCKER-severity Finding.
"""

from typing import Optional, Dict, Any
from governance.base import Inspector
from governance.schemas import Finding


class KillSwitchInspector(Inspector):
    """L0 inspector for paper-trading kill switches in active strategies.

    Wraps risk_monitor.check_kill_switches() and surfaces each triggered
    kill switch as a BLOCKER-severity Finding. Kill switches are hard stops:
    - Drawdown beyond stage4a_max_drawdown
    - Data confidence below dq_min_confidence
    - Unresolved corporate actions on the strategy's ticker

    Each triggered switch causes the affected idea's paper trading to PAUSE
    (behavior in portfolio_executor). This inspector records that pause as
    a governance Finding.
    """

    name = "KillSwitchInspector"
    level = "L0"

    def inspect(
        self, scope: str, ctx: Dict[str, Any]
    ) -> Optional[Finding]:
        """Wrap kill-switch outcome into a governance Finding.

        Args:
            scope: Context identifier (e.g. "portfolio_risk" or "idea:567")
            ctx: Dictionary containing:
                - "kill_switches": output dict from risk_monitor.check_kill_switches()
                  with keys "triggered" (list of dicts) and "count" (int)

        Returns:
            Finding with status PASS/FAIL and severity INFO/BLOCKER.
            - PASS/INFO if no kill switches triggered
            - FAIL/BLOCKER if one or more kill switches triggered (each trigger
              is a dict with idea_id, trigger type, and detail string)
        """
        kill_switches = ctx.get("kill_switches", {})
        triggered = kill_switches.get("triggered", [])
        count = kill_switches.get("count", 0)

        # No kill switches triggered — all active strategies are healthy
        if not triggered:
            return Finding(
                agent=self.name,
                level=self.level,
                scope=scope,
                status="PASS",
                severity="INFO",
                evidence={
                    "triggered_count": 0,
                    "active_strategies_healthy": True,
                },
                local_recommendation="No kill switches active. All paper-trading strategies remain live.",
            )

        # One or more kill switches triggered — strategies are paused
        trigger_details = []
        for t in triggered:
            trigger_details.append({
                "idea_id": t.get("idea_id"),
                "trigger": t.get("trigger"),
                "detail": t.get("detail"),
            })

        # Build evidence and recommendation based on trigger types
        trigger_types = set(t.get("trigger") for t in triggered)

        recommendation_parts = []
        if "drawdown" in trigger_types:
            recommendation_parts.append(
                "Review drawdown-breached strategies for exit signals or P&L lock-in."
            )
        if "data_confidence" in trigger_types:
            recommendation_parts.append(
                "Check data quality issues (corporate actions, gaps, stunted bars)."
            )
        if "corporate_action" in trigger_types:
            recommendation_parts.append(
                "Resolve unresolved corporate actions before resuming paper trading."
            )

        local_recommendation = " ".join(recommendation_parts)

        return Finding(
            agent=self.name,
            level=self.level,
            scope=scope,
            status="FAIL",
            severity="BLOCKER",
            evidence={
                "triggered_count": count,
                "triggers": trigger_details,
                "trigger_types": sorted(trigger_types),
            },
            local_recommendation=local_recommendation or "Investigate kill-switch triggers and resolve before resuming paper trading.",
            escalate_to="PortfolioExecutor" if count > 1 else "RiskMonitor",
        )
