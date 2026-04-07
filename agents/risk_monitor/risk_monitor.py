import json, logging
from datetime import datetime
from agents.base_agent import BaseAgent
from config.settings import MODEL_FAST, GATE_CONFIG
from data.database import db_session

logger = logging.getLogger(__name__)

class RiskMonitor(BaseAgent):
    name = "RiskMonitor"
    description = "Risk control and exposure monitoring"
    default_model = MODEL_FAST

    def check_position_risk(self, account_summary, open_trades):
        nav            = account_summary.get("nav", 0)
        margin_used    = account_summary.get("margin_used", 0)
        unrealized_pnl = account_summary.get("unrealized_pnl", 0)
        margin_util    = margin_used / nav if nav > 0 else 0
        pnl_pct        = unrealized_pnl / nav if nav > 0 else 0
        alerts = []
        if margin_util > 0.50:
            alerts.append({"level":"WARN","msg":f"High margin: {margin_util:.1%}"})
        if margin_util > 0.80:
            alerts.append({"level":"ERROR","msg":f"CRITICAL margin: {margin_util:.1%}"})
        if pnl_pct < -GATE_CONFIG.stage4a_max_drawdown:
            alerts.append({"level":"ERROR","msg":f"Portfolio DD breached: {pnl_pct:.1%}"})
        for a in alerts:
            self.log_daemon(a["level"], a["msg"])
        return {
            "nav": nav, "margin_util": round(margin_util,4),
            "unrealized_pnl": unrealized_pnl, "pnl_pct": round(pnl_pct,4),
            "open_trades": len(open_trades), "alerts": alerts,
            "risk_level": "critical" if any(a["level"]=="ERROR" for a in alerts) else "warning" if alerts else "normal",
        }

    def check_strategy_drawdown(self, idea_id):
        with db_session() as conn:
            trades = conn.execute("SELECT pnl FROM paper_trades WHERE idea_id=? AND status='closed' ORDER BY closed_at", (idea_id,)).fetchall()
        if not trades:
            return {"idea_id": idea_id, "status": "no_data"}
        import numpy as np
        pnls    = [t["pnl"] for t in trades]
        cum_pnl = np.cumsum(pnls)
        peak    = np.maximum.accumulate(cum_pnl)
        dd      = ((peak - cum_pnl) / (np.abs(peak) + 1e-9)).max()
        breached = dd > GATE_CONFIG.stage4a_max_drawdown
        if breached:
            self.log_daemon("ERROR", f"Strategy [{idea_id}] drawdown breached: {dd:.1%}")
        return {"idea_id": idea_id, "drawdown": round(float(dd),4), "breached": breached, "trade_count": len(pnls)}

    def pipeline_health_report(self):
        with db_session() as conn:
            total    = conn.execute("SELECT COUNT(*) as n FROM alpha_ideas").fetchone()["n"]
            by_stage = conn.execute("SELECT stage, status, COUNT(*) as n FROM alpha_ideas GROUP BY stage, status").fetchall()
            daily_spend = conn.execute("SELECT COALESCE(SUM(cost_usd),0) as total FROM ai_usage WHERE created_at >= date('now')").fetchone()["total"]
            error_count = conn.execute("SELECT COUNT(*) as n FROM daemon_logs WHERE level='ERROR' AND created_at >= datetime('now','-1 hour')").fetchone()["n"]
        stage_map = {f"{r['stage']}:{r['status']}": r["n"] for r in by_stage}
        health = "healthy"
        if error_count > 10: health = "degraded"
        if error_count > 50: health = "critical"
        return {"total_ideas": total, "stages": stage_map, "daily_spend": round(float(daily_spend),4), "errors_1h": error_count, "health": health, "checked_at": datetime.utcnow().isoformat()}

    def run(self, task):
        action = task.get("action","health")
        if action == "health":
            return self.pipeline_health_report()
        elif action == "position_risk":
            return self.check_position_risk(task.get("account_summary",{}), task.get("open_trades",[]))
        elif action == "strategy_drawdown":
            return self.check_strategy_drawdown(task.get("idea_id"))
        return {"error": f"Unknown action: {action}"}
