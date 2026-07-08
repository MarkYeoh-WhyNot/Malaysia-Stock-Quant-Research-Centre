import json, logging
from datetime import datetime
from agents.base_agent import BaseAgent
from config.settings import MODEL_FAST, GATE_CONFIG, KLCI_BY_SYMBOL
from data.database import db_session

_BANK_SECTOR = "Banking"

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
        """Drawdown from the daily NAV series (paper_equity) so open-position
        losses count; falls back to closed-trade cumulative PnL for ideas that
        predate NAV tracking."""
        import numpy as np
        with db_session() as conn:
            navs = conn.execute(
                "SELECT nav FROM paper_equity WHERE idea_id=? ORDER BY date", (idea_id,)
            ).fetchall()
            trade_count = conn.execute(
                "SELECT COUNT(*) as n FROM paper_trades WHERE idea_id=? AND status='closed'",
                (idea_id,),
            ).fetchone()["n"]
        if len(navs) >= 2:
            series = np.array([r["nav"] for r in navs], dtype=float)
            peak = np.maximum.accumulate(series)
            dd = float(((peak - series) / peak).max())
        elif trade_count:
            with db_session() as conn:
                trades = conn.execute(
                    "SELECT pnl FROM paper_trades WHERE idea_id=? AND status='closed' ORDER BY closed_at",
                    (idea_id,),
                ).fetchall()
            cum_pnl = np.cumsum([t["pnl"] for t in trades])
            peak = np.maximum.accumulate(cum_pnl)
            dd = float(((peak - cum_pnl) / (np.abs(peak) + 1e-9)).max())
        else:
            return {"idea_id": idea_id, "status": "no_data"}
        breached = dd > GATE_CONFIG.stage4a_max_drawdown
        if breached:
            self.log_daemon("ERROR", f"Strategy [{idea_id}] drawdown breached: {dd:.1%}")
        return {"idea_id": idea_id, "drawdown": round(dd, 4), "breached": breached, "trade_count": trade_count}

    def portfolio_risk_snapshot(self):
        """Phase 4.2 (audit §10): portfolio-level + Malaysia-specific
        concentration across all OPEN paper positions. Exposure per ticker is
        units × entry_price (cost basis). Persists a risk_snapshots row and flags
        breaches of single-name / sector / bank limits.
        """
        with db_session() as conn:
            rows = conn.execute(
                "SELECT pair, units, entry_price FROM paper_trades WHERE status='open'"
            ).fetchall()

        by_ticker, by_sector, bank_exposure, gross = {}, {}, 0.0, 0.0
        for r in rows:
            val = float((r["units"] or 0) * (r["entry_price"] or 0))
            if val <= 0:
                continue
            gross += val
            by_ticker[r["pair"]] = by_ticker.get(r["pair"], 0.0) + val
            sector = (KLCI_BY_SYMBOL.get(r["pair"], {}) or {}).get("sector", "Unknown")
            by_sector[sector] = by_sector.get(sector, 0.0) + val
            if sector == _BANK_SECTOR:
                bank_exposure += val

        if gross <= 0:
            return {"open_positions": 0, "gross_exposure_myr": 0.0,
                    "concentration_ok": True, "detail": "no open positions"}

        max_single_pct = max(by_ticker.values()) / gross
        max_sector, max_sector_val = max(by_sector.items(), key=lambda kv: kv[1])
        max_sector_pct = max_sector_val / gross
        bank_pct = bank_exposure / gross

        breaches = []
        if max_single_pct > GATE_CONFIG.max_single_name_pct:
            breaches.append(f"single-name {max_single_pct:.0%} > {GATE_CONFIG.max_single_name_pct:.0%}")
        if max_sector_pct > GATE_CONFIG.max_sector_pct:
            breaches.append(f"sector {max_sector} {max_sector_pct:.0%} > {GATE_CONFIG.max_sector_pct:.0%}")
        if bank_pct > GATE_CONFIG.max_bank_pct:
            breaches.append(f"bank {bank_pct:.0%} > {GATE_CONFIG.max_bank_pct:.0%}")
        concentration_ok = not breaches

        kill = self.check_kill_switches()
        kill_active = bool(kill["triggered"])
        detail = json.dumps({
            "by_sector_pct": {k: round(v / gross, 3) for k, v in by_sector.items()},
            "breaches": breaches, "kill_switches": kill["triggered"],
        })

        with db_session() as conn:
            conn.execute("""
                INSERT INTO risk_snapshots
                  (open_positions, gross_exposure_myr, max_single_pct, max_sector,
                   max_sector_pct, bank_pct, concentration_ok, kill_switch_active, detail)
                VALUES (?,?,?,?,?,?,?,?,?)
            """, (len(rows), round(gross, 2), round(max_single_pct, 4), max_sector,
                  round(max_sector_pct, 4), round(bank_pct, 4),
                  1 if concentration_ok else 0, 1 if kill_active else 0, detail))

        for b in breaches:
            self.log_daemon("WARN", f"Concentration breach: {b}")
        if breaches:
            try:
                from scripts.alerts import send_alert
                send_alert(f"Portfolio concentration breach: {'; '.join(breaches)}",
                          level="WARNING")
            except Exception:
                pass
        return {
            "open_positions": len(rows), "gross_exposure_myr": round(gross, 2),
            "max_single_pct": round(max_single_pct, 4),
            "max_sector": max_sector, "max_sector_pct": round(max_sector_pct, 4),
            "bank_pct": round(bank_pct, 4), "concentration_ok": concentration_ok,
            "breaches": breaches, "kill_switch_active": kill_active,
            "kill_switches": kill["triggered"],
        }

    def check_kill_switches(self):
        """Phase 4.3 (audit §10.3): hard-stop triggers across active paper
        strategies — drawdown breach, low data confidence, or an unresolved
        suspected corporate action. Paper-only (no live wiring): surfaces status,
        does not liquidate.
        """
        triggered = []
        with db_session() as conn:
            active = conn.execute(
                "SELECT id, ticker FROM alpha_ideas WHERE stage='stage4a' AND status='active'"
            ).fetchall()
        for idea in active:
            dd = self.check_strategy_drawdown(idea["id"])
            if dd.get("breached"):
                triggered.append({"idea_id": idea["id"], "trigger": "drawdown",
                                  "detail": f"DD {dd['drawdown']:.1%}"})
            with db_session() as conn:
                dq = conn.execute(
                    "SELECT confidence_score FROM data_quality_checks WHERE idea_id=? "
                    "ORDER BY created_at DESC LIMIT 1", (idea["id"],)
                ).fetchone()
                unresolved = conn.execute(
                    "SELECT COUNT(*) n FROM corporate_actions WHERE ticker=? "
                    "AND validation_status='suspected'", (idea["ticker"] or "",)
                ).fetchone()["n"]
            if dq and dq["confidence_score"] is not None and \
                    dq["confidence_score"] < GATE_CONFIG.dq_min_confidence:
                triggered.append({"idea_id": idea["id"], "trigger": "data_confidence",
                                  "detail": f"{dq['confidence_score']}/100"})
            if unresolved:
                triggered.append({"idea_id": idea["id"], "trigger": "corporate_action",
                                  "detail": f"{unresolved} unresolved"})
        for t in triggered:
            self.log_daemon("ERROR", f"KILL SWITCH [{t['idea_id']}] {t['trigger']}: {t['detail']}")
        if triggered:
            try:
                from scripts.alerts import send_alert
                send_alert(
                    f"{len(triggered)} kill switch(es) triggered: " +
                    "; ".join(f"idea {t['idea_id']} {t['trigger']} ({t['detail']})"
                             for t in triggered),
                    level="CRITICAL")
            except Exception:
                pass
        return {"triggered": triggered, "count": len(triggered)}

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
        elif action == "portfolio_risk":
            return self.portfolio_risk_snapshot()
        elif action == "kill_switches":
            return self.check_kill_switches()
        return {"error": f"Unknown action: {action}"}
