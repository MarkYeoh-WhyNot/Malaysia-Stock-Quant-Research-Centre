import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Optional

_PROGRESS_FILE = "/tmp/openclaw_progress.json"

from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from data.database import db_session, init_db
from config.settings import AI_DAILY_BUDGET_USD, key_health_check

app = FastAPI(title="OpenClaw Mission Control", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_executor = ThreadPoolExecutor(max_workers=4)


@app.on_event("startup")
async def startup():
    init_db()


# ─── Helpers ────────────────────────────────────────────────────────────────

async def _in_thread(fn):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, fn)


# ─── Health ─────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    kh = key_health_check()
    # Query last daemon scan time from daemon_logs
    last_scan_time = None
    last_scan_secs = None
    try:
        with db_session() as conn:
            row = conn.execute(
                "SELECT created_at FROM daemon_logs WHERE source='ResearchDaemon' "
                "AND message LIKE 'Scan cycle%' ORDER BY id DESC LIMIT 1"
            ).fetchone()
        if row:
            ts = row["created_at"][:19].replace(" ", "T")
            dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S")
            last_scan_time = ts
            last_scan_secs = int((datetime.utcnow() - dt).total_seconds())
    except Exception:
        pass
    return {
        "status": "ok",
        "version": "2.0.0",
        "time": datetime.utcnow().isoformat(),
        "last_scan_time": last_scan_time,
        "last_scan_secs": last_scan_secs,
        "key_health": {
            "key_preview": kh["key_preview"],
            "healthy": kh["healthy"],
            "issues": kh["issues"],
        },
    }


# ─── Agent Progress ──────────────────────────────────────────────────────────

@app.get("/api/agent-progress")
def agent_progress():
    """Return active backtest progress keyed by idea_id."""
    data: dict = {}
    try:
        if os.path.exists(_PROGRESS_FILE):
            with open(_PROGRESS_FILE, "r") as fh:
                raw = json.load(fh)
            now = datetime.utcnow()
            # Purge stale entries (> 10 min old — backtest should never take that long)
            for idea_id, entry in raw.items():
                try:
                    ts = datetime.strptime(entry["ts"][:19], "%Y-%m-%dT%H:%M:%S")
                    age_secs = (now - ts).total_seconds()
                    if age_secs < 600:
                        data[idea_id] = entry
                except Exception:
                    data[idea_id] = entry
    except Exception:
        pass
    return {"progress": data, "as_of": datetime.utcnow().isoformat()}


# ─── Mission Control ─────────────────────────────────────────────────────────

@app.get("/api/mission-control")
def mission_control():
    with db_session() as conn:
        stages  = conn.execute("SELECT stage, COUNT(*) as n FROM alpha_ideas WHERE status != 'rejected' GROUP BY stage").fetchall()
        today   = datetime.utcnow().strftime("%Y-%m-%d")
        spend   = conn.execute("SELECT COALESCE(SUM(cost_usd),0) as total, COUNT(*) as calls FROM ai_usage WHERE created_at LIKE ?", (f"{today}%",)).fetchone()
        logs    = conn.execute("SELECT level, source, message, created_at FROM daemon_logs ORDER BY id DESC LIMIT 50").fetchall()
        totals  = conn.execute("SELECT COUNT(*) as total, SUM(CASE WHEN status='rejected' THEN 1 ELSE 0 END) as rejected, SUM(CASE WHEN stage='stage5' THEN 1 ELSE 0 END) as live FROM alpha_ideas").fetchone()
        models  = conn.execute("SELECT model, SUM(cost_usd) as cost, COUNT(*) as calls FROM ai_usage WHERE created_at LIKE ? GROUP BY model", (f"{today}%",)).fetchall()
        # Active stages: stages that have ideas currently awaiting processing
        active_raw = conn.execute("""
            SELECT stage FROM alpha_ideas
            WHERE (stage='gate0' AND status='pending')
               OR (stage IN ('stage1','stage2','stage3','stage4a') AND status='active')
            GROUP BY stage
        """).fetchall()
    stage_map = {r["stage"]: r["n"] for r in stages}
    active_stages = [r["stage"] for r in active_raw]
    return {
        "pipeline": {"total_ideas": totals["total"], "total_rejected": totals["rejected"], "live_strategies": totals["live"], "stages": stage_map, "active_stages": active_stages},
        "ai_usage": {"today_spend": round(float(spend["total"]), 4), "budget": AI_DAILY_BUDGET_USD, "budget_pct": round(float(spend["total"]) / AI_DAILY_BUDGET_USD * 100, 1), "total_calls": spend["calls"], "by_model": [{"model": r["model"], "cost": round(float(r["cost"]), 4), "calls": r["calls"]} for r in models]},
        "daemon_logs": [{"level": r["level"], "source": r["source"], "message": r["message"], "time": r["created_at"]} for r in logs],
        "timestamp": datetime.utcnow().isoformat(),
    }


# ─── Analytics ───────────────────────────────────────────────────────────────

@app.get("/api/analytics")
def analytics():
    with db_session() as conn:
        # Stage funnel (all-time)
        funnel = conn.execute("""
            SELECT stage, status, COUNT(*) as n FROM alpha_ideas GROUP BY stage, status
        """).fetchall()

        # Gate acceptance rates
        gate_rates = conn.execute("""
            SELECT gate, decision, COUNT(*) as n FROM gate_decisions GROUP BY gate, decision
        """).fetchall()

        # Daily idea creation (last 30 days)
        daily = conn.execute("""
            SELECT date(created_at) as day, COUNT(*) as n
            FROM alpha_ideas
            WHERE created_at >= date('now', '-30 days')
            GROUP BY day ORDER BY day
        """).fetchall()

        # Daily spend (last 14 days)
        daily_spend = conn.execute("""
            SELECT date(created_at) as day, SUM(cost_usd) as cost, COUNT(*) as calls
            FROM ai_usage
            WHERE created_at >= date('now', '-14 days')
            GROUP BY day ORDER BY day
        """).fetchall()

        # Agent performance
        agent_stats = conn.execute("""
            SELECT agent, COUNT(*) as calls, SUM(cost_usd) as cost,
                   AVG(input_tokens) as avg_in, AVG(output_tokens) as avg_out
            FROM ai_usage GROUP BY agent ORDER BY cost DESC
        """).fetchall()

        # Ticker distribution
        pairs = conn.execute("""
            SELECT ticker, COUNT(*) as n,
                   AVG(COALESCE(backtest_sharpe, 0)) as avg_sharpe
            FROM alpha_ideas WHERE ticker IS NOT NULL GROUP BY ticker ORDER BY n DESC
        """).fetchall()

        # Pipeline events last 7 days
        events = conn.execute("""
            SELECT date(created_at) as day, event_type, COUNT(*) as n
            FROM pipeline_events
            WHERE created_at >= date('now', '-7 days')
            GROUP BY day, event_type ORDER BY day
        """).fetchall()

    # Build gate acceptance map
    gate_map = {}
    for r in gate_rates:
        g = r["gate"]
        if g not in gate_map:
            gate_map[g] = {"approve": 0, "reject": 0}
        gate_map[g][r["decision"]] = r["n"]
    gate_acceptance = [
        {"gate": g, "approve": v["approve"], "reject": v["reject"],
         "rate": round(v["approve"] / max(v["approve"] + v["reject"], 1) * 100, 1)}
        for g, v in gate_map.items()
    ]

    # Build stage funnel
    stage_order = ["gate0", "stage1", "stage2", "stage3", "stage4a", "stage4b", "stage5"]
    funnel_map = {}
    for r in funnel:
        s = r["stage"]
        if s not in funnel_map:
            funnel_map[s] = {"active": 0, "rejected": 0, "pending": 0}
        funnel_map[s][r["status"]] = r["n"]
    funnel_data = [
        {"stage": s, **funnel_map.get(s, {"active": 0, "rejected": 0, "pending": 0})}
        for s in stage_order
    ]

    return {
        "funnel": funnel_data,
        "gate_acceptance": gate_acceptance,
        "daily_ideas": [dict(r) for r in daily],
        "daily_spend": [{"day": r["day"], "cost": round(float(r["cost"]), 4), "calls": r["calls"]} for r in daily_spend],
        "agent_stats": [{"agent": r["agent"], "calls": r["calls"], "cost": round(float(r["cost"]), 4), "avg_in": round(float(r["avg_in"] or 0)), "avg_out": round(float(r["avg_out"] or 0))} for r in agent_stats],
        "tickers": [{"ticker": r["ticker"], "count": r["n"], "avg_sharpe": round(float(r["avg_sharpe"]), 3)} for r in pairs],
        "pipeline_events": [dict(r) for r in events],
    }


# ─── Pipeline / Ideas ────────────────────────────────────────────────────────

@app.get("/api/pipeline/ideas")
def get_ideas(stage: Optional[str] = None, status: Optional[str] = None, limit: int = 50):
    with db_session() as conn:
        where, params = [], []
        if stage:  where.append("ai.stage=?");  params.append(stage)
        if status: where.append("ai.status=?"); params.append(status)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        # LEFT JOIN latest backtest_run to expose QC metrics in the ideas table
        ideas = conn.execute(f"""
            SELECT ai.*,
                br.sharpe_net       AS bt_sharpe_net,
                br.sharpe_gross     AS bt_sharpe_gross,
                br.sharpe_oos       AS bt_sharpe_oos,
                br.trade_count      AS bt_trade_count,
                br.regimes_positive AS bt_regimes_positive,
                br.sanity_flags     AS bt_sanity_flags
            FROM alpha_ideas ai
            LEFT JOIN (
                SELECT idea_id, sharpe_net, sharpe_gross, sharpe_oos,
                       trade_count, regimes_positive, sanity_flags,
                       MAX(id) AS latest_id
                FROM backtest_runs
                GROUP BY idea_id
            ) br ON br.idea_id = ai.id
            {clause}
            ORDER BY ai.id DESC LIMIT ?
        """, params + [limit]).fetchall()
        total = conn.execute(
            f"SELECT COUNT(*) as n FROM alpha_ideas ai {clause}", params
        ).fetchone()["n"]
    return {"total": total, "ideas": [dict(r) for r in ideas]}


@app.get("/api/pipeline/ideas/{idea_id}")
def get_idea(idea_id: int):
    with db_session() as conn:
        idea   = conn.execute("SELECT * FROM alpha_ideas WHERE id=?", (idea_id,)).fetchone()
        if not idea:
            raise HTTPException(status_code=404, detail="Idea not found")
        events = conn.execute("SELECT * FROM pipeline_events WHERE idea_id=? ORDER BY id DESC", (idea_id,)).fetchall()
        gates  = conn.execute("SELECT * FROM gate_decisions WHERE idea_id=? ORDER BY id DESC", (idea_id,)).fetchall()
        btruns = conn.execute("SELECT * FROM backtest_runs WHERE idea_id=? ORDER BY id DESC LIMIT 5", (idea_id,)).fetchall()
        ptrades = conn.execute("SELECT * FROM paper_trades WHERE idea_id=? ORDER BY id DESC LIMIT 20", (idea_id,)).fetchall()
    return {
        "idea": dict(idea),
        "events": [dict(r) for r in events],
        "gate_decisions": [dict(r) for r in gates],
        "backtest_runs": [dict(r) for r in btruns],
        "paper_trades": [dict(r) for r in ptrades],
    }


@app.get("/api/pipeline/gate-queue")
def gate_queue():
    with db_session() as conn:
        # Ideas pending gate scoring
        pending_g0   = conn.execute("SELECT * FROM alpha_ideas WHERE stage='gate0' AND status='pending' ORDER BY id DESC LIMIT 20").fetchall()
        pending_s1   = conn.execute("SELECT * FROM alpha_ideas WHERE stage='stage1' AND status='active' AND research_score IS NULL ORDER BY id DESC LIMIT 20").fetchall()
        pending_s2   = conn.execute("SELECT * FROM alpha_ideas WHERE stage='stage2' AND status='active' ORDER BY id DESC LIMIT 20").fetchall()
        recent_gates = conn.execute("SELECT gd.*, ai.title, ai.ticker FROM gate_decisions gd JOIN alpha_ideas ai ON ai.id=gd.idea_id ORDER BY gd.id DESC LIMIT 30").fetchall()
    return {
        "gate0_pending": [dict(r) for r in pending_g0],
        "stage1_pending": [dict(r) for r in pending_s1],
        "stage2_pending": [dict(r) for r in pending_s2],
        "recent_decisions": [dict(r) for r in recent_gates],
    }


# ─── Detailed Analytics ──────────────────────────────────────────────────────

@app.get("/api/pipeline/analytics/detailed")
def detailed_analytics():
    today = datetime.utcnow().strftime("%Y-%m-%d")
    with db_session() as conn:
        # Rejection by stage
        rejection_by_stage = conn.execute("""
            SELECT stage, COUNT(*) as n FROM alpha_ideas
            WHERE status='rejected' GROUP BY stage ORDER BY stage
        """).fetchall()

        # Full status × stage matrix for donut
        status_dist = conn.execute("""
            SELECT stage, status, COUNT(*) as n FROM alpha_ideas GROUP BY stage, status
        """).fetchall()

        # Gate 0 pass / fail from gate_decisions
        g0_decisions = conn.execute("""
            SELECT decision, COUNT(*) as n FROM gate_decisions
            WHERE gate='gate0' GROUP BY decision
        """).fetchall()

        # Stage 2+ active count (ideas that made it past both gates)
        stage2_plus = conn.execute("""
            SELECT COUNT(*) as n FROM alpha_ideas
            WHERE stage IN ('stage2','stage3','stage4a','stage4b','stage5')
            AND status != 'rejected'
        """).fetchone()["n"]

        # Today's spend
        today_spend = conn.execute(
            "SELECT COALESCE(SUM(cost_usd),0) as total FROM ai_usage WHERE created_at LIKE ?",
            (f"{today}%",)
        ).fetchone()["total"]

        # Total ideas (for coach stub)
        total_ideas = conn.execute("SELECT COUNT(*) as n FROM alpha_ideas").fetchone()["n"]

    # Gate 0 pass rate
    g0_map = {r["decision"]: r["n"] for r in g0_decisions}
    g0_approve = g0_map.get("approve", 0)
    g0_total   = g0_approve + g0_map.get("reject", 0)
    g0_pass_rate = round(g0_approve / max(g0_total, 1) * 100, 1)

    # Build donut segments from stage × status matrix
    dist = {}
    for r in status_dist:
        dist[f"{r['stage']}:{r['status']}"] = r["n"]

    total_rejected = sum(v for k, v in dist.items() if k.endswith(":rejected"))
    donut_segments = [
        {"label": "Gate 0 Pending", "value": dist.get("gate0:pending", 0),  "color": "#3b82f6"},
        {"label": "Gate 0 Active",  "value": dist.get("gate0:active", 0),   "color": "#60a5fa"},
        {"label": "Stage 1",        "value": dist.get("stage1:active", 0) + dist.get("stage1:pending", 0), "color": "#06b6d4"},
        {"label": "Stage 2",        "value": dist.get("stage2:active", 0) + dist.get("stage2:pending", 0), "color": "#8b5cf6"},
        {"label": "Stage 3",        "value": dist.get("stage3:active", 0) + dist.get("stage3:pending", 0), "color": "#f97316"},
        {"label": "Stage 4A",       "value": dist.get("stage4a:active", 0) + dist.get("stage4a:pending", 0), "color": "#10b981"},
        {"label": "Stage 5 Live",   "value": dist.get("stage5:active", 0),  "color": "#ec4899"},
        {"label": "Rejected",       "value": total_rejected,                 "color": "#ef4444"},
    ]
    donut_segments = [s for s in donut_segments if s["value"] > 0]

    # Coach performance — stub until coaching is wired up
    coach_performance = [
        {
            "coach": "Generic (No Coach)",
            "total_ideas": total_ideas,
            "explored": total_ideas,
            "gate_pass_rate": g0_pass_rate,
        }
    ]

    return {
        "rejection_by_stage": [{"stage": r["stage"], "count": r["n"]} for r in rejection_by_stage],
        "status_distribution": donut_segments,
        "coach_performance": coach_performance,
        "gate0_pass_rate": g0_pass_rate,
        "stage2_plus_count": stage2_plus,
        "today_spend": round(float(today_spend), 4),
        "total_ideas": total_ideas,
    }


# ─── Agent Team ───────────────────────────────────────────────────────────────

@app.get("/api/agent-team")
def agent_team():
    now       = datetime.utcnow()
    ts_now    = now.strftime("%Y-%m-%dT%H:%M:%S")
    cutoff_2m = (now - timedelta(minutes=2)).strftime("%Y-%m-%dT%H:%M:%S")
    cutoff_5m = (now - timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%S")
    cutoff_1h = (now - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S")

    with db_session() as conn:
        # ── StrategyResearcher ────────────────────────────────────────────────
        sr_gate0_pending = conn.execute(
            "SELECT COUNT(*) as n FROM alpha_ideas WHERE stage='gate0' AND status='pending'"
        ).fetchone()["n"]
        sr_stage1_active = conn.execute(
            "SELECT COUNT(*) as n FROM alpha_ideas WHERE stage='stage1' AND status='active'"
        ).fetchone()["n"]
        # Last gate decision (most meaningful "last action")
        sr_last_gate = conn.execute("""
            SELECT gd.decision, gd.created_at, gd.idea_id, ai.title, ai.ticker,
                   ai.novelty_score, ai.logic_score
            FROM gate_decisions gd
            JOIN alpha_ideas ai ON ai.id = gd.idea_id
            WHERE gd.gate = 'gate0'
            ORDER BY gd.id DESC LIMIT 1
        """).fetchone()
        # Pipeline-derived active: pending gate0 ideas OR active stage1 ideas OR recent update
        sr_pipeline_active = (sr_gate0_pending > 0) or (sr_stage1_active > 0)
        sr_ts_active = bool(conn.execute(
            "SELECT 1 FROM alpha_ideas WHERE updated_at >= ? LIMIT 1", (cutoff_2m,)
        ).fetchone())
        sr_is_active = sr_pipeline_active or sr_ts_active
        sr_last_ts = sr_last_gate["created_at"] if sr_last_gate else None
        # "Working on" — most recent gate0/stage1 idea
        sr_working_idea = conn.execute("""
            SELECT id, title, ticker, stage FROM alpha_ideas
            WHERE (stage='gate0' AND status='pending')
               OR (stage='stage1' AND status='active')
            ORDER BY updated_at DESC LIMIT 1
        """).fetchone()

        # ── DataEngineer ──────────────────────────────────────────────────────
        de_stage2_active = conn.execute(
            "SELECT COUNT(*) as n FROM alpha_ideas WHERE stage='stage2' AND status='active'"
        ).fetchone()["n"]
        de_pipeline_active = de_stage2_active > 0
        de_ts_active = bool(conn.execute(
            "SELECT 1 FROM backtest_runs WHERE created_at >= ? LIMIT 1", (cutoff_5m,)
        ).fetchone())
        de_is_active = de_pipeline_active or de_ts_active
        # Most informative data log: prefer fetch/cache messages
        de_log = conn.execute("""
            SELECT message, created_at FROM daemon_logs
            WHERE (message LIKE '%Fetch%' OR message LIKE '%fetch%'
                   OR message LIKE '%cache%' OR message LIKE '%bars%'
                   OR message LIKE '%DataEngineer%')
            ORDER BY id DESC LIMIT 1
        """).fetchone()
        de_last_ts = de_log["created_at"] if de_log else None
        # "Working on" — most recent stage2 idea for data fetch
        de_working_idea = conn.execute("""
            SELECT id, title, ticker FROM alpha_ideas
            WHERE stage='stage2' AND status='active'
            ORDER BY updated_at DESC LIMIT 1
        """).fetchone()

        # ── BacktestEngineer ──────────────────────────────────────────────────
        bt_last = conn.execute("""
            SELECT br.id, br.idea_id, br.passed, br.train_sharpe, br.val_sharpe, br.test_sharpe,
                   br.created_at, ai.title, ai.ticker
            FROM backtest_runs br
            JOIN alpha_ideas ai ON ai.id = br.idea_id
            ORDER BY br.id DESC LIMIT 1
        """).fetchone()
        bt_queued = conn.execute("""
            SELECT COUNT(*) as n FROM alpha_ideas ai
            WHERE ai.stage='stage2' AND ai.status='active'
              AND NOT EXISTS (SELECT 1 FROM backtest_runs br WHERE br.idea_id=ai.id)
        """).fetchone()["n"]
        bt_ts_active = bool(conn.execute(
            "SELECT 1 FROM backtest_runs WHERE created_at >= ? LIMIT 1", (cutoff_5m,)
        ).fetchone())
        bt_pipeline_active = bt_queued > 0
        bt_is_active = bt_pipeline_active or bt_ts_active
        bt_last_ts = bt_last["created_at"] if bt_last else None
        # "Working on" — next unbacktested stage2 idea
        bt_working_idea = conn.execute("""
            SELECT ai.id, ai.title, ai.ticker FROM alpha_ideas ai
            WHERE ai.stage='stage2' AND ai.status='active'
              AND NOT EXISTS (SELECT 1 FROM backtest_runs br WHERE br.idea_id=ai.id)
            ORDER BY ai.updated_at DESC LIMIT 1
        """).fetchone()

        # ── RiskMonitor ───────────────────────────────────────────────────────
        rm_errors_1h = conn.execute(
            "SELECT COUNT(*) as n FROM daemon_logs WHERE level='ERROR' AND created_at >= ?",
            (cutoff_1h,)
        ).fetchone()["n"]
        rm_watching = conn.execute(
            "SELECT COUNT(*) as n FROM alpha_ideas WHERE stage IN ('stage4a','stage5') AND status='active'"
        ).fetchone()["n"]
        rm_research_active = conn.execute(
            "SELECT COUNT(*) as n FROM alpha_ideas WHERE stage IN ('stage1','stage2','stage3') AND status='active'"
        ).fetchone()["n"]
        rm_total_active = conn.execute(
            "SELECT COUNT(*) as n FROM alpha_ideas WHERE status='active'"
        ).fetchone()["n"]
        rm_last_log = conn.execute(
            "SELECT message, created_at FROM daemon_logs ORDER BY id DESC LIMIT 1"
        ).fetchone()
        rm_last_ts = rm_last_log["created_at"] if rm_last_log else None

        # ── PortfolioExecutor ─────────────────────────────────────────────────
        pe_open_trades = conn.execute(
            "SELECT COUNT(*) as n FROM paper_trades WHERE status='open'"
        ).fetchone()["n"]
        pe_stage4_ideas = conn.execute(
            "SELECT COUNT(*) as n FROM alpha_ideas WHERE stage IN ('stage4a','stage5') AND status='active'"
        ).fetchone()["n"]
        pe_last_trade = conn.execute("""
            SELECT pt.status, pt.opened_at, pt.pnl, ai.title, ai.ticker
            FROM paper_trades pt
            JOIN alpha_ideas ai ON ai.id = pt.idea_id
            ORDER BY pt.id DESC LIMIT 1
        """).fetchone()
        pe_is_active = pe_open_trades > 0 or pe_stage4_ideas > 0
        pe_last_ts = pe_last_trade["opened_at"] if pe_last_trade else None
        # "Working on" — active stage4a/5 idea
        pe_working_idea = conn.execute("""
            SELECT id, title, ticker FROM alpha_ideas
            WHERE stage IN ('stage4a','stage5') AND status='active'
            ORDER BY updated_at DESC LIMIT 1
        """).fetchone()

    def _trunc(text: str, maxlen: int = 45) -> str:
        if not text:
            return ""
        return (text[:maxlen] + "…") if len(text) > maxlen else text

    def _ts_ago(ts_str: str | None) -> str:
        """Convert a UTC timestamp string to a human-readable 'X ago' string."""
        if not ts_str:
            return ""
        try:
            # SQLite datetime('now') returns "YYYY-MM-DD HH:MM:SS" (space separator)
            # normalise to ISO format before parsing
            normalised = ts_str[:19].replace(" ", "T")
            dt = datetime.strptime(normalised, "%Y-%m-%dT%H:%M:%S")
            secs = int((now - dt).total_seconds())
            if secs < 60:    return f"{secs}s ago"
            if secs < 3600:  return f"{secs//60}m ago"
            if secs < 86400: return f"{secs//3600}h ago"
            return f"{secs//86400}d ago"
        except Exception:
            return ""

    # Build last-action strings
    if sr_last_gate:
        _outcome = "pass" if sr_last_gate["decision"] == "approve" else "fail"
        sr_last_action = (
            f"Last: Scored idea [{sr_last_gate['idea_id']}] {sr_last_gate['ticker']} — {_outcome}"
        )
    else:
        sr_last_action = "No gate decisions yet"

    if de_log:
        de_last_action = f"Last: {_trunc(de_log['message'], 52)}"
    else:
        de_last_action = "No data fetch activity yet"

    if bt_last:
        if bt_ts_active:
            bt_last_action = f"Last: Backtesting [{bt_last['idea_id']}] {bt_last['ticker']} — running"
        else:
            _pass  = "pass" if bt_last["passed"] else "fail"
            _sharpe = bt_last["test_sharpe"] or 0
            bt_last_action = (
                f"Last: Backtested [{bt_last['idea_id']}] {bt_last['ticker']} — {_pass}"
                f" (Sharpe {_sharpe:.2f})"
            )
    else:
        bt_last_action = "No backtest runs yet"

    if rm_watching > 0:
        rm_task = f"Watching {rm_watching} paper trade idea{'s' if rm_watching != 1 else ''}"
    elif rm_research_active > 0:
        rm_task = f"Watching {rm_research_active} idea{'s' if rm_research_active != 1 else ''} in factor dev"
    elif rm_total_active > 0:
        rm_task = f"Monitoring {rm_total_active} active idea{'s' if rm_total_active != 1 else ''} in pipeline"
    else:
        rm_task = "Pipeline empty — monitoring for activity"
    if rm_errors_1h > 0:
        rm_task += f" · {rm_errors_1h} error{'s' if rm_errors_1h != 1 else ''} in last hour"

    if pe_last_trade:
        if pe_open_trades > 0:
            pe_task = f"{pe_open_trades} open trade{'s' if pe_open_trades != 1 else ''}: {pe_last_trade['ticker']} — {_trunc(pe_last_trade['title'], 30)}"
        else:
            _pnl = pe_last_trade["pnl"]
            _pnl_str = f"PnL={_pnl:+.4f}" if _pnl is not None else "PnL=—"
            pe_task = f"Last: {pe_last_trade['ticker']} {pe_last_trade['status']} {_pnl_str}"
    else:
        pe_task = "No paper trades active"

    # Build pipeline-derived current_task and working_on strings
    if sr_pipeline_active and sr_working_idea:
        _sr_stage_label = "Screening" if sr_working_idea["stage"] == "gate0" else "Researching"
        sr_current_task = f"{_sr_stage_label} [{sr_working_idea['id']}] {_trunc(sr_working_idea['title'], 35)}"
        sr_working_on = f"Working on: [{sr_working_idea['id']}] {sr_working_idea['title']}"
    elif sr_is_active:
        sr_current_task = f"Scoring ideas… ({sr_gate0_pending} pending)"
        sr_working_on = sr_current_task
    else:
        sr_current_task = sr_last_action
        sr_working_on = ""

    if de_pipeline_active and de_working_idea:
        de_current_task = f"Fetching data for [{de_working_idea['id']}] {de_working_idea['ticker']}"
        de_working_on = f"Working on: [{de_working_idea['id']}] {de_working_idea['title']}"
    elif de_is_active:
        de_current_task = de_last_action
        de_working_on = de_last_action
    else:
        de_current_task = de_last_action
        de_working_on = ""

    if bt_pipeline_active and bt_working_idea:
        bt_current_task = f"Queued: [{bt_working_idea['id']}] {_trunc(bt_working_idea['title'], 35)}"
        bt_working_on = f"Working on: [{bt_working_idea['id']}] {bt_working_idea['title']}"
    elif bt_ts_active and bt_last:
        bt_current_task = f"Running backtest [{bt_last['idea_id']}] {bt_last['ticker']}…"
        bt_working_on = f"Working on: [{bt_last['idea_id']}] {bt_last['title']}"
    else:
        bt_current_task = bt_last_action
        bt_working_on = ""

    if pe_is_active and pe_working_idea:
        pe_working_on = f"Working on: [{pe_working_idea['id']}] {pe_working_idea['title']}"
    else:
        pe_working_on = ""

    agents = [
        {
            "name":          "StrategyResearcher",
            "display_name":  "Strategy Researcher",
            "subtitle":      "Gate 0 screening, deep research, idea generation",
            "status":        "ACTIVE" if sr_is_active else "IDLE",
            "pending_tasks": sr_gate0_pending + sr_stage1_active,
            "current_task":  sr_current_task,
            "last_action":   sr_last_action,
            "last_updated":  _ts_ago(sr_last_ts),
            "working_on":    sr_working_on,
            "pipeline_stage": "gate0" if sr_gate0_pending > 0 else ("stage1" if sr_stage1_active > 0 else ""),
        },
        {
            "name":          "DataEngineer",
            "display_name":  "Data Engineer",
            "subtitle":      "Yahoo Finance fetch, feature engineering, cache",
            "status":        "ACTIVE" if de_is_active else "IDLE",
            "pending_tasks": de_stage2_active,
            "current_task":  de_current_task,
            "last_action":   de_last_action,
            "last_updated":  _ts_ago(de_last_ts),
            "working_on":    de_working_on,
            "pipeline_stage": "stage2" if de_pipeline_active else "",
        },
        {
            "name":          "BacktestEngineer",
            "display_name":  "Backtest Engineer",
            "subtitle":      "Vectorised NumPy backtest, K-fold validation",
            "status":        "ACTIVE" if bt_is_active else "IDLE",
            "pending_tasks": bt_queued,
            "current_task":  bt_current_task,
            "last_action":   bt_last_action,
            "last_updated":  _ts_ago(bt_last_ts),
            "working_on":    bt_working_on,
            "pipeline_stage": "stage2" if bt_pipeline_active else ("stage3" if bt_ts_active else ""),
        },
        {
            "name":          "RiskMonitor",
            "display_name":  "Risk Monitor",
            "subtitle":      "Drawdown monitoring, pipeline health, alerts",
            "status":        "ACTIVE",
            "pending_tasks": rm_errors_1h,
            "current_task":  rm_task,
            "last_action":   rm_task,
            "last_updated":  _ts_ago(rm_last_ts),
            "working_on":    "",
            "pipeline_stage": "",
        },
        {
            "name":          "PortfolioExecutor",
            "display_name":  "Portfolio Executor",
            "subtitle":      "Paper trading, position sizing, exit management",
            "status":        "ACTIVE" if pe_is_active else "IDLE",
            "pending_tasks": pe_stage4_ideas,
            "current_task":  pe_task,
            "last_action":   pe_task,
            "last_updated":  _ts_ago(pe_last_ts),
            "working_on":    pe_working_on,
            "pipeline_stage": "stage4a" if pe_stage4_ideas > 0 else "",
        },
    ]

    return {
        "agents":        agents,
        "total_active":  sum(1 for a in agents if a["status"] == "ACTIVE"),
        "total_pending": sum(a["pending_tasks"] for a in agents),
        "as_of":         ts_now,
    }


class AdvanceBody(BaseModel):
    action: str = "advance"  # advance | reject
    notes: str = ""


@app.post("/api/pipeline/ideas/{idea_id}/advance")
def advance_idea(idea_id: int, body: AdvanceBody):
    stage_map = {
        "gate0": "stage1", "stage1": "stage2", "stage2": "stage3",
        "stage3": "stage4a", "stage4a": "stage4b", "stage4b": "stage5",
    }
    with db_session() as conn:
        row = conn.execute("SELECT * FROM alpha_ideas WHERE id=?", (idea_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Idea not found")
        if body.action == "advance":
            next_stage = stage_map.get(row["stage"], row["stage"])
            conn.execute("UPDATE alpha_ideas SET stage=?, status='active', updated_at=datetime('now') WHERE id=?", (next_stage, idea_id))
            conn.execute("INSERT INTO pipeline_events (idea_id,stage,event_type,agent,notes) VALUES (?,?,'advanced','dashboard',?)",
                         (idea_id, row["stage"], body.notes or "Manual advance via dashboard"))
            conn.execute("INSERT INTO gate_decisions (idea_id,gate,decision,decided_by,rationale) VALUES (?,?,'approve','dashboard',?)",
                         (idea_id, row["stage"], body.notes or "Manual approval"))
        else:
            conn.execute("UPDATE alpha_ideas SET status='rejected', updated_at=datetime('now') WHERE id=?", (idea_id,))
            conn.execute("INSERT INTO pipeline_events (idea_id,stage,event_type,agent,notes) VALUES (?,?,'rejected','dashboard',?)",
                         (idea_id, row["stage"], body.notes or "Manual rejection via dashboard"))
    return {"ok": True, "idea_id": idea_id, "action": body.action}


# ─── Backtest Lab ─────────────────────────────────────────────────────────────

@app.get("/api/backtest/runs")
def backtest_runs(idea_id: Optional[int] = None, limit: int = 50):
    with db_session() as conn:
        where, params = [], []
        if idea_id:
            where.append("br.idea_id=?"); params.append(idea_id)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        rows = conn.execute(f"""
            SELECT br.*, ai.title, ai.ticker as idea_ticker, ai.timeframe as idea_tf
            FROM backtest_runs br
            JOIN alpha_ideas ai ON ai.id = br.idea_id
            {clause} ORDER BY br.id DESC LIMIT ?
        """, params + [limit]).fetchall()
    return {"runs": [dict(r) for r in rows]}


class BacktestTrigger(BaseModel):
    idea_id: int


@app.post("/api/backtest/trigger")
async def trigger_backtest(body: BacktestTrigger):
    def _run():
        from agents.backtest_engineer.backtest_engineer import BacktestEngineer
        return BacktestEngineer().run({"action": "backtest", "idea_id": body.idea_id})
    result = await _in_thread(_run)
    return result


# ─── Backtest Lab — Detail, List, Decision ────────────────────────────────────

@app.get("/api/backtest/list")
def backtest_list():
    """List ideas that have at least one backtest run (most recent run per idea)."""
    with db_session() as conn:
        rows = conn.execute("""
            SELECT ai.id, ai.title, ai.ticker, ai.timeframe, ai.stage, ai.status,
                   br.id AS bt_id, br.sharpe_gross, br.sharpe_net,
                   br.sharpe_is, br.sharpe_oos, br.oos_degradation,
                   br.trade_count, br.regimes_positive, br.passed,
                   br.verdict, br.sanity_flags, br.holding_period_class,
                   br.test_dd, br.win_rate,
                   br.created_at AS bt_date
            FROM alpha_ideas ai
            JOIN (
                SELECT idea_id, MAX(id) AS latest_id
                FROM backtest_runs GROUP BY idea_id
            ) latest ON latest.idea_id = ai.id
            JOIN backtest_runs br ON br.id = latest.latest_id
            ORDER BY br.id DESC
        """).fetchall()
    return {"ideas": [dict(r) for r in rows], "total": len(rows)}


def _build_bt_verdict(bt: dict) -> tuple:
    sn  = bt.get("sharpe_net")  or 0.0
    so  = bt.get("sharpe_oos")  or 0.0
    rp  = bt.get("regimes_positive")
    tc  = bt.get("trade_count") or 0
    dd  = bt.get("test_dd")     or bt.get("val_dd") or 0.0
    flags = bt.get("sanity_flags") or ""
    passed = bt.get("passed", 0)
    if passed:
        parts = [f"Net Sharpe {sn:.2f} clears threshold"]
        if so > 0:              parts.append(f"OOS Sharpe {so:.2f} positive")
        if rp and rp >= 2:     parts.append(f"Robust across {rp}/3 vol regimes")
        if tc >= 30:            parts.append(f"{tc} trades (adequate sample)")
        return "PASS", "; ".join(parts)
    else:
        issues = []
        if sn < 1.0:           issues.append(f"Net Sharpe {sn:.2f} < 1.0")
        if so < 0:             issues.append(f"OOS Sharpe {so:.2f} negative (overfitting risk)")
        if rp is not None and rp < 2: issues.append(f"Only {rp}/3 regimes positive")
        if tc < 30:            issues.append(f"Only {tc} trades (need ≥30)")
        if dd > 0.25:          issues.append(f"Drawdown {dd:.1%} > 25%")
        if flags:              issues.append(f"Sanity: {flags}")
        return "FAIL", "; ".join(issues) if issues else "Failed quality gates"


@app.get("/api/backtest/{idea_id}")
def get_backtest_detail(idea_id: int):
    """Full backtest detail: idea + all runs + equity series + computed verdict."""
    with db_session() as conn:
        idea = conn.execute("SELECT * FROM alpha_ideas WHERE id=?", (idea_id,)).fetchone()
        if not idea:
            raise HTTPException(status_code=404, detail="Idea not found")
        idea = dict(idea)
        runs = conn.execute(
            "SELECT * FROM backtest_runs WHERE idea_id=? ORDER BY id DESC", (idea_id,)
        ).fetchall()
        runs = [dict(r) for r in runs]
        series = conn.execute(
            "SELECT date, strategy_pct, benchmark_pct, drawdown_pct, is_oos "
            "FROM backtest_series WHERE idea_id=? ORDER BY date", (idea_id,)
        ).fetchall()
        series = [dict(r) for r in series]
    latest = runs[0] if runs else {}
    verdict = latest.get("verdict") or ""
    verdict_reason = latest.get("verdict_reason") or ""
    if not verdict and latest:
        verdict, verdict_reason = _build_bt_verdict(latest)
    return {
        "idea": idea, "runs": runs, "latest": latest,
        "series": series, "verdict": verdict, "verdict_reason": verdict_reason,
    }


class BtDecisionBody(BaseModel):
    decision: str   # "approve" or "reject"
    notes: str = ""


@app.post("/api/backtest/{idea_id}/decision")
def backtest_decision(idea_id: int, body: BtDecisionBody):
    if body.decision not in ("approve", "reject"):
        raise HTTPException(status_code=400, detail="decision must be 'approve' or 'reject'")
    with db_session() as conn:
        idea = conn.execute("SELECT id, stage FROM alpha_ideas WHERE id=?", (idea_id,)).fetchone()
        if not idea:
            raise HTTPException(status_code=404, detail="Idea not found")
        rationale = body.notes or (
            "Human approval via Backtest Lab" if body.decision == "approve"
            else "Human rejection via Backtest Lab"
        )
        conn.execute("""
            UPDATE backtest_runs SET verdict=?, verdict_reason=?
            WHERE id=(SELECT MAX(id) FROM backtest_runs WHERE idea_id=?)
        """, (body.decision.upper(), rationale, idea_id))
        conn.execute("""
            INSERT INTO gate_decisions (idea_id, gate, decision, decided_by, rationale)
            VALUES (?, 'gate3', ?, 'dashboard', ?)
        """, (idea_id, body.decision, rationale))
        if body.decision == "approve":
            conn.execute("""
                UPDATE alpha_ideas
                SET stage='stage4a', status='active', updated_at=datetime('now')
                WHERE id=?
            """, (idea_id,))
            conn.execute("""
                INSERT INTO pipeline_events (idea_id, stage, event_type, agent, notes)
                VALUES (?, 'stage4a', 'advanced', 'dashboard',
                        'Backtest approved — advancing to paper trading')
            """, (idea_id,))
    return {"ok": True, "idea_id": idea_id, "decision": body.decision}


# ─── Factor Sandbox ───────────────────────────────────────────────────────────

class SandboxBody(BaseModel):
    title: str
    hypothesis: str = ""
    ticker: str = "1155.KL"
    timeframe: str = "1d"
    factor_formula: str


@app.post("/api/sandbox/run")
async def sandbox_run(body: SandboxBody):
    import re
    from data.database import db_session

    slug = re.sub(r"[^a-z0-9]+", "-", body.title.lower()).strip("-")
    slug = f"sandbox-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{slug[:40]}"

    with db_session() as conn:
        conn.execute("""
            INSERT OR IGNORE INTO alpha_ideas
            (slug, title, hypothesis, ticker, timeframe, factor_formula, data_sources, stage, status, novelty_score, logic_score)
            VALUES (?, ?, ?, ?, ?, ?, '[]', 'stage2', 'active', 0.7, 0.7)
        """, (slug, body.title, body.hypothesis, body.ticker, body.timeframe, body.factor_formula))
        row = conn.execute("SELECT id FROM alpha_ideas WHERE slug=?", (slug,)).fetchone()
        idea_id = row["id"]

    def _run():
        from agents.backtest_engineer.backtest_engineer import BacktestEngineer
        return BacktestEngineer().run({"action": "backtest", "idea_id": idea_id})

    result = await _in_thread(_run)
    result["idea_id"] = idea_id
    result["slug"] = slug
    return result


# ─── Paper Trades ─────────────────────────────────────────────────────────────

@app.get("/api/paper-trades")
def paper_trades(idea_id: Optional[int] = None, status: Optional[str] = None, limit: int = 100):
    with db_session() as conn:
        where, params = [], []
        if idea_id: where.append("pt.idea_id=?"); params.append(idea_id)
        if status:  where.append("pt.status=?");  params.append(status)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        rows = conn.execute(f"""
            SELECT pt.*, ai.title as idea_title
            FROM paper_trades pt
            JOIN alpha_ideas ai ON ai.id = pt.idea_id
            {clause} ORDER BY pt.id DESC LIMIT ?
        """, params + [limit]).fetchall()
        stats = conn.execute("""
            SELECT COUNT(*) as total,
                   COALESCE(SUM(CASE WHEN status='open' THEN 1 ELSE 0 END), 0) as open_count,
                   COALESCE(SUM(CASE WHEN status='closed' THEN 1 ELSE 0 END), 0) as closed_count,
                   COALESCE(SUM(CASE WHEN status='closed' THEN pnl ELSE 0 END), 0) as total_pnl,
                   COALESCE(SUM(CASE WHEN status='closed' AND pnl>0 THEN 1 ELSE 0 END), 0) as wins
            FROM paper_trades
        """).fetchone()
    return {
        "trades": [dict(r) for r in rows],
        "stats": dict(stats),
    }


class PaperEntryBody(BaseModel):
    idea_id: int
    pair: str
    direction: str  # long | short
    signal: str = ""
    stop_pips: int = 25


@app.post("/api/paper-trades/entry")
async def paper_entry(body: PaperEntryBody):
    def _run():
        from agents.portfolio_executor.portfolio_executor import PortfolioExecutor
        return PortfolioExecutor().run({
            "action": "paper_entry",
            "idea_id": body.idea_id,
            "pair": body.pair,
            "direction": body.direction,
            "signal": body.signal,
            "stop_pips": body.stop_pips,
        })
    return await _in_thread(_run)


@app.put("/api/paper-trades/{trade_id}/exit")
async def paper_exit(trade_id: int):
    def _run():
        from agents.portfolio_executor.portfolio_executor import PortfolioExecutor
        return PortfolioExecutor().run({"action": "paper_exit", "trade_id": trade_id})
    return await _in_thread(_run)


@app.get("/api/paper-trades/evaluate/{idea_id}")
async def evaluate_paper(idea_id: int):
    def _run():
        from agents.portfolio_executor.portfolio_executor import PortfolioExecutor
        return PortfolioExecutor().run({"action": "evaluate_paper", "idea_id": idea_id})
    return await _in_thread(_run)


# ─── Knowledge Base ───────────────────────────────────────────────────────────

@app.get("/api/kb/stats")
def kb_stats():
    with db_session() as conn:
        docs     = conn.execute("SELECT COUNT(*) as n FROM kb_documents").fetchone()["n"]
        concepts = conn.execute("SELECT COUNT(*) as n FROM kb_concepts").fetchone()["n"]
        links    = conn.execute("SELECT COUNT(*) as n FROM kb_links").fetchone()["n"]
        by_domain = conn.execute("SELECT domain, COUNT(*) as n FROM kb_documents GROUP BY domain").fetchall()
        top_concepts = conn.execute("SELECT name, domain, count FROM kb_concepts ORDER BY count DESC LIMIT 20").fetchall()
    return {
        "total_documents": docs,
        "total_concepts": concepts,
        "total_links": links,
        "by_domain": {r["domain"]: r["n"] for r in by_domain},
        "top_concepts": [dict(r) for r in top_concepts],
    }


@app.get("/api/kb/documents")
def kb_documents(domain: Optional[str] = None, limit: int = 50, offset: int = 0):
    with db_session() as conn:
        where, params = [], []
        if domain: where.append("domain=?"); params.append(domain)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        rows  = conn.execute(f"SELECT id, slug, title, domain, summary, tags, source_url, created_at FROM kb_documents {clause} ORDER BY id DESC LIMIT ? OFFSET ?", params + [limit, offset]).fetchall()
        total = conn.execute(f"SELECT COUNT(*) as n FROM kb_documents {clause}", params).fetchone()["n"]
    return {"total": total, "documents": [dict(r) for r in rows]}


@app.get("/api/kb/search")
def kb_search(q: str, domain: Optional[str] = None, limit: int = 20):
    if not q.strip():
        return {"results": []}
    terms = [t.strip() for t in q.lower().split() if len(t.strip()) > 2]
    if not terms:
        return {"results": []}
    like_clauses = " AND ".join([f"(LOWER(title) LIKE ? OR LOWER(summary) LIKE ? OR LOWER(tags) LIKE ?)" for _ in terms])
    params = []
    for t in terms:
        like = f"%{t}%"
        params += [like, like, like]
    sql = f"SELECT id, slug, title, domain, summary, tags, source_url, created_at FROM kb_documents WHERE {like_clauses}"
    if domain:
        sql += " AND domain=?"
        params.append(domain)
    sql += f" ORDER BY updated_at DESC LIMIT {limit}"
    with db_session() as conn:
        rows = conn.execute(sql, params).fetchall()
    return {"results": [dict(r) for r in rows]}


@app.get("/api/kb/concepts")
def kb_concepts(q: Optional[str] = None, limit: int = 50):
    with db_session() as conn:
        if q:
            like = f"%{q.lower()}%"
            rows = conn.execute("SELECT * FROM kb_concepts WHERE LOWER(name) LIKE ? OR LOWER(description) LIKE ? ORDER BY count DESC LIMIT ?", (like, like, limit)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM kb_concepts ORDER BY count DESC LIMIT ?", (limit,)).fetchall()
    return {"concepts": [dict(r) for r in rows]}


class KBIngestBody(BaseModel):
    content: str
    title: str
    domain: str = "other"
    source_url: str = ""


@app.post("/api/kb/ingest")
async def kb_ingest(body: KBIngestBody):
    def _run():
        from knowledge.ingestion.kb_ingester import KBIngester
        return KBIngester().run({
            "action": "ingest_text",
            "content": body.content,
            "title": body.title,
            "domain": body.domain,
            "source_url": body.source_url,
        })
    return await _in_thread(_run)


class KBIngestURLBody(BaseModel):
    url: str
    title: str = ""
    domain: str = "other"


@app.post("/api/kb/ingest-url")
async def kb_ingest_url(body: KBIngestURLBody):
    def _run():
        from knowledge.ingestion.kb_ingester import KBIngester
        return KBIngester().run({"action": "ingest_url", "url": body.url, "title": body.title, "domain": body.domain})
    return await _in_thread(_run)


@app.post("/api/kb/ingest-pdf")
async def kb_ingest_pdf(file: UploadFile = File(...)):
    import io
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    pdf_bytes = await file.read()
    if len(pdf_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    def _run():
        import pdfplumber
        from knowledge.ingestion.kb_ingester import KBIngester

        # Extract text from all pages, cap at 50 000 chars
        text_parts = []
        title = ""
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text() or ""
                    text_parts.append(page_text)
                    # Use first non-empty line of page 0 as title fallback
                    if i == 0 and not title:
                        for line in page_text.splitlines():
                            line = line.strip()
                            if len(line) > 5:
                                title = line[:120]
                                break
        except Exception as e:
            return {"error": f"PDF extraction failed: {e}"}

        full_text = "\n\n".join(text_parts)[:50000]
        if not full_text.strip():
            return {"error": "Could not extract any text from PDF"}

        # Fall back to filename (without extension) if no title found on page 1
        if not title:
            title = file.filename.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").title()

        source_url = f"pdf_upload:{file.filename}"
        return KBIngester().ingest_text(content=full_text, title=title, source_url=source_url)

    return await _in_thread(_run)


# ─── KB Management (delete / reassign) ──────────────────────────────────────

_KB_VALID_DOMAINS = {
    "price_action", "fundamental", "event_driven", "institutional",
    "macro", "commodity", "sector_rotation", "behavioural", "statistical_modelling",
}


@app.delete("/api/kb/document/{doc_id}")
def kb_delete_document(doc_id: int):
    with db_session() as conn:
        row = conn.execute("SELECT id, title FROM kb_documents WHERE id=?", (doc_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Document not found")
        conn.execute("DELETE FROM kb_links WHERE source_id=? OR target_id=?", (doc_id, doc_id))
        conn.execute("DELETE FROM kb_documents WHERE id=?", (doc_id,))
    return {"deleted": True, "doc_id": doc_id}


class KBBulkDeleteBody(BaseModel):
    doc_ids: list


@app.delete("/api/kb/documents/bulk")
def kb_delete_bulk(body: KBBulkDeleteBody):
    ids = [int(i) for i in body.doc_ids if str(i).lstrip("-").isdigit()]
    if not ids:
        return {"deleted": 0, "doc_ids": []}
    placeholders = ",".join("?" for _ in ids)
    with db_session() as conn:
        for doc_id in ids:
            conn.execute("DELETE FROM kb_links WHERE source_id=? OR target_id=?", (doc_id, doc_id))
        conn.execute(f"DELETE FROM kb_documents WHERE id IN ({placeholders})", ids)
    return {"deleted": len(ids), "doc_ids": ids}


class KBDomainUpdateBody(BaseModel):
    domain: str


@app.patch("/api/kb/document/{doc_id}/domain")
def kb_update_domain(doc_id: int, body: KBDomainUpdateBody):
    if body.domain not in _KB_VALID_DOMAINS:
        raise HTTPException(status_code=400, detail=f"Invalid domain: {body.domain}")
    with db_session() as conn:
        row = conn.execute("SELECT id FROM kb_documents WHERE id=?", (doc_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Document not found")
        conn.execute(
            "UPDATE kb_documents SET domain=?, updated_at=datetime('now') WHERE id=?",
            (body.domain, doc_id),
        )
    return {"updated": True, "doc_id": doc_id, "domain": body.domain}


@app.delete("/api/kb/concept/{concept_id}")
def kb_delete_concept(concept_id: int):
    with db_session() as conn:
        row = conn.execute("SELECT id FROM kb_concepts WHERE id=?", (concept_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Concept not found")
        conn.execute("DELETE FROM kb_concepts WHERE id=?", (concept_id,))
    return {"deleted": True, "concept_id": concept_id}


# ─── Logs & Usage ────────────────────────────────────────────────────────────

@app.get("/api/logs")
def get_logs(level: Optional[str] = None, limit: int = 100):
    with db_session() as conn:
        where, params = [], []
        if level and level != "ALL":
            where.append("level=?"); params.append(level.upper())
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        logs = conn.execute(f"SELECT * FROM daemon_logs {clause} ORDER BY id DESC LIMIT ?", params + [limit]).fetchall()
    return {"logs": [dict(r) for r in logs]}


@app.get("/api/ai-usage")
def ai_usage(days: int = 7):
    with db_session() as conn:
        since = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        usage = conn.execute("SELECT date(created_at) as day, model, agent, SUM(cost_usd) as cost, COUNT(*) as calls FROM ai_usage WHERE created_at >= ? GROUP BY day, model, agent ORDER BY day DESC", (since,)).fetchall()
    return {"usage": [dict(r) for r in usage]}


# ─── System Direction ────────────────────────────────────────────────────────

@app.get("/api/system/direction")
def system_direction():
    """Return the OpenClaw system direction document as structured JSON,
    with live KB angle coverage pulled from the database."""
    with db_session() as conn:
        kb_by_domain = conn.execute(
            "SELECT domain, COUNT(*) as n FROM kb_documents GROUP BY domain"
        ).fetchall()
        total_kb = conn.execute("SELECT COUNT(*) as n FROM kb_documents").fetchone()["n"]
        total_ideas = conn.execute("SELECT COUNT(*) as n FROM alpha_ideas").fetchone()["n"]
        today = datetime.utcnow().strftime("%Y-%m-%d")
        today_spend = float(conn.execute(
            "SELECT COALESCE(SUM(cost_usd),0) as t FROM ai_usage WHERE created_at LIKE ?",
            (f"{today}%",)
        ).fetchone()["t"])

    domain_counts = {r["domain"]: r["n"] for r in kb_by_domain}

    angles = [
        {"id": "price_action",         "label": "Price Action",         "description": "Technical signals, momentum, mean reversion"},
        {"id": "fundamental",           "label": "Fundamental",          "description": "PE, ROE, earnings, valuation, dividends"},
        {"id": "event_driven",          "label": "Event Driven",         "description": "PEAD, dividend capture, announcements"},
        {"id": "institutional",         "label": "Institutional",        "description": "EPF, KWAP, GLC, foreign fund flows"},
        {"id": "macro",                 "label": "Macro",                "description": "OPR, BNM, GDP, inflation, MYR, global macro"},
        {"id": "commodity",             "label": "Commodity",            "description": "CPO, palm oil, crude oil, aluminium, tin"},
        {"id": "sector_rotation",       "label": "Sector Rotation",      "description": "Sector cycles, thematic investing"},
        {"id": "behavioural",           "label": "Behavioural",          "description": "Retail sentiment, overreaction, anomalies"},
        {"id": "statistical_modelling", "label": "Statistical Modelling","description": "GARCH, HMM, factor models, ML"},
    ]
    for a in angles:
        a["doc_count"] = domain_counts.get(a["id"], 0)
        a["target"] = 5   # docs-per-angle minimum before idea generation
        a["coverage_pct"] = min(100, round(a["doc_count"] / a["target"] * 100))
        a["ready"] = a["doc_count"] >= a["target"]

    gate_thresholds = {
        "gate0":           {"novelty": 0.60, "logic": 0.70, "feasibility": 0.60},
        "stage2_sharpe":   1.1,
        "stage2_tv_gap":   0.30,
        "cross_section_ic": 0.05,
        "cross_section_tstat": 1.5,
        "cross_section_stocks": 15,
        "stage4a_sharpe":  1.0,
        "stage4a_max_dd":  0.15,
    }

    min_trades = {
        "INTRADAY":    {"min": 100, "note": "flag as indicative only"},
        "SHORT_TERM":  {"min": 50,  "note": "1–10 days"},
        "MEDIUM_TERM": {"min": 30,  "note": "10–60 days"},
        "LONG_TERM":   {"min": 15,  "note": ">60 days"},
    }

    transaction_costs = {
        "commission_pct": 0.08,
        "stamp_duty_pct": 0.15,
        "stamp_duty_cap_myr": 200,
        "clearing_pct": 0.03,
        "clearing_cap_myr": 1000,
        "slippage": {"BLUE_CHIP": 0.05, "MID_CAP": 0.25, "SMALL_CAP": 0.75},
        "min_liquidity_myr": 500000,
    }

    known_issues = [
        {"status": "fixed",   "issue": "load_dotenv() not called — all API keys were empty"},
        {"status": "fixed",   "issue": "Backtest infinite loop — status not set to processing"},
        {"status": "fixed",   "issue": "_link_document_concept() FK bug — links not created"},
        {"status": "fixed",   "issue": "FX contamination — strategy_researcher had forex prompts"},
        {"status": "fixed",   "issue": "KB garbage ingestion — no relevance filter existed"},
        {"status": "fixed",   "issue": "Gate 0 feasibility missing — only novelty+logic scored"},
        {"status": "fixed",   "issue": "Rejection memory missing — system blind to past failures"},
        {"status": "fixed",   "issue": "Red-Blue debate not Bursa-grounded — generic debate"},
        {"status": "fixed",   "issue": "Formula verification missing — code not checked vs text"},
        {"status": "fixed",   "issue": "Domain classification inconsistent — two systems existed"},
        {"status": "fixed",   "issue": "Gate 0 scoring bug — novelty/logic always 0.00 (JSON parse failure + silent 0.0 fallback)"},
        {"status": "pending", "issue": "Cross-sectional validation fully wired into pipeline"},
        {"status": "pending", "issue": "Broker connection for paper/live trading"},
        {"status": "pending", "issue": "SSL/HTTPS for dashboard"},
        {"status": "pending", "issue": "D3 knowledge graph (when KB hits 200+ docs)"},
    ]

    bursa_constraints = [
        "Long-only strategies only (short-selling heavily restricted)",
        "T+3 settlement — affects short-term strategy feasibility",
        "Minimum lot size: 100 shares (affects small-cap liquidity)",
        "Stamp duty: 0.15% buy-side, capped RM200 (real cost)",
        "Brokerage: ~0.08% per side minimum",
        "Trading hours: 9:00–12:30 and 14:30–17:00 MYT only",
        "Circuit breakers: halt if stock moves >30% in a day",
        "EPF dominates: ~15% of market cap, rebalancing is predictable",
        "OPR sensitivity: banking stocks move with BNM rate decisions",
        "CPO correlation: plantation stocks follow palm oil futures",
    ]

    angles_ready = sum(1 for a in angles if a["ready"])

    return {
        "last_updated": "April 2026",
        "core_purpose": "Find genuine, statistically robust alpha factors in Bursa Malaysia equity markets. Prove them cross-sectionally. Deploy them safely with human oversight at every capital decision point.",
        "design_philosophy": "Quality over quantity. 10 robust, well-validated strategies beats 300 hastily generated noise ideas. Every component must earn its place. The system should get smarter every day, not just bigger.",
        "success_metrics": [
            {"rank": 1, "metric": "First idea reaches Stage 3 with IC > 0.05 across 15+ stocks"},
            {"rank": 2, "metric": "First idea completes 30-day paper trade with Sharpe >= 1.0"},
            {"rank": 3, "metric": "First live strategy deployed with positive alpha after costs"},
            {"rank": 4, "metric": "KB reaches 50 quality docs across all 9 research angles"},
            {"rank": 5, "metric": "Daily budget stays under $10 while pipeline processes meaningful ideas"},
        ],
        "research_angles": angles,
        "angles_ready_count": angles_ready,
        "angles_total": len(angles),
        "gate_thresholds": gate_thresholds,
        "min_trades": min_trades,
        "transaction_costs": transaction_costs,
        "bursa_constraints": bursa_constraints,
        "known_issues": known_issues,
        "system_state": {
            "total_kb_docs": total_kb,
            "total_ideas": total_ideas,
            "today_spend_usd": round(today_spend, 4),
            "daily_budget_usd": AI_DAILY_BUDGET_USD,
            "kb_target": 50,
            "ideas_target": 45,
        },
    }


# ─── Arsenal ─────────────────────────────────────────────────────────────────

@app.get("/api/system/arsenal")
def system_arsenal(angle: Optional[str] = None):
    """Return all quantitative techniques from TechniqueLibrary, enriched with
    strategy profile data (exit logic, hold rationale, IC benchmarks) where available.
    """
    from knowledge.ingestion.technique_library import TechniqueLibrary
    lib = TechniqueLibrary()
    techniques = lib.to_api_list()

    # ── Enrich with strategy_profiles data ───────────────────────────────────
    try:
        with db_session() as conn:
            profiles = conn.execute("SELECT * FROM strategy_profiles").fetchall()
        profile_map = {dict(r)["strategy_key"]: dict(r) for r in profiles}
    except Exception:
        profile_map = {}

    # Map technique library keys → strategy_profile keys (where they differ)
    _KEY_ALIAS: dict[str, str] = {
        "hidden_markov_model":  "hmm_regime_detector",
        "garch":                "garch_volatility_overlay",
        "bollinger_squeeze":    "bollinger_squeeze_breakout",
        "information_coefficient": "cross_sectional_momentum",
    }

    for t in techniques:
        key = t.get("key", "")
        sp  = profile_map.get(key) or profile_map.get(_KEY_ALIAS.get(key, ""))
        if sp:
            t["phenomenon"]        = sp.get("phenomenon")
            t["bursa_nuance"]      = sp.get("bursa_nuance")
            t["entry_condition"]   = sp.get("entry_condition")
            t["entry_universe"]    = sp.get("entry_universe")
            t["entry_rebalance"]   = sp.get("entry_rebalance")
            t["exit_type"]         = sp.get("exit_type")
            t["exit_condition"]    = sp.get("exit_condition")
            t["exit_rationale"]    = sp.get("exit_rationale")
            t["stop_loss_pct"]     = sp.get("stop_loss_pct")
            t["profit_target_pct"] = sp.get("profit_target_pct")
            t["min_hold_days"]     = sp.get("min_hold_days")
            t["max_hold_days"]     = sp.get("max_hold_days")
            t["hold_rationale"]    = sp.get("hold_rationale")
            t["use_when"]          = sp.get("use_when")
            t["avoid_when"]        = sp.get("avoid_when")
            # override ic_benchmark with richer profile version if available
            if sp.get("ic_benchmark"):
                t["ic_benchmark"]  = sp.get("ic_benchmark")
        else:
            # Ensure fields always present in response even without a profile
            for field in ("phenomenon", "bursa_nuance", "entry_condition",
                          "entry_universe", "entry_rebalance", "exit_type",
                          "exit_condition", "exit_rationale", "stop_loss_pct",
                          "profit_target_pct", "min_hold_days", "max_hold_days",
                          "hold_rationale"):
                t.setdefault(field, None)

    if angle:
        techniques = [t for t in techniques if t["angle"] == angle]
    implemented   = sum(1 for t in techniques if t["implemented"])
    total         = len(techniques)
    by_angle: dict = {}
    for t in techniques:
        by_angle.setdefault(t["angle"], []).append(t)
    return {
        "total":       total,
        "implemented": implemented,
        "pending":     total - implemented,
        "by_angle":    by_angle,
        "techniques":  techniques,
    }


# ─── Event Engine ────────────────────────────────────────────────────────────

@app.get("/api/events")
def get_events(limit: int = 50, event_type: str = "all", action: str = "all", hours: int = 24):
    """Return recent market_events ordered by detected_at DESC."""
    since = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
    with db_session() as conn:
        query = "SELECT * FROM market_events WHERE detected_at >= ?"
        params: list = [since]
        if event_type != "all":
            query += " AND event_type = ?"
            params.append(event_type)
        if action != "all":
            query += " AND action_taken = ?"
            params.append(action)
        query += " ORDER BY detected_at DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


@app.get("/api/events/stats")
def get_event_stats():
    """Return event engine stats for today."""
    today = datetime.utcnow().strftime("%Y-%m-%d")
    with db_session() as conn:
        total_today = conn.execute(
            "SELECT COUNT(*) as n FROM market_events WHERE detected_at LIKE ?",
            (f"{today}%",)
        ).fetchone()["n"]

        gate0_today = conn.execute(
            "SELECT COUNT(*) as n FROM market_events WHERE detected_at LIKE ? AND action_taken='gate0_idea'",
            (f"{today}%",)
        ).fetchone()["n"]

        alerts_today = conn.execute(
            "SELECT COUNT(*) as n FROM market_events WHERE detected_at LIKE ? AND action_taken='alert'",
            (f"{today}%",)
        ).fetchone()["n"]

        kb_today = conn.execute(
            "SELECT COUNT(*) as n FROM market_events WHERE detected_at LIKE ? AND action_taken='kb_only'",
            (f"{today}%",)
        ).fetchone()["n"]

        by_type = {
            r["event_type"]: r["n"]
            for r in conn.execute(
                "SELECT event_type, COUNT(*) as n FROM market_events WHERE detected_at LIKE ? GROUP BY event_type",
                (f"{today}%",)
            ).fetchall()
        }

        by_source = {
            r["source"]: r["n"]
            for r in conn.execute(
                "SELECT source, COUNT(*) as n FROM market_events WHERE detected_at LIKE ? GROUP BY source",
                (f"{today}%",)
            ).fetchall()
        }

        # Watcher heartbeat — last EventWatcher log
        last_cycle_row = conn.execute(
            "SELECT created_at FROM daemon_logs WHERE source='EventWatcher' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        last_cycle = last_cycle_row["created_at"] if last_cycle_row else None

    return {
        "total_today": total_today,
        "gate0_ideas_today": gate0_today,
        "alerts_sent_today": alerts_today,
        "kb_ingested_today": kb_today,
        "by_type": by_type,
        "by_source": by_source,
        "last_cycle": last_cycle,
    }


@app.get("/api/events/calendar")
def get_event_calendar(days_ahead: int = 30):
    """Return upcoming economic calendar events."""
    today = datetime.utcnow().strftime("%Y-%m-%d")
    cutoff = (datetime.utcnow() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
    with db_session() as conn:
        rows = conn.execute("""
            SELECT * FROM economic_calendar
            WHERE scheduled_date >= ? AND scheduled_date <= ?
            ORDER BY scheduled_date, scheduled_time
        """, (today, cutoff)).fetchall()
    return [dict(r) for r in rows]


# ─── Static UI ───────────────────────────────────────────────────────────────

ui_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ui")
if os.path.exists(ui_path):
    app.mount("/", StaticFiles(directory=ui_path, html=True), name="ui")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("dashboard.api.server:app", host="0.0.0.0", port=8001, reload=True)
