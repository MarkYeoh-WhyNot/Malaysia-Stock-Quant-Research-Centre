import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from data.database import db_session, init_db
from config.settings import AI_DAILY_BUDGET_USD

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
    return {"status": "ok", "version": "2.0.0", "time": datetime.utcnow().isoformat()}


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
    stage_map = {r["stage"]: r["n"] for r in stages}
    return {
        "pipeline": {"total_ideas": totals["total"], "total_rejected": totals["rejected"], "live_strategies": totals["live"], "stages": stage_map},
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

        # Pair distribution
        pairs = conn.execute("""
            SELECT pair, COUNT(*) as n,
                   AVG(COALESCE(backtest_sharpe, 0)) as avg_sharpe
            FROM alpha_ideas WHERE pair IS NOT NULL GROUP BY pair ORDER BY n DESC
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
        "pairs": [{"pair": r["pair"], "count": r["n"], "avg_sharpe": round(float(r["avg_sharpe"]), 3)} for r in pairs],
        "pipeline_events": [dict(r) for r in events],
    }


# ─── Pipeline / Ideas ────────────────────────────────────────────────────────

@app.get("/api/pipeline/ideas")
def get_ideas(stage: Optional[str] = None, status: Optional[str] = None, limit: int = 50):
    with db_session() as conn:
        where, params = [], []
        if stage:  where.append("stage=?");  params.append(stage)
        if status: where.append("status=?"); params.append(status)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        ideas  = conn.execute(f"SELECT * FROM alpha_ideas {clause} ORDER BY id DESC LIMIT ?", params + [limit]).fetchall()
        total  = conn.execute(f"SELECT COUNT(*) as n FROM alpha_ideas {clause}", params).fetchone()["n"]
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
        recent_gates = conn.execute("SELECT gd.*, ai.title, ai.pair FROM gate_decisions gd JOIN alpha_ideas ai ON ai.id=gd.idea_id ORDER BY gd.id DESC LIMIT 30").fetchall()
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
    cutoff_5m = (datetime.utcnow() - timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%S")
    cutoff_1h = (datetime.utcnow() - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S")

    with db_session() as conn:
        # StrategyResearcher
        sr_pending = conn.execute(
            "SELECT COUNT(*) as n FROM alpha_ideas WHERE stage='gate0' AND status='pending'"
        ).fetchone()["n"]
        sr_active = conn.execute(
            "SELECT COUNT(*) as n FROM alpha_ideas WHERE stage='stage1' AND status='active'"
        ).fetchone()["n"]
        sr_recent = conn.execute(
            "SELECT title, updated_at FROM alpha_ideas WHERE stage IN ('gate0','stage1') AND status IN ('pending','active') ORDER BY updated_at DESC LIMIT 1"
        ).fetchone()
        sr_is_active = bool(conn.execute(
            "SELECT 1 FROM alpha_ideas WHERE stage IN ('gate0','stage1') AND status IN ('pending','active') AND updated_at >= ? LIMIT 1",
            (cutoff_5m,)
        ).fetchone())

        # DataEngineer
        de_pending = conn.execute(
            "SELECT COUNT(*) as n FROM alpha_ideas WHERE stage='stage2' AND status='active'"
        ).fetchone()["n"]
        de_is_active = bool(conn.execute(
            "SELECT 1 FROM backtest_runs WHERE created_at >= ? LIMIT 1", (cutoff_5m,)
        ).fetchone())
        de_log = conn.execute(
            "SELECT message FROM daemon_logs WHERE source LIKE '%data%' OR message LIKE '%Fetch%' OR message LIKE '%fetch%' ORDER BY id DESC LIMIT 1"
        ).fetchone()

        # BacktestEngineer
        bt_recent = conn.execute(
            "SELECT br.id, ai.title, ai.pair FROM backtest_runs br JOIN alpha_ideas ai ON ai.id=br.idea_id ORDER BY br.id DESC LIMIT 1"
        ).fetchone()
        bt_pending = conn.execute(
            "SELECT COUNT(*) as n FROM alpha_ideas ai WHERE ai.stage='stage2' AND ai.status='active' AND NOT EXISTS (SELECT 1 FROM backtest_runs br WHERE br.idea_id=ai.id)"
        ).fetchone()["n"]
        bt_is_active = bool(conn.execute(
            "SELECT 1 FROM backtest_runs WHERE created_at >= ? LIMIT 1", (cutoff_5m,)
        ).fetchone())

        # RiskMonitor — always active
        rm_alerts = conn.execute(
            "SELECT COUNT(*) as n FROM daemon_logs WHERE level='ERROR' AND created_at >= ?", (cutoff_1h,)
        ).fetchone()["n"]

        # PortfolioExecutor
        pe_pending = conn.execute(
            "SELECT COUNT(*) as n FROM alpha_ideas WHERE stage IN ('stage4a','stage5') AND status='active'"
        ).fetchone()["n"]
        pe_is_active = pe_pending > 0
        pe_recent = conn.execute(
            "SELECT ai.title FROM paper_trades pt JOIN alpha_ideas ai ON ai.id=pt.idea_id WHERE pt.status='open' ORDER BY pt.id DESC LIMIT 1"
        ).fetchone()

    def _task(text, maxlen=40):
        if not text:
            return "Idle"
        return (text[:maxlen] + "…") if len(text) > maxlen else text

    agents = [
        {
            "name": "StrategyResearcher",
            "display_name": "Strategy Researcher",
            "subtitle": "策略研究、文献分析、因子挖掘",
            "status": "ACTIVE" if sr_is_active else "WAITING",
            "pending_tasks": sr_pending + sr_active,
            "current_task": _task(sr_recent["title"] if sr_recent else "Scanning for alpha ideas"),
        },
        {
            "name": "DataEngineer",
            "display_name": "Data Engineer",
            "subtitle": "数据管理、清洗、特征工程",
            "status": "ACTIVE" if de_is_active else "WAITING",
            "pending_tasks": de_pending,
            "current_task": _task(de_log["message"] if de_log else "Monitoring data pipeline"),
        },
        {
            "name": "BacktestEngineer",
            "display_name": "Backtest Engineer",
            "subtitle": "回测框架、量化分析、结果评估",
            "status": "ACTIVE" if bt_is_active else "WAITING",
            "pending_tasks": bt_pending,
            "current_task": _task(
                f"Backtesting {bt_recent['pair']} — {bt_recent['title']}" if bt_recent else "Awaiting factor signals"
            ),
        },
        {
            "name": "RiskMonitor",
            "display_name": "Risk Monitor",
            "subtitle": "风控审查、异常检测、exposure监控",
            "status": "ACTIVE",
            "pending_tasks": rm_alerts,
            "current_task": "Monitoring pipeline health",
        },
        {
            "name": "PortfolioExecutor",
            "display_name": "Portfolio Executor",
            "subtitle": "信号执行、仓位管理、交易路由",
            "status": "ACTIVE" if pe_is_active else "WAITING",
            "pending_tasks": pe_pending,
            "current_task": _task(
                f"Managing position: {pe_recent['title']}" if pe_recent else "Awaiting paper trade signals"
            ),
        },
    ]

    return {
        "agents": agents,
        "total_active": sum(1 for a in agents if a["status"] == "ACTIVE"),
        "total_pending": sum(a["pending_tasks"] for a in agents),
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
            SELECT br.*, ai.title, ai.pair as idea_pair, ai.timeframe as idea_tf
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


# ─── Factor Sandbox ───────────────────────────────────────────────────────────

class SandboxBody(BaseModel):
    title: str
    hypothesis: str = ""
    pair: str = "1155.KL"
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
            (slug, title, hypothesis, pair, timeframe, factor_formula, data_sources, stage, status, novelty_score, logic_score)
            VALUES (?, ?, ?, ?, ?, ?, '[]', 'stage2', 'active', 0.7, 0.7)
        """, (slug, body.title, body.hypothesis, body.pair, body.timeframe, body.factor_formula))
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


# ─── Static UI ───────────────────────────────────────────────────────────────

ui_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ui")
if os.path.exists(ui_path):
    app.mount("/", StaticFiles(directory=ui_path, html=True), name="ui")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("dashboard.api.server:app", host="0.0.0.0", port=8001, reload=True)
