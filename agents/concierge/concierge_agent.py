"""ConciergeAgent — dashboard chat that turns a natural-language idea into a
structured strategy, feeds it through the factor sandbox into the gated pipeline,
and reports progress back conversationally.

Guardrails (deliberate):
  - Ideas enter at stage2 (skip Gate 0) via the shared submit_sandbox_idea(),
    which still runs the feasibility + long-only pre-checks.
  - The toolset can submit ideas and read status ONLY. It has NO tool to approve
    live trading, force a gate pass, or delete — the Stage 4a→4b human gate is
    untouched.
  - Its own daily budget sub-cap (CONCIERGE_DAILY_BUDGET_USD) is checked before
    each turn so chat can't starve the research pipeline.
"""
from __future__ import annotations

import json

from agents.base_agent import BaseAgent, get_agent_daily_spend
from config.settings import (
    CONCIERGE_MODEL, CONCIERGE_DAILY_BUDGET_USD, CONCIERGE_MAX_TOOL_ITERS,
    KLCI_STOCKS, KLCI_BY_SYMBOL, KLCI_SECTORS,
    MARKET_NAME, MARKET_BRIEF, TICKER_EXAMPLE,
)
from data.database import db_session

# First token of the profile's example, e.g. "1155.KL" or "BTC/USDT"
_EXAMPLE_TICKER = TICKER_EXAMPLE.split(" ")[0]

TOOLS = [
    {
        "name": "submit_strategy_idea",
        "description": f"Submit a fully-specified long-only {MARKET_NAME} strategy into "
                       "the factor sandbox. It enters at the backtest stage and the "
                       "pipeline carries it through backtest -> red/blue -> paper "
                       "automatically. Use ONLY after you have a concrete factor_formula "
                       f"and a valid ticker like {_EXAMPLE_TICKER}. "
                       "Never for short-selling/pairs/intraday/derivatives.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Short strategy name"},
                "hypothesis": {"type": "string", "description": "1-2 sentence economic rationale"},
                "ticker": {"type": "string", "description": f"One or more tickers, comma-separated (e.g. {_EXAMPLE_TICKER})"},
                "timeframe": {"type": "string", "description": "Bar/holding timeframe, e.g. 1d, 1wk"},
                "factor_formula": {"type": "string", "description": "Concrete entry/exit rule in terms of price/volume/indicators"},
            },
            "required": ["title", "hypothesis", "ticker", "factor_formula"],
        },
    },
    {
        "name": "get_idea_status",
        "description": "Look up the current pipeline stage, status, latest backtest "
                       "metrics/verdict, and paper performance for one idea by id.",
        "input_schema": {
            "type": "object",
            "properties": {"idea_id": {"type": "integer"}},
            "required": ["idea_id"],
        },
    },
    {
        "name": "list_session_ideas",
        "description": "List the ideas submitted in this chat session with their current stage/status.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "search_knowledge_base",
        "description": f"Search the research knowledge base for relevant {MARKET_NAME} "
                       "strategy knowledge, prior ideas, or rejection lessons before "
                       "proposing a strategy.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "resolve_tickers",
        "description": f"Resolve asset or sector names to {MARKET_NAME} tickers in the "
                       f"tradable universe (e.g. name or group -> {_EXAMPLE_TICKER}).",
        "input_schema": {
            "type": "object",
            "properties": {"names": {"type": "array", "items": {"type": "string"}}},
            "required": ["names"],
        },
    },
]


def _system_prompt() -> str:
    from agents.backtest_engineer import signal_dsl
    universe = "\n".join(
        f"  {s['symbol']} — {s['name']} ({s['sector']})" for s in KLCI_STOCKS)
    return f"""You are the Concierge for Mark's Research Centre — the {MARKET_NAME}
quantitative research pipeline. A human chats with you to test strategy ideas.

YOUR JOB: turn a natural-language idea into a concrete, long-only, daily-or-slower
strategy and submit it via submit_strategy_idea. Then it runs the real pipeline
(backtest -> red/blue debate -> paper trading) and you report progress when asked.

MARKET STRUCTURE YOU OPERATE IN:
{MARKET_BRIEF}

HARD RULES — never violate:
- Long-only ONLY. No short-selling, pairs, long/short, market-neutral,
  derivatives, margin, or leverage.
- No intraday/scalping/HFT — daily bars or slower.
- Instruments from this market's universe only, ticker format like
  {_EXAMPLE_TICKER} (resolve names with resolve_tickers).
- You may submit ideas and report status. You may NOT approve or trigger LIVE
  trading — moving a paper strategy to live is a human-only decision. If asked,
  explain that you can get an idea paper-trading-ready but the human makes the
  live call.
- If an idea is infeasible, say so plainly instead of forcing it through.

Write factor_formula in terms the backtester can parse — prefer these conditions:
{signal_dsl.leaf_catalog_text()}

Tradable universe:
{universe}

Sectors: {', '.join(KLCI_SECTORS)}

Be concise and concrete. When you submit an idea, tell the user its id and that
you're tracking it. When they ask how it's doing, call get_idea_status."""


class ConciergeAgent(BaseAgent):
    name = "Concierge"
    description = "Chat concierge: NL idea -> factor sandbox -> pipeline, with status reporting"
    default_model = CONCIERGE_MODEL

    # ── Tool implementations ────────────────────────────────────────────────
    def _tool_resolve_tickers(self, names: list) -> dict:
        matches = {}
        for name in names or []:
            low = name.lower().strip()
            hits = []
            # exact/substring on company name
            for s in KLCI_STOCKS:
                if low in s["name"].lower() or low == s["symbol"].lower():
                    hits.append(s["symbol"])
            # sector match (e.g. "banks" -> Banking names)
            for sector in KLCI_SECTORS:
                if low.rstrip("s") in sector.lower():
                    hits += [s["symbol"] for s in KLCI_STOCKS if s["sector"] == sector]
            matches[name] = sorted(set(hits))
        return {"matches": matches}

    def _tool_submit(self, session_id: int, args: dict) -> dict:
        from pipeline.sandbox import submit_sandbox_idea
        brief = {
            "title": args.get("title"),
            "hypothesis": args.get("hypothesis"),
            "ticker": args.get("ticker"),
            "timeframe": args.get("timeframe") or "1d",
            "factor_formula": args.get("factor_formula"),
        }
        result = submit_sandbox_idea(brief, run_backtest=False, source="concierge")
        if result.get("ok"):
            with db_session() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO concierge_idea_links (session_id, idea_id) "
                    "VALUES (?, ?)", (session_id, result["idea_id"]))
        return result

    def _tool_idea_status(self, idea_id: int) -> dict:
        with db_session() as conn:
            idea = conn.execute(
                "SELECT id, title, ticker, stage, status, rejection_reason "
                "FROM alpha_ideas WHERE id=?", (idea_id,)).fetchone()
            if not idea:
                return {"error": f"No idea #{idea_id}"}
            bt = conn.execute(
                "SELECT run_type, passed, verdict, verdict_reason, sharpe_net, max_dd, "
                "excess_vs_ew_ann_return, benchmark_pass, capacity_pass "
                "FROM backtest_runs WHERE idea_id=? ORDER BY id DESC LIMIT 1",
                (idea_id,)).fetchone()
            paper = conn.execute(
                "SELECT COUNT(*) n FROM paper_trades WHERE idea_id=?", (idea_id,)
            ).fetchone()["n"]
        out = {**dict(idea), "backtest": dict(bt) if bt else None, "paper_trades": paper}
        return out

    def _tool_list_ideas(self, session_id: int) -> dict:
        with db_session() as conn:
            rows = conn.execute(
                "SELECT a.id, a.title, a.stage, a.status FROM concierge_idea_links l "
                "JOIN alpha_ideas a ON a.id = l.idea_id WHERE l.session_id=? "
                "ORDER BY l.id DESC", (session_id,)).fetchall()
        return {"ideas": [dict(r) for r in rows]}

    def _tool_search_kb(self, query: str) -> dict:
        try:
            from knowledge.search.retriever import retrieve
            hits = retrieve(query, k=5, hops=1)
            return {"results": [
                {"title": h.get("title"), "type": h.get("node_type"),
                 "summary": (h.get("summary") or "")[:300]} for h in hits[:5]]}
        except Exception as e:
            return {"error": f"KB search failed: {e}", "results": []}

    def _dispatch(self, session_id: int, name: str, args: dict):
        if name == "submit_strategy_idea":
            return self._tool_submit(session_id, args)
        if name == "get_idea_status":
            return self._tool_idea_status(int(args.get("idea_id")))
        if name == "list_session_ideas":
            return self._tool_list_ideas(session_id)
        if name == "search_knowledge_base":
            return self._tool_search_kb(args.get("query", ""))
        if name == "resolve_tickers":
            return self._tool_resolve_tickers(args.get("names", []))
        return {"error": f"unknown tool {name}"}

    # ── Session + turn handling ─────────────────────────────────────────────
    def ensure_session(self, session_id: int | None) -> int:
        with db_session() as conn:
            if session_id:
                row = conn.execute(
                    "SELECT id FROM concierge_sessions WHERE id=?", (session_id,)).fetchone()
                if row:
                    return session_id
            cur = conn.execute(
                "INSERT INTO concierge_sessions (label) VALUES (?)", ("chat",))
            return cur.lastrowid

    def _history(self, session_id: int, limit: int = 12) -> list:
        with db_session() as conn:
            rows = conn.execute(
                "SELECT role, content FROM concierge_messages WHERE session_id=? "
                "AND role IN ('user','assistant') ORDER BY id DESC LIMIT ?",
                (session_id, limit)).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    def handle(self, session_id: int | None, message: str) -> dict:
        """Run one chat turn. Returns {session_id, reply, tool_calls, idea_ids}."""
        spend = get_agent_daily_spend(self.name)
        if spend >= CONCIERGE_DAILY_BUDGET_USD:
            return {"session_id": session_id, "reply":
                    f"I've reached my daily chat budget (${spend:.2f} / "
                    f"${CONCIERGE_DAILY_BUDGET_USD:.2f}). Try again tomorrow, or ask "
                    f"Mark to raise CONCIERGE_DAILY_BUDGET_USD.",
                    "tool_calls": [], "idea_ids": [], "budget_exceeded": True}

        session_id = self.ensure_session(session_id)
        convo = self._history(session_id) + [{"role": "user", "content": message}]

        with db_session() as conn:
            conn.execute(
                "INSERT INTO concierge_messages (session_id, role, content) VALUES (?, 'user', ?)",
                (session_id, message))

        try:
            result = self.call_claude_tools(
                system=_system_prompt(), messages=convo, tools=TOOLS,
                tool_handler=lambda n, a: self._dispatch(session_id, n, a),
                model=self.default_model, task_label="concierge_chat",
                max_iters=CONCIERGE_MAX_TOOL_ITERS,
            )
        except Exception as e:
            self.log_daemon("WARN", f"Concierge turn failed: {e}")
            return {"session_id": session_id,
                    "reply": f"Sorry — I hit an error handling that ({e}). "
                             f"Please try rephrasing, or check the daily budget.",
                    "tool_calls": [], "idea_ids": [], "error": str(e)}
        reply = result["text"] or "(no reply)"
        idea_ids = [tc["result"]["idea_id"] for tc in result["tool_calls"]
                    if tc["name"] == "submit_strategy_idea"
                    and isinstance(tc.get("result"), dict) and tc["result"].get("ok")]

        with db_session() as conn:
            conn.execute(
                "INSERT INTO concierge_messages (session_id, role, content, tool_calls_json) "
                "VALUES (?, 'assistant', ?, ?)",
                (session_id, reply, json.dumps(result["tool_calls"], default=str)))
            conn.execute(
                "UPDATE concierge_sessions SET last_active=datetime('now') WHERE id=?",
                (session_id,))

        return {"session_id": session_id, "reply": reply,
                "tool_calls": result["tool_calls"], "idea_ids": idea_ids}

    def run(self, task: dict) -> dict:
        return self.handle(task.get("session_id"), task.get("message", ""))
