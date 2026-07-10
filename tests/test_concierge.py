"""ConciergeAgent — tool implementations, budget sub-cap, tool-use loop, guardrails.

No network: the Anthropic client is mocked where a Claude call would occur.
"""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from data.database import db_session, init_db
from agents.concierge.concierge_agent import ConciergeAgent, TOOLS


@pytest.fixture(autouse=True)
def _clean():
    init_db()
    _cleanup()
    yield
    _cleanup()


def _cleanup():
    with db_session() as conn:
        conn.execute("DELETE FROM backtest_runs WHERE idea_id IN "
                     "(SELECT id FROM alpha_ideas WHERE title LIKE 'CCX %')")
        conn.execute("DELETE FROM alpha_ideas WHERE title LIKE 'CCX %'")
        conn.execute("DELETE FROM concierge_idea_links WHERE session_id IN "
                     "(SELECT id FROM concierge_sessions WHERE label='test')")
        conn.execute("DELETE FROM concierge_messages WHERE session_id IN "
                     "(SELECT id FROM concierge_sessions WHERE label='test')")
        conn.execute("DELETE FROM concierge_sessions WHERE label='test'")


def _session():
    with db_session() as conn:
        cur = conn.execute("INSERT INTO concierge_sessions (label) VALUES ('test')")
        return cur.lastrowid


# ── Guardrail: the toolset cannot reach live trading ──────────────────────────
def test_toolset_has_no_live_or_destructive_tools():
    names = {t["name"] for t in TOOLS}
    assert names == {"submit_strategy_idea", "get_idea_status", "list_session_ideas",
                     "search_knowledge_base", "resolve_tickers", "suggest_techniques",
                     "get_pine_script"}
    blob = " ".join(names).lower()
    for forbidden in ("live", "approve", "delete", "promote", "stage4b"):
        assert forbidden not in blob


# ── Pine Script tool ──────────────────────────────────────────────────────────
def test_pine_script_not_ready_before_backtest():
    agent = ConciergeAgent()
    with db_session() as conn:
        cur = conn.execute(
            "INSERT INTO alpha_ideas (slug, title, hypothesis, ticker, timeframe, "
            "factor_formula, stage, status, novelty_score, logic_score, "
            "feasibility_score) VALUES ('ccx-ps-pending','CCX pine pending','h',"
            "'1155.KL','1d','f','stage2','pending',0.8,0.8,0.8)")
        idea_id = cur.lastrowid
    r = agent._tool_get_pine_script(idea_id)
    assert r["ok"] is False and r["status"] == "not_backtested_yet"


def test_pine_script_returns_code_when_present():
    agent = ConciergeAgent()
    with db_session() as conn:
        cur = conn.execute(
            "INSERT INTO alpha_ideas (slug, title, hypothesis, ticker, timeframe, "
            "factor_formula, stage, status, novelty_score, logic_score, "
            "feasibility_score) VALUES ('ccx-ps-ready','CCX pine ready','h',"
            "'1155.KL','1d','f','stage2','rejected',0.8,0.8,0.8)")
        idea_id = cur.lastrowid
        conn.execute(
            "INSERT INTO backtest_runs (idea_id, run_type, pinescript) "
            "VALUES (?, 'klse_daily', ?)", (idea_id, "//@version=5\nstrategy(\"x\")\n"))
    r = agent._tool_get_pine_script(idea_id)
    assert r["ok"] is True and "strategy(" in r["pinescript"]


def test_pine_script_not_applicable_for_basket():
    agent = ConciergeAgent()
    with db_session() as conn:
        cur = conn.execute(
            "INSERT INTO alpha_ideas (slug, title, hypothesis, ticker, timeframe, "
            "factor_formula, stage, status, novelty_score, logic_score, "
            "feasibility_score) VALUES ('ccx-ps-basket','CCX pine basket','h',"
            "'UNIVERSE','1d','f','stage3','active',0.8,0.8,0.8)")
        idea_id = cur.lastrowid
        conn.execute(
            "INSERT INTO backtest_runs (idea_id, run_type, pinescript) "
            "VALUES (?, 'cross_sectional', NULL)", (idea_id,))
    r = agent._tool_get_pine_script(idea_id)
    assert r["ok"] is False and r["status"] == "not_applicable"


# ── Technique Arsenal wiring ──────────────────────────────────────────────────
def test_system_prompt_carries_technique_arsenal_index():
    from agents.concierge.concierge_agent import _system_prompt
    p = _system_prompt()
    assert "TECHNIQUE ARSENAL" in p
    assert "suggest_techniques" in p
    # Bursa default mode → a Bursa-library key should be in the index
    assert "kalman_filter" in p


def test_suggest_techniques_by_key_and_by_shape():
    c = ConciergeAgent()
    by_key = c._tool_suggest_techniques({"key": "kalman_filter"})
    assert "TECHNIQUE:" in by_key["techniques"]
    assert "WHEN TO USE" in by_key["techniques"]
    ranked = c._tool_suggest_techniques({"strategy_type": "momentum"})
    assert ranked["techniques"].strip()
    unknown = c._tool_suggest_techniques({"key": "no_such_technique"})
    assert "not found" in unknown["techniques"]


def test_suggest_techniques_full_detail_carries_arsenal_v2_example():
    # market-aware: picks keys from whichever library is active this MARKET_MODE
    from knowledge.ingestion.technique_library import TECHNIQUE_LIBRARY
    c = ConciergeAgent()
    # validated example for a representable technique
    with_key = next(k for k, t in TECHNIQUE_LIBRARY.items()
                    if "dsl" in t["example"] or "factor_spec" in t["example"])
    detail = c._tool_suggest_techniques({"key": with_key})["techniques"]
    assert "VALIDATED EXAMPLE" in detail
    # honest no-example for an unimplemented concept, naming the gap
    none_key, none_tech = next(
        (k, t) for k, t in TECHNIQUE_LIBRARY.items()
        if "none" in t["example"] and t["representability"]["missing_leaves"])
    detail = c._tool_suggest_techniques({"key": none_key})["techniques"]
    assert "NO EXECUTABLE EXAMPLE" in detail
    assert none_tech["representability"]["missing_leaves"][0] in detail


def test_system_prompt_pins_verbatim_verdict_rule():
    from agents.concierge.concierge_agent import _system_prompt
    p = _system_prompt()
    assert "VERBATIM" in p
    assert "rejection_reason" in p and "verdict_reason" in p
    assert "was not recorded" in p
    # ma_level leaf catalog inheritance — the concierge sees the new vocabulary
    assert "ma_level" in p


# ── Tool implementations ──────────────────────────────────────────────────────
def test_resolve_tickers_maps_names_and_sectors():
    c = ConciergeAgent()
    out = c._tool_resolve_tickers(["Maybank", "banks"])
    assert out["matches"]["Maybank"] == ["1155.KL"]
    assert "1155.KL" in out["matches"]["banks"]  # Banking sector expansion
    assert len(out["matches"]["banks"]) >= 5


def test_submit_links_idea_to_session():
    c = ConciergeAgent()
    sid = _session()
    r = c._tool_submit(sid, {
        "title": "CCX weekly momentum", "hypothesis": "weekly momentum on Tenaga, hold weeks",
        "ticker": "5347.KL", "factor_formula": "close crosses above sma(50)"})
    assert r["ok"] is True
    listed = c._tool_list_ideas(sid)
    assert any(i["id"] == r["idea_id"] for i in listed["ideas"])


def test_submit_rejects_short_selling():
    c = ConciergeAgent()
    sid = _session()
    r = c._tool_submit(sid, {
        "title": "CCX short", "hypothesis": "short sell weak banks",
        "ticker": "1155.KL", "factor_formula": "short when overbought for days"})
    assert r["ok"] is False


def test_idea_status_reports_stage():
    c = ConciergeAgent()
    sid = _session()
    r = c._tool_submit(sid, {
        "title": "CCX status probe", "hypothesis": "momentum on Maxis over weeks",
        "ticker": "6012.KL", "factor_formula": "close above sma(100) uptrend"})
    st = c._tool_idea_status(r["idea_id"])
    assert st["stage"] == "stage2"
    assert st["id"] == r["idea_id"]


# ── Budget sub-cap ────────────────────────────────────────────────────────────
def test_budget_subcap_blocks_before_any_claude_call():
    c = ConciergeAgent()
    with patch("agents.concierge.concierge_agent.get_agent_daily_spend", return_value=999.0), \
         patch.object(c, "call_claude_tools") as mock_call:
        out = c.handle(None, "test an idea")
    assert out["budget_exceeded"] is True
    mock_call.assert_not_called()


# ── Tool-use loop (mocked Anthropic client) ──────────────────────────────────
def _fake_tool_use_response(tool_name, tool_input):
    block = SimpleNamespace(type="tool_use", name=tool_name, input=tool_input, id="tu_1")
    return SimpleNamespace(
        stop_reason="tool_use", content=[block],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5))


def _fake_text_response(text):
    block = SimpleNamespace(type="text", text=text)
    return SimpleNamespace(
        stop_reason="end_turn", content=[block],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5))


def test_handle_runs_tool_loop_and_persists(monkeypatch):
    c = ConciergeAgent()
    # First model turn calls resolve_tickers; second returns final text.
    responses = iter([
        _fake_tool_use_response("resolve_tickers", {"names": ["Maybank"]}),
        _fake_text_response("Maybank is 1155.KL. Want me to submit a strategy on it?"),
    ])
    monkeypatch.setattr(c.client.messages, "create", lambda **kw: next(responses))
    monkeypatch.setattr("agents.concierge.concierge_agent.get_agent_daily_spend",
                        lambda *a: 0.0)

    out = c.handle(None, "what ticker is Maybank?")
    assert "1155.KL" in out["reply"]
    assert any(tc["name"] == "resolve_tickers" for tc in out["tool_calls"])
    # message history persisted for the session
    with db_session() as conn:
        n = conn.execute("SELECT COUNT(*) n FROM concierge_messages WHERE session_id=?",
                         (out["session_id"],)).fetchone()["n"]
    assert n >= 2  # user + assistant
    with db_session() as conn:
        conn.execute("DELETE FROM concierge_messages WHERE session_id=?", (out["session_id"],))
        conn.execute("DELETE FROM concierge_sessions WHERE id=?", (out["session_id"],))
