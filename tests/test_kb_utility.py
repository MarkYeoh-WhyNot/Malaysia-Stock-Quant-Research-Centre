"""KB-utility wiring: provenance storage, rejection edges, funnel split."""
import json

import pytest

from data.database import db_session, init_db
from agents.researcher.strategy_researcher import StrategyResearcher


@pytest.fixture(autouse=True)
def _setup():
    init_db()
    _cleanup()
    yield
    _cleanup()


def _cleanup():
    with db_session() as conn:
        rows = conn.execute(
            "SELECT id FROM alpha_ideas WHERE title LIKE 'TESTKBU%'").fetchall()
        ids = [r["id"] for r in rows]
        for iid in ids:
            node = conn.execute(
                "SELECT id FROM kb_nodes WHERE ref_table='alpha_ideas' AND ref_id=?",
                (iid,)).fetchone()
            if node:
                conn.execute("DELETE FROM kb_edges WHERE source_id=? OR target_id=?",
                             (node["id"], node["id"]))
                conn.execute("DELETE FROM kb_fts WHERE node_id=?", (node["id"],))
                conn.execute("DELETE FROM kb_nodes WHERE id=?", (node["id"],))
        if ids:
            marks = ",".join("?" * len(ids))
            conn.execute(f"DELETE FROM alpha_ideas WHERE id IN ({marks})", ids)
        conn.execute("DELETE FROM rejection_patterns WHERE example_title LIKE 'TESTKBU%'")


def test_save_idea_stores_kb_context():
    sr = StrategyResearcher()
    iid = sr.save_idea({
        "title": "TESTKBU grounded idea",
        "hypothesis": "x",
        "ticker": "1155.KL",
        "factor_formula": "rsi below 30 with high volume for kbu test",
        "kb_context": ["tech-rsi-mean-reversion", "2026-01-01-some-note"],
    })
    with db_session() as conn:
        ctx = conn.execute(
            "SELECT kb_context FROM alpha_ideas WHERE id=?", (iid,)).fetchone()["kb_context"]
    assert json.loads(ctx) == ["tech-rsi-mean-reversion", "2026-01-01-some-note"]


def test_save_idea_without_context_is_null():
    sr = StrategyResearcher()
    iid = sr.save_idea({
        "title": "TESTKBU ungrounded idea",
        "hypothesis": "x",
        "ticker": "1023.KL",
        "factor_formula": "sma cross twenty fifty ungrounded kbu test",
    })
    with db_session() as conn:
        ctx = conn.execute(
            "SELECT kb_context FROM alpha_ideas WHERE id=?", (iid,)).fetchone()["kb_context"]
    assert ctx is None


def test_rejection_creates_graph_edge():
    from knowledge.ingestion.rejection_memory import RejectionMemory
    sr = StrategyResearcher()
    iid = sr.save_idea({
        "title": "TESTKBU momentum banking idea",
        "hypothesis": "momentum on Maybank",
        "ticker": "1155.KL",
        "factor_formula": "sma crossover momentum trend kbu reject test",
    })
    RejectionMemory().record_rejection(iid, "no edge — low sharpe below threshold", "stage2")
    with db_session() as conn:
        idea_node = conn.execute(
            "SELECT id FROM kb_nodes WHERE ref_table='alpha_ideas' AND ref_id=?",
            (iid,)).fetchone()
        assert idea_node is not None, "idea node should be created on rejection"
        edge = conn.execute(
            "SELECT e.relation, n.node_type FROM kb_edges e "
            "JOIN kb_nodes n ON n.id = e.target_id "
            "WHERE e.source_id=? AND e.relation='rejected_because'",
            (idea_node["id"],)).fetchone()
    assert edge is not None, "rejected_because edge should exist"
    assert edge["node_type"] == "rejection_pattern"


def test_funnel_split_counts_grounded_vs_ungrounded():
    import scripts.research_daemon as rd
    sr = StrategyResearcher()
    sr.save_idea({
        "title": "TESTKBU funnel grounded",
        "hypothesis": "x", "ticker": "5347.KL",
        "factor_formula": "bollinger squeeze funnel grounded kbu",
        "kb_context": ["tech-bollinger-squeeze"],
    })
    sr.save_idea({
        "title": "TESTKBU funnel plain",
        "hypothesis": "x", "ticker": "6012.KL",
        "factor_formula": "gap fill funnel plain kbu",
    })
    d = rd.ResearchDaemon.__new__(rd.ResearchDaemon)
    counts = d._funnel_counts(1)
    assert counts["kb_gen"] >= 1
    assert counts["plain_gen"] >= 1
    assert counts["generated"] == counts["kb_gen"] + counts["plain_gen"]
