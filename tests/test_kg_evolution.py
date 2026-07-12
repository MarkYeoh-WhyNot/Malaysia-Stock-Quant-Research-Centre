"""Regression tests for the knowledge-graph evolution (Slices 1–5 of
docs/knowledge_graph_evolution_design.md).

The evolution shipped with no dedicated automated coverage — its proof was
manual/one-shot. These lock in the load-bearing invariants so a later edit to
store.py, the schema, or the ingesters can't silently regress the truth graph.

House pattern: share the local DB, isolate by a `test-kg-` slug prefix, and
clean up before + after. Tests are surgical (they exercise the ingester
building blocks, not whole-table scans) so they stay fast and deterministic.
"""
import pytest

from data.database import db_session, init_db
from knowledge.graph import store
from knowledge.ingestion import evidence_graph, alias_seeder
from knowledge.search.retriever import retrieve_facets
from scripts.graph_health_check import run_health_check

# Slugs the evidence-graph helpers mint that don't carry our test prefix.
_FIXED_INGESTER_SLUGS = ("strategy-999000001",)
_FAKE_IDEA_ID = 999000001
_AGENT_NAME = "TestKGParserAgent"


def _purge(exact=()):
    agent_slug = f"agent-{evidence_graph._slug_hash(_AGENT_NAME)}"
    with db_session() as conn:
        ids = {r["id"] for r in conn.execute(
            "SELECT id FROM kb_nodes WHERE slug LIKE 'test-kg-%'")}
        for slug in tuple(exact) + _FIXED_INGESTER_SLUGS + (agent_slug,):
            row = conn.execute("SELECT id FROM kb_nodes WHERE slug=?", (slug,)).fetchone()
            if row:
                ids.add(row["id"])
        # any finding node promoted from our planted governance row
        for r in conn.execute(
                "SELECT id FROM kb_nodes WHERE node_type='finding' AND slug LIKE 'finding-%' "
                "AND ref_table='governance_findings' AND ref_id IN "
                "(SELECT id FROM governance_findings WHERE agent=?)", (_AGENT_NAME,)):
            ids.add(r["id"])
        for nid in ids:
            conn.execute("DELETE FROM kb_edges WHERE source_id=? OR target_id=?", (nid, nid))
            conn.execute("DELETE FROM kb_fts WHERE node_id=?", (nid,))
            conn.execute("DELETE FROM kb_nodes WHERE id=?", (nid,))
        conn.execute("DELETE FROM governance_findings WHERE agent=?", (_AGENT_NAME,))


@pytest.fixture(autouse=True)
def _setup():
    init_db()
    _purge()
    yield
    _purge()


def _edge_exists(source_id, target_id, relation):
    with db_session() as conn:
        return conn.execute(
            "SELECT 1 FROM kb_edges WHERE source_id=? AND target_id=? AND relation=?",
            (source_id, target_id, relation)).fetchone() is not None


# ── Slice 1: node-type registry validation ───────────────────────────────────

def test_registry_rejects_unknown_node_type():
    """store.upsert_node must refuse a type absent from kb_node_type_registry
    (app-level discipline that replaced the dropped DB CHECK)."""
    with pytest.raises(ValueError):
        store.upsert_node("not_a_registered_type", "test-kg-bad-type", title="x")
    # a registered type still works
    nid = store.upsert_node("strategy", "test-kg-good-type", title="ok")
    assert nid


# ── Slice 1 (§5.5): idea → strategy promotion rule ───────────────────────────

def test_qualifies_promotion_rule():
    """Only evaluated ideas become strategies; a raw gate0 candidate does not."""
    assert evidence_graph._qualifies({"signal_signature": "sig"}, False, False) == "has_signature"
    assert evidence_graph._qualifies({}, True, False) == "has_backtest"
    assert evidence_graph._qualifies({}, False, True) == "has_gate_decision"
    assert evidence_graph._qualifies({"stage": "stage2"}, False, False) == "stage:stage2"
    # raw candidate with nothing to show → not promoted
    assert evidence_graph._qualifies({"stage": "gate0"}, False, False) is None
    assert evidence_graph._qualifies({}, False, False) is None


def test_promote_idea_wires_compiled_to_strategy():
    """_promote_idea_to_strategy emits a strategy node and links the raw idea
    node to it via `compiled_to`."""
    idea_node = store.upsert_node(
        "idea", "test-kg-idea-1", title="raw idea", ref=("alpha_ideas", _FAKE_IDEA_ID))
    strat = evidence_graph._promote_idea_to_strategy(
        {"id": _FAKE_IDEA_ID, "title": "promoted", "stage": "stage2"}, "has_backtest")
    assert store.get_node(slug=f"strategy-{_FAKE_IDEA_ID}") is not None
    assert _edge_exists(idea_node, strat, "compiled_to")


# ── Slice 1.5: DSL leaves become parser-honesty nodes ────────────────────────

def test_ingest_leaves_registers_every_dsl_leaf():
    from agents.backtest_engineer.signal_dsl import LEAVES
    n = evidence_graph.ingest_leaves()
    assert n == len(LEAVES)
    # a known leaf is now a graph node
    assert store.get_node(slug="leaf-rsi") is not None


# ── Slice 2: deterministic alias resolution ──────────────────────────────────

def test_alias_seed_and_resolve():
    alias_seeder.seed_aliases()
    assert alias_seeder.resolve("btcusdt") == "BTC"
    assert alias_seeder.resolve("XBT") == "BTC"          # case-insensitive
    assert alias_seeder.resolve("dpsr") == "deflated_probabilistic_sharpe_ratio"
    # an unknown term passes through unchanged
    assert alias_seeder.resolve("test-kg-nonexistent") == "test-kg-nonexistent"


# ── Slice 3: governance findings → finding/agent/risk graph ──────────────────

def test_ingest_findings_wires_agent_and_risk():
    """A parser-honesty governance finding is promoted to a `finding` node with
    reported_by(agent) and exposed_to(risk=parser_approximation) edges."""
    with db_session() as conn:
        conn.execute(
            "INSERT INTO governance_findings "
            "(agent, level, scope, status, severity, evidence, local_recommendation) "
            "VALUES (?, 'L0', 'parser', 'FAIL', 'BLOCKER', ?, ?)",
            (_AGENT_NAME,
             "parser approximated ma_level as ema_cross — representability violation",
             "return representable:false"))
    evidence_graph.ingest_findings()

    fnode = store.get_node(slug=f"finding-{evidence_graph._slug_hash('|'.join((_AGENT_NAME, 'L0', 'parser', 'BLOCKER', 'FAIL')))}")
    agent_node = store.get_node(slug=f"agent-{evidence_graph._slug_hash(_AGENT_NAME)}")
    risk_node = store.get_node(slug="risk-parser_approximation")
    assert fnode is not None and agent_node is not None and risk_node is not None
    assert _edge_exists(fnode["id"], agent_node["id"], "reported_by")
    assert _edge_exists(fnode["id"], risk_node["id"], "exposed_to")


# ── Slice 5: typed retrieval facets ──────────────────────────────────────────

def test_retrieve_facets_groups_by_relation():
    """retrieve_facets returns a packet grouped by what each fact IS to the query
    (past failures, signatures, risks, leaves) rather than a flat list."""
    term = "zzqxfundcarry"  # unique FTS token so the seed resolves to our strategy
    strat = store.upsert_node("strategy", "test-kg-strat",
                              title=f"{term} strategy", summary=term)
    sig = store.upsert_node("signature", "test-kg-sig", title="funding carry signature")
    gate = store.upsert_node("gate_decision", "test-kg-gate", title="gate reject",
                             summary="cost drag killed the edge")
    leaf = store.upsert_node("leaf", "test-kg-leaf", title="funding_zscore")
    risk = store.upsert_node("risk", "test-kg-risk", title="funding_data_gap")
    store.add_edge(strat, sig, "shares_signature", origin="heuristic")
    store.add_edge(strat, gate, "failed", origin="heuristic")
    store.add_edge(strat, leaf, "uses_leaf", origin="heuristic")
    store.add_edge(strat, risk, "exposed_to", origin="heuristic")

    facets = retrieve_facets(term, k=5)

    assert any(term in t for t in facets["direct_matches"])
    assert "funding carry signature" in facets["related_signatures"]
    assert f"{term} strategy" in facets["past_failures"]
    assert any("cost drag" in r for r in facets["common_rejection_reasons"])
    assert "funding_data_gap" in facets["open_risks"]
    assert "funding_zscore" in facets["leaves_used"]


# ── Slice 5: anti-garbage health check ───────────────────────────────────────

def test_health_check_flags_unregistered_node_type():
    """A node whose type isn't in the registry (planted via raw SQL, bypassing
    store validation) must surface as a BLOCKER."""
    with db_session() as conn:
        conn.execute(
            "INSERT INTO kb_nodes (node_type, slug, title, content_hash) "
            "VALUES ('test_kg_bogus_type', 'test-kg-bad-node', 'bad', 'h')")

    result = run_health_check(record=False)

    assert result["blockers"] >= 1
    assert any(f["check"] == "unregistered_node_type"
               and "test_kg_bogus_type" in f["detail"]
               and f["severity"] == "BLOCKER"
               for f in result["detail"])
