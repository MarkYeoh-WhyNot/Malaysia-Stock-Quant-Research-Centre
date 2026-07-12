"""scripts/cleanup_duplicate_finding_nodes.py: collapses duplicate governance
`finding` nodes down to distinct (agent, level, scope, severity, status)
states, WITHOUT touching the disjoint finding-campaign-* namespace
(campaign_findings.py) — that script has no governance_findings row behind
it, so wiping it would delete it permanently (ingest_findings() only rebuilds
from governance_findings).

House pattern: share the local DB, isolate by a fixed test agent name /
campaign slug, purge before + after.
"""
import pytest

from data.database import db_session, init_db
from knowledge.graph import store
from knowledge.ingestion import evidence_graph
from knowledge.ingestion.campaign_findings import record_campaign_finding
from scripts.cleanup_duplicate_finding_nodes import _finding_node_count, _wipe_finding_nodes

_AGENT_NAME = "TestCleanupDupInspector"
_CAMPAIGN_SLUG = "test-cleanup-dup-campaign"


def _purge():
    with db_session() as conn:
        for r in conn.execute(
                "SELECT id FROM kb_nodes WHERE node_type='finding' "
                "AND ref_table='governance_findings' AND ref_id IN "
                "(SELECT id FROM governance_findings WHERE agent=?)", (_AGENT_NAME,)):
            conn.execute("DELETE FROM kb_edges WHERE source_id=? OR target_id=?",
                        (r["id"], r["id"]))
            conn.execute("DELETE FROM kb_fts WHERE node_id=?", (r["id"],))
            conn.execute("DELETE FROM kb_nodes WHERE id=?", (r["id"],))
        row = conn.execute(
            "SELECT id FROM kb_nodes WHERE slug=?",
            (f"finding-campaign-{_CAMPAIGN_SLUG}",)).fetchone()
        if row:
            conn.execute("DELETE FROM kb_edges WHERE source_id=? OR target_id=?",
                        (row["id"], row["id"]))
            conn.execute("DELETE FROM kb_fts WHERE node_id=?", (row["id"],))
            conn.execute("DELETE FROM kb_nodes WHERE id=?", (row["id"],))
        conn.execute("DELETE FROM governance_findings WHERE agent=?", (_AGENT_NAME,))


@pytest.fixture(autouse=True)
def _setup():
    init_db()
    _purge()
    yield
    _purge()


def _insert_governance_row(status="FAIL"):
    with db_session() as conn:
        conn.execute(
            "INSERT INTO governance_findings "
            "(agent, level, scope, status, severity, evidence) VALUES "
            "(?, 'L0', 'test-scope', ?, 'BLOCKER', 'test evidence')",
            (_AGENT_NAME, status))


def test_wipe_collapses_duplicate_rows_but_spares_campaign_findings():
    # Three governance_findings rows, same state -> pre-fix ingest_findings()
    # would have minted 3 separate finding-{fid} nodes (one per row id).
    for _ in range(3):
        _insert_governance_row("FAIL")
    evidence_graph.ingest_findings()

    campaign_id = record_campaign_finding(
        slug=_CAMPAIGN_SLUG, title="t", summary="s", direction="confirmed")

    with db_session() as conn:
        gov_nodes_before = conn.execute(
            "SELECT COUNT(*) AS n FROM kb_nodes WHERE node_type='finding' "
            "AND ref_table='governance_findings' AND ref_id IN "
            "(SELECT id FROM governance_findings WHERE agent=?)",
            (_AGENT_NAME,)).fetchone()["n"]
    # Already collapsed to 1 by the content-keyed slug fix — nothing left
    # for the cleanup script to do on THIS agent, but exercise the wipe path
    # regardless (this pins that it doesn't touch campaign findings).
    assert gov_nodes_before == 1

    deleted = _wipe_finding_nodes()
    assert deleted >= 1

    with db_session() as conn:
        campaign_row = conn.execute(
            "SELECT id FROM kb_nodes WHERE id=?", (campaign_id,)).fetchone()
        gov_rows_after = conn.execute(
            "SELECT id FROM kb_nodes WHERE node_type='finding' "
            "AND ref_table='governance_findings' AND ref_id IN "
            "(SELECT id FROM governance_findings WHERE agent=?)",
            (_AGENT_NAME,)).fetchall()
    assert campaign_row is not None, "cleanup must not delete finding-campaign-* nodes"
    assert gov_rows_after == []

    stats = evidence_graph.ingest_findings()
    with db_session() as conn:
        rebuilt = conn.execute(
            "SELECT id FROM kb_nodes WHERE node_type='finding' "
            "AND ref_table='governance_findings' AND ref_id IN "
            "(SELECT id FROM governance_findings WHERE agent=?)",
            (_AGENT_NAME,)).fetchall()
    assert len(rebuilt) == 1, "rebuild must collapse back to one node per state"


def test_finding_node_count_excludes_campaign_findings():
    record_campaign_finding(
        slug=_CAMPAIGN_SLUG, title="t", summary="s", direction="confirmed")
    _insert_governance_row("FAIL")
    evidence_graph.ingest_findings()

    with db_session() as conn:
        raw_total = conn.execute(
            "SELECT COUNT(*) AS n FROM kb_nodes WHERE node_type='finding'"
        ).fetchone()["n"]
    reported = _finding_node_count()
    assert reported < raw_total, (
        "_finding_node_count must exclude finding-campaign-* so the "
        "cleanup script's before/after report isn't inflated by a "
        "namespace it never touches")
