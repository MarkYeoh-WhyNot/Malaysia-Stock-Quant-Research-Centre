"""Regression tests for campaign-findings ingestion (Phase 1 of the ideation
loop wiring): alpha-hunt campaign verdicts must land in the knowledge graph as
`finding` nodes so the generator's GraphRAG and red/blue retrieval surface
them next cycle.

House pattern: share the local DB, isolate by a slug prefix
(finding-campaign-test-cf-*), and clean up before + after.
"""
import json

import pytest

from data.database import db_session, init_db
from knowledge.ingestion.campaign_findings import (
    MIN_TRIALS_FOR_FALSIFIED, emit_alpha_hunt_findings, record_campaign_finding,
)

_PREFIX = "finding-campaign-test-cf-"
# Emitter-minted slugs derive from generated_at month + config/pair/tf below.
_EMITTER_SLUGS_LIKE = "finding-campaign-alpha-hunt-2099-01%"


def _purge():
    with db_session() as conn:
        ids = {r["id"] for r in conn.execute(
            "SELECT id FROM kb_nodes WHERE slug LIKE ? OR slug LIKE ?",
            (_PREFIX + "%", _EMITTER_SLUGS_LIKE))}
        for nid in ids:
            conn.execute("DELETE FROM kb_edges WHERE source_id=? OR target_id=?",
                         (nid, nid))
            conn.execute("DELETE FROM kb_fts WHERE node_id=?", (nid,))
            conn.execute("DELETE FROM kb_nodes WHERE id=?", (nid,))


@pytest.fixture(autouse=True)
def _setup():
    init_db()
    _purge()
    yield
    _purge()


def _node(slug):
    with db_session() as conn:
        return conn.execute("SELECT * FROM kb_nodes WHERE slug=?", (slug,)).fetchone()


def _edges_from(node_id):
    with db_session() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT relation, target_id FROM kb_edges WHERE source_id=?", (node_id,))]


def test_record_is_idempotent_and_wires_leaf_edges():
    kwargs = dict(
        slug="test-cf-idem", title="test finding", summary="a test verdict",
        direction="falsified", leaf_names=("funding_level", "funding_zscore"),
        content=json.dumps({"ic": 0.018}),
    )
    nid1 = record_campaign_finding(**kwargs)
    nid2 = record_campaign_finding(**kwargs)
    assert nid1 == nid2

    row = _node(_PREFIX + "idem")
    assert row is not None and row["node_type"] == "finding"
    assert row["domain"] == "research-record"

    uses_leaf = [e for e in _edges_from(nid1) if e["relation"] == "uses_leaf"]
    assert len(uses_leaf) == 2
    # re-recording must not duplicate edges (UNIQUE(source,target,relation))
    assert len([e for e in _edges_from(nid2) if e["relation"] == "uses_leaf"]) == 2


def test_record_rejects_bad_direction_and_skips_missing_targets():
    with pytest.raises(ValueError):
        record_campaign_finding(slug="test-cf-bad", title="x", summary="x",
                                direction="maybe")

    nid = record_campaign_finding(
        slug="test-cf-noref", title="x", summary="x", direction="watch",
        refines_slugs=("finding-campaign-test-cf-does-not-exist",),
    )
    assert all(e["relation"] != "refines" for e in _edges_from(nid))


def test_emitter_skips_underpowered_zero_survivor_runs():
    report = {"generated_at": "2099-01-01T00:00:00", "stage_a_trials": 40,
              "survivors": [], "finalist_results": [], "stage_b_finalists": 0}
    stats = emit_alpha_hunt_findings(report)
    assert stats == {"falsified": 0, "confirmed": 0}
    assert _node("finding-campaign-alpha-hunt-2099-01-no-edge") is None


def test_emitter_records_falsified_direction_at_scale():
    tree = {"entry": {"leaf": "sma_cross", "fast": 10, "slow": 30,
                      "direction": "above"}}
    report = {"generated_at": "2099-01-01T00:00:00",
              "stage_a_trials": MIN_TRIALS_FOR_FALSIFIED,
              "survivors": [], "stage_b_finalists": 3,
              "finalist_results": [{"dsl": tree}],
              "pairs": ["BTC/USDT"], "timeframes": ["1d"],
              "market_mode": "crypto"}
    stats = emit_alpha_hunt_findings(report)
    assert stats["falsified"] == 1

    row = _node("finding-campaign-alpha-hunt-2099-01-no-edge")
    assert row is not None
    assert "falsified" in (row["tags"] or "")
    with db_session() as conn:
        leaf_edges = conn.execute(
            "SELECT COUNT(*) AS n FROM kb_edges e JOIN kb_nodes t ON t.id=e.target_id "
            "WHERE e.source_id=? AND e.relation='uses_leaf' AND t.slug='leaf-sma_cross'",
            (row["id"],)).fetchone()["n"]
    assert leaf_edges == 1


def test_emitter_records_each_survivor_as_confirmed():
    tree = {"entry": {"leaf": "funding_level", "above": -0.0001}}
    report = {"generated_at": "2099-01-01T00:00:00", "stage_a_trials": 900,
              "stage_b_finalists": 1,
              "survivors": [{"config": "fund_lvl_1bp", "pair": "BTC/USDT",
                             "tf": "1d", "family": "carry", "dsl": tree,
                             "idea_id": 1, "test_sharpe_net": 1.2,
                             "deflated_hurdle": 0.9, "n_trials": 900}],
              "finalist_results": []}
    stats = emit_alpha_hunt_findings(report)
    assert stats == {"falsified": 0, "confirmed": 1}
    row = _node("finding-campaign-alpha-hunt-2099-01-fund_lvl_1bp-BTCUSDT-1d")
    assert row is not None
    assert "confirmed" in (row["tags"] or "")


def test_findings_are_retrievable_as_seeds():
    record_campaign_finding(
        slug="test-cf-retrieval", title="zzcfretrievaltoken funding carry verdict",
        summary="zzcfretrievaltoken unique probe for retriever seeding",
        direction="confirmed",
    )
    from knowledge.search.retriever import retrieve
    hits = retrieve("zzcfretrievaltoken", k=4, node_types=["finding"])
    assert any(h["slug"] == _PREFIX + "retrieval" for h in hits)
