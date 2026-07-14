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
from knowledge.graph import store
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
        # detect_triggers() advances a global cursor — reset it so this
        # test's edge doesn't leak into (or get masked by) other tests.
        conn.execute("DELETE FROM revisit_state WHERE key='finding_scan:last_node_id'")


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


def test_contradicts_slugs_wires_a_trigger_ready_edge():
    """2026-07-13 follow-up audit (task 1): `contradicts_slugs` has zero
    production callers (neither the alpha_hunt emitter nor the campaign
    backfill script ever passes it) and had zero test coverage of its own —
    the two halves of the chain (this producer, revisit.py's consumer) were
    only ever verified in isolation, via store.add_edge directly in
    test_revisit.py. This proves the actual production entry point wires an
    edge that revisit.detect_triggers() picks up, end to end."""
    pattern_id = store.upsert_node(
        "rejection_pattern", slug=_PREFIX + "pattern", title="test pattern")
    # record_campaign_finding upserts the finding node itself — the
    # contradicts target must already exist (missing targets are skipped,
    # not fabricated), so create the pattern first.
    finding_id = record_campaign_finding(
        slug="test-cf-contradicts", title="test contradicting finding",
        summary="a test verdict that contradicts an old rejection pattern",
        direction="confirmed",
        contradicts_slugs=(_PREFIX + "pattern",),
    )

    contradicts = [e for e in _edges_from(finding_id) if e["relation"] == "contradicts"]
    assert contradicts == [{"relation": "contradicts", "target_id": pattern_id}]

    import unittest.mock as mock
    from pipeline import revisit
    with mock.patch.object(revisit, "_current_vol_tercile", return_value=None), \
         mock.patch.object(revisit, "_current_macro_regime", return_value=None):
        triggers = revisit.detect_triggers()

    hits = [t for t in triggers if t["type"] == "contradicting_finding"
           and t["finding_id"] == finding_id]
    assert len(hits) == 1
    assert hits[0]["pattern_slug"] == _PREFIX + "pattern"


def test_findings_are_retrievable_as_seeds():
    record_campaign_finding(
        slug="test-cf-retrieval", title="zzcfretrievaltoken funding carry verdict",
        summary="zzcfretrievaltoken unique probe for retriever seeding",
        direction="confirmed",
    )
    from knowledge.search.retriever import retrieve
    hits = retrieve("zzcfretrievaltoken", k=4, node_types=["finding"])
    assert any(h["slug"] == _PREFIX + "retrieval" for h in hits)
