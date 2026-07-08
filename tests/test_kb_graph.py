"""Knowledge graph schema, store, and migration tests (dev DB, sentinel slugs)."""
import pytest

from data.database import db_session, init_db
from knowledge.graph import store
from knowledge.graph.migrate import migrate_kb_graph

SLUG_A = "test-graph-node-a"
SLUG_B = "test-graph-node-b"


@pytest.fixture(autouse=True)
def _setup():
    init_db()
    _cleanup()
    yield
    _cleanup()


def _cleanup():
    with db_session() as conn:
        rows = conn.execute(
            "SELECT id FROM kb_nodes WHERE slug LIKE 'test-graph-%'").fetchall()
        ids = [r["id"] for r in rows]
        if ids:
            marks = ",".join("?" * len(ids))
            conn.execute(f"DELETE FROM kb_edges WHERE source_id IN ({marks}) "
                         f"OR target_id IN ({marks})", ids + ids)
            conn.execute(f"DELETE FROM kb_fts WHERE node_id IN ({marks})", ids)
            conn.execute(f"DELETE FROM kb_nodes WHERE id IN ({marks})", ids)


def test_upsert_node_creates_and_syncs_fts():
    nid = store.upsert_node("note", SLUG_A, title="Dividend capture on Bursa",
                            summary="Ex-date behaviour", content="long text here")
    with db_session() as conn:
        fts = conn.execute("SELECT * FROM kb_fts WHERE node_id=?", (nid,)).fetchone()
    assert fts is not None and "Dividend" in fts["title"]


def test_upsert_node_resets_extraction_on_change():
    nid = store.upsert_node("note", SLUG_A, title="v1", summary="s1")
    with db_session() as conn:
        conn.execute("UPDATE kb_nodes SET extracted_at=datetime('now') WHERE id=?", (nid,))
    # same content -> extracted_at untouched
    store.upsert_node("note", SLUG_A, title="v1", summary="s1")
    with db_session() as conn:
        assert conn.execute("SELECT extracted_at FROM kb_nodes WHERE id=?",
                            (nid,)).fetchone()["extracted_at"] is not None
    # changed content -> reset to pending
    store.upsert_node("note", SLUG_A, title="v2", summary="s2")
    with db_session() as conn:
        assert conn.execute("SELECT extracted_at FROM kb_nodes WHERE id=?",
                            (nid,)).fetchone()["extracted_at"] is None


def test_add_edge_unique_keeps_max_weight():
    a = store.upsert_node("note", SLUG_A, title="A")
    b = store.upsert_node("note", SLUG_B, title="B")
    assert store.add_edge(a, b, "supports", weight=0.5)
    assert store.add_edge(a, b, "supports", weight=0.9)  # upsert, not dup
    assert store.add_edge(a, b, "supports", weight=0.2)  # keeps max
    with db_session() as conn:
        rows = conn.execute(
            "SELECT weight FROM kb_edges WHERE source_id=? AND target_id=? "
            "AND relation='supports'", (a, b)).fetchall()
    assert len(rows) == 1 and rows[0]["weight"] == 0.9


def test_add_edge_rejects_self_loops_and_bad_relations():
    a = store.upsert_node("note", SLUG_A, title="A")
    b = store.upsert_node("note", SLUG_B, title="B")
    assert store.add_edge(a, a, "supports") is False
    assert store.add_edge(a, b, "made_up_relation") is False


def test_migration_idempotent():
    c1 = migrate_kb_graph()
    with db_session() as conn:
        n1 = conn.execute("SELECT COUNT(*) AS n FROM kb_nodes").fetchone()["n"]
        e1 = conn.execute("SELECT COUNT(*) AS n FROM kb_edges").fetchone()["n"]
    migrate_kb_graph()
    with db_session() as conn:
        n2 = conn.execute("SELECT COUNT(*) AS n FROM kb_nodes").fetchone()["n"]
        e2 = conn.execute("SELECT COUNT(*) AS n FROM kb_edges").fetchone()["n"]
    assert (n1, e1) == (n2, e2)
    assert c1["techniques"] >= 20  # technique library seeds


def test_fts_reconcile_heals_missing_rows():
    nid = store.upsert_node("note", SLUG_A, title="Reconcile target",
                            summary="unique reconcile summary")
    with db_session() as conn:
        conn.execute("DELETE FROM kb_fts WHERE node_id=?", (nid,))
    result = store.fts_reconcile()
    assert result["added"] >= 1
    with db_session() as conn:
        assert conn.execute("SELECT 1 FROM kb_fts WHERE node_id=?",
                            (nid,)).fetchone() is not None
