"""Vault exporter and extractor tests."""
import os
import re

import pytest

from data.database import db_session, init_db
from knowledge.graph import store

SLUG_A = "test-exp-note-a"
SLUG_B = "test-exp-note-b"


@pytest.fixture(autouse=True)
def _setup():
    init_db()
    _cleanup()
    yield
    _cleanup()


def _cleanup():
    with db_session() as conn:
        rows = conn.execute(
            "SELECT id FROM kb_nodes WHERE slug LIKE 'test-exp-%'").fetchall()
        ids = [r["id"] for r in rows]
        if ids:
            marks = ",".join("?" * len(ids))
            conn.execute(f"DELETE FROM kb_edges WHERE source_id IN ({marks}) "
                         f"OR target_id IN ({marks})", ids + ids)
            conn.execute(f"DELETE FROM kb_fts WHERE node_id IN ({marks})", ids)
            conn.execute(f"DELETE FROM kb_nodes WHERE id IN ({marks})", ids)


def test_export_roundtrip(tmp_path):
    from scripts.export_obsidian import export_vault
    a = store.upsert_node("note", SLUG_A, title="Export A", domain="event_driven",
                          summary="summary A", tags=["dividend"])
    b = store.upsert_node("note", SLUG_B, title="Export B", domain="macro",
                          summary="summary B")
    store.add_edge(a, b, "refines", weight=0.7, origin="manual")

    result = export_vault(str(tmp_path / "vault"))
    assert result["notes"] >= 2

    path = tmp_path / "vault" / "notes" / f"{SLUG_A}.md"
    text = path.read_text()
    # frontmatter parses
    assert text.startswith("---")
    assert "type: note" in text and 'domain: "event_driven"' in text
    # wikilink round-trips to the edge target
    links = re.findall(r"\[\[([^\]]+)\]\]", text)
    assert SLUG_B in links
    assert "### refines" in text


def test_export_wipes_stale_files(tmp_path):
    from scripts.export_obsidian import export_vault
    out = tmp_path / "vault"
    store.upsert_node("note", SLUG_A, title="Only note")
    export_vault(str(out))
    stale = out / "notes" / "stale-file.md"
    stale.write_text("should be removed")
    export_vault(str(out))
    assert not stale.exists()


def test_extractor_writes_edges_and_marks_done(monkeypatch):
    from knowledge.graph.extractor import GraphExtractor
    a = store.upsert_node("note", SLUG_A, title="EPF flows note",
                          summary="EPF rebalancing moves banking stocks")
    b = store.upsert_node("note", SLUG_B, title="Banking OPR sensitivity",
                          summary="Banks track OPR decisions")

    fake_response = {"notes": [{
        "slug": SLUG_A,
        "relations": [
            {"target_slug": SLUG_B, "relation": "supports", "weight": 0.8,
             "reason": "same mechanism"},
            {"target_slug": "hallucinated-slug", "relation": "supports",
             "weight": 0.9, "reason": "must be dropped"},
        ],
        "new_concepts": [],
    }]}
    monkeypatch.setattr(GraphExtractor, "call_claude_json",
                        lambda self, *a, **k: fake_response)

    ex = GraphExtractor.__new__(GraphExtractor)
    import logging
    ex.logger = logging.getLogger("test")

    result = ex._extract_batch([
        {"id": a, "slug": SLUG_A, "title": "EPF flows note",
         "domain": "institutional", "summary": "EPF rebalancing moves banking stocks"},
    ])
    assert result["processed"] == 1
    with db_session() as conn:
        edges = conn.execute(
            "SELECT relation, weight FROM kb_edges WHERE source_id=?", (a,)
        ).fetchall()
        node = conn.execute("SELECT extracted_at FROM kb_nodes WHERE id=?",
                            (a,)).fetchone()
    # hallucinated target dropped; only the valid edge written
    assert len(edges) == 1 and edges[0]["relation"] == "supports"
    assert node["extracted_at"] is not None
