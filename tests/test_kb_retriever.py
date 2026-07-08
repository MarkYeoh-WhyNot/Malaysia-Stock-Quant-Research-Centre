"""Retriever tests: FTS ranking, no-key fallback, graph traversal scoring."""
import pytest

from data.database import db_session, init_db
from knowledge.graph import store
from knowledge.search import fts, embeddings
from knowledge.search.retriever import retrieve, assemble_context, HOP_DECAY

SLUGS = ["test-ret-dividend", "test-ret-settlement", "test-ret-unrelated"]


@pytest.fixture(autouse=True)
def _setup(monkeypatch):
    init_db()
    _cleanup()
    # any accidental network call must fail loudly, not silently pass
    import requests

    def _boom(*a, **k):
        raise AssertionError("network call attempted without VOYAGE_API_KEY")
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    monkeypatch.setattr(requests, "post", _boom)
    yield
    _cleanup()


def _cleanup():
    with db_session() as conn:
        rows = conn.execute(
            "SELECT id FROM kb_nodes WHERE slug LIKE 'test-ret-%'").fetchall()
        ids = [r["id"] for r in rows]
        if ids:
            marks = ",".join("?" * len(ids))
            conn.execute(f"DELETE FROM kb_edges WHERE source_id IN ({marks}) "
                         f"OR target_id IN ({marks})", ids + ids)
            conn.execute(f"DELETE FROM kb_fts WHERE node_id IN ({marks})", ids)
            conn.execute(f"DELETE FROM kb_nodes WHERE id IN ({marks})", ids)


def _seed_chain():
    """dividend --supports(0.8)--> settlement ; unrelated stands alone."""
    a = store.upsert_node("note", SLUGS[0], title="Dividend capture strategy",
                          summary="Buying before zzqxdividend ex-date on Bursa")
    b = store.upsert_node("note", SLUGS[1], title="T+3 settlement friction",
                          summary="Settlement timing constrains short holds")
    c = store.upsert_node("note", SLUGS[2], title="Palm oil inventory cycles",
                          summary="CPO stockpiles drive plantation earnings")
    store.add_edge(a, b, "supports", weight=0.8, origin="manual")
    return a, b, c


def test_fts_ranks_relevant_first():
    _seed_chain()
    hits = fts.fts_search("zzqxdividend capture ex-date")
    assert hits, "expected FTS hits"
    top = store.get_node(node_id=hits[0][0])
    assert top["slug"] == SLUGS[0]


def test_fts_syntax_error_falls_back():
    _seed_chain()
    # unbalanced parens/AND would crash raw FTS5 MATCH; sanitizer must cope
    hits = fts.fts_search('zzqxdividend AND (capture')
    assert any(store.get_node(node_id=h[0])["slug"] == SLUGS[0] for h in hits)


def test_embeddings_disabled_is_clean_noop():
    assert embeddings.enabled() is False
    assert embeddings.cosine_search("anything") == []
    assert embeddings.embed_pending() == 0


def test_graph_walk_pulls_linked_note_with_decayed_score():
    _seed_chain()
    results = retrieve("zzqxdividend capture ex-date", k=5, hops=2)
    by_slug = {r["slug"]: r for r in results}
    assert SLUGS[0] in by_slug, "seed note missing"
    assert SLUGS[1] in by_slug, "graph-linked note not pulled in"
    seed = by_slug[SLUGS[0]]
    linked = by_slug[SLUGS[1]]
    # linked score = seed_score * edge_weight(0.8) * decay^1
    assert linked["score"] == pytest.approx(seed["score"] * 0.8 * HOP_DECAY, rel=0.05)
    assert linked["via"] and linked["via"][0][0] == "supports"


def test_contradicts_surfaced_and_flagged():
    a = store.upsert_node("note", SLUGS[0], title="Dividend capture works",
                          summary="zzqxdividend capture earns alpha")
    b = store.upsert_node("note", SLUGS[1], title="Costs kill dividend capture",
                          summary="Transaction costs exceed the ex-date drop")
    store.add_edge(a, b, "contradicts", weight=0.9, origin="manual")
    results = retrieve("zzqxdividend capture", k=5, hops=1)
    linked = next((r for r in results if r["slug"] == SLUGS[1]), None)
    assert linked is not None, "contradicting note must still be surfaced"
    assert linked["contradicts"] is True


def test_assemble_context_includes_paths_and_flags():
    _seed_chain()
    results = retrieve("zzqxdividend capture ex-date", k=5, hops=2)
    ctx = assemble_context(results)
    assert "KNOWLEDGE GRAPH CONTEXT" in ctx
    assert "supports" in ctx  # relationship path shown to the agent
