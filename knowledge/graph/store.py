"""GraphRAG node/edge store — the ONLY write path for kb_nodes/kb_edges/kb_fts.

All graph writes funnel through here so the FTS index stays in sync without
triggers (three processes share the SQLite file; legacy writers that bypass
this module are healed by the nightly fts_reconcile daemon job).
"""
import hashlib
import json
import logging

from data.database import db_session

logger = logging.getLogger(__name__)

NODE_TYPES = ("note", "concept", "technique", "idea", "rejection_pattern")

RELATIONS = (
    "supports", "contradicts", "refines", "derived_from", "about_ticker",
    "uses_technique", "rejected_because", "shared_concept", "shared_tag",
    "mentions",
)


def content_hash(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update((p or "").encode("utf-8", errors="replace"))
        h.update(b"\x00")
    return h.hexdigest()


def upsert_node(node_type: str, slug: str, title: str = "", domain: str = "",
                summary: str = "", tags=None, content: str = "",
                ref: tuple[str, int] | None = None) -> int:
    """Insert or update a node; sync its kb_fts row; return node id.

    If the content hash changed, extracted_at is reset to NULL so the LLM
    edge extractor re-processes the node on its next pass.
    """
    if node_type not in NODE_TYPES:
        raise ValueError(f"Invalid node_type: {node_type}")
    tags_str = json.dumps(tags) if isinstance(tags, (list, dict)) else (tags or "")
    chash = content_hash(title, summary, content, tags_str)
    ref_table, ref_id = ref if ref else (None, None)

    with db_session() as conn:
        existing = conn.execute(
            "SELECT id, content_hash FROM kb_nodes WHERE slug=? "
            "OR (ref_table IS NOT NULL AND ref_table=? AND ref_id=?)",
            (slug, ref_table, ref_id),
        ).fetchone()

        if existing:
            node_id = existing["id"]
            if existing["content_hash"] != chash:
                conn.execute("""
                    UPDATE kb_nodes
                    SET title=?, domain=?, summary=?, tags=?, content_hash=?,
                        extracted_at=NULL, updated_at=datetime('now')
                    WHERE id=?
                """, (title, domain, summary, tags_str, chash, node_id))
                _sync_fts(conn, node_id, title, summary, content, tags_str)
        else:
            conn.execute("""
                INSERT INTO kb_nodes
                    (node_type, ref_table, ref_id, slug, title, domain,
                     summary, tags, content_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (node_type, ref_table, ref_id, slug, title, domain,
                  summary, tags_str, chash))
            node_id = conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
            _sync_fts(conn, node_id, title, summary, content, tags_str)
    return node_id


def _sync_fts(conn, node_id: int, title: str, summary: str,
              content: str, tags: str):
    conn.execute("DELETE FROM kb_fts WHERE node_id=?", (node_id,))
    conn.execute(
        "INSERT INTO kb_fts (title, summary, content, tags, node_id) "
        "VALUES (?, ?, ?, ?, ?)",
        (title or "", summary or "", (content or "")[:20000], tags or "", node_id),
    )


def add_edge(source_id: int, target_id: int, relation: str,
             weight: float = 1.0, origin: str = "llm") -> bool:
    """Idempotent typed edge; concurrent duplicates collapse onto one row,
    keeping the strongest weight. Returns False for self-loops/bad relations."""
    if source_id == target_id:
        return False
    if relation not in RELATIONS:
        logger.warning(f"[GraphStore] Dropping edge with unknown relation {relation!r}")
        return False
    weight = max(0.0, min(1.0, float(weight)))
    with db_session() as conn:
        conn.execute("""
            INSERT INTO kb_edges (source_id, target_id, relation, weight, origin)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(source_id, target_id, relation)
            DO UPDATE SET weight=max(weight, excluded.weight)
        """, (source_id, target_id, relation, weight, origin))
    return True


def get_node(node_id: int = None, slug: str = None):
    with db_session() as conn:
        if node_id is not None:
            return conn.execute("SELECT * FROM kb_nodes WHERE id=?", (node_id,)).fetchone()
        return conn.execute("SELECT * FROM kb_nodes WHERE slug=?", (slug,)).fetchone()


def neighbors(node_id: int) -> list[dict]:
    """Edges touching a node in either direction, with the far node's info."""
    with db_session() as conn:
        rows = conn.execute("""
            SELECT e.relation, e.weight, e.origin,
                   CASE WHEN e.source_id=? THEN 'out' ELSE 'in' END AS direction,
                   n.id AS node_id, n.slug, n.title, n.node_type, n.domain
            FROM kb_edges e
            JOIN kb_nodes n ON n.id = CASE WHEN e.source_id=? THEN e.target_id
                                           ELSE e.source_id END
            WHERE e.source_id=? OR e.target_id=?
            ORDER BY e.weight DESC
        """, (node_id, node_id, node_id, node_id)).fetchall()
    return [dict(r) for r in rows]


def graph_json(limit: int = 500, domain: str = None) -> dict:
    """Nodes+edges payload for the dashboard graph view."""
    with db_session() as conn:
        if domain:
            nodes = conn.execute(
                "SELECT id, slug, title, node_type, domain, summary FROM kb_nodes "
                "WHERE domain=? ORDER BY updated_at DESC LIMIT ?",
                (domain, limit)).fetchall()
        else:
            nodes = conn.execute(
                "SELECT id, slug, title, node_type, domain, summary FROM kb_nodes "
                "ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()
        ids = [n["id"] for n in nodes]
        edges = []
        if ids:
            marks = ",".join("?" * len(ids))
            edges = conn.execute(
                f"SELECT source_id, target_id, relation, weight FROM kb_edges "
                f"WHERE source_id IN ({marks}) AND target_id IN ({marks})",
                ids + ids).fetchall()
    return {
        "nodes": [{"id": n["id"], "slug": n["slug"], "title": n["title"],
                   "type": n["node_type"], "domain": n["domain"],
                   "summary": (n["summary"] or "")[:280]} for n in nodes],
        "edges": [{"source": e["source_id"], "target": e["target_id"],
                   "relation": e["relation"], "weight": e["weight"]} for e in edges],
    }


def ensure_node_for_document(doc_id: int) -> int | None:
    """Lazy bridge: make sure a kb_documents row has a note node (used when a
    legacy write path created a doc without going through upsert_node)."""
    with db_session() as conn:
        doc = conn.execute("SELECT * FROM kb_documents WHERE id=?", (doc_id,)).fetchone()
    if not doc:
        return None
    return upsert_node(
        "note", slug=doc["slug"], title=doc["title"] or "",
        domain=doc["domain"] or "", summary=doc["summary"] or "",
        tags=doc["tags"] or "", content=doc["content"] or "",
        ref=("kb_documents", doc_id),
    )


def fts_reconcile() -> dict:
    """Self-heal the FTS index: remove orphaned rows, index missing nodes.
    Run nightly by the daemon; makes app-level sync safe without triggers."""
    added, removed = 0, 0
    with db_session() as conn:
        orphans = conn.execute("""
            SELECT f.rowid FROM kb_fts f
            LEFT JOIN kb_nodes n ON n.id = f.node_id
            WHERE n.id IS NULL
        """).fetchall()
        for row in orphans:
            conn.execute("DELETE FROM kb_fts WHERE rowid=?", (row["rowid"],))
            removed += 1
        missing = conn.execute("""
            SELECT n.id FROM kb_nodes n
            LEFT JOIN kb_fts f ON f.node_id = n.id
            WHERE f.node_id IS NULL
        """).fetchall()
    for row in missing:
        node = get_node(node_id=row["id"])
        if node:
            with db_session() as conn:
                _sync_fts(conn, node["id"], node["title"], node["summary"],
                          "", node["tags"])
            added += 1
    if added or removed:
        logger.info(f"[GraphStore] FTS reconcile: +{added} indexed, -{removed} orphans")
    return {"added": added, "removed": removed}
