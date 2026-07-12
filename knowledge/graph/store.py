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

# Fallback set only — the authoritative list lives in kb_node_type_registry and
# is validated at write time (see _check_node_type). Keep in sync for the case
# where the registry table is missing (very old DB / first init).
NODE_TYPES = (
    "note", "concept", "technique", "idea", "rejection_pattern",
    "strategy", "signature", "backtest_run", "gate_decision", "risk",
    "finding", "leaf", "agent",
)

RELATIONS = (
    # concept-graph (original)
    "supports", "contradicts", "refines", "derived_from", "about_ticker",
    "uses_technique", "rejected_because", "shared_concept", "shared_tag",
    "mentions",
    # evidence / truth graph (2026-07-12)
    "produced", "failed", "passed", "shares_signature", "reported_by",
    "blocks", "measured_by", "exposed_to", "affects", "compiled_to",
    "uses_leaf",
)

# Lazily-loaded cache of valid node types from kb_node_type_registry.
_NODE_TYPE_CACHE: set | None = None


def _check_node_type(conn, node_type: str) -> None:
    """Validate node_type against kb_node_type_registry (cached, refreshed on
    miss). Raises ValueError for an unregistered type — the same discipline
    add_edge applies to relations, without a DB-level CHECK."""
    global _NODE_TYPE_CACHE
    if _NODE_TYPE_CACHE is not None and node_type in _NODE_TYPE_CACHE:
        return
    try:
        rows = conn.execute(
            "SELECT node_type FROM kb_node_type_registry WHERE status='active'"
        ).fetchall()
        _NODE_TYPE_CACHE = {r["node_type"] for r in rows} or set(NODE_TYPES)
    except Exception:
        _NODE_TYPE_CACHE = set(NODE_TYPES)
    if node_type not in _NODE_TYPE_CACHE:
        raise ValueError(
            f"Invalid node_type {node_type!r} (not in kb_node_type_registry)")


def content_hash(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update((p or "").encode("utf-8", errors="replace"))
        h.update(b"\x00")
    return h.hexdigest()


def upsert_node(node_type: str, slug: str, title: str = "", domain: str = "",
                summary: str = "", tags=None, content: str = "",
                ref: tuple[str, int] | None = None,
                confidence: float | None = None,
                review_state: str | None = None,
                ingestion_version: str | None = None) -> int:
    """Insert or update a node; sync its kb_fts row; return node id.

    If the content hash changed, extracted_at is reset to NULL so the LLM
    edge extractor re-processes the node on its next pass.

    confidence / review_state / ingestion_version are optional provenance
    fields (deterministic ingesters stamp ingestion_version; the feedback loop
    drives review_state). Passing None leaves the existing value untouched.
    """
    tags_str = json.dumps(tags) if isinstance(tags, (list, dict)) else (tags or "")
    chash = content_hash(title, summary, content, tags_str)
    ref_table, ref_id = ref if ref else (None, None)

    with db_session() as conn:
        _check_node_type(conn, node_type)
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

        # Optional provenance columns — only overwrite when a value is supplied.
        sets, params = [], []
        if confidence is not None:
            sets.append("confidence=?"); params.append(confidence)
        if review_state is not None:
            sets.append("review_state=?"); params.append(review_state)
        if ingestion_version is not None:
            sets.append("ingestion_version=?"); params.append(ingestion_version)
        if sets:
            params.append(node_id)
            conn.execute(f"UPDATE kb_nodes SET {', '.join(sets)} WHERE id=?", params)
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
             weight: float = 1.0, origin: str = "llm",
             count_evidence: bool = False) -> bool:
    """Idempotent typed edge; concurrent duplicates collapse onto one row,
    keeping the strongest weight. Returns False for self-loops/bad relations.

    count_evidence=True bumps evidence_count on a repeat sighting (for rollup
    edges where the same relation is re-observed across many source rows); it
    stays False for the default 1:1 evidence-node wiring where the distinct
    nodes already ARE the evidence.
    """
    if source_id == target_id:
        return False
    if relation not in RELATIONS:
        logger.warning(f"[GraphStore] Dropping edge with unknown relation {relation!r}")
        return False
    weight = max(0.0, min(1.0, float(weight)))
    inc = 1 if count_evidence else 0
    with db_session() as conn:
        conn.execute("""
            INSERT INTO kb_edges (source_id, target_id, relation, weight, origin, last_seen_at)
            VALUES (?, ?, ?, ?, ?, datetime('now'))
            ON CONFLICT(source_id, target_id, relation)
            DO UPDATE SET weight=max(weight, excluded.weight),
                          evidence_count=evidence_count + ?,
                          last_seen_at=datetime('now')
        """, (source_id, target_id, relation, weight, origin, inc))
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


def graph_json(limit: int = 500, domain: str = None, since: str = None,
               node_type: str = None) -> dict:
    """Nodes+edges payload for the dashboard graph view.

    With `since` (an as_of timestamp from a previous call), returns only
    nodes updated and edges created after that moment — the live-view delta
    protocol. Delta edges may reference nodes outside the delta; the client
    ignores edges whose endpoints it doesn't hold (an hourly full refresh
    reconciles, including deletions which deltas can't see).

    node_type accepts a single type or a comma-separated list (saved views pass
    the set of types a lens needs, e.g. "strategy,gate_decision,rejection_pattern").
    """
    from datetime import datetime, timezone
    as_of = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    filters, params = [], []
    if domain:
        filters.append("domain=?")
        params.append(domain)
    if node_type:
        types = [t.strip() for t in node_type.split(",") if t.strip()]
        if types:
            filters.append(f"node_type IN ({','.join('?' * len(types))})")
            params.extend(types)
    if since:
        filters.append("updated_at > ?")
        params.append(since)
    where = ("WHERE " + " AND ".join(filters)) if filters else ""

    with db_session() as conn:
        nodes = conn.execute(
            f"SELECT id, slug, title, node_type, domain, summary, review_state FROM kb_nodes "
            f"{where} ORDER BY updated_at DESC LIMIT ?",
            params + [limit]).fetchall()
        edges = []
        if since:
            edges = conn.execute(
                "SELECT source_id, target_id, relation, weight FROM kb_edges "
                "WHERE created_at > ? LIMIT ?", (since, limit * 4)).fetchall()
        else:
            ids = [n["id"] for n in nodes]
            if ids:
                marks = ",".join("?" * len(ids))
                edges = conn.execute(
                    f"SELECT source_id, target_id, relation, weight FROM kb_edges "
                    f"WHERE source_id IN ({marks}) AND target_id IN ({marks})",
                    ids + ids).fetchall()
    return {
        "as_of": as_of,
        "delta": bool(since),
        "nodes": [{"id": n["id"], "slug": n["slug"], "title": n["title"],
                   "type": n["node_type"], "domain": n["domain"],
                   "review_state": n["review_state"],
                   "summary": (n["summary"] or "")[:280]} for n in nodes],
        "edges": [{"source": e["source_id"], "target": e["target_id"],
                   "relation": e["relation"], "weight": e["weight"]} for e in edges],
    }


def subgraph_json(node_id: int, hops: int = 2, max_nodes: int = 150) -> dict:
    """K-hop neighborhood of a node as a flat table-friendly payload — the
    'extract subgraph' feature. BFS over kb_edges in both directions."""
    frontier = {node_id}
    seen = {node_id}
    rows = []
    with db_session() as conn:
        for hop in range(1, max(1, hops) + 1):
            if not frontier or len(seen) >= max_nodes:
                break
            marks = ",".join("?" * len(frontier))
            ids = list(frontier)
            edges = conn.execute(f"""
                SELECT e.source_id, e.target_id, e.relation, e.weight,
                       n1.slug AS source_slug, n1.title AS source_title,
                       n2.slug AS target_slug, n2.title AS target_title,
                       n2.node_type AS target_type, n1.node_type AS source_type
                FROM kb_edges e
                JOIN kb_nodes n1 ON n1.id = e.source_id
                JOIN kb_nodes n2 ON n2.id = e.target_id
                WHERE e.source_id IN ({marks}) OR e.target_id IN ({marks})
            """, ids + ids).fetchall()
            next_frontier = set()
            for e in edges:
                rows.append({"hop": hop, "source": e["source_slug"],
                             "source_title": e["source_title"],
                             "source_type": e["source_type"],
                             "relation": e["relation"], "weight": e["weight"],
                             "target": e["target_slug"],
                             "target_title": e["target_title"],
                             "target_type": e["target_type"]})
                for nid in (e["source_id"], e["target_id"]):
                    if nid not in seen and len(seen) < max_nodes:
                        seen.add(nid)
                        next_frontier.add(nid)
            frontier = next_frontier
        center = conn.execute(
            "SELECT slug, title FROM kb_nodes WHERE id=?", (node_id,)).fetchone()
    # dedupe edge rows (an edge can be touched from both endpoints)
    unique = {(r["source"], r["target"], r["relation"]): r for r in rows}
    return {"center": dict(center) if center else None,
            "hops": hops, "node_count": len(seen),
            "edges": sorted(unique.values(), key=lambda r: (r["hop"], -r["weight"]))}


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
