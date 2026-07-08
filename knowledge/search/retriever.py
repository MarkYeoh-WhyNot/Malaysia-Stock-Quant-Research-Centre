"""GraphRAG retriever — the public entry point for all KB retrieval.

Algorithm (Obsidian-style graph-aware retrieval):
1. Entry points: hybrid search — FTS5 BM25 (always on) blended with Voyage
   cosine similarity when embeddings are enabled.
2. Graph walk: BFS 1..hops from the seed notes over typed weighted edges,
   like following [[wikilinks]] outward; each hop discounts relevance.
3. Score + assemble: every touched node scored by its best path
   (seed_score × Π(edge_weight) × decay^hop), deduped, top-k returned with
   the path that reached it (so agents see WHY a note is in context).
"""
import logging

from data.database import db_session
from knowledge.search import fts, embeddings

logger = logging.getLogger(__name__)

HOP_DECAY = 0.5
SEED_LIMIT = 12
BM25_WEIGHT = 0.6
COSINE_WEIGHT = 0.4

# Relation modifiers: contradictions stay visible (flagged) but rank slightly
# lower; weak co-occurrence relations propagate less relevance.
RELATION_MODIFIER = {
    "contradicts": 0.8,
    "shared_tag": 0.6,
    "shared_concept": 0.8,
    "mentions": 0.7,
}


def retrieve(query: str, k: int = 8, hops: int = 2,
             node_types: list[str] | None = None,
             domain: str | None = None) -> list[dict]:
    # ── 1. Seeds: hybrid FTS + optional cosine ────────────────────────────
    fts_hits = dict(fts.fts_search(query, k=30, node_types=node_types, domain=domain))
    cos_hits = dict(embeddings.cosine_search(query, k=30)) if embeddings.enabled() else {}

    seed_scores: dict[int, float] = {}
    if cos_hits:
        for nid in set(fts_hits) | set(cos_hits):
            seed_scores[nid] = (BM25_WEIGHT * fts_hits.get(nid, 0.0)
                                + COSINE_WEIGHT * cos_hits.get(nid, 0.0))
    else:
        seed_scores = dict(fts_hits)

    seeds = sorted(seed_scores.items(), key=lambda x: -x[1])[:SEED_LIMIT]
    if not seeds:
        return []

    # ── 2. Graph walk: BFS with decay, keeping each node's best path ─────
    best: dict[int, dict] = {
        nid: {"score": s, "via": [], "contradicts": False}
        for nid, s in seeds
    }
    frontier = dict(seeds)
    for hop in range(1, max(0, hops) + 1):
        if not frontier:
            break
        next_frontier: dict[int, float] = {}
        ids = list(frontier.keys())
        marks = ",".join("?" * len(ids))
        with db_session() as conn:
            edges = conn.execute(f"""
                SELECT e.source_id, e.target_id, e.relation, e.weight,
                       n1.slug AS source_slug, n2.slug AS target_slug
                FROM kb_edges e
                JOIN kb_nodes n1 ON n1.id = e.source_id
                JOIN kb_nodes n2 ON n2.id = e.target_id
                WHERE e.source_id IN ({marks}) OR e.target_id IN ({marks})
            """, ids + ids).fetchall()

        for e in edges:
            for from_id, to_id, from_slug in (
                (e["source_id"], e["target_id"], e["source_slug"]),
                (e["target_id"], e["source_id"], e["target_slug"]),
            ):
                if from_id not in frontier:
                    continue
                modifier = RELATION_MODIFIER.get(e["relation"], 1.0)
                score = (frontier[from_id] * float(e["weight"] or 1.0)
                         * modifier * (HOP_DECAY ** hop))
                if score <= 0.01:
                    continue
                entry = best.get(to_id)
                if entry is None or score > entry["score"]:
                    prev_via = best.get(from_id, {}).get("via", [])
                    best[to_id] = {
                        "score": score,
                        "via": prev_via + [(e["relation"], from_slug)],
                        "contradicts": e["relation"] == "contradicts"
                                       or (entry or {}).get("contradicts", False),
                    }
                    next_frontier[to_id] = max(next_frontier.get(to_id, 0), score)
        frontier = next_frontier

    # ── 3. Materialize, filter, top-k ──────────────────────────────────────
    ids = list(best.keys())
    marks = ",".join("?" * len(ids))
    with db_session() as conn:
        rows = conn.execute(f"""
            SELECT id, slug, title, node_type, domain, summary, ref_table, ref_id
            FROM kb_nodes WHERE id IN ({marks})
        """, ids).fetchall()

    results = []
    for r in rows:
        if node_types and r["node_type"] not in node_types:
            # seeds were filtered already; graph hops may surface other types —
            # keep them (that's the point of the graph) unless caller filtered
            pass
        meta = best[r["id"]]
        results.append({
            "node_id": r["id"], "slug": r["slug"], "title": r["title"],
            "node_type": r["node_type"], "domain": r["domain"],
            "summary": r["summary"], "score": round(meta["score"], 4),
            "via": meta["via"], "contradicts": meta["contradicts"],
            "ref_table": r["ref_table"], "ref_id": r["ref_id"],
        })
    results.sort(key=lambda x: -x["score"])
    return results[:k]


def assemble_context(results: list[dict], max_chars: int = 4000) -> str:
    """Pack retrieval results into a prompt-ready context block, including the
    relationship paths so the agent sees how knowledge connects."""
    if not results:
        return ""
    parts = ["KNOWLEDGE GRAPH CONTEXT (connected notes from the research KB):"]
    used = len(parts[0])
    for r in results:
        via = ""
        if r["via"]:
            chain = " → ".join(f"[{rel}] {slug}" for rel, slug in r["via"])
            via = f" (reached via {chain})"
        flag = " ⚠ CONTRADICTS related note" if r["contradicts"] else ""
        block = (f"\n• [{r['node_type']}/{r['domain']}] {r['title']}{via}{flag}\n"
                 f"  {(r['summary'] or '')[:400]}")
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "".join(parts)
