"""Idempotent migration of the legacy KB into the GraphRAG node/edge layer.

Non-destructive: kb_documents/kb_concepts/kb_links are read, never modified.
Safe to run repeatedly — kb_nodes UNIQUE(ref_table, ref_id) and slug keys make
every step a no-op on re-run.
"""
import logging
import re

from data.database import db_session
from knowledge.graph import store

logger = logging.getLogger(__name__)


def _slugify(text: str, prefix: str = "") -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")[:60]
    return f"{prefix}{s}" if s else f"{prefix}unnamed"


def migrate_kb_graph() -> dict:
    counts = {"notes": 0, "concepts": 0, "techniques": 0,
              "rejection_patterns": 0, "ideas": 0, "edges": 0}

    # 1. kb_documents → note nodes
    with db_session() as conn:
        docs = conn.execute("SELECT * FROM kb_documents").fetchall()
    doc_node: dict[int, int] = {}
    for d in docs:
        nid = store.upsert_node(
            "note", slug=d["slug"], title=d["title"] or "",
            domain=d["domain"] or "", summary=d["summary"] or "",
            tags=d["tags"] or "", content=d["content"] or "",
            ref=("kb_documents", d["id"]),
        )
        doc_node[d["id"]] = nid
        counts["notes"] += 1

    # 2. kb_concepts → concept nodes
    with db_session() as conn:
        concepts = conn.execute("SELECT * FROM kb_concepts").fetchall()
    for c in concepts:
        store.upsert_node(
            "concept", slug=_slugify(c["name"], "concept-"),
            title=c["name"], domain=c["domain"] or "",
            summary=c["description"] or "",
            ref=("kb_concepts", c["id"]),
        )
        counts["concepts"] += 1

    # 3. TechniqueLibrary dict → technique nodes
    try:
        from knowledge.ingestion.technique_library import TECHNIQUE_LIBRARY
        for key, t in TECHNIQUE_LIBRARY.items():
            summary_bits = [
                f"When to use: {'; '.join(t.get('when_to_use', []))}",
                f"When to avoid: {'; '.join(t.get('when_to_avoid', []))}",
                f"Bursa applicability: {t.get('bursa_applicability', '')}",
                f"Complexity: {t.get('complexity', '')} | "
                f"Overfitting risk: {t.get('overfitting_risk', '')}",
            ]
            store.upsert_node(
                "technique", slug=f"tech-{_slugify(key)}",
                title=t.get("name", key), domain=t.get("angle", ""),
                summary="\n".join(summary_bits),
                tags=t.get("strategy_types", []),
            )
            counts["techniques"] += 1
    except Exception as e:
        logger.warning(f"[Migrate] Technique library skipped: {e}")

    # 4. rejection_patterns → rejection_pattern nodes
    with db_session() as conn:
        patterns = conn.execute("SELECT * FROM rejection_patterns").fetchall()
    for p in patterns:
        title = f"{p['factor_type'] or 'unknown'} / {p['reason_category'] or 'unknown'}"
        store.upsert_node(
            "rejection_pattern",
            slug=_slugify(f"{p['factor_type']}-{p['sector']}-{p['reason_category']}", "reject-"),
            title=title, domain="",
            summary=(f"Rejected {p['count']}x (last {p['last_seen']}). "
                     f"Sector: {p['sector'] or 'any'}. Example: {p['example_title'] or '-'}"),
            ref=("rejection_patterns", p["id"]),
        )
        counts["rejection_patterns"] += 1

    # 5. alpha_ideas → idea nodes. (Historical derived_from edges are not
    # reconstructable — ideas don't store their source doc; new seeds get the
    # edge live from AlphaSeedGenerator.digest.)
    with db_session() as conn:
        ideas = conn.execute(
            "SELECT id, slug, title, hypothesis, ticker, stage, status FROM alpha_ideas"
        ).fetchall()
    for i in ideas:
        store.upsert_node(
            "idea", slug=f"idea-{i['slug']}"[:120],
            title=i["title"] or "", domain="",
            summary=(f"[{i['stage']}/{i['status']}] {i['hypothesis'] or ''}")[:2000],
            tags=[i["ticker"]] if i["ticker"] else [],
            ref=("alpha_ideas", i["id"]),
        )
        counts["ideas"] += 1

    # 6. Legacy kb_links (doc-to-doc substring co-occurrence) → kb_edges at
    # low weight; these are weak signals compared to LLM-extracted relations.
    with db_session() as conn:
        links = conn.execute("SELECT * FROM kb_links").fetchall()
    for l in links:
        src = doc_node.get(l["source_id"])
        tgt = doc_node.get(l["target_id"])
        relation = l["relation"] if l["relation"] in store.RELATIONS else "mentions"
        if src and tgt and store.add_edge(src, tgt, relation, weight=0.3,
                                          origin="migration"):
            counts["edges"] += 1

    logger.info(f"[Migrate] KB graph migration complete: {counts}")
    return counts
