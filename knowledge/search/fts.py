"""FTS5 keyword search over kb_nodes (BM25-ranked entry-point finder)."""
import logging
import re

from data.database import db_session

logger = logging.getLogger(__name__)


def _sanitize_query(query: str) -> str:
    """Turn free text into a safe FTS5 MATCH expression: quoted tokens OR'd
    together (FTS5 syntax characters in user input would otherwise raise)."""
    tokens = re.findall(r"[A-Za-z0-9.]{2,}", query or "")
    if not tokens:
        return ""
    return " OR ".join(f'"{t}"' for t in tokens[:12])


def fts_search(query: str, k: int = 30, node_types: list[str] | None = None,
               domain: str | None = None) -> list[tuple[int, float]]:
    """Returns [(node_id, score 0..1)] best-first. Falls back to LIKE on any
    FTS5 error so retrieval never hard-fails on odd input."""
    match_expr = _sanitize_query(query)
    if not match_expr:
        return []

    filters, params = [], []
    if node_types:
        filters.append(f"n.node_type IN ({','.join('?' * len(node_types))})")
        params.extend(node_types)
    if domain:
        filters.append("n.domain=?")
        params.append(domain)
    where_extra = (" AND " + " AND ".join(filters)) if filters else ""

    try:
        with db_session() as conn:
            rows = conn.execute(f"""
                SELECT f.node_id, rank
                FROM kb_fts f
                JOIN kb_nodes n ON n.id = f.node_id
                WHERE kb_fts MATCH ?{where_extra}
                ORDER BY rank LIMIT ?
            """, [match_expr] + params + [k]).fetchall()
        # FTS5 rank is negative BM25 (more negative = better). Normalize 0..1.
        return [(r["node_id"], _norm_rank(r["rank"])) for r in rows]
    except Exception as e:
        logger.warning(f"[FTS] MATCH failed ({e}) — falling back to LIKE")
        return _like_fallback(query, k, node_types, domain)


def _norm_rank(rank: float) -> float:
    """FTS5 bm25 rank: negative, closer to 0 = worse; more negative = better.
    Map to 0..1 via score = -rank / (1 + -rank)."""
    r = max(0.0, -float(rank))
    return r / (1.0 + r)


def _like_fallback(query: str, k: int, node_types, domain) -> list[tuple[int, float]]:
    tokens = [t for t in re.findall(r"[A-Za-z0-9.]{2,}", (query or "").lower())][:6]
    if not tokens:
        return []
    clauses, params = [], []
    for t in tokens:
        clauses.append("(LOWER(title) LIKE ? OR LOWER(summary) LIKE ? OR LOWER(tags) LIKE ?)")
        params.extend([f"%{t}%"] * 3)
    filters = []
    if node_types:
        filters.append(f"node_type IN ({','.join('?' * len(node_types))})")
        params.extend(node_types)
    if domain:
        filters.append("domain=?")
        params.append(domain)
    where = " OR ".join(clauses)
    if filters:
        where = f"({where}) AND " + " AND ".join(filters)
    with db_session() as conn:
        rows = conn.execute(
            f"SELECT id FROM kb_nodes WHERE {where} ORDER BY updated_at DESC LIMIT ?",
            params + [k]).fetchall()
    return [(r["id"], 0.3) for r in rows]  # flat modest score — no ranking signal
