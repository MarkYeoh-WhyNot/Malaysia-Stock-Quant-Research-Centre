"""Optional semantic search via Voyage AI embeddings.

Activated only when VOYAGE_API_KEY is set in the environment; every function
is a clean no-op without it, so the retriever degrades to FTS-only. Plain
requests POST — no SDK dependency. Vectors stored as float32 blobs in
kb_embeddings (brute-force numpy cosine is fine at this KB's scale).
"""
import logging
import os

import numpy as np
import requests

from data.database import db_session

logger = logging.getLogger(__name__)

VOYAGE_URL = "https://api.voyageai.com/v1/embeddings"
MODEL = "voyage-3-lite"


def enabled() -> bool:
    return bool(os.environ.get("VOYAGE_API_KEY"))


def embed(texts: list[str]) -> np.ndarray | None:
    """Embed up to 128 texts; returns (n, dim) float32 array or None on failure."""
    if not enabled() or not texts:
        return None
    try:
        resp = requests.post(
            VOYAGE_URL,
            headers={"Authorization": f"Bearer {os.environ['VOYAGE_API_KEY']}"},
            json={"model": MODEL, "input": texts[:128]},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        return np.array([d["embedding"] for d in data], dtype=np.float32)
    except Exception as e:
        logger.warning(f"[Embeddings] Voyage request failed: {e}")
        return None


def embed_pending(batch: int = 64) -> int:
    """Embed nodes with no embedding or a stale content_hash. Returns count."""
    if not enabled():
        return 0
    with db_session() as conn:
        pending = conn.execute("""
            SELECT n.id, n.title, n.summary, n.content_hash
            FROM kb_nodes n
            LEFT JOIN kb_embeddings e ON e.node_id = n.id
            WHERE e.node_id IS NULL OR e.content_hash != n.content_hash
            LIMIT ?
        """, (batch,)).fetchall()
    if not pending:
        return 0

    texts = [f"{r['title'] or ''}\n{r['summary'] or ''}"[:8000] for r in pending]
    vectors = embed(texts)
    if vectors is None:
        return 0

    with db_session() as conn:
        for row, vec in zip(pending, vectors):
            conn.execute("""
                INSERT INTO kb_embeddings (node_id, model, dim, vector, content_hash,
                                           updated_at)
                VALUES (?, ?, ?, ?, ?, datetime('now'))
                ON CONFLICT(node_id) DO UPDATE SET
                    model=excluded.model, dim=excluded.dim, vector=excluded.vector,
                    content_hash=excluded.content_hash, updated_at=datetime('now')
            """, (row["id"], MODEL, len(vec), vec.tobytes(), row["content_hash"]))
    logger.info(f"[Embeddings] Embedded {len(pending)} nodes")
    return len(pending)


def cosine_search(query: str, k: int = 30) -> list[tuple[int, float]]:
    """[(node_id, cosine 0..1)] best-first; [] when disabled or on failure."""
    if not enabled():
        return []
    qv = embed([query])
    if qv is None:
        return []
    qv = qv[0]
    qv = qv / (np.linalg.norm(qv) + 1e-9)

    with db_session() as conn:
        rows = conn.execute("SELECT node_id, vector, dim FROM kb_embeddings").fetchall()
    if not rows:
        return []

    ids = np.array([r["node_id"] for r in rows])
    mat = np.vstack([np.frombuffer(r["vector"], dtype=np.float32) for r in rows])
    mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
    sims = mat @ qv
    # cosine of normalized embeddings is [-1, 1]; clip to [0, 1] for scoring
    sims = np.clip(sims, 0.0, 1.0)
    order = np.argsort(-sims)[:k]
    return [(int(ids[i]), float(sims[i])) for i in order]
