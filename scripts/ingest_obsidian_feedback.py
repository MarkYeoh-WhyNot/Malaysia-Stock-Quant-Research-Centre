#!/usr/bin/env python3
"""Ingest human feedback authored in Obsidian (vault/feedback/) back into the loop.

This is the return path of the Obsidian bridge. scripts/export_obsidian.py writes
the read-only graph out; this reads the ONE human-editable zone (vault/feedback/)
back in. Each feedback note carries YAML frontmatter (target / verdict / rating /
tags / reviewer) plus a free-form body.

Effects (all idempotent, keyed off a content hash in kb_feedback):
  - Any body note      -> a human `note` kb_node linked to the target (mentions),
                          which syncs kb_fts so the retriever grounds future
                          generation on your reasoning.
  - verdict = reject   -> RejectionMemory.record_rejection (stops the loop from
                          regenerating this class) + alpha_ideas.status='rejected'
                          + an audit row in gate_decisions.
  - verdict = promote  -> audit row in gate_decisions (human gate).
  - verdict = watch    -> audit row in gate_decisions (human gate).
  - rating / tags      -> stored on the kb_feedback row and folded into the note
                          node's tags, feeding idea-generation prioritisation.

Re-ingesting an unchanged file is a no-op. The DB — not the vault — is the
source of truth for the ingested state; deleting a file does NOT undo it.

Usage: python scripts/ingest_obsidian_feedback.py [feedback_dir]
       (defaults to vault/feedback/)
"""
import hashlib
import json
import logging
import os
import sys

import yaml

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from data.database import db_session
from knowledge.graph import store
from knowledge.ingestion.rejection_memory import RejectionMemory

logger = logging.getLogger(__name__)

DEFAULT_FEEDBACK_DIR = os.path.join(_REPO_ROOT, "vault", "feedback")
VALID_VERDICTS = {"promote", "reject", "watch"}
SKIP_FILES = {"_template.md", "readme.md"}


def _parse_feedback_file(path: str) -> dict | None:
    """Split a note into (frontmatter dict, body). Returns None if unusable."""
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    if not text.lstrip().startswith("---"):
        return None
    body = text.split("---", 2)
    if len(body) < 3:
        return None
    try:
        meta = yaml.safe_load(body[1]) or {}
    except yaml.YAMLError as e:
        logger.warning(f"[FeedbackIngest] Bad YAML in {os.path.basename(path)}: {e}")
        return None
    if not isinstance(meta, dict):
        return None
    # Strip HTML comments from the body so the template's hint text isn't ingested.
    raw_body = body[2]
    note_lines = [ln for ln in raw_body.splitlines()
                  if not ln.strip().startswith("<!--")
                  and not ln.strip().endswith("-->")
                  and "<!--" not in ln]
    note = "\n".join(note_lines).strip()
    return {"meta": meta, "note": note}


def _normalise(meta: dict, note: str) -> dict | None:
    target = str(meta.get("target") or "").strip()
    if not target:
        return None  # a feedback note with no target is inert
    verdict = str(meta.get("verdict") or "").strip().lower() or None
    if verdict and verdict not in VALID_VERDICTS:
        logger.warning(f"[FeedbackIngest] Unknown verdict {verdict!r} on {target}; ignoring verdict")
        verdict = None
    rating = meta.get("rating")
    try:
        rating = int(rating) if rating not in (None, "") else None
        if rating is not None:
            rating = max(1, min(5, rating))
    except (TypeError, ValueError):
        rating = None
    tags = meta.get("tags") or []
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]
    tags = [str(t).strip() for t in tags if str(t).strip()]
    reviewer = str(meta.get("reviewer") or "human").strip() or "human"
    return {"target": target, "verdict": verdict, "rating": rating,
            "tags": tags, "note": note, "reviewer": reviewer}


def _content_hash(fb: dict) -> str:
    blob = json.dumps({
        "verdict": fb["verdict"], "rating": fb["rating"],
        "tags": sorted(fb["tags"]), "note": fb["note"],
    }, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _apply_note_node(fb: dict, node_id: int) -> None:
    """Human note -> first-class kb_node linked to the target (syncs FTS)."""
    if not fb["note"] and not fb["tags"]:
        return
    slug = f"note-fb-{fb['target']}-{fb['reviewer']}".lower()[:120]
    summary = (fb["note"].splitlines()[0] if fb["note"] else "")[:200]
    note_node = store.upsert_node(
        node_type="note", slug=slug,
        title=f"{fb['reviewer']}'s review of {fb['target']}",
        domain="human_feedback",
        summary=summary,
        tags=fb["tags"],
        content=fb["note"],
    )
    if node_id:
        store.add_edge(note_node, node_id, "mentions", weight=1.0, origin="human")


def ingest_dir(feedback_dir: str = DEFAULT_FEEDBACK_DIR) -> dict:
    if not os.path.isdir(feedback_dir):
        logger.info(f"[FeedbackIngest] No feedback dir at {feedback_dir}; nothing to do")
        return {"applied": 0, "unchanged": 0, "skipped": 0, "rejects": 0}

    applied = unchanged = skipped = rejects = 0
    for fname in sorted(os.listdir(feedback_dir)):
        if not fname.endswith(".md") or fname.lower() in SKIP_FILES:
            continue
        path = os.path.join(feedback_dir, fname)
        parsed = _parse_feedback_file(path)
        if not parsed:
            skipped += 1
            continue
        fb = _normalise(parsed["meta"], parsed["note"])
        if not fb:
            skipped += 1
            continue

        chash = _content_hash(fb)

        # Resolve the target node (and underlying idea, if any).
        with db_session() as conn:
            node = conn.execute(
                "SELECT id, node_type, ref_table, ref_id FROM kb_nodes WHERE slug=?",
                (fb["target"],),
            ).fetchone()
            prior = conn.execute(
                "SELECT content_hash, verdict, applied_at FROM kb_feedback "
                "WHERE target_slug=? AND reviewer=?",
                (fb["target"], fb["reviewer"]),
            ).fetchone()

        if node is None:
            logger.warning(f"[FeedbackIngest] {fname}: target {fb['target']!r} not found in kb_nodes; skipping")
            skipped += 1
            continue

        node_id = node["id"]
        idea_id = node["ref_id"] if node["ref_table"] == "alpha_ideas" else None
        # ref_id is not guaranteed to point at a live alpha_ideas row (legacy /
        # synthetic KB nodes exist). Verify before applying idea-scoped effects,
        # else the gate_decisions FK fails. Missing row -> note-only feedback.
        if idea_id is not None:
            with db_session() as conn:
                if conn.execute("SELECT 1 FROM alpha_ideas WHERE id=?", (idea_id,)).fetchone() is None:
                    logger.info(f"[FeedbackIngest] {fname}: idea {idea_id} no longer exists; note-only")
                    idea_id = None

        if prior and prior["content_hash"] == chash and prior["applied_at"]:
            unchanged += 1
            continue

        prior_verdict = prior["verdict"] if prior else None

        # 1) Note node (retriever grounding) — reflects the latest text/tags.
        _apply_note_node(fb, node_id)

        # 2) Verdict effects.
        if fb["verdict"] and idea_id is not None:
            with db_session() as conn:
                conn.execute(
                    "INSERT INTO gate_decisions (idea_id, gate, decision, decided_by, rationale) "
                    "VALUES (?, 'human_review', ?, ?, ?)",
                    (idea_id, fb["verdict"], fb["reviewer"], (fb["note"] or "")[:500]),
                )
            if fb["verdict"] == "reject":
                # Only feed RejectionMemory on the first transition into reject,
                # so re-edits of the note don't inflate the pattern counts.
                if prior_verdict != "reject":
                    RejectionMemory().record_rejection(
                        idea_id,
                        reason=fb["note"] or "human review: rejected",
                        stage="human_review",
                    )
                    rejects += 1
                with db_session() as conn:
                    conn.execute(
                        "UPDATE alpha_ideas SET status='rejected', updated_at=datetime('now') WHERE id=?",
                        (idea_id,),
                    )
        elif fb["verdict"] and idea_id is None:
            logger.info(f"[FeedbackIngest] {fname}: verdict on non-idea target {fb['target']!r}; "
                        f"logged as note only")

        # 2b) Drive the graph node's review_state from the verdict (Slice 2
        # provenance): promote -> trusted, reject -> deprecated, watch ->
        # human_reviewed. This is the human half of the trust state machine.
        _REVIEW_STATE = {"promote": "trusted", "reject": "deprecated", "watch": "human_reviewed"}
        if fb["verdict"] in _REVIEW_STATE:
            with db_session() as conn:
                conn.execute(
                    "UPDATE kb_nodes SET review_state=?, updated_at=datetime('now') WHERE id=?",
                    (_REVIEW_STATE[fb["verdict"]], node_id),
                )

        # 3) Upsert the authoritative feedback row.
        with db_session() as conn:
            conn.execute("""
                INSERT INTO kb_feedback
                    (target_slug, node_id, idea_id, reviewer, verdict, rating,
                     tags, note, content_hash, source_path, applied_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
                ON CONFLICT(target_slug, reviewer) DO UPDATE SET
                    node_id=excluded.node_id, idea_id=excluded.idea_id,
                    verdict=excluded.verdict, rating=excluded.rating,
                    tags=excluded.tags, note=excluded.note,
                    content_hash=excluded.content_hash, source_path=excluded.source_path,
                    applied_at=excluded.applied_at, updated_at=excluded.updated_at
            """, (
                fb["target"], node_id, idea_id, fb["reviewer"], fb["verdict"],
                fb["rating"], json.dumps(fb["tags"]), fb["note"], chash, path,
            ))
        applied += 1
        logger.info(f"[FeedbackIngest] Applied {fname} "
                    f"(target={fb['target']} verdict={fb['verdict']} rating={fb['rating']})")

    result = {"applied": applied, "unchanged": unchanged,
              "skipped": skipped, "rejects": rejects}
    logger.info(f"[FeedbackIngest] {result}")
    return result


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    d = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FEEDBACK_DIR
    print(ingest_dir(d))


if __name__ == "__main__":
    main()
