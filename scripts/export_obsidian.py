#!/usr/bin/env python3
"""One-way Obsidian vault export of the knowledge graph.

Writes one Markdown file per kb_node into vault/ with YAML frontmatter and
typed [[wikilinks]], so the graph opens natively in Obsidian (graph view,
backlinks, search).

WIPE-AND-REWRITE for the GENERATED type-folders (ideas/, techniques/, …): the
DB is the source of truth for those, so never hand-edit them. The one exception
is vault/feedback/ — that zone is human-authored, git-tracked, NEVER wiped, and
read back into the loop by scripts/ingest_obsidian_feedback.py. That is the
feedback surface: edit there, not in the generated notes.

Usage: python scripts/export_obsidian.py [output_dir]
"""
import json
import logging
import os
import shutil
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from data.database import db_session

logger = logging.getLogger(__name__)

DEFAULT_VAULT = os.path.join(_REPO_ROOT, "vault")

TYPE_FOLDER = {
    "note": "notes",
    "concept": "concepts",
    "technique": "techniques",
    "idea": "ideas",
    "rejection_pattern": "rejections",
    # evidence / truth-graph node types (2026-07-12)
    "strategy": "strategies",
    "signature": "signatures",
    "backtest_run": "backtests",
    "gate_decision": "gate_decisions",
    "risk": "risks",
    "finding": "findings",
    "leaf": "leaves",
    "agent": "agents",
}

# Human-editable zone — preserved across exports and ingested back into the DB.
FEEDBACK_FOLDER = "feedback"

FEEDBACK_TEMPLATE = """---
target:            # slug of the note this is about, e.g. idea-987654320  (see the generated note's filename)
verdict:           # promote | reject | watch   (leave blank for a note-only review)
rating:            # 1-5, optional
tags: []           # e.g. [lookahead, overfit, promising]
reviewer: mark
---

<!-- Write your analysis below. It is indexed into the KB search (kb_fts) and
     surfaces as grounding context in future idea generation. A `reject` verdict
     also feeds RejectionMemory so the loop stops regenerating this class. -->
"""

FEEDBACK_README = """# Feedback zone — your writeback into the research loop

This is the ONLY folder in the vault you should hand-edit. Everything else is
regenerated (wipe-and-rewrite) from the database every day.

## How to give feedback
1. Duplicate `_TEMPLATE.md` (Obsidian: right-click - Make a copy).
2. Name it after the target, e.g. `idea-987654320.md` (optional but tidy).
3. Fill the frontmatter `target:` with the slug of the note you're reviewing —
   it's the generated note's filename without `.md`.
4. Set a `verdict`, `rating`, and/or `tags`, and write your reasoning in the body.
5. Commit + push (git transport). The VPS ingests it on pull.

## What each field does
- **verdict: reject**  -> RejectionMemory + gate_decisions; the loop stops
  regenerating this class of idea. Your body text becomes the rejection reason.
- **verdict: promote / watch** -> logged as a human gate_decision (audit trail).
- **body note** -> indexed into kb_fts as a human note linked to the target, so
  the retriever surfaces your reasoning when grounding future generation.
- **rating / tags** -> stored on the feedback record and folded into the note's
  tags, feeding idea-generation prioritization.

Re-ingesting an unchanged file is a no-op; editing it supersedes the prior take.
Never delete a file to "undo" — the DB keeps the last ingested state.
"""


def _safe_slug(slug: str) -> str:
    """Slugs may contain '/' (e.g. crypto pairs like BTC/USDT), which would break
    both file paths and Obsidian [[wikilinks]]. Map any path/link-hostile char to
    '-' — applied identically to filenames and link targets so links still resolve.
    """
    out = str(slug or "")
    for ch in ("/", "\\", ":", "|", "#", "^", "[", "]"):
        out = out.replace(ch, "-")
    return out.strip("-") or "unnamed"


def _yaml_escape(value: str) -> str:
    return '"' + str(value or "").replace("\\", "\\\\").replace('"', '\\"') + '"'


def _frontmatter(node: dict) -> str:
    tags = []
    try:
        raw = node["tags"]
        tags = json.loads(raw) if raw and raw.startswith("[") else ([raw] if raw else [])
    except Exception:
        tags = []
    lines = [
        "---",
        f"id: {node['id']}",
        f"type: {node['node_type']}",
        f"domain: {_yaml_escape(node['domain'])}",
        f"tags: [{', '.join(_yaml_escape(t) for t in tags if t)}]",
        f"created: {_yaml_escape(node['created_at'])}",
    ]
    # Provenance fields (present on evidence-graph nodes); keep Dataview-friendly.
    if _has(node, "review_state"):
        lines.append(f"review_state: {_yaml_escape(node['review_state'])}")
    if _has(node, "confidence"):
        lines.append(f"confidence: {node['confidence']}")
    lines.append("---")
    return "\n".join(lines)


def _has(node: dict, key: str) -> bool:
    try:
        return node[key] is not None
    except (KeyError, IndexError):
        return False


def export_vault(out_dir: str = DEFAULT_VAULT) -> dict:
    with db_session() as conn:
        nodes = conn.execute("SELECT * FROM kb_nodes").fetchall()
        edges = conn.execute("""
            SELECT e.source_id, e.target_id, e.relation, e.weight,
                   n2.slug AS target_slug
            FROM kb_edges e JOIN kb_nodes n2 ON n2.id = e.target_id
        """).fetchall()
        # note bodies come from the underlying document content
        doc_content = {r["id"]: r["content"] for r in conn.execute(
            "SELECT id, content FROM kb_documents").fetchall()}

    out_edges: dict[int, list] = {}
    for e in edges:
        out_edges.setdefault(e["source_id"], []).append(e)

    # Wipe ONLY the generated type-folders — never the human feedback zone.
    for folder in set(TYPE_FOLDER.values()):
        gen_dir = os.path.join(out_dir, folder)
        if os.path.isdir(gen_dir):
            shutil.rmtree(gen_dir)
        os.makedirs(gen_dir, exist_ok=True)

    # Seed / refresh the feedback zone without touching the user's own files.
    fb_dir = os.path.join(out_dir, FEEDBACK_FOLDER)
    os.makedirs(fb_dir, exist_ok=True)
    with open(os.path.join(fb_dir, "_TEMPLATE.md"), "w", encoding="utf-8") as fh:
        fh.write(FEEDBACK_TEMPLATE)
    with open(os.path.join(fb_dir, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(FEEDBACK_README)

    written = 0
    for n in nodes:
        node = dict(n)
        parts = [_frontmatter(node), f"\n# {node['title'] or node['slug']}\n"]
        if node["summary"]:
            parts.append(node["summary"] + "\n")
        if node["ref_table"] == "kb_documents" and node["ref_id"] in doc_content:
            content = (doc_content[node["ref_id"]] or "")[:2000]
            if content:
                parts.append(f"\n## Excerpt\n\n{content}\n")

        links = out_edges.get(node["id"], [])
        if links:
            parts.append("\n## Links\n")
            by_relation: dict[str, list] = {}
            for e in links:
                by_relation.setdefault(e["relation"], []).append(e)
            for relation in sorted(by_relation):
                parts.append(f"\n### {relation}\n")
                for e in by_relation[relation]:
                    parts.append(f"- [[{_safe_slug(e['target_slug'])}]] (weight {e['weight']:.2f})")
            parts.append("")

        folder = TYPE_FOLDER.get(node["node_type"], "notes")
        path = os.path.join(out_dir, folder, f"{_safe_slug(node['slug'])}.md")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(parts))
        written += 1

    logger.info(f"[VaultExport] Wrote {written} notes to {out_dir}")
    return {"notes": written, "edges": len(edges), "path": out_dir}


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VAULT
    print(export_vault(out))


if __name__ == "__main__":
    main()
