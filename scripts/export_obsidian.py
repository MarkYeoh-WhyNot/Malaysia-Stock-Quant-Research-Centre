#!/usr/bin/env python3
"""One-way Obsidian vault export of the knowledge graph.

Writes one Markdown file per kb_node into vault/ (gitignored) with YAML
frontmatter and typed [[wikilinks]], so the graph opens natively in Obsidian
(graph view, backlinks, search). WIPE-AND-REWRITE: never point this at a
vault you edit by hand — the DB is the source of truth.

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
}


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
        "---",
    ]
    return "\n".join(lines)


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

    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    for folder in TYPE_FOLDER.values():
        os.makedirs(os.path.join(out_dir, folder), exist_ok=True)

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
                    parts.append(f"- [[{e['target_slug']}]] (weight {e['weight']:.2f})")
            parts.append("")

        folder = TYPE_FOLDER.get(node["node_type"], "notes")
        path = os.path.join(out_dir, folder, f"{node['slug']}.md")
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
