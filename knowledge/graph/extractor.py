"""LLM typed-edge extraction for the knowledge graph.

Turns each note's text into typed, weighted relations against EXISTING nodes.
Hallucination guard: Claude may only choose targets from a provided candidate
list (FTS-similar nodes + all concept/technique slugs); unknown slugs are
dropped. Haiku, batched, incremental (content_hash/extracted_at), and
budget-capped through BaseAgent.
"""
import json
import logging

from agents.base_agent import BaseAgent
from config.settings import MODEL_FAST
from data.database import db_session
from knowledge.graph import store
from knowledge.search import fts

logger = logging.getLogger(__name__)

SYSTEM = """You are a knowledge-graph curator for a Bursa Malaysia quantitative
equity research system. You link research notes to existing graph nodes with
typed relations. Only use target slugs from the provided candidate list.
Output only valid JSON."""

RELATION_GUIDE = """Relations (choose the most specific that applies):
- supports: this note provides evidence FOR the target's claim/technique
- contradicts: this note provides evidence AGAINST the target
- refines: this note adds nuance/parameters/conditions to the target
- derived_from: this note's idea originates from the target
- about_ticker: the note is specifically about the target stock/concept
- uses_technique: the note applies the target technique
- mentions: weaker topical connection (use sparingly)"""


class GraphExtractor(BaseAgent):
    name = "GraphExtractor"
    description = "Extracts typed knowledge-graph edges from KB notes"
    default_model = MODEL_FAST

    def extract_pending(self, batch: int = 8, max_notes: int = 40) -> dict:
        """Process notes whose edges are missing or stale. Returns counts."""
        with db_session() as conn:
            pending = conn.execute("""
                SELECT id, slug, title, domain, summary
                FROM kb_nodes
                WHERE node_type IN ('note', 'idea', 'rejection_pattern')
                  AND (extracted_at IS NULL OR extracted_at < updated_at)
                  AND summary IS NOT NULL AND summary != ''
                ORDER BY updated_at DESC
                LIMIT ?
            """, (max_notes,)).fetchall()

        if not pending:
            return {"processed": 0, "edges_added": 0, "concepts_created": 0}

        processed = edges_added = concepts_created = 0
        for i in range(0, len(pending), batch):
            chunk = [dict(r) for r in pending[i:i + batch]]
            try:
                result = self._extract_batch(chunk)
                processed += result["processed"]
                edges_added += result["edges_added"]
                concepts_created += result["concepts_created"]
            except RuntimeError as e:
                # budget cap — stop cleanly, remaining notes stay pending
                logger.warning(f"[GraphExtractor] Stopping (budget): {e}")
                break
            except Exception as e:
                logger.error(f"[GraphExtractor] Batch failed: {e}")

        self.log_daemon(
            "INFO",
            f"Graph extraction: {processed} notes, +{edges_added} edges, "
            f"+{concepts_created} concepts"
        )
        return {"processed": processed, "edges_added": edges_added,
                "concepts_created": concepts_created}

    # ------------------------------------------------------------------

    def _candidates_for(self, note: dict, k: int = 15) -> list[dict]:
        """FTS-similar nodes this note could plausibly link to."""
        query = f"{note['title']} {note['summary'][:200]}"
        hits = fts.fts_search(query, k=k + 1)
        ids = [nid for nid, _ in hits if nid != note["id"]][:k]
        if not ids:
            return []
        marks = ",".join("?" * len(ids))
        with db_session() as conn:
            rows = conn.execute(
                f"SELECT id, slug, title, node_type FROM kb_nodes WHERE id IN ({marks})",
                ids).fetchall()
        return [dict(r) for r in rows]

    def _shared_candidates(self) -> list[dict]:
        """All concept + technique nodes (small, shared across the batch)."""
        with db_session() as conn:
            rows = conn.execute(
                "SELECT id, slug, title, node_type FROM kb_nodes "
                "WHERE node_type IN ('concept', 'technique') LIMIT 300"
            ).fetchall()
        return [dict(r) for r in rows]

    def _extract_batch(self, notes: list[dict]) -> dict:
        shared = self._shared_candidates()
        note_blocks, candidate_slugs = [], {}
        for n in notes:
            cands = self._candidates_for(n)
            slugs = {c["slug"]: c["id"] for c in cands}
            slugs.update({c["slug"]: c["id"] for c in shared})
            candidate_slugs[n["slug"]] = slugs
            cand_lines = "\n".join(
                f"  - {c['slug']} ({c['node_type']}: {c['title'][:60]})"
                for c in (cands + shared)[:60]
            )
            note_blocks.append(
                f"NOTE slug={n['slug']} [{n['domain']}] {n['title']}\n"
                f"Summary: {n['summary'][:600]}\n"
                f"Candidate targets:\n{cand_lines}"
            )

        prompt = f"""{RELATION_GUIDE}

For each note below, identify its typed relations to candidate targets, plus
any genuinely NEW concepts (entities like "EPF rebalancing", "T+3 settlement")
not already in the candidates.

{chr(10).join(note_blocks)}

Return JSON:
{{
  "notes": [
    {{
      "slug": "<note slug>",
      "relations": [
        {{"target_slug": "<candidate slug>", "relation": "supports|contradicts|refines|derived_from|about_ticker|uses_technique|mentions", "weight": 0.7, "reason": "one line"}}
      ],
      "new_concepts": [
        {{"name": "...", "description": "one line", "domain": "one of the 9 angles"}}
      ]
    }}
  ]
}}
Rules: max 6 relations per note, weight 0.0-1.0 by confidence, only use
candidate slugs, new_concepts only for genuinely missing entities (max 3)."""

        result = self.call_claude_json(
            SYSTEM, [{"role": "user", "content": prompt}],
            model=MODEL_FAST, max_tokens=3000, task_label="graph_extract",
        )
        if "error" in result:
            logger.error(f"[GraphExtractor] Parse failure — skipping batch")
            return {"processed": 0, "edges_added": 0, "concepts_created": 0}

        note_by_slug = {n["slug"]: n for n in notes}
        processed = edges_added = concepts_created = 0
        for entry in result.get("notes", []):
            note = note_by_slug.get(entry.get("slug"))
            if not note:
                continue
            slugs = candidate_slugs.get(note["slug"], {})

            for c in (entry.get("new_concepts") or [])[:3]:
                name = (c.get("name") or "").strip()
                if not name:
                    continue
                cid = store.upsert_node(
                    "concept", slug=self._concept_slug(name), title=name,
                    domain=c.get("domain", ""), summary=c.get("description", ""),
                )
                if store.add_edge(note["id"], cid, "mentions",
                                  weight=0.6, origin="llm"):
                    edges_added += 1
                    concepts_created += 1

            for rel in (entry.get("relations") or [])[:6]:
                target_id = slugs.get(rel.get("target_slug"))
                if not target_id:
                    continue  # hallucinated or out-of-list target — drop
                if store.add_edge(note["id"], target_id,
                                  rel.get("relation", "mentions"),
                                  weight=float(rel.get("weight", 0.5)),
                                  origin="llm"):
                    edges_added += 1

            with db_session() as conn:
                conn.execute(
                    "UPDATE kb_nodes SET extracted_at=datetime('now') WHERE id=?",
                    (note["id"],))
            processed += 1

        return {"processed": processed, "edges_added": edges_added,
                "concepts_created": concepts_created}

    @staticmethod
    def _concept_slug(name: str) -> str:
        import re
        return "concept-" + re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:60]

    def run(self, task: dict) -> dict:
        return self.extract_pending(
            batch=int(task.get("batch", 8)),
            max_notes=int(task.get("max_notes", 40)),
        )
