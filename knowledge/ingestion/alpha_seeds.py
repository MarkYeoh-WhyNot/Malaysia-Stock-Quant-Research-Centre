import json
import logging
import re
from datetime import datetime

from agents.base_agent import BaseAgent
from config.settings import MODEL_MAIN
from data.database import db_session

logger = logging.getLogger(__name__)

SYSTEM = (
    "You are a senior quant researcher specialising in "
    "Bursa Malaysia equity markets."
)


class AlphaSeedGenerator(BaseAgent):
    name = "AlphaSeedGenerator"
    description = "Extract actionable alpha hypotheses from KB documents and seed the pipeline"
    default_model = MODEL_MAIN

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _slugify(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")

    # ------------------------------------------------------------------
    # digest(doc_id) → dict
    # ------------------------------------------------------------------

    def digest(self, doc_id: int) -> dict:
        """Extract alpha hypotheses from a single KB document and seed ideas.

        Returns {"skipped": True} if the document has already been processed
        (seeded=1) or does not exist.
        """
        with db_session() as conn:
            row = conn.execute(
                "SELECT id, title, summary, content, seeded "
                "FROM kb_documents WHERE id=?",
                (doc_id,),
            ).fetchone()

        if not row:
            return {"skipped": True, "reason": "not_found"}
        if row["seeded"]:
            return {"skipped": True, "reason": "already_seeded", "doc_id": doc_id}

        title   = row["title"] or f"Document {doc_id}"
        summary = row["summary"] or ""
        content = row["content"] or ""

        # ── Claude call ─────────────────────────────────────────────────
        prompt = f"""Read this article and extract actionable alpha.

Title: {title}
Summary: {summary}
Content: {content[:3000]}

Do exactly THREE things:

A) CORE INSIGHT (1 sentence):
   The single most actionable trading insight from this content.

B) MARKET MECHANISM (2 sentences):
   WHY does this edge exist specifically in Bursa Malaysia?
   What investor behaviour or market structure causes it?

C) TESTABLE HYPOTHESES (generate 2-4):
   Convert insights into specific, testable alpha ideas.
   Each must have a specific .KL ticker OR sector,
   a measurable entry signal, expected return,
   and a failure condition.

Return JSON:
{{
  "core_insight": "...",
  "mechanism": "...",
  "hypotheses": [
    {{
      "title": "Short descriptive name under 80 chars",
      "hypothesis": "Full explanation of the trade logic",
      "ticker": "1155.KL or sector name",
      "timeframe": "1d or 1wk",
      "factor_formula": "Signal description",
      "data_sources": ["Yahoo Finance", "Bursa announcements"],
      "novelty_score": 0.0,
      "logic_score": 0.0,
      "confidence": 0.0
    }}
  ]
}}"""

        result = self.call_claude_json(
            SYSTEM,
            [{"role": "user", "content": prompt}],
            model=self.default_model,
            max_tokens=2048,
            task_label="alpha_seed_digest",
        )

        if "error" in result:
            self.log_daemon("WARN", f"AlphaSeed digest failed for doc {doc_id}: {result.get('error')}")
            return {"skipped": True, "reason": "claude_error", "doc_id": doc_id, "error": result["error"]}

        core_insight = result.get("core_insight", "")
        mechanism    = result.get("mechanism", "")
        hypotheses   = result.get("hypotheses", [])
        if not isinstance(hypotheses, list):
            hypotheses = []

        # ── Save ideas ─────────────────────────────────────────────────
        today      = datetime.utcnow().strftime("%Y-%m-%d")
        ideas_saved = 0

        for h in hypotheses:
            if not isinstance(h, dict) or not h.get("title"):
                continue

            h_title    = str(h["title"])[:80]
            slug_body  = self._slugify(h_title)[:60]
            slug       = f"seed-{today}-{slug_body}"
            hypothesis = h.get("hypothesis", "")
            ticker     = h.get("ticker", "")
            timeframe  = h.get("timeframe", "1d")
            formula    = h.get("factor_formula", "")
            sources    = json.dumps(h.get("data_sources", []))
            novelty    = float(h.get("novelty_score") or 0.0)
            logic      = float(h.get("logic_score") or 0.0)

            with db_session() as conn:
                conn.execute("""
                    INSERT OR IGNORE INTO alpha_ideas
                        (slug, title, hypothesis, pair, timeframe, factor_formula,
                         data_sources, stage, status, novelty_score, logic_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 'gate0', 'pending', ?, ?)
                """, (slug, h_title, hypothesis, ticker, timeframe, formula,
                      sources, novelty, logic))

                saved_row = conn.execute(
                    "SELECT id FROM alpha_ideas WHERE slug=?", (slug,)
                ).fetchone()

            if saved_row:
                idea_id = saved_row["id"]
                ideas_saved += 1
                self.log_daemon("INFO", f"AlphaSeed: created idea [{idea_id}] {h_title}")

        # ── Mark document as seeded; update summary ────────────────────
        new_summary = f"[DIGESTED] {core_insight}" if core_insight else summary
        with db_session() as conn:
            conn.execute(
                "UPDATE kb_documents SET seeded=1, summary=?, updated_at=datetime('now') WHERE id=?",
                (new_summary, doc_id),
            )

        self.log_daemon(
            "INFO",
            f"AlphaSeed: digested doc [{doc_id}] '{title[:50]}' "
            f"→ {len(hypotheses)} hypotheses, {ideas_saved} ideas saved",
        )

        return {
            "doc_id":              doc_id,
            "title":               title,
            "core_insight":        core_insight,
            "mechanism":           mechanism,
            "hypotheses_generated": len(hypotheses),
            "ideas_saved":         ideas_saved,
        }

    # ------------------------------------------------------------------
    # process_undigested(limit) → dict
    # ------------------------------------------------------------------

    def process_undigested(self, limit: int = 10) -> dict:
        """Digest all KB documents where seeded=0, up to *limit* docs."""
        with db_session() as conn:
            docs = conn.execute(
                "SELECT id FROM kb_documents WHERE seeded=0 ORDER BY id LIMIT ?",
                (limit,),
            ).fetchall()

        processed        = 0
        total_ideas      = 0
        skipped          = 0

        for doc in docs:
            result = self.digest(doc["id"])
            if result.get("skipped"):
                skipped += 1
            else:
                processed       += 1
                total_ideas     += result.get("ideas_saved", 0)

        return {
            "processed":           processed,
            "total_ideas_created": total_ideas,
            "skipped":             skipped,
        }

    # ------------------------------------------------------------------
    # run(task) → dict
    # ------------------------------------------------------------------

    def run(self, task: dict) -> dict:
        action = task.get("action", "process")
        if action == "digest":
            return self.digest(task["doc_id"])
        if action == "process":
            return self.process_undigested(task.get("limit", 10))
        return {"error": f"Unknown action: {action}"}
