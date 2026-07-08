import json
import logging
import re
from datetime import datetime

from agents.base_agent import BaseAgent
from config import settings
from config.settings import MODEL_MAIN
from data.database import db_session

logger = logging.getLogger(__name__)

# Persona (from active market profile — was hardcoded "Bursa Malaysia equity
# markets" regardless of MARKET_MODE, verified live 2026-07-09).
SYSTEM = settings.ALPHA_SEED_SYSTEM


def is_market_feasible(h: dict) -> tuple[bool, str]:
    """Return (feasible: bool, reason: str) for a hypothesis dict.

    Layer 2 quality gate — rejects strategies that cannot be executed in the
    active market: short-selling/pairs/options-style phrases (reuses
    settings.BLOCKED_MODES — the same list Gate 0 and the sandbox path already
    enforce, instead of a second hardcoded phrase list to keep in sync), and
    tickers that don't match this market's format (settings.TICKER_REGEX).
    """
    text      = " ".join([
        str(h.get("title", "")),
        str(h.get("hypothesis", "")),
        str(h.get("factor_formula", "")),
    ]).lower()
    ticker    = str(h.get("ticker", "")).strip()

    # Check for infeasible phrases in combined text
    for phrase in settings.BLOCKED_MODES:
        if phrase in text:
            return False, f"contains infeasible phrase: '{phrase}'"

    # Check ticker field: a single bare token (no spaces) that doesn't match
    # this market's ticker format is a wrong-market symbol (e.g. "AAPL" in
    # Bursa mode, "1155.KL" in crypto mode); multi-word "sector name" strings
    # and "KLCI"/index labels pass through untouched.
    if ticker and not ticker.lower().startswith("sector") and ticker.lower() != "klci":
        if " " not in ticker and not settings.TICKER_REGEX.fullmatch(ticker):
            return False, (
                f"ticker '{ticker}' does not match this market's format "
                f"({settings.TICKER_EXAMPLE})"
            )

    return True, "ok"


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

    def digest(self, doc_id: int, confidence_cap: float = 1.0) -> dict:
        """Extract alpha hypotheses from a single KB document and seed ideas.

        Args:
            doc_id:         KB document to process.
            confidence_cap: Maximum confidence allowed for seeded hypotheses (0.0–1.0).
                            Pass 0.65 for 'partial' relevance docs (ASEAN/EM context)
                            so generated ideas enter the pipeline with lower weight.
                            Default 1.0 = no cap (full confidence).

        Returns {"skipped": True} if the document has already been processed
        (seeded=1) or does not exist.
        """
        with db_session() as conn:
            row = conn.execute(
                "SELECT id, slug, title, summary, content, seeded "
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
        ticker_example = settings.TICKER_EXAMPLE.split(" (")[0]
        data_sources_json = json.dumps(settings.DATA_SOURCES_EXAMPLE)
        prompt = f"""Read this article and extract actionable alpha.

Title: {title}
Summary: {summary}
Content: {content[:3000]}

Do exactly THREE things:

A) CORE INSIGHT (1 sentence):
   The single most actionable trading insight from this content.

B) MARKET MECHANISM (2 sentences):
   WHY does this edge exist specifically in {settings.MARKET_NAME}?
   What investor behaviour or market structure causes it?

C) TESTABLE HYPOTHESES (generate 2-4):
   Convert insights into specific, testable alpha ideas.
   Each must have a specific {ticker_example}-style ticker OR sector,
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
      "ticker": "{ticker_example} or sector name",
      "timeframe": "1d or 1wk",
      "factor_formula": "Signal description",
      "data_sources": {data_sources_json},
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
            error_msg = result.get("error", "unknown")
            self.log_daemon(
                "WARN",
                f"AlphaSeed digest failed for doc {doc_id}: {error_msg}",
            )
            # On hard parse failures (json_parse_failed, invalid JSON, etc.) mark
            # the document seeded=1 so the daemon does not retry it every cycle.
            # The summary is updated with a failure note for auditability.
            if "parse" in str(error_msg).lower() or "json" in str(error_msg).lower():
                with db_session() as conn:
                    conn.execute(
                        "UPDATE kb_documents SET seeded=1, "
                        "summary=?, updated_at=datetime('now') WHERE id=?",
                        (f"[PARSE_FAILED] {(summary or '')[:200]}", doc_id),
                    )
                self.log_daemon(
                    "WARN",
                    f"AlphaSeed: doc {doc_id} marked seeded=1 after parse failure "
                    f"— will not retry",
                )
            return {
                "skipped": True, "reason": "claude_error",
                "doc_id": doc_id, "error": error_msg,
            }

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
            hypothesis = h.get("hypothesis", "")
            ticker     = h.get("ticker", "")
            formula    = h.get("factor_formula", "")
            confidence = min(float(h.get("confidence") or 0.0), confidence_cap)

            # Layer 3: minimum quality threshold
            if confidence < 0.5:
                self.log_daemon(
                    "INFO",
                    f"AlphaSeed: skipped low-confidence hypothesis '{h_title[:50]}' "
                    f"(confidence={confidence:.2f})",
                )
                continue
            if not ticker or not formula or len(hypothesis) < 50:
                self.log_daemon(
                    "INFO",
                    f"AlphaSeed: skipped incomplete hypothesis '{h_title[:50]}' "
                    f"(missing ticker/formula or hypothesis too short)",
                )
                continue

            # Layer 2: market feasibility filter
            feasible, reason = is_market_feasible(h)
            if not feasible:
                self.log_daemon(
                    "INFO",
                    f"AlphaSeed: rejected non-feasible hypothesis '{h_title[:50]}' — {reason}",
                )
                continue

            slug_body  = self._slugify(h_title)[:60]
            slug       = f"seed-{today}-{slug_body}"
            timeframe  = h.get("timeframe", "1d")
            sources    = json.dumps(h.get("data_sources", []))
            novelty    = float(h.get("novelty_score") or 0.0)
            logic      = float(h.get("logic_score") or 0.0)

            with db_session() as conn:
                conn.execute("""
                    INSERT OR IGNORE INTO alpha_ideas
                        (slug, title, hypothesis, ticker, timeframe, factor_formula,
                         data_sources, stage, status, novelty_score, logic_score,
                         kb_context)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 'gate0', 'pending', ?, ?, ?)
                """, (slug, h_title, hypothesis, ticker, timeframe, formula,
                      sources, novelty, logic,
                      json.dumps([row["slug"]])))  # provenance: KB doc that seeded this

                saved_row = conn.execute(
                    "SELECT id FROM alpha_ideas WHERE slug=?", (slug,)
                ).fetchone()

            if saved_row:
                idea_id = saved_row["id"]
                ideas_saved += 1
                self.log_daemon("INFO", f"AlphaSeed: created idea [{idea_id}] {h_title}")
                # Knowledge graph: idea node + derived_from edge to source doc
                try:
                    from knowledge.graph import store as graph_store
                    idea_node = graph_store.upsert_node(
                        "idea", slug=f"idea-{slug}"[:120], title=h_title,
                        summary=(hypothesis or "")[:2000],
                        tags=[ticker] if ticker else [],
                        ref=("alpha_ideas", idea_id),
                    )
                    doc_node = graph_store.ensure_node_for_document(doc_id)
                    if doc_node:
                        graph_store.add_edge(idea_node, doc_node, "derived_from",
                                             weight=0.9, origin="heuristic")
                except Exception as _ge:
                    self.log_daemon("WARN", f"AlphaSeed: graph edge failed: {_ge}")

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
        """Digest all KB documents where seeded=0, up to *limit* docs.

        Skips 'irrelevant' and 'generic' documents (no seeding for those tiers).
        Applies confidence_cap=0.65 for 'partial' relevance documents.
        """
        with db_session() as conn:
            docs = conn.execute(
                "SELECT id, status FROM kb_documents "
                "WHERE seeded=0 AND status NOT IN ('irrelevant', 'generic') "
                "ORDER BY id LIMIT ?",
                (limit,),
            ).fetchall()

        processed        = 0
        total_ideas      = 0
        skipped          = 0

        for doc in docs:
            cap = 0.65 if doc["status"] == "partial" else 1.0
            result = self.digest(doc["id"], confidence_cap=cap)
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
