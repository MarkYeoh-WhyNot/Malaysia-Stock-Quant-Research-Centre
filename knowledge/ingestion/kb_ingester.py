import asyncio
import json
import logging
import re
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse
import aiohttp
import requests as _requests
from agents.base_agent import BaseAgent
from config import settings
from config.settings import MODEL_FAST, MODEL_MAIN, BRAVE_SEARCH_API_KEY
from data.database import db_session

logger = logging.getLogger(__name__)

SYSTEM = (
    "You are a quantitative finance knowledge engineer. Extract structured information "
    f"from research documents to build a searchable knowledge base for {settings.MARKET_NAME} research."
)

# Single unified taxonomy — matches DiversityEngine's 8 research angles exactly.
# The domain field in kb_documents now uses these values so check_balance() can
# query by domain directly instead of using keyword/source_url heuristics.
VALID_DOMAINS = {
    "price_action",          # Technical analysis, price momentum, chart patterns
    "fundamental",           # Value investing, earnings quality, fundamental factors
    "event_driven",          # Post-earnings drift, dividend capture, corporate events
    "institutional",         # EPF flows, GLC ownership, institutional trading patterns
    "macro",                 # OPR cycle, MYR macro impacts on sector returns
    "commodity",             # CPO price impact on plantation stocks
    "sector_rotation",       # sector rotation, defensive vs cyclical
    "behavioural",           # Investor behaviour biases, market anomalies
    "statistical_modelling", # GARCH/ARIMA, factor models, ML, cointegration, HMM, Monte Carlo
}

# Map legacy / internal domain names → unified angle names
DOMAIN_TO_ANGLE = {
    "price_action":          "price_action",
    "fundamental":           "fundamental",
    "event_driven":          "event_driven",
    "institutional":         "institutional",
    "macro":                 "macro",
    "commodity":             "commodity",
    "sector_rotation":       "sector_rotation",
    "behavioural":           "behavioural",
    "statistical_modelling": "statistical_modelling",
    # Legacy domain names (pre-unification)
    "research":              "price_action",
    "analysis-methods":      "statistical_modelling",
    "quant-philosophy":      "statistical_modelling",
    "mental-models":         "behavioural",
    "factor-data":           "statistical_modelling",
    "infrastructure":        "price_action",
    "portfolio-management":  "fundamental",
    "risk-management":       "macro",
    "alpha-ideas":           "event_driven",
    "market-structure":      "price_action",
    "technical":             "price_action",
    "fx":                    "price_action",
    "risk":                  "macro",
    "execution":             "price_action",
    "other":                 None,   # triggers auto-classification
}

# Ordered list used for domain classification prompts
INFER_DOMAINS = [
    "price_action",          # Technical analysis, price momentum, moving averages, RSI, MACD
    "fundamental",           # Value investing, earnings quality, P/E, ROE, dividend yield
    "event_driven",          # Post-earnings drift, dividend capture, corporate events
    "institutional",         # EPF/GLC ownership, pension fund flows, index rebalancing
    "macro",                 # OPR cycle, BNM policy, macroeconomic sector impacts
    "commodity",             # CPO/plantation stocks, aluminium/Press Metal, energy sector
    "sector_rotation",       # Sector momentum, cyclical/defensive rotation, industry trends
    "behavioural",           # Investor sentiment, anomalies, behavioural biases
    "statistical_modelling", # GARCH/ARIMA, HMM regime detection, factor models, PCA, ML, cointegration, Kalman, Monte Carlo, Bayesian
]


BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"


def _normalise_domain(domain: str) -> str:
    """Map any domain string to a valid unified angle name. Returns 'price_action' as default."""
    if domain in VALID_DOMAINS:
        return domain
    mapped = DOMAIN_TO_ANGLE.get(domain)
    if mapped is not None:
        return mapped
    return "price_action"   # safe default for unknown domains


class BraveSearchFetcher:
    """Fetches web search results from the Brave Search API for KB ingestion."""

    def __init__(self, api_key: str = ""):
        self.api_key = api_key or BRAVE_SEARCH_API_KEY

    def search_and_extract(self, query: str, num_results: int = 3) -> list:
        """Search Brave and return list of {title, url, description, content}.

        Each item's 'content' combines the description and extra_snippets fields
        so it can be passed directly to KBIngester.ingest_text().
        Returns [] if the API key is missing or the request fails.
        """
        if not self.api_key:
            logger.warning("BraveSearchFetcher: BRAVE_SEARCH_API_KEY not set")
            return []
        try:
            resp = _requests.get(
                BRAVE_SEARCH_URL,
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "X-Subscription-Token": self.api_key,
                },
                params={
                    "q": query,
                    "count": num_results,
                    "text_decorations": "false",
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning(f"BraveSearchFetcher: request failed for '{query}': {e}")
            return []

        results = []
        for item in data.get("web", {}).get("results", [])[:num_results]:
            title    = item.get("title", "").strip()
            url      = item.get("url", "")
            desc     = item.get("description", "").strip()
            snippets = item.get("extra_snippets", [])

            content_parts = []
            if desc:
                content_parts.append(desc)
            content_parts.extend(s for s in snippets if s)
            content = "\n\n".join(content_parts)

            if title and content:
                results.append({"title": title, "url": url, "description": desc, "content": content})

        return results


class KBIngester(BaseAgent):
    name = "KBIngester"
    description = "Knowledge base ingestion: documents, concept extraction, and graph linking"
    default_model = MODEL_MAIN

    # ------------------------------------------------------------------
    # Text fetching
    # ------------------------------------------------------------------

    async def _fetch_url(self, url: str, timeout: int = 30) -> str:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
        }
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                resp.raise_for_status()
                content_type = resp.headers.get("Content-Type", "")
                if "text" in content_type or "json" in content_type:
                    return await resp.text()
                return ""

    @staticmethod
    def _strip_html(html: str) -> str:
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"&[a-zA-Z]+;", " ", text)
        text = re.sub(r"\s{3,}", "\n\n", text)
        return text.strip()

    def _brave_fallback(self, url: str, title: str = "", domain: str = "other",
                        original_error: str = "") -> dict:
        """Fallback: use Brave Search when direct URL fetch fails.

        Builds a query from the supplied title or the URL path segments,
        ingests the top result, and returns a standard ingest dict with
        'brave_fallback': True and 'original_url' set.
        Returns {'error': ..., 'brave_fallback': True} if Brave also fails.
        """
        if title:
            query = title
        else:
            parsed    = urlparse(url)
            path_parts = [p for p in parsed.path.split("/") if p and len(p) > 3]
            query      = f"{parsed.netloc} {' '.join(path_parts[:3])}".strip()

        self.log_daemon("INFO", f"KB: Brave fallback for '{query}' (original error: {original_error})")
        fetcher = BraveSearchFetcher()
        results = fetcher.search_and_extract(query, num_results=1)

        if not results:
            return {"error": f"URL fetch failed ({original_error}) and Brave fallback found no results",
                    "url": url, "brave_fallback": True}

        r      = results[0]
        result = self.ingest_text(r["content"], r["title"], domain, source_url=r["url"])
        result["brave_fallback"] = True
        result["original_url"]   = url
        return result

    @staticmethod
    def _slug(title: str) -> str:
        s = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
        ts = datetime.utcnow().strftime("%Y-%m-%d")
        return f"{ts}-{s[:60]}"

    # ------------------------------------------------------------------
    # Summarisation & concept extraction
    # ------------------------------------------------------------------

    def _summarise(self, content: str, title: str, domain: str) -> dict:
        truncated = content[:6000]
        ticker_example = settings.TICKER_EXAMPLE.split(" (")[0]
        prompt = f"""Analyse this research document for a {settings.MARKET_NAME} quantitative
research knowledge base.

Title: {title}
Domain: {domain}

Content (truncated):
{truncated}

Return JSON:
{{
  "summary": "3-5 sentence summary focused on relevance to {settings.MARKET_NAME} alpha research",
  "tags": ["tag1", "tag2"],
  "key_concepts": [
    {{"name": "...", "description": "...", "domain": "{domain}"}}
  ],
  "trading_relevance": 0.0,
  "strategy_types": ["momentum|mean_reversion|value|quality|event_driven|macro|technical|fundamental"],
  "applicable_tickers": ["{ticker_example}"],
  "time_horizon": "intraday|swing|position|multi-year",
  "data_requirements": ["..."]
}}"""
        return self.call_claude_json(
            SYSTEM, [{"role": "user", "content": prompt}],
            model=MODEL_MAIN, max_tokens=2048, task_label="kb_summarise"
        )

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def _upsert_document(self, slug: str, title: str, domain: str,
                         content: str, summary: str, source_url: str,
                         tags: list, status: str = "indexed") -> int:
        # Content-hash dedup: the slug is date-prefixed, so the same article
        # re-ingested on a different day would otherwise create a duplicate.
        from knowledge.graph import store as graph_store
        chash = graph_store.content_hash(title, content[:50000])
        with db_session() as conn:
            dup = conn.execute(
                "SELECT id, slug FROM kb_documents "
                "WHERE content_hash=? AND slug != ? LIMIT 1",
                (chash, slug),
            ).fetchone()
        if dup:
            logger.info(f"Duplicate content detected — reusing doc {dup['slug']} instead of {slug}")
            with db_session() as conn:
                conn.execute(
                    "UPDATE kb_documents SET updated_at=datetime('now') WHERE id=?",
                    (dup["id"],),
                )
            return dup["id"]

        with db_session() as conn:
            conn.execute("""
                INSERT INTO kb_documents (slug, title, domain, content, summary, source_url, tags, status, content_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(slug) DO UPDATE SET
                    summary=excluded.summary, tags=excluded.tags,
                    updated_at=datetime('now'), status=excluded.status,
                    content_hash=excluded.content_hash
            """, (slug, title, domain, content[:50000], summary, source_url,
                  json.dumps(tags), status, chash))
            row = conn.execute("SELECT id FROM kb_documents WHERE slug=?", (slug,)).fetchone()

        # Every ingested doc immediately becomes a graph note node
        try:
            graph_store.upsert_node(
                "note", slug=slug, title=title, domain=domain,
                summary=summary, tags=tags, content=content,
                ref=("kb_documents", row["id"]),
            )
        except Exception as e:
            logger.warning(f"Graph node creation failed for {slug}: {e}")
        return row["id"]

    def _upsert_concept(self, name: str, description: str, domain: str) -> int:
        with db_session() as conn:
            conn.execute("""
                INSERT INTO kb_concepts (name, description, domain, count)
                VALUES (?, ?, ?, 1)
                ON CONFLICT(name) DO UPDATE SET
                    count=count+1, description=excluded.description
            """, (name.lower().strip(), description, domain))
            row = conn.execute("SELECT id FROM kb_concepts WHERE name=?", (name.lower().strip(),)).fetchone()
        return row["id"]

    def _link_document_concept(self, doc_id: int, concept_name: str):
        """Create doc-to-doc links via concept phrase matching in summaries/titles.

        Searches for the full concept phrase in other documents' summaries and titles.
        For short concepts (≤ 3 words) this works well. For long phrases it may
        find no matches — that is intentional; tag-based linking (see _link_by_tags)
        handles broader cross-document connectivity.
        """
        try:
            from knowledge.graph import store as graph_store
            term = concept_name.lower().strip()
            with db_session() as conn:
                related = conn.execute(
                    "SELECT id FROM kb_documents "
                    "WHERE id != ? AND ("
                    "  tags LIKE ? OR LOWER(title) LIKE ? OR LOWER(summary) LIKE ?"
                    ") LIMIT 10",
                    (doc_id, f"%{term}%", f"%{term}%", f"%{term}%"),
                ).fetchall()
            src_node = graph_store.ensure_node_for_document(doc_id)
            for rel in related:
                tgt_node = graph_store.ensure_node_for_document(rel["id"])
                if src_node and tgt_node:
                    graph_store.add_edge(src_node, tgt_node, "shared_concept",
                                         weight=0.4, origin="heuristic")
        except Exception as e:
            logger.warning(f"Concept linking failed for doc {doc_id}: {e}")

    def _link_by_tags(self, doc_id: int, tags: list):
        """Create doc-to-doc links for every shared tag between this doc and existing docs.

        Tags are short normalized strings (e.g. ["momentum", "EPF"] for Bursa or
        ["momentum", "on-chain"] for crypto).
        For each tag, finds other documents that contain that same tag anywhere in their
        tags JSON, title, or summary, then creates a 'shared_tag' kb_links row.
        Called once after a document is fully saved with its tags.
        """
        if not tags:
            return
        try:
            from knowledge.graph import store as graph_store
            src_node = graph_store.ensure_node_for_document(doc_id)
            if not src_node:
                return
            for tag in tags:
                term = str(tag).lower().strip()
                if len(term) < 3:
                    continue  # skip trivial terms
                with db_session() as conn:
                    related = conn.execute(
                        "SELECT id FROM kb_documents "
                        "WHERE id != ? AND ("
                        "  LOWER(tags) LIKE ? OR LOWER(title) LIKE ? OR LOWER(summary) LIKE ?"
                        ") LIMIT 8",
                        (doc_id, f"%{term}%", f"%{term}%", f"%{term}%"),
                    ).fetchall()
                for rel in related:
                    tgt_node = graph_store.ensure_node_for_document(rel["id"])
                    if tgt_node:
                        graph_store.add_edge(src_node, tgt_node, "shared_tag",
                                             weight=0.4, origin="heuristic")
        except Exception as e:
            logger.warning(f"Tag linking failed for doc {doc_id}: {e}")

    # ------------------------------------------------------------------
    # Relevance check (Layer 1 quality gate)
    # ------------------------------------------------------------------

    @staticmethod
    def _score_to_category(score: float) -> str:
        """Map a relevance score to its 5-tier category label."""
        if score < 0.20:  return "irrelevant"
        if score < 0.40:  return "generic"
        if score < 0.60:  return "partial"
        if score < 0.80:  return "relevant"
        return "direct"

    def relevance_check(self, title: str, content_preview: str) -> dict:
        """Call Claude Haiku to rate this market's relevance (0.0–1.0).

        Returns {'relevance': float, 'category': str, 'reason': str}.
        Category is one of: irrelevant / generic / partial / relevant / direct.
        Defaults to {'relevance': 1.0, 'category': 'relevant', 'reason': 'check_failed'}
        on any error so that ingest is never blocked by a transient API issue.

        The target market and 5-tier scale text come from the active market
        profile (settings.RELEVANCE_TARGET / settings.RELEVANCE_SCALE) — this
        is the same scale ResearchHunter._is_relevant uses, kept in one place
        per market instead of two hardcoded near-duplicates.
        """
        prompt = (
            f"Rate this content's relevance to {settings.RELEVANCE_TARGET}.\n\n"
            f"Title: {title}\n"
            f"Content preview: {content_preview[:500]}\n\n"
            f"Use this 5-tier scale:\n\n"
            f"{settings.RELEVANCE_SCALE}\n\n"
            f"Return JSON only:\n"
            f'{{"relevance": 0.0, "category": "irrelevant|generic|partial|relevant|direct", "reason": "one sentence"}}'
        )
        try:
            result = self.call_claude_json(
                f"You are a relevance classifier for a {settings.RELEVANCE_TARGET} research system. "
                "Return only the requested JSON — no other text.",
                [{"role": "user", "content": prompt}],
                model=MODEL_FAST,
                max_tokens=100,
                task_label="kb_relevance_check",
            )
            relevance = float(result.get("relevance", 1.0))
            # Accept Claude's category if valid; otherwise derive from score
            raw_cat = str(result.get("category", ""))
            category = raw_cat if raw_cat in {"irrelevant", "generic", "partial", "relevant", "direct"} \
                       else self._score_to_category(relevance)
            reason = str(result.get("reason", ""))
            return {"relevance": relevance, "category": category, "reason": reason}
        except Exception as e:
            self.log_daemon("WARN", f"KB relevance check failed: {e}")
            return {"relevance": 1.0, "category": "relevant", "reason": "check_failed"}

    # ------------------------------------------------------------------
    # Public ingest methods
    # ------------------------------------------------------------------

    def ingest_text(self, content: str, title: str, domain: str = "other",
                    source_url: str = "") -> dict:
        # Track whether we need Claude to classify the domain after summarisation
        needs_classification = (domain == "other" or domain not in VALID_DOMAINS)
        domain = _normalise_domain(domain)  # safe fallback for non-'other' unknowns
        slug = self._slug(title)

        # ── Layer 1: relevance gate ─────────────────────────────────────────────
        # Cheap Haiku call before the expensive Sonnet summarise.
        # 5-tier result: irrelevant / generic / partial / relevant / direct
        rel = self.relevance_check(title, content)
        relevance_score    = rel["relevance"]
        relevance_category = rel["category"]   # one of the 5 tier labels
        relevance_reason   = rel["reason"]

        # Always save the doc regardless of tier (knowledge is never deleted)
        # but tag 'generic' docs so downstream logic can filter them.
        extra_tags_pre = []
        if relevance_category == "generic":
            extra_tags_pre.append("generic")

        meta = self._summarise(content, title, domain)
        if "error" in meta:
            self.log_daemon("WARN", f"Summarisation failed for '{title}': {meta.get('error')}")
            meta = {"summary": "", "tags": [], "key_concepts": []}

        summary  = meta.get("summary", "")
        tags     = extra_tags_pre + meta.get("tags", [])
        concepts = meta.get("key_concepts", [])

        # Use category as the DB status so process_undigested() can filter intelligently
        doc_status = relevance_category  # irrelevant / generic / partial / relevant / direct
        doc_id = self._upsert_document(slug, title, domain, content, summary, source_url, tags, doc_status)

        # ── Auto-classify domain if it was 'other' or unrecognised ─────────────
        # classify_domain() calls Claude Haiku and writes the result back to DB.
        if needs_classification:
            domain = self.classify_domain(doc_id, title, summary)

        concept_ids = []
        for c in concepts:
            if not isinstance(c, dict) or not c.get("name"):
                continue
            cid = self._upsert_concept(c["name"], c.get("description", ""), c.get("domain", domain))
            self._link_document_concept(doc_id, c["name"])
            concept_ids.append(cid)

        self._link_by_tags(doc_id, tags)

        self.log_daemon(
            "INFO",
            f"Ingested doc [{doc_id}] '{title}' ({len(concepts)} concepts, {len(tags)} tags) "
            f"relevance={relevance_score:.2f} category={relevance_category}",
        )

        # ── Seeding logic — tier-aware ──────────────────────────────────────────
        #
        # irrelevant (<0.20): save only — no seeding
        # generic    (0.20–0.40): save only — no seeding (transferable but not market-specific)
        # partial    (0.40–0.60): seed with confidence cap 0.65 (adjacent context, lower weight)
        # relevant   (0.60–0.80): seed normally (this market's RELEVANCE_TARGET)
        # direct     (0.80+):     seed immediately, priority processing
        #
        if relevance_category in ("irrelevant", "generic"):
            self.log_daemon(
                "INFO",
                f"KB: skipped seeding {relevance_category} doc [{doc_id}] "
                f"(score={relevance_score:.2f}: {relevance_reason})",
            )
        else:
            confidence_cap = 0.65 if relevance_category == "partial" else 1.0
            priority       = relevance_category == "direct"
            try:
                from knowledge.ingestion.alpha_seeds import AlphaSeedGenerator
                seed_result = AlphaSeedGenerator().digest(doc_id, confidence_cap=confidence_cap)
                if not seed_result.get("skipped"):
                    self.log_daemon(
                        "INFO",
                        f"AlphaSeed: {seed_result['hypotheses_generated']} hypotheses from "
                        f"'{title[:50]}' (category={relevance_category}"
                        f"{', confidence_cap=0.65' if confidence_cap < 1.0 else ''}"
                        f"{', PRIORITY' if priority else ''})",
                    )
            except Exception as e:
                self.log_daemon("WARN", f"AlphaSeed failed for doc {doc_id}: {e}")

        return {
            "doc_id":              doc_id,
            "slug":                slug,
            "title":               title,
            "domain":              domain,
            "summary":             summary,
            "tags":                tags,
            "concepts_extracted":  len(concepts),
            "trading_relevance":   meta.get("trading_relevance", 0.0),
            "relevance_score":     relevance_score,
            "relevance_category":  relevance_category,
            "relevance_reason":    relevance_reason,
            # keep legacy key for backward compat with Telegram bot old code
            "low_relevance":       relevance_category in ("irrelevant", "generic"),
        }

    async def ingest_url(self, url: str, title: str = "", domain: str = "other") -> dict:
        self.log_daemon("INFO", f"Ingesting URL: {url}")
        try:
            raw = await self._fetch_url(url)
        except aiohttp.ClientResponseError as e:
            log_level = "INFO" if 400 <= e.status < 500 else "WARN"
            self.log_daemon(log_level, f"URL fetch failed (HTTP {e.status}): {url} — trying Brave fallback")
            return self._brave_fallback(url, title, domain, original_error=f"HTTP {e.status}")
        except Exception as e:
            self.log_daemon("WARN", f"URL fetch failed: {url} — {e} — trying Brave fallback")
            return self._brave_fallback(url, title, domain, original_error=str(e))

        content = self._strip_html(raw) if "<html" in raw.lower() else raw
        if not title:
            title_match = re.search(r"<title[^>]*>(.*?)</title>", raw, re.IGNORECASE | re.DOTALL)
            title = title_match.group(1).strip() if title_match else url.split("/")[-1]

        return self.ingest_text(content, title, domain, source_url=url)

    # ------------------------------------------------------------------
    # Domain classification
    # ------------------------------------------------------------------

    def classify_domain(self, doc_id: int, title: str, summary: str) -> str:
        """Call Claude (Haiku) to infer the best unified angle domain for a document.

        Returns one of the 8 unified angle names. Falls back to 'price_action' on error.
        Always updates the DB record with the classified domain.
        """
        # Angle descriptions come from the active market's RESEARCH_ANGLES — the
        # same content DiversityEngine hunts against — instead of a third
        # hardcoded copy of the taxonomy.
        domain_list = "\n".join(
            f"  {d}: {settings.RESEARCH_ANGLES.get(d, {}).get('description', d)}"
            for d in INFER_DOMAINS
        )
        prompt = (
            f"Classify this {settings.MARKET_NAME} research document into exactly one research angle.\n\n"
            f"Title: {title}\n"
            f"Summary: {summary[:600]}\n\n"
            f"Available angles:\n{domain_list}\n\n"
            f'Return JSON only: {{"domain": "<angle-name>", "confidence": 0.0, '
            f'"reason": "one sentence"}}'
        )
        try:
            result = self.call_claude_json(
                f"You are a knowledge base classifier for a {settings.MARKET_NAME} research system. "
                "Return only the requested JSON — no other text.",
                [{"role": "user", "content": prompt}],
                model=MODEL_FAST,
                max_tokens=120,
                task_label="kb_classify_domain",
            )
            raw_domain = result.get("domain", "price_action")
            domain = _normalise_domain(raw_domain)
            with db_session() as conn:
                conn.execute(
                    "UPDATE kb_documents SET domain=?, updated_at=datetime('now') WHERE id=?",
                    (domain, doc_id),
                )
            self.log_daemon("INFO", f"KB domain classified [{doc_id}] → '{domain}' "
                                    f"(confidence={result.get('confidence', 0):.2f})")
            return domain
        except Exception as e:
            self.log_daemon("WARN", f"Domain classification failed for doc {doc_id}: {e}")
            return "price_action"

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, domain: str = None, limit: int = 10) -> list:
        """GraphRAG retrieval (FTS/BM25 + optional embeddings + graph walk).
        Keeps the legacy return shape (list of dicts with title/summary/
        domain/slug) so Telegram /search and other callers are unchanged."""
        try:
            from knowledge.search.retriever import retrieve
            results = retrieve(query, k=limit, hops=2, domain=domain)
            return [{
                # keep legacy id semantics: kb_documents.id for note nodes
                "id": r["ref_id"] if r["ref_table"] == "kb_documents" else r["node_id"],
                "slug": r["slug"], "title": r["title"],
                "domain": r["domain"], "summary": r["summary"],
                "tags": "", "created_at": "",
                "score": r["score"], "node_type": r["node_type"],
                "via": r["via"], "contradicts": r["contradicts"],
            } for r in results]
        except Exception as e:
            logger.warning(f"GraphRAG retrieve failed, falling back to LIKE: {e}")
        # Legacy LIKE fallback (pre-GraphRAG behaviour)
        terms = [t.strip() for t in query.lower().split() if len(t.strip()) > 2]
        if not terms:
            return []
        like_clauses = " AND ".join([f"(LOWER(title) LIKE ? OR LOWER(summary) LIKE ? OR LOWER(tags) LIKE ?)" for _ in terms])
        params = []
        for t in terms:
            like = f"%{t}%"
            params += [like, like, like]
        sql = f"SELECT id, slug, title, domain, summary, tags, created_at FROM kb_documents WHERE {like_clauses}"
        if domain:
            sql += " AND domain=?"
            params.append(domain)
        sql += f" ORDER BY updated_at DESC LIMIT {limit}"
        with db_session() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def search_concepts(self, query: str, limit: int = 20) -> list:
        like = f"%{query.lower()}%"
        with db_session() as conn:
            rows = conn.execute("""
                SELECT id, name, description, domain, count
                FROM kb_concepts
                WHERE LOWER(name) LIKE ? OR LOWER(description) LIKE ?
                ORDER BY count DESC
                LIMIT ?
            """, (like, like, limit)).fetchall()
        return [dict(r) for r in rows]

    def kb_stats(self) -> dict:
        with db_session() as conn:
            docs = conn.execute("SELECT COUNT(*) as n FROM kb_documents").fetchone()["n"]
            concepts = conn.execute("SELECT COUNT(*) as n FROM kb_concepts").fetchone()["n"]
            links = conn.execute("SELECT COUNT(*) as n FROM kb_links").fetchone()["n"]
            by_domain = conn.execute(
                "SELECT domain, COUNT(*) as n FROM kb_documents GROUP BY domain"
            ).fetchall()
        return {
            "total_documents": docs,
            "total_concepts": concepts,
            "total_links": links,
            "by_domain": {r["domain"]: r["n"] for r in by_domain},
        }

    # ------------------------------------------------------------------
    # run()
    # ------------------------------------------------------------------

    def run(self, task: dict) -> dict:
        action = task.get("action", "ingest_text")

        if action == "ingest_text":
            return self.ingest_text(
                task.get("content", ""),
                task.get("title", "Untitled"),
                task.get("domain", "other"),
                task.get("source_url", ""),
            )
        elif action == "ingest_url":
            return asyncio.run(self.ingest_url(
                task["url"],
                task.get("title", ""),
                task.get("domain", "other"),
            ))
        elif action == "search":
            return {"results": self.search(
                task["query"],
                task.get("domain"),
                int(task.get("limit", 10)),
            )}
        elif action == "search_concepts":
            return {"results": self.search_concepts(task["query"], int(task.get("limit", 20)))}
        elif action == "stats":
            return self.kb_stats()
        return {"error": f"Unknown action: {action}"}
