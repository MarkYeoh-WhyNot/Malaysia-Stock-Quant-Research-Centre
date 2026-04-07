import asyncio
import json
import logging
import re
from datetime import datetime
from typing import Optional
import aiohttp
from agents.base_agent import BaseAgent
from config.settings import MODEL_FAST, MODEL_MAIN
from data.database import db_session

logger = logging.getLogger(__name__)

SYSTEM = """You are a quantitative finance knowledge engineer. Extract structured information
from research documents to build a searchable knowledge base for Bursa Malaysia equity research."""

VALID_DOMAINS = {"fx", "macro", "technical", "fundamental", "risk", "execution", "research", "other"}


class KBIngester(BaseAgent):
    name = "KBIngester"
    description = "Knowledge base ingestion: documents, concept extraction, and graph linking"
    default_model = MODEL_MAIN

    # ------------------------------------------------------------------
    # Text fetching
    # ------------------------------------------------------------------

    async def _fetch_url(self, url: str, timeout: int = 30) -> str:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; OpenClaw/1.0; research-bot)"}
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
        prompt = f"""Analyse this research document for an FX trading knowledge base.

Title: {title}
Domain: {domain}

Content (truncated):
{truncated}

Return JSON:
{{
  "summary": "3-5 sentence summary focused on trading relevance",
  "tags": ["tag1", "tag2"],
  "key_concepts": [
    {{"name": "...", "description": "...", "domain": "{domain}"}}
  ],
  "trading_relevance": 0.0,
  "strategy_types": ["carry|momentum|mean_reversion|macro|technical|fundamental"],
  "applicable_pairs": ["EUR_USD"],
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
                         tags: list) -> int:
        with db_session() as conn:
            conn.execute("""
                INSERT INTO kb_documents (slug, title, domain, content, summary, source_url, tags, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'indexed')
                ON CONFLICT(slug) DO UPDATE SET
                    summary=excluded.summary, tags=excluded.tags,
                    updated_at=datetime('now'), status='indexed'
            """, (slug, title, domain, content[:50000], summary, source_url, json.dumps(tags)))
            row = conn.execute("SELECT id FROM kb_documents WHERE slug=?", (slug,)).fetchone()
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

    def _link_document_concept(self, doc_id: int, concept_id: int):
        # kb_links uses doc-to-doc FKs; we track doc↔concept via a tag-style
        # entry using source_id=doc_id, target_id=doc_id (self-link) won't work
        # either, so we store in a separate lightweight approach: annotate the
        # kb_links row with relation='concept' and target_id=source_id to avoid
        # FK violation.  The concept lookup is already covered by kb_concepts.
        # Silently skip if FK would be violated (concept IDs ≠ document IDs).
        try:
            with db_session() as conn:
                # Only insert if target_id exists as a document
                target_doc = conn.execute(
                    "SELECT id FROM kb_documents WHERE id=?", (concept_id,)
                ).fetchone()
                if not target_doc:
                    return  # concept_id is not a doc ID — skip FK link
                existing = conn.execute(
                    "SELECT id FROM kb_links WHERE source_id=? AND target_id=? AND relation='contains'",
                    (doc_id, concept_id)
                ).fetchone()
                if not existing:
                    conn.execute(
                        "INSERT INTO kb_links (source_id, target_id, relation, weight) VALUES (?, ?, 'contains', 1.0)",
                        (doc_id, concept_id)
                    )
        except Exception:
            pass  # FK link is supplementary — never fail ingest over it

    # ------------------------------------------------------------------
    # Public ingest methods
    # ------------------------------------------------------------------

    def ingest_text(self, content: str, title: str, domain: str = "other",
                    source_url: str = "") -> dict:
        domain = domain if domain in VALID_DOMAINS else "other"
        slug = self._slug(title)

        meta = self._summarise(content, title, domain)
        if "error" in meta:
            self.log_daemon("WARN", f"Summarisation failed for '{title}': {meta.get('error')}")
            meta = {"summary": "", "tags": [], "key_concepts": []}

        summary = meta.get("summary", "")
        tags = meta.get("tags", [])
        concepts = meta.get("key_concepts", [])

        doc_id = self._upsert_document(slug, title, domain, content, summary, source_url, tags)

        concept_ids = []
        for c in concepts:
            if not isinstance(c, dict) or not c.get("name"):
                continue
            cid = self._upsert_concept(c["name"], c.get("description", ""), c.get("domain", domain))
            self._link_document_concept(doc_id, cid)
            concept_ids.append(cid)

        self.log_daemon("INFO", f"Ingested doc [{doc_id}] '{title}' ({len(concepts)} concepts, {len(tags)} tags)")
        return {
            "doc_id": doc_id,
            "slug": slug,
            "title": title,
            "domain": domain,
            "summary": summary,
            "tags": tags,
            "concepts_extracted": len(concepts),
            "trading_relevance": meta.get("trading_relevance", 0.0),
        }

    async def ingest_url(self, url: str, title: str = "", domain: str = "other") -> dict:
        self.log_daemon("INFO", f"Ingesting URL: {url}")
        try:
            raw = await self._fetch_url(url)
        except Exception as e:
            self.log_daemon("WARN", f"URL fetch failed: {url} — {e}")
            return {"error": str(e), "url": url}

        content = self._strip_html(raw) if "<html" in raw.lower() else raw
        if not title:
            title_match = re.search(r"<title[^>]*>(.*?)</title>", raw, re.IGNORECASE | re.DOTALL)
            title = title_match.group(1).strip() if title_match else url.split("/")[-1]

        return self.ingest_text(content, title, domain, source_url=url)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, domain: str = None, limit: int = 10) -> list:
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
