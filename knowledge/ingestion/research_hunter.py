"""
ResearchHunter — fetch academic papers from Semantic Scholar and arXiv,
ingest abstracts into the OpenClaw knowledge base.

Sources:
  - Semantic Scholar Graph API (API key required; falls back to arXiv if absent)
  - arXiv q-fin section (no key required)
"""
import logging
import xml.etree.ElementTree as ET
from typing import Optional

import requests

from agents.base_agent import BaseAgent
from config.settings import MODEL_FAST, SEMANTIC_SCHOLAR_API_KEY
from knowledge.ingestion.kb_ingester import KBIngester

logger = logging.getLogger(__name__)

SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
ARXIV_URL            = "https://export.arxiv.org/api/query"

QUERY_SYSTEM = (
    "You are a research librarian generating academic database search queries for "
    "quantitative equity research focused on Bursa Malaysia and ASEAN emerging markets."
)


class ResearchHunter(BaseAgent):
    name = "ResearchHunter"
    description = "Fetches academic papers from Semantic Scholar and arXiv, ingests into KB"
    default_model = MODEL_FAST

    MAX_PAPERS = 10

    # ── Query generation ──────────────────────────────────────────────────────

    def _generate_queries(self, topic: str, context: str) -> list:
        prompt = f"""Generate 3-5 academic search queries to find relevant literature for:

Topic:   {topic}
Context: {context[:600]}

Each query should target one of these angles:
1. The specific strategy factor (momentum, value, mean-reversion, event-driven, etc.)
2. ASEAN / emerging-market / Malaysian equity markets
3. Quantitative finance or factor investing more broadly

Return a JSON array of short query strings (6-10 words each). Example:
["momentum premium ASEAN emerging market equities",
 "post-earnings drift Malaysia stock returns",
 "value factor Bursa Malaysia"]"""

        result = self.call_claude_json(
            QUERY_SYSTEM,
            [{"role": "user", "content": prompt}],
            model=MODEL_FAST,
            max_tokens=512,
            task_label="generate_search_queries",
        )
        if isinstance(result, list) and result:
            return [str(q) for q in result[:5]]
        # Fallback: build simple queries from topic words
        words = topic.replace("-", " ").split()[:4]
        base  = " ".join(words)
        return [
            f"{base} equity strategy",
            f"{base} ASEAN emerging markets",
            f"{base} quantitative finance",
        ]

    # ── Semantic Scholar ──────────────────────────────────────────────────────

    def _search_semantic_scholar(self, query: str) -> list:
        if not SEMANTIC_SCHOLAR_API_KEY:
            return []
        try:
            headers = {
                "User-Agent": "OpenClaw/1.0 research-bot",
                "x-api-key": SEMANTIC_SCHOLAR_API_KEY,
            }
            resp = requests.get(
                SEMANTIC_SCHOLAR_URL,
                params={"query": query, "fields": "title,abstract,year,openAccessPdf", "limit": 5},
                timeout=15,
                headers=headers,
            )
            resp.raise_for_status()
            data  = resp.json()
            papers = []
            for p in data.get("data", []):
                if not p.get("title"):
                    continue
                papers.append({
                    "title":    p["title"],
                    "abstract": p.get("abstract") or "",
                    "year":     p.get("year"),
                    "source":   "semantic_scholar",
                })
            return papers
        except Exception as e:
            logger.debug(f"Semantic Scholar search failed for '{query}': {e}")
            return []

    # ── arXiv ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_arxiv_xml(xml_text: str) -> list:
        ns     = {"atom": "http://www.w3.org/2005/Atom"}
        papers = []
        try:
            root = ET.fromstring(xml_text)
            for entry in root.findall("atom:entry", ns):
                title_el   = entry.find("atom:title", ns)
                summary_el = entry.find("atom:summary", ns)
                if title_el is None:
                    continue
                title   = " ".join((title_el.text or "").split())
                summary = " ".join((summary_el.text or "").split()) if summary_el is not None else ""
                papers.append({"title": title, "abstract": summary, "year": None, "source": "arxiv"})
        except ET.ParseError as e:
            logger.warning(f"arXiv XML parse error: {e}")
        return papers

    def _search_arxiv(self, query: str) -> list:
        try:
            resp = requests.get(
                ARXIV_URL,
                params={
                    "search_query": f"all:{query}",
                    "start":        0,
                    "max_results":  5,
                    "sortBy":       "relevance",
                },
                timeout=15,
            )
            resp.raise_for_status()
            return self._parse_arxiv_xml(resp.text)
        except Exception as e:
            logger.warning(f"arXiv search failed for '{query}': {e}")
            return []

    # ── Main hunt ─────────────────────────────────────────────────────────────

    def hunt(self, topic: str, context: str, angle_tag: str = "") -> dict:
        """
        Fetch up to MAX_PAPERS relevant papers and ingest them into the KB.

        Args:
            topic:     Strategy title or research topic.
            context:   Hypothesis or additional context for query generation.
            angle_tag: Optional diversity-engine angle name (e.g. 'price_action').
                       When set, source_url is prefixed with 'diversity_hunt:<angle_tag>'.

        Returns:
            {"papers_found": int, "papers_ingested": int, "titles": [...], "queries": [...]}
        """
        queries = self._generate_queries(topic, context)
        sources = "Semantic Scholar + arXiv" if SEMANTIC_SCHOLAR_API_KEY else "arXiv only (no SEMANTIC_SCHOLAR_API_KEY)"
        self.log_daemon("INFO", f"ResearchHunter hunting '{topic}' with {len(queries)} queries [{sources}]")

        seen_titles: set = set()
        all_papers: list = []

        for query in queries:
            if len(all_papers) >= self.MAX_PAPERS:
                break
            for paper in self._search_semantic_scholar(query):
                key = paper["title"].lower().strip()
                if key and key not in seen_titles:
                    seen_titles.add(key)
                    all_papers.append(paper)
                if len(all_papers) >= self.MAX_PAPERS:
                    break

            if len(all_papers) < self.MAX_PAPERS:
                for paper in self._search_arxiv(query):
                    key = paper["title"].lower().strip()
                    if key and key not in seen_titles:
                        seen_titles.add(key)
                        all_papers.append(paper)
                    if len(all_papers) >= self.MAX_PAPERS:
                        break

        # Ingest into KB
        source_prefix = f"diversity_hunt:{angle_tag}" if angle_tag else "research_hunt"
        kb             = KBIngester()
        ingested       = 0
        titles_ingested: list = []

        for paper in all_papers:
            if not paper.get("abstract"):
                continue
            try:
                lines = [f"Title: {paper['title']}"]
                if paper.get("year"):
                    lines.append(f"Year: {paper['year']}")
                lines.append(f"\nAbstract:\n{paper['abstract']}")
                content = "\n".join(lines)

                kb.ingest_text(
                    content=content,
                    title=paper["title"],
                    domain="research",
                    source_url=f"{source_prefix}:{paper['source']}",
                )
                ingested += 1
                titles_ingested.append(paper["title"])
            except Exception as e:
                logger.warning(f"KB ingest failed for '{paper['title']}': {e}")

        self.log_daemon(
            "INFO",
            f"ResearchHunter complete: {len(all_papers)} found, {ingested} ingested "
            f"(topic='{topic[:50]}')",
        )
        return {
            "papers_found":    len(all_papers),
            "papers_ingested": ingested,
            "titles":          titles_ingested,
            "queries":         queries,
        }

    # ── run() ─────────────────────────────────────────────────────────────────

    def run(self, task: dict) -> dict:
        return self.hunt(
            topic=task.get("topic", ""),
            context=task.get("context", ""),
            angle_tag=task.get("angle_tag", ""),
        )
