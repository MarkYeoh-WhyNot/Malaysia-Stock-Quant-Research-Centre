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

# Brave Search — quality domain allowlist for the research hunt
BRAVE_RESEARCH_DOMAINS = (
    ".edu", ".ac.", "ssrn.com", "researchgate.net", "papers.ssrn.com",
)

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

    # ── Relevance pre-filter ──────────────────────────────────────────────────

    @staticmethod
    def _score_to_category(score: float) -> str:
        """Map a relevance score to its 5-tier category label."""
        if score < 0.20:  return "irrelevant"
        if score < 0.40:  return "generic"
        if score < 0.60:  return "partial"
        if score < 0.80:  return "relevant"
        return "direct"

    def _is_relevant(self, title: str, abstract: str) -> dict:
        """Call Claude Haiku to rate Bursa Malaysia equity relevance (0.0-1.0).

        Returns {'relevance': float, 'category': str, 'reason': str}.
        Category is one of: irrelevant / generic / partial / relevant / direct.
        Defaults to {'relevance': 1.0, 'category': 'relevant', 'reason': 'check_failed'}
        on any error so that ingest is never blocked by a transient API issue.

        5-tier scoring:
          0.00–0.20  irrelevant  — wrong market/asset class, crypto, forex, CFD, non-financial
          0.20–0.40  generic     — general finance theory, no EM/Asian context
          0.40–0.60  partial     — ASEAN / emerging-market / Asian equity context
          0.60–0.80  relevant    — Bursa Malaysia or Malaysian equity specific
          0.80–1.00  direct      — actionable KLSE intelligence
        """
        prompt = (
            f"Rate this academic paper's relevance to Bursa Malaysia equity trading.\n\n"
            f"Title: {title}\n"
            f"Abstract: {abstract[:400]}\n\n"
            f"Use this 5-tier scale:\n\n"
            f"  0.00–0.20  irrelevant — completely wrong market or asset class\n"
            f"    Examples: Australian CFD trading, cryptocurrency, forex pairs,\n"
            f"    US options pricing, bond market mechanics, ML for cybersecurity\n\n"
            f"  0.20–0.40  generic — general finance, transferable concepts only\n"
            f"    Examples: General momentum theory, generic valuation frameworks,\n"
            f"    factor investing with no regional context, portfolio theory\n\n"
            f"  0.40–0.60  partial — emerging market or Asian market context\n"
            f"    Examples: ASEAN equity research, Southeast Asia fund flows,\n"
            f"    EM factor models, Asian market microstructure, China/India/HK equity\n\n"
            f"  0.60–0.80  relevant — Bursa Malaysia or Malaysian equity specific\n"
            f"    Examples: KLSE stock returns, Malaysian market anomalies,\n"
            f"    Bursa market microstructure, BNM policy effects, FBM KLCI factors\n\n"
            f"  0.80–1.00  direct — actionable KLSE intelligence\n"
            f"    Examples: Specific KLSE stock analysis, EPF flow studies,\n"
            f"    CPO-plantation correlation, GLC ownership effects, Bursa volatility\n\n"
            f"Return JSON only:\n"
            f'{{"relevance": 0.0, "category": "irrelevant|generic|partial|relevant|direct", "reason": "one sentence"}}'
        )
        try:
            result = self.call_claude_json(
                "You are a relevance classifier for a Bursa Malaysia equity research system. "
                "Return only the requested JSON — no other text.",
                [{"role": "user", "content": prompt}],
                model=MODEL_FAST,
                max_tokens=100,
                task_label="paper_relevance_check",
            )
            relevance = float(result.get("relevance", 1.0))
            raw_cat   = str(result.get("category", ""))
            category  = raw_cat if raw_cat in {"irrelevant", "generic", "partial", "relevant", "direct"} \
                        else self._score_to_category(relevance)
            return {
                "relevance": relevance,
                "category":  category,
                "reason":    str(result.get("reason", "")),
            }
        except Exception as e:
            logger.debug(f"Relevance check failed for '{title}': {e}")
            return {"relevance": 1.0, "category": "relevant", "reason": "check_failed"}

    # ── Main hunt ─────────────────────────────────────────────────────────────

    def hunt(self, topic: str, context: str, angle_tag: str = "", domain: str = "price_action") -> dict:
        """
        Fetch up to MAX_PAPERS relevant papers and ingest them into the KB.

        Args:
            topic:     Strategy title or research topic.
            context:   Hypothesis or additional context for query generation.
            angle_tag: Optional diversity-engine angle name (e.g. 'price_action').
                       When set, source_url is prefixed with 'diversity_hunt:<angle_tag>'.
            domain:    Unified angle domain to store in kb_documents.domain.

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
                # Relevance pre-filter — only hard-skip 'irrelevant' papers (<0.20)
                # to avoid paying for a Sonnet summarise call on clearly wrong content.
                # generic/partial/relevant/direct are all passed to ingest_text() which
                # applies tier-aware seeding logic.
                rel = self._is_relevant(paper["title"], paper["abstract"])
                if rel["category"] == "irrelevant":
                    self.log_daemon(
                        "INFO",
                        f"ResearchHunter: skipped '{paper['title'][:60]}' "
                        f"(irrelevant, score={rel['relevance']:.2f}, reason={rel['reason']})",
                    )
                    continue

                lines = [f"Title: {paper['title']}"]
                if paper.get("year"):
                    lines.append(f"Year: {paper['year']}")
                lines.append(f"\nAbstract:\n{paper['abstract']}")
                content = "\n".join(lines)

                kb.ingest_text(
                    content=content,
                    title=paper["title"],
                    domain=domain,
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

    # ── Brave Search hunt ─────────────────────────────────────────────────────

    def brave_search_hunt(self, topic: str, domain: str = "price_action") -> dict:
        """Search Brave for research articles on a topic, filter to quality domains, ingest top 3.

        Query template: "{topic} trading strategy research Bursa Malaysia"
        Domain filter: .edu, .ac., ssrn.com, researchgate.net, papers.ssrn.com

        Returns {"papers_found": int, "papers_ingested": int, "titles": list}
        """
        from config.settings import BRAVE_SEARCH_API_KEY
        from knowledge.ingestion.kb_ingester import BraveSearchFetcher, BRAVE_SEARCH_URL

        if not BRAVE_SEARCH_API_KEY:
            self.log_daemon("WARN", "ResearchHunter.brave_search_hunt: BRAVE_SEARCH_API_KEY not set — skipping")
            return {"papers_found": 0, "papers_ingested": 0, "titles": []}

        query = f"{topic} trading strategy research Bursa Malaysia"
        self.log_daemon("INFO", f"ResearchHunter.brave_search_hunt: '{query}'")

        try:
            resp = requests.get(
                BRAVE_SEARCH_URL,
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "X-Subscription-Token": BRAVE_SEARCH_API_KEY,
                },
                params={"q": query, "count": 10, "text_decorations": "false"},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            self.log_daemon("WARN", f"ResearchHunter.brave_search_hunt: API error: {e}")
            return {"papers_found": 0, "papers_ingested": 0, "titles": []}

        all_results = data.get("web", {}).get("results", [])
        filtered = [
            r for r in all_results
            if any(d in r.get("url", "") for d in BRAVE_RESEARCH_DOMAINS)
        ]
        self.log_daemon(
            "INFO",
            f"ResearchHunter.brave_search_hunt: {len(all_results)} total, "
            f"{len(filtered)} from quality domains",
        )

        kb             = KBIngester()
        ingested       = 0
        titles_ingested: list = []

        for item in filtered[:3]:
            title    = item.get("title", "").strip()
            url      = item.get("url", "")
            desc     = item.get("description", "").strip()
            snippets = item.get("extra_snippets", [])

            content_parts = [f"Title: {title}"]
            if desc:
                content_parts.append(desc)
            content_parts.extend(s for s in snippets if s)
            content = "\n\n".join(content_parts)

            if not content.strip() or not title:
                continue

            try:
                rel = self._is_relevant(title, desc)
                if rel["category"] == "irrelevant":
                    self.log_daemon(
                        "INFO",
                        f"ResearchHunter.brave_search_hunt: skipped '{title[:60]}' "
                        f"(irrelevant, score={rel['relevance']:.2f})",
                    )
                    continue

                kb.ingest_text(
                    content=content,
                    title=title,
                    domain=domain,
                    source_url=f"brave_hunt:{url}",
                )
                ingested += 1
                titles_ingested.append(title)
            except Exception as e:
                logger.warning(f"ResearchHunter.brave_search_hunt: ingest failed for '{title}': {e}")

        self.log_daemon(
            "INFO",
            f"ResearchHunter.brave_search_hunt complete: {len(filtered)} quality results, "
            f"{ingested} ingested (topic='{topic[:50]}')",
        )
        return {"papers_found": len(filtered), "papers_ingested": ingested, "titles": titles_ingested}

    # ── run() ─────────────────────────────────────────────────────────────────

    def run(self, task: dict) -> dict:
        return self.hunt(
            topic=task.get("topic", ""),
            context=task.get("context", ""),
            angle_tag=task.get("angle_tag", ""),
        )
