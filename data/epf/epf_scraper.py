"""
EPFScraper — EPF (Employees Provident Fund) substantial shareholder tracker.

EPF is Malaysia's mandatory retirement fund (~MYR 1 trillion AUM). As a
price-insensitive mandate-driven buyer, its accumulation patterns create
predictable price support on Bursa Malaysia stocks.

Sources:
  1. Brave Search API (primary): recent EPF substantial shareholder announcements
  2. i3investor EPF page (supplementary): structured shareholding change table

Substantial shareholder disclosures are triggered whenever EPF's stake crosses
5% or changes by ±1% above the 5% threshold — Bursa Malaysia regulations.
"""
import logging
import re
from datetime import datetime
from typing import Optional, List

import requests
from bs4 import BeautifulSoup

from config.settings import BRAVE_SEARCH_API_KEY, KLCI_STOCKS
from data.database import db_session

logger = logging.getLogger(__name__)

BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"

# ── Company name → .KL ticker lookup ─────────────────────────────────────────
_TICKER_TO_NAME: dict[str, str] = {s["symbol"]: s["name"] for s in KLCI_STOCKS}
_NAME_TO_TICKER: dict[str, str] = {}

for _s in KLCI_STOCKS:
    _NAME_TO_TICKER[_s["name"].lower()] = _s["symbol"]

# Extended aliases for how EPF announcements refer to companies
_EPF_ALIASES: dict[str, str] = {
    # Banking
    "maybank": "1155.KL", "malayan banking": "1155.KL", "mbb": "1155.KL",
    "public bank": "1295.KL", "pbbank": "1295.KL",
    "cimb": "1023.KL", "cimb group": "1023.KL",
    "rhb": "1066.KL", "rhb bank": "1066.KL", "rhb banking": "1066.KL",
    "hong leong bank": "5819.KL", "hlb": "5819.KL",
    "ambank": "1015.KL", "ammb": "1015.KL", "ambank group": "1015.KL",
    "hong leong financial": "1082.KL", "hlcap": "1082.KL",
    # Utilities / Energy
    "tenaga": "5347.KL", "tenaga nasional": "5347.KL", "tnb": "5347.KL",
    "petronas gas": "6033.KL", "petgas": "6033.KL",
    "ytl": "4677.KL", "ytl corporation": "4677.KL", "ytl corp": "4677.KL",
    # Telecoms
    "celcomdigi": "6947.KL", "celcom digi": "6947.KL", "cdb": "6947.KL",
    "maxis": "6012.KL",
    "telekom": "4863.KL", "telekom malaysia": "4863.KL", "tm": "4863.KL",
    # Healthcare
    "ihh": "5225.KL", "ihh healthcare": "5225.KL",
    "hartalega": "5168.KL", "harta": "5168.KL",
    # Materials / Industrial
    "press metal": "8869.KL", "press metal aluminium": "8869.KL", "pmetal": "8869.KL",
    "sime darby": "4197.KL", "sime": "4197.KL",
    "gamuda": "5398.KL",
    # Chemicals
    "petronas chemicals": "5183.KL", "pchem": "5183.KL", "petchem": "5183.KL",
    # Plantations
    "ioi": "1961.KL", "ioi corporation": "1961.KL", "ioi corp": "1961.KL",
    "sime darby plantation": "5285.KL", "sdplant": "5285.KL",
    "klk": "2445.KL", "kuala lumpur kepong": "2445.KL",
    # Consumer
    "genting": "3182.KL",
    "genting malaysia": "4715.KL", "genm": "4715.KL",
    "nestle": "4707.KL", "nestle malaysia": "4707.KL",
    "ppb": "4065.KL", "ppb group": "4065.KL",
    "ql": "5296.KL", "ql resources": "5296.KL",
    # Transport / Infrastructure
    "misc": "3816.KL", "misc berhad": "3816.KL",
    "dialog": "7277.KL", "dialog group": "7277.KL",
}
_NAME_TO_TICKER.update(_EPF_ALIASES)

# Browser headers for web scraping
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,ms;q=0.8",
}


def _resolve_ticker(text: str) -> Optional[str]:
    """Map a company name or abbreviation to a .KL Yahoo Finance ticker."""
    t = text.lower().strip()
    # Exact match
    if t in _NAME_TO_TICKER:
        return _NAME_TO_TICKER[t]
    # Substring match (longest name wins to avoid false positives)
    best_len, best_tk = 0, None
    for name, ticker in _NAME_TO_TICKER.items():
        if len(name) >= 4 and name in t and len(name) > best_len:
            best_len, best_tk = len(name), ticker
    return best_tk


class EPFScraper:
    """Fetch and analyse EPF substantial shareholder disclosures on Bursa Malaysia."""

    # ── DB helpers ────────────────────────────────────────────────────────────

    def _ensure_table(self):
        """Create epf_holdings table if absent (idempotent)."""
        with db_session() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS epf_holdings (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker      TEXT NOT NULL,
                    company     TEXT,
                    date        TEXT NOT NULL,
                    epf_pct     REAL NOT NULL,
                    prev_pct    REAL,
                    change_pct  REAL,
                    direction   TEXT DEFAULT 'stable',
                    source      TEXT,
                    created_at  TEXT DEFAULT (datetime('now')),
                    UNIQUE(ticker, date)
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_epf_ticker ON epf_holdings(ticker)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_epf_date ON epf_holdings(date)"
            )

    def _upsert_holding(self, record: dict):
        with db_session() as conn:
            conn.execute("""
                INSERT INTO epf_holdings
                  (ticker, company, date, epf_pct, prev_pct, change_pct, direction, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker, date) DO UPDATE SET
                  epf_pct=excluded.epf_pct,
                  prev_pct=excluded.prev_pct,
                  change_pct=excluded.change_pct,
                  direction=excluded.direction,
                  source=excluded.source
            """, (
                record["ticker"],
                record.get("company", _TICKER_TO_NAME.get(record["ticker"], "")),
                record["date"],
                record["epf_pct"],
                record.get("prev_epf_pct"),
                record.get("change_pct"),
                record.get("direction", "stable"),
                record.get("source", "unknown"),
            ))

    # ── Source 1: Brave Search ────────────────────────────────────────────────

    def _brave_search_epf(self) -> List[dict]:
        """Search Brave API for recent EPF substantial shareholder announcements."""
        if not BRAVE_SEARCH_API_KEY:
            logger.warning("BRAVE_SEARCH_API_KEY not set — skipping Brave EPF search")
            return []

        queries = [
            "EPF substantial shareholder Bursa Malaysia 2025 ownership percent",
            "\"Employees Provident Fund\" substantial shareholder Bursa 2025",
            "EPF KWSP increases stake Bursa Malaysia stock 2025",
        ]

        all_results: List[dict] = []
        seen_urls: set[str]    = set()

        for query in queries:
            try:
                resp = requests.get(
                    BRAVE_SEARCH_URL,
                    headers={
                        "X-Subscription-Token": BRAVE_SEARCH_API_KEY,
                        "Accept": "application/json",
                    },
                    params={"q": query, "count": 10, "search_lang": "en"},
                    timeout=15,
                )
                resp.raise_for_status()
                data = resp.json()
                for r in data.get("web", {}).get("results", []):
                    url = r.get("url", "")
                    if url in seen_urls:
                        continue
                    seen_urls.add(url)
                    all_results.append({
                        "title":       r.get("title", ""),
                        "url":         url,
                        "description": r.get("description", ""),
                        "extra":       " ".join(r.get("extra_snippets", [])),
                    })
            except Exception as e:
                logger.debug(f"Brave EPF search failed for query '{query[:40]}': {e}")

        logger.info(f"Brave EPF search: {len(all_results)} results across {len(queries)} queries")
        return all_results

    # ── Source 2: i3investor EPF page ─────────────────────────────────────────

    def _scrape_i3investor_epf(self) -> List[dict]:
        """Scrape i3investor's EPF shareholding change tracker page."""
        url = "https://klse.i3investor.com/web/stkpick/epf.jsp"
        raw: List[dict] = []
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=25)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            # i3investor renders a table: Date | Company | No. of Shares | % Held | +/-
            for table in soup.find_all("table"):
                rows = table.find_all("tr")
                if len(rows) < 2:
                    continue
                for row in rows[1:]:
                    cells = [c.get_text(" ", strip=True) for c in row.find_all(["td", "th"])]
                    if len(cells) < 3:
                        continue
                    combined = " | ".join(cells)
                    # Only include rows that mention a company name we can resolve
                    raw.append({
                        "title":       f"EPF i3investor: {cells[0]} {cells[1] if len(cells) > 1 else ''}",
                        "url":         url,
                        "description": combined,
                        "extra":       "",
                    })

            # Fallback: scan all text blobs for EPF percentage mentions
            if not raw:
                for el in soup.find_all(["p", "div", "li"]):
                    text = el.get_text(" ", strip=True)
                    if re.search(r'\bepf\b|\bkwsp\b', text, re.IGNORECASE) and "%" in text:
                        raw.append({
                            "title":       text[:80],
                            "url":         url,
                            "description": text,
                            "extra":       "",
                        })

            logger.info(f"i3investor EPF page: {len(raw)} rows scraped")
        except Exception as e:
            logger.debug(f"i3investor EPF scrape failed: {e}")
        return raw

    # ── Parser ────────────────────────────────────────────────────────────────

    def parse_epf_holdings(self, raw_data: List[dict]) -> List[dict]:
        """Parse raw search/scrape results into structured EPF holding records.

        For each item, extracts: ticker, company, date, epf_pct,
        prev_epf_pct, change_pct, direction.
        Deduplicates on (ticker, date).
        """
        holdings: List[dict] = []
        seen: set[tuple]     = set()

        for item in raw_data:
            full_text = " ".join(filter(None, [
                item.get("title", ""),
                item.get("description", ""),
                item.get("extra", ""),
            ]))

            # Must mention EPF / KWSP
            if not re.search(r'\bepf\b|\bkwsp\b|kumpulan wang simpanan', full_text, re.IGNORECASE):
                continue

            # ── Extract percentage(s) ─────────────────────────────────────────
            pct_matches = re.findall(
                r'(\d{1,2}(?:\.\d{1,4})?)\s*(?:%|per\s*cent)',
                full_text, re.IGNORECASE,
            )
            # Realistic EPF ownership range: 5%–30%
            pcts = [float(p) for p in pct_matches if 5.0 <= float(p) <= 30.0]
            if not pcts:
                continue
            epf_pct = pcts[0]

            # ── Extract date ──────────────────────────────────────────────────
            date_str = datetime.utcnow().strftime("%Y-%m-%d")
            date_pat = (
                r'(\d{4}-\d{2}-\d{2}|'
                r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|'
                r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4})'
            )
            dm = re.search(date_pat, full_text, re.IGNORECASE)
            if dm:
                raw_d = dm.group(1)
                for fmt in (
                    "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%d-%m-%y", "%d/%m/%y",
                    "%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b. %d, %Y",
                ):
                    try:
                        date_str = datetime.strptime(raw_d, fmt).strftime("%Y-%m-%d")
                        break
                    except ValueError:
                        continue

            # ── Resolve company → ticker ──────────────────────────────────────
            ticker  = None
            company = ""

            # Pattern 1: "EPF ... in CompanyName"
            for pat in (
                r'(?:in|stake in|shares in|of)\s+([A-Z][A-Za-z\s\-&]{2,35}?)(?:\s+Bh?d\.?|Berhad|Holdings|Group|Corp\.?)?(?:\s+\(|,|\.|$)',
                r'([A-Z][A-Za-z\s\-&]{2,30}?)(?:\s+Bh?d\.?|Berhad)?\s+(?:shares|stake|holding|equity)',
            ):
                m = re.search(pat, full_text)
                if m:
                    cname = m.group(1).strip()
                    resolved = _resolve_ticker(cname)
                    if resolved:
                        ticker  = resolved
                        company = cname
                        break

            # Pattern 2: Scan full text for any known alias (longest match wins)
            if not ticker:
                ticker = _resolve_ticker(full_text)
                if ticker:
                    company = _TICKER_TO_NAME.get(ticker, "")

            if not ticker:
                continue

            # Deduplicate
            key = (ticker, date_str)
            if key in seen:
                continue
            seen.add(key)

            # ── Direction from keyword signals ────────────────────────────────
            direction = "stable"
            if re.search(r'\b(?:acquir|increas|bought|purchas|add|accumul|rais)\w*\b',
                         full_text, re.IGNORECASE):
                direction = "accumulating"
            elif re.search(r'\b(?:dispos|reduc|sell|sold|decreas|trim|lower)\w*\b',
                           full_text, re.IGNORECASE):
                direction = "distributing"

            # ── Infer previous % and change ───────────────────────────────────
            prev_pct   = None
            change_pct = None
            if len(pcts) >= 2:
                if direction == "accumulating":
                    prev_pct, epf_pct = min(pcts[:2]), max(pcts[:2])
                elif direction == "distributing":
                    prev_pct, epf_pct = max(pcts[:2]), min(pcts[:2])
                else:
                    prev_pct = pcts[1]
                change_pct = round(epf_pct - prev_pct, 4)
                # Refine direction based on computed delta
                if change_pct > 0.05:
                    direction = "accumulating"
                elif change_pct < -0.05:
                    direction = "distributing"

            record = {
                "ticker":      ticker,
                "company":     company or _TICKER_TO_NAME.get(ticker, ticker),
                "date":        date_str,
                "epf_pct":     round(epf_pct, 4),
                "prev_epf_pct": round(prev_pct, 4) if prev_pct is not None else None,
                "change_pct":  round(change_pct, 4) if change_pct is not None else None,
                "direction":   direction,
                "source":      item.get("url", "unknown"),
            }
            holdings.append(record)
            logger.debug(
                f"EPF parsed: {ticker} {epf_pct:.2f}% ({direction}) on {date_str}"
            )

        logger.info(f"parse_epf_holdings: {len(holdings)} records from {len(raw_data)} raw items")
        return holdings

    # ── Accumulation signal ───────────────────────────────────────────────────

    def compute_accumulation_signal(self, ticker: str) -> dict:
        """Compute EPF accumulation/distribution signal from last 4 DB records.

        Counts how many of the last 4 periods show increasing EPF ownership.

        Returns:
            {ticker, quarters_increasing, total_change_4q,
             signal_strength, current_pct, trend}
        """
        base = {
            "ticker":              ticker,
            "quarters_increasing": 0,
            "total_change_4q":     0.0,
            "signal_strength":     "weak",
            "current_pct":         0.0,
            "trend":               "stable",
        }

        with db_session() as conn:
            rows = conn.execute("""
                SELECT epf_pct, prev_pct, change_pct, direction, date
                FROM epf_holdings
                WHERE ticker = ?
                ORDER BY date DESC
                LIMIT 4
            """, (ticker,)).fetchall()

        if not rows:
            return {**base, "error": f"No EPF data for {ticker} in DB"}

        records = [dict(r) for r in rows]
        current_pct = records[0]["epf_pct"]

        quarters_increasing = 0
        total_change        = 0.0
        for r in records:
            chg = r.get("change_pct") or 0.0
            total_change += chg
            if chg > 0.05 or r.get("direction") == "accumulating":
                quarters_increasing += 1

        # Signal strength thresholds
        if quarters_increasing >= 3:
            strength = "strong"
            trend    = "accumulating"
        elif quarters_increasing >= 2:
            strength = "moderate"
            trend    = "accumulating"
        elif quarters_increasing == 0 and total_change < -0.10:
            strength = "moderate"
            trend    = "distributing"
        else:
            strength = "weak"
            if total_change > 0.05:
                trend = "accumulating"
            elif total_change < -0.05:
                trend = "distributing"
            else:
                trend = "stable"

        return {
            "ticker":              ticker,
            "quarters_increasing": quarters_increasing,
            "total_change_4q":     round(total_change, 4),
            "signal_strength":     strength,
            "current_pct":         current_pct,
            "trend":               trend,
        }

    # ── Public entry point ────────────────────────────────────────────────────

    def fetch_epf_disclosures(self) -> List[dict]:
        """Fetch, parse, and store latest EPF shareholding disclosures.

        Strategy:
          1. Brave Search API — recent news/announcements
          2. i3investor EPF page — structured table (always run as supplement)

        Upserts all parsed records into epf_holdings table.

        Returns:
            List of parsed holding dicts (may be empty if all sources fail).
        """
        self._ensure_table()

        raw: List[dict] = []

        # Primary: Brave Search
        raw.extend(self._brave_search_epf())

        # Supplement: i3investor (always run to catch different coverage)
        raw.extend(self._scrape_i3investor_epf())

        if not raw:
            logger.warning("EPF: Brave Search and i3investor both returned no data")
            return []

        holdings = self.parse_epf_holdings(raw)

        for record in holdings:
            try:
                self._upsert_holding(record)
            except Exception as e:
                logger.debug(f"EPF upsert failed [{record.get('ticker')}]: {e}")

        logger.info(
            f"fetch_epf_disclosures: {len(raw)} raw items → "
            f"{len(holdings)} holdings stored"
        )
        return holdings
