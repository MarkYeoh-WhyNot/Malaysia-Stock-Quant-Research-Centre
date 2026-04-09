"""
AnalystCoverageMonitor — detects new analyst research coverage on Bursa Malaysia stocks.

Tracks "initiation" events (first-ever coverage by a new analyst house) which
create systematic re-rating opportunities as institutional fund managers gain
research access and information asymmetry collapses.

Sources:
  1. Brave Search API — "Bursa Malaysia analyst initiate coverage 2025"
  2. i3investor analyst blog — klse.i3investor.com/web/blog/analysis

Tables:
  analyst_coverage_history — every report seen per (ticker, analyst_house, date)
  analyst_alerts           — Gate 0 ideas + Telegram alerts generated
"""
import json
import logging
import re
from datetime import datetime, timedelta
from typing import Optional

import requests
from bs4 import BeautifulSoup

from config.settings import BRAVE_SEARCH_API_KEY, KLCI_STOCKS
from data.database import db_session

logger = logging.getLogger(__name__)

BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"

# ── Analyst house registry ────────────────────────────────────────────────────
# Maps text fragments (and URL slugs) → canonical house name.
# Order doesn't matter — resolution uses longest-match wins.
_ANALYST_HOUSES: dict[str, str] = {
    # Maybank
    "maybank investment banking":   "Maybank IB Research",
    "maybank investment bank":      "Maybank IB Research",
    "maybank kim eng":              "Maybank IB Research",
    "maybank ib research":          "Maybank IB Research",
    "maybank ib":                   "Maybank IB Research",
    "maybank investment":           "Maybank IB Research",
    "maybank research":             "Maybank IB Research",
    "maybankib":                    "Maybank IB Research",
    "maybank":                      "Maybank IB Research",
    # CIMB
    "cimb investment bank":         "CIMB Research",
    "cimb securities":              "CIMB Research",
    "cimb research":                "CIMB Research",
    "cimbresearch":                 "CIMB Research",
    "cimb":                         "CIMB Research",
    # Kenanga
    "kenanga investment bank":      "Kenanga Research",
    "kenanga research":             "Kenanga Research",
    "kenanga ib":                   "Kenanga Research",
    "kenanga":                      "Kenanga Research",
    # RHB
    "rhb investment bank":          "RHB Research",
    "rhb investment":               "RHB Research",
    "rhb research":                 "RHB Research",
    "rhb invest":                   "RHB Research",
    "rhbinvest":                    "RHB Research",
    "rhb":                          "RHB Research",
    # Hong Leong
    "hong leong investment bank":   "Hong Leong Investment Bank",
    "hong leong invest":            "Hong Leong Investment Bank",
    "hong leong ib":                "Hong Leong Investment Bank",
    "hong leong research":          "Hong Leong Investment Bank",
    "hlib research":                "Hong Leong Investment Bank",
    "hlib":                         "Hong Leong Investment Bank",
    "hlinvest":                     "Hong Leong Investment Bank",
    "hong leong":                   "Hong Leong Investment Bank",
    # Affin
    "affin hwang capital":          "Affin Hwang Capital",
    "affin hwang research":         "Affin Hwang Capital",
    "affin hwang":                  "Affin Hwang Capital",
    "affin research":               "Affin Hwang Capital",
    "affin":                        "Affin Hwang Capital",
    # AmInvest / AmBank
    "aminvestment":                 "AmInvest",
    "am investment":                "AmInvest",
    "aminvest":                     "AmInvest",
    "amresearch":                   "AmInvest",
    "am research":                  "AmInvest",
    "am invest":                    "AmInvest",
    "ambank":                       "AmInvest",
    "ammb":                         "AmInvest",
    # PublicInvest
    "publicinvest research":        "PublicInvest Research",
    "public invest research":       "PublicInvest Research",
    "publicinvest":                 "PublicInvest Research",
    "public invest":                "PublicInvest Research",
    "public bank research":         "PublicInvest Research",
    "pbinvest":                     "PublicInvest Research",
    # MIDF
    "midf amanah investment":       "MIDF Research",
    "midf amanah":                  "MIDF Research",
    "midf research":                "MIDF Research",
    "midf":                         "MIDF Research",
    # UOB Kay Hian
    "uob kay hian research":        "UOB Kay Hian",
    "uob kay hian":                 "UOB Kay Hian",
    "uobkh":                        "UOB Kay Hian",
    "uob":                          "UOB Kay Hian",
    # TA Securities
    "ta securities research":       "TA Securities",
    "ta securities":                "TA Securities",
    "ta research":                  "TA Securities",
    "tasec":                        "TA Securities",
    # Alliance
    "alliance bank research":       "Alliance Bank Research",
    "alliance bank":                "Alliance Bank Research",
    "alliance research":            "Alliance Bank Research",
    "alliance":                     "Alliance Bank Research",
    # BIMB
    "bimb securities research":     "BIMB Securities",
    "bimb securities":              "BIMB Securities",
    "bimb":                         "BIMB Securities",
    # Phillip Capital
    "phillip capital research":     "Phillip Capital",
    "phillip capital":              "Phillip Capital",
    "phillip securities":           "Phillip Capital",
    "phillip":                      "Phillip Capital",
    # Inter Pacific
    "inter pacific research":       "Inter Pacific Research",
    "inter-pacific research":       "Inter Pacific Research",
    "interpacific research":        "Inter Pacific Research",
    "inter pacific":                "Inter Pacific Research",
    "interpacific":                 "Inter Pacific Research",
    # Others
    "apex securities":              "Apex Securities",
    "apex":                         "Apex Securities",
    "mplus":                        "M+ Online",
    "m+":                           "M+ Online",
    "rakuten trade":                "Rakuten Trade",
    "rakuten":                      "Rakuten Trade",
    "stockbiz":                     "StockBiz",
    "malacca securities":           "Malacca Securities",
    "malaccasec":                   "Malacca Securities",
}

# URL slug → canonical house (for URL-based fallback detection).
# Longer/more-specific slugs must appear BEFORE shorter ones so the hostname
# pass (which returns on first match) prefers the more specific entry.
_URL_HOUSE_SLUGS: dict[str, str] = {
    # Maybank — long forms first
    "maybankib":        "Maybank IB Research",
    "kimeng":           "Maybank IB Research",
    "maybank":          "Maybank IB Research",
    # CIMB
    "cimbsecurities":   "CIMB Research",
    "cimbresearch":     "CIMB Research",
    "cimb":             "CIMB Research",
    # Kenanga
    "kenangaib":        "Kenanga Research",
    "kenanga":          "Kenanga Research",
    # RHB
    "rhbinvest":        "RHB Research",
    "rhbresearch":      "RHB Research",
    "rhb":              "RHB Research",
    # Hong Leong
    "hlinvest":         "Hong Leong Investment Bank",
    "hongleong":        "Hong Leong Investment Bank",
    "hlbank":           "Hong Leong Investment Bank",
    "hlib":             "Hong Leong Investment Bank",
    # Affin
    "affinhwang":       "Affin Hwang Capital",
    "affinresearch":    "Affin Hwang Capital",
    "affin":            "Affin Hwang Capital",
    # AmInvest
    "aminvestment":     "AmInvest",
    "amresearch":       "AmInvest",
    "aminvest":         "AmInvest",
    # PublicInvest
    "publicinvest":     "PublicInvest Research",
    "publicbank":       "PublicInvest Research",
    "pbinvest":         "PublicInvest Research",
    # MIDF
    "midfamanah":       "MIDF Research",
    "midf":             "MIDF Research",
    # UOB Kay Hian
    "uobkh":            "UOB Kay Hian",
    "uob":              "UOB Kay Hian",
    # TA Securities
    "tasec":            "TA Securities",
    # Alliance
    "alliancedbs":      "Alliance Bank Research",
    "alliance":         "Alliance Bank Research",
    # BIMB
    "bimb":             "BIMB Securities",
    # Phillip Capital
    "phillip":          "Phillip Capital",
    # Inter Pacific
    "interpacific":     "Inter Pacific Research",
    # Others
    "apexsec":          "Apex Securities",
    "rakuten":          "Rakuten Trade",
    "malacca":          "Malacca Securities",
}

# ── Company → ticker lookup (reuse from config) ───────────────────────────────
_TICKER_TO_NAME: dict[str, str] = {s["symbol"]: s["name"] for s in KLCI_STOCKS}
_NAME_TO_TICKER: dict[str, str] = {}
for _s in KLCI_STOCKS:
    _NAME_TO_TICKER[_s["name"].lower()] = _s["symbol"]

_EXTRA_ALIASES: dict[str, str] = {
    "maybank": "1155.KL", "malayan banking": "1155.KL",
    "public bank": "1295.KL", "pbbank": "1295.KL",
    "cimb": "1023.KL", "cimb group": "1023.KL",
    "tenaga": "5347.KL", "tnb": "5347.KL",
    "petronas chemicals": "5183.KL", "pchem": "5183.KL",
    "ihh": "5225.KL", "ihh healthcare": "5225.KL",
    "press metal": "8869.KL", "pmetal": "8869.KL",
    "celcomdigi": "6947.KL", "cdb": "6947.KL",
    "maxis": "6012.KL",
    "rhb": "1066.KL", "rhb bank": "1066.KL",
    "ioi": "1961.KL", "ioi corp": "1961.KL",
    "sime darby plantation": "5285.KL",
    "hong leong bank": "5819.KL", "hlb": "5819.KL",
    "genting": "3182.KL",
    "genting malaysia": "4715.KL", "genm": "4715.KL",
    "telekom": "4863.KL", "tm": "4863.KL",
    "nestle": "4707.KL",
    "ppb": "4065.KL", "ppb group": "4065.KL",
    "petronas gas": "6033.KL", "petgas": "6033.KL",
    "misc": "3816.KL",
    "hartalega": "5168.KL", "harta": "5168.KL",
    "klk": "2445.KL", "kuala lumpur kepong": "2445.KL",
    "dialog": "7277.KL",
    "ambank": "1015.KL", "ammb": "1015.KL",
    "sime darby": "4197.KL",
    "ytl": "4677.KL",
    "gamuda": "5398.KL",
    "ql resources": "5296.KL", "ql": "5296.KL",
    # Common non-KLCI names that may appear in initiation reports
    "genetec": "0104.KL",
    "inari": "0166.KL",
    "frontken": "0128.KL",
    "myeg": "0138.KL", "my e.g.": "0138.KL",
    "aeon": "6599.KL",
    "padini": "7052.KL",
    "bumi armada": "5210.KL",
    "tom tom": "5247.KL",
    "eco world": "8206.KL",
    "eco world international": "5283.KL",
    "sunway": "5211.KL",
    "sunway reit": "5176.KL",
    "ioiprop": "1635.KL",
    "ioi prop": "1635.KL",
    "ioi properties": "1635.KL",
    "axiata": "6888.KL",
    "axiata group": "6888.KL",
    "digi": "6947.KL",
    "petronas dagangan": "5681.KL",
    "petdag": "5681.KL",
    "dutch lady": "3026.KL",
    "fraser neave": "3689.KL",
    "f&n": "3689.KL",
}
_NAME_TO_TICKER.update(_EXTRA_ALIASES)

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
    """Map a company name string to a .KL ticker (longest match wins)."""
    t = text.lower().strip()
    if t in _NAME_TO_TICKER:
        return _NAME_TO_TICKER[t]
    # Also try direct Bursa code: 4-digit number
    bursa_code = re.search(r'\b(\d{4})\b', text)
    if bursa_code:
        code = bursa_code.group(1)
        if any(s["bursa_code"] == code for s in KLCI_STOCKS):
            return f"{code}.KL"
    # Longest substring match
    best_len, best_tk = 0, None
    for name, ticker in _NAME_TO_TICKER.items():
        if len(name) >= 4 and name in t and len(name) > best_len:
            best_len, best_tk = len(name), ticker
    return best_tk


def _resolve_analyst_house(text: str, url: str = "") -> str:
    """Map text/URL to a canonical analyst house name (longest match wins).

    Resolution order:
      1. Longest substring match in the full text body (snippet/title/extra)
      2. URL hostname match — broker's own domain (highest confidence; e.g.
         rhb.com.my → "RHB Research" even if snippet never says "RHB")
      3. URL full-path match — catches broker-tagged slugs on third-party sites
         (e.g. i3investor.com/.../kenanga-research-2025)
    """
    t = text.lower()
    best_len, best_name = 0, ""

    # Pass 1: full-text longest-match
    for fragment, canonical in _ANALYST_HOUSES.items():
        if fragment in t and len(fragment) > best_len:
            best_len, best_name = len(fragment), canonical

    if best_name:
        return best_name

    if not url:
        return "Unknown"

    # Pass 2: hostname-only check (most authoritative — broker's own domain)
    m = re.search(r'https?://(?:www\.)?([^/?#]+)', url.lower())
    hostname = m.group(1) if m else ""
    # Strip common TLD suffixes so "rhb.com.my" → "rhb"
    hostname_core = re.sub(r'\.(com|com\.my|my|sg|net|org)(\.[a-z]{2})?$', '', hostname)
    for slug, canonical in _URL_HOUSE_SLUGS.items():
        if slug in hostname_core:
            return canonical

    # Pass 3: full URL path scan (lower confidence — slug may appear in article path)
    url_clean = re.sub(r'https?://(www\.)?', '', url.lower())
    for slug, canonical in _URL_HOUSE_SLUGS.items():
        if slug in url_clean:
            return canonical

    return "Unknown"


def _classify_report_type(text: str) -> str:
    """Classify report as initiate/upgrade/downgrade/maintain."""
    t = text.lower()
    if re.search(r'\binitiat\w*\b|\binitiates?\b|\bfirst coverage\b|\bnew coverage\b', t):
        return "initiate"
    if re.search(r'\bupgrad\w*\b|\bupgrades?\b', t):
        return "upgrade"
    if re.search(r'\bdowngrad\w*\b|\bdowngrades?\b', t):
        return "downgrade"
    if re.search(r'\bmaintain\w*\b|\breiterat\w*\b|\bneutral\b|\bhold\b', t):
        return "maintain"
    return "other"


def _extract_target_price(text: str) -> Optional[float]:
    """Extract target price from text (e.g. 'TP: 2.80', 'target price RM 3.50')."""
    patterns = [
        r'(?:tp|target\s+price|price\s+target)\s*:?\s*(?:rm\s*)?(\d+(?:\.\d{1,2})?)',
        r'(?:rm\s*)?(\d+(?:\.\d{2}))\s+(?:target|tp)',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            try:
                tp = float(m.group(1))
                if 0.10 <= tp <= 500.0:   # realistic Bursa price range
                    return round(tp, 2)
            except ValueError:
                pass
    return None


def _parse_date(text: str, fallback: Optional[str] = None) -> str:
    """Extract and normalise the first date found in text."""
    date_pat = (
        r'(\d{4}-\d{2}-\d{2}|'
        r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|'
        r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4})'
    )
    m = re.search(date_pat, text, re.IGNORECASE)
    if m:
        raw = m.group(1)
        for fmt in (
            "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%d-%m-%y", "%d/%m/%y",
            "%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b. %d, %Y",
        ):
            try:
                return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
    return fallback or datetime.utcnow().strftime("%Y-%m-%d")


class AnalystCoverageMonitor:
    """Monitors new analyst research coverage on Bursa Malaysia stocks."""

    # ── DB helpers ────────────────────────────────────────────────────────────

    def _ensure_tables(self):
        """Create analyst DB tables if absent (idempotent)."""
        with db_session() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS analyst_coverage_history (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker          TEXT NOT NULL,
                    company         TEXT,
                    analyst_house   TEXT NOT NULL,
                    report_type     TEXT NOT NULL,
                    target_price    REAL,
                    date            TEXT NOT NULL,
                    is_first_coverage INTEGER DEFAULT 0,
                    created_at      TEXT DEFAULT (datetime('now')),
                    UNIQUE(ticker, analyst_house, date)
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_cov_ticker "
                "ON analyst_coverage_history(ticker)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_cov_date "
                "ON analyst_coverage_history(date)"
            )
            conn.execute("""
                CREATE TABLE IF NOT EXISTS analyst_alerts (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker        TEXT NOT NULL,
                    analyst_house TEXT,
                    alert_type    TEXT NOT NULL,
                    date          TEXT NOT NULL,
                    idea_id       INTEGER,
                    created_at    TEXT DEFAULT (datetime('now'))
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_alert_ticker "
                "ON analyst_alerts(ticker)"
            )

    def _save_coverage(self, report: dict, is_first: bool) -> int:
        """Upsert a coverage record; return the row id."""
        with db_session() as conn:
            conn.execute("""
                INSERT INTO analyst_coverage_history
                  (ticker, company, analyst_house, report_type,
                   target_price, date, is_first_coverage)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker, analyst_house, date) DO UPDATE SET
                  report_type=excluded.report_type,
                  target_price=excluded.target_price,
                  is_first_coverage=excluded.is_first_coverage
            """, (
                report["ticker"],
                report.get("company", _TICKER_TO_NAME.get(report["ticker"], "")),
                report["analyst_house"],
                report["report_type"],
                report.get("target_price"),
                report["date"],
                1 if is_first else 0,
            ))
            row = conn.execute("""
                SELECT id FROM analyst_coverage_history
                WHERE ticker=? AND analyst_house=? AND date=?
            """, (report["ticker"], report["analyst_house"], report["date"])).fetchone()
            return row["id"] if row else -1

    def _save_alert(self, ticker: str, analyst_house: str,
                    alert_type: str, date: str, idea_id: Optional[int] = None):
        with db_session() as conn:
            conn.execute("""
                INSERT INTO analyst_alerts
                  (ticker, analyst_house, alert_type, date, idea_id)
                VALUES (?, ?, ?, ?, ?)
            """, (ticker, analyst_house, alert_type, date, idea_id))

    # ── Source 1: Brave Search ────────────────────────────────────────────────

    def _brave_search_reports(self, days_back: int = 1) -> list[dict]:
        """Use Brave API to find recent analyst reports/initiations."""
        if not BRAVE_SEARCH_API_KEY:
            logger.warning("BRAVE_SEARCH_API_KEY not set — skipping Brave analyst search")
            return []

        queries = [
            "Bursa Malaysia analyst initiate coverage 2025",
            "KLSE new analyst report initiation site:i3investor.com",
            "Bursa Malaysia buy initiate coverage research report",
            "Malaysia stock analyst coverage initiate BUY target price 2025",
        ]

        results: list[dict] = []
        seen: set[str]      = set()

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
                    if url in seen:
                        continue
                    seen.add(url)
                    results.append({
                        "title":       r.get("title", ""),
                        "url":         url,
                        "description": r.get("description", ""),
                        "extra":       " ".join(r.get("extra_snippets", [])),
                    })
            except Exception as e:
                logger.debug(f"Brave analyst search failed ({query[:40]}): {e}")

        logger.info(f"Brave analyst search: {len(results)} results")
        return results

    # ── Source 2: i3investor analyst blog ─────────────────────────────────────

    def _scrape_i3investor_analysis(self) -> list[dict]:
        """Scrape i3investor analysis blog for recent analyst reports."""
        url = "https://klse.i3investor.com/web/blog/analysis"
        raw: list[dict] = []
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=25)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            # i3investor blog lists posts with title/author/date
            for article in soup.find_all(["article", "div"],
                                          class_=re.compile(r'post|article|blog|entry', re.I)):
                title_el = article.find(["h1", "h2", "h3", "a"])
                title    = title_el.get_text(strip=True) if title_el else ""
                if not title:
                    continue
                # Author / brokerage
                author_el  = article.find(class_=re.compile(r'author|writer|source', re.I))
                author_txt = author_el.get_text(strip=True) if author_el else ""
                # Date
                date_el  = article.find(["time", "span"],
                                         class_=re.compile(r'date|time|publish', re.I))
                date_txt = date_el.get_text(strip=True) if date_el else ""
                # Description
                body_el  = article.find(["p", "div"],
                                         class_=re.compile(r'excerpt|summary|content|body', re.I))
                body_txt = body_el.get_text(" ", strip=True) if body_el else ""

                raw.append({
                    "title":       title,
                    "url":         url,
                    "description": f"{author_txt} {date_txt} {body_txt}",
                    "extra":       "",
                })

            # Fallback: extract all <a> links with analyst keywords
            if not raw:
                for a in soup.find_all("a", href=True):
                    txt = a.get_text(strip=True)
                    if re.search(r'initiat|research|analyst|coverage|target', txt, re.I):
                        raw.append({
                            "title":       txt,
                            "url":         a["href"] if a["href"].startswith("http") else url,
                            "description": txt,
                            "extra":       "",
                        })

            logger.info(f"i3investor analysis: {len(raw)} items scraped")
        except Exception as e:
            logger.debug(f"i3investor analysis scrape failed: {e}")
        return raw

    # ── Parser ────────────────────────────────────────────────────────────────

    def _parse_reports(self, raw_items: list[dict]) -> list[dict]:
        """Parse raw search/scrape results into structured analyst report dicts."""
        reports: list[dict] = []
        seen: set[tuple]    = set()

        for item in raw_items:
            full = " ".join(filter(None, [
                item.get("title", ""),
                item.get("description", ""),
                item.get("extra", ""),
            ]))

            # Must contain analyst-related keywords
            if not re.search(
                r'\binitiat\w*\b|\bcoverage\b|\banalyst\b|\bresearch\b|\btarget price\b|\btp\b',
                full, re.IGNORECASE,
            ):
                continue

            # Must mention Bursa/Malaysia/KLSE context
            if not re.search(
                r'\bbursa\b|\bklse\b|\bklci\b|\bmalay\w*\b|\b\.KL\b',
                full, re.IGNORECASE,
            ):
                # Allow i3investor URLs (already Bursa-focused)
                if "i3investor" not in item.get("url", "").lower():
                    continue

            ticker       = _resolve_ticker(full)
            if not ticker:
                # Try extracting from the title specifically
                ticker = _resolve_ticker(item.get("title", ""))
            if not ticker:
                continue

            analyst_house = _resolve_analyst_house(full, url=item.get("url", ""))
            report_type   = _classify_report_type(full)
            target_price  = _extract_target_price(full)
            date_str      = _parse_date(full)
            company       = _TICKER_TO_NAME.get(ticker, ticker)

            key = (ticker, analyst_house, date_str)
            if key in seen:
                continue
            seen.add(key)

            reports.append({
                "ticker":        ticker,
                "company":       company,
                "analyst_house": analyst_house,
                "report_type":   report_type,
                "target_price":  target_price,
                "date":          date_str,
                "source_url":    item.get("url", ""),
            })
            logger.debug(
                f"Analyst report: {ticker} [{report_type}] by {analyst_house} "
                f"TP={target_price} on {date_str}"
            )

        logger.info(f"_parse_reports: {len(reports)} reports from {len(raw_items)} items")
        return reports

    # ── Coverage detection ────────────────────────────────────────────────────

    def is_first_coverage(self, ticker: str, analyst_house: str) -> bool:
        """Return True if this is the first time any analyst has covered ticker,
        OR if analyst_house has never covered ticker before."""
        with db_session() as conn:
            # First-ever coverage globally for this ticker?
            any_row = conn.execute(
                "SELECT id FROM analyst_coverage_history WHERE ticker=? LIMIT 1",
                (ticker,),
            ).fetchone()
            if not any_row:
                return True
            # First coverage by this specific house?
            house_row = conn.execute(
                "SELECT id FROM analyst_coverage_history "
                "WHERE ticker=? AND analyst_house=? LIMIT 1",
                (ticker, analyst_house),
            ).fetchone()
            return house_row is None

    def fetch_coverage_history(self, ticker: str) -> list[dict]:
        """Return all historical coverage records for a ticker."""
        with db_session() as conn:
            rows = conn.execute("""
                SELECT date, analyst_house, report_type, target_price, is_first_coverage
                FROM analyst_coverage_history
                WHERE ticker=?
                ORDER BY date DESC
            """, (ticker,)).fetchall()
        return [dict(r) for r in rows]

    # ── Gate 0 idea generation ────────────────────────────────────────────────

    def _create_idea(self, report: dict, is_first: bool) -> dict:
        """Create a Gate 0 alpha idea for a first-coverage initiation."""
        ticker        = report["ticker"]
        company       = report["company"]
        analyst_house = report["analyst_house"]
        report_type   = report["report_type"]
        tp            = report.get("target_price")

        today_str = datetime.utcnow().strftime("%Y-%m-%d")
        slug = (
            f"analyst-init-{re.sub(r'[^a-z0-9]', '', ticker.lower())}"
            f"-{re.sub(r'[^a-z0-9]', '', analyst_house.lower()[:12])}"
            f"-{today_str}"
        )

        with db_session() as conn:
            existing = conn.execute(
                "SELECT id FROM alpha_ideas WHERE slug=?", (slug,)
            ).fetchone()
        if existing:
            return {"created": False, "idea_id": existing["id"],
                    "reason": "Analyst idea already exists for today"}

        coverage_kind = (
            "first-ever analyst research coverage"
            if is_first else f"first coverage from {analyst_house}"
        )
        tp_str = f" with target price RM {tp:.2f}" if tp else ""

        title = f"Analyst Initiation — {company} ({ticker})"
        hypothesis = (
            f"{analyst_house} initiates coverage on {ticker} ({company}) "
            f"with {report_type.upper()} rating{tp_str}. "
            f"This represents {coverage_kind}. "
            f"Market historically underprices the visibility uplift from coverage initiation: "
            f"(1) Institutional mandate barrier removed — fund managers require analyst coverage "
            f"before investing under their mandates; "
            f"(2) Information diffusion — initiation report distributes fundamental analysis "
            f"to a wider investor base, reducing information asymmetry; "
            f"(3) Visibility uplift — stock appears in screening databases for the first time. "
            f"Empirically, first-coverage BUY initiations outperform KLCI by 8-15% over 60 days. "
            f"Effect strongest for small/mid-cap stocks where information asymmetry is highest."
        )
        formula = (
            f"Entry trigger: {analyst_house} initiation report published for {ticker}. "
            f"Entry: buy at market open on next trading day after initiation date. "
            f"Position size: 5% of portfolio NAV. "
            f"Primary exit: 60 calendar days from entry. "
            f"Secondary exit: price reaches analyst TP of RM {tp:.2f}" if tp
            else "Secondary exit: price +15% from entry. "
        )
        formula += (
            f" Stop-loss: -8% from entry. "
            f"Monitor for: subsequent initiations by other analysts (adds momentum), "
            f"earnings surprise in next reporting window (Feb/May/Aug/Nov). "
            f"Data: Bursa analyst reports, Yahoo Finance {ticker} OHLCV."
        )

        with db_session() as conn:
            conn.execute("""
                INSERT OR IGNORE INTO alpha_ideas
                  (slug, title, hypothesis, ticker, timeframe, factor_formula,
                   data_sources, stage, status, novelty_score, logic_score)
                VALUES (?, ?, ?, ?, '60d', ?, ?, 'gate0', 'pending', ?, ?)
            """, (
                slug, title, hypothesis, ticker, formula,
                json.dumps([
                    f"Bursa Malaysia analyst coverage ({analyst_house})",
                    f"Yahoo Finance {ticker} daily OHLCV",
                    "i3investor analyst reports",
                ]),
                0.65,   # novelty: documented effect, ticker-specific timing is alpha
                0.78,   # logic: institutional mandate + info asymmetry mechanism is strong
            ))
            row = conn.execute(
                "SELECT id FROM alpha_ideas WHERE slug=?", (slug,)
            ).fetchone()
            idea_id = row["id"]

            conn.execute("""
                INSERT INTO pipeline_events
                  (idea_id, stage, event_type, agent, notes)
                VALUES (?, 'gate0', 'created', 'AnalystCoverageMonitor', ?)
            """, (idea_id, f"Auto-created: {analyst_house} {report_type} on {ticker}"))

            conn.execute(
                "INSERT INTO daemon_logs (level, source, message) VALUES "
                "('INFO', 'AnalystCoverageMonitor', ?)",
                (f"Created Gate 0 idea [{idea_id}] '{title}'",),
            )

        return {"created": True, "idea_id": idea_id, "reason": "First analyst coverage"}

    # ── Process a single report ───────────────────────────────────────────────

    def process_new_report(self, report: dict) -> dict:
        """Save report to DB; if first coverage, create Gate 0 idea + alert.

        Returns:
            {saved: bool, is_first: bool, idea_created: bool, idea_id: int|None}
        """
        ticker        = report["ticker"]
        analyst_house = report["analyst_house"]
        is_first      = self.is_first_coverage(ticker, analyst_house)

        cov_id  = self._save_coverage(report, is_first)
        idea_id = None

        if is_first and report["report_type"] in ("initiate", "upgrade"):
            result  = self._create_idea(report, is_first)
            idea_id = result.get("idea_id")
            self._save_alert(
                ticker, analyst_house,
                "first_coverage" if is_first else "initiation",
                report["date"], idea_id,
            )
            logger.info(
                f"FIRST COVERAGE: {ticker} by {analyst_house} "
                f"({report['report_type']}) → idea [{idea_id}]"
            )
        elif report["report_type"] == "upgrade":
            self._save_alert(ticker, analyst_house, "upgrade", report["date"])

        return {
            "saved":         cov_id > 0,
            "is_first":      is_first,
            "idea_created":  idea_id is not None,
            "idea_id":       idea_id,
            "coverage_id":   cov_id,
        }

    # ── Public entry point ────────────────────────────────────────────────────

    def fetch_new_reports(self, days_back: int = 1) -> list[dict]:
        """Fetch, parse, and process all new analyst reports.

        Steps:
          1. Brave Search — initiation/coverage keywords
          2. i3investor analysis blog
          3. Parse into structured reports
          4. process_new_report() for each
          5. Return list of processed reports with metadata

        Args:
            days_back: how many days back to consider "new" (used for logging)

        Returns:
            List of report dicts enriched with {is_first, idea_created, idea_id}.
        """
        self._ensure_tables()

        raw: list[dict] = []
        raw.extend(self._brave_search_reports(days_back))
        raw.extend(self._scrape_i3investor_analysis())

        reports = self._parse_reports(raw)

        enriched: list[dict] = []
        for report in reports:
            try:
                result = self.process_new_report(report)
                enriched.append({**report, **result})
            except Exception as e:
                logger.warning(f"process_new_report failed [{report.get('ticker')}]: {e}")

        first_count   = sum(1 for r in enriched if r.get("is_first"))
        upgrade_count = sum(1 for r in enriched if r.get("report_type") == "upgrade")
        logger.info(
            f"fetch_new_reports: {len(raw)} raw → {len(reports)} parsed → "
            f"{len(enriched)} processed ({first_count} first-coverage, "
            f"{upgrade_count} upgrades)"
        )

        with db_session() as conn:
            conn.execute(
                "INSERT INTO daemon_logs (level, source, message) VALUES "
                "('INFO', 'AnalystCoverageMonitor', ?)",
                (
                    f"Analyst scan: {len(raw)} raw items, {len(reports)} parsed, "
                    f"{first_count} first-coverage, {upgrade_count} upgrades",
                ),
            )

        return enriched

    # ── Reporting helpers (for Telegram) ─────────────────────────────────────

    def recent_events(self, days: int = 7) -> dict:
        """Return analyst events from the last N days grouped by type."""
        cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        with db_session() as conn:
            rows = conn.execute("""
                SELECT h.ticker, h.company, h.analyst_house, h.report_type,
                       h.target_price, h.date, h.is_first_coverage
                FROM analyst_coverage_history h
                WHERE h.date >= ?
                ORDER BY h.date DESC
            """, (cutoff,)).fetchall()

        firsts:    list[dict] = []
        upgrades:  list[dict] = []
        downgrades: list[dict] = []
        maintains: list[dict] = []

        for r in rows:
            entry = dict(r)
            if r["is_first_coverage"]:
                firsts.append(entry)
            elif r["report_type"] == "upgrade":
                upgrades.append(entry)
            elif r["report_type"] == "downgrade":
                downgrades.append(entry)
            else:
                maintains.append(entry)

        return {
            "days":        days,
            "cutoff_date": cutoff,
            "first_coverage": firsts,
            "upgrades":       upgrades,
            "downgrades":     downgrades,
            "maintains":      maintains,
            "total":          len(rows),
        }
