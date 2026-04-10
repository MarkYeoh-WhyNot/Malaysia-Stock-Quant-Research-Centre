"""
BursaScraper — scrapes KLSE Screener announcements via its JSON AJAX endpoint.
Uses company name → ticker mapping for KLCI stocks.
"""
import hashlib
import json
import logging
import re
from datetime import datetime, date, timedelta

import requests
from bs4 import BeautifulSoup

from data.database import db_session
from config.settings import KLCI_STOCKS

logger = logging.getLogger(__name__)

KLSE_ANNOUNCEMENTS_URL = "https://www.klsescreener.com/v2/announcements"

HEADERS_HTML = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

HEADERS_AJAX = {
    "User-Agent": HEADERS_HTML["User-Agent"],
    "Accept": "application/json, text/plain, */*",
    "X-Requested-With": "XMLHttpRequest",
    "Referer": KLSE_ANNOUNCEMENTS_URL,
}

ANNOUNCEMENT_TYPES = {
    "financial results": "earnings",
    "quarterly report": "earnings",
    "quarterly": "earnings",
    "annual report": "earnings",
    "dividend": "dividend_declared",
    "interim dividend": "dividend_declared",
    "final dividend": "dividend_declared",
    "special dividend": "dividend_declared",
    "single-tier": "dividend_declared",
    "distribution": "dividend_declared",
    "bonus issue": "bonus_issue",
    "bonus shares": "bonus_issue",
    "rights issue": "rights_issue",
    "rights entitlement": "rights_issue",
    "contract": "contract_win",
    "letter of award": "contract_win",
    "letter of intent": "contract_win",
    "subcontract": "contract_win",
    "memorandum of understanding": "mou",
    "mou": "mou",
    "acquisition": "acquisition",
    "proposed acquisition": "acquisition",
    "disposal": "disposal",
    "proposed disposal": "disposal",
    "substantial shareholder": "shareholding_change",
    "notice of change": "shareholding_change",
    "change in boardroom": "boardroom_change",
    "placement": "placement",
    "private placement": "placement",
    "share buyback": "buyback",
    "buy back": "buyback",
}

# Build company name → ticker mapping from KLCI universe
# Both full name and short uppercase name variants
_COMPANY_MAP: dict[str, str] = {}
for _stock in KLCI_STOCKS:
    name_lower = _stock["name"].lower()
    _COMPANY_MAP[name_lower] = _stock["symbol"]
    # Also map first word (e.g. "maybank" → 1155.KL)
    first_word = name_lower.split()[0]
    if first_word not in _COMPANY_MAP:
        _COMPANY_MAP[first_word] = _stock["symbol"]


def _company_to_ticker(company_name: str) -> str | None:
    """Look up ticker for a company name from KLCI universe."""
    name_lower = company_name.lower().strip()
    # Exact match
    if name_lower in _COMPANY_MAP:
        return _COMPANY_MAP[name_lower]
    # Partial match — company name contains known name
    for known, ticker in _COMPANY_MAP.items():
        if len(known) > 4 and known in name_lower:
            return ticker
    return None


def _make_event_id(company: str, subject: str, date_str: str) -> str:
    raw = f"{company}|{subject}|{date_str}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _already_seen(event_id: str) -> bool:
    try:
        with db_session() as conn:
            row = conn.execute(
                "SELECT id FROM market_events WHERE event_id = ?", (event_id,)
            ).fetchone()
        return row is not None
    except Exception:
        return False


def _classify_subject(subject: str) -> str:
    subject_lower = subject.lower()
    for keyword, event_type in ANNOUNCEMENT_TYPES.items():
        if keyword in subject_lower:
            return event_type
    return "general"


def _parse_klse_time(time_str: str) -> str:
    """Parse KLSE Screener time string like '2026-04-10 - 6:38 pm' to ISO."""
    time_str = time_str.strip()
    # Format: "2026-04-10 - 6:38 pm"
    m = re.match(r"(\d{4}-\d{2}-\d{2})\s*-\s*(.+)", time_str)
    if m:
        date_part = m.group(1)
        time_part = m.group(2).strip()
        for fmt in ("%I:%M %p", "%H:%M"):
            try:
                t = datetime.strptime(time_part, fmt).time()
                dt = datetime.fromisoformat(date_part).replace(
                    hour=t.hour, minute=t.minute
                )
                return dt.isoformat()
            except ValueError:
                pass
    # Already ISO-like
    try:
        return datetime.fromisoformat(time_str).isoformat()
    except Exception:
        pass
    return datetime.utcnow().isoformat()


class BursaScraper:
    """Fetches recent Bursa Malaysia announcements via KLSE Screener AJAX."""

    def __init__(self):
        self._session = requests.Session()
        # Prime session with cookies
        try:
            self._session.get(
                KLSE_ANNOUNCEMENTS_URL,
                headers=HEADERS_HTML,
                timeout=10,
            )
        except Exception:
            pass

    def fetch_announcements(self, hours_back: int = 1) -> list:
        """
        Fetch new Bursa announcements from KLSE Screener AJAX endpoint.
        Returns list of event dicts not yet in market_events.
        """
        try:
            resp = self._session.get(
                KLSE_ANNOUNCEMENTS_URL,
                headers=HEADERS_AJAX,
                params={"ajax": "1"},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning(f"BursaScraper: fetch failed: {exc}")
            return []

        html = data.get("html", "")
        if not html:
            logger.debug("BursaScraper: empty HTML in response")
            return []

        soup = BeautifulSoup(html, "lxml")
        cutoff = datetime.utcnow() - timedelta(hours=hours_back)
        today_str = date.today().isoformat()
        results = []

        links = soup.find_all("a", href=re.compile(r"/v2/announcements/view/"))
        for link in links:
            try:
                card = link.find("div", class_="cardmy")
                if not card:
                    continue

                header = card.find("div", class_="card-header")
                body_div = card.find("div", class_="card-body")
                if not header or not body_div:
                    continue

                spans = header.find_all("span")
                company = spans[0].get_text(strip=True) if spans else ""
                time_str = spans[1].get_text(strip=True) if len(spans) > 1 else ""

                # Subject is the first non-empty div text in card-body
                subject = ""
                for d in body_div.find_all("div"):
                    text = d.get_text(strip=True)
                    if text:
                        subject = text
                        break

                if not company or not subject:
                    continue

                published_at = _parse_klse_time(time_str)

                # Age filter
                try:
                    pub_dt = datetime.fromisoformat(published_at)
                    if pub_dt < cutoff:
                        continue
                except Exception:
                    pass

                event_id = _make_event_id(company, subject, today_str)
                if _already_seen(event_id):
                    continue

                # Try to look up ticker from company name
                ticker = _company_to_ticker(company)

                detail_href = link.get("href", "")
                raw_url = (
                    f"https://www.klsescreener.com{detail_href}"
                    if detail_href
                    else None
                )

                results.append({
                    "event_id": event_id,
                    "source": "bursa",
                    "ticker": ticker,
                    "company": company,
                    "event_type": _classify_subject(subject),
                    "headline": subject,
                    "body": None,
                    "raw_url": raw_url,
                    "published_at": published_at,
                })

            except Exception as exc:
                logger.debug(f"BursaScraper: row parse error: {exc}")
                continue

        logger.info(f"BursaScraper: found {len(results)} new announcements")
        return results

    def fetch_announcement_detail(self, url: str) -> str:
        """Fetch full text of a Bursa announcement. Returns cleaned text."""
        if not url:
            return ""
        try:
            resp = self._session.get(url, headers=HEADERS_HTML, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")
            for tag in soup(["script", "style", "nav", "header", "footer"]):
                tag.decompose()
            text = soup.get_text(separator=" ", strip=True)
            text = re.sub(r"\s+", " ", text).strip()
            return text[:3000]
        except Exception as exc:
            logger.debug(f"BursaScraper: detail fetch failed for {url}: {exc}")
            return ""
