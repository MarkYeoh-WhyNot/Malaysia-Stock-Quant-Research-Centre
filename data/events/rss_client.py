"""
RSSClient — uses Brave Search API as news source for Bursa-relevant market news.
RSS feeds are blocked on this VPS; Brave Search replaces them.
Deduplicates against market_events table using SHA-256 event IDs.
"""
import hashlib
import logging
import os
import time
from datetime import datetime, timezone

import requests

from config.settings import MARKET_MODE
from data.database import db_session

logger = logging.getLogger(__name__)

BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"

# Native RSS is blocked on this VPS (see module docstring) — Brave Search
# substitutes for it. Queries are market-specific so crypto and Bursa each
# get their own relevant news, not a shared/wrong feed.
BURSA_QUERIES = [
    "Malaysia stock market news today",
    "Bursa Malaysia announcement today",
    "palm oil CPO price today",
    "Malaysia economy news today",
    "KLCI market today",
    "Petronas Malaysia news today",
]

CRYPTO_QUERIES = [
    "bitcoin ethereum news today",
    "crypto market news today",
    "CoinDesk bitcoin today",
    "cryptocurrency regulation news today",
    "altcoin news today",
    "crypto exchange news today",
]

BRAVE_QUERIES = CRYPTO_QUERIES if MARKET_MODE == "crypto" else BURSA_QUERIES


def _make_event_id(url: str, title: str) -> str:
    raw = f"{url}|{title}"
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


class RSSClient:
    """Fetches Bursa-relevant news via Brave Search API (replaces blocked RSS feeds)."""

    def __init__(self):
        self.api_key = os.getenv("BRAVE_SEARCH_API_KEY", "")
        if not self.api_key:
            logger.warning("RSSClient: BRAVE_SEARCH_API_KEY not set — news search disabled")

    def _search(self, query: str) -> list:
        """
        Run a single Brave Search query.
        Returns list of new result dicts not yet in market_events.
        """
        if not self.api_key:
            return []

        try:
            resp = requests.get(
                BRAVE_SEARCH_URL,
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "X-Subscription-Token": self.api_key,
                },
                params={
                    "q": query,
                    "count": 5,
                    "freshness": "pd",  # past day only
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response else "?"
            logger.warning(f"RSSClient: Brave Search HTTP {status} for '{query}'")
            return []
        except Exception as exc:
            logger.warning(f"RSSClient: Brave Search error for '{query}': {exc}")
            return []

        results = []
        web_results = data.get("web", {}).get("results", [])
        for item in web_results:
            title = item.get("title", "") or ""
            url = item.get("url", "") or ""
            description = item.get("description", "") or ""
            # Brave returns age as a string like "3 hours ago" — use now as fallback
            age_str = item.get("age", "")
            published_at = _parse_brave_age(age_str)

            if not title or not url:
                continue

            event_id = _make_event_id(url, title)
            if _already_seen(event_id):
                continue

            results.append({
                "event_id": event_id,
                "source": "brave_search",
                "headline": title,
                "body": description[:1000] if description else None,
                "raw_url": url,
                "published_at": published_at,
                "feed_name": f"brave:{query[:40]}",
            })

        logger.debug(f"RSSClient: '{query[:40]}' → {len(results)} new results")
        return results

    def brave_search_news(self) -> list:
        """
        Search all configured queries via Brave Search API.
        Returns combined deduplicated list of new entries.
        """
        if not self.api_key:
            logger.info("RSSClient: skipping Brave Search (no API key)")
            return []

        seen_ids: set[str] = set()
        all_entries = []

        for query in BRAVE_QUERIES:
            try:
                results = self._search(query)
                for r in results:
                    if r["event_id"] not in seen_ids:
                        seen_ids.add(r["event_id"])
                        all_entries.append(r)
            except Exception as exc:
                logger.warning(f"RSSClient: query error for '{query}': {exc}")
            time.sleep(1)  # rate-limit between queries

        logger.info(f"RSSClient: Brave Search → {len(all_entries)} new entries")
        return all_entries

    # Keep fetch_all_feeds() as an alias so EventWatcher's call site doesn't change
    def fetch_all_feeds(self) -> list:
        """Alias for brave_search_news() — keeps EventWatcher call site unchanged."""
        return self.brave_search_news()


def _parse_brave_age(age_str: str) -> str:
    """
    Convert Brave's human-readable age string to an approximate ISO datetime.
    Examples: '3 hours ago', '1 day ago', '45 minutes ago'
    Falls back to utcnow() if unparseable.
    """
    now = datetime.utcnow()
    if not age_str:
        return now.isoformat()

    age_lower = age_str.lower().strip()
    try:
        import re
        m = re.search(r"(\d+)\s+(minute|hour|day|week)", age_lower)
        if m:
            n = int(m.group(1))
            unit = m.group(2)
            from datetime import timedelta
            delta_map = {
                "minute": timedelta(minutes=n),
                "hour":   timedelta(hours=n),
                "day":    timedelta(days=n),
                "week":   timedelta(weeks=n),
            }
            dt = now - delta_map[unit]
            return dt.isoformat()
    except Exception:
        pass

    return now.isoformat()
