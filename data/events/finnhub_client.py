"""
FinnhubClient — fetches economic calendar events and market news from Finnhub.
Free tier API: https://finnhub.io/register
Add FINNHUB_API_KEY to .env
"""
import hashlib
import logging
import os
import time
from datetime import datetime, timedelta, timezone

import requests

from data.database import db_session

logger = logging.getLogger(__name__)

FINNHUB_BASE = "https://finnhub.io/api/v1"

HIGH_IMPORTANCE_EVENTS = [
    "Fed Interest Rate Decision",
    "FOMC Statement",
    "FOMC Minutes",
    "China Manufacturing PMI",
    "China Caixin Manufacturing PMI",
    "China Non-Manufacturing PMI",
    "US Non-Farm Payrolls",
    "US CPI",
    "US Core CPI",
    "ECB Interest Rate Decision",
    "Bank of England Rate Decision",
    "Malaysia Interest Rate Decision",
    "Malaysia CPI",
    "Malaysia GDP",
    "Crude Oil Inventories",
    "Bank of Japan Rate Decision",
    "US GDP",
    "US Retail Sales",
    "China GDP",
]

# Map event name patterns to structured event_type codes
EVENT_TYPE_MAP = {
    "Malaysia Interest Rate Decision": "bnm_opr",
    "Malaysia CPI": "malaysia_cpi",
    "Malaysia GDP": "malaysia_gdp",
    "Fed Interest Rate Decision": "fed_decision",
    "FOMC": "fed_decision",
    "China Manufacturing PMI": "china_pmi",
    "China Caixin": "china_pmi",
    "China Non-Manufacturing PMI": "china_pmi",
    "China GDP": "china_gdp",
    "ECB Interest Rate Decision": "ecb_decision",
    "Bank of England": "boe_decision",
    "Bank of Japan": "boj_decision",
    "US Non-Farm Payrolls": "us_nfp",
    "US CPI": "us_cpi",
    "US GDP": "us_gdp",
    "Crude Oil Inventories": "crude_inventories",
}

MALAYSIA_KEYWORDS = [
    "malaysia", "klci", "bursa", "ringgit", "maybank",
    "petronas", "cimb", "tenaga", "palm oil", "asean",
    "southeast asia", "myr",
]


def _map_event_type(event_name: str) -> str:
    name_lower = event_name.lower()
    for pattern, etype in EVENT_TYPE_MAP.items():
        if pattern.lower() in name_lower:
            return etype
    return "macro_event"


def _make_news_event_id(url: str, title: str) -> str:
    raw = f"{url}|{title}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class FinnhubClient:
    """Fetches economic calendar and news from Finnhub free API."""

    def __init__(self):
        self.api_key = os.getenv("FINNHUB_API_KEY", "")
        if not self.api_key:
            logger.warning("FinnhubClient: FINNHUB_API_KEY not set — calendar features disabled")

    def _get(self, path: str, params: dict) -> dict | list | None:
        if not self.api_key:
            return None
        params["token"] = self.api_key
        try:
            resp = requests.get(f"{FINNHUB_BASE}{path}", params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as exc:
            if exc.response.status_code == 429:
                logger.warning("FinnhubClient: rate limited — sleeping 60s")
                time.sleep(60)
            else:
                logger.warning(f"FinnhubClient: HTTP error {exc.response.status_code} for {path}")
            return None
        except Exception as exc:
            logger.warning(f"FinnhubClient: request failed for {path}: {exc}")
            return None

    def fetch_economic_calendar(self, days_ahead: int = 30) -> int:
        """
        Fetch upcoming high-importance economic events and upsert into economic_calendar.
        Returns count of events added/updated.
        """
        today = datetime.utcnow().date()
        to_date = today + timedelta(days=days_ahead)
        data = self._get("/calendar/economic", {
            "from": today.isoformat(),
            "to": to_date.isoformat(),
        })
        if not data or not isinstance(data, dict):
            return 0

        events = data.get("economicCalendar", [])
        if not events:
            return 0

        count = 0
        with db_session() as conn:
            for ev in events:
                event_name = ev.get("event", "")
                if not event_name:
                    continue

                # Filter to high-importance events only
                is_high = any(
                    h.lower() in event_name.lower()
                    for h in HIGH_IMPORTANCE_EVENTS
                )
                if not is_high:
                    continue

                scheduled_date = ev.get("time", "")[:10] if ev.get("time") else ""
                if not scheduled_date:
                    continue

                scheduled_time_raw = ev.get("time", "")
                scheduled_time = scheduled_time_raw[11:16] if len(scheduled_time_raw) > 10 else None

                event_type = _map_event_type(event_name)
                country = ev.get("country", "")
                importance = "high"  # we've already filtered to high importance

                try:
                    conn.execute("""
                        INSERT INTO economic_calendar
                          (event_name, event_type, scheduled_date, scheduled_time,
                           country, importance, forecast_value, previous_value)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(event_name, scheduled_date) DO UPDATE SET
                          forecast_value=excluded.forecast_value,
                          previous_value=excluded.previous_value
                    """, (
                        event_name,
                        event_type,
                        scheduled_date,
                        scheduled_time,
                        country,
                        importance,
                        str(ev.get("estimate", "")) or None,
                        str(ev.get("prev", "")) or None,
                    ))
                    count += 1
                except Exception as exc:
                    logger.debug(f"FinnhubClient: calendar upsert failed: {exc}")

        logger.info(f"FinnhubClient: upserted {count} calendar events")
        return count

    def fetch_market_news(self, category: str = "general") -> list:
        """
        Fetch general market news filtered to Malaysia/ASEAN/commodity topics.
        Returns list of event dicts in RSSClient format.
        """
        data = self._get("/news", {"category": category})
        if not data or not isinstance(data, list):
            return []

        results = []
        kw_lower = [kw.lower() for kw in MALAYSIA_KEYWORDS]
        for item in data:
            headline = item.get("headline", "") or ""
            summary = item.get("summary", "") or ""
            text_lower = (headline + " " + summary).lower()
            if not any(kw in text_lower for kw in kw_lower):
                continue

            url = item.get("url", "") or ""
            published_ts = item.get("datetime", 0)
            try:
                published_at = datetime.fromtimestamp(published_ts, tz=timezone.utc).isoformat()
            except Exception:
                published_at = datetime.utcnow().isoformat()

            event_id = _make_news_event_id(url, headline)
            results.append({
                "event_id": event_id,
                "source": "finnhub",
                "headline": headline,
                "body": summary[:1000] if summary else None,
                "raw_url": url,
                "published_at": published_at,
                "feed_name": f"finnhub_{category}",
            })

        logger.debug(f"FinnhubClient: {len(results)} Malaysia-relevant news items")
        return results

    def fetch_company_news(self, symbol: str, days_back: int = 1) -> list:
        """
        Fetch news for a specific company ticker from Finnhub.
        Note: Finnhub uses symbols like "CIMB.KL" — works for major KLSE stocks.
        Returns list of news dicts.
        """
        today = datetime.utcnow().date()
        from_date = today - timedelta(days=days_back)
        data = self._get("/company-news", {
            "symbol": symbol,
            "from": from_date.isoformat(),
            "to": today.isoformat(),
        })
        if not data or not isinstance(data, list):
            return []

        results = []
        for item in data:
            headline = item.get("headline", "") or ""
            url = item.get("url", "") or ""
            published_ts = item.get("datetime", 0)
            try:
                published_at = datetime.fromtimestamp(published_ts, tz=timezone.utc).isoformat()
            except Exception:
                published_at = datetime.utcnow().isoformat()

            event_id = _make_news_event_id(url, headline)
            results.append({
                "event_id": event_id,
                "source": "finnhub",
                "headline": headline,
                "body": (item.get("summary", "") or "")[:1000],
                "raw_url": url,
                "published_at": published_at,
            })
        return results

    def check_upcoming_events(self) -> list:
        """
        Query economic_calendar for events in next 48h.
        Returns list of upcoming events for monitoring priority.
        """
        now = datetime.utcnow()
        cutoff = (now + timedelta(hours=48)).isoformat()[:10]
        today = now.isoformat()[:10]
        with db_session() as conn:
            rows = conn.execute("""
                SELECT * FROM economic_calendar
                WHERE scheduled_date >= ? AND scheduled_date <= ?
                  AND importance = 'high' AND processed = 0
                ORDER BY scheduled_date, scheduled_time
            """, (today, cutoff)).fetchall()
        return [dict(r) for r in rows]
