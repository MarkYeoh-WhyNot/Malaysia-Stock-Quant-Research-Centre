"""
KLSE Screener base HTTP client.
Single fetch_page() function used by both fundamental_scraper and screener modules.
"""
import logging
import time

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

BASE_URL = "https://www.klsescreener.com/v2"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.klsescreener.com/v2/",
    "Connection": "keep-alive",
}


_POST_HEADERS = {
    **HEADERS,
    "Content-Type": "application/x-www-form-urlencoded",
    "X-Requested-With": "XMLHttpRequest",
}


def fetch_page(path: str, params: dict = None) -> BeautifulSoup | None:
    """GET {BASE_URL}{path} with polite 1.5s delay. Returns BeautifulSoup or None on error."""
    time.sleep(1.5)
    url = f"{BASE_URL}{path}"
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=20)
        if r.status_code == 200:
            return BeautifulSoup(r.text, "html.parser")
        logger.warning(f"[KLSEClient] HTTP {r.status_code} for {url}")
        return None
    except Exception as e:
        logger.warning(f"[KLSEClient] Fetch error {url}: {e}")
        return None


def post_screener(data: dict) -> tuple[BeautifulSoup | None, str]:
    """POST to /v2/screener/quote_results (AJAX endpoint). Returns (soup, url)."""
    time.sleep(1.5)
    url = f"{BASE_URL}/screener/quote_results"
    payload = {"getquote": "1", **data}
    try:
        r = requests.post(url, data=payload, headers=_POST_HEADERS, timeout=30)
        if r.status_code == 200:
            return BeautifulSoup(r.text, "html.parser"), url
        logger.warning(f"[KLSEClient] HTTP {r.status_code} for POST {url}")
        return None, url
    except Exception as e:
        logger.warning(f"[KLSEClient] POST error {url}: {e}")
        return None, url


def log_daemon(level: str, source: str, message: str) -> None:
    """Write a log entry to daemon_logs table (non-blocking)."""
    try:
        from data.database import db_session
        with db_session() as conn:
            conn.execute(
                "INSERT INTO daemon_logs (level, source, message) VALUES (?, ?, ?)",
                (level.upper(), source, message),
            )
    except Exception:
        pass
    getattr(logger, level.lower(), logger.info)(f"[{source}] {message}")
