"""
KLSE Screener base HTTP client.
Single fetch_page() function used by both fundamental_scraper and screener modules.
Also houses the canonical SLUG_MAP (stock code → klsescreener.com URL slug) and the
KLSEScreenerClient wrapper class used by tests and other modules.
"""
import logging
import time

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ── Canonical stock-code → klsescreener.com URL-slug mapping ─────────────────
# Keys are stock codes (numeric string, as used in klsescreener.com URLs).
# Values are the URL slug used in /stocks/view/{code}/{slug} paths.
# Covers FBM KLCI 30 + FBM70 additional stocks.
SLUG_MAP: dict[str, str] = {
    # ── KLCI 30 (original) ──────────────────────────────────────────────────
    "1023": "cimb-group-holdings-berhad",
    "1155": "malayan-banking-berhad",
    "1295": "public-bank-berhad",
    "5285": "sime-darby-plantation-berhad",
    "5347": "tenaga-nasional-berhad",
    "4863": "telekom-malaysia-berhad",
    "6947": "celcomdigi-berhad",
    "5225": "ihh-healthcare-berhad",
    "2291": "ioi-corporation-berhad",
    "5182": "kuala-lumpur-kepong-berhad",
    "1066": "rhb-bank-berhad",
    "5819": "hong-leong-bank-berhad",
    "1082": "hong-leong-financial-group-berhad",
    "4197": "sime-darby-berhad",
    "5398": "petronas-gas-berhad",
    "5183": "petronas-dagangan-berhad",
    "6033": "misc-berhad",
    "4715": "genting-berhad",
    "3182": "genting-malaysia-berhad",
    "5681": "maxis-berhad",
    "6888": "axiata-group-berhad",
    "1961": "ppb-group-berhad",
    "7277": "dialog-group-berhad",
    "5168": "hartalega-holdings-berhad",
    "5069": "hap-seng-plantations-berhad",
    # ── FBM70 Additional ────────────────────────────────────────────────────
    # Consumer
    "6599": "aeon-co-m-bhd",
    "5196": "berjaya-food-berhad",
    "3026": "dutch-lady-milk-industries-berhad",
    "4707": "nestle-malaysia-berhad",
    "7052": "padini-holdings-berhad",
    "7084": "ql-resources-berhad",
    "7103": "spritzer-bhd",
    # Healthcare
    "5878": "kpj-healthcare-berhad",
    "7081": "pharmaniaga-berhad",
    "7153": "kossan-rubber-industries-berhad",
    # Technology / EMS
    "0166": "inari-amertron-berhad",
    "3867": "malaysian-pacific-industries-berhad",
    "5005": "unisem-m-berhad",
    "0128": "frontken-corporation-berhad",
    "0097": "vitrox-corporation-berhad",
    "0208": "greatech-technology-berhad",
    # Industrial / Building Materials
    "7162": "astino-berhad",
    "5026": "engtex-group-berhad",
    "3794": "lafarge-malaysia-berhad",
    "8869": "press-metal-aluminium-holdings-berhad",
    # REITs
    "5106": "axis-reit-managers-berhad",
    "5227": "igb-reit",
    "5079": "mrcb-quill-reit",
    "5212": "pavilion-reit",
    "5176": "sunway-real-estate-investment-trust",
    # Property
    "1061": "ioi-properties-group-berhad",
    "8583": "mah-sing-group-berhad",
    "5288": "sime-darby-property-berhad",
    "8664": "sp-setia-berhad",
    "5148": "uem-sunrise-berhad",
    # Utilities
    "5264": "malakoff-corporation-berhad",
    "6742": "ytl-power-international-berhad",
    # Media / Services
    "6399": "astro-malaysia-holdings-berhad",
    "6084": "the-star-media-group-berhad",
    # Transport / Logistics
    "5014": "malaysia-airports-holdings-berhad",
    "2194": "mmc-corporation-berhad",
    "5246": "westports-holdings-berhad",
    # Auto
    "5248": "bermaz-auto-berhad",
    "5983": "mbm-resources-berhad",
    # Gloves
    "7113": "top-glove-corporation-berhad",
    "7106": "supermax-corporation-berhad",
    # Plantation (additional)
    "5254": "boustead-plantations-berhad",
    "5222": "felda-global-ventures-holdings-berhad",
    "5012": "ta-ann-holdings-berhad",
}

# ── Yahoo Finance ticker → klsescreener.com stock code lookup ─────────────────
# Derived automatically from SLUG_MAP (code becomes the lookup key for .KL tickers).
_YF_TO_CODE: dict[str, str] = {f"{code}.KL": code for code in SLUG_MAP}


class KLSEScreenerClient:
    """Thin wrapper exposing the canonical FBM70 universe and SLUG_MAP.

    Used by backtesting and screening modules to access the canonical stock list
    without importing from fundamental_scraper (which imports this module).

    Usage:
        from data.klse_screener.client import KLSEScreenerClient
        c = KLSEScreenerClient()
        print(len(c.SLUG_MAP))  # ≥ 60
        tickers = c.get_all_tickers()
    """

    SLUG_MAP = SLUG_MAP

    def get_all_tickers(self) -> list[str]:
        """Return all Yahoo Finance tickers in the universe (e.g. ['1155.KL', ...])."""
        return sorted(_YF_TO_CODE.keys())

    def get_code(self, ticker: str) -> str | None:
        """Convert a Yahoo Finance ticker (e.g. '1155.KL') to klsescreener stock code."""
        return _YF_TO_CODE.get(ticker)

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
