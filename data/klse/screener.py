"""
KLSE Screener — scrapes klsescreener.com for Bursa Malaysia fundamental data.
Falls back gracefully to the hardcoded KLCI_STOCKS universe in settings.py.
"""
import logging
import time
from typing import Optional
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://klsescreener.com/",
}

_SESSION = requests.Session()
_SESSION.headers.update(_HEADERS)

# ---------------------------------------------------------------------------
# klsescreener.com — primary source
# ---------------------------------------------------------------------------

def _scrape_klsescreener_page(url: str, timeout: int = 20) -> list:
    """Parse a klsescreener.com table page into a list of stock dicts."""
    try:
        resp = _SESSION.get(url, timeout=timeout)
        resp.raise_for_status()
    except Exception as e:
        logger.warning(f"klsescreener fetch failed ({url}): {e}")
        return []

    soup = BeautifulSoup(resp.text, "lxml")

    # They render a DataTable — rows have class 'stock-row' or are plain <tr>
    table = soup.find("table", {"id": "screener-content"}) or soup.find("table")
    if not table:
        logger.warning("klsescreener: no table found in response")
        return []

    headers = []
    header_row = table.find("thead")
    if header_row:
        headers = [th.get_text(strip=True).lower() for th in header_row.find_all("th")]

    rows = []
    for tr in table.find("tbody").find_all("tr") if table.find("tbody") else table.find_all("tr")[1:]:
        cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
        if not cells or len(cells) < 3:
            continue
        row = dict(zip(headers, cells)) if headers else {}

        # Try to extract symbol from first cell link
        first_link = tr.find("a")
        symbol_raw = ""
        if first_link:
            href = first_link.get("href", "")
            # href format: /v2/stocks/view/1155 or /v2/stocks/1155
            parts = [p for p in href.split("/") if p.isdigit()]
            symbol_raw = parts[-1] if parts else cells[0]
        else:
            symbol_raw = cells[0].split()[0] if cells else ""

        if not symbol_raw:
            continue

        # Normalise to Yahoo Finance .KL format
        ticker = symbol_raw if symbol_raw.endswith(".KL") else f"{symbol_raw}.KL"
        # Handle stapled securities like "5235SS"
        if not symbol_raw.replace("SS", "").replace("WB", "").isdigit():
            # Skip if we can't resolve it cleanly
            pass

        stock = {
            "symbol":      ticker,
            "bursa_code":  symbol_raw,
            "name":        row.get("name", row.get("company", cells[1] if len(cells) > 1 else "")),
            "price":       _safe_float(row.get("price", cells[2] if len(cells) > 2 else "")),
            "chg_pct":     _safe_float(row.get("chg%", row.get("change%", ""))),
            "market_cap":  _safe_float(row.get("mkt cap", row.get("market cap", "")).replace("B","").replace("M","")),
            "pe":          _safe_float(row.get("p/e", row.get("pe", ""))),
            "eps":         _safe_float(row.get("eps", "")),
            "dy_pct":      _safe_float(row.get("dy%", row.get("div yield", ""))),
            "roe_pct":     _safe_float(row.get("roe%", row.get("roe", ""))),
            "sector":      row.get("sector", ""),
        }
        rows.append(stock)

    logger.info(f"klsescreener scraped {len(rows)} stocks from {url}")
    return rows


def _scrape_klsescreener_index(index: str = "FBMKLCI") -> list:
    """Attempt to get KLCI constituents via klsescreener index page."""
    urls = [
        f"https://klsescreener.com/v2/screener/index/{index}",
        f"https://klsescreener.com/v2/indices/{index}",
        f"https://klsescreener.com/v2/screener/",
    ]
    for url in urls:
        rows = _scrape_klsescreener_page(url)
        if rows:
            return rows
        time.sleep(0.5)
    return []


# ---------------------------------------------------------------------------
# i3investor.com — fallback source
# ---------------------------------------------------------------------------

def _scrape_i3investor_klci() -> list:
    """Scrape i3investor for KLCI component list."""
    url = "https://klse.i3investor.com/web/stkstat/list/KLCI"
    try:
        resp = _SESSION.get(url, timeout=20)
        resp.raise_for_status()
    except Exception as e:
        logger.warning(f"i3investor fetch failed: {e}")
        return []

    soup = BeautifulSoup(resp.text, "lxml")
    rows = []
    for tr in soup.select("table tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(cells) < 3:
            continue
        code = cells[0].strip()
        if not code or not code[0].isdigit():
            continue
        rows.append({
            "symbol":     f"{code}.KL",
            "bursa_code": code,
            "name":       cells[1].strip() if len(cells) > 1 else "",
            "price":      _safe_float(cells[2]) if len(cells) > 2 else None,
            "sector":     cells[-1].strip() if cells else "",
        })
    logger.info(f"i3investor scraped {len(rows)} KLCI stocks")
    return rows


# ---------------------------------------------------------------------------
# yfinance info enrichment
# ---------------------------------------------------------------------------

def enrich_with_yfinance(stocks: list, max_stocks: int = 30) -> list:
    """Add live P/E, market cap, sector from yfinance for missing data."""
    try:
        import yfinance as yf
    except ImportError:
        return stocks

    enriched = []
    for s in stocks[:max_stocks]:
        if s.get("pe") and s.get("sector"):
            enriched.append(s)
            continue
        try:
            info = yf.Ticker(s["symbol"]).fast_info
            s.setdefault("price",      getattr(info, "last_price", None))
            s.setdefault("market_cap", getattr(info, "market_cap", None))
        except Exception:
            pass
        enriched.append(s)
        time.sleep(0.1)
    return enriched


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_klci_constituents(use_cache: bool = True, enrich: bool = False) -> list:
    """
    Return the FBM KLCI top-30 constituent list with fundamental data.

    Priority:
      1. Live scrape from klsescreener.com
      2. Fallback to i3investor.com
      3. Fallback to hardcoded KLCI_STOCKS from settings
    """
    from config.settings import KLCI_STOCKS

    # Try live scrape
    rows = _scrape_klsescreener_index("FBMKLCI")
    if not rows:
        rows = _scrape_i3investor_klci()

    if rows:
        # Merge scraped data with hardcoded list to fill in missing fields
        scraped_by_code = {r["bursa_code"]: r for r in rows}
        merged = []
        for stock in KLCI_STOCKS:
            code = stock["bursa_code"]
            live  = scraped_by_code.get(code, {})
            merged.append({**stock, **{k: v for k, v in live.items() if v}})
        if enrich:
            merged = enrich_with_yfinance(merged)
        return merged

    # Hardcoded fallback — always works
    logger.warning("All scrape attempts failed — using hardcoded KLCI_STOCKS")
    return list(KLCI_STOCKS)


def screen_stocks(
    min_pe: Optional[float] = None,
    max_pe: Optional[float] = None,
    min_dy: Optional[float] = None,
    min_roe: Optional[float] = None,
    sector: Optional[str] = None,
    limit: int = 20,
) -> list:
    """
    Filter KLCI stocks by fundamental criteria.
    Returns list of matching stocks, sorted by market cap desc.
    """
    stocks = get_klci_constituents(enrich=True)
    results = []
    for s in stocks:
        if min_pe  is not None and (s.get("pe")  or 0) < min_pe:  continue
        if max_pe  is not None and (s.get("pe")  or 999) > max_pe: continue
        if min_dy  is not None and (s.get("dy_pct") or 0) < min_dy:  continue
        if min_roe is not None and (s.get("roe_pct") or 0) < min_roe: continue
        if sector  and sector.lower() not in (s.get("sector") or "").lower(): continue
        results.append(s)
    return sorted(results, key=lambda x: x.get("market_cap") or 0, reverse=True)[:limit]


def get_sector_breakdown() -> dict:
    """Return count of KLCI stocks by sector."""
    from collections import Counter
    stocks = get_klci_constituents()
    return dict(Counter(s.get("sector", "Unknown") for s in stocks))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(val: str) -> Optional[float]:
    if not val:
        return None
    clean = val.replace(",", "").replace("%", "").replace("RM", "").strip()
    if not clean or clean in ("-", "N/A", "—"):
        return None
    try:
        return float(clean)
    except ValueError:
        return None
