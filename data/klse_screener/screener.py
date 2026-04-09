"""
KLSEProactiveScreener — runs predefined fundamental + technical screens against
klsescreener.com and returns matching stocks for each screen.
"""
import logging
import time

from data.klse_screener.client import fetch_page, post_screener, log_daemon

logger = logging.getLogger(__name__)


class KLSEProactiveScreener:
    """
    Submits GET requests to https://www.klsescreener.com/v2/ with filter parameters.
    The page returns a full HTML response containing a results table.

    Column order (confirmed from screenshots):
    Name | Code | Category | Price | Change | Change% | 52w | Volume |
    EPS | DPS | NTA | PE | DY | ROE | PTBV | Cap | Indicators
    """

    # Pre-defined screens — field names match klsescreener.com POST form exactly.
    # Endpoint: POST /v2/screener/quote_results  (getquote=1 always added by post_screener)
    # Sector IDs (Main Market): Plantation=10, Financial Services=7
    # Subsector IDs: Banking=27
    SCREENS: dict = {
        "high_yield_value": {
            "params": {"min_dy": 5, "max_pe": 15, "min_roe": 8, "min_marketcap": 500},
            "description": "High DY + low PE + solid ROE",
            "idea_angle": "dividend_capture",
        },
        "momentum_breakout": {
            "params": {
                "price_gt_sma_50": "50",
                "rsi": "neutral",
                "price_gainer": "1",
                "yoy": "1",
            },
            "description": "Above SMA50, gaining, profitable",
            "idea_angle": "momentum",
        },
        "oversold_quality": {
            "params": {"rsi": "oversold", "min_roe": 10, "max_pe": 20, "min_marketcap": 1000},
            "description": "RSI oversold + quality fundamentals",
            "idea_angle": "mean_reversion",
        },
        "low_pb_reversion": {
            "params": {"max_ptbv": 1.2, "min_roe": 8, "price_gt_sma_50": "50"},
            "description": "P/B below book + ROE + uptrend",
            "idea_angle": "value",
        },
        "earnings_momentum": {
            "params": {
                "yoy": "1",
                "qoq": "1",
                "ryoy": "1",
                "min_marketcap": 300,
            },
            "description": "Improving earnings and revenue",
            "idea_angle": "event_driven",
        },
        "net_cash_defensive": {
            "params": {"netcash": "on", "min_dy": 3, "min_marketcap": 500},
            "description": "Net cash + dividend yield",
            "idea_angle": "defensive",
        },
        "plantation_oversold": {
            "params": {"sector": "10", "rsi": "oversold", "min_dy": 2},
            "description": "Plantation stocks oversold",
            "idea_angle": "cpo_lag",
        },
        "banking_value": {
            "params": {
                "subsector": "27",
                "max_ptbv": 1.3,
                "min_dy": 4,
                "min_roe": 10,
            },
            "description": "Banking below book with yield",
            "idea_angle": "value",
        },
    }

    # Column indices from live /v2/screener/quote_results response (POST)
    # Headers: Name Code Category Price Change Change% 52week Volume EPS DPS NTA PE DY ROE PTBV MCap.(M) Indicators
    _COL_MAP = {
        "name":     0,
        "code":     1,
        "category": 2,
        "price":    3,
        "change":   4,
        "change_pct": 5,
        "52w":      6,
        "volume":   7,
        "eps":      8,
        "dps":      9,
        "nta":      10,
        "pe":       11,
        "dy":       12,
        "roe":      13,
        "pb":       14,
        "cap":      15,
        "indicators": 16,
    }

    @staticmethod
    def _safe_float(text) -> float | None:
        if not text:
            return None
        t = str(text).strip().replace(",", "")
        for unit in ("%", "B", "b", "M", "m", "sen", "RM"):
            t = t.replace(unit, "")
        t = t.strip()
        if not t or t in ("-", "N/A", "n/a", "--", "–"):
            return None
        try:
            return float(t)
        except ValueError:
            return None

    def _discover_form_params(self) -> None:
        """Fetch base screener page and log all form field names at DEBUG level."""
        soup = fetch_page("/")
        if not soup:
            return
        form = soup.find("form")
        if not form:
            # Try finding any form-like element
            forms = soup.find_all("form")
            if forms:
                form = forms[0]
        if not form:
            log_daemon("DEBUG", "KLSEScreener", "No <form> found on screener page")
            return
        fields = []
        for el in form.find_all(["input", "select"]):
            name = el.get("name")
            typ = el.get("type", "select")
            if name:
                fields.append(f"{name}({typ})")
        log_daemon(
            "DEBUG",
            "KLSEScreener",
            f"Form fields discovered: {', '.join(fields[:60])}",
        )

    def run_screen(self, params: dict, screen_name: str) -> list:
        """
        Run a single screen: POST to /v2/screener/quote_results with params dict.

        Returns list of stock dicts with fields:
            ticker, name, price, dy, pe, pb, roe, eps, dps, nta, cap
        """
        time.sleep(2)

        soup, req_url = post_screener(params)

        # ── Debug: log full URL + params ──────────────────────────────────────
        param_str = "&".join(f"{k}={v}" for k, v in params.items())
        log_daemon("DEBUG", "KLSEScreener", f"{screen_name}: POST {req_url} [{param_str}]")

        if not soup:
            log_daemon("WARN", "KLSEScreener", f"{screen_name}: POST returned None")
            return []

        page_text = soup.get_text(separator=" ")

        # ── Check result count ────────────────────────────────────────────────
        count_text = ""
        for line in page_text.splitlines():
            if "stock(s) found" in line or "stocks found" in line.lower():
                count_text = line.strip()
                break

        import re
        m = re.search(r"(\d+)\s+stock", count_text, re.IGNORECASE)
        if count_text:
            log_daemon("DEBUG", "KLSEScreener", f"{screen_name}: {count_text}")
        if m and int(m.group(1)) == 0:
            return []

        # ── Find results table ────────────────────────────────────────────────
        # Identify by headers containing 'Code' and 'DY' and 'Price'
        target_table = None
        for table in soup.find_all("table"):
            header_text = " ".join(
                th.get_text(strip=True) for th in table.find_all("th")
            ).upper()
            if "CODE" in header_text and "PRICE" in header_text:
                target_table = table
                break
            # Also check first row
            first_tr = table.find("tr")
            if first_tr:
                row_text = first_tr.get_text(separator=" ").upper()
                if "CODE" in row_text and "PRICE" in row_text:
                    target_table = table
                    break

        if not target_table:
            html_preview = str(soup)[:500].replace("\n", " ")
            log_daemon(
                "WARN",
                "KLSEScreener",
                f"{screen_name}: results table not found. HTML[0:500]: {html_preview}",
            )
            return []

        # ── Build dynamic column map from header row ──────────────────────────
        col_map = dict(self._COL_MAP)  # start with defaults
        header_cells = []
        first_tr = target_table.find("tr")
        if first_tr:
            header_cells = [
                c.get_text(strip=True).upper()
                for c in first_tr.find_all(["th", "td"])
            ]

        if header_cells:
            _header_field_map = {
                "NAME":      "name",
                "CODE":      "code",
                "PRICE":     "price",
                "EPS":       "eps",
                "DPS":       "dps",
                "NTA":       "nta",
                "PE":        "pe",
                "DY":        "dy",
                "ROE":       "roe",
                "PTBV":      "pb",
                "P/B":       "pb",
                "CAP":       "cap",
                "MCAP.(M)":  "cap",
                "VOLUME":    "volume",
                "CHANGE%":   "change_pct",
                "52WEEK":    "52w",
                "INDICATORS": "indicators",
            }
            for i, hdr in enumerate(header_cells):
                for key, field in _header_field_map.items():
                    if key == hdr.strip():
                        col_map[field] = i
                        break

        # ── Parse data rows ───────────────────────────────────────────────────
        stocks = []
        rows = target_table.find_all("tr")
        # Skip first row (header)
        for tr in rows[1:]:
            cells = tr.find_all("td")
            if not cells or len(cells) < 4:
                continue

            def _cell(field: str) -> str:
                i = col_map.get(field)
                if i is None or i >= len(cells):
                    return ""
                return cells[i].get_text(strip=True)

            code_raw = _cell("code")
            if not code_raw:
                continue
            ticker = f"{code_raw}.KL" if not code_raw.endswith(".KL") else code_raw

            stock = {
                "ticker":     ticker,
                "code":       code_raw,
                "name":       _cell("name"),
                "price":      self._safe_float(_cell("price")),
                "dy":         self._safe_float(_cell("dy")),
                "pe":         self._safe_float(_cell("pe")),
                "pb":         self._safe_float(_cell("pb")),
                "roe":        self._safe_float(_cell("roe")),
                "eps":        self._safe_float(_cell("eps")),
                "dps":        self._safe_float(_cell("dps")),
                "nta":        self._safe_float(_cell("nta")),
                "cap":        self._safe_float(_cell("cap")),
                "screen":     screen_name,
            }
            stocks.append(stock)

        return stocks

    def run_all_screens(self) -> dict:
        """Run all 8 screens sequentially. Returns full results dict."""
        # First run: discover form params (logged at DEBUG level)
        self._discover_form_params()

        results = {}
        for name, config in self.SCREENS.items():
            stocks = self.run_screen(config["params"], name)
            results[name] = {
                "stocks": stocks,
                "count": len(stocks),
                "description": config["description"],
                "idea_angle": config["idea_angle"],
            }
            time.sleep(2)

        total = sum(r["count"] for r in results.values())
        summary = ", ".join(f"{k}={v['count']}" for k, v in results.items())
        log_daemon(
            "INFO",
            "KLSEScreener",
            f"8 screens complete: {total} total matches — {summary}",
        )
        return results
