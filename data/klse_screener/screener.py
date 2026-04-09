"""
KLSEProactiveScreener — runs predefined fundamental + technical screens against
klsescreener.com and returns matching stocks for each screen.
"""
import logging
import time

from data.klse_screener.client import fetch_page, log_daemon

logger = logging.getLogger(__name__)


class KLSEProactiveScreener:
    """
    Submits GET requests to https://www.klsescreener.com/v2/ with filter parameters.
    The page returns a full HTML response containing a results table.

    Column order (confirmed from screenshots):
    Name | Code | Category | Price | Change | Change% | 52w | Volume |
    EPS | DPS | NTA | PE | DY | ROE | PTBV | Cap | Indicators
    """

    # Pre-defined screens — params will be adjusted once form discovery runs
    SCREENS: dict = {
        "high_yield_value": {
            "params": {"dy_min": 5, "pe_max": 15, "roe_min": 8, "market_cap_min": 500},
            "description": "High DY + low PE + solid ROE",
            "idea_angle": "dividend_capture",
        },
        "momentum_breakout": {
            "params": {
                "sma": "SMA50",
                "rsi": "Neutral",
                "price_change": "Gainers",
                "net_profit_yoy": "On",
            },
            "description": "Above SMA50, gaining, profitable",
            "idea_angle": "momentum",
        },
        "oversold_quality": {
            "params": {"rsi": "Oversold", "roe_min": 10, "pe_max": 20, "market_cap_min": 1000},
            "description": "RSI oversold + quality fundamentals",
            "idea_angle": "mean_reversion",
        },
        "low_pb_reversion": {
            "params": {"ptbv_max": 1.2, "roe_min": 8, "sma": "SMA50"},
            "description": "P/B below book + ROE + uptrend",
            "idea_angle": "value",
        },
        "earnings_momentum": {
            "params": {
                "net_profit_yoy": "On",
                "net_profit_qoq": "On",
                "revenue_yoy": "On",
                "market_cap_min": 300,
            },
            "description": "Improving earnings and revenue",
            "idea_angle": "event_driven",
        },
        "net_cash_defensive": {
            "params": {"net_cash": "1", "dy_min": 3, "market_cap_min": 500},
            "description": "Net cash + dividend yield",
            "idea_angle": "defensive",
        },
        "plantation_oversold": {
            "params": {"sector": "Plantation", "rsi": "Oversold", "dy_min": 2},
            "description": "Plantation stocks oversold",
            "idea_angle": "cpo_lag",
        },
        "banking_value": {
            "params": {
                "subsector": "Banking",
                "ptbv_max": 1.3,
                "dy_min": 4,
                "roe_min": 10,
            },
            "description": "Banking below book with yield",
            "idea_angle": "value",
        },
    }

    # Column indices as discovered from live screenshots
    # Order: Name Code Category Price Change Change% 52w Volume EPS DPS NTA PE DY ROE PTBV Cap Indicators
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
        Run a single screen: GET klsescreener.com/v2/ with params dict.

        Returns list of stock dicts with fields:
            ticker, name, price, dy, pe, pb, roe, eps, dps, nta, cap
        """
        time.sleep(2)

        soup = fetch_page("/", params=params)
        if not soup:
            log_daemon("WARN", "KLSEScreener", f"{screen_name}: fetch returned None")
            return []

        page_text = soup.get_text(separator=" ")

        # ── Check result count ────────────────────────────────────────────────
        count_text = ""
        for line in page_text.splitlines():
            if "stock(s) found" in line or "stocks found" in line.lower():
                count_text = line.strip()
                break
        # If "0 stock(s) found" → return early
        import re
        m = re.search(r"(\d+)\s+stock", count_text, re.IGNORECASE)
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
            log_daemon(
                "WARN", "KLSEScreener", f"{screen_name}: results table not found"
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
                "NAME":    "name",
                "CODE":    "code",
                "PRICE":   "price",
                "EPS":     "eps",
                "DPS":     "dps",
                "NTA":     "nta",
                "PE":      "pe",
                "DY":      "dy",
                "ROE":     "roe",
                "PTBV":    "pb",
                "P/B":     "pb",
                "CAP":     "cap",
                "VOLUME":  "volume",
                "CHANGE%": "change_pct",
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
