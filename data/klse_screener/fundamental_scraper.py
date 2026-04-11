"""
KLSEFundamentalScraper — single-fetch fundamental data for KLCI stocks.

One GET to /stocks/view/{code}/{slug} retrieves ALL tab content (quarterly reports,
dividends, financials). Tabs are CSS show/hide — no AJAX calls needed.
"""
import logging
import time
from datetime import date

from dateutil import parser as dateutil_parser

from data.klse_screener.client import fetch_page, log_daemon, SLUG_MAP

logger = logging.getLogger(__name__)

# SLUG_MAP is now canonical in client.py — imported above.
# Kept as a local re-export for backward compatibility with any direct imports.
SLUG_MAP = SLUG_MAP


def _parse_date(text: str) -> str | None:
    """Parse a date string like '16 Mar 2026' → '2026-03-16'. Returns None on failure."""
    if not text or not text.strip() or text.strip() == "-":
        return None
    try:
        return dateutil_parser.parse(text.strip(), dayfirst=True).strftime("%Y-%m-%d")
    except Exception:
        return None


class KLSEFundamentalScraper:
    """Scrapes fundamental, quarterly, and dividend data from klsescreener.com."""

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _safe_float(text) -> float | None:
        """Convert text to float, stripping units (%, B, M, sen)."""
        if not text:
            return None
        text = str(text).strip().replace(",", "")
        for unit in ("%", "B", "b", "M", "m", "sen", "RM", "rm"):
            text = text.replace(unit, "")
        text = text.strip()
        if not text or text in ("-", "N/A", "n/a", "--"):
            return None
        try:
            return float(text)
        except ValueError:
            return None

    @staticmethod
    def _get_text(cell) -> str:
        """Extract clean text from a BS4 element."""
        return cell.get_text(separator=" ", strip=True) if cell else ""

    # ── Main entry point ──────────────────────────────────────────────────────

    def fetch_all(self, ticker: str) -> dict:
        """
        Single HTTP fetch for ALL data: fundamentals, quarterly history, dividend history.

        Returns:
            {
                "ticker": str,
                "fundamentals": dict,
                "quarterly_history": list[dict],
                "dividend_history": list[dict],
                "fetched_at": str (ISO date),
            }
        """
        code = ticker.replace(".KL", "")
        slug = SLUG_MAP.get(code)
        if not slug:
            return {"error": f"Unknown ticker {ticker} — add to SLUG_MAP"}

        soup = fetch_page(f"/stocks/view/{code}/{slug}")
        if not soup:
            return {"error": f"Fetch failed for {ticker}"}

        fundamentals = self._parse_summary(soup, ticker)
        quarterly = self._parse_quarterly(soup)
        dividends = self._parse_dividends(soup)

        return {
            "ticker": ticker,
            "fundamentals": fundamentals,
            "quarterly_history": quarterly,
            "dividend_history": dividends,
            "fetched_at": date.today().isoformat(),
        }

    # ── Section parsers ───────────────────────────────────────────────────────

    def _parse_summary(self, soup, ticker: str) -> dict:
        """Parse the summary sidebar table and technical indicators section."""
        result = {
            "ticker": ticker,
            "name": None,
            "price": None,
            "dy": None,
            "dps_ttm": None,
            "eps_ttm": None,
            "pe": None,
            "pb": None,
            "roe": None,
            "nta": None,
            "rsi_14": None,
            "stoch_14": None,
            "market_cap_b": None,
        }

        # ── Stock name ────────────────────────────────────────────────────────
        # Try <title> first: "CIMB: CIMB GROUP HOLDINGS BERHAD | KLSE ..."
        title_tag = soup.find("title")
        if title_tag:
            title_text = title_tag.get_text(strip=True)
            if ":" in title_text:
                result["name"] = title_text.split(":", 1)[1].split("|")[0].strip()
        # Fallback: <h1> or <h2>
        if not result["name"]:
            for tag in ("h1", "h2"):
                el = soup.find(tag)
                if el:
                    result["name"] = el.get_text(strip=True)[:80]
                    break

        # ── Price ─────────────────────────────────────────────────────────────
        # klsescreener shows price in a large element with class containing 'price'
        # or inside a dedicated span/div
        for selector in (
            {"class": "price"},
            {"id": "price"},
            {"class": "stock-price"},
            {"class": "last-price"},
        ):
            el = soup.find(attrs=selector)
            if el:
                result["price"] = self._safe_float(el.get_text(strip=True))
                if result["price"]:
                    break

        # ── Summary table — build label→value dict ────────────────────────────
        label_map: dict[str, str] = {}

        # Strategy: walk all <tr> in the page; if a row has exactly 2 <td> cells
        # the first is likely a label and the second a value
        for tr in soup.find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) == 2:
                lbl = self._get_text(tds[0]).strip().rstrip(":")
                val = self._get_text(tds[1]).strip()
                if lbl:
                    label_map[lbl] = val

        # Also handle <th> labels paired with <td> values
        for tr in soup.find_all("tr"):
            ths = tr.find_all("th")
            tds = tr.find_all("td")
            if len(ths) >= 1 and len(tds) >= 1:
                lbl = self._get_text(ths[0]).strip().rstrip(":")
                val = self._get_text(tds[0]).strip()
                if lbl:
                    label_map.setdefault(lbl, val)

        # Map known labels to result fields
        _FIELD_MAP = {
            "ROE":        ("roe",          False),
            "P/E":        ("pe",           False),
            "PE":         ("pe",           False),
            "EPS":        ("eps_ttm",      False),
            "DPS":        ("dps_ttm",      False),
            "DY":         ("dy",           True),   # True = strip % sign
            "NTA":        ("nta",          False),
            "P/B":        ("pb",           False),
            "PB":         ("pb",           False),
            "PTBV":       ("pb",           False),
            "Market Cap": ("market_cap_b", False),
            "Cap":        ("market_cap_b", False),
        }
        for label, (field, _strip_pct) in _FIELD_MAP.items():
            if label in label_map and result[field] is None:
                result[field] = self._safe_float(label_map[label])

        # ── Technical indicators ──────────────────────────────────────────────
        # RSI row structure: <td>RSI(14)</td>
        #                    <td><span class="ta ...">Neutral</span>  38.4</td>
        # The numeric value is a text node AFTER the span inside the second cell.
        import re as _re
        for tr in soup.find_all("tr"):
            cells = tr.find_all("td")
            if len(cells) >= 2:
                label = self._get_text(cells[0]).strip()
                if label not in ("RSI(14)", "Stochastic(14)"):
                    continue
                value_cell = cells[-1]
                # Remove any <span> children and read the remaining text
                for span in value_cell.find_all("span"):
                    span.decompose()
                raw_val = value_cell.get_text(strip=True)
                # Extract trailing number (e.g. "38.4" from "Neutral38.4" or just "38.4")
                m = _re.search(r"[\d]+\.?[\d]*$", raw_val)
                v = self._safe_float(m.group()) if m else None
                if label == "RSI(14)" and result["rsi_14"] is None:
                    result["rsi_14"] = v
                elif label == "Stochastic(14)" and result["stoch_14"] is None:
                    result["stoch_14"] = v

        return result

    def _parse_quarterly(self, soup) -> list:
        """Parse the Quarter Reports tab.

        Returns list of quarterly records (up to 20), sorted by q_date descending.
        """
        records = []

        # Find the quarterly table by looking for a table whose headers contain
        # at least 3 of: EPS, DPS, NTA, Revenue, Quarter, ROE, QoQ, YoY
        QUARTERLY_HEADERS = {"EPS", "DPS", "NTA", "Revenue", "Quarter", "ROE", "QoQ", "YoY"}

        target_table = None
        for table in soup.find_all("table"):
            headers = {th.get_text(strip=True) for th in table.find_all("th")}
            # Also check first row cells for column headers
            first_row = table.find("tr")
            if first_row:
                headers |= {td.get_text(strip=True) for td in first_row.find_all("td")}
            if len(headers & QUARTERLY_HEADERS) >= 3:
                target_table = table
                break

        # Fallback: look near the #quarter_reports anchor
        if not target_table:
            anchor = soup.find("a", {"name": "quarter_reports"}) or soup.find(
                "div", {"id": "quarter_reports"}
            )
            if anchor:
                target_table = anchor.find_next("table")

        if not target_table:
            return records

        # Build column index from header row(s)
        col_names = []
        header_rows = target_table.find_all("tr")
        for hr in header_rows:
            cells = hr.find_all(["th", "td"])
            texts = [self._get_text(c) for c in cells]
            # A header row has recognisable column names
            if any(t in QUARTERLY_HEADERS for t in texts):
                col_names = texts
                break

        if not col_names:
            # Fallback: use first row
            first = target_table.find("tr")
            if first:
                col_names = [self._get_text(c) for c in first.find_all(["th", "td"])]

        # Map column index for known fields
        def _col(name: str) -> int | None:
            for aliases in (
                [name],
                [name.upper()],
                [name.lower()],
            ):
                for alias in aliases:
                    try:
                        return col_names.index(alias)
                    except ValueError:
                        pass
            # Partial match
            for i, cn in enumerate(col_names):
                if name.lower() in cn.lower():
                    return i
            return None

        idx = {
            "eps":            _col("EPS"),
            "dps":            _col("DPS"),
            "nta":            _col("NTA"),
            "revenue":        _col("Revenue") or _col("Rev"),
            "pl":             _col("P/L") or _col("PL") or _col("Net Profit") or _col("Profit"),
            "quarter":        _col("Quarter") or _col("Q"),
            "q_date":         _col("Q Date") or _col("Qdate") or _col("Date"),
            "financial_year": _col("Financial Year") or _col("FY"),
            "announced":      _col("Announced") or _col("Announce"),
            "roe":            _col("ROE"),
            "qoq_pct":        _col("QoQ%") or _col("QoQ"),
            "yoy_pct":        _col("YoY%") or _col("YoY"),
        }

        for tr in target_table.find_all("tr"):
            cells = tr.find_all("td")
            if not cells:
                continue

            # Skip year-header rows (first cell spans multiple columns)
            first_td = cells[0]
            if first_td.get("colspan") and int(first_td.get("colspan", 1)) > 1:
                continue
            # Skip rows that look like column headers
            if len(cells) < 5:
                continue

            def _cell(key: str) -> str:
                i = idx.get(key)
                if i is None or i >= len(cells):
                    return ""
                return self._get_text(cells[i])

            q_date_raw = _cell("q_date")
            q_date = _parse_date(q_date_raw) or q_date_raw

            rec = {
                "eps":            self._safe_float(_cell("eps")),
                "dps":            self._safe_float(_cell("dps")),
                "nta":            self._safe_float(_cell("nta")),
                "revenue":        _cell("revenue") or None,
                "pl":             _cell("pl") or None,
                "quarter":        self._safe_float(_cell("quarter")),
                "q_date":         q_date or None,
                "financial_year": _cell("financial_year") or None,
                "announced":      _parse_date(_cell("announced")) or _cell("announced") or None,
                "roe":            self._safe_float(_cell("roe")),
                "qoq_pct":        self._safe_float(_cell("qoq_pct")),
                "yoy_pct":        self._safe_float(_cell("yoy_pct")),
            }

            # Skip rows where core fields are all None (blank rows)
            if not any([rec["eps"], rec["dps"], rec["q_date"]]):
                continue

            if isinstance(rec["quarter"], float):
                rec["quarter"] = int(rec["quarter"])

            records.append(rec)

        # Sort by q_date descending, return last 20 quarters
        records.sort(key=lambda r: r.get("q_date") or "", reverse=True)
        return records[:20]

    def _parse_dividends(self, soup) -> list:
        """Parse the Dividends tab.

        Returns list of dividend records, sorted by ex_date descending.
        """
        records = []

        DIV_HEADERS = {"Announced", "Subject", "EX Date", "Payment Date", "Amount"}

        target_table = None
        for table in soup.find_all("table"):
            headers = {th.get_text(strip=True) for th in table.find_all("th")}
            first_row = table.find("tr")
            if first_row:
                headers |= {td.get_text(strip=True) for td in first_row.find_all("td")}
            if len(headers & DIV_HEADERS) >= 3:
                target_table = table
                break

        # Fallback: near #dividends anchor
        if not target_table:
            anchor = soup.find("a", {"name": "dividends"}) or soup.find(
                "div", {"id": "dividends"}
            )
            if anchor:
                target_table = anchor.find_next("table")

        if not target_table:
            return records

        # Build column index
        col_names = []
        for hr in target_table.find_all("tr"):
            cells = hr.find_all(["th", "td"])
            texts = [self._get_text(c) for c in cells]
            if any(t in DIV_HEADERS for t in texts):
                col_names = texts
                break

        def _col(name: str) -> int | None:
            for i, cn in enumerate(col_names):
                if name.lower() == cn.lower() or name.lower() in cn.lower():
                    return i
            return None

        idx = {
            "announced":      _col("Announced"),
            "financial_year": _col("Financial Year") or _col("FY"),
            "subject":        _col("Subject"),
            "ex_date":        _col("EX Date") or _col("Ex Date") or _col("ExDate"),
            "payment_date":   _col("Payment Date") or _col("Pay Date"),
            "amount":         _col("Amount"),
            "indicator":      _col("Indicator") or _col("Type"),
        }

        for tr in target_table.find_all("tr"):
            cells = tr.find_all("td")
            if not cells or len(cells) < 4:
                continue

            def _cell(key: str) -> str:
                i = idx.get(key)
                if i is None or i >= len(cells):
                    return ""
                return self._get_text(cells[i])

            amount_raw = _cell("amount")
            amount_f = self._safe_float(amount_raw)
            if amount_f is None:
                continue  # skip non-data rows

            subject = _cell("subject")
            if "interim" in subject.lower():
                dtype = "interim"
            elif "special" in subject.lower():
                dtype = "special"
            elif "final" in subject.lower():
                dtype = "final"
            else:
                dtype = "other"

            ex_date = _parse_date(_cell("ex_date"))
            if not ex_date:
                continue

            rec = {
                "announced":      _parse_date(_cell("announced")) or _cell("announced") or None,
                "financial_year": _cell("financial_year") or None,
                "subject":        subject or None,
                "ex_date":        ex_date,
                "payment_date":   _parse_date(_cell("payment_date")) or None,
                "dps_sen":        round(amount_f * 100, 4),   # MYR → sen
                "dividend_type":  dtype,
            }
            records.append(rec)

        records.sort(key=lambda r: r.get("ex_date") or "", reverse=True)
        return records

    # ── Bulk refresh ──────────────────────────────────────────────────────────

    def refresh_all_klci(self) -> dict:
        """Fetch and upsert fundamental/quarterly/dividend data for all SLUG_MAP stocks.

        Upserts into:  fundamental_data, quarterly_history, dividend_history tables.
        Returns:  {success: N, failed: N, stocks: [...]}
        """
        from data.database import db_session

        success, failed = 0, 0
        stocks_done = []

        for code, slug in SLUG_MAP.items():
            ticker = f"{code}.KL"
            try:
                data = self.fetch_all(ticker)
                if "error" in data:
                    logger.warning(f"[FundScraper] {ticker}: {data['error']}")
                    failed += 1
                    stocks_done.append({"ticker": ticker, "ok": False})
                    time.sleep(2)
                    continue

                fetched_at = data["fetched_at"]
                fund = data["fundamentals"]

                with db_session() as conn:
                    # Upsert fundamental_data
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO fundamental_data
                          (ticker, name, price, dy, dps_ttm, eps_ttm, pe, pb, roe, nta,
                           rsi_14, stoch_14, market_cap_b, fetched_at)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            ticker,
                            fund.get("name"),
                            fund.get("price"),
                            fund.get("dy"),
                            fund.get("dps_ttm"),
                            fund.get("eps_ttm"),
                            fund.get("pe"),
                            fund.get("pb"),
                            fund.get("roe"),
                            fund.get("nta"),
                            fund.get("rsi_14"),
                            fund.get("stoch_14"),
                            fund.get("market_cap_b"),
                            fetched_at,
                        ),
                    )

                    # Upsert quarterly rows
                    for q in data["quarterly_history"]:
                        if not q.get("q_date"):
                            continue
                        conn.execute(
                            """
                            INSERT OR IGNORE INTO quarterly_history
                              (ticker, q_date, quarter, financial_year, announced,
                               eps, dps, nta, revenue, pl, roe, qoq_pct, yoy_pct)
                            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                            """,
                            (
                                ticker,
                                q["q_date"],
                                q.get("quarter"),
                                q.get("financial_year"),
                                q.get("announced"),
                                q.get("eps"),
                                q.get("dps"),
                                q.get("nta"),
                                q.get("revenue"),
                                q.get("pl"),
                                q.get("roe"),
                                q.get("qoq_pct"),
                                q.get("yoy_pct"),
                            ),
                        )

                    # Upsert dividend rows
                    for d in data["dividend_history"]:
                        if not d.get("ex_date"):
                            continue
                        conn.execute(
                            """
                            INSERT OR IGNORE INTO dividend_history
                              (ticker, ex_date, payment_date, announced, financial_year,
                               subject, dps_sen, dividend_type)
                            VALUES (?,?,?,?,?,?,?,?)
                            """,
                            (
                                ticker,
                                d["ex_date"],
                                d.get("payment_date"),
                                d.get("announced"),
                                d.get("financial_year"),
                                d.get("subject"),
                                d.get("dps_sen"),
                                d.get("dividend_type"),
                            ),
                        )

                success += 1
                stocks_done.append({"ticker": ticker, "ok": True})
                log_daemon(
                    "INFO",
                    "FundScraper",
                    f"{ticker} OK — Q:{len(data['quarterly_history'])} D:{len(data['dividend_history'])}",
                )

            except Exception as e:
                logger.error(f"[FundScraper] {ticker} exception: {e}", exc_info=True)
                failed += 1
                stocks_done.append({"ticker": ticker, "ok": False})

            time.sleep(2)

        summary = f"refresh_all_klci: {success} ok / {failed} failed"
        log_daemon("INFO", "FundScraper", summary)
        return {"success": success, "failed": failed, "stocks": stocks_done}
