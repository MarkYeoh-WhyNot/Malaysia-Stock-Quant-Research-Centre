"""
KLSE Fundamental Scanner — screens KLCI stocks by value, momentum,
dividend calendar, and earnings calendar criteria.
"""
import logging
import time
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


class FundamentalScanner:
    """
    Screens FBM KLCI stocks using a combination of live fundamental data
    (from KLSEScreener) and Yahoo Finance price history.
    """

    def __init__(self):
        # Lazy-import to avoid circular deps at module load time
        from data.klse.screener import get_klci_constituents, screen_stocks
        from data.yahoo.client import get_multi_prices
        self._get_constituents  = get_klci_constituents
        self._screen_stocks     = screen_stocks
        self._get_multi_prices  = get_multi_prices

    # ------------------------------------------------------------------
    # 1. Value stocks
    # ------------------------------------------------------------------

    def scan_value_stocks(
        self,
        max_pe: float = 15.0,
        min_roe: float = 10.0,
        min_dy: float = 3.0,
    ) -> list:
        """
        Return KLCI stocks meeting value criteria: low P/E, high ROE, decent yield.

        Returns list of dicts enriched with pe, roe_pct, dy_pct, price.
        """
        results = self._screen_stocks(max_pe=max_pe, min_dy=min_dy, min_roe=min_roe)
        logger.info(
            f"FundamentalScanner.scan_value_stocks: {len(results)} stocks "
            f"(PE≤{max_pe}, ROE≥{min_roe}%, DY≥{min_dy}%)"
        )
        return results

    # ------------------------------------------------------------------
    # 2. Momentum stocks
    # ------------------------------------------------------------------

    def scan_momentum_stocks(self, lookback_days: int = 20) -> list:
        """
        Find KLCI stocks with the strongest price momentum over the last
        `lookback_days` trading days.

        Returns top-10 stocks sorted by % gain, each dict includes:
          symbol, name, sector, momentum_pct, start_price, end_price
        """
        stocks = self._get_constituents()
        symbols = [s["symbol"] for s in stocks]
        name_map = {s["symbol"]: s for s in stocks}

        momentum_rows = []
        # Fetch in bulk using module-level get_multi_prices
        try:
            # period that covers lookback_days * 1.4 to account for weekends/holidays
            calendar_days = int(lookback_days * 1.5) + 5
            prices = self._get_multi_prices(
                symbols,
                period=f"{calendar_days}d",
            )
        except Exception as e:
            logger.warning(f"Momentum scan bulk fetch failed: {e}")
            return []

        for symbol, df in prices.items():
            if df is None or df.empty or "close" not in df.columns:
                continue
            try:
                # Use last `lookback_days` rows
                df = df.sort_index()
                if len(df) < 5:
                    continue
                slice_df    = df.tail(lookback_days)
                start_price = float(slice_df["close"].iloc[0])
                end_price   = float(slice_df["close"].iloc[-1])
                if start_price <= 0:
                    continue
                pct = (end_price - start_price) / start_price * 100
                meta = name_map.get(symbol, {})
                momentum_rows.append({
                    "symbol":       symbol,
                    "name":         meta.get("name", symbol),
                    "sector":       meta.get("sector", ""),
                    "momentum_pct": round(pct, 2),
                    "start_price":  round(start_price, 4),
                    "end_price":    round(end_price, 4),
                })
            except Exception as e:
                logger.debug(f"Momentum calc error for {symbol}: {e}")

        top10 = sorted(momentum_rows, key=lambda x: x["momentum_pct"], reverse=True)[:10]
        logger.info(f"FundamentalScanner.scan_momentum_stocks: top-10 of {len(momentum_rows)} stocks")
        return top10

    # ------------------------------------------------------------------
    # 3. Dividend calendar
    # ------------------------------------------------------------------

    def scan_dividend_calendar(self, days_ahead: int = 14) -> list:
        """
        Find KLCI stocks with an ex-dividend date within the next `days_ahead` days.

        Uses yfinance calendar data where available, then enriches with
        i3investor dividend announcements as a secondary source.

        Returns list of dicts:
          symbol, name, ex_date, dividend_amount, current_yield_pct
        """
        results = []
        stocks  = self._get_constituents()
        cutoff  = datetime.utcnow() + timedelta(days=days_ahead)

        for s in stocks:
            symbol = s["symbol"]
            try:
                import yfinance as yf
                ticker = yf.Ticker(symbol)

                # yfinance returns calendar as a DataFrame or dict
                cal = ticker.calendar
                if cal is None:
                    continue

                # Newer yfinance: calendar is a dict with 'Ex-Dividend Date' key
                ex_date_raw = None
                if isinstance(cal, dict):
                    ex_date_raw = cal.get("Ex-Dividend Date") or cal.get("exDividendDate")
                elif hasattr(cal, "T"):
                    # Older format: DataFrame transposed
                    row = cal.T
                    if "Ex-Dividend Date" in row.columns:
                        ex_date_raw = row["Ex-Dividend Date"].iloc[0]

                if not ex_date_raw:
                    continue

                # Normalise to datetime
                if isinstance(ex_date_raw, (int, float)):
                    ex_dt = datetime.utcfromtimestamp(ex_date_raw)
                elif hasattr(ex_date_raw, "to_pydatetime"):
                    ex_dt = ex_date_raw.to_pydatetime()
                elif isinstance(ex_date_raw, datetime):
                    ex_dt = ex_date_raw
                else:
                    ex_dt = datetime.strptime(str(ex_date_raw)[:10], "%Y-%m-%d")

                if ex_dt > datetime.utcnow() and ex_dt <= cutoff:
                    # Get dividend amount and yield from fast_info
                    div_amount = 0.0
                    curr_yield = 0.0
                    try:
                        info = ticker.fast_info
                        div_amount = getattr(info, "last_dividend_value", 0.0) or 0.0
                        curr_yield = getattr(info, "dividend_yield", 0.0) or 0.0
                        if curr_yield and curr_yield < 1:
                            curr_yield = round(curr_yield * 100, 2)
                    except Exception:
                        pass

                    results.append({
                        "symbol":           symbol,
                        "name":             s.get("name", symbol),
                        "sector":           s.get("sector", ""),
                        "ex_date":          ex_dt.strftime("%Y-%m-%d"),
                        "dividend_amount":  round(div_amount, 4),
                        "current_yield_pct": curr_yield,
                    })

                time.sleep(0.2)  # Be respectful with yfinance calls

            except Exception as e:
                logger.debug(f"Dividend calendar error for {symbol}: {e}")

        results.sort(key=lambda x: x["ex_date"])
        logger.info(f"FundamentalScanner.scan_dividend_calendar: {len(results)} stocks with ex-date in next {days_ahead}d")
        return results

    # ------------------------------------------------------------------
    # 4. Earnings calendar
    # ------------------------------------------------------------------

    def scan_earnings_calendar(self) -> list:
        """
        Find KLCI stocks announcing results in the next 30 days.

        Attempts yfinance earnings dates first, then falls back to
        klsescreener.com fundamental data (uses last_eps as a proxy).

        Returns list of dicts:
          symbol, name, expected_date, last_eps, sector
        """
        results = []
        stocks  = self._get_constituents()
        horizon = datetime.utcnow() + timedelta(days=30)

        for s in stocks:
            symbol = s["symbol"]
            try:
                import yfinance as yf
                ticker = yf.Ticker(symbol)

                # Try earnings_dates DataFrame (yfinance ≥ 0.2)
                try:
                    ed = ticker.earnings_dates
                    if ed is not None and not ed.empty:
                        upcoming = ed[ed.index > datetime.utcnow()]
                        upcoming = upcoming[upcoming.index <= horizon]
                        if not upcoming.empty:
                            date_str = upcoming.index[0].strftime("%Y-%m-%d")
                            # EPS estimate column varies by yfinance version
                            eps_col  = [c for c in upcoming.columns if "eps" in c.lower() or "estimate" in c.lower()]
                            last_eps = float(upcoming[eps_col[0]].iloc[0]) if eps_col else s.get("eps")
                            results.append({
                                "symbol":        symbol,
                                "name":          s.get("name", symbol),
                                "sector":        s.get("sector", ""),
                                "expected_date": date_str,
                                "last_eps":      last_eps,
                            })
                except Exception:
                    pass  # No earnings dates — skip silently

                time.sleep(0.15)

            except Exception as e:
                logger.debug(f"Earnings calendar error for {symbol}: {e}")

        results.sort(key=lambda x: x.get("expected_date") or "9999")
        logger.info(f"FundamentalScanner.scan_earnings_calendar: {len(results)} upcoming earnings")
        return results
