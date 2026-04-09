"""
MPOBScraper — Crude Palm Oil (CPO) price data fetcher for OpenClaw.

Source priority:
  1. MPOB BEPI official website (Malaysian Palm Oil Board)
  2. cpo.com.my spot price page
  3. yfinance FCPO.KL (Bursa Malaysia CPO futures, best for historical data)

CPO prices are quoted in MYR per metric tonne (e.g. 3800.00 MYR/t).
Plantation stocks on Bursa Malaysia: revenue is largely CPO-denominated,
so CPO price leads plantation stock prices with a 1–5 day lag.
"""
import logging
import re
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup

from data.database import db_session

logger = logging.getLogger(__name__)

# ── Plantation universe ───────────────────────────────────────────────────────
PLANTATION_TICKERS = [
    "5285.KL",  # Sime Darby Plantation
    "1961.KL",  # IOI Corporation
    "2445.KL",  # Kuala Lumpur Kepong (KLK)
    "4065.KL",  # PPB Group
    "5069.KL",  # Hap Seng Plantations
]

PLANTATION_NAMES = {
    "5285.KL": "Sime Darby Plantation",
    "1961.KL": "IOI Corporation",
    "2445.KL": "Kuala Lumpur Kepong",
    "4065.KL": "PPB Group",
    "5069.KL": "Hap Seng Plantations",
}

# Realistic CPO price range guard (MYR/tonne)
_CPO_MIN, _CPO_MAX = 1500.0, 9000.0

# CPO=F is CME palm oil futures, quoted in USD/tonne (~800–1500 USD/t)
# Convert to MYR using an approximate rate (updated periodically)
_USD_TO_MYR_APPROX = 4.47

# yfinance CPO sources: (ticker, currency, myr_multiplier)
# CPO=F: CME palm oil USD/tonne → multiply by ~4.47 for MYR
# PALM.L: WisdomTree Palm Oil ETP on LSE, NOT a direct MYR/t proxy — skip for price
_YF_CPO_SOURCES = [
    ("CPO=F",  "USD", _USD_TO_MYR_APPROX),   # CME palm oil futures, best coverage
]

# Browser-like request headers
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,ms;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1",
    "Connection": "keep-alive",
}


def _is_valid_cpo(price: float) -> bool:
    return _CPO_MIN <= price <= _CPO_MAX


class MPOBScraper:
    """Fetches and caches CPO spot/futures prices; computes plantation lag signals."""

    # ── DB helpers ────────────────────────────────────────────────────────────

    def _ensure_table(self):
        """Create cpo_prices table if absent (idempotent)."""
        with db_session() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cpo_prices (
                    date    TEXT PRIMARY KEY,
                    price   REAL NOT NULL,
                    source  TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_cpo_date ON cpo_prices(date)"
            )

    def _upsert_price(self, date: str, price: float, source: str):
        with db_session() as conn:
            conn.execute(
                "INSERT INTO cpo_prices (date, price, source) VALUES (?, ?, ?) "
                "ON CONFLICT(date) DO UPDATE SET price=excluded.price, source=excluded.source",
                (date, price, source),
            )

    # ── Source 1: MPOB BEPI ───────────────────────────────────────────────────

    def _fetch_mpob_bepi(self) -> Optional[dict]:
        """Scrape MPOB BEPI for today's CPO spot price (official Malaysian gov source)."""
        url = "https://bepi.mpob.gov.my/index.php/en/"
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=20)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            text = soup.get_text(" ", strip=True)

            # MPOB BEPI shows prices in tables; CPO row is typically labelled
            # "Crude Palm Oil" or "CPO" followed by a RM/tonne value
            patterns = [
                r'(?:Crude Palm Oil|CPO)[^0-9\n]{0,40}?(\d{3,5}(?:\.\d{1,2})?)',
                r'CPO[^0-9]{0,20}(\d{4,5}(?:\.\d{1,2})?)',
            ]
            for pat in patterns:
                m = re.search(pat, text, re.IGNORECASE)
                if m:
                    price = float(m.group(1))
                    if _is_valid_cpo(price):
                        return {
                            "date": datetime.utcnow().strftime("%Y-%m-%d"),
                            "price_myr_per_tonne": price,
                            "source": "mpob_bepi",
                        }

            # Table-based search fallback
            for table in soup.find_all("table"):
                for row in table.find_all("tr"):
                    cells = [c.get_text(strip=True) for c in row.find_all(["td", "th"])]
                    for i, cell in enumerate(cells):
                        if re.search(r'\bcpo\b|crude palm', cell, re.IGNORECASE):
                            for j in range(i + 1, min(i + 5, len(cells))):
                                m2 = re.search(r'(\d{3,5}(?:\.\d{1,2})?)', cells[j])
                                if m2:
                                    price = float(m2.group(1))
                                    if _is_valid_cpo(price):
                                        return {
                                            "date": datetime.utcnow().strftime("%Y-%m-%d"),
                                            "price_myr_per_tonne": price,
                                            "source": "mpob_bepi",
                                        }
        except Exception as e:
            logger.debug(f"MPOB BEPI fetch failed: {e}")
        return None

    # ── Source 2: cpo.com.my ──────────────────────────────────────────────────

    def _fetch_cpocommy(self) -> Optional[dict]:
        """Scrape cpo.com.my for current CPO spot price."""
        url = "https://www.cpo.com.my/prices/"
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=20)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            text = soup.get_text(" ", strip=True)

            patterns = [
                r'(?:CPO|palm oil)[^0-9\n]{0,40}?(\d{3,5}(?:\.\d{1,2})?)',
                r'(?:spot|current|today)[^0-9\n]{0,30}?(\d{3,5}(?:\.\d{1,2})?)',
                r'RM\s*(\d{3,5}(?:\.\d{1,2})?)',
            ]
            for pat in patterns:
                m = re.search(pat, text, re.IGNORECASE)
                if m:
                    price = float(m.group(1))
                    if _is_valid_cpo(price):
                        return {
                            "date": datetime.utcnow().strftime("%Y-%m-%d"),
                            "price_myr_per_tonne": price,
                            "source": "cpocommy",
                        }
        except Exception as e:
            logger.debug(f"cpo.com.my fetch failed: {e}")
        return None

    # ── Source 3: yfinance CPO futures (CPO=F, CME, USD/tonne) ───────────────

    def _fetch_yfinance_cpo(self) -> Optional[dict]:
        """Fetch latest CPO price via yfinance.

        Uses CPO=F (CME palm oil futures, USD/tonne) and converts to MYR/tonne
        with _USD_TO_MYR_APPROX. Validation range: 700–2000 USD/t.
        """
        for sym, ccy, multiplier in _YF_CPO_SOURCES:
            try:
                t    = yf.Ticker(sym)
                hist = t.history(period="5d", interval="1d", auto_adjust=True)
                if hist.empty:
                    continue
                hist.columns = [c.lower() for c in hist.columns]
                if "close" not in hist.columns:
                    continue
                if hist.index.tz is not None:
                    hist.index = hist.index.tz_localize(None)
                raw_price = float(hist["close"].iloc[-1])
                date      = hist.index[-1].strftime("%Y-%m-%d")
                # Validate raw USD price — CPO=F realistic range: 700–2000 USD/t
                if not (700 <= raw_price <= 2000):
                    logger.debug(f"yfinance {sym}: raw={raw_price:.2f} out of USD range — skip")
                    continue
                myr_price = round(raw_price * multiplier, 2)
                logger.debug(
                    f"yfinance {sym}: {raw_price:.2f} {ccy}/t → {myr_price:.2f} MYR/t ({date})"
                )
                return {
                    "date":                date,
                    "price_myr_per_tonne": myr_price,
                    "source":             f"yfinance:{sym} ({ccy}→MYR×{multiplier})",
                }
            except Exception as e:
                logger.debug(f"yfinance {sym} failed: {e}")
        return None

    # ── Public: fetch today's price ───────────────────────────────────────────

    def fetch_daily_cpo_price(self) -> dict:
        """Fetch the latest CPO spot price from the first available source.

        Returns:
            {"date": str, "price_myr_per_tonne": float, "source": str}
        Raises:
            RuntimeError if all three sources fail.
        """
        self._ensure_table()

        for fn, label in [
            (self._fetch_mpob_bepi,   "MPOB BEPI"),
            (self._fetch_cpocommy,    "cpo.com.my"),
            (self._fetch_yfinance_cpo, "yfinance"),
        ]:
            try:
                result = fn()
            except Exception as e:
                logger.debug(f"CPO source {label} raised: {e}")
                result = None

            if result:
                logger.info(
                    f"CPO price: {result['price_myr_per_tonne']:.2f} MYR/t "
                    f"({result['date']}) via {result['source']}"
                )
                self._upsert_price(
                    result["date"],
                    result["price_myr_per_tonne"],
                    result["source"],
                )
                return result

        raise RuntimeError(
            "All CPO sources failed (MPOB BEPI, cpo.com.my, yfinance). "
            "Check network connectivity or website structure changes."
        )

    # ── Public: historical data ───────────────────────────────────────────────

    def get_historical_cpo(self, days: int = 365) -> pd.DataFrame:
        """Return a DataFrame of historical CPO prices, fetching from yfinance if needed.

        Stores results in the cpo_prices SQLite table for caching.
        Falls back to cached data if yfinance is unavailable.

        Returns:
            DataFrame with columns: date (datetime), price (float), source (str)
        """
        self._ensure_table()
        cutoff     = datetime.utcnow() - timedelta(days=days)
        cutoff_str = cutoff.strftime("%Y-%m-%d")

        # How much do we already have?
        with db_session() as conn:
            cached = conn.execute(
                "SELECT date, price, source FROM cpo_prices "
                "WHERE date >= ? ORDER BY date",
                (cutoff_str,),
            ).fetchall()

        existing = pd.DataFrame([dict(r) for r in cached]) if cached else pd.DataFrame()
        if not existing.empty:
            existing["date"] = pd.to_datetime(existing["date"])
            # Sufficient if we have at least 80% of expected trading days
            expected_days = int(days * 252 / 365)
            if len(existing) >= expected_days * 0.80:
                logger.info(f"CPO historical: {len(existing)} rows from cache (sufficient)")
                return existing.sort_values("date").reset_index(drop=True)

        # Fetch from yfinance
        logger.info(f"CPO historical: fetching {days}d from yfinance...")
        fetched: list[dict] = []

        for sym, ccy, multiplier in _YF_CPO_SOURCES:
            try:
                t    = yf.Ticker(sym)
                hist = t.history(start=cutoff_str, interval="1d", auto_adjust=True)
                if hist.empty:
                    continue
                hist.columns = [c.lower() for c in hist.columns]
                if "close" not in hist.columns:
                    continue
                if hist.index.tz is not None:
                    hist.index = hist.index.tz_localize(None)

                for dt, row in hist.iterrows():
                    raw_price = float(row["close"])
                    # Validate USD price range for CPO=F (700–2000 USD/t)
                    if 700 <= raw_price <= 2000:
                        myr_price = round(raw_price * multiplier, 2)
                        fetched.append({
                            "date":   dt.strftime("%Y-%m-%d"),
                            "price":  myr_price,
                            "source": f"yfinance:{sym}",
                        })
                if fetched:
                    logger.info(f"CPO historical: {len(fetched)} rows from yfinance {sym}")
                    break
            except Exception as e:
                logger.debug(f"yfinance {sym} history failed: {e}")

        if fetched:
            with db_session() as conn:
                for r in fetched:
                    conn.execute(
                        "INSERT INTO cpo_prices (date, price, source) VALUES (?, ?, ?) "
                        "ON CONFLICT(date) DO UPDATE SET "
                        "price=excluded.price, source=excluded.source",
                        (r["date"], r["price"], r["source"]),
                    )

        # Return everything we have (cached + newly fetched)
        with db_session() as conn:
            rows = conn.execute(
                "SELECT date, price, source FROM cpo_prices "
                "WHERE date >= ? ORDER BY date",
                (cutoff_str,),
            ).fetchall()

        df = pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
        logger.info(f"CPO historical: returning {len(df)} rows")
        return df.reset_index(drop=True)

    # ── Public: lag signal ────────────────────────────────────────────────────

    def compute_lag_signal(self, plantation_ticker: str, max_lag: int = 5) -> dict:
        """Compute Spearman correlation between CPO daily returns and lagged stock returns.

        For each lag k in 1..max_lag, computes:
            rank_corr(CPO_return[t], stock_return[t+k])

        The best lag is the one with the highest |correlation|.
        A signal > 0.30 is considered statistically meaningful on ~250 obs.

        Args:
            plantation_ticker: Yahoo Finance .KL ticker, e.g. "5285.KL"
            max_lag:           Maximum lag days to test

        Returns:
            {
              ticker, best_lag_days, best_lag_corr, is_significant,
              signal_today, predicted_direction, lag_correlations, n_observations
            }
        """
        base = {
            "ticker":              plantation_ticker,
            "best_lag_days":       0,
            "best_lag_corr":       0.0,
            "is_significant":      False,
            "signal_today":        0.0,
            "predicted_direction": "neutral",
            "lag_correlations":    {},
            "n_observations":      0,
        }

        # ── CPO data ──────────────────────────────────────────────────────────
        cpo_df = self.get_historical_cpo(days=400)
        if cpo_df.empty or len(cpo_df) < 30:
            return {**base, "error": "Insufficient CPO price history (need ≥30 rows)"}

        # ── Stock data ────────────────────────────────────────────────────────
        try:
            stock = yf.Ticker(plantation_ticker)
            hist  = stock.history(period="2y", interval="1d", auto_adjust=True)
            if hist.empty:
                raise ValueError(f"No price data returned for {plantation_ticker}")
            hist.columns = [c.lower() for c in hist.columns]
            if hist.index.tz is not None:
                hist.index = hist.index.tz_localize(None)
            stock_series = hist["close"].rename("stock_close")
        except Exception as e:
            return {**base, "error": f"Stock fetch failed for {plantation_ticker}: {e}"}

        # ── Merge on common dates ─────────────────────────────────────────────
        cpo_series = (
            cpo_df.set_index("date")["price"]
            .rename("cpo_price")
        )
        cpo_series.index = pd.to_datetime(cpo_series.index)
        # Average if multiple CPO rows on same date (shouldn't happen, safety net)
        cpo_series = cpo_series.groupby(level=0).mean()

        merged = (
            stock_series.to_frame()
            .join(cpo_series, how="inner")
            .sort_index()
        )

        if len(merged) < 30:
            return {
                **base,
                "error": f"Only {len(merged)} overlapping dates — need ≥30",
            }

        # ── Returns ───────────────────────────────────────────────────────────
        merged["cpo_ret"]   = merged["cpo_price"].pct_change()
        merged["stock_ret"] = merged["stock_close"].pct_change()
        merged.dropna(inplace=True)

        if len(merged) < 20:
            return {**base, "error": "Fewer than 20 clean return rows after dropna"}

        # ── Spearman correlations at each lag ─────────────────────────────────
        # Spearman = Pearson of ranks; computed via pandas without scipy
        best_lag  = 1
        best_corr = 0.0
        lag_corrs: dict[int, float] = {}

        for lag in range(1, max_lag + 1):
            cpo_x    = merged["cpo_ret"].iloc[:-lag]
            stock_y  = merged["stock_ret"].iloc[lag:]

            if len(cpo_x) < 20:
                continue

            corr = float(
                pd.Series(cpo_x.values, name="x").corr(
                    pd.Series(stock_y.values, name="y"),
                    method="spearman",
                )
            )
            if pd.isna(corr):
                corr = 0.0
            lag_corrs[lag] = round(corr, 4)
            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_lag  = lag

        # ── Today's signal ────────────────────────────────────────────────────
        signal_today   = float(merged["cpo_ret"].iloc[-1]) if len(merged) > 0 else 0.0
        is_significant = abs(best_corr) > 0.30

        if not is_significant or abs(signal_today) < 0.005:
            direction = "neutral"
        elif (signal_today > 0 and best_corr > 0) or (signal_today < 0 and best_corr < 0):
            direction = "up"
        else:
            direction = "down"

        result = {
            "ticker":              plantation_ticker,
            "best_lag_days":       best_lag,
            "best_lag_corr":       round(best_corr, 4),
            "is_significant":      is_significant,
            "signal_today":        round(signal_today, 6),
            "predicted_direction": direction,
            "lag_correlations":    lag_corrs,
            "n_observations":      len(merged),
        }

        logger.info(
            f"CPO lag [{plantation_ticker}]: "
            f"best_lag={best_lag}d corr={best_corr:+.3f} "
            f"sig={is_significant} signal={signal_today:+.4f} → {direction} "
            f"(n={len(merged)})"
        )
        return result
