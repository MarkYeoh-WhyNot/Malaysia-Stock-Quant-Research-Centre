import logging
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
from agents.base_agent import BaseAgent
from config.settings import MODEL_FAST, BASE_DIR, DEFAULT_SYMBOLS
from data.database import db_session
from data.yahoo.client import extract_tickers, get_historical_data, get_latest_prices, get_multi_info, BARS_PER_YEAR

logger = logging.getLogger(__name__)

CACHE_DIR = BASE_DIR / "data" / "cache"


class DataEngineer(BaseAgent):
    name = "DataEngineer"
    description = "Bursa Malaysia price data fetching, caching, and feature engineering via Yahoo Finance"
    default_model = MODEL_FAST

    # ── Cache ──────────────────────────────────────────────────────────────────

    def _cache_path(self, symbol: str, interval: str) -> Path:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        safe = symbol.replace(".", "_")
        return CACHE_DIR / f"{safe}_{interval}.parquet"

    def _is_stale(self, path: Path, max_age_hours: int = 12) -> bool:
        """Equity data needs less frequent refresh than FX (markets closed overnight)."""
        if not path.exists():
            return True
        age = datetime.utcnow() - datetime.utcfromtimestamp(path.stat().st_mtime)
        return age > timedelta(hours=max_age_hours)

    def _load_cache(self, path: Path) -> pd.DataFrame:
        try:
            return pd.read_parquet(path)
        except Exception as e:
            logger.warning(f"Cache read failed {path}: {e}")
            return pd.DataFrame()

    def _save_cache(self, df: pd.DataFrame, path: Path):
        try:
            df.to_parquet(path)
        except Exception as e:
            logger.warning(f"Cache write failed {path}: {e}")

    # ── Fetch ──────────────────────────────────────────────────────────────────

    def fetch_prices(
        self,
        symbol: str,
        interval: str = "1d",
        days: int = 730,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV for a Bursa Malaysia stock (Yahoo Finance .KL ticker).
        Caches to parquet; default stale threshold is 12h (adequate for daily bars).

        Accepts dirty ticker strings such as sector descriptions or comma-separated lists
        (e.g. "Healthcare sector (e.g., 5225.KL, 5878.KL)") — the first valid .KL code
        is extracted and used as the primary ticker.
        """
        # Sanitize: extract the primary .KL ticker from any description string
        candidates = extract_tickers(symbol)
        primary = candidates[0]
        if primary != symbol:
            self.log_daemon(
                "INFO",
                f"DataEngineer: ticker sanitized '{symbol[:80]}' → '{primary}'",
            )
        symbol = primary

        path = self._cache_path(symbol, interval)
        if use_cache and not self._is_stale(path):
            df = self._load_cache(path)
            if not df.empty:
                self.log_daemon("INFO", f"Cache hit: {symbol} {interval} ({len(df)} bars)")
                return df

        self.log_daemon("INFO", f"Fetching {symbol} {interval} {days}d from Yahoo Finance")
        df = get_historical_data(symbol, interval=interval, days=days)

        if df.empty:
            self.log_daemon("WARN", f"No data returned for {symbol} {interval}")
            return df

        self._save_cache(df, path)
        self.log_daemon("INFO", f"Cached {len(df)} bars for {symbol} {interval}")
        return df

    # ── Feature engineering ────────────────────────────────────────────────────

    @staticmethod
    def _ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        rs    = gain / loss.replace(0, 1e-9)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        cp   = df["close"].shift(1)
        tr   = pd.concat([df["high"] - df["low"],
                          (df["high"] - cp).abs(),
                          (df["low"]  - cp).abs()], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def _adl(df: pd.DataFrame) -> pd.Series:
        """Accumulation/Distribution Line."""
        clv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / \
              (df["high"] - df["low"]).replace(0, 1e-9)
        return (clv * df["volume"]).cumsum()

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add equity-relevant technical and statistical features to a price DataFrame.
        Designed for daily (1d) Bursa Malaysia data.
        """
        if df.empty:
            return df
        out   = df.copy()
        close = df["close"]

        # ── Moving averages ──────────────────────────────────────────────────
        for p in (5, 10, 20, 50, 100, 200):
            out[f"sma_{p}"]  = close.rolling(p).mean()
            out[f"ema_{p}"]  = self._ema(close, p)

        # ── MACD ────────────────────────────────────────────────────────────
        out["macd"]         = self._ema(close, 12) - self._ema(close, 26)
        out["macd_signal"]  = self._ema(out["macd"], 9)
        out["macd_hist"]    = out["macd"] - out["macd_signal"]

        # ── Oscillators ─────────────────────────────────────────────────────
        out["rsi_14"]       = self._rsi(close, 14)
        out["rsi_7"]        = self._rsi(close, 7)

        # ── Bollinger Bands ──────────────────────────────────────────────────
        bb_mid              = close.rolling(20).mean()
        bb_std              = close.rolling(20).std()
        out["bb_upper"]     = bb_mid + 2 * bb_std
        out["bb_lower"]     = bb_mid - 2 * bb_std
        out["bb_pct"]       = (close - out["bb_lower"]) / (out["bb_upper"] - out["bb_lower"] + 1e-9)
        out["bb_width"]     = (out["bb_upper"] - out["bb_lower"]) / bb_mid

        # ── Volatility ───────────────────────────────────────────────────────
        out["atr_14"]       = self._atr(df, 14)
        out["atr_pct"]      = out["atr_14"] / close
        out["hist_vol_20"]  = close.pct_change().rolling(20).std() * np.sqrt(252)

        # ── Returns ──────────────────────────────────────────────────────────
        for p in (1, 3, 5, 10, 20, 60):
            out[f"ret_{p}d"]     = close.pct_change(p)
            out[f"log_ret_{p}d"] = np.log(close / close.shift(p))

        # ── Rate of change ────────────────────────────────────────────────────
        for p in (10, 20, 50, 120):
            out[f"roc_{p}"] = close.pct_change(p) * 100

        # ── Volume features ──────────────────────────────────────────────────
        if "volume" in df.columns and df["volume"].sum() > 0:
            out["vol_sma_20"]   = df["volume"].rolling(20).mean()
            out["vol_ratio"]    = df["volume"] / out["vol_sma_20"]
            out["vwap_20"]      = (close * df["volume"]).rolling(20).sum() / df["volume"].rolling(20).sum()
            out["adl"]          = self._adl(df)

        # ── Trend signals ─────────────────────────────────────────────────────
        out["above_sma50"]   = (close > out["sma_50"]).astype(int)
        out["above_sma200"]  = (close > out["sma_200"]).astype(int)
        out["golden_cross"]  = ((out["sma_50"] > out["sma_200"]) &
                                (out["sma_50"].shift(1) <= out["sma_200"].shift(1))).astype(int)
        out["death_cross"]   = ((out["sma_50"] < out["sma_200"]) &
                                (out["sma_50"].shift(1) >= out["sma_200"].shift(1))).astype(int)

        # ── 52-week high/low proximity ────────────────────────────────────────
        out["52w_high"]      = close.rolling(252, min_periods=50).max()
        out["52w_low"]       = close.rolling(252, min_periods=50).min()
        out["dist_52w_high"] = (close - out["52w_high"]) / out["52w_high"]
        out["dist_52w_low"]  = (close - out["52w_low"])  / out["52w_low"]

        return out

    # ── Screener snapshot ──────────────────────────────────────────────────────

    def get_universe_snapshot(self) -> list:
        """Return latest prices and basic stats for all KLCI stocks."""
        from config.settings import DEFAULT_SYMBOLS
        return get_latest_prices(DEFAULT_SYMBOLS)

    def get_universe_fundamentals(self, symbols: list = None) -> list:
        """Return yfinance fundamental info for the KLCI universe."""
        if symbols is None:
            symbols = DEFAULT_SYMBOLS
        return get_multi_info(symbols)

    # ── Bulk refresh ──────────────────────────────────────────────────────────

    def refresh_universe(
        self,
        symbols: list = None,
        intervals: list = None,
        days: int = 730,
    ) -> dict:
        if symbols  is None: symbols   = DEFAULT_SYMBOLS
        if intervals is None: intervals = ["1d"]
        results = {}
        for sym in symbols:
            results[sym] = {}
            for iv in intervals:
                try:
                    df = self.fetch_prices(sym, iv, days=days, use_cache=False)
                    results[sym][iv] = len(df)
                except Exception as e:
                    logger.error(f"Refresh failed {sym} {iv}: {e}")
                    results[sym][iv] = -1
        return results

    # ── Cache stats ───────────────────────────────────────────────────────────

    def cache_stats(self) -> dict:
        stats = {}
        if CACHE_DIR.exists():
            for f in CACHE_DIR.glob("*.parquet"):
                try:
                    df = pd.read_parquet(f)
                    age_h = (datetime.utcnow() - datetime.utcfromtimestamp(
                        f.stat().st_mtime)).total_seconds() / 3600
                    stats[f.stem] = {"rows": len(df), "age_hours": round(age_h, 1)}
                except Exception:
                    stats[f.stem] = {"rows": 0, "age_hours": -1}
        return stats

    # ── run() ──────────────────────────────────────────────────────────────────

    def run(self, task: dict) -> dict:
        action = task.get("action", "fetch")

        if action == "fetch":
            symbol   = task.get("symbol") or task.get("pair", "1155.KL")
            interval = task.get("interval") or task.get("timeframe", "1d")
            days     = int(task.get("days", 730))
            use_cache = bool(task.get("use_cache", True))
            df = self.fetch_prices(symbol, interval, days, use_cache)
            return {"symbol": symbol, "interval": interval, "bars": len(df), "ok": not df.empty}

        elif action == "features":
            symbol   = task.get("symbol") or task.get("pair", "1155.KL")
            interval = task.get("interval") or task.get("timeframe", "1d")
            df = self.fetch_prices(symbol, interval)
            if df.empty:
                return {"error": "No data"}
            featured = self.compute_features(df)
            return {"symbol": symbol, "interval": interval,
                    "bars": len(featured), "columns": list(featured.columns)}

        elif action == "refresh":
            symbols   = task.get("symbols", DEFAULT_SYMBOLS)
            intervals = task.get("intervals", ["1d"])
            days      = int(task.get("days", 730))
            result    = self.refresh_universe(symbols, intervals, days)
            return {"action": "refresh", "result": result}

        elif action == "snapshot":
            return {"snapshot": self.get_universe_snapshot()}

        elif action == "fundamentals":
            symbols = task.get("symbols", DEFAULT_SYMBOLS[:10])
            return {"fundamentals": self.get_universe_fundamentals(symbols)}

        elif action == "cache_stats":
            return {"cache": self.cache_stats()}

        return {"error": f"Unknown action: {action}"}
