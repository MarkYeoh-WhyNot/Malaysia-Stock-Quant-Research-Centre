import json
import logging
import os
import numpy as np
import pandas as pd
from agents.base_agent import BaseAgent
from config.settings import MODEL_FAST, MODEL_MAIN, GATE_CONFIG, KLCI_BY_SYMBOL, DEFAULT_SYMBOLS
from data.database import db_session
from data.yahoo.client import get_historical_data, BARS_PER_YEAR

logger = logging.getLogger(__name__)

SYSTEM = """You are a quantitative backtesting engineer specialising in Bursa Malaysia equities.
Parse equity strategy descriptions into structured signal parameters for vectorised backtesting.
Output only valid JSON."""


_PROGRESS_FILE = "/tmp/openclaw_progress.json"


class BacktestEngineer(BaseAgent):
    name = "BacktestEngineer"
    description = "Vectorised KLSE equity backtesting, Gate 2/3 evaluation (Stage 2-3)"
    default_model = MODEL_MAIN

    def _log_progress(self, idea_id: int, pct: int, msg: str) -> None:
        """Write backtest progress to a shared file for the API server to read."""
        try:
            data: dict = {}
            if os.path.exists(_PROGRESS_FILE):
                with open(_PROGRESS_FILE, "r") as fh:
                    data = json.load(fh)
            from datetime import datetime as _dt
            data[str(idea_id)] = {"pct": pct, "msg": msg, "ts": _dt.utcnow().isoformat()}
            with open(_PROGRESS_FILE, "w") as fh:
                json.dump(data, fh)
        except Exception:
            pass

    def _clear_progress(self, idea_id: int) -> None:
        """Remove completed entry from progress file."""
        try:
            if not os.path.exists(_PROGRESS_FILE):
                return
            with open(_PROGRESS_FILE, "r") as fh:
                data = json.load(fh)
            data.pop(str(idea_id), None)
            with open(_PROGRESS_FILE, "w") as fh:
                json.dump(data, fh)
        except Exception:
            pass

    # ── Data fetch ────────────────────────────────────────────────────────────

    def _fetch_prices(self, symbol: str, interval: str = "1d", days: int = 1825) -> pd.DataFrame:
        """Fetch via Yahoo Finance; use data cache if available."""
        try:
            from agents.data_engineer.data_engineer import DataEngineer
            de = DataEngineer()
            return de.fetch_prices(symbol, interval, days, use_cache=True)
        except Exception:
            return get_historical_data(symbol, interval=interval, days=days)

    # ── Factor parsing ────────────────────────────────────────────────────────

    def _parse_factor(self, factor_formula: str, title: str, hypothesis: str) -> dict:
        prompt = f"""Parse this Bursa Malaysia equity strategy into structured signal parameters.

Factor formula: {factor_formula}
Strategy title: {title}
Hypothesis: {hypothesis}

Return JSON:
{{
  "signal_type": "sma_crossover|ema_crossover|rsi|momentum|bollinger|macd|value|quality|volume_breakout",
  "fast_period": 20,
  "slow_period": 50,
  "rsi_period": 14,
  "rsi_oversold": 35,
  "rsi_overbought": 65,
  "bb_period": 20,
  "bb_std": 2.0,
  "momentum_period": 20,
  "macd_signal_period": 9,
  "volume_ma_period": 20,
  "volume_threshold": 1.5,
  "stop_loss_pct": 0.08,
  "take_profit_pct": 0.15,
  "long_only": true,
  "notes": "brief signal description"
}}"""
        result = self.call_claude_json(
            SYSTEM,
            [{"role": "user", "content": prompt}],
            model=MODEL_FAST,
            task_label="parse_factor",
        )
        return result if isinstance(result, dict) else {}

    # ── Signal computation ────────────────────────────────────────────────────

    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        rs    = gain / loss.replace(0, 1e-9)
        return 100 - (100 / (1 + rs))

    def _compute_signals(self, df: pd.DataFrame, params: dict) -> pd.Series:
        close  = df["close"]
        stype  = params.get("signal_type", "momentum")
        # KLSE equities: long-only by default (short-selling is restricted to
        # designated securities only)
        long_only = bool(params.get("long_only", True))

        if stype in ("sma_crossover", "ema_crossover"):
            fp = int(params.get("fast_period", 20))
            sp = int(params.get("slow_period", 50))
            if stype == "sma_crossover":
                fast_ma, slow_ma = close.rolling(fp).mean(), close.rolling(sp).mean()
            else:
                fast_ma = close.ewm(span=fp, adjust=False).mean()
                slow_ma = close.ewm(span=sp, adjust=False).mean()
            raw = np.where(fast_ma > slow_ma, 1, 0 if long_only else -1)

        elif stype == "rsi":
            rsi      = self._rsi(close, int(params.get("rsi_period", 14)))
            oversold = params.get("rsi_oversold", 35)
            overbought = params.get("rsi_overbought", 65)
            # Long when oversold, exit (or short) when overbought
            long_sig  = (rsi < oversold).astype(int)
            short_sig = (rsi > overbought).astype(int)
            raw = np.where(long_sig, 1, np.where(short_sig, 0 if long_only else -1, np.nan))
            # Forward-fill to hold position
            raw = pd.Series(raw, index=df.index).ffill().fillna(0).values

        elif stype == "bollinger":
            period   = int(params.get("bb_period", 20))
            std_mult = float(params.get("bb_std", 2.0))
            mid   = close.rolling(period).mean()
            std   = close.rolling(period).std()
            upper = mid + std_mult * std
            lower = mid - std_mult * std
            raw   = np.where(close < lower, 1, np.where(close > upper, 0 if long_only else -1, np.nan))
            raw   = pd.Series(raw, index=df.index).ffill().fillna(0).values

        elif stype == "macd":
            fp    = int(params.get("fast_period", 12))
            sp    = int(params.get("slow_period", 26))
            sig_p = int(params.get("macd_signal_period", 9))
            macd_line   = close.ewm(span=fp, adjust=False).mean() - close.ewm(span=sp, adjust=False).mean()
            signal_line = macd_line.ewm(span=sig_p, adjust=False).mean()
            raw = np.where(macd_line > signal_line, 1, 0 if long_only else -1)

        elif stype == "volume_breakout":
            vol_ma    = df["volume"].rolling(int(params.get("volume_ma_period", 20))).mean()
            vol_thresh = float(params.get("volume_threshold", 1.5))
            ret_1d    = close.pct_change()
            # Long on high-volume up days; exit on high-volume down days
            raw = np.where(
                (df["volume"] > vol_thresh * vol_ma) & (ret_1d > 0), 1,
                np.where((df["volume"] > vol_thresh * vol_ma) & (ret_1d < 0), 0 if long_only else -1, np.nan)
            )
            raw = pd.Series(raw, index=df.index).ffill().fillna(0).values

        else:  # momentum / default
            period = int(params.get("momentum_period", 20))
            ret    = close.pct_change(period)
            raw    = np.where(ret > 0, 1, 0 if long_only else -1)

        return pd.Series(raw, index=df.index, dtype=float)

    # ── Performance metrics ───────────────────────────────────────────────────

    def _compute_performance(self, df: pd.DataFrame, signals: pd.Series, interval: str) -> dict:
        close = df["close"].values
        sig   = signals.fillna(0).values

        # Execute at next-bar open (shift signal by 1)
        sig_shifted     = np.roll(sig, 1)
        sig_shifted[0]  = 0.0

        bar_returns     = np.diff(close) / np.where(close[:-1] != 0, close[:-1], 1e-9)
        # Bursa transaction cost: ~0.1% brokerage + 0.03% stamp duty each way
        TRANSACTION_COST = 0.0013
        signal_changes  = np.abs(np.diff(sig_shifted))
        net_returns     = sig_shifted[1:] * bar_returns - signal_changes * TRANSACTION_COST

        n = len(net_returns)
        if n < 20 or np.std(net_returns) < 1e-10:
            return {"sharpe": 0.0, "max_dd": 1.0, "win_rate": 0.0, "profit_factor": 0.0, "total_trades": 0}

        # Annualised Sharpe
        ann   = BARS_PER_YEAR.get(interval, 252)
        sharpe = (np.mean(net_returns) / np.std(net_returns)) * np.sqrt(ann)

        # Max drawdown
        cum   = np.cumprod(1 + np.clip(net_returns, -0.5, 0.5))
        peak  = np.maximum.accumulate(cum)
        dd    = (peak - cum) / np.where(peak != 0, peak, 1e-9)
        max_dd = float(dd.max())

        # Win rate and profit factor
        pos  = net_returns[net_returns > 0]
        neg  = net_returns[net_returns < 0]
        nz   = net_returns[net_returns != 0]
        win_rate = len(pos) / max(len(nz), 1)
        gross_win  = float(pos.sum()) if len(pos) > 0 else 0.0
        gross_loss = float(abs(neg.sum())) if len(neg) > 0 else 1e-9
        profit_factor = gross_win / gross_loss

        total_trades = int(np.sum(np.abs(np.diff(sig_shifted)) > 0))

        return {
            "sharpe":        round(float(sharpe), 3),
            "max_dd":        round(max_dd, 4),
            "win_rate":      round(win_rate, 4),
            "profit_factor": round(float(profit_factor), 3),
            "total_trades":  total_trades,
            "ann_return":    round(float(np.mean(net_returns)) * ann, 4),
        }

    # ── Spearman correlation (no scipy needed) ────────────────────────────────

    @staticmethod
    def _spearman(x: np.ndarray, y: np.ndarray) -> float:
        """Spearman rank correlation via pandas rank (handles ties, no scipy needed)."""
        n = len(x)
        if n < 4:
            return np.nan
        rx = pd.Series(x).rank(method="average").values.astype(float)
        ry = pd.Series(y).rank(method="average").values.astype(float)
        mx, my = rx.mean(), ry.mean()
        num   = np.mean((rx - mx) * (ry - my))
        denom = rx.std(ddof=0) * ry.std(ddof=0)
        return float(num / denom) if denom > 1e-10 else np.nan

    # ── Cross-sectional validation ────────────────────────────────────────────

    def cross_sectional_test(self, factor_formula: str, idea_id: int) -> dict:
        """Test whether a factor generalises across the full KLCI universe.

        For each of the 30 KLCI stocks:
          - Fetches 2yr daily prices
          - Computes the factor signal using the same parsed params as the single-stock backtest
          - Records per-stock Information Coefficient (IC): Spearman(signal, fwd_return)

        Also computes cross-sectional IC at each trading date:
          IC(t) = Spearman across stocks of {signal_t, return_t+1}
          Mean IC and IC t-stat = mean / (std / sqrt(T)) measure factor breadth.

        Quintile Sharpe: at each date go long top-quintile (6 stocks) by signal,
        equal-weight portfolio.

        Returns a dict with mean_ic, ic_tstat, stocks_positive_ic, best_stocks,
        worst_stocks, factor_is_real, and saves IC columns to the latest
        backtest_runs row for this idea.
        """
        with db_session() as conn:
            row = conn.execute("SELECT * FROM alpha_ideas WHERE id=?", (idea_id,)).fetchone()
        if not row:
            return {"error": f"Idea {idea_id} not found", "factor_is_real": False}

        formula = factor_formula or row["factor_formula"] or ""
        params  = self._parse_factor(formula, row["title"], row["hypothesis"] or "")
        if not params or "error" in params:
            params = {"signal_type": "momentum", "momentum_period": 20, "long_only": True}

        self.log_daemon("INFO", f"CrossSect [{idea_id}]: fetching 2yr data for {len(DEFAULT_SYMBOLS)} KLCI stocks")

        # ── Build signal + forward-return panels ─────────────────────────────
        signal_series: dict[str, pd.Series] = {}
        return_series: dict[str, pd.Series] = {}

        for symbol in DEFAULT_SYMBOLS:
            try:
                df = self._fetch_prices(symbol, "1d", days=730)
                if df.empty or len(df) < 60:
                    continue
                sig     = self._compute_signals(df, params)
                fwd_ret = df["close"].pct_change().shift(-1)   # next-bar return
                valid   = sig.notna() & fwd_ret.notna() & (sig != 0)
                if valid.sum() < 20:
                    continue
                signal_series[symbol] = sig[valid]
                return_series[symbol] = fwd_ret[valid]
            except Exception as e:
                self.log_daemon("WARN", f"CrossSect: skipped {symbol}: {e}")

        n_stocks = len(signal_series)
        if n_stocks < 5:
            self.log_daemon("WARN", f"CrossSect [{idea_id}]: only {n_stocks} stocks have data")
            return {
                "error": "insufficient stock data for cross-sectional test",
                "factor_is_real": False,
                "idea_id": idea_id,
            }

        # Align into panels (union of dates; NaN where stock missing on a date)
        sig_panel = pd.DataFrame(signal_series)
        ret_panel = pd.DataFrame(return_series)
        common_idx = sig_panel.index.intersection(ret_panel.index)
        sig_panel  = sig_panel.loc[common_idx]
        ret_panel  = ret_panel.loc[common_idx]

        # Keep rows where at least 5 stocks have valid values
        enough = (sig_panel.notna().sum(axis=1) >= 5) & (ret_panel.notna().sum(axis=1) >= 5)
        sig_panel = sig_panel[enough]
        ret_panel = ret_panel[enough]

        if len(sig_panel) < 30:
            return {
                "error": "insufficient common dates for cross-sectional IC",
                "factor_is_real": False,
                "idea_id": idea_id,
            }

        # ── Cross-sectional IC series (IC at each trading date) ───────────────
        ic_series: list[float] = []
        portfolio_rets: list[float] = []

        for date in sig_panel.index:
            sig_row = sig_panel.loc[date].dropna()
            ret_row = ret_panel.loc[date].dropna()
            common_stocks = sig_row.index.intersection(ret_row.index)
            if len(common_stocks) < 5:
                continue

            sv = sig_row[common_stocks].values
            rv = ret_row[common_stocks].values

            ic = self._spearman(sv, rv)
            if not np.isnan(ic):
                ic_series.append(ic)

            # Top-quintile portfolio (top ~20% of available stocks that day)
            n_q = max(1, len(common_stocks) // 5)
            top_idx = np.argsort(sv)[-n_q:]
            if len(top_idx) > 0:
                portfolio_rets.append(float(np.mean(rv[top_idx])))

        if not ic_series:
            return {
                "error": "IC series is empty (constant signal?)",
                "factor_is_real": False,
                "idea_id": idea_id,
            }

        ic_arr  = np.array(ic_series)
        mean_ic = float(np.mean(ic_arr))
        ic_std  = float(np.std(ic_arr, ddof=1))
        ic_tstat = (mean_ic / (ic_std / np.sqrt(len(ic_arr)))) if ic_std > 1e-10 else 0.0

        # ── Per-stock IC (how predictive is the factor within each ticker) ────
        stock_ics: dict[str, float] = {}
        for sym in sig_panel.columns:
            sig_ts = sig_panel[sym].dropna()
            ret_ts = ret_panel[sym].dropna()
            overlap = sig_ts.index.intersection(ret_ts.index)
            if len(overlap) < 20:
                continue
            ic = self._spearman(sig_ts[overlap].values, ret_ts[overlap].values)
            if not np.isnan(ic):
                stock_ics[sym] = ic

        stocks_positive_ic = sum(1 for v in stock_ics.values() if v > 0)
        sorted_ics = sorted(stock_ics.items(), key=lambda x: x[1], reverse=True)
        best_stocks  = [
            {"symbol": s, "ic": round(ic, 4),
             "name": KLCI_BY_SYMBOL.get(s, {}).get("name", s)}
            for s, ic in sorted_ics[:5]
        ]
        worst_stocks = [
            {"symbol": s, "ic": round(ic, 4),
             "name": KLCI_BY_SYMBOL.get(s, {}).get("name", s)}
            for s, ic in sorted_ics[-5:]
        ]

        # ── Quintile portfolio Sharpe ─────────────────────────────────────────
        quintile_sharpe = 0.0
        if len(portfolio_rets) > 20:
            pr  = np.array(portfolio_rets)
            std = float(np.std(pr, ddof=1))
            quintile_sharpe = round(float(np.mean(pr) / std * np.sqrt(252)), 3) if std > 1e-10 else 0.0

        # ── Gate: factor is real? ─────────────────────────────────────────────
        factor_is_real = (
            mean_ic > 0.05
            and ic_tstat > 1.5
            and stocks_positive_ic > 15
        )

        result = {
            "idea_id":            idea_id,
            "mean_ic":            round(mean_ic, 4),
            "ic_tstat":           round(ic_tstat, 3),
            "stocks_tested":      n_stocks,
            "stocks_positive_ic": stocks_positive_ic,
            "quintile_sharpe":    quintile_sharpe,
            "best_stocks":        best_stocks,
            "worst_stocks":       worst_stocks,
            "factor_is_real":     factor_is_real,
            "ic_periods":         len(ic_series),
        }

        # ── Persist IC columns to latest backtest_run for this idea ──────────
        try:
            with db_session() as conn:
                conn.execute("""
                    UPDATE backtest_runs
                    SET mean_ic=?, ic_tstat=?, stocks_positive_ic=?, best_stocks=?
                    WHERE id = (
                        SELECT id FROM backtest_runs WHERE idea_id=?
                        ORDER BY created_at DESC LIMIT 1
                    )
                """, (
                    round(mean_ic, 4), round(ic_tstat, 3),
                    stocks_positive_ic, json.dumps(best_stocks),
                    idea_id,
                ))
        except Exception as e:
            self.log_daemon("WARN", f"CrossSect: failed to save IC stats for [{idea_id}]: {e}")

        self.log_daemon(
            "INFO" if factor_is_real else "WARN",
            f"CrossSect [{idea_id}] {'REAL' if factor_is_real else 'WEAK'} "
            f"mean_IC={mean_ic:.3f} t={ic_tstat:.2f} pos_stocks={stocks_positive_ic}/{n_stocks} "
            f"q_sharpe={quintile_sharpe:.2f}",
        )
        return result

    # ── Holding period classification ─────────────────────────────────────────

    @staticmethod
    def classify_holding_period(timeframe: str, factor_formula: str, hypothesis: str) -> str:
        """Classify strategy into INTRADAY / SHORT_TERM / MEDIUM_TERM / LONG_TERM.

        Based on keywords in factor_formula and hypothesis text. Returns one of:
          INTRADAY    — < 1 day (intraday, tick, scalp, 1min/5min)
          SHORT_TERM  — 1-10 trading days
          MEDIUM_TERM — 10-60 trading days (most KLSE strategies)
          LONG_TERM   — > 60 trading days
        """
        blob = f"{timeframe} {factor_formula} {hypothesis}".lower()

        intraday_kw  = ["intraday", "scalp", "tick", "1 minute", "5 minute", "15 minute",
                        "60 minute", "1min", "5min", "15min", "hourly", "hft"]
        short_kw     = ["1 day", "2 day", "3 day", "4 day", "5 day", "1-5 day", "1-3 day",
                        "1 week", "t+1", "t+2", "t+3", "overnight", "few days"]
        long_kw      = ["3 month", "6 month", "12 month", "annual", "quarterly",
                        "long-term", "long term", "buy and hold", "60 day", "90 day"]

        if any(kw in blob for kw in intraday_kw):
            return "INTRADAY"
        if any(kw in blob for kw in long_kw):
            return "LONG_TERM"
        if any(kw in blob for kw in short_kw):
            return "SHORT_TERM"
        return "MEDIUM_TERM"   # default: most KLSE strategies hold weeks–months

    # Minimum trade count requirements per holding period class (Fix 6)
    _MIN_TRADES = {
        "INTRADAY":    100,
        "SHORT_TERM":   50,
        "MEDIUM_TERM":  30,
        "LONG_TERM":    15,
    }

    # Sharpe thresholds per holding period class (Fix 4)
    _SHARPE_THRESHOLDS = {
        "INTRADAY":    1.1,   # indicative only — needs tick data
        "SHORT_TERM":  1.1,
        "MEDIUM_TERM": 1.1,
        "LONG_TERM":   0.8,   # fewer trades, lower bar
    }

    # ── Formula verification ──────────────────────────────────────────────────

    def verify_formula(self, params: dict, factor_formula: str, df: pd.DataFrame) -> dict:
        """Verify that the parsed signal code matches the formula description.

        Runs the signal on the last 20 bars and asks Claude to confirm the
        signals are directionally correct and match the formula intent.

        Returns dict with keys: verified (bool), confidence (float), issue (str).
        """
        if df.empty or len(df) < 30:
            return {"verified": False, "confidence": 0.0, "issue": "insufficient data for verification"}

        sample_df = df.iloc[-20:].copy()
        try:
            signals = self._compute_signals(sample_df, params)
        except Exception as e:
            return {"verified": False, "confidence": 0.0, "issue": f"signal computation error: {e}"}

        # Build human-readable samples
        close_sample = sample_df["close"].round(4).tolist()
        signal_sample = signals.fillna(0).astype(int).tolist()
        dates_sample  = [str(d)[:10] for d in sample_df.index.tolist()]

        bars_table = "\n".join(
            f"  {dates_sample[i]}: close={close_sample[i]:.3f}  signal={signal_sample[i]}"
            for i in range(len(dates_sample))
        )

        verify_prompt = f"""The factor formula says: {factor_formula}

The code produced these signals on the last 20 bars (1=long, 0=flat):
{bars_table}

Does the signal output match what the formula describes?
Are the entry/exit points logical given the price data?
Is the direction correct (long when signal=1, flat/no position when signal=0)?

Return JSON only:
{{
  "verified": true,
  "confidence": 0.0,
  "issue": "description of any problem found, or empty string if verified"
}}"""

        result = self.call_claude_json(
            SYSTEM,
            [{"role": "user", "content": verify_prompt}],
            model=MODEL_FAST,
            task_label="verify_formula",
        )
        verified   = bool(result.get("verified", False))
        confidence = float(result.get("confidence", 0.0))
        issue      = result.get("issue", "")

        if not verified or confidence < 0.7:
            self.log_daemon("ERROR", f"Formula verification failed (confidence={confidence:.2f}): {issue}")
        else:
            self.log_daemon("INFO", f"Formula verified with confidence {confidence:.2f}")

        return {
            "verified":   verified,
            "confidence": confidence,
            "issue":      issue,
        }

    # ── Train / val / test split ─────────────────────────────────────────────

    @staticmethod
    def _split(df: pd.DataFrame) -> tuple:
        n = len(df)
        t = int(n * GATE_CONFIG.stage3_data_split_train)
        v = int(n * GATE_CONFIG.stage3_data_split_val)
        return df.iloc[:t], df.iloc[t:t + v], df.iloc[t + v:]

    # ── Main backtest pipeline ────────────────────────────────────────────────

    def backtest_idea(self, idea_id: int) -> dict:
        with db_session() as conn:
            row = conn.execute("SELECT * FROM alpha_ideas WHERE id=?", (idea_id,)).fetchone()
        if not row:
            return {"error": f"Idea {idea_id} not found"}

        symbol   = row["ticker"] or "1155.KL"
        interval = row["timeframe"] or "1d"
        stock    = KLCI_BY_SYMBOL.get(symbol, {})

        self.log_daemon("INFO", f"Backtesting [{idea_id}] {row['title']} — {symbol} {interval}")
        self._log_progress(idea_id, 10, f"Fetching price data for {symbol}")

        # Parse factor formula
        params = self._parse_factor(
            row["factor_formula"] or "",
            row["title"],
            row["hypothesis"] or "",
        )
        if not params or "error" in params:
            params = {"signal_type": "momentum", "momentum_period": 20, "long_only": True}

        # Fetch 5 years of daily data for robust train/val/test split
        df = self._fetch_prices(symbol, interval, days=1825)
        if df.empty or len(df) < 150:
            self.log_daemon("WARN", f"Insufficient data [{idea_id}]: {len(df)} bars for {symbol}")
            return {"error": "Insufficient historical data", "idea_id": idea_id, "symbol": symbol}

        self._log_progress(idea_id, 30, f"Computing factor signals ({symbol})")
        # Classify holding period for appropriate thresholds and warnings
        hp_class = self.classify_holding_period(
            interval, row["factor_formula"] or "", row["hypothesis"] or ""
        )
        self.log_daemon("INFO", f"Backtest [{idea_id}] holding_period_class={hp_class}")

        # Verify formula before full backtest
        verification = self.verify_formula(params, row["factor_formula"] or "", df)
        needs_review    = 0 if (verification["verified"] and verification["confidence"] >= 0.7) else 1
        verification_note = verification.get("issue", "") or ""

        # INTRADAY on daily bars — flag immediately, do not trust results
        if hp_class == "INTRADAY":
            needs_review = 1
            verification_note = (
                (verification_note + " | " if verification_note else "")
                + "INTRADAY strategy backtested on daily bars — indicative only, needs tick data"
            )
            self.log_daemon(
                "WARN",
                f"Backtest [{idea_id}] INTRADAY strategy on daily OHLCV — results are indicative only",
            )

        self._log_progress(idea_id, 50, "Running train/val/test split")
        train_df, val_df, test_df = self._split(df)

        results = {}
        for split_name, split_df in (("train", train_df), ("val", val_df), ("test", test_df)):
            if len(split_df) < 40:
                results[split_name] = {
                    "sharpe": 0.0, "max_dd": 1.0, "win_rate": 0.0,
                    "profit_factor": 0.0, "total_trades": 0, "ann_return": 0.0,
                }
                continue
            sig = self._compute_signals(split_df, params)
            results[split_name] = self._compute_performance(split_df, sig, interval)

        self._log_progress(idea_id, 70, "Computing Sharpe and drawdown")
        train_r = results["train"]
        val_r   = results["val"]
        test_r  = results["test"]
        train_val_gap = abs(train_r["sharpe"] - val_r["sharpe"])

        # Per holding-period-class thresholds
        sharpe_threshold = self._SHARPE_THRESHOLDS.get(hp_class, GATE_CONFIG.stage3_min_sharpe)
        min_trades       = self._MIN_TRADES.get(hp_class, 30)
        actual_trades    = test_r.get("total_trades", 0)

        # Gate 2 — in-sample quality
        gate2_pass = (
            train_r["sharpe"] >= sharpe_threshold
            and val_r["sharpe"]   >= sharpe_threshold
            and train_r["max_dd"] <= GATE_CONFIG.stage3_max_drawdown
            and val_r["max_dd"]   <= GATE_CONFIG.stage3_max_drawdown
            and train_val_gap     <= GATE_CONFIG.stage3_max_train_val_gap
        )

        # Gate 3 — out-of-sample test
        gate3_pass = (
            gate2_pass
            and test_r["sharpe"]  >= sharpe_threshold
            and test_r["max_dd"]  <= GATE_CONFIG.stage3_max_drawdown
        )

        # Fix 6 — minimum trade count gate
        trade_count_pass = actual_trades >= min_trades
        trade_count_note = ""
        if not trade_count_pass:
            trade_count_note = (
                f"Insufficient trades: {actual_trades} found, minimum {min_trades} "
                f"for {hp_class} strategies. Statistical significance requires more signals."
            )
            self.log_daemon(
                "WARN",
                f"Backtest [{idea_id}] trade count gate FAILED: "
                f"{actual_trades} trades < {min_trades} minimum for {hp_class}",
            )
            # Record in rejection memory
            try:
                from knowledge.ingestion.rejection_memory import RejectionMemory
                RejectionMemory().record_rejection(idea_id, trade_count_note, "stage2_trades")
            except Exception:
                pass

        overall_pass = gate3_pass and trade_count_pass
        self._log_progress(idea_id, 90, "Running cross-sectional IC check")

        run_id = None
        try:
            with db_session() as conn:
                full_note = " | ".join(filter(None, [verification_note, trade_count_note]))
                conn.execute("""
                    INSERT INTO backtest_runs
                      (idea_id, run_type, pair, timeframe, factor_formula,
                       train_sharpe, val_sharpe, test_sharpe,
                       train_dd, val_dd, test_dd,
                       train_val_gap, total_trades, win_rate, profit_factor,
                       params, result_data, passed, needs_review, verification_note,
                       holding_period_class, trade_count)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    idea_id, "klse_daily", symbol, interval, row["factor_formula"],
                    train_r["sharpe"], val_r["sharpe"], test_r["sharpe"],
                    train_r["max_dd"], val_r["max_dd"],  test_r["max_dd"],
                    round(train_val_gap, 3), test_r["total_trades"],
                    test_r["win_rate"], test_r["profit_factor"],
                    json.dumps(params), json.dumps(results),
                    1 if overall_pass else 0,
                    needs_review, full_note or None,
                    hp_class, actual_trades,
                ))
                run_id = conn.execute(
                    "SELECT id FROM backtest_runs WHERE idea_id=? ORDER BY created_at DESC LIMIT 1",
                    (idea_id,),
                ).fetchone()["id"]

                new_stage  = "stage3" if overall_pass else "stage2"
                new_status = "active"  if overall_pass else "rejected"
                conn.execute("""
                    UPDATE alpha_ideas
                    SET backtest_sharpe=?, backtest_dd=?, stage=?, status=?, updated_at=datetime('now')
                    WHERE id=?
                """, (test_r["sharpe"], test_r["max_dd"], new_stage, new_status, idea_id))

                conn.execute("""
                    INSERT INTO pipeline_events (idea_id, stage, event_type, agent, notes)
                    VALUES (?, 'stage2', ?, 'BacktestEngineer', ?)
                """, (idea_id,
                      "advanced" if overall_pass else "rejected",
                      f"Train={train_r['sharpe']:.2f} Val={val_r['sharpe']:.2f} "
                      f"Test={test_r['sharpe']:.2f} DD={test_r['max_dd']:.1%} "
                      f"AnnRet={test_r.get('ann_return',0):.1%}"))

                conn.execute("""
                    INSERT INTO gate_decisions (idea_id, gate, decision, decided_by, rationale)
                    VALUES (?, 'gate2_3', ?, 'BacktestEngineer', ?)
                """, (idea_id,
                      "approve" if overall_pass else "reject",
                      f"G2={'PASS' if gate2_pass else 'FAIL'} G3={'PASS' if gate3_pass else 'FAIL'} "
                      f"gap={train_val_gap:.2f}"))
            self.log_daemon("INFO", f"Backtest saved for idea {idea_id} — pass={overall_pass}")
        except Exception as e:
            self.log_daemon("ERROR", f"Backtest save FAILED for idea {idea_id}: {e}")
            raise

        self._log_progress(idea_id, 100, "Complete")
        self._clear_progress(idea_id)
        self.log_daemon(
            "INFO" if overall_pass else "WARN",
            f"Backtest [{idea_id}] {symbol} {'PASSED' if overall_pass else 'FAILED'} "
            f"train={train_r['sharpe']} val={val_r['sharpe']} test={test_r['sharpe']}",
        )
        hp_warnings = []
        if hp_class == "INTRADAY":
            hp_warnings.append("Indicative only — needs tick data for reliable results")
        if hp_class == "SHORT_TERM":
            hp_warnings.append(
                "Daily bar backtest may overstate performance for strategies held < 10 days"
            )
        if not trade_count_pass:
            hp_warnings.append(trade_count_note)

        return {
            "idea_id":             idea_id,
            "run_id":              run_id,
            "symbol":              symbol,
            "company":             stock.get("name", symbol),
            "interval":            interval,
            "gate2_pass":          gate2_pass,
            "gate3_pass":          gate3_pass,
            "trade_count_pass":    trade_count_pass,
            "train":               train_r,
            "val":                 val_r,
            "test":                test_r,
            "train_val_gap":       round(train_val_gap, 3),
            "params":              params,
            "bars_total":          len(df),
            "factor_formula":      row["factor_formula"] or "",
            "needs_review":        bool(needs_review),
            "verification":        verification,
            "holding_period_class": hp_class,
            "actual_trades":       actual_trades,
            "min_trades_required": min_trades,
            "hp_warnings":         hp_warnings,
        }

    # ── run() ─────────────────────────────────────────────────────────────────

    def run(self, task: dict) -> dict:
        action = task.get("action", "backtest")
        if action == "backtest":
            idea_id = task.get("idea_id")
            if not idea_id:
                return {"error": "idea_id required"}
            return self.backtest_idea(idea_id)
        if action == "cross_sectional":
            idea_id = task.get("idea_id")
            if not idea_id:
                return {"error": "idea_id required"}
            return self.cross_sectional_test(task.get("factor_formula", ""), idea_id)
        return {"error": f"Unknown action: {action}"}
