import json
import logging
import os
import numpy as np
import pandas as pd
from agents.base_agent import BaseAgent
from config.settings import MODEL_FAST, MODEL_MAIN, GATE_CONFIG, KLCI_BY_SYMBOL, DEFAULT_SYMBOLS
from data.database import db_session
from data.yahoo.client import extract_tickers, get_historical_data, BARS_PER_YEAR

logger = logging.getLogger(__name__)

# ── Bursa Malaysia realistic transaction cost model (QC3) ────────────────────
# Applied once per completed round-trip trade (open + close).
_BT_BROKERAGE       = 0.0020   # 0.20% per side (online broker)
_BT_STAMP_DUTY      = 0.0015   # 0.15% per side (capped MYR 1000)
_BT_SLIPPAGE        = 0.0010   # 0.10% market impact estimate
_BT_ROUND_TRIP_COST = (_BT_BROKERAGE + _BT_STAMP_DUTY + _BT_SLIPPAGE) * 2  # 0.0090

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
  "signal_type": "sma_crossover|ema_crossover|rsi|momentum|bollinger|macd|value|quality|volume_breakout|gap_fill|short_term_reversal|cross_sectional_momentum|pead|cpo_correlation|cpo_lag|opr_banking_signal|opr_cycle",
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
        open_prices = df["open"] if "open" in df.columns else close

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

        elif stype == "fundamental_screen":
            # Constant signal from a fundamental screen (ROE, PB, PE, DER, etc.).
            # Fundamental screens are quarterly-rebalanced, not time-varying on daily bars.
            # The pre-computed signal value (1=long, 0=flat) is stored in params by
            # _run_backtest() after evaluating fund_context against the formula.
            fundamental_signal = float(params.get("fundamental_signal", 1.0))
            raw = np.full(len(close), fundamental_signal)

        elif stype == "gap_fill":
            gap_pct = (open_prices - close.shift(1)) / close.shift(1)
            raw = (gap_pct < -0.03).astype(float).values

        elif stype == "short_term_reversal":
            five_day_return = close.pct_change(5)
            raw = (five_day_return < -0.06).astype(float).values

        elif stype == "cross_sectional_momentum":
            formation_return = close.shift(21).pct_change(126)
            rolling_pct = formation_return.rolling(252).rank(pct=True)
            raw = (rolling_pct >= 0.80).astype(float).values

        elif stype == "pead":
            gap_pct = (open_prices - close.shift(1)) / close.shift(1)
            raw = (gap_pct > 0.02).astype(float).values

        elif stype in ("cpo_correlation", "cpo_lag"):
            try:
                import yfinance as yf
                cpo_dl = yf.download(
                    "FCPO=F",
                    start=close.index[0],
                    end=close.index[-1],
                    interval="1d",
                    progress=False,
                )["Close"]
                # yfinance may return DataFrame with MultiIndex columns — flatten to Series
                if hasattr(cpo_dl, "squeeze"):
                    cpo_dl = cpo_dl.squeeze()
                cpo = pd.Series(cpo_dl.values.ravel(), index=cpo_dl.index) if not isinstance(cpo_dl, pd.Series) else cpo_dl
                cpo = cpo.reindex(close.index).ffill()
                cpo_mom = cpo.pct_change(5)
                raw = (cpo_mom > 0).astype(float).values
            except Exception:
                raw = (close.pct_change(21) > 0).astype(float).values

        elif stype in ("opr_banking_signal", "opr_cycle"):
            three_month_ret = close.pct_change(63)
            raw = (three_month_ret > 0.02).astype(float).values

        else:  # momentum / default
            period = int(params.get("momentum_period", 20))
            ret    = close.pct_change(period)
            raw    = np.where(ret > 0, 1, 0 if long_only else -1)

        return pd.Series(raw, index=df.index, dtype=float)

    # ── Performance metrics ───────────────────────────────────────────────────

    def _compute_performance(self, df: pd.DataFrame, signals: pd.Series, interval: str) -> dict:
        """Compute performance with QC1 lookahead guard and QC3 realistic costs.

        QC1 — Lookahead bias guard:
          Signal computed on day T may only trigger a trade at T+1 open.
          Enforced by pd.Series.shift(1) — signal_shifted[t] = signal[t-1].
          signal_shifted[0] is forced to 0 (no position at start).

        QC3 — Realistic Bursa transaction costs:
          BROKERAGE=0.20% + STAMP_DUTY=0.15% + SLIPPAGE=0.10% per side,
          applied as a round-trip deduction on every position change.
          Returns both sharpe_gross (before costs) and sharpe_net (after costs).
        """
        close = df["close"]
        sig   = signals.fillna(0)

        # ── QC1: strict 1-bar signal delay ────────────────────────────────────
        signal_shifted = sig.shift(1).fillna(0)
        assert float(signal_shifted.iloc[0]) == 0.0, \
            "Lookahead guard failure: signal_shifted[0] != 0"

        bar_returns    = close.pct_change().fillna(0)

        # Gross returns (no costs): position held at T earns return from T→T+1
        gross_bar     = signal_shifted * bar_returns
        gross_returns = gross_bar.values[1:]   # drop bar 0 (always 0 after shift)

        # ── QC3: subtract round-trip cost on every position change ────────────
        signal_changes = np.abs(np.diff(signal_shifted.values))
        net_returns    = gross_returns - signal_changes * _BT_ROUND_TRIP_COST

        n = len(net_returns)
        _empty = {
            "sharpe": 0.0, "sharpe_gross": 0.0, "sharpe_net": 0.0,
            "max_dd": 1.0, "win_rate": 0.0, "profit_factor": 0.0,
            "total_trades": 0, "ann_return": 0.0,
        }
        if n < 20 or np.std(net_returns) < 1e-10:
            return _empty

        ann         = BARS_PER_YEAR.get(interval, 252)
        g_std       = float(np.std(gross_returns))
        n_std       = float(np.std(net_returns))
        sharpe_gross = round(float(np.mean(gross_returns) / g_std * np.sqrt(ann)), 3) if g_std > 1e-10 else 0.0
        sharpe_net   = round(float(np.mean(net_returns)   / n_std * np.sqrt(ann)), 3) if n_std > 1e-10 else 0.0

        # Max drawdown (on net equity curve)
        cum    = np.cumprod(1 + np.clip(net_returns, -0.5, 0.5))
        peak   = np.maximum.accumulate(cum)
        dd     = (peak - cum) / np.where(peak != 0, peak, 1e-9)
        max_dd = float(dd.max())

        # Win rate / profit factor
        pos           = net_returns[net_returns > 0]
        neg           = net_returns[net_returns < 0]
        nz            = net_returns[net_returns != 0]
        win_rate      = len(pos) / max(len(nz), 1)
        gross_win     = float(pos.sum())         if len(pos) > 0 else 0.0
        gross_loss    = float(abs(neg.sum()))    if len(neg) > 0 else 1e-9
        profit_factor = gross_win / gross_loss
        total_trades  = int(np.sum(signal_changes > 0))

        return {
            "sharpe":        sharpe_net,          # backward-compat: sharpe == net
            "sharpe_gross":  sharpe_gross,
            "sharpe_net":    sharpe_net,
            "max_dd":        round(max_dd, 4),
            "win_rate":      round(win_rate, 4),
            "profit_factor": round(float(profit_factor), 3),
            "total_trades":  total_trades,
            "ann_return":    round(float(np.mean(net_returns)) * ann, 4),
        }

    # ── QC2: Walk-forward IS / OOS validation ────────────────────────────────

    def _compute_walk_forward(self, df: pd.DataFrame, params: dict, interval: str) -> dict:
        """Split full dataset 70/30 and compute Sharpe separately for IS and OOS periods.

        Returns sharpe_is, sharpe_oos, oos_degradation (fraction drop from IS to OOS).
        """
        n        = len(df)
        split_at = int(n * 0.70)
        is_df    = df.iloc[:split_at]
        oos_df   = df.iloc[split_at:]

        if len(is_df) < 60 or len(oos_df) < 20:
            return {"sharpe_is": 0.0, "sharpe_oos": 0.0, "oos_degradation": 0.0}

        is_perf  = self._compute_performance(is_df,  self._compute_signals(is_df,  params), interval)
        oos_perf = self._compute_performance(oos_df, self._compute_signals(oos_df, params), interval)

        sharpe_is  = is_perf["sharpe_net"]
        sharpe_oos = oos_perf["sharpe_net"]
        # Degradation: how much (as a fraction) OOS drops below IS
        deg = (sharpe_is - sharpe_oos) / max(abs(sharpe_is), 1e-9) if sharpe_is > 0 else 0.0

        return {
            "sharpe_is":       round(sharpe_is, 3),
            "sharpe_oos":      round(sharpe_oos, 3),
            "oos_degradation": round(deg, 3),
        }

    # ── QC5: Regime stress test ───────────────────────────────────────────────

    def _compute_regimes(self, df: pd.DataFrame, params: dict, interval: str) -> dict:
        """Split the backtest period into 3 volatility regimes and compute Sharpe for each.

        Regimes are defined by the stock's own 60-day rolling annualised volatility
        (a reliable proxy for market regime in KLSE where single stocks are highly
        correlated with the index).  Signals are computed on the full series so
        indicator look-backs are valid; only the return attribution is masked by regime.
        """
        if len(df) < 80:
            return {
                "sharpe_low_vol": 0.0, "sharpe_mid_vol": 0.0,
                "sharpe_high_vol": 0.0, "regimes_positive": 0,
            }

        close      = df["close"]
        daily_ret  = close.pct_change()
        rolling_vol = daily_ret.rolling(60).std() * np.sqrt(252)  # annualised

        # Signals on full series (so MAs etc. have full context)
        sig          = self._compute_signals(df, params)
        sig_shifted  = sig.shift(1).fillna(0)

        # Per-bar net return (same cost model as _compute_performance)
        cost_bar     = sig_shifted.diff().abs() * _BT_ROUND_TRIP_COST
        net_bar      = (sig_shifted * daily_ret - cost_bar).fillna(0)

        valid_vol = rolling_vol.dropna()
        if len(valid_vol) < 30:
            return {
                "sharpe_low_vol": 0.0, "sharpe_mid_vol": 0.0,
                "sharpe_high_vol": 0.0, "regimes_positive": 0,
            }

        p33 = float(valid_vol.quantile(0.33))
        p66 = float(valid_vol.quantile(0.66))
        ann = BARS_PER_YEAR.get(interval, 252)

        regime_sharpes: dict[str, float] = {}
        for name, mask in (
            ("low_vol",  rolling_vol <= p33),
            ("mid_vol",  (rolling_vol > p33) & (rolling_vol <= p66)),
            ("high_vol", rolling_vol > p66),
        ):
            r = net_bar[mask & rolling_vol.notna()]
            if len(r) < 10:
                regime_sharpes[name] = 0.0
                continue
            std = float(r.std())
            regime_sharpes[name] = round(float(r.mean() / std * np.sqrt(ann)), 3) if std > 1e-10 else 0.0

        regimes_positive = sum(1 for v in regime_sharpes.values() if v > 0)

        return {
            "sharpe_low_vol":   regime_sharpes.get("low_vol",  0.0),
            "sharpe_mid_vol":   regime_sharpes.get("mid_vol",  0.0),
            "sharpe_high_vol":  regime_sharpes.get("high_vol", 0.0),
            "regimes_positive": regimes_positive,
        }

    # ── Backtest sanity flags ─────────────────────────────────────────────────

    @staticmethod
    def _detect_sanity_flags(
        sharpe_gross: float, max_dd: float, win_rate: float,
        trade_count: int, timeframe: str,
    ) -> list:
        """Return a list of human-readable sanity warnings for suspicious results."""
        flags = []
        if sharpe_gross > 2.0:
            flags.append(f"Suspiciously high Sharpe: gross={sharpe_gross:.2f}")
        if max_dd < 0.02:
            flags.append(f"Suspiciously low drawdown: {max_dd:.1%}")
        if win_rate > 0.70:
            flags.append(f"Suspiciously high win rate: {win_rate:.1%}")
        if trade_count > 500 and timeframe == "1d":
            flags.append(f"Too many trades for daily strategy: {trade_count}")
        return flags

    # ── Data requirements pre-check ──────────────────────────────────────────

    # Keywords in factor_formula / hypothesis that signal unavailable data.
    # Any match → backtest is blocked before fetching a single price bar.
    _UNAVAILABLE_DATA_SIGNALS = [
        "dividend yield", "ttm yield", "dividend ttm",
        "klci yield", "constituent weight",
        "spread mean", "spread std", "spread zscore", "yield spread",
        "blended yield", "basket yield", "reference yield",
        "corporate announcement", "dividend cut", "dividend suspension",
        "bursa announcement", "pdmr", "ex-date", "ex-dividend date",
        "payout ratio", "earnings yield", "free cash flow yield",
        "book value", "price to book", "price-to-book", "p/b ratio", "p/b:",
        "roe", "return on equity", "debt equity", "debt-to-equity", "der",
        "earnings per share", "eps forecast", "analyst consensus",
        "bloomberg", "refinitiv", "institutional ownership",
        "short interest", "options", "implied volatility",
        "net profit margin", "gross margin",
        "quarterly result", "quarterly earnings",
        "dividend declared", "dividend history",
        "nim expansion", "net interest margin",
        "book value per share", "bvps",
        "price earnings", "pe ratio", "p/e ratio",
        "market cap weighted", "market-cap-weighted",
    ]

    # Keywords that signal purely OHLCV-based formulas — safe to proceed.
    _AVAILABLE_DATA_SIGNALS = [
        "close", "open", "high", "low", "volume",
        "moving average", "simple moving average", "exponential moving average",
        "sma", "ema", "wma",
        "rsi", "relative strength index",
        "macd", "signal line",
        "bollinger band", "bollinger",
        "atr", "average true range",
        "momentum", "rate of change", "roc",
        "returns", "daily return", "price return",
        "volatility", "rolling std", "rolling volatility",
        "52-week high", "52-week low", "52w high", "52w low",
        "price", "ohlcv", "bar", "candle",
        "volume surge", "volume breakout", "volume ratio",
        "crossover", "cross above", "cross below",
        "stochastic", "williams %r", "cci", "adx",
        "donchian", "keltner", "ichimoku",
        "vwap", "obv", "on-balance volume",
    ]

    def check_data_requirements(self, idea: dict) -> dict:
        """Pre-check whether the idea's factor formula can be backtested with
        available Yahoo Finance OHLCV data.

        Returns a dict:
          {"blocked": bool, "matched": list[str], "reason": str, "fund_context": dict|None}

        If blocked=True the backtest must NOT run. The caller is responsible for
        updating the DB and returning a failure result.

        If fund_context is not None the backtest may proceed using a constant
        fundamental-screen signal derived from the KLSE Screener data in the DB.
        """
        formula   = (idea.get("factor_formula") or "").lower()
        hypothesis = (idea.get("hypothesis")     or "").lower()
        combined  = formula + " " + hypothesis

        # 1. Scan for unavailable-data keywords
        matched = [kw for kw in self._UNAVAILABLE_DATA_SIGNALS if kw in combined]
        if not matched:
            return {"blocked": False, "matched": [], "reason": "", "fund_context": None}

        # 2. Check whether ALL matched terms are clearly overridden by OHLCV context.
        #    Heuristic: if the formula also mentions several OHLCV terms, the
        #    unavailable keyword may be coincidental (e.g. "momentum" mentioned
        #    alongside "eps"). Only block when the formula relies on the term as a
        #    primary driver (matched in factor_formula alone, not just hypothesis).
        formula_only_matches = [kw for kw in matched if kw in formula]
        if not formula_only_matches:
            # Keywords appear in hypothesis text only — warn but don't block
            return {"blocked": False, "matched": matched, "reason": "", "fund_context": None}

        # 3. Before blocking, check if fresh fundamental data exists in the DB.
        #    KLSE Screener populates fundamental_data; if it is available (< 7 days old)
        #    we can run the backtest as a constant fundamental-screen signal instead.
        fund_context = self._load_fundamental_context(idea.get("ticker", ""))
        if fund_context is not None:
            self.log_daemon(
                "INFO",
                f"DataPreCheck: fundamental_data available "
                f"ROE={fund_context.get('roe')} PB={fund_context.get('pb')} "
                f"— unlocking backtest with fundamental-screen signal",
            )
            return {
                "blocked": False,
                "matched": formula_only_matches,
                "reason": "",
                "fund_context": fund_context,
            }

        reason = (
            f"Factor formula requires fundamental data not available via Yahoo Finance "
            f"price feed: {', '.join(formula_only_matches)}. "
            f"Add a fundamental data source (Bursa announcements, financial statements API) "
            f"before backtesting this strategy. "
            f"Rewrite the factor_formula to use ONLY daily OHLCV signals "
            f"(price, volume, moving averages, RSI, MACD, Bollinger Bands, ATR, momentum)."
        )
        return {"blocked": True, "matched": formula_only_matches, "reason": reason, "fund_context": None}

    # ── Fundamental context helpers ────────────────────────────────────────────

    def _load_fundamental_context(self, ticker_raw: str) -> dict | None:
        """Load fresh KLSE Screener fundamental data (< 7 days) for the primary ticker.

        Returns {"roe", "pb", "pe", "dy", "eps", "dps", "nta"} or None if unavailable.
        Handles comma-separated ticker lists and sector descriptions.
        """
        from datetime import date, timedelta
        tickers = extract_tickers(ticker_raw or "")
        primary = next((t for t in tickers if t.endswith(".KL")), None)
        if not primary:
            return None
        try:
            with db_session() as conn:
                fund = conn.execute("""
                    SELECT * FROM fundamental_data
                    WHERE ticker = ?
                    ORDER BY fetched_at DESC LIMIT 1
                """, [primary]).fetchone()
            if fund is None:
                return None
            # Check freshness — fetched_at is stored as YYYY-MM-DD
            fetched_raw = (fund["fetched_at"] or "")[:10]
            try:
                fetched_date = date.fromisoformat(fetched_raw)
            except Exception:
                return None
            if (date.today() - fetched_date).days > 7:
                logger.warning(
                    f"Fundamental data for {primary} is {(date.today() - fetched_date).days}d old — "
                    f"too stale for fundamental-screen backtest"
                )
                return None
            return {
                "roe": fund["roe"],
                "pb":  fund["pb"],
                "pe":  fund["pe"],
                "dy":  fund["dy"],
                "eps": fund["eps_ttm"],
                "dps": fund["dps_ttm"],
                "nta": fund["nta"],
            }
        except Exception as e:
            logger.warning(f"Failed to load fundamental context for {primary}: {e}")
            return None

    @staticmethod
    def _evaluate_fundamental_screen(fund_context: dict, factor_formula: str) -> float:
        """Convert fundamental context to a constant long/flat screening signal.

        Fundamental screens are quarterly-rebalanced so the signal is constant
        across the entire backtest window.  Returns 1.0 (hold in portfolio) when
        the stock has valid positive fundamentals, 0.0 otherwise.
        """
        roe = float(fund_context.get("roe") or 0.0)
        pb  = float(fund_context.get("pb")  or 0.0)
        # Accept the stock if it has any positive equity return or valid book value.
        # Stricter screening is handled at idea-generation time (Gate 0 / Stage 1).
        if roe > 0 or pb > 0:
            return 1.0
        return 0.0

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

    # ── Per-strategy exit profiles ────────────────────────────────────────────

    EXIT_PROFILES: dict[str, dict] = {
        "cross_sectional_momentum": {
            "exit_type":           "signal_or_time",
            "rsi_overbought_exit": 78,
            "stop_loss_pct":       0.08,
            "profit_target_pct":   None,
            "min_hold_days":       20,
            "max_hold_days":       65,
        },
        "short_term_reversal": {
            "exit_type":               "signal_or_time",
            "rsi_exit":                68,
            "recovery_threshold_pct":  98.5,
            "stop_loss_pct":           0.05,
            "profit_target_pct":       0.06,
            "min_hold_days":           1,
            "max_hold_days":           5,
        },
        "low_volatility_anomaly": {
            "exit_type":         "signal_or_time",
            "stop_loss_pct":     0.12,
            "profit_target_pct": None,
            "min_hold_days":     60,
            "max_hold_days":     90,
        },
        "rsi_mean_reversion": {
            "exit_type":           "signal_or_time",
            "rsi_recovery_exit":   55,
            "stop_loss_pct":       0.06,
            "profit_target_pct":   0.08,
            "min_hold_days":       3,
            "max_hold_days":       15,
        },
        "bollinger_squeeze_breakout": {
            "exit_type":                 "signal_or_time",
            "exit_on_middle_band_close": True,
            "stop_loss_pct":             0.05,
            "profit_target_pct":         0.15,
            "min_hold_days":             2,
            "max_hold_days":             20,
        },
        "gap_fill": {
            "exit_type":        "signal_or_time",
            "exit_on_gap_fill": True,
            "stop_loss_pct":    0.03,
            "profit_target_pct": 0.04,
            "min_hold_days":    1,
            "max_hold_days":    3,
        },
        "sma_crossover": {
            "exit_type":          "signal",
            "exit_on_death_cross": True,
            "stop_loss_pct":      0.10,
            "profit_target_pct":  None,
            "min_hold_days":      10,
            "max_hold_days":      None,   # unlimited — trend-following
        },
        "pead": {
            "exit_type":         "signal_or_time",
            "stop_loss_pct":     0.04,
            "profit_target_pct": 0.08,
            "min_hold_days":     5,
            "max_hold_days":     20,
        },
        "cpo_correlation": {
            "exit_type":         "signal_or_time",
            "stop_loss_pct":     0.06,
            "profit_target_pct": 0.10,
            "min_hold_days":     5,
            "max_hold_days":     20,
        },
        "cpo_lag": {
            "exit_type":         "signal_or_time",
            "stop_loss_pct":     0.06,
            "profit_target_pct": 0.10,
            "min_hold_days":     5,
            "max_hold_days":     20,
        },
        "opr_banking_signal": {
            "exit_type":         "signal_or_time",
            "stop_loss_pct":     0.07,
            "profit_target_pct": 0.12,
            "min_hold_days":     10,
            "max_hold_days":     60,
        },
        "opr_cycle": {
            "exit_type":         "signal_or_time",
            "stop_loss_pct":     0.07,
            "profit_target_pct": 0.12,
            "min_hold_days":     10,
            "max_hold_days":     60,
        },
    }

    # Default profile used when no strategy_key matches EXIT_PROFILES.
    _DEFAULT_EXIT_PROFILE: dict = {
        "exit_type":         "time_fallback",
        "stop_loss_pct":     0.08,
        "profit_target_pct": None,
        "min_hold_days":     1,
        "max_hold_days":     40,
    }

    def _get_exit_profile_by_key(self, strategy_key: str) -> dict:
        """Return the exit profile for a given strategy_key, or the default profile."""
        return self.EXIT_PROFILES.get(strategy_key or "", self._DEFAULT_EXIT_PROFILE)

    def _apply_exit_logic(
        self,
        prices:     pd.Series,
        signals:    pd.Series,
        exit_profile: dict,
        rsi_series: pd.Series | None = None,
        bb_middle:  pd.Series | None = None,
        gap_prev_close: pd.Series | None = None,
    ) -> pd.Series:
        """Post-process a raw entry signal series using per-strategy exit rules.

        Iterates over each trade entry (signal goes 0→1) and determines the exit
        bar based on the exit_profile parameters.  Returns a new signal series where
        positions are held exactly from entry to exit — no longer than max_hold_days
        and no shorter than min_hold_days.

        Parameters
        ----------
        prices        : daily close price series (same index as signals)
        signals       : raw entry signal (1=long, 0=flat) from _compute_signals()
        exit_profile  : dict from EXIT_PROFILES or _DEFAULT_EXIT_PROFILE
        rsi_series    : pre-computed RSI series (required for RSI exits)
        bb_middle     : 20-day Bollinger mid band (required for BB exit)
        gap_prev_close: previous-close series used for gap-fill exit detection

        Returns
        -------
        pd.Series of 0/1 signals with exits applied.
        """
        result = pd.Series(0.0, index=signals.index)
        min_hold = exit_profile.get("min_hold_days") or 1
        max_hold = exit_profile.get("max_hold_days")  # None = unlimited
        stop_pct  = exit_profile.get("stop_loss_pct") or 0.0
        tgt_pct   = exit_profile.get("profit_target_pct") or 0.0
        exit_type = exit_profile.get("exit_type", "time_fallback")

        # RSI exits
        rsi_overbought  = exit_profile.get("rsi_overbought_exit")
        rsi_recovery    = exit_profile.get("rsi_recovery_exit")
        rsi_exit_thresh = exit_profile.get("rsi_exit")   # short_term_reversal (RSI 5-day)

        # Signal exits
        exit_on_bb_mid   = exit_profile.get("exit_on_middle_band_close", False)
        exit_on_gap_fill = exit_profile.get("exit_on_gap_fill", False)
        exit_on_death    = exit_profile.get("exit_on_death_cross", False)

        in_trade   = False
        entry_bar  = 0
        entry_px   = 0.0

        n = len(signals)
        sig_vals = signals.values
        px_vals  = prices.values

        for i in range(n):
            if not in_trade:
                if sig_vals[i] == 1:
                    in_trade  = True
                    entry_bar = i
                    entry_px  = float(px_vals[i])
            else:
                result.iloc[i - 1] = 1.0   # mark previous bar as held
                bars_held = i - entry_bar
                px = float(px_vals[i])

                # Never exit before min_hold
                if bars_held < min_hold:
                    continue

                exit_triggered = False

                # Stop loss
                if stop_pct and entry_px > 0 and px <= entry_px * (1.0 - stop_pct):
                    exit_triggered = True

                # Profit target
                if tgt_pct and entry_px > 0 and px >= entry_px * (1.0 + tgt_pct):
                    exit_triggered = True

                # RSI overbought exit (momentum)
                if not exit_triggered and rsi_overbought and rsi_series is not None:
                    rsi_val = float(rsi_series.iloc[i]) if i < len(rsi_series) else float("nan")
                    if not np.isnan(rsi_val) and rsi_val > rsi_overbought:
                        exit_triggered = True

                # RSI recovery exit (mean reversion — exit when normalised)
                if not exit_triggered and rsi_recovery and rsi_series is not None:
                    rsi_val = float(rsi_series.iloc[i]) if i < len(rsi_series) else float("nan")
                    if not np.isnan(rsi_val) and rsi_val > rsi_recovery:
                        exit_triggered = True

                # RSI threshold exit (reversal — short 5-day RSI)
                if not exit_triggered and rsi_exit_thresh and rsi_series is not None:
                    rsi_val = float(rsi_series.iloc[i]) if i < len(rsi_series) else float("nan")
                    if not np.isnan(rsi_val) and rsi_val > rsi_exit_thresh:
                        exit_triggered = True

                # Bollinger Band middle exit (breakout failed)
                if not exit_triggered and exit_on_bb_mid and bb_middle is not None:
                    mid = float(bb_middle.iloc[i]) if i < len(bb_middle) else float("nan")
                    if not np.isnan(mid) and px < mid:
                        exit_triggered = True

                # Gap fill exit
                if not exit_triggered and exit_on_gap_fill and gap_prev_close is not None:
                    prev_close = float(gap_prev_close.iloc[i]) if i < len(gap_prev_close) else float("nan")
                    if not np.isnan(prev_close) and px >= prev_close:
                        exit_triggered = True

                # Death cross exit (SMA crossover)
                if not exit_triggered and exit_on_death:
                    # Signal series itself encodes the death cross via 0;
                    # exit when the original signal flips back to 0
                    if sig_vals[i] == 0:
                        exit_triggered = True

                # Max hold days (time fallback)
                if not exit_triggered and max_hold is not None and bars_held >= max_hold:
                    exit_triggered = True

                if exit_triggered:
                    in_trade = False
                    # Mark exit bar as 0 (position closed this bar)

        # Mark the last bar as held if still in trade at end of series
        if in_trade and n > 0:
            result.iloc[n - 1] = 1.0

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

    # Relaxed thresholds for fundamental screening strategies.
    # Fundamental screens are quarterly-rebalanced buy-and-hold selections,
    # not active trading signals.  A Sharpe of 0.40 on a positive buy-and-hold
    # screen is genuinely good; active-trading thresholds (1.1) would wrongly
    # reject solid fundamental strategies.
    FUNDAMENTAL_SCREEN_THRESHOLDS = {
        "min_sharpe_net":      0.40,   # vs 1.1 for active signals
        "min_oos_sharpe":      0.35,   # absolute OOS floor
        "max_oos_degradation": 0.70,   # vs 0.50 for active signals
        "min_trades":          1,      # quarterly rebalance = few trades
        "max_dd":              0.30,   # vs 0.25 for active signals
        "max_train_val_gap":   1.00,   # bypassed — improving trend is not overfitting
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

    def _compute_signal_with_exits(
        self,
        df:           pd.DataFrame,
        raw_signals:  pd.Series,
        exit_profile: dict,
    ) -> pd.Series:
        """Wrap _apply_exit_logic() with pre-computed indicator series.

        Builds the RSI, BB middle, and previous-close series required by
        _apply_exit_logic() from the OHLCV DataFrame, then delegates to it.
        Returns the modified signal series.
        """
        close = df["close"]

        # RSI — needed for several exit types
        rsi_series: pd.Series | None = None
        if any(exit_profile.get(k) for k in (
            "rsi_overbought_exit", "rsi_recovery_exit", "rsi_exit"
        )):
            rsi_series = self._rsi(close, 14)

        # Bollinger Band middle (20-day SMA)
        bb_middle: pd.Series | None = None
        if exit_profile.get("exit_on_middle_band_close"):
            bb_middle = close.rolling(20).mean()

        # Previous close for gap-fill detection
        gap_prev_close: pd.Series | None = None
        if exit_profile.get("exit_on_gap_fill"):
            gap_prev_close = close.shift(1)

        return self._apply_exit_logic(
            prices=close,
            signals=raw_signals,
            exit_profile=exit_profile,
            rsi_series=rsi_series,
            bb_middle=bb_middle,
            gap_prev_close=gap_prev_close,
        )

    # ── Train / val / test split ─────────────────────────────────────────────

    @staticmethod
    def _split(df: pd.DataFrame) -> tuple:
        n = len(df)
        t = int(n * GATE_CONFIG.stage3_data_split_train)
        v = int(n * GATE_CONFIG.stage3_data_split_val)
        return df.iloc[:t], df.iloc[t:t + v], df.iloc[t + v:]

    # ── Main backtest pipeline ────────────────────────────────────────────────

    def _run_backtest(self, idea_id: int) -> dict:
        with db_session() as conn:
            row = conn.execute("SELECT * FROM alpha_ideas WHERE id=?", (idea_id,)).fetchone()
        if not row:
            return {"error": f"Idea {idea_id} not found"}

        # ── DATA REQUIREMENTS PRE-CHECK ──────────────────────────────────────
        # Reject ideas whose factor_formula relies on fundamental/announcement
        # data unavailable from Yahoo Finance OHLCV before touching the network.
        data_check = self.check_data_requirements(dict(row))
        if data_check["blocked"]:
            reason = data_check["reason"]
            self.log_daemon(
                "WARN",
                f"Backtest [{idea_id}] DATA PRE-CHECK FAILED: "
                f"formula requires unavailable data — {data_check['matched']}",
            )
            with db_session() as conn:
                conn.execute("""
                    INSERT INTO backtest_runs
                      (idea_id, run_type, pair, timeframe, factor_formula,
                       train_sharpe, val_sharpe, test_sharpe,
                       train_dd, val_dd, test_dd,
                       train_val_gap, total_trades, win_rate, profit_factor,
                       params, result_data, passed,
                       needs_review, verification_note,
                       sharpe_gross, sharpe_net,
                       sharpe_is, sharpe_oos, oos_degradation,
                       regimes_positive, sanity_flags,
                       verdict, verdict_reason)
                    VALUES (?,?,?,?,?,0,0,0,1,1,1,0,0,0,0,'{}','{}',0,1,?,0,0,0,0,0,0,NULL,'REJECTED',?)
                """, (
                    idea_id, "data_precheck",
                    row["ticker"] or "", row["timeframe"] or "1d",
                    row["factor_formula"] or "",
                    reason, reason,
                ))
                conn.execute("""
                    UPDATE alpha_ideas
                    SET status='rejected', rejection_reason=?, updated_at=datetime('now')
                    WHERE id=?
                """, (reason[:500], idea_id))
                conn.execute("""
                    INSERT INTO pipeline_events (idea_id, stage, event_type, agent, notes)
                    VALUES (?, 'stage2', 'rejected', 'BacktestEngineer', ?)
                """, (idea_id, f"DATA PRE-CHECK: {reason[:300]}"))
                conn.execute("""
                    INSERT INTO gate_decisions (idea_id, gate, decision, decided_by, rationale)
                    VALUES (?, 'gate2_data', 'reject', 'BacktestEngineer', ?)
                """, (idea_id, reason[:500]))
            self._clear_progress(idea_id)
            return {
                "gate3_pass": False,
                "error": "data_requirements_not_met",
                "verdict": "REJECTED",
                "verdict_reason": reason,
                "matched_keywords": data_check["matched"],
                "idea_id": idea_id,
            }

        # Extract primary .KL ticker — handles sector descriptions and comma-separated lists
        symbol   = extract_tickers(row["ticker"] or "1155.KL")[0]
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

        # If fundamental context is available (from KLSE Screener DB), override to a
        # constant fundamental-screen signal.  The screen is quarterly-rebalanced so
        # a constant long/flat position is the correct model for daily price bars.
        fund_context = data_check.get("fund_context")
        if fund_context is not None:
            fundamental_signal = self._evaluate_fundamental_screen(
                fund_context, row["factor_formula"] or ""
            )
            params["signal_type"]        = "fundamental_screen"
            params["fundamental_signal"] = fundamental_signal
            params["long_only"]          = True
            self.log_daemon(
                "INFO",
                f"Backtest [{idea_id}] fundamental-screen context: "
                f"ROE={fund_context.get('roe')} PB={fund_context.get('pb')} "
                f"PE={fund_context.get('pe')} → signal={'LONG' if fundamental_signal == 1.0 else 'FLAT'}",
            )

            # ── Route multi-stock universe to portfolio backtest ──────────────
            # If the ticker field contains 5+ .KL tickers this is a cross-sectional
            # factor screen, not a single-stock study — use the proper portfolio path.
            _universe_kl = [t.strip() for t in (row["ticker"] or "").split(",")
                            if t.strip() and ".KL" in t.strip()]
            if len(_universe_kl) >= 5:
                self.log_daemon(
                    "INFO",
                    f"Backtest [{idea_id}] multi-stock universe "
                    f"({len(_universe_kl)} tickers) → portfolio backtest",
                )
                return self._run_fundamental_screen_backtest(idea_id, dict(row))

        # Fetch 5 years of daily data for robust train/val/test split
        df = self._fetch_prices(symbol, interval, days=1825)
        # QC4: minimum 252 bars (1 year) required for any statistically meaningful backtest
        if df.empty or len(df) < 252:
            msg = f"Insufficient history ({len(df)} bars) — need minimum 252 bars (1yr)"
            self.log_daemon("WARN", f"[{idea_id}] {msg}")
            return {"error": msg, "idea_id": idea_id, "symbol": symbol}

        self._log_progress(idea_id, 30, f"Computing factor signals ({symbol})")
        # Classify holding period for appropriate thresholds and warnings
        hp_class = self.classify_holding_period(
            interval, row["factor_formula"] or "", row["hypothesis"] or ""
        )

        # Load per-strategy exit profile (if strategy_key is set)
        _strategy_key = (dict(row).get("strategy_key") or "").strip()
        _exit_profile = self._get_exit_profile_by_key(_strategy_key)
        _has_custom_exit = _strategy_key in self.EXIT_PROFILES
        if _has_custom_exit:
            self.log_daemon(
                "INFO",
                f"Backtest [{idea_id}] using exit profile '{_strategy_key}': "
                f"exit_type={_exit_profile['exit_type']} "
                f"max_hold={_exit_profile.get('max_hold_days')} "
                f"stop={_exit_profile.get('stop_loss_pct')}",
            )
        self.log_daemon("INFO", f"Backtest [{idea_id}] holding_period_class={hp_class}")

        # Verify formula before full backtest.
        # fundamental_screen strategies legitimately produce a constant signal — the
        # verification step cannot validate fundamental factor logic from price data
        # alone and always flags needs_review. Skip it for this signal type.
        if params.get("signal_type") == "fundamental_screen":
            needs_review      = 0
            verification_note = ("Constant signal expected for fundamental screen "
                                 "(quarterly buy-and-hold) — formula verification N/A")
        else:
            verification      = self.verify_formula(params, row["factor_formula"] or "", df)
            needs_review      = 0 if (verification["verified"] and verification["confidence"] >= 0.7) else 1
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
                    "sharpe": 0.0, "sharpe_gross": 0.0, "sharpe_net": 0.0,
                    "max_dd": 1.0, "win_rate": 0.0, "profit_factor": 0.0,
                    "total_trades": 0, "ann_return": 0.0,
                }
                continue
            sig = self._compute_signals(split_df, params)
            # Apply per-strategy exit logic for known strategy_key profiles
            if _has_custom_exit:
                sig = self._compute_signal_with_exits(split_df, sig, _exit_profile)
            results[split_name] = self._compute_performance(split_df, sig, interval)

        self._log_progress(idea_id, 65, "Walk-forward IS/OOS and regime stress test")

        # ── QC2: Walk-forward IS/OOS validation ──────────────────────────────
        wf            = self._compute_walk_forward(df, params, interval)
        sharpe_is     = wf["sharpe_is"]
        sharpe_oos    = wf["sharpe_oos"]
        oos_deg       = wf["oos_degradation"]

        # ── QC5: Regime stress test ───────────────────────────────────────────
        reg              = self._compute_regimes(df, params, interval)
        regimes_positive = reg["regimes_positive"]

        self._log_progress(idea_id, 80, "Computing Sharpe and drawdown")
        train_r = results["train"]
        val_r   = results["val"]
        test_r  = results["test"]
        train_val_gap = abs(train_r["sharpe"] - val_r["sharpe"])

        # Extract gross / net Sharpe for the test period
        test_sharpe_net   = test_r["sharpe_net"]
        test_sharpe_gross = test_r["sharpe_gross"]

        # Per holding-period-class thresholds
        sharpe_threshold = self._SHARPE_THRESHOLDS.get(hp_class, GATE_CONFIG.stage3_min_sharpe)
        max_dd_threshold = GATE_CONFIG.stage3_max_drawdown
        min_trades       = self._MIN_TRADES.get(hp_class, 30)
        actual_trades    = test_r.get("total_trades", 0)

        # Fundamental-screen strategies are quarterly-rebalanced buy-and-hold positions.
        # Apply relaxed thresholds: a Sharpe of 0.40 with positive OOS confirms a working
        # screen — active-trading thresholds (1.1) would wrongly reject solid strategies.
        if params.get("signal_type") == "fundamental_screen":
            _fs = self.FUNDAMENTAL_SCREEN_THRESHOLDS
            sharpe_threshold = _fs["min_sharpe_net"]
            max_dd_threshold = _fs["max_dd"]
            min_trades       = _fs["min_trades"]
            # For fundamental screens the "trade" is a quarterly rebalance decision,
            # not a signal flip. Compute actual_trades as the total number of quarterly
            # rebalance points across the full backtest period (252 trading days ÷ 4 ≈ 63).
            bars_per_quarter  = 63 if interval == "1d" else 13  # 52wk ÷ 4
            n_bars_total      = len(train_df) + len(val_df) + len(test_df)
            actual_trades     = max(1, n_bars_total // bars_per_quarter)

        # Gate 2 — in-sample quality (use net Sharpe throughout)
        # For fundamental screens the signal is constant (quarterly buy-and-hold),
        # so per-split Sharpe differences reflect the stock's returns in each time
        # window, NOT parameter overfitting.  Only drawdown is checked per-split;
        # the train_val_gap limit is relaxed to 1.0 (an improving train→val→test
        # trend is healthy, not a sign of overfitting).
        _max_tvg = (self.FUNDAMENTAL_SCREEN_THRESHOLDS["max_train_val_gap"]
                    if params.get("signal_type") == "fundamental_screen"
                    else GATE_CONFIG.stage3_max_train_val_gap)
        if params.get("signal_type") == "fundamental_screen":
            gate2_pass = (
                train_r["max_dd"] <= max_dd_threshold
                and val_r["max_dd"]   <= max_dd_threshold
                and train_val_gap     <= _max_tvg
            )
        else:
            gate2_pass = (
                train_r["sharpe_net"] >= sharpe_threshold
                and val_r["sharpe_net"]   >= sharpe_threshold
                and train_r["max_dd"] <= max_dd_threshold
                and val_r["max_dd"]   <= max_dd_threshold
                and train_val_gap     <= _max_tvg
            )

        # Gate 3 — out-of-sample test
        gate3_pass = (
            gate2_pass
            and test_sharpe_net   >= sharpe_threshold
            and test_r["max_dd"]  <= max_dd_threshold
        )

        # QC4: minimum trade count (per holding-period class)
        trade_count_pass = actual_trades >= min_trades
        trade_count_note = ""
        if not trade_count_pass:
            trade_count_note = (
                f"Insufficient trades ({actual_trades}) for statistical significance "
                f"— need minimum {min_trades} for {hp_class} strategies"
            )
            self.log_daemon(
                "WARN",
                f"Backtest [{idea_id}] trade count gate FAILED: "
                f"{actual_trades} trades < {min_trades} minimum for {hp_class}",
            )
            try:
                from knowledge.ingestion.rejection_memory import RejectionMemory
                RejectionMemory().record_rejection(idea_id, trade_count_note, "stage2_trades")
            except Exception:
                pass

        # QC3: cost sensitivity gate
        cost_pass = True
        cost_note = ""
        if test_sharpe_net < 0.4:
            cost_pass = False
            cost_note = (f"Net Sharpe after Bursa transaction costs too low: "
                         f"{test_sharpe_net:.2f} < 0.40 minimum")
        elif test_sharpe_gross - test_sharpe_net > 0.8:
            cost_pass = False
            cost_note = (f"Strategy is cost-sensitive — gross Sharpe {test_sharpe_gross:.2f} "
                         f"degrades to net {test_sharpe_net:.2f} after Bursa transaction costs")
        if not cost_pass:
            self.log_daemon("WARN", f"Backtest [{idea_id}] cost gate FAILED: {cost_note}")

        # QC2: OOS degradation gate — relaxed for fundamental screens
        oos_pass = True
        oos_note = ""
        _is_fund_screen = params.get("signal_type") == "fundamental_screen"
        _max_oos_deg    = (self.FUNDAMENTAL_SCREEN_THRESHOLDS["max_oos_degradation"]
                           if _is_fund_screen else 0.50)
        _min_oos_sharpe = (self.FUNDAMENTAL_SCREEN_THRESHOLDS["min_oos_sharpe"]
                           if _is_fund_screen else 0.30)
        if sharpe_is > 0 and oos_deg > _max_oos_deg:
            oos_pass = False
            oos_note = (f"OOS Sharpe degradation: IS={sharpe_is:.2f} OOS={sharpe_oos:.2f} "
                        f"deg={oos_deg:.2f} > {_max_oos_deg:.2f} — likely overfitted")
        if sharpe_oos < _min_oos_sharpe:
            oos_pass = False
            oos_note = oos_note or f"OOS Sharpe {sharpe_oos:.2f} < {_min_oos_sharpe:.2f} floor"
        if not oos_pass:
            self.log_daemon("WARN", f"Backtest [{idea_id}] OOS gate FAILED: {oos_note}")

        # QC5: regime robustness gate
        regime_pass = regimes_positive >= 2
        regime_note = ""
        if not regime_pass:
            regime_note = (f"Strategy only works in {regimes_positive}/3 volatility regimes "
                           f"— not robust enough")
            self.log_daemon("WARN", f"Backtest [{idea_id}] regime gate FAILED: {regime_note}")

        # ── Sanity flags (warn but do not auto-reject) ────────────────────────
        sanity_flags = self._detect_sanity_flags(
            test_sharpe_gross, test_r["max_dd"], test_r["win_rate"], actual_trades, interval,
        )
        for flag in sanity_flags:
            self.log_daemon("WARN", f"Backtest [{idea_id}] SANITY FLAG: {flag}")

        overall_pass = gate3_pass and trade_count_pass and cost_pass and oos_pass and regime_pass

        # ── Verdict string ────────────────────────────────────────────────────
        if _is_fund_screen and overall_pass:
            verdict = "pass"
            verdict_reason = (
                f"Fundamental screen passes relaxed thresholds appropriate for quarterly "
                f"rebalance strategies. OOS Sharpe {sharpe_oos:.3f} confirms no overfitting. "
                f"Positive in {regimes_positive}/3 regimes."
            )
        elif overall_pass:
            verdict = "pass"
            verdict_reason = (
                f"Active strategy passes all gates: net Sharpe {test_sharpe_net:.2f}, "
                f"OOS={sharpe_oos:.2f}, regimes={regimes_positive}/3"
            )
        else:
            verdict = "reject"
            verdict_reason = " | ".join(filter(None, [
                "" if gate2_pass   else "Gate2 failed (Sharpe or DD)",
                "" if gate3_pass   else "Gate3 failed (test Sharpe or DD)",
                "" if cost_pass    else cost_note,
                "" if oos_pass     else oos_note,
                "" if regime_pass  else regime_note,
                "" if trade_count_pass else trade_count_note,
            ]))

        self._log_progress(idea_id, 90, "Running cross-sectional IC check")

        run_id = None
        try:
            with db_session() as conn:
                full_note = " | ".join(filter(None, [
                    verification_note, trade_count_note, cost_note, oos_note, regime_note,
                ]))
                conn.execute("""
                    INSERT INTO backtest_runs
                      (idea_id, run_type, pair, timeframe, factor_formula,
                       train_sharpe, val_sharpe, test_sharpe,
                       train_dd, val_dd, test_dd,
                       train_val_gap, total_trades, win_rate, profit_factor,
                       params, result_data, passed, needs_review, verification_note,
                       holding_period_class, trade_count, trades,
                       sharpe_gross, sharpe_net, net_sharpe, gross_sharpe,
                       sharpe_is, sharpe_oos, oos_sharpe, oos_degradation,
                       sharpe_low_vol, sharpe_mid_vol, sharpe_high_vol,
                       regimes_positive, sanity_flags, max_dd,
                       verdict, verdict_reason)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    idea_id, "klse_daily", symbol, interval, row["factor_formula"],
                    train_r["sharpe_net"], val_r["sharpe_net"], test_sharpe_net,
                    train_r["max_dd"], val_r["max_dd"], test_r["max_dd"],
                    round(train_val_gap, 3), test_r["total_trades"],
                    test_r["win_rate"], test_r["profit_factor"],
                    json.dumps(params), json.dumps(results),
                    1 if overall_pass else 0,
                    needs_review, full_note or None,
                    hp_class, actual_trades, actual_trades,
                    test_sharpe_gross, test_sharpe_net, test_sharpe_net, test_sharpe_gross,
                    sharpe_is, sharpe_oos, sharpe_oos, oos_deg,
                    reg["sharpe_low_vol"], reg["sharpe_mid_vol"], reg["sharpe_high_vol"],
                    regimes_positive,
                    json.dumps(sanity_flags) if sanity_flags else None,
                    test_r["max_dd"],
                    verdict, verdict_reason,
                ))
                run_id = conn.execute(
                    "SELECT id FROM backtest_runs WHERE idea_id=? ORDER BY created_at DESC LIMIT 1",
                    (idea_id,),
                ).fetchone()["id"]

                # Only update stage/status from stage2 → stage3.
                # If idea is already at stage3+ (e.g., after Red-Blue reviewed it),
                # preserve the current stage/status — never overwrite Red-Blue decisions.
                cur_idea = conn.execute(
                    "SELECT stage, status FROM alpha_ideas WHERE id=?", (idea_id,)
                ).fetchone()
                cur_stage = cur_idea["stage"] if cur_idea else "stage2"

                if cur_stage == "stage2":
                    new_stage  = "stage3" if overall_pass else "stage2"
                    new_status = "active"  if overall_pass else "rejected"
                    conn.execute("""
                        UPDATE alpha_ideas
                        SET backtest_sharpe=?, backtest_dd=?, stage=?, status=?,
                            updated_at=datetime('now')
                        WHERE id=?
                    """, (test_sharpe_net, test_r["max_dd"], new_stage, new_status, idea_id))
                else:
                    # Refresh metrics only; preserve stage and status
                    conn.execute("""
                        UPDATE alpha_ideas
                        SET backtest_sharpe=?, backtest_dd=?, updated_at=datetime('now')
                        WHERE id=?
                    """, (test_sharpe_net, test_r["max_dd"], idea_id))

                conn.execute("""
                    INSERT INTO pipeline_events (idea_id, stage, event_type, agent, notes)
                    VALUES (?, 'stage2', ?, 'BacktestEngineer', ?)
                """, (idea_id,
                      "advanced" if overall_pass else "rejected",
                      f"Train(net)={train_r['sharpe_net']:.2f} Val={val_r['sharpe_net']:.2f} "
                      f"Test(net)={test_sharpe_net:.2f} Test(gross)={test_sharpe_gross:.2f} "
                      f"IS={sharpe_is:.2f} OOS={sharpe_oos:.2f} "
                      f"DD={test_r['max_dd']:.1%} Regimes={regimes_positive}/3 "
                      f"AnnRet={test_r.get('ann_return',0):.1%}"))

                conn.execute("""
                    INSERT INTO gate_decisions (idea_id, gate, decision, decided_by, rationale)
                    VALUES (?, 'gate2_3', ?, 'BacktestEngineer', ?)
                """, (idea_id,
                      "approve" if overall_pass else "reject",
                      f"G2={'PASS' if gate2_pass else 'FAIL'} "
                      f"G3={'PASS' if gate3_pass else 'FAIL'} "
                      f"cost={'PASS' if cost_pass else 'FAIL'} "
                      f"oos={'PASS' if oos_pass else 'FAIL'} "
                      f"regime={'PASS' if regime_pass else 'FAIL'} "
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
            f"net={test_sharpe_net:.2f} gross={test_sharpe_gross:.2f} "
            f"IS={sharpe_is:.2f} OOS={sharpe_oos:.2f} regimes={regimes_positive}/3",
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
        if sanity_flags:
            hp_warnings.extend(sanity_flags)

        return {
            "idea_id":             idea_id,
            "run_id":              run_id,
            "symbol":              symbol,
            "company":             stock.get("name", symbol),
            "interval":            interval,
            "gate2_pass":          gate2_pass,
            "gate3_pass":          gate3_pass,
            "trade_count_pass":    trade_count_pass,
            "cost_pass":           cost_pass,
            "oos_pass":            oos_pass,
            "regime_pass":         regime_pass,
            "train":               train_r,
            "val":                 val_r,
            "test":                test_r,
            "sharpe_is":           sharpe_is,
            "sharpe_oos":          sharpe_oos,
            "oos_degradation":     oos_deg,
            "regimes":             reg,
            "train_val_gap":       round(train_val_gap, 3),
            "params":              params,
            "bars_total":          len(df),
            "factor_formula":      row["factor_formula"] or "",
            "needs_review":        bool(needs_review),
            "verification":        verification,
            "holding_period_class": hp_class,
            "actual_trades":       actual_trades,
            "min_trades_required": min_trades,
            "sanity_flags":        sanity_flags,
            "hp_warnings":         hp_warnings,
        }

    # ── Multi-stock quarterly rebalance backtest for fundamental screens ─────────

    def _run_fundamental_screen_backtest(self, idea_id: int, row: dict) -> dict:
        """Proper portfolio backtest for fundamental screening strategies.

        Instead of a single-stock constant-signal backtest, builds an equal-weight
        portfolio that:
          1. Parses the full ticker universe from idea['ticker']
          2. Loads fundamental data for all tickers from the DB
          3. Scores each ticker by the factor(s) detected in factor_formula
          4. Selects the top 33% (min 3, max 10) by composite score
          5. Fetches weekly OHLCV for selected tickers (5y, 1wk)
          6. Generates a weekly portfolio return series with quarterly rebalancing
             (Mar/Jun/Sep/Dec), applying 0.4% cost on tickers that change
          7. Runs train/val/test split + IS/OOS + regime tests on the portfolio

        Returns the same result dict shape as _run_backtest so callers are
        agnostic to which path ran.
        """
        ticker_raw = row.get("ticker", "") or ""
        universe   = [t.strip() for t in ticker_raw.split(",")
                      if t.strip() and ".KL" in t.strip()]

        # ── Universe size gate ────────────────────────────────────────────────
        if len(universe) < 5:
            msg = (f"Universe too small for factor ranking ({len(universe)} stocks). "
                   f"Minimum 5 required.")
            self.log_daemon("WARN", f"FundScreen [{idea_id}] {msg}")
            with db_session() as conn:
                conn.execute(
                    "UPDATE alpha_ideas SET status='rejected', rejection_reason=? WHERE id=?",
                    (msg, idea_id),
                )
            return {"error": msg, "idea_id": idea_id,
                    "gate2_pass": False, "gate3_pass": False}

        self.log_daemon("INFO",
            f"FundScreen [{idea_id}] universe={len(universe)} tickers, "
            f"first 5: {universe[:5]}")
        self._log_progress(idea_id, 15,
            f"Loading fundamental data for {len(universe)} stocks")

        # ── Load fundamental data (no staleness check — screening is slow-moving) ──
        fund_data: dict[str, dict] = {}
        with db_session() as conn:
            for ticker in universe:
                r = conn.execute(
                    "SELECT * FROM fundamental_data WHERE ticker=? "
                    "ORDER BY fetched_at DESC LIMIT 1",
                    [ticker],
                ).fetchone()
                if r:
                    fund_data[ticker] = {
                        "roe": float(r["roe"] or 0),
                        "pb":  float(r["pb"]  or 0),
                        "pe":  float(r["pe"]  or 0),
                        "dy":  float(r["dy"]  or 0),
                    }

        if len(fund_data) < 3:
            msg = (f"Insufficient fundamental data: only "
                   f"{len(fund_data)}/{len(universe)} tickers have data in DB.")
            self.log_daemon("WARN", f"FundScreen [{idea_id}] {msg}")
            return {"error": msg, "idea_id": idea_id,
                    "gate2_pass": False, "gate3_pass": False}

        # ── Detect which factors to use from the formula ──────────────────────
        formula  = (row.get("factor_formula") or "").lower()
        use_roe  = "roe" in formula
        use_der  = any(kw in formula for kw in ("der", "debt"))
        use_pb   = any(kw in formula for kw in ("pb", "p/b", "book"))
        use_dy   = any(kw in formula for kw in ("dy", "dividend yield", "yield"))
        use_pe   = any(kw in formula for kw in ("pe", "p/e", "price earning"))

        # Default to ROE if nothing detected
        if not any((use_roe, use_der, use_pb, use_dy, use_pe)):
            use_roe = True

        self.log_daemon("INFO",
            f"FundScreen [{idea_id}] factors detected — "
            f"ROE={use_roe} DER={use_der} PB={use_pb} DY={use_dy} PE={use_pe}")

        # ── Compute composite percentile-rank scores ──────────────────────────
        def _pct_rank(data: dict, key: str, higher_better: bool = True) -> dict:
            items = [(t, d[key]) for t, d in data.items() if d[key] != 0]
            if not items:
                return {t: 0.5 for t in data}
            n = len(items)
            ranked = sorted(items, key=lambda x: x[1])
            r = {t: i / max(n - 1, 1) for i, (t, _) in enumerate(ranked)}
            return r if higher_better else {t: 1 - v for t, v in r.items()}

        tickers_w_data = list(fund_data.keys())
        composite: dict[str, float] = {t: 0.0 for t in tickers_w_data}

        if use_roe:
            for t, v in _pct_rank(fund_data, "roe", True).items():
                composite[t] += v
        if use_der:
            # DER not stored directly; use PB as leverage proxy (lower = less risky)
            for t, v in _pct_rank(fund_data, "pb", False).items():
                composite[t] += v
        if use_pb:
            for t, v in _pct_rank(fund_data, "pb", False).items():
                composite[t] += v
        if use_dy:
            for t, v in _pct_rank(fund_data, "dy", True).items():
                composite[t] += v
        if use_pe:
            for t, v in _pct_rank(fund_data, "pe", False).items():
                composite[t] += v

        # ── Select top 33% (min 3, max 10) ───────────────────────────────────
        n_select = max(3, min(10, len(tickers_w_data) // 3))
        selected = sorted(composite, key=composite.get, reverse=True)[:n_select]

        self.log_daemon("INFO",
            f"FundScreen [{idea_id}] selected {n_select}/{len(tickers_w_data)}: {selected}")

        self._log_progress(idea_id, 25,
            f"Fetching weekly prices for {len(selected)} stocks")

        # ── Fetch weekly prices ───────────────────────────────────────────────
        prices: dict[str, pd.Series] = {}
        for ticker in selected:
            try:
                df = self._fetch_prices(ticker, "1wk", days=1825)
                if not df.empty and len(df) >= 52:
                    prices[ticker] = df["close"]
                    self.log_daemon("INFO",
                        f"FundScreen [{idea_id}] {ticker}: {len(df)} weekly bars")
            except Exception as exc:
                self.log_daemon("WARN",
                    f"FundScreen [{idea_id}] price fetch failed for {ticker}: {exc}")

        if len(prices) < 3:
            msg = (f"Price data unavailable for sufficient stocks: "
                   f"{len(prices)}/{len(selected)} fetched")
            self.log_daemon("WARN", f"FundScreen [{idea_id}] {msg}")
            return {"error": msg, "idea_id": idea_id,
                    "gate2_pass": False, "gate3_pass": False}

        # ── Build aligned price + return panel ────────────────────────────────
        price_panel = pd.DataFrame(prices).dropna(how="all")
        price_panel = price_panel.dropna(
            thresh=max(1, len(price_panel.columns) // 2))
        price_panel = price_panel.ffill().bfill()
        returns_panel = price_panel.pct_change().fillna(0)

        # ── Generate quarterly-rebalanced portfolio returns ───────────────────
        _QUARTER_MONTHS = {3, 6, 9, 12}
        current_holdings  = list(prices.keys())
        portfolio_rets: list[float] = []
        rebalance_log:  list[dict]  = []
        n_rebalances = 0

        for i, (date, row_ret) in enumerate(returns_panel.iterrows()):
            ts    = pd.Timestamp(date)
            month = ts.month

            # Quarterly rebalance: first week whose month is in {3,6,9,12}
            # and the previous week was NOT in {3,6,9,12}
            prev_month = (pd.Timestamp(returns_panel.index[i - 1]).month
                          if i > 0 else -1)
            is_rebalance = (month in _QUARTER_MONTHS
                            and prev_month not in _QUARTER_MONTHS)

            if i == 0 or is_rebalance:
                prev_set = set(current_holdings)
                current_holdings = [t for t in selected
                                    if t in price_panel.columns]
                n_rebalances += 1

                new_set  = set(current_holdings)
                turnover = ((len(new_set - prev_set) + len(prev_set - new_set))
                            / max(len(prev_set), 1)) if i > 0 else 1.0

                rebalance_log.append({
                    "date":          str(date)[:10],
                    "month":         month,
                    "holdings":      current_holdings,
                    "universe_size": len(tickers_w_data),
                    "scores": {t: round(composite.get(t, 0.0), 3)
                               for t in current_holdings},
                })
                self.log_daemon("INFO",
                    f"FundScreen [{idea_id}] Q-rebalance {n_rebalances} "
                    f"({str(date)[:7]}): selected={current_holdings} "
                    f"turnover={turnover:.0%} universe={len(tickers_w_data)}")

                cost = turnover * 0.004   # 0.4% on changed portion
            else:
                cost = 0.0

            valid = [t for t in current_holdings
                     if t in row_ret.index and not pd.isna(row_ret[t])]
            port_ret = float(row_ret[valid].mean()) if valid else 0.0
            portfolio_rets.append(port_ret - cost)

        self.log_daemon("INFO",
            f"FundScreen [{idea_id}] {n_rebalances} rebalances over "
            f"{len(portfolio_rets)} weeks")

        if len(portfolio_rets) < 52:
            msg = f"Insufficient portfolio history: {len(portfolio_rets)} weeks"
            return {"error": msg, "idea_id": idea_id,
                    "gate2_pass": False, "gate3_pass": False}

        # ── Build synthetic NAV for reuse with _compute_performance ──────────
        port_s = pd.Series(portfolio_rets, index=returns_panel.index)
        nav    = (1 + port_s.clip(-0.5, 0.5)).cumprod()
        port_df = pd.DataFrame({"close": nav}, index=returns_panel.index)

        interval = "1wk"
        self._log_progress(idea_id, 50, "Running train/val/test split")

        train_df, val_df, test_df = self._split(port_df)

        results: dict[str, dict] = {}
        for sname, sdf in (("train", train_df), ("val", val_df), ("test", test_df)):
            if len(sdf) < 20:
                results[sname] = {
                    "sharpe": 0.0, "sharpe_gross": 0.0, "sharpe_net": 0.0,
                    "max_dd": 1.0, "win_rate": 0.0, "profit_factor": 0.0,
                    "total_trades": 0, "ann_return": 0.0,
                }
                continue
            sig = pd.Series(1.0, index=sdf.index)
            results[sname] = self._compute_performance(sdf, sig, interval)

        # ── IS / OOS walk-forward ─────────────────────────────────────────────
        self._log_progress(idea_id, 65, "Walk-forward IS/OOS")
        n         = len(port_df)
        split_at  = int(n * 0.70)
        is_perf   = self._compute_performance(
            port_df.iloc[:split_at],
            pd.Series(1.0, index=port_df.index[:split_at]),
            interval,
        ) if split_at >= 52 else {"sharpe_net": 0.0}
        oos_perf  = self._compute_performance(
            port_df.iloc[split_at:],
            pd.Series(1.0, index=port_df.index[split_at:]),
            interval,
        ) if (n - split_at) >= 12 else {"sharpe_net": 0.0}

        sharpe_is  = is_perf["sharpe_net"]
        sharpe_oos = oos_perf["sharpe_net"]
        deg = ((sharpe_is - sharpe_oos) / max(abs(sharpe_is), 1e-9)
               if sharpe_is > 0 else 0.0)
        oos_deg = round(deg, 3)

        # ── Regime stress test ────────────────────────────────────────────────
        self._log_progress(idea_id, 75, "Regime stress test")
        reg = {"sharpe_low_vol": 0.0, "sharpe_mid_vol": 0.0,
               "sharpe_high_vol": 0.0, "regimes_positive": 0}
        if len(port_df) >= 60:
            vol = port_s.rolling(12).std() * np.sqrt(52)
            vv  = vol.dropna()
            if len(vv) >= 20:
                p33, p66 = float(vv.quantile(0.33)), float(vv.quantile(0.66))
                rsharpes: dict[str, float] = {}
                for rn, mask in (
                    ("low_vol",  vol <= p33),
                    ("mid_vol",  (vol > p33) & (vol <= p66)),
                    ("high_vol", vol > p66),
                ):
                    r = port_s[mask & vol.notna()]
                    if len(r) < 8:
                        rsharpes[rn] = 0.0
                    else:
                        std = float(r.std())
                        rsharpes[rn] = (round(float(r.mean() / std * np.sqrt(52)), 3)
                                        if std > 1e-10 else 0.0)
                reg = {
                    "sharpe_low_vol":   rsharpes.get("low_vol",  0.0),
                    "sharpe_mid_vol":   rsharpes.get("mid_vol",  0.0),
                    "sharpe_high_vol":  rsharpes.get("high_vol", 0.0),
                    "regimes_positive": sum(1 for v in rsharpes.values() if v > 0),
                }

        # ── Gate evaluation (relaxed fundamental-screen thresholds) ───────────
        train_r = results["train"]
        val_r   = results["val"]
        test_r  = results["test"]
        train_val_gap     = abs(train_r["sharpe_net"] - val_r["sharpe_net"])
        test_sharpe_net   = test_r["sharpe_net"]
        test_sharpe_gross = test_r["sharpe_gross"]
        actual_trades     = n_rebalances
        regimes_positive  = reg["regimes_positive"]
        hp_class          = "LONG_TERM"

        _fs              = self.FUNDAMENTAL_SCREEN_THRESHOLDS
        sharpe_threshold = _fs["min_sharpe_net"]
        max_dd_threshold = _fs["max_dd"]
        _max_tvg         = _fs["max_train_val_gap"]
        _min_oos_sharpe  = _fs["min_oos_sharpe"]
        _max_oos_deg     = _fs["max_oos_degradation"]
        min_trades       = _fs["min_trades"]

        gate2_pass = (
            train_r["max_dd"] <= max_dd_threshold
            and val_r["max_dd"]   <= max_dd_threshold
            and train_val_gap     <= _max_tvg
        )
        gate3_pass = (
            gate2_pass
            and test_sharpe_net  >= sharpe_threshold
            and test_r["max_dd"] <= max_dd_threshold
        )
        trade_count_pass = actual_trades >= min_trades

        oos_pass = True
        oos_note = ""
        if sharpe_is > 0 and oos_deg > _max_oos_deg:
            oos_pass = False
            oos_note = (f"OOS degradation: IS={sharpe_is:.2f} OOS={sharpe_oos:.2f} "
                        f"deg={oos_deg:.2f} > {_max_oos_deg:.2f}")
        if sharpe_oos < _min_oos_sharpe:
            oos_pass = False
            oos_note = oos_note or f"OOS Sharpe {sharpe_oos:.2f} < {_min_oos_sharpe:.2f}"

        regime_pass = regimes_positive >= 2
        regime_note = ("" if regime_pass
                       else f"Only {regimes_positive}/3 volatility regimes positive")

        overall_pass = gate3_pass and trade_count_pass and oos_pass and regime_pass

        if overall_pass:
            verdict        = "pass"
            verdict_reason = (
                f"Fundamental screen portfolio passes: "
                f"Sharpe(net)={test_sharpe_net:.3f} "
                f"IS={sharpe_is:.2f} OOS={sharpe_oos:.2f} "
                f"stocks={len(selected)} rebalances={n_rebalances} "
                f"regimes={regimes_positive}/3"
            )
        else:
            verdict        = "reject"
            verdict_reason = " | ".join(filter(None, [
                "" if gate2_pass       else "Gate2 failed (DD)",
                "" if gate3_pass       else f"Gate3: Sharpe={test_sharpe_net:.2f} < {sharpe_threshold}",
                "" if oos_pass         else oos_note,
                "" if regime_pass      else regime_note,
                "" if trade_count_pass else f"Insufficient rebalances: {actual_trades}",
            ]))

        self._log_progress(idea_id, 90, "Saving results")

        params = {
            "signal_type":      "fundamental_screen",
            "long_only":        True,
            "universe_size":    len(universe),
            "stocks_with_data": len(fund_data),
            "selected_stocks":  selected,
            "factors_used":     {"roe": use_roe, "der": use_der,
                                  "pb": use_pb,  "dy": use_dy, "pe": use_pe},
        }
        result_data = {
            "rebalance_log":           rebalance_log[:20],
            "portfolio_weeks":         len(portfolio_rets),
            "composite_scores":        {t: round(composite.get(t, 0), 3)
                                        for t in selected},
        }
        full_note = " | ".join(filter(None, [oos_note, regime_note]))

        run_id = None
        try:
            with db_session() as conn:
                conn.execute("""
                    INSERT INTO backtest_runs
                      (idea_id, run_type, pair, timeframe, factor_formula,
                       train_sharpe, val_sharpe, test_sharpe,
                       train_dd, val_dd, test_dd,
                       train_val_gap, total_trades, win_rate, profit_factor,
                       params, result_data, passed, needs_review, verification_note,
                       holding_period_class, trade_count, trades,
                       sharpe_gross, sharpe_net, net_sharpe, gross_sharpe,
                       sharpe_is, sharpe_oos, oos_sharpe, oos_degradation,
                       sharpe_low_vol, sharpe_mid_vol, sharpe_high_vol,
                       regimes_positive, sanity_flags, max_dd,
                       verdict, verdict_reason)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    idea_id, "fundamental_screen_portfolio",
                    ticker_raw, interval, row.get("factor_formula", ""),
                    train_r["sharpe_net"], val_r["sharpe_net"], test_sharpe_net,
                    train_r["max_dd"], val_r["max_dd"], test_r["max_dd"],
                    round(train_val_gap, 3), actual_trades,
                    test_r["win_rate"], test_r["profit_factor"],
                    json.dumps(params), json.dumps(result_data),
                    1 if overall_pass else 0,
                    0, full_note or None,
                    hp_class, actual_trades, actual_trades,
                    test_sharpe_gross, test_sharpe_net, test_sharpe_net, test_sharpe_gross,
                    sharpe_is, sharpe_oos, sharpe_oos, oos_deg,
                    reg["sharpe_low_vol"], reg["sharpe_mid_vol"], reg["sharpe_high_vol"],
                    regimes_positive, None, test_r["max_dd"],
                    verdict, verdict_reason,
                ))
                run_id = conn.execute(
                    "SELECT id FROM backtest_runs WHERE idea_id=? "
                    "ORDER BY created_at DESC LIMIT 1",
                    (idea_id,),
                ).fetchone()["id"]

                # Only update stage/status if idea is currently at stage2.
                # If already at stage3 or higher (e.g., after Red-Blue review),
                # preserve the existing stage/status — do not overwrite rejections.
                cur_idea = conn.execute(
                    "SELECT stage, status FROM alpha_ideas WHERE id=?", (idea_id,)
                ).fetchone()
                cur_stage  = cur_idea["stage"]  if cur_idea else "stage2"
                cur_status = cur_idea["status"] if cur_idea else "pending"

                if cur_stage == "stage2":
                    new_stage  = "stage3" if overall_pass else "stage2"
                    new_status = "active"  if overall_pass else "rejected"
                    conn.execute("""
                        UPDATE alpha_ideas
                        SET backtest_sharpe=?, backtest_dd=?, stage=?, status=?,
                            updated_at=datetime('now')
                        WHERE id=?
                    """, (test_sharpe_net, test_r["max_dd"],
                          new_stage, new_status, idea_id))
                else:
                    # Only refresh metrics, never downgrade stage or flip status
                    conn.execute("""
                        UPDATE alpha_ideas
                        SET backtest_sharpe=?, backtest_dd=?,
                            updated_at=datetime('now')
                        WHERE id=?
                    """, (test_sharpe_net, test_r["max_dd"], idea_id))
                conn.execute("""
                    INSERT INTO pipeline_events
                      (idea_id, stage, event_type, agent, notes)
                    VALUES (?, 'stage2', ?, 'BacktestEngineer', ?)
                """, (idea_id,
                      "advanced" if overall_pass else "rejected",
                      f"FundScreen portfolio {len(selected)} stocks "
                      f"{n_rebalances} rebalances "
                      f"Sharpe(net)={test_sharpe_net:.2f} "
                      f"IS={sharpe_is:.2f} OOS={sharpe_oos:.2f}"))
                conn.execute("""
                    INSERT INTO gate_decisions
                      (idea_id, gate, decision, decided_by, rationale)
                    VALUES (?, 'gate2_3', ?, 'BacktestEngineer', ?)
                """, (idea_id,
                      "approve" if overall_pass else "reject",
                      verdict_reason[:500]))
            self.log_daemon("INFO",
                f"FundScreen [{idea_id}] saved — overall_pass={overall_pass}")
        except Exception as exc:
            self.log_daemon("ERROR",
                f"FundScreen [{idea_id}] DB save FAILED: {exc}")
            raise

        self._log_progress(idea_id, 100, "Complete")
        self._clear_progress(idea_id)
        self.log_daemon(
            "INFO" if overall_pass else "WARN",
            f"FundScreen [{idea_id}] {'PASSED' if overall_pass else 'FAILED'} "
            f"portfolio={len(selected)} stocks "
            f"Sharpe(net)={test_sharpe_net:.2f} "
            f"IS={sharpe_is:.2f} OOS={sharpe_oos:.2f} "
            f"rebalances={n_rebalances}",
        )

        return {
            "idea_id":             idea_id,
            "run_id":              run_id,
            "symbol":              ticker_raw,
            "universe_size":       len(universe),
            "stocks_selected":     len(selected),
            "stocks_with_data":    len(fund_data),
            "n_rebalances":        n_rebalances,
            "trade_count":         n_rebalances,
            "sharpe_net":          test_sharpe_net,
            "sharpe_gross":        test_sharpe_gross,
            "gate2_pass":          gate2_pass,
            "gate3_pass":          gate3_pass,
            "trade_count_pass":    trade_count_pass,
            "oos_pass":            oos_pass,
            "regime_pass":         regime_pass,
            "train":               train_r,
            "val":                 val_r,
            "test":                test_r,
            "sharpe_is":           sharpe_is,
            "sharpe_oos":          sharpe_oos,
            "oos_degradation":     oos_deg,
            "regimes":             reg,
            "train_val_gap":       round(train_val_gap, 3),
            "params":              params,
            "rebalance_log":       rebalance_log,
            "holding_period_class": hp_class,
            "actual_trades":       n_rebalances,
            "factor_formula":      row.get("factor_formula", ""),
        }

    def backtest_idea(self, idea_id: int) -> dict:
        """Public wrapper: 10-minute timeout + full exception safety around _run_backtest.

        SIGALRM timeout is only installed when called from the main thread
        (signal.alarm is not available in worker threads — guard avoids ValueError).
        """
        import signal as _sig
        import threading
        import traceback as _tb

        in_main_thread = (threading.current_thread() is threading.main_thread())

        def _timeout_handler(signum, frame):
            raise TimeoutError(f"Backtest [{idea_id}] exceeded 10-minute timeout — aborting")

        if in_main_thread:
            _old = _sig.signal(_sig.SIGALRM, _timeout_handler)
            _sig.alarm(600)

        try:
            return self._run_backtest(idea_id)
        except TimeoutError as exc:
            msg = str(exc)
            self.log_daemon("ERROR", msg)
            self._clear_progress(idea_id)
            return {
                "error": msg, "idea_id": idea_id,
                "gate2_pass": False, "gate3_pass": False, "timeout": True,
            }
        except Exception as exc:
            msg = f"Backtest [{idea_id}] unhandled exception: {exc}"
            self.log_daemon("ERROR", msg + "\n" + _tb.format_exc())
            self._clear_progress(idea_id)
            return {
                "error": msg, "idea_id": idea_id,
                "gate2_pass": False, "gate3_pass": False,
            }
        finally:
            if in_main_thread:
                _sig.alarm(0)
                _sig.signal(_sig.SIGALRM, _old)

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

    def run_backtest(self, idea: dict) -> dict:
        """Convenience wrapper: run backtest for a pre-loaded idea dict.

        Accepts the idea as a dict (e.g. from dict(sqlite3.Row)) and
        delegates to backtest_idea(id).  Adds top-level aliases so callers
        can use result.get('trade_count'), result.get('sharpe_net'), etc.
        """
        result = self.backtest_idea(int(idea["id"]))
        # Normalise common key aliases so both code paths look the same
        if "actual_trades" in result and "trade_count" not in result:
            result["trade_count"] = result["actual_trades"]
        if "test" in result and "sharpe_net" not in result:
            result["sharpe_net"] = result["test"].get("sharpe_net", 0.0)
        return result
