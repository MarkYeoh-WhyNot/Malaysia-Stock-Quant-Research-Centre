import json
import logging
import os
import numpy as np
import pandas as pd
from agents.base_agent import BaseAgent
from config.settings import (
    MODEL_MAIN, GATE_CONFIG, KLCI_BY_SYMBOL, DEFAULT_SYMBOLS,
    PAPER_CAPITAL_MYR, PAPER_ALLOC_PCT, BURSA_MIN_DAILY_VALUE_MYR,
    bursa_trade_cost, bursa_slippage_tier,
    MARKET_RULES_VERSION, FEE_MODEL_VERSION,
    BENCHMARK_SYMBOL,
    FUNDING_INTERVAL_HOURS, AVG_FUNDING_RATE_PER_INTERVAL,
)
from data.database import db_session


def _stamp_versions(conn):
    """Stamp the active market-rule / fee-model versions onto the row just
    inserted on this connection (uses last_insert_rowid()). Traceability so any
    backtest_runs row can be tied back to the cost assumptions in force."""
    conn.execute(
        "UPDATE backtest_runs SET market_rules_version=?, fee_model_version=? "
        "WHERE id=last_insert_rowid()",
        (MARKET_RULES_VERSION, FEE_MODEL_VERSION),
    )
from data.market_data import extract_tickers, get_historical_data, BARS_PER_YEAR
from config.settings import bars_per_day as _bars_per_day
from config.settings import FETCH_DAYS_BY_INTERVAL as _FETCH_DAYS
from agents.backtest_engineer import engine
from agents.backtest_engineer.engine import _funding_bar_sum
from agents.backtest_engineer import signal_parsing
from agents.backtest_engineer.signal_parsing import SYSTEM
from agents.backtest_engineer import cross_sectional
from agents.backtest_engineer import gates
from agents.backtest_engineer import fundamental_screen

logger = logging.getLogger(__name__)

from config.settings import PROGRESS_FILE as _PROGRESS_FILE_PATH
_PROGRESS_FILE = str(_PROGRESS_FILE_PATH)

# Capacity-adjusted-Sharpe impact coefficient (disclosed rough estimate, used
# ONLY for the reported capacity haircut — never the gated number): market
# impact per side ≈ this × ADV participation. 5% ADV participation ⇒ ~0.5%/side.
_CAPACITY_IMPACT_COEF = 0.10


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

    def _fetch_funding_history(self, symbol: str, days: int = 1825) -> pd.DataFrame:
        """Cached historical funding settlements (empty on Bursa/yahoo backend)."""
        try:
            from agents.data_engineer.data_engineer import DataEngineer
            return DataEngineer().fetch_funding(symbol, days=days, use_cache=True)
        except Exception:
            from data.market_data import get_funding_rate_history
            return get_funding_rate_history(symbol, days=days)

    def _equal_weight_klci_returns(self, interval: str = "1d", days: int = 1825) -> pd.Series:
        """Daily equal-weight KLCI buy-and-hold return series (Phase 3.2 benchmark).

        The audit (§8.4) requires every strategy to beat a *simple* baseline;
        equal-weight KLCI is the harder of the two it lists. Fetching 30
        constituents is expensive, so the series is memoised on the instance per
        interval — the daemon reuses one BacktestEngineer across ideas, so the
        universe is fetched at most once per process run (all cached).
        """
        cache = getattr(self, "_ew_ret_cache", None)
        if cache is None:
            cache = self._ew_ret_cache = {}
        if interval not in cache:
            rets = []
            for sym in DEFAULT_SYMBOLS:
                try:
                    d = self._fetch_prices(sym, interval, days=days)
                    if not d.empty and "close" in d:
                        rets.append(d["close"].pct_change())
                except Exception:
                    continue
            cache[interval] = (
                pd.concat(rets, axis=1).mean(axis=1) if rets else pd.Series(dtype=float)
            )
        return cache[interval]

    # ── Factor parsing ────────────────────────────────────────────────────────

    def _parse_factor(self, factor_formula: str, title: str, hypothesis: str) -> dict:
        """Parse the idea's free text into a signal DSL condition tree.

        Honesty contract: if the idea cannot be expressed with the available
        leaves, the parser must say so ({"representable": false, "reason"})
        and the idea is rejected with that reason — it is NEVER silently
        genericized onto the nearest template. See signal_parsing.parse_factor
        for the full implementation (kept as a thin instance-method delegate
        here so instance-level test/harness monkeypatches of _parse_factor
        keep intercepting it — see signal_parsing.py's module docstring).
        """
        return signal_parsing.parse_factor(self, factor_formula, title, hypothesis)

    # ── Signal computation ────────────────────────────────────────────────────

    def _fetch_cpo_series(self, index: pd.DatetimeIndex) -> pd.Series:
        """CPO futures closes aligned to the given index (for cpo_change
        leaves). Returns NaN series on failure — the leaf then never fires,
        which surfaces in the deterministic never-fires verify gate instead
        of silently degrading to a price proxy."""
        try:
            import yfinance as yf
            cpo_dl = yf.download("FCPO=F", start=index[0], end=index[-1],
                                 interval="1d", progress=False)["Close"]
            if hasattr(cpo_dl, "squeeze"):
                cpo_dl = cpo_dl.squeeze()
            cpo = (pd.Series(cpo_dl.values.ravel(), index=cpo_dl.index)
                   if not isinstance(cpo_dl, pd.Series) else cpo_dl)
            return cpo.reindex(index).ffill()
        except Exception as e:
            logger.warning(f"CPO series fetch failed: {e}")
            return pd.Series(np.nan, index=index)

    def _fetch_funding_column(self, symbol: str,
                              index: pd.DatetimeIndex) -> pd.Series:
        """Last settled funding rate ffill'd to each bar (for funding_* leaves).

        Backward-looking by construction (a bar sees only the most recent
        SETTLED rate); the engine's shift(1) then adds the trade delay. NaN
        series when no symbol/history — the leaf never fires and the
        deterministic verify gate rejects, instead of silently degrading."""
        if not symbol:
            return pd.Series(np.nan, index=index)
        try:
            fund = self._fetch_funding_history(symbol)
            if fund is None or fund.empty or "funding_rate" not in fund:
                return pd.Series(np.nan, index=index)
            return fund["funding_rate"].reindex(index, method="ffill")
        except Exception as e:
            logger.warning(f"funding column fetch failed for {symbol}: {e}")
            return pd.Series(np.nan, index=index)

    # ── Performance metrics ───────────────────────────────────────────────────

    def _cost_rates(self, df: pd.DataFrame, interval: str = "1d") -> dict:
        """Per-side Bursa cost rates for this stock (QC3).

        Uses the shared cost model in config.settings — commission, buy-side
        stamp duty (RM1,000 cap), clearing (RM1,000 cap), and slippage tiered by
        the stock's average DAILY traded value. On sub-daily bars the per-bar
        mean is scaled up by bars-per-day so the tier thresholds (which are
        daily-ADV figures) classify correctly; at 1d the factor is exactly 1.0.
        Rates are expressed as a fraction of trade value at the paper-capital
        notional, so the caps are reflected realistically for our trade size.
        """
        adv_value = 0.0
        if "volume" in df.columns and len(df):
            lookback = max(60, int(60 * _bars_per_day(interval)))  # ~60 days of bars
            tail = df.tail(lookback)
            adv_value = float((tail["close"] * tail["volume"]).mean()) * _bars_per_day(interval)
        tier = bursa_slippage_tier(adv_value)
        notional = PAPER_CAPITAL_MYR * PAPER_ALLOC_PCT
        # Phase 1.1: use the fee schedule in force at the backtest's midpoint so a
        # run spanning the 2023-07-13 stamp-duty remission applies the rate that
        # actually applied (single-schedule approximation per run). Falls back to
        # the current constants when no dated schedule is available.
        as_of = None
        if len(df):
            try:
                as_of = df.index[len(df) // 2].strftime("%Y-%m-%d")
            except Exception:
                as_of = None
        from data.fee_schedule import bursa_trade_cost_asof
        return {
            "buy":  bursa_trade_cost_asof(notional, "buy", tier, as_of) / notional,
            "sell": bursa_trade_cost_asof(notional, "sell", tier, as_of) / notional,
            "tier": tier,
            "adv_value_myr": adv_value,
            "fee_as_of": as_of,
        }

    # ── Data requirements pre-check ──────────────────────────────────────────

    # Keywords in factor_formula / hypothesis that signal unavailable data.
    # Any match → backtest is blocked before fetching a single price bar.
    _UNAVAILABLE_DATA_SIGNALS = [
        "dividend yield", "ttm yield", "dividend ttm",
        "klci yield", "constituent weight",
        "spread mean", "spread std", "spread zscore", "yield spread",
        "blended yield", "basket yield", "reference yield",
        # NOTE: ex-date / ex-dividend timing is NO LONGER blocked — the
        # dividends column is kept from yfinance and the DSL's div_days_to_ex
        # leaf can express event-timed entries around announced ex-dates.
        "corporate announcement", "dividend cut", "dividend suspension",
        "bursa announcement", "pdmr",
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

    # ── Newey-West t-stat (autocorrelation-robust) ────────────────────────────

    # DSL leaf → technique-library node slug (only leaves with a real
    # counterpart node; unmapped leaves are simply skipped)
    _LEAF_TO_TECH = {
        "sma_cross": "tech-sma-crossover",
        "ema_cross": "tech-sma-crossover",
        "rsi": "tech-rsi-mean-reversion",
        "bollinger": "tech-bollinger-squeeze",
        "gap": "tech-gap-fill",
        "reversal": "tech-short-term-reversal",
        "rolling_rank": "tech-cross-sectional-momentum",
        "cpo_change": "tech-cpo-correlation",
        "div_days_to_ex": "tech-event-study",
    }

    def _link_idea_to_techniques(self, idea_id: int, title: str, dsl_tree: dict):
        """On a passing backtest, connect the idea node to the technique nodes
        its DSL leaves employ — the KB Explorer then shows which knowledge
        actually produces surviving strategies."""
        try:
            from knowledge.graph import store
            leaves: set = set()

            def _walk(node):
                if not isinstance(node, dict):
                    return
                if "leaf" in node:
                    leaves.add(node["leaf"])
                for c in node.get("children", []):
                    _walk(c)
                if "child" in node:
                    _walk(node["child"])

            for part in ("entry", "exit"):
                if dsl_tree.get(part):
                    _walk(dsl_tree[part])

            idea_node = store.upsert_node(
                "idea", slug=f"idea-{idea_id}-passed"[:120], title=title or f"idea {idea_id}",
                ref=("alpha_ideas", idea_id),
            )
            for leaf in leaves:
                tech_slug = self._LEAF_TO_TECH.get(leaf)
                if not tech_slug:
                    continue
                tech = store.get_node(slug=tech_slug)
                if tech:
                    store.add_edge(idea_node, tech["id"], "uses_technique",
                                   weight=0.9, origin="heuristic")
        except Exception as e:
            logger.warning(f"Technique linking failed for [{idea_id}]: {e}")

    # ── Cross-sectional validation ────────────────────────────────────────────

    def cross_sectional_test(self, factor_formula: str, idea_id: int,
                             factor: dict | None = None,
                             interval: str = "1d", days: int = 730) -> dict:
        """Test whether a factor generalises across the full universe.
        See cross_sectional.cross_sectional_test for the full implementation
        (kept as a thin instance-method delegate: research_daemon.py calls
        this as self.backtest_engineer.cross_sectional_test(...), a public
        daemon-facing entry point)."""
        return cross_sectional.cross_sectional_test(
            self, factor_formula, idea_id, factor=factor, interval=interval, days=days)

    # ── Cross-sectional basket backtest ───────────────────────────────────────

    def _run_cross_sectional_backtest(self, idea_id: int, row: dict,
                                      params: dict) -> dict:
        """Gated long-top/short-bottom basket backtest across the universe.
        See cross_sectional.run_cross_sectional_backtest for the full
        implementation (kept as a thin instance-method delegate)."""
        return cross_sectional.run_cross_sectional_backtest(self, idea_id, row, params)


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
        # Numeric-first: an idea whose ACTUAL bar interval is sub-daily is
        # SUBDAILY (crypto 15m/1h/4h — really backtested on those bars), not
        # keyword-guessed INTRADAY (which means "sub-daily thesis forced onto
        # daily bars" and carries the indicative-only caveat).
        if _bars_per_day(timeframe) > 1.0:
            return "SUBDAILY"

        blob = f"{timeframe} {factor_formula} {hypothesis}".lower()

        intraday_kw  = ["intraday", "scalp", "tick", "1 minute", "5 minute", "15 minute",
                        "60 minute", "1min", "5min", "15min", "hourly", "hft"]
        short_kw     = ["1 day", "2 day", "3 day", "4 day", "5 day", "1-5 day", "1-3 day",
                        "1 week", "t+1", "t+2", "overnight", "few days"]
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
        "SUBDAILY":    100,   # sub-daily bars produce plenty; demand real evidence
        "SHORT_TERM":   50,
        "MEDIUM_TERM":  30,
        "LONG_TERM":    15,
    }

    # Sharpe thresholds per holding period class (Fix 4)
    _SHARPE_THRESHOLDS = {
        "INTRADAY":    1.1,   # indicative only — needs tick data
        "SUBDAILY":    1.1,   # genuinely backtested on its own bars
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

    # ── Noise-aware train/val gap tolerance ────────────────────────────────────
    # ── Formula verification ──────────────────────────────────────────────────

    def verify_formula(self, params: dict, factor_formula: str, df: pd.DataFrame) -> dict:
        """Verify that the parsed signal code matches the formula description.
        See signal_parsing.verify_formula for the full implementation (kept as
        a thin instance-method delegate so instance-level monkeypatches of
        verify_formula keep intercepting it)."""
        return signal_parsing.verify_formula(self, params, factor_formula, df)

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
            rsi_series = engine._rsi(close, 14)

        # Bollinger Band middle (20-day SMA)
        bb_middle: pd.Series | None = None
        if exit_profile.get("exit_on_middle_band_close"):
            bb_middle = close.rolling(20).mean()

        # Previous close for gap-fill detection
        gap_prev_close: pd.Series | None = None
        if exit_profile.get("exit_on_gap_fill"):
            gap_prev_close = close.shift(1)

        return engine._apply_exit_logic(
            self,
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

    def _data_quality_gate(self, idea_id: int, symbol: str,
                           df: pd.DataFrame, interval: str) -> dict:
        """Gate DQ (Phase 1.2/1.3): score the price data, flag suspected
        corporate-action gaps, persist to data_quality_checks, and decide pass.

        Returns {"passed": bool, "confidence_score": float, "notes": str}.
        Fails open (passes) if the gate is disabled or scoring errors.
        """
        from data.data_quality import (
            compute_data_confidence, detect_corporate_action_anomalies)
        try:
            dq = compute_data_confidence(df, interval)
            # Corp-action gap detection only makes sense where corporate
            # actions exist (splits/dividends → unexplained adjusted-price
            # gaps). On crypto there are none — a >25% bar on a volatile alt
            # is a real market move, and penalising it false-rejected whole
            # pairs at the DQ door (SOL/USDT: 59.9/100 for 4 genuine moves).
            from config.settings import HAS_CORPORATE_ACTIONS
            anomalies = []
            if HAS_CORPORATE_ACTIONS:
                # Gap threshold is a daily-move figure; per-bar moves shrink
                # with bar size, so scale by sqrt(bar-fraction-of-day), floored
                # at 3%. Exactly the configured value at 1d (Bursa parity).
                _gap = GATE_CONFIG.dq_corp_action_gap
                _bpd = _bars_per_day(interval)
                if _bpd > 1.0:
                    _gap = max(0.03, _gap / np.sqrt(_bpd))
                anomalies = detect_corporate_action_anomalies(df, _gap)
        except Exception as _dq_exc:
            self.log_daemon("WARN", f"[{idea_id}] Gate DQ scoring failed: {_dq_exc}")
            return {"passed": True, "confidence_score": 0.0, "notes": "dq error (fail-open)"}

        score = dq["confidence_score"]
        # Each suspected unhandled corporate action dents confidence.
        if anomalies:
            score = max(0.0, score - 10.0 * len(anomalies))
            dq["notes"] += f"; {len(anomalies)} suspected corp-action gap(s)"

        passed = (not GATE_CONFIG.dq_gate_enabled) or (score >= GATE_CONFIG.dq_min_confidence)

        try:
            with db_session() as conn:
                conn.execute("""
                    INSERT INTO data_quality_checks
                      (idea_id, ticker, source, bars, price_completeness,
                       volume_completeness, stale_price_frac, missing_day_frac,
                       corporate_action_flag, confidence_score, passed, notes)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """, (idea_id, symbol, "yfinance", dq["bars"],
                      dq["price_completeness"], dq["volume_completeness"],
                      dq["stale_price_frac"], dq["missing_day_frac"],
                      1 if anomalies else 0, round(score, 1),
                      1 if passed else 0, dq["notes"]))
                for a in anomalies:
                    conn.execute("""
                        INSERT OR IGNORE INTO corporate_actions
                          (ticker, event_date, event_type, adjustment_factor,
                           source, validation_status, notes)
                        VALUES (?,?,?,?,?,?,?)
                    """, (symbol, a["date"], "suspected_gap",
                          1.0 + a["pct_change"], "gap_detector", "suspected",
                          f"{a['pct_change']:+.1%} overnight move "
                          f"({a['prev_close']}→{a['close']})"))
        except Exception as _w_exc:
            self.log_daemon("WARN", f"[{idea_id}] Gate DQ persist failed: {_w_exc}")

        if not passed:
            self.log_daemon(
                "WARN", f"[{idea_id}] Gate DQ FAIL: {symbol} confidence "
                        f"{score}/100 ({dq['notes']})")
        return {"passed": passed, "confidence_score": round(score, 1),
                "notes": dq["notes"]}

    def _reject_idea(self, idea_id: int, row, run_type: str, reason: str,
                     reason_category: str = "stage2") -> dict:
        """Uniform rejection: backtest_runs stub row, alpha_ideas status,
        pipeline event, gate decision, and RejectionMemory. Used by the data
        pre-check, unrepresentable-DSL, verify, and robustness gates."""
        self.log_daemon("WARN", f"Backtest [{idea_id}] REJECTED ({run_type}): {reason}")
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
            """, (idea_id, run_type,
                  row["ticker"] or "", row["timeframe"] or "1d",
                  row["factor_formula"] or "", reason, reason))
            _stamp_versions(conn)
            conn.execute("""
                UPDATE alpha_ideas
                SET status='rejected', rejection_reason=?, updated_at=datetime('now')
                WHERE id=?
            """, (reason[:500], idea_id))
            conn.execute("""
                INSERT INTO pipeline_events (idea_id, stage, event_type, agent, notes)
                VALUES (?, 'stage2', 'rejected', 'BacktestEngineer', ?)
            """, (idea_id, f"{run_type}: {reason[:300]}"))
            conn.execute("""
                INSERT INTO gate_decisions (idea_id, gate, decision, decided_by, rationale)
                VALUES (?, ?, 'reject', 'BacktestEngineer', ?)
            """, (idea_id, f"gate2_{run_type}"[:30], reason[:500]))
        try:
            from knowledge.ingestion.rejection_memory import RejectionMemory
            # reason_category is ALREADY precise here (unrepresentable/
            # duplicate/data_quality/verify_failed) — pass it explicitly so
            # rejection_patterns/the KG bucket use it directly instead of
            # re-guessing from free text (2026-07-13 fix). Still passed
            # positionally as `stage` too, unchanged, since strategy_
            # cemetery.rejected_at_stage already relies on this exact value
            # (pipeline/revisit.py's chain-revival guard checks it).
            RejectionMemory().record_rejection(idea_id, reason, reason_category,
                                               reason_category=reason_category)
        except Exception:
            pass
        self._clear_progress(idea_id)
        return {"gate3_pass": False, "error": run_type, "verdict": "REJECTED",
                "verdict_reason": reason, "idea_id": idea_id}

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
                _stamp_versions(conn)
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

        # ── Cross-sectional route (BEFORE the primary-ticker reduction: a
        # basket idea spans the whole universe, collapsing it to ticker #1
        # would silently turn it into a single-name test). Convention: the
        # structured spec travels in factor_formula as "xs:" + JSON (written
        # by the sandbox/researcher), since alpha_ideas has no params column.
        _ff = (row["factor_formula"] or "").strip()
        if _ff.startswith("xs:"):
            try:
                _xs_params = json.loads(_ff[3:])
                assert _xs_params.get("signal_type") == "cross_sectional"
            except Exception:
                return self._reject_idea(
                    idea_id, row, "xs_spec",
                    "malformed cross-sectional spec (xs: prefix but invalid JSON)",
                    reason_category="unrepresentable")
            return self._run_cross_sectional_backtest(idea_id, row, _xs_params)

        # Extract primary .KL ticker — handles sector descriptions and comma-separated lists
        symbol   = extract_tickers(row["ticker"] or "1155.KL")[0]
        interval = row["timeframe"] or "1d"
        stock    = KLCI_BY_SYMBOL.get(symbol, {})

        self.log_daemon("INFO", f"Backtesting [{idea_id}] {row['title']} — {symbol} {interval}")
        self._log_progress(idea_id, 10, f"Fetching price data for {symbol}")

        # Optimizer promotion: if a parameter sweep chose this idea's winning
        # config, use its exact DSL tree — re-parsing the free-text formula
        # could drift from what was actually swept and selected.
        params = None
        try:
            with db_session() as conn:
                _opt = conn.execute(
                    "SELECT winner_json FROM optimizer_runs WHERE idea_id=? "
                    "AND status='done' ORDER BY id DESC LIMIT 1", (idea_id,)
                ).fetchone()
            if _opt and _opt["winner_json"]:
                _w = json.loads(_opt["winner_json"])
                if _w.get("dsl"):
                    params = {"signal_type": "dsl", "dsl": _w["dsl"],
                              "representable": True}
                    self.log_daemon(
                        "INFO", f"Backtest [{idea_id}] using optimizer winner DSL "
                                f"({_w.get('instrument')} {_w.get('timeframe')})")
        except Exception:
            params = None

        # Parse factor formula into a signal DSL tree. Honesty gate: if the
        # idea can't be expressed by the available conditions, REJECT with
        # the parser's reason — never silently substitute a momentum proxy
        # (the historical failure that flattened every thesis into the same
        # template and made real edges indistinguishable from noise).
        if params is None:
            params = self._parse_factor(
                row["factor_formula"] or "",
                row["title"],
                row["hypothesis"] or "",
            )
        if not params.get("representable"):
            _reason = params.get("reason", "factor not representable as signal conditions")
            # Attempt to turn this dead end into a new DSL leaf instead of a
            # permanent one (Mark-approved 2026-07-13) — best-effort, never
            # blocks the rejection itself from being recorded.
            try:
                from agents.leaf_synthesizer.leaf_synthesizer import LeafSynthesizer
                LeafSynthesizer().synthesize(
                    idea_id, row["hypothesis"] or "", row["factor_formula"] or "", _reason)
            except Exception as _ls_exc:
                self.log_daemon("WARN", f"[{idea_id}] Leaf synthesis attempt "
                                        f"failed: {_ls_exc}")
            return self._reject_idea(
                idea_id, row, "dsl_unrepresentable", _reason,
                reason_category="unrepresentable",
            )

        _universe_kl = [t.strip() for t in (row["ticker"] or "").split(",")
                        if t.strip() and ".KL" in t.strip()]
        if params.get("route") == "fundamental_screen":
            if len(_universe_kl) >= 5:
                self.log_daemon(
                    "INFO",
                    f"Backtest [{idea_id}] parser routed to fundamental screen "
                    f"({len(_universe_kl)} tickers) → portfolio backtest",
                )
                return self._run_fundamental_screen_backtest(idea_id, dict(row))
            # Single-name fundamental idea: usable only if screener context
            # exists (handled by the fund_context override below); otherwise
            # honest rejection beats a price proxy.
            if data_check.get("fund_context") is None:
                return self._reject_idea(
                    idea_id, row, "dsl_unrepresentable",
                    "fundamental screen needs 5+ tickers or cached screener "
                    "fundamentals for this name — neither available",
                    reason_category="unrepresentable",
                )
            params = {"signal_type": "fundamental_screen", "long_only": True}

        # Semantic dedup, layer 2: canonical DSL signature. Two ideas that
        # parse to the same condition tree on the same ticker are the same
        # strategy no matter how the titles are worded.
        if params.get("signal_type") == "dsl":
            from agents.backtest_engineer import signal_dsl
            dsl_sig = "dsl:" + signal_dsl.canonical_signature(params["dsl"], row["ticker"] or "")
            with db_session() as conn:
                dup = conn.execute(
                    "SELECT id, title FROM alpha_ideas "
                    "WHERE signal_signature=? AND id != ? AND status != 'rejected' LIMIT 1",
                    (dsl_sig, idea_id),
                ).fetchone()
                if dup:
                    return self._reject_idea(
                        idea_id, row, "duplicate_signal",
                        f"parses to the same signal as live idea [{dup['id']}] "
                        f"'{dup['title'][:60]}' — reworded duplicate",
                        reason_category="duplicate",
                    )
                conn.execute(
                    "UPDATE alpha_ideas SET signal_signature=? WHERE id=?",
                    (dsl_sig, idea_id),
                )

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

        # Fetch history for a robust train/val/test split. Depth is per-interval
        # (profile-driven): sub-daily intervals fetch fewer calendar days but far
        # more bars; 1d/1wk keep the historical 5-year window.
        df = self._fetch_prices(symbol, interval, days=_FETCH_DAYS.get(interval, 1825))
        # QC4: minimum 252 bars required for any statistically meaningful backtest
        if df.empty or len(df) < 252:
            msg = f"Insufficient history ({len(df)} bars) — need minimum 252 bars"
            self.log_daemon("WARN", f"[{idea_id}] {msg}")
            return {"error": msg, "idea_id": idea_id, "symbol": symbol}

        # ── Real funding drag (crypto): attach per-bar settlements BEFORE the
        # split so train/val/test, walk-forward, regimes and robustness draws
        # all inherit correct alignment for free. Pre-history bars get the
        # modeled constant; the run is labeled "historical" only when the real
        # series covers ≥90% of the window (disclosed in result_data).
        funding_source = "none" if not FUNDING_INTERVAL_HOURS else "modeled"
        if FUNDING_INTERVAL_HOURS:
            try:
                _fund = self._fetch_funding_history(symbol)
                if _fund is not None and not _fund.empty:
                    _fseries = _funding_bar_sum(_fund, df.index)
                    _first_settle = _fund.index[0]
                    _covered = df.index >= _first_settle
                    _coverage = float(_covered.mean())
                    _bar_days = 365.0 / BARS_PER_YEAR.get(interval, 252)
                    _spb = _bar_days * 24.0 / FUNDING_INTERVAL_HOURS
                    # bars before funding history began: modeled constant
                    _fseries[~_covered] = AVG_FUNDING_RATE_PER_INTERVAL * _spb
                    df["funding_bar_sum"] = _fseries
                    funding_source = ("historical" if _coverage >= 0.90
                                      else f"mixed({_coverage:.0%} historical)")
            except Exception as _f_exc:
                self.log_daemon("WARN", f"[{idea_id}] funding history unavailable, "
                                        f"using modeled constant: {_f_exc}")

        # funding_* DSL leaves: attach the backward-looking signal column ONCE
        # so splits, walk-forward, regimes and robustness draws all see it
        # without re-fetching (distinct from funding_bar_sum, which is the
        # drag realized INSIDE each bar's holding window).
        if params.get("signal_type") == "dsl":
            from agents.backtest_engineer import signal_dsl as _sdsl
            if ("funding_rate" in _sdsl.required_columns(params["dsl"])
                    and "funding_rate" not in df.columns):
                df["funding_rate"] = self._fetch_funding_column(symbol, df.index)

        # ── Liquidity floor: reject names too thin to trade realistically ─────
        _liq = self._cost_rates(df, interval)
        if _liq["adv_value_myr"] < BURSA_MIN_DAILY_VALUE_MYR:
            msg = (f"Liquidity floor: {symbol} avg daily traded value "
                   f"RM{_liq['adv_value_myr']:,.0f} < RM{BURSA_MIN_DAILY_VALUE_MYR:,.0f}")
            self.log_daemon("WARN", f"[{idea_id}] {msg}")
            with db_session() as conn:
                conn.execute(
                    "UPDATE alpha_ideas SET status='rejected', "
                    "rejection_reason=?, updated_at=datetime('now') WHERE id=?",
                    (msg, idea_id),
                )
            return {"error": msg, "idea_id": idea_id, "symbol": symbol,
                    "gate3_pass": False}

        # ── Capacity test (Phase 3.4, audit §8.5) ─────────────────────────────
        # How many days to enter/exit the position without exceeding
        # capacity_max_participation of ADV? Rarely binds at paper scale but
        # required before scaling capital.
        _notional = PAPER_CAPITAL_MYR * PAPER_ALLOC_PCT
        _adv = _liq["adv_value_myr"]
        capacity_pct_adv = (_notional / _adv) if _adv > 0 else float("inf")
        _daily_cap = _adv * GATE_CONFIG.capacity_max_participation
        days_to_enter = (_notional / _daily_cap) if _daily_cap > 0 else float("inf")
        capacity_pass = ((not GATE_CONFIG.capacity_gate_enabled)
                         or days_to_enter <= GATE_CONFIG.capacity_max_days)
        capacity_note = ""
        if not capacity_pass:
            capacity_note = (f"Capacity: {days_to_enter:.1f} days to enter at "
                             f"{GATE_CONFIG.capacity_max_participation:.0%} ADV "
                             f"(> {GATE_CONFIG.capacity_max_days:.0f})")

        # ── Survivorship: production-eligibility (Phase 2.3) ──────────────────
        # A single-current-constituent backtest over a multi-year window predates
        # our point-in-time membership coverage → research-grade, not production.
        from data.universe import is_production_eligible
        _window_start = None
        try:
            _window_start = df.index[0].strftime("%Y-%m-%d")
        except Exception:
            pass
        production_eligible = is_production_eligible(_window_start)

        # ── Gate DQ: data-quality gate (Phase 1.2/1.3) ────────────────────────
        # Reject before expensive backtesting if the price data can't be trusted.
        dq = self._data_quality_gate(idea_id, symbol, df, interval)
        if not dq["passed"]:
            msg = (f"Gate DQ: data confidence {dq['confidence_score']}/100 "
                   f"< {GATE_CONFIG.dq_min_confidence} ({dq['notes']})")
            return self._reject_idea(idea_id, row, "data_quality", msg,
                                     reason_category="data_quality")

        self._log_progress(idea_id, 30, f"Computing factor signals ({symbol})")
        # Classify holding period for appropriate thresholds and warnings
        hp_class = self.classify_holding_period(
            interval, row["factor_formula"] or "", row["hypothesis"] or ""
        )

        # Load per-strategy exit profile (if strategy_key is set)
        _strategy_key = (dict(row).get("strategy_key") or "").strip()
        _exit_profile = engine._get_exit_profile_by_key(self, _strategy_key)
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
        verification = {"verified": False, "confidence": 0.0, "issue": ""}
        pinescript = None   # only the dsl branch below can populate this
        if params.get("signal_type") == "fundamental_screen":
            needs_review      = 0
            verification_note = ("Constant signal expected for fundamental screen "
                                 "(quarterly buy-and-hold) — formula verification N/A")
        elif params.get("signal_type") == "dsl":
            # Deterministic verification, and it BLOCKS — unlike the legacy
            # LLM verify_formula which only ever set needs_review=1.
            _full_sig = engine._compute_signals(self, df, params)
            _n_bars = len(_full_sig)
            _fire_frac = float((_full_sig > 0).sum()) / max(_n_bars, 1)
            if _fire_frac == 0.0:
                return self._reject_idea(
                    idea_id, row, "dsl_verify",
                    "signal never fires on 5y of data — conditions unreachable "
                    "(check thresholds vs. this stock's actual ranges)",
                    reason_category="verify_failed",
                )
            if _full_sig.nunique() <= 1:
                return self._reject_idea(
                    idea_id, row, "dsl_verify",
                    "signal is constant — always-on position is buy-and-hold, "
                    "not a strategy",
                    reason_category="verify_failed",
                )
            if _fire_frac > 0.90:
                return self._reject_idea(
                    idea_id, row, "dsl_verify",
                    f"signal long {_fire_frac:.0%} of all bars — effectively "
                    f"buy-and-hold with noise, conditions carry no information",
                    reason_category="verify_failed",
                )
            needs_review = 0
            verification_note = (f"DSL deterministic verify passed: long "
                                 f"{_fire_frac:.1%} of bars")
            verification = {"verified": True, "confidence": 1.0, "issue": ""}

            # Concierge Pine Script export: generated from this EXACT verified
            # tree, independent of whether the idea later clears gate3 — a
            # legitimately executable signal is worth showing even if its
            # Sharpe/DD don't pass. Declines (pinescript stays None) for a
            # leaf Pine can't express (funding/dividends/CPO) — never a
            # fabricated approximation.
            pinescript = None
            try:
                from agents.backtest_engineer.pinescript_gen import generate_pinescript
                from config.settings import ALLOW_SHORT as _allow_short_ps
                _ps = generate_pinescript(params["dsl"], row["title"], interval,
                                          _allow_short_ps)
                if _ps.get("ok"):
                    pinescript = _ps["code"]
            except Exception as _ps_exc:
                self.log_daemon("WARN", f"[{idea_id}] Pine Script generation "
                                        f"failed (non-blocking): {_ps_exc}")
        else:
            verification      = self.verify_formula(params, row["factor_formula"] or "", df)
            needs_review      = 0 if (verification["verified"] and verification["confidence"] >= 0.7) else 1
            verification_note = verification.get("issue", "") or ""

        # INTRADAY on daily bars — flag immediately, do not trust results.
        # (SUBDAILY is different: the backtest genuinely ran on the idea's own
        # sub-daily bars, so no indicative-only caveat — but the model limits
        # are still worth stating on every sub-daily run.)
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
        elif hp_class == "SUBDAILY":
            verification_note = (
                (verification_note + " | " if verification_note else "")
                + f"Sub-daily ({interval}) backtest — liquidation modeled per bar close, "
                  "slippage constant across timeframes"
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
            sig = engine._compute_signals(self, split_df, params)
            # Apply per-strategy exit logic for known strategy_key profiles
            if _has_custom_exit:
                sig = self._compute_signal_with_exits(split_df, sig, _exit_profile)
            results[split_name] = engine._compute_performance(self, split_df, sig, interval)

        self._log_progress(idea_id, 65, "Walk-forward IS/OOS and regime stress test")

        # ── QC2: Walk-forward IS/OOS validation ──────────────────────────────
        wf            = engine._compute_walk_forward(self, df, params, interval)
        sharpe_is     = wf["sharpe_is"]
        sharpe_oos    = wf["sharpe_oos"]
        oos_deg       = wf["oos_degradation"]

        # ── QC5: Regime stress test ───────────────────────────────────────────
        reg              = engine._compute_regimes(self, df, params, interval)
        regimes_positive = reg["regimes_positive"]

        self._log_progress(idea_id, 80, "Computing Sharpe and drawdown")
        train_r = results["train"]
        val_r   = results["val"]
        test_r  = results["test"]
        gate_result = gates.evaluate_gates(
            self, idea_id=idea_id, params=params, hp_class=hp_class, interval=interval,
            df=df, symbol=symbol, train_df=train_df, val_df=val_df, test_df=test_df,
            train_r=train_r, val_r=val_r, test_r=test_r,
            sharpe_is=sharpe_is, sharpe_oos=sharpe_oos, oos_deg=oos_deg,
            regimes_positive=regimes_positive, regime_sharpes=reg,
            capacity_pass=capacity_pass, capacity_note=capacity_note,
        )
        train_val_gap           = gate_result["train_val_gap"]
        test_sharpe_net         = gate_result["test_sharpe_net"]
        test_sharpe_gross       = gate_result["test_sharpe_gross"]
        min_trades              = gate_result["min_trades"]
        actual_trades           = gate_result["actual_trades"]
        _max_tvg                = gate_result["_max_tvg"]
        n_trials                = gate_result["n_trials"]
        deflated_hurdle         = gate_result["deflated_hurdle"]
        psr_test                = gate_result["psr_test"]
        psr_trainval            = gate_result["psr_trainval"]
        gate2_pass               = gate_result["gate2_pass"]
        gate3_pass               = gate_result["gate3_pass"]
        tier1_pass               = gate_result["tier1_pass"]
        tier2_flags              = gate_result["tier2_flags"]
        # Two-tier outcome (2026-07-14): passed the Tier-1 statistical/risk core
        # but tripped an advisory (Tier-2) shape check → HELD for human review,
        # neither auto-advanced nor auto-rejected.
        held_for_review          = bool(tier1_pass and tier2_flags)
        trade_count_pass        = gate_result["trade_count_pass"]
        trade_count_note        = gate_result["trade_count_note"]
        cost_pass               = gate_result["cost_pass"]
        cost_note                = gate_result["cost_note"]
        oos_pass                = gate_result["oos_pass"]
        oos_note                 = gate_result["oos_note"]
        regime_pass              = gate_result["regime_pass"]
        regime_note              = gate_result["regime_note"]
        robustness_score         = gate_result["robustness_score"]
        benchmark_sharpe         = gate_result["benchmark_sharpe"]
        excess_ann_return        = gate_result["excess_ann_return"]
        equal_weight_sharpe      = gate_result["equal_weight_sharpe"]
        excess_vs_ew_ann_return  = gate_result["excess_vs_ew_ann_return"]
        benchmark_pass           = gate_result["benchmark_pass"]
        sanity_flags             = gate_result["sanity_flags"]
        overall_pass             = gate_result["overall_pass"]
        verdict                  = gate_result["verdict"]
        verdict_reason           = gate_result["verdict_reason"]


        # ── Fidelity reports (NOT gated): fill robustness + capacity haircut ──
        # Headline metrics on the full window, plus (a) the same edge under a
        # 2-bar conservative fill — if the Sharpe collapses the "edge" was
        # living on the signal bar's own move, and (b) a capacity-adjusted
        # Sharpe that adds a size-aware market-impact haircut proportional to
        # ADV participation. Both are reported alongside the untouched gated
        # numbers so we can see whether the edge survives realistic execution.
        cagr_full = ulcer_full = fill_robustness = capacity_adjusted_sharpe = None
        dd_dur_full = None
        sharpe_net_conservative = None
        try:
            _hs = engine._compute_signals(self, df, params, symbol=symbol)
            if _has_custom_exit:
                _hs = self._compute_signal_with_exits(df, _hs, _exit_profile)
            _head = engine._compute_performance(self, df, _hs, interval)               # research fill (lag 1)
            _cons = engine._compute_performance(self, df, _hs, interval, lag=2)        # conservative fill
            cagr_full   = _head.get("cagr")
            ulcer_full  = _head.get("ulcer_index")
            dd_dur_full = _head.get("dd_duration_bars")
            _sh, _shc = _head.get("sharpe_net", 0.0), _cons.get("sharpe_net", 0.0)
            sharpe_net_conservative = _shc
            fill_robustness = round(_shc / _sh, 3) if abs(_sh) > 1e-9 else 0.0
            # Capacity impact: linear in participation (notional/ADV). Disclosed
            # rough estimate — 0.10 × participation per side (5% ADV ⇒ ~0.5%/side).
            _partic = capacity_pct_adv if capacity_pct_adv not in (None, float("inf")) else 0.0
            _impact = _CAPACITY_IMPACT_COEF * _partic
            capacity_adjusted_sharpe = engine._compute_performance(
                self, df, _hs, interval, extra_cost_per_side=_impact).get("sharpe_net")
        except Exception as _fid_exc:
            self.log_daemon("WARN", f"[{idea_id}] fidelity reports failed: {_fid_exc}")

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
                       verdict, verdict_reason, pinescript)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    idea_id, "klse_daily", symbol, interval, row["factor_formula"],
                    train_r["sharpe_net"], val_r["sharpe_net"], test_sharpe_net,
                    train_r["max_dd"], val_r["max_dd"], test_r["max_dd"],
                    round(train_val_gap, 3), test_r["total_trades"],
                    test_r["win_rate"], test_r["profit_factor"],
                    json.dumps(params),
                    json.dumps({**results, "funding_source": funding_source}),
                    1 if overall_pass else 0,
                    (1 if (held_for_review or needs_review) else 0), full_note or None,
                    hp_class, actual_trades, actual_trades,
                    test_sharpe_gross, test_sharpe_net, test_sharpe_net, test_sharpe_gross,
                    sharpe_is, sharpe_oos, sharpe_oos, oos_deg,
                    reg["sharpe_low_vol"], reg["sharpe_mid_vol"], reg["sharpe_high_vol"],
                    regimes_positive,
                    json.dumps(sanity_flags) if sanity_flags else None,
                    test_r["max_dd"],
                    verdict, verdict_reason, pinescript,
                ))
                _stamp_versions(conn)
                run_id = conn.execute(
                    "SELECT id FROM backtest_runs WHERE idea_id=? ORDER BY created_at DESC LIMIT 1",
                    (idea_id,),
                ).fetchone()["id"]
                conn.execute("""
                    UPDATE backtest_runs
                    SET n_trials=?, deflated_hurdle=?, benchmark_sharpe=?,
                        excess_ann_return=?, robustness_score=?,
                        equal_weight_sharpe=?, excess_vs_ew_ann_return=?, benchmark_pass=?,
                        capacity_pct_adv=?, days_to_enter=?, capacity_pass=?,
                        production_eligible=?, universe_asof=?,
                        leverage_used=?, funding_drag_pct=?,
                        psr_test=?, psr_trainval=?,
                        cagr=?, ulcer_index=?, dd_duration_bars=?,
                        sharpe_net_conservative=?, fill_robustness=?,
                        capacity_adjusted_sharpe=?, advisory_flags=?
                    WHERE id=?
                """, (n_trials, round(deflated_hurdle, 3),
                      round(benchmark_sharpe, 3), round(excess_ann_return, 4),
                      round(robustness_score, 3) if robustness_score is not None else None,
                      round(equal_weight_sharpe, 3), round(excess_vs_ew_ann_return, 4),
                      1 if benchmark_pass else 0,
                      round(capacity_pct_adv, 4) if capacity_pct_adv != float("inf") else None,
                      round(days_to_enter, 3) if days_to_enter != float("inf") else None,
                      1 if capacity_pass else 0,
                      1 if production_eligible else 0, _window_start,
                      # WS3: leverage/funding traceability — 1.0/0.0 on Bursa,
                      # test_r carries the real values on crypto (see _compute_performance).
                      test_r.get("leverage_used"), test_r.get("funding_drag_pct"),
                      None if psr_test is None else round(psr_test, 4),
                      None if psr_trainval is None else round(psr_trainval, 4),
                      cagr_full, ulcer_full, dd_dur_full,
                      sharpe_net_conservative, fill_robustness,
                      capacity_adjusted_sharpe,
                      json.dumps(tier2_flags) if tier2_flags else None,
                      run_id))

                # Only update stage/status from stage2 → stage3.
                # If idea is already at stage3+ (e.g., after Red-Blue reviewed it),
                # preserve the current stage/status — never overwrite Red-Blue decisions.
                cur_idea = conn.execute(
                    "SELECT stage, status FROM alpha_ideas WHERE id=?", (idea_id,)
                ).fetchone()
                cur_stage = cur_idea["stage"] if cur_idea else "stage2"

                if cur_stage == "stage2":
                    if overall_pass:
                        new_stage, new_status = "stage3", "active"
                    elif held_for_review:
                        new_stage, new_status = "stage2", "needs_review"
                    else:
                        new_stage, new_status = "stage2", "rejected"
                    conn.execute("""
                        UPDATE alpha_ideas
                        SET backtest_sharpe=?, backtest_dd=?, stage=?, status=?,
                            updated_at=datetime('now')
                        WHERE id=?
                    """, (test_sharpe_net, test_r["max_dd"], new_stage, new_status, idea_id))

                if cur_stage != "stage2":
                    # Refresh metrics only; preserve stage and status
                    conn.execute("""
                        UPDATE alpha_ideas
                        SET backtest_sharpe=?, backtest_dd=?, updated_at=datetime('now')
                        WHERE id=?
                    """, (test_sharpe_net, test_r["max_dd"], idea_id))

                _event_type = ("advanced" if overall_pass
                               else "needs_review" if held_for_review else "rejected")
                _gate_decision = ("approve" if overall_pass
                                  else "review" if held_for_review else "reject")
                conn.execute("""
                    INSERT INTO pipeline_events (idea_id, stage, event_type, agent, notes)
                    VALUES (?, 'stage2', ?, 'BacktestEngineer', ?)
                """, (idea_id, _event_type,
                      f"Train(net)={train_r['sharpe_net']:.2f} Val={val_r['sharpe_net']:.2f} "
                      f"Test(net)={test_sharpe_net:.2f} Test(gross)={test_sharpe_gross:.2f} "
                      f"IS={sharpe_is:.2f} OOS={sharpe_oos:.2f} "
                      f"DD={test_r['max_dd']:.1%} Regimes={regimes_positive}/3 "
                      f"AnnRet={test_r.get('ann_return',0):.1%}"))

                conn.execute("""
                    INSERT INTO gate_decisions (idea_id, gate, decision, decided_by, rationale)
                    VALUES (?, 'gate2_3', ?, 'BacktestEngineer', ?)
                """, (idea_id, _gate_decision,
                      f"tier1={'PASS' if tier1_pass else 'FAIL'} "
                      f"G2={'PASS' if gate2_pass else 'FAIL'} "
                      f"G3={'PASS' if gate3_pass else 'FAIL'} "
                      f"oos={'PASS' if oos_pass else 'FAIL'} "
                      + (f"advisory_tripped={[f['flag'] for f in tier2_flags]} "
                         if tier2_flags else "")
                      + f"gap={train_val_gap:.2f}"))
            _outcome = ("PASS" if overall_pass else
                        "NEEDS_REVIEW" if held_for_review else "REJECT")
            self.log_daemon("INFO", f"Backtest saved for idea {idea_id} — outcome={_outcome}"
                            + (f" advisory={[f['flag'] for f in tier2_flags]}"
                               if held_for_review else ""))
        except Exception as e:
            self.log_daemon("ERROR", f"Backtest save FAILED for idea {idea_id}: {e}")
            raise

        # Knowledge graph: surviving ideas link to the techniques they used,
        # so the graph accumulates evidence about which knowledge works.
        # (Outside the save transaction — the store opens its own connection.)
        if overall_pass and params.get("signal_type") == "dsl":
            self._link_idea_to_techniques(idea_id, row["title"], params["dsl"])

        # ── Save equity curve to backtest_series ─────────────────────────────
        try:
            sig_full = engine._compute_signals(self, df, params)
            if _has_custom_exit:
                sig_full = self._compute_signal_with_exits(df, sig_full, _exit_profile)
            # Same single-source series as the gated Sharpe. Fixes the prior
            # divergence: this block used to omit funding/leverage AND default
            # costs to "1d" regardless of the run's real interval, so the
            # dashboard drawdown curve did not match the gated max_dd on crypto
            # / sub-daily runs.
            net_bar = engine._net_return_series(self, df, sig_full, interval)["net"]
            equity  = (1 + net_bar.clip(-0.5, 0.5)).cumprod()
            oos_start   = int(len(df) * 0.70)
            peak        = equity.expanding().max()
            dd_series   = (equity - peak) / peak.clip(lower=1e-9)
            bench_curve = None
            try:
                _bdf = self._fetch_prices(BENCHMARK_SYMBOL, interval, days=1825)
                if not _bdf.empty:
                    _bret = _bdf["close"].pct_change().reindex(df.index).fillna(0)
                    bench_curve = (1 + _bret).cumprod()
            except Exception:
                bench_curve = None
            trades = engine._reconstruct_trades(self, df, sig_full, interval)
            with db_session() as conn:
                conn.execute("DELETE FROM backtest_series WHERE idea_id=?", (idea_id,))
                rows_eq = [
                    (idea_id, engine.series_date_key(d, interval),
                     float(v) - 1.0,
                     float(bench_curve.iloc[i]) - 1.0 if bench_curve is not None else 0.0,
                     float(dd_series.iloc[i]), 1 if i >= oos_start else 0)
                    for i, (d, v) in enumerate(zip(equity.index, equity.values))
                ]
                conn.executemany(
                    "INSERT INTO backtest_series "
                    "(idea_id, date, strategy_pct, benchmark_pct, drawdown_pct, is_oos) "
                    "VALUES (?,?,?,?,?,?)", rows_eq,
                )
                conn.execute("DELETE FROM backtest_trades WHERE idea_id=?", (idea_id,))
                conn.executemany(
                    "INSERT INTO backtest_trades "
                    "(idea_id, seq, direction, entry_date, exit_date, entry_price, "
                    " exit_price, bars_held, gross_pct, cost_pct, net_pct, is_oos) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    [(idea_id, t["seq"], t["direction"], t["entry_date"], t["exit_date"],
                      t["entry_price"], t["exit_price"], t["bars_held"], t["gross_pct"],
                      t["cost_pct"], t["net_pct"], t["is_oos"]) for t in trades],
                )
            self.log_daemon("INFO",
                f"Backtest [{idea_id}] saved {len(rows_eq)} equity points + "
                f"{len(trades)} reconstructed trades")
        except Exception as _eq_exc:
            self.log_daemon("WARN",
                f"Backtest [{idea_id}] could not save equity series: {_eq_exc}")

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
            "tier1_pass":          tier1_pass,
            "tier2_flags":         tier2_flags,
            "held_for_review":     held_for_review,
            "overall_pass":        overall_pass,
            "verdict":             verdict,
            "trade_count_pass":    trade_count_pass,
            "cost_pass":           cost_pass,
            "oos_pass":            oos_pass,
            "regime_pass":         regime_pass,
            # PSR principal rule (subsumes the old deflation binary — kept as
            # a derived key for callers/harness that attribute failures)
            "psr_test":            None if psr_test is None else round(psr_test, 4),
            "psr_trainval":        None if psr_trainval is None else round(psr_trainval, 4),
            "deflation_pass":      (psr_test is None
                                    or psr_test >= GATE_CONFIG.psr_confidence_test),
            "benchmark_pass":      benchmark_pass,
            "capacity_pass":       capacity_pass,
            "capacity_pct_adv":    None if capacity_pct_adv == float("inf") else round(capacity_pct_adv, 4),
            "days_to_enter":       None if days_to_enter == float("inf") else round(days_to_enter, 3),
            "production_eligible":  production_eligible,
            "n_trials":            n_trials,
            "deflated_hurdle":     round(deflated_hurdle, 3),
            "benchmark_sharpe":    round(benchmark_sharpe, 3),
            "excess_ann_return":   round(excess_ann_return, 4),
            "equal_weight_sharpe": round(equal_weight_sharpe, 3),
            "excess_vs_ew_ann_return": round(excess_vs_ew_ann_return, 4),
            "train":               train_r,
            "val":                 val_r,
            "test":                test_r,
            "sharpe_is":           sharpe_is,
            "sharpe_oos":          sharpe_oos,
            "oos_degradation":     oos_deg,
            "regimes":             reg,
            "train_val_gap":       round(train_val_gap, 3),
            "train_val_gap_tol":   round(_max_tvg, 3),
            "funding_source":      funding_source,
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
        See fundamental_screen.run_fundamental_screen_backtest for the full
        implementation (kept as a thin instance-method delegate, consistent
        with the other _run_*_backtest routing methods)."""
        return fundamental_screen.run_fundamental_screen_backtest(self, idea_id, row)


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
