import json
import logging
import os
import numpy as np
import pandas as pd
from agents.base_agent import BaseAgent
from config.settings import (
    MODEL_FAST, MODEL_MAIN, GATE_CONFIG, KLCI_BY_SYMBOL, DEFAULT_SYMBOLS,
    PAPER_CAPITAL_MYR, PAPER_ALLOC_PCT, BURSA_MIN_DAILY_VALUE_MYR,
    bursa_trade_cost, bursa_slippage_tier,
    MARKET_RULES_VERSION, FEE_MODEL_VERSION,
    BENCHMARK_SYMBOL,
    DEFAULT_LEVERAGE, MAX_LEVERAGE, LIQUIDATION_BUFFER,
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

logger = logging.getLogger(__name__)

from config.settings import MARKET_NAME as _MARKET_NAME_FOR_SYSTEM

SYSTEM = f"""You are a quantitative backtesting engineer specialising in {_MARKET_NAME_FOR_SYSTEM}.
Parse strategy descriptions into structured signal parameters for vectorised backtesting.
Output only valid JSON."""


from config.settings import PROGRESS_FILE as _PROGRESS_FILE_PATH
_PROGRESS_FILE = str(_PROGRESS_FILE_PATH)


def _funding_bar_sum(funding: pd.DataFrame, bar_index: pd.DatetimeIndex) -> pd.Series:
    """Resample 8h funding settlements onto a bar index with NO smearing.

    Bar labeled T (bar-open convention, Binance klines) covers the holding
    window [T, next_open). Each settlement timestamp S is assigned to bar
    `searchsorted(S, side="right") - 1` — the bar whose window contains it —
    and rates are SUMMED per bar (1d → 3 settlements summed; 1h/15m → a
    settlement lands in exactly one bar; 1wk → 21 summed). Settlement
    timestamps carry millisecond jitter (e.g. 16:00:00.001) which searchsorted
    handles naturally. Settlements before the first bar are dropped.

    Lookahead: none — the engine charges this on the LAGGED position
    (signal_shifted), i.e. rates realized during a window the position was
    already held through; the rate never forms the signal (the DSL's
    `funding_rate` column is a separate, backward-looking ffill series).
    """
    out = pd.Series(0.0, index=bar_index)
    if funding is None or funding.empty or "funding_rate" not in funding:
        return out
    ts = funding.index
    pos = bar_index.searchsorted(ts, side="right") - 1
    valid = pos >= 0
    if not valid.any():
        return out
    sums = pd.Series(funding["funding_rate"].values[valid]).groupby(
        pos[valid]).sum()
    out.iloc[sums.index] = sums.values
    return out


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
        genericized onto the nearest template (the historical failure mode
        that flattened every thesis into 20-day momentum).

        The catalog shows parameter RANGES only, no example values — the old
        prompt pre-filled defaults (20/50/14/35/65...) and Haiku anchored on
        them instead of extracting the idea's own parameters.
        """
        from agents.backtest_engineer import signal_dsl
        from config.settings import ALLOW_SHORT, MARKET_NAME

        short_shape = ""
        short_rule = "Long-only (Bursa short-selling restricted): the entry tree describes when to be LONG."
        if ALLOW_SHORT:
            short_shape = (
                '  "short_entry": <condition tree or null — when to go SHORT, if the strategy '
                'has a short thesis>,\n'
                '  "short_exit": <condition tree or null — when to cover the short; null means '
                'hold short while short_entry is true>,\n'
            )
            short_rule = ("Long AND short are both supported (perpetuals). A tree may set entry/exit "
                         "(long leg), short_entry/short_exit (short leg), or both if the strategy "
                         "genuinely trades both directions — most single-direction ideas need only one "
                         "leg. If the strategy is a pure short thesis, entry/exit may be null.")

        prompt = f"""Translate this {MARKET_NAME} strategy into a signal condition tree.

Factor formula: {factor_formula}
Strategy title: {title}
Hypothesis: {hypothesis}

AVAILABLE CONDITIONS (parameters MUST come from the strategy text; ranges are hard limits):
{signal_dsl.leaf_catalog_text()}

Combinators: {{"op": "AND"|"OR", "children": [<node>, <node>, ...]}} and {{"op": "NOT", "child": <node>}}.
A leaf node looks like {{"leaf": "<name>", <params>}}.

Return JSON, one of these three shapes:

1. Representable as price/volume/dividend/CPO conditions:
{{
  "representable": true,
  "entry": <condition tree or null — when to be LONG>,
  "exit": <condition tree or null — when to flatten the long; null means hold while entry condition is true>,
{short_shape}  "notes": "one line: how the tree captures the strategy's actual thesis"
}}

2. A fundamental screen across 5+ stocks (ROE/PB/PE/DY ranking or filtering):
{{"representable": true, "route": "fundamental_screen"}}

3. NOT expressible with the available conditions (requires data or logic none of the leaves provide):
{{"representable": false, "reason": "one specific sentence — what the strategy needs that is unavailable"}}

Rules:
- Extract every numeric parameter from the strategy text. If the text gives no value for a
  required parameter, the strategy is underspecified — use shape 3 with reason "parameter X unspecified".
- NEVER approximate an unrelated mechanism with a price proxy. If the thesis is about earnings
  surprises, analyst coverage, sentiment, or anything with no matching leaf, use shape 3.
- {short_rule}"""
        result = self.call_claude_json(
            SYSTEM,
            [{"role": "user", "content": prompt}],
            model=MODEL_FAST,
            task_label="parse_factor",
        )
        if not isinstance(result, dict):
            return {"representable": False, "reason": "parser returned non-JSON"}
        if "error" in result:
            return {"representable": False, "reason": "parser JSON parse failure"}

        if result.get("route") == "fundamental_screen":
            return {"signal_type": "fundamental_screen", "route": "fundamental_screen",
                    "representable": True}

        if not result.get("representable"):
            return {"representable": False,
                    "reason": result.get("reason", "not representable (no reason given)")}

        tree = {"entry": result.get("entry"), "exit": result.get("exit")}
        if ALLOW_SHORT:
            if result.get("short_entry"):
                tree["short_entry"] = result.get("short_entry")
            if result.get("short_exit"):
                tree["short_exit"] = result.get("short_exit")
        errors = signal_dsl.validate(tree)
        if errors:
            return {"representable": False,
                    "reason": f"invalid condition tree: {'; '.join(errors[:4])}"}
        return {
            "signal_type": "dsl",
            "representable": True,
            "dsl": tree,
            "long_only": not ALLOW_SHORT,
            "notes": result.get("notes", ""),
        }

    # ── Signal computation ────────────────────────────────────────────────────

    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        rs    = gain / loss.replace(0, 1e-9)
        return 100 - (100 / (1 + rs))

    def _compute_signals(self, df: pd.DataFrame, params: dict,
                         symbol: str = "") -> pd.Series:
        close  = df["close"]
        stype  = params.get("signal_type", "momentum")
        # KLSE equities: long-only by default (short-selling is restricted to
        # designated securities only)
        long_only = bool(params.get("long_only", True))
        open_prices = df["open"] if "open" in df.columns else close

        if stype == "dsl":
            # Condition-tree path (see signal_dsl.py). Legacy template branches
            # below stay intact: paper trading replays stored backtest params.
            from agents.backtest_engineer import signal_dsl
            frame = df
            needed = signal_dsl.required_columns(params["dsl"])
            if "cpo_close" in needed and "cpo_close" not in df.columns:
                frame = df.copy()
                frame["cpo_close"] = self._fetch_cpo_series(df.index)
            if "dividends" in needed and "dividends" not in df.columns:
                if frame is df:
                    frame = df.copy()
                frame["dividends"] = 0.0  # no dividend data cached — leaf never fires
            if "funding_rate" in needed and "funding_rate" not in df.columns:
                if frame is df:
                    frame = df.copy()
                # Last SETTLED rate ffill'd to each bar (backward-looking; the
                # engine's shift(1) adds the trade delay). Without a symbol —
                # or on the yahoo backend — the series is empty → NaN column →
                # the leaf never fires and deterministic verify rejects it.
                frame["funding_rate"] = self._fetch_funding_column(symbol, df.index)
            return signal_dsl.signal_from_dsl(frame, params["dsl"])

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

        else:  # momentum / default (legacy)
            if stype not in ("momentum",):
                # The old parser could emit types with no branch (value,
                # quality) which silently became momentum — the new DSL
                # parser can't, but legacy stored params might. Be loud.
                logger.warning(
                    f"_compute_signals: legacy signal_type {stype!r} has no "
                    f"branch — falling back to momentum (legacy-compat only)"
                )
            period = int(params.get("momentum_period", 20))
            ret    = close.pct_change(period)
            raw    = np.where(ret > 0, 1, 0 if long_only else -1)

        return pd.Series(raw, index=df.index, dtype=float)

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

    def _compute_performance(self, df: pd.DataFrame, signals: pd.Series, interval: str,
                             leverage: float | None = None) -> dict:
        """Compute performance with QC1 lookahead guard and QC3 realistic costs.

        QC1 — Lookahead bias guard:
          Signal computed on day T may only trigger a trade at T+1 open.
          Enforced by pd.Series.shift(1) — signal_shifted[t] = signal[t-1].
          signal_shifted[0] is forced to 0 (no position at start).

        QC3 — Realistic transaction costs (see _cost_rates):
          asymmetric buy/sell rates where the market has them (Bursa stamp
          duty is buy-side only, capped), slippage tiered by liquidity,
          deducted on every position change. Returns both sharpe_gross
          (before costs) and sharpe_net (after costs).

        WS3 — signed positions, funding, leverage/liquidation (crypto only —
        every term below is a documented no-op on Bursa, see the settings
        each is sourced from):
          `signals` may be -1/0/1 (short/flat/long); the existing gross-return
          and delta-based turnover-cost math already handles the sign correctly
          with no changes (short * negative bar return = correct short PnL;
          a short's entry/exit is still a "sell"/"buy" cost event).
          Funding: when the caller attached a `funding_bar_sum` column to df
          (real historical settlements resampled per bar — see
          _funding_bar_sum), the drag uses the REAL per-bar series. Otherwise
          it falls back to AVG_FUNDING_RATE_PER_INTERVAL, a disclosed MODELED
          AVERAGE, scaled by how many settlements the bar spans. 0.0 on Bursa
          (no perp funding).
          Leverage: `leverage` defaults to DEFAULT_LEVERAGE (1.0 — unleveraged
          unless the caller explicitly requests more, capped at MAX_LEVERAGE).
          Liquidation: at leverage > 1, any single bar whose leveraged loss
          exceeds 1/leverage * (1 - LIQUIDATION_BUFFER) is capped at that
          floor — a simplified per-bar liquidation model (this system has no
          intraday data, so path-dependent intra-bar liquidation isn't
          modelable; a single daily bar breaching the threshold is the
          realistic granularity available).
        """
        close = df["close"]
        sig   = signals.fillna(0)

        # ── QC1: strict 1-bar signal delay ────────────────────────────────────
        signal_shifted = sig.shift(1).fillna(0)
        assert float(signal_shifted.iloc[0]) == 0.0, \
            "Lookahead guard failure: signal_shifted[0] != 0"

        bar_returns    = close.pct_change().fillna(0)

        # Gross returns (no costs): position held at T earns return from T→T+1.
        # Sign-correct for short positions (signal_shifted == -1) with no
        # special-casing — a short earns the negative of the bar return.
        gross_bar     = signal_shifted * bar_returns
        gross_returns = gross_bar.values[1:]   # drop bar 0 (always 0 after shift)

        # ── QC3: per-side costs on every position change ──────────────────────
        rates          = self._cost_rates(df, interval)
        deltas         = np.diff(signal_shifted.values)
        buys           = np.clip(deltas, 0, None)
        sells          = np.clip(-deltas, 0, None)
        signal_changes = np.abs(deltas)
        cost_adj_returns = gross_returns - buys * rates["buy"] - sells * rates["sell"]

        # ── WS3: funding accrual (crypto only) ─────────────────────────────────
        leverage = min(leverage if leverage else DEFAULT_LEVERAGE, MAX_LEVERAGE)
        bar_days = 365.0 / BARS_PER_YEAR.get(interval, 252)
        settlements_per_bar = (bar_days * 24.0 / FUNDING_INTERVAL_HOURS) if FUNDING_INTERVAL_HOURS else 0.0
        if "funding_bar_sum" in df.columns and FUNDING_INTERVAL_HOURS:
            # REAL settlements realized inside each bar's holding window,
            # charged on the lagged position — no lookahead (the rate is paid
            # during the window, never used to form the signal). Guarded on
            # FUNDING_INTERVAL_HOURS: funding cannot exist on Bursa even if a
            # stray column appears.
            funding_drag_bar = (signal_shifted.values[1:]
                                * df["funding_bar_sum"].values[1:])
        else:
            funding_drag_bar = (signal_shifted.values[1:] * AVG_FUNDING_RATE_PER_INTERVAL
                               * settlements_per_bar)
        pre_leverage_returns = cost_adj_returns - funding_drag_bar

        # ── WS3: leverage + bounded per-bar liquidation ────────────────────────
        net_returns = pre_leverage_returns * leverage
        if leverage > 1.0:
            liq_floor = -(1.0 / leverage) * (1.0 - LIQUIDATION_BUFFER)
            net_returns = np.where(net_returns < liq_floor, liq_floor, net_returns)
        # PnL CONTRIBUTION sign (negative = funding cost/drag, positive = funding
        # income) — funding_drag_bar is SUBTRACTED from returns above, so the
        # contribution is its negative, not the raw subtracted amount.
        funding_drag_pct = float(-np.sum(funding_drag_bar) * leverage)

        n = len(net_returns)
        _empty = {
            "sharpe": 0.0, "sharpe_gross": 0.0, "sharpe_net": 0.0,
            "max_dd": 1.0, "win_rate": 0.0, "profit_factor": 0.0,
            "total_trades": 0, "ann_return": 0.0,
            "leverage_used": leverage, "funding_drag_pct": round(funding_drag_pct, 4),
            "n_obs": n, "skew": 0.0, "kurt": 3.0,
        }
        if n < 20 or np.std(net_returns) < 1e-10:
            return _empty

        ann         = BARS_PER_YEAR.get(interval, 252)
        g_std       = float(np.std(gross_returns))
        n_std       = float(np.std(net_returns))
        sharpe_gross = round(float(np.mean(gross_returns) / g_std * np.sqrt(ann)), 3) if g_std > 1e-10 else 0.0
        sharpe_net   = round(float(np.mean(net_returns)   / n_std * np.sqrt(ann)), 3) if n_std > 1e-10 else 0.0
        # Return moments for the PSR pass rule (kurt non-excess; normal = 3)
        from agents.backtest_engineer.stats import moments as _moments
        ret_skew, ret_kurt = _moments(net_returns)

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
            "leverage_used": leverage,
            "funding_drag_pct": round(funding_drag_pct, 4),
            "n_obs":         n,
            "skew":          round(ret_skew, 3),
            "kurt":          round(ret_kurt, 3),
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
        rolling_vol = daily_ret.rolling(60).std() * np.sqrt(BARS_PER_YEAR.get(interval, 252))  # annualised

        # Signals on full series (so MAs etc. have full context)
        sig          = self._compute_signals(df, params)
        sig_shifted  = sig.shift(1).fillna(0)

        # Per-bar net return (same cost model as _compute_performance)
        rates        = self._cost_rates(df, interval)
        deltas       = sig_shifted.diff().fillna(0)
        cost_bar     = deltas.clip(lower=0) * rates["buy"] + (-deltas).clip(lower=0) * rates["sell"]
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
        # Daily-or-faster only (weekly was never flagged — preserved); the
        # threshold scales with bar frequency so sub-daily isn't over-flagged.
        if _bars_per_day(timeframe) >= 1.0 and trade_count > 500 * _bars_per_day(timeframe):
            flags.append(f"Too many trades for a {timeframe} strategy: {trade_count}")
        return flags

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

    def _robustness_check(self, test_df: pd.DataFrame, dsl_tree: dict,
                          base_sharpe: float, interval: str) -> float:
        """QC7: fraction of ±20% parameter perturbations whose test-split net
        Sharpe stays above robustness_sharpe_ratio × base. Seeded for
        reproducibility; vectorized, no LLM cost."""
        from agents.backtest_engineer import signal_dsl
        rng = np.random.RandomState(1234)
        ok, valid = 0, 0
        for _ in range(GATE_CONFIG.robustness_draws):
            perturbed = signal_dsl.perturb_tree(dsl_tree, rng)
            try:
                sig = self._compute_signals(
                    test_df, {"signal_type": "dsl", "dsl": perturbed})
                perf = self._compute_performance(test_df, sig, interval)
                valid += 1
                if perf["sharpe_net"] > GATE_CONFIG.robustness_sharpe_ratio * base_sharpe:
                    ok += 1
            except Exception as e:
                logger.warning(f"Robustness draw failed: {e}")
        return ok / max(valid, 1)

    @staticmethod
    def _nw_tstat(series: np.ndarray, max_lag: int | None = None) -> float:
        """t-stat of the series mean using Newey-West (Bartlett kernel) standard
        errors. Daily IC observations are autocorrelated, so the iid t-stat
        (mean / (std/√n)) overstates significance; this corrects for it."""
        n = len(series)
        if n < 3:
            return 0.0
        x = series - series.mean()
        if max_lag is None:
            max_lag = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
        max_lag = max(0, min(max_lag, n - 1))
        gamma0 = float(np.dot(x, x)) / n
        lrv = gamma0
        for lag in range(1, max_lag + 1):
            w = 1.0 - lag / (max_lag + 1.0)
            lrv += 2.0 * w * (float(np.dot(x[lag:], x[:-lag])) / n)
        if lrv <= 1e-12:
            return 0.0
        return float(series.mean() / np.sqrt(lrv / n))

    # ── Cross-sectional validation ────────────────────────────────────────────

    def cross_sectional_test(self, factor_formula: str, idea_id: int,
                             factor: dict | None = None,
                             interval: str = "1d", days: int = 730) -> dict:
        """Test whether a factor generalises across the full universe.

        Two modes:
          * ``factor`` supplied ({"name":..., "params":{...}} from the factor
            registry) → per-name CONTINUOUS scores (proper Spearman rank IC)
            at the requested ``interval``. This is the cross-sectional
            strategy path.
          * no ``factor`` → LEGACY path, byte-stable: the idea's parsed
            binary/ternary entry signal on daily bars (the stage2 veto that
            checks a single-name idea also works across the universe).

        Cross-sectional IC at each date: Spearman across names of
        {score_t, return_t+1}; mean IC + Newey-West t-stat measure breadth.
        Quintile portfolio: long top ~20% by score, equal weight (plus a
        top-minus-bottom spread when a factor is supplied and shorts are
        allowed — informational).

        Gate thresholds come from GATE_CONFIG (xs_min_mean_ic /
        xs_min_ic_tstat / xs_min_positive_names — defaults equal the values
        previously hardcoded here; crypto overrides positive-names to 12/20).
        """
        with db_session() as conn:
            row = conn.execute("SELECT * FROM alpha_ideas WHERE id=?", (idea_id,)).fetchone()
        if not row:
            return {"error": f"Idea {idea_id} not found", "factor_is_real": False}

        params = None
        if factor is None:
            formula = factor_formula or row["factor_formula"] or ""
            params  = self._parse_factor(formula, row["title"], row["hypothesis"] or "")
            if not params or "error" in params:
                params = {"signal_type": "momentum", "momentum_period": 20, "long_only": True}
        else:
            from agents.backtest_engineer import factors as factor_registry
            interval = interval or "1d"
            days = _FETCH_DAYS.get(interval, days)

        self.log_daemon(
            "INFO", f"CrossSect [{idea_id}]: {interval} data for "
                    f"{len(DEFAULT_SYMBOLS)} names "
                    f"({'factor ' + factor['name'] if factor else 'legacy signal'})")

        # ── Build score + forward-return panels ──────────────────────────────
        signal_series: dict[str, pd.Series] = {}
        return_series: dict[str, pd.Series] = {}

        for symbol in DEFAULT_SYMBOLS:
            try:
                df = self._fetch_prices(symbol, interval if factor else "1d",
                                        days=days)
                if df.empty or len(df) < 60:
                    continue
                if factor is not None:
                    for _col in factor_registry.required_columns(factor["name"]):
                        if _col == "funding_rate" and _col not in df.columns:
                            df[_col] = self._fetch_funding_column(symbol, df.index)
                    sig = factor_registry.compute_factor(
                        factor["name"], df, factor.get("params"))
                    fwd_ret = df["close"].pct_change().shift(-1)
                    # continuous scores: 0 is a legitimate value — keep it
                    valid = sig.notna() & fwd_ret.notna()
                else:
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
        spread_rets: list[float] = []   # top-minus-bottom (factor mode, shorts allowed)
        from config.settings import ALLOW_SHORT as _allow_short
        _spread_ok = factor is not None and _allow_short

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
            if _spread_ok:
                bot_idx = np.argsort(sv)[:n_q]
                spread_rets.append(float(np.mean(rv[top_idx]) - np.mean(rv[bot_idx])))

        if not ic_series:
            return {
                "error": "IC series is empty (constant signal?)",
                "factor_is_real": False,
                "idea_id": idea_id,
            }

        ic_arr  = np.array(ic_series)
        mean_ic = float(np.mean(ic_arr))
        ic_std  = float(np.std(ic_arr, ddof=1))
        # Daily ICs are strongly autocorrelated (signals persist for weeks), so
        # the naive iid t-stat overstates significance. Use Newey-West SEs.
        ic_tstat_iid = (mean_ic / (ic_std / np.sqrt(len(ic_arr)))) if ic_std > 1e-10 else 0.0
        ic_tstat = self._nw_tstat(ic_arr)

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
        _ann_xs = BARS_PER_YEAR.get(interval if factor else "1d", 252)
        quintile_sharpe = 0.0
        if len(portfolio_rets) > 20:
            pr  = np.array(portfolio_rets)
            std = float(np.std(pr, ddof=1))
            quintile_sharpe = round(float(np.mean(pr) / std * np.sqrt(_ann_xs)), 3) if std > 1e-10 else 0.0
        spread_sharpe = 0.0
        if len(spread_rets) > 20:
            sr_ = np.array(spread_rets)
            std = float(np.std(sr_, ddof=1))
            spread_sharpe = round(float(np.mean(sr_) / std * np.sqrt(_ann_xs)), 3) if std > 1e-10 else 0.0

        # ── Gate: factor is real? (thresholds from GATE_CONFIG — defaults are
        # the previously hardcoded North Star values; crypto overrides the
        # positive-names count proportional to its universe) ──────────────────
        factor_is_real = (
            mean_ic > GATE_CONFIG.xs_min_mean_ic
            and ic_tstat > GATE_CONFIG.xs_min_ic_tstat
            and stocks_positive_ic > GATE_CONFIG.xs_min_positive_names
        )

        result = {
            "idea_id":            idea_id,
            "mean_ic":            round(mean_ic, 4),
            "ic_tstat":           round(ic_tstat, 3),
            "ic_tstat_iid":       round(ic_tstat_iid, 3),
            "stocks_tested":      n_stocks,
            "stocks_positive_ic": stocks_positive_ic,
            "quintile_sharpe":    quintile_sharpe,
            "spread_sharpe":      spread_sharpe,   # top-minus-bottom (0.0 when n/a)
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

    # ── Cross-sectional basket backtest ───────────────────────────────────────

    def _run_cross_sectional_backtest(self, idea_id: int, row: dict,
                                      params: dict) -> dict:
        """Gated long-top/short-bottom basket backtest across the universe.

        params contract (idea params JSON / sandbox / researcher):
          {"signal_type": "cross_sectional",
           "factor": {"name": <FACTORS key>, "params": {...}},
           "top_n": 3-5, "bottom_n": 0-5,     # bottom_n forced 0 if not ALLOW_SHORT
           "rebalance_bars": int,             # e.g. 7 = weekly on 1d bars
           "interval": "1d"}

        Method: per-name CONTINUOUS factor scores (factors.py registry) ranked
        cross-sectionally at each rebalance bar; weights take effect the NEXT
        bar (same shift(1) no-lookahead convention as the single-name engine);
        equal-weight legs; per-name per-side costs from the profile cost model
        on turnover; REAL per-bar funding drag per leg (crypto). The resulting
        NAV runs through the standard gate stack at FULL stage3 thresholds
        (deliberately NOT the relaxed fundamental-screen thresholds) plus the
        IC gate (cross_sectional_test in factor mode), the equal-weight
        benchmark gate, and the deflated-Sharpe multiple-testing hurdle.

        A PASS advances the idea to stage3 but it is PARKED there — basket
        paper-trading doesn't exist yet (single-name executor); the daemon's
        promotion guards key off run_type='cross_sectional'. Honest, disclosed.
        """
        from agents.backtest_engineer import factors as factor_registry
        from config.settings import ALLOW_SHORT as _allow_short

        factor = params.get("factor") or {}
        fname = factor.get("name", "")
        try:
            fparams = factor_registry.validate_factor(fname, factor.get("params") or {})
        except ValueError as exc:
            return self._reject_idea(idea_id, row, "xs_factor",
                                     f"cross-sectional factor invalid: {exc}",
                                     reason_category="unrepresentable")

        interval = params.get("interval") or row["timeframe"] or "1d"
        rebalance_bars = max(1, int(params.get("rebalance_bars", 7)))
        top_n = max(1, int(params.get("top_n", 4)))
        bottom_n = int(params.get("bottom_n", 0))
        if not _allow_short:
            bottom_n = 0   # Bursa: long-only basket, structurally
        days = _FETCH_DAYS.get(interval, 1825)
        needs_funding = "funding_rate" in factor_registry.required_columns(fname)

        self.log_daemon(
            "INFO", f"XSect backtest [{idea_id}]: factor={fname}{fparams} "
                    f"top{top_n}/bottom{bottom_n} rebal={rebalance_bars} bars "
                    f"{interval} across {len(DEFAULT_SYMBOLS)} names")
        self._log_progress(idea_id, 15, f"Building {len(DEFAULT_SYMBOLS)}-name factor panel")

        # ── Panel build ───────────────────────────────────────────────────────
        closes: dict[str, pd.Series] = {}
        scores: dict[str, pd.Series] = {}
        fundmap: dict[str, pd.Series] = {}
        side_rate: dict[str, float] = {}
        coverage_notes: list[str] = []
        for symbol in DEFAULT_SYMBOLS:
            try:
                df = self._fetch_prices(symbol, interval, days=days)
                if df.empty or len(df) < 100:
                    coverage_notes.append(f"{symbol}: {0 if df.empty else len(df)} bars — excluded")
                    continue
                if needs_funding and "funding_rate" not in df.columns:
                    df["funding_rate"] = self._fetch_funding_column(symbol, df.index)
                scores[symbol] = factor_registry.compute_factor(fname, df, fparams)
                closes[symbol] = df["close"]
                if FUNDING_INTERVAL_HOURS:
                    _f = self._fetch_funding_history(symbol)
                    fundmap[symbol] = _funding_bar_sum(_f, df.index)
                _r = self._cost_rates(df, interval)
                side_rate[symbol] = (_r["buy"] + _r["sell"]) / 2.0
            except Exception as exc:
                coverage_notes.append(f"{symbol}: {exc}")

        if len(scores) < max(5, top_n + bottom_n):
            return self._reject_idea(
                idea_id, row, "xs_universe",
                f"only {len(scores)} names usable for the basket "
                f"(need ≥{max(5, top_n + bottom_n)})",
                reason_category="data")

        close_p = pd.DataFrame(closes).sort_index()
        score_p = pd.DataFrame(scores).reindex(close_p.index)
        ret_p = close_p.pct_change().fillna(0.0)
        fund_p = (pd.DataFrame(fundmap).reindex(close_p.index).fillna(0.0)
                  if fundmap else pd.DataFrame(0.0, index=close_p.index,
                                               columns=close_p.columns))
        n_bars = len(close_p)
        if n_bars < 252:
            return self._reject_idea(idea_id, row, "xs_history",
                                     f"only {n_bars} common bars (need ≥252)",
                                     reason_category="data")

        # ── Rebalance loop: ranks at bar close → weights from the NEXT bar ────
        weights = pd.DataFrame(0.0, index=close_p.index, columns=close_p.columns)
        current = pd.Series(0.0, index=close_p.columns)
        n_rebalances = 0
        for i in range(0, n_bars, rebalance_bars):
            sv = score_p.iloc[i].dropna()
            if len(sv) >= max(5, top_n + bottom_n):
                new_w = pd.Series(0.0, index=close_p.columns)
                ranked = sv.sort_values()
                new_w[ranked.index[-top_n:]] = 1.0 / top_n
                if bottom_n > 0:
                    new_w[ranked.index[:bottom_n]] = -1.0 / bottom_n
                if not new_w.equals(current):
                    n_rebalances += 1
                current = new_w
            weights.iloc[i] = current
        weights = weights.replace(0.0, np.nan).ffill().fillna(0.0)
        # one-bar execution delay — weights decided at bar i apply to i+1
        w_held = weights.shift(1).fillna(0.0)

        # ── Portfolio returns: PnL − turnover costs − funding per leg ─────────
        gross = (w_held * ret_p).sum(axis=1)
        turnover = (weights - weights.shift(1)).abs().fillna(0.0)
        rate_vec = pd.Series(side_rate).reindex(close_p.columns).fillna(
            float(np.mean(list(side_rate.values()))))
        costs = (turnover * rate_vec).sum(axis=1)
        funding = (w_held * fund_p).sum(axis=1)   # long pays +funding, short receives
        port_ret = gross - costs - funding
        nav = (1.0 + port_ret).cumprod()

        port_df = pd.DataFrame({"close": nav,
                                "open": nav, "high": nav, "low": nav,
                                "volume": 1.0,
                                # zero column DISABLES the modeled-constant
                                # fallback in _compute_performance — funding is
                                # already embedded in the NAV per leg above.
                                "funding_bar_sum": 0.0},
                               index=close_p.index)

        # ── Standard gate scaffolding on the NAV ──────────────────────────────
        self._log_progress(idea_id, 55, "Gating basket NAV")
        train_df, val_df, test_df = self._split(port_df)
        one = lambda d: pd.Series(1.0, index=d.index)
        train_r = self._compute_performance(train_df, one(train_df), interval)
        val_r   = self._compute_performance(val_df,   one(val_df),   interval)
        test_r  = self._compute_performance(test_df,  one(test_df),  interval)
        test_sharpe_net   = test_r["sharpe_net"]
        test_sharpe_gross = test_r["sharpe_gross"]
        train_val_gap = abs(train_r["sharpe"] - val_r["sharpe"])
        _ann = BARS_PER_YEAR.get(interval, 252)
        _max_tvg = self._train_val_gap_tolerance(
            train_r["sharpe"], val_r["sharpe"], len(train_df), len(val_df),
            _ann, GATE_CONFIG.stage3_max_train_val_gap)

        # IS/OOS walk-forward on the NAV (70/30 row split)
        _n70 = int(n_bars * 0.70)
        _is, _oos = port_ret.iloc[:_n70], port_ret.iloc[_n70:]
        _sh = lambda r: (float(np.mean(r) / np.std(r) * np.sqrt(_ann))
                         if len(r) > 20 and np.std(r) > 1e-12 else 0.0)
        sharpe_is, sharpe_oos = _sh(_is.values), _sh(_oos.values)
        oos_deg = (sharpe_is - sharpe_oos) / abs(sharpe_is) if abs(sharpe_is) > 1e-9 else 0.0

        # Regime stress: vol-tercile buckets of the NAV's own returns
        _rv = port_ret.rolling(60).std()
        _q1, _q2 = _rv.quantile(0.33), _rv.quantile(0.66)
        regimes = {"sharpe_low_vol": _sh(port_ret[_rv <= _q1].values),
                   "sharpe_mid_vol": _sh(port_ret[(_rv > _q1) & (_rv <= _q2)].values),
                   "sharpe_high_vol": _sh(port_ret[_rv > _q2].values)}
        regimes_positive = sum(1 for v in regimes.values() if v > 0)

        # IC gate — the factor itself must be real across the universe
        ic = self.cross_sectional_test(row["factor_formula"] or fname, idea_id,
                                       factor={"name": fname, "params": fparams},
                                       interval=interval)
        ic_pass = bool(ic.get("factor_is_real"))

        # Benchmark gate — RISK-ADJUSTED (2026-07-10): the basket's net Sharpe
        # must beat holding the universe equal-weight. Raw ann returns are
        # computed for the report only — comparing raw return punished
        # market-neutral books in bull markets (category error).
        _ew = self._equal_weight_klci_returns(interval)
        _ew = _ew.reindex(close_p.index).fillna(0.0)
        _yrs = max(n_bars / _ann, 1e-9)
        strat_ann = float(nav.iloc[-1] ** (1.0 / _yrs) - 1.0)
        ew_ann = float((1.0 + _ew).prod() ** (1.0 / _yrs) - 1.0)
        _ew_std = float(np.std(_ew.values))
        ew_sharpe = (float(np.mean(_ew.values) / _ew_std * np.sqrt(_ann))
                     if _ew_std > 1e-12 else 0.0)
        _pr_std = float(np.std(port_ret.values))
        full_window_sharpe_net = (float(np.mean(port_ret.values) / _pr_std
                                        * np.sqrt(_ann))
                                  if _pr_std > 1e-12 else 0.0)
        # Like-for-like: full-window basket Sharpe vs full-window EW Sharpe.
        benchmark_pass = ((not GATE_CONFIG.benchmark_gate_enabled)
                          or full_window_sharpe_net >= ew_sharpe
                                                + GATE_CONFIG.benchmark_min_excess_ann)

        # Principal pass rule: deflated PSR (same machinery as the DSL path).
        from agents.backtest_engineer.stats import psr as _psr, deflated_sr_star
        with db_session() as conn:
            n_trials = conn.execute(
                "SELECT COUNT(DISTINCT idea_id) AS n FROM backtest_runs "
                "WHERE created_at >= datetime('now', ?)",
                (f"-{int(GATE_CONFIG.deflation_window_days)} days",),
            ).fetchone()["n"] + 1
            try:
                _sw = conn.execute(
                    "SELECT COALESCE(SUM(n_configs),0) AS n FROM optimizer_runs "
                    "WHERE idea_id=? AND status='done'", (idea_id,)).fetchone()["n"]
                n_trials += int(_sw or 0)
            except Exception:
                pass
        deflated_hurdle = deflated_sr_star(n_trials, n_bars, _ann)   # = SR*

        hp_class = "MEDIUM_TERM"
        dd_threshold = GATE_CONFIG.stage3_max_drawdown
        min_rebals = self._MIN_TRADES.get(hp_class, 30)
        trade_count_pass = n_rebalances >= min_rebals
        from agents.backtest_engineer.stats import moments as _moments
        # Principal rule on the FULL-WINDOW NAV evidence (same reasoning as
        # the DSL path: the test slice alone lacks statistical power; the OOS
        # walk-forward + regime guards enforce out-of-sample honesty).
        _fw_sk, _fw_ku = _moments(port_ret.values)
        psr_test = _psr(full_window_sharpe_net, deflated_hurdle,
                        len(port_ret), _ann, _fw_sk, _fw_ku)
        _tv_ret = port_ret.iloc[:len(train_df) + len(val_df)].values
        _tv_sk, _tv_ku = _moments(_tv_ret)
        _tv_std = float(np.std(_tv_ret))
        _tv_sr = (float(np.mean(_tv_ret) / _tv_std * np.sqrt(_ann))
                  if _tv_std > 1e-12 else 0.0)
        # diagnostic, not gated
        psr_trainval = _psr(_tv_sr,
                            deflated_sr_star(n_trials, max(len(_tv_ret), 2), _ann),
                            len(_tv_ret), _ann, _tv_sk, _tv_ku)
        # Same discipline as the DSL path: psr_trainval is DIAGNOSTIC only
        # (gating it would double-charge the evidence the full-window PSR
        # already weighs); gate2 carries the risk-mandate guards.
        gate2_pass = (train_r["max_dd"] <= dd_threshold
                      and val_r["max_dd"] <= dd_threshold
                      and train_val_gap <= _max_tvg)
        gate3_pass = (gate2_pass
                      and psr_test >= GATE_CONFIG.psr_confidence_test
                      and test_r["max_dd"] <= dd_threshold)
        # (the old cost_pass = net >= 0.4 was dominated by the Sharpe gate —
        # subsumed by PSR; XS costs are embedded per leg in the NAV already)
        oos_pass = sharpe_oos >= 0.30 and oos_deg <= 0.50
        regime_pass = regimes_positive >= 2
        overall_pass = (gate3_pass and trade_count_pass and oos_pass
                        and regime_pass and benchmark_pass
                        and ic_pass)

        verdict = "PASS" if overall_pass else "REJECTED"
        verdict_reason = " | ".join(filter(None, [
            "" if gate2_pass else "Gate2: train/val DD cap or train-val gap",
            "" if gate3_pass else (f"Gate3: test PSR {psr_test:.2f} < "
                                   f"{GATE_CONFIG.psr_confidence_test} "
                                   f"({n_trials} trials), or DD"),
            "" if trade_count_pass else f"rebalances {n_rebalances} < {min_rebals}",
            "" if oos_pass else f"OOS {sharpe_oos:.2f} (deg {oos_deg:.2f})",
            "" if regime_pass else f"regimes {regimes_positive}/3",
            "" if benchmark_pass else (f"benchmark: net Sharpe {test_sharpe_net:.2f} "
                                       f"< EW Sharpe {ew_sharpe:.2f} "
                                       f"(returns {strat_ann:.1%} vs {ew_ann:.1%}, report-only)"),
            "" if ic_pass else f"IC gate: mean_ic={ic.get('mean_ic')} t={ic.get('ic_tstat')} pos={ic.get('stocks_positive_ic')}",
        ]))

        result_data = {
            "factor": {"name": fname, "params": fparams},
            "top_n": top_n, "bottom_n": bottom_n,
            "rebalance_bars": rebalance_bars, "interval": interval,
            "names_used": sorted(scores.keys()),
            "coverage_notes": coverage_notes[:10],
            "n_rebalances": n_rebalances,
            "avg_turnover_cost_bar": round(float(costs.mean()), 6),
            "avg_funding_bar": round(float(funding.mean()), 6),
            "strat_ann_return": round(strat_ann, 4),
            "ew_ann_return": round(ew_ann, 4),
            "ic": {k: ic.get(k) for k in ("mean_ic", "ic_tstat",
                                           "stocks_positive_ic", "spread_sharpe",
                                           "quintile_sharpe")},
            "n_trials": n_trials, "deflated_hurdle": round(deflated_hurdle, 3),
            "universe_asof": "2026-07-09",
            "survivorship_note": ("universe is the CURRENT 20 majors — results "
                                  "carry survivorship bias"),
            "parked": bool(overall_pass),
        }

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
                idea_id, "cross_sectional",
                row["ticker"] or "UNIVERSE", interval, row["factor_formula"] or fname,
                train_r["sharpe_net"], val_r["sharpe_net"], test_sharpe_net,
                train_r["max_dd"], val_r["max_dd"], test_r["max_dd"],
                round(train_val_gap, 3), n_rebalances,
                test_r["win_rate"], test_r["profit_factor"],
                json.dumps(params), json.dumps(result_data),
                1 if overall_pass else 0, 0, None,
                hp_class, n_rebalances, n_rebalances,
                test_sharpe_gross, test_sharpe_net, test_sharpe_net, test_sharpe_gross,
                sharpe_is, sharpe_oos, sharpe_oos, round(oos_deg, 3),
                regimes["sharpe_low_vol"], regimes["sharpe_mid_vol"],
                regimes["sharpe_high_vol"], regimes_positive, None, test_r["max_dd"],
                verdict, verdict_reason or None,
            ))
            _stamp_versions(conn)
            if overall_pass:
                conn.execute(
                    "UPDATE alpha_ideas SET backtest_sharpe=?, backtest_dd=?, "
                    "stage='stage3', status='active', updated_at=datetime('now') "
                    "WHERE id=? AND stage='stage2'",
                    (test_sharpe_net, test_r["max_dd"], idea_id))
                conn.execute(
                    "INSERT INTO pipeline_events (idea_id, stage, event_type, agent, notes) "
                    "VALUES (?, 'stage3', 'parked', 'BacktestEngineer', ?)",
                    (idea_id, "PASSED all gates — PARKED at stage3: awaiting basket "
                              "paper-trading support (single-name executor cannot "
                              "run a cross-sectional book)"))
            else:
                conn.execute(
                    "UPDATE alpha_ideas SET status='rejected', rejection_reason=?, "
                    "updated_at=datetime('now') WHERE id=?",
                    (verdict_reason[:500], idea_id))
            conn.execute(
                "INSERT INTO pipeline_events (idea_id, stage, event_type, agent, notes) "
                "VALUES (?, 'stage2', ?, 'BacktestEngineer', ?)",
                (idea_id, "advanced" if overall_pass else "rejected",
                 f"XSect {fname} top{top_n}/bot{bottom_n} net={test_sharpe_net:.2f} "
                 f"IC={ic.get('mean_ic')} hurdle={deflated_hurdle:.2f}"))
            conn.execute(
                "INSERT INTO gate_decisions (idea_id, gate, decision, decided_by, rationale) "
                "VALUES (?, 'gate2_3_xs', ?, 'BacktestEngineer', ?)",
                (idea_id, "approve" if overall_pass else "reject",
                 (verdict_reason or "all gates passed")[:500]))

        self._clear_progress(idea_id)
        self.log_daemon(
            "INFO" if overall_pass else "WARN",
            f"XSect [{idea_id}] {verdict} {fname} net={test_sharpe_net:.2f} "
            f"IS={sharpe_is:.2f} OOS={sharpe_oos:.2f} IC={ic.get('mean_ic')} "
            f"rebals={n_rebalances} hurdle={deflated_hurdle:.2f}")

        return {"idea_id": idea_id, "run_type": "cross_sectional",
                "gate3_pass": overall_pass, "overall_pass": overall_pass,
                "verdict": verdict, "verdict_reason": verdict_reason,
                "test_sharpe_net": test_sharpe_net,
                "sharpe_is": sharpe_is, "sharpe_oos": sharpe_oos,
                "train_val_gap": round(train_val_gap, 3),
                "n_rebalances": n_rebalances,
                "psr_test": round(psr_test, 4), "psr_trainval": round(psr_trainval, 4),
                "deflated_hurdle": round(deflated_hurdle, 3), "n_trials": n_trials,
                "ic": result_data["ic"], "benchmark_pass": benchmark_pass,
                "strat_ann_return": strat_ann, "ew_ann_return": ew_ann,
                "ew_sharpe": round(ew_sharpe, 3),
                "parked": bool(overall_pass),
                "gates": {"gate2_pass": gate2_pass, "gate3_pass": gate3_pass,
                          "trade_count_pass": trade_count_pass,
                          "oos_pass": oos_pass, "regime_pass": regime_pass,
                          "benchmark_pass": benchmark_pass, "ic_pass": ic_pass}}

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
    # A fixed |train − val| Sharpe threshold (the old 0.30) is far tighter than
    # the sampling noise of a Sharpe estimated on a ~20% validation slice: the
    # standard error of an annualised Sharpe over a few hundred bars is ~0.5–0.7,
    # so the SE of the gap is ~0.8. A 0.30 fixed cap therefore rejects genuinely
    # stationary edges the majority of the time (proven by the calibration
    # harness: winner pass rate ~45%, 100% of rejections here). The gap is only
    # evidence of OVERFITTING when it exceeds what noise alone would produce, so
    # the tolerance is the max of the fixed floor and a k-sigma noise band.
    _TVG_SIGMA_K = 2.0   # allow gaps up to 2σ of sampling noise before flagging

    @staticmethod
    def _sharpe_stderr(sharpe_ann: float, n_bars: int, ann: float) -> float:
        """Standard error of an annualised Sharpe estimate (Lo 2002, IID form):
        se(SR_per_bar) = sqrt((1 + 0.5·SR_per_bar²)/n); annualise by sqrt(ann)."""
        if n_bars < 2 or ann <= 0:
            return float("inf")
        sr_pb = sharpe_ann / np.sqrt(ann)
        return float(np.sqrt((1.0 + 0.5 * sr_pb * sr_pb) / n_bars) * np.sqrt(ann))

    @classmethod
    def _train_val_gap_tolerance(cls, train_sharpe: float, val_sharpe: float,
                                 n_train: int, n_val: int, ann: float,
                                 floor: float) -> float:
        """Max allowable |train − val| Sharpe before it counts as overfitting.
        Never below ``floor`` (with lots of data a real 0.30 gap still matters),
        but widened to a k-sigma band of the gap's sampling noise for short
        slices — so genuine stationary edges are not rejected on Sharpe noise."""
        se_gap = float(np.hypot(cls._sharpe_stderr(train_sharpe, n_train, ann),
                                cls._sharpe_stderr(val_sharpe, n_val, ann)))
        return max(float(floor), cls._TVG_SIGMA_K * se_gap)

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
            RejectionMemory().record_rejection(idea_id, reason, reason_category)
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
            return self._reject_idea(
                idea_id, row, "dsl_unrepresentable",
                params.get("reason", "factor not representable as signal conditions"),
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
        verification = {"verified": False, "confidence": 0.0, "issue": ""}
        if params.get("signal_type") == "fundamental_screen":
            needs_review      = 0
            verification_note = ("Constant signal expected for fundamental screen "
                                 "(quarterly buy-and-hold) — formula verification N/A")
        elif params.get("signal_type") == "dsl":
            # Deterministic verification, and it BLOCKS — unlike the legacy
            # LLM verify_formula which only ever set needs_review=1.
            _full_sig = self._compute_signals(df, params)
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
        # Trade count is a statistical-significance requirement on the WHOLE
        # edge estimate, so it is counted over the full backtest window
        # (train+val+test, ~5yr), not the test slice alone. The old test-slice
        # count (~252 bars) made MEDIUM_TERM structurally unpassable: 10-60 day
        # holds cap out at 25 trades in 252 bars, below the 30-trade minimum —
        # the gate rejected the very class it was defined for, regardless of
        # edge (calibration finding, 2026-07-10).
        actual_trades = (train_r.get("total_trades", 0)
                         + val_r.get("total_trades", 0)
                         + test_r.get("total_trades", 0))

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
        # Noise-aware tolerance (see _train_val_gap_tolerance): the fundamental
        # branch keeps its own relaxed constant; active DSL signals get a
        # tolerance widened from the fixed floor by the Sharpe sampling noise of
        # these specific train/val slice lengths.
        if params.get("signal_type") == "fundamental_screen":
            _max_tvg = self.FUNDAMENTAL_SCREEN_THRESHOLDS["max_train_val_gap"]
        else:
            _max_tvg = self._train_val_gap_tolerance(
                train_r["sharpe"], val_r["sharpe"],
                len(train_df), len(val_df),
                BARS_PER_YEAR.get(interval, 252),
                GATE_CONFIG.stage3_max_train_val_gap)
        # ── Principal pass rule (2026-07-10 redesign): deflated PSR ──────────
        # ONE statistically grounded rule replaces the fixed per-class Sharpe
        # thresholds and the separate deflation binary: pass iff we are
        # confident the TRUE net Sharpe beats SR*, the expected max Sharpe of
        # the recent search's noise trials, given this sample's length and
        # return moments. Evidence-scaled: a moderate edge with years of data
        # can qualify; a strong-looking short fluke cannot. Confidence levels
        # live in GateConfig and are calibrated by the harness strength tiers.
        from agents.backtest_engineer.stats import psr as _psr, deflated_sr_star
        _ann_qc6 = BARS_PER_YEAR.get(interval, 252)
        _n_bars = max(len(train_df) + len(val_df) + len(test_df), 2)
        with db_session() as conn:
            # Trial count over a ROLLING window (not all history — a forever-
            # growing N silently raised the bar for every future idea
            # regardless of its own quality; audit finding, 2026-07-10).
            n_trials = conn.execute(
                "SELECT COUNT(DISTINCT idea_id) AS n FROM backtest_runs "
                "WHERE created_at >= datetime('now', ?)",
                (f"-{int(GATE_CONFIG.deflation_window_days)} days",),
            ).fetchone()["n"] + 1
            # Honest multiple-testing accounting: if this idea's params were
            # chosen by a parameter sweep, every swept config was a trial.
            try:
                _sweep_trials = conn.execute(
                    "SELECT COALESCE(SUM(n_configs), 0) AS n FROM optimizer_runs "
                    "WHERE idea_id=? AND status='done'", (idea_id,)
                ).fetchone()["n"]
                n_trials += int(_sweep_trials or 0)
            except Exception:
                pass
        deflated_hurdle = deflated_sr_star(n_trials, _n_bars, _ann_qc6)  # = SR*

        if params.get("signal_type") == "fundamental_screen":
            gate2_pass = (
                train_r["max_dd"] <= max_dd_threshold
                and val_r["max_dd"]   <= max_dd_threshold
                and train_val_gap     <= _max_tvg
            )
            gate3_pass = (
                gate2_pass
                and test_sharpe_net   >= sharpe_threshold
                and test_r["max_dd"]  <= max_dd_threshold
            )
            psr_test = psr_trainval = None
            full_window_sharpe_net = test_sharpe_net
        else:
            # The principal rule weighs the FULL WINDOW's evidence: that is
            # where a moderate true edge has statistical power (a 20% test
            # slice alone has SE ≈ 1 annualized — no realistic edge can be
            # 95%-confident on it, and its noise-max benchmark is huge).
            # Test-slice honesty is enforced by the ORTHOGONAL guards: the
            # OOS walk-forward gate (fresh-30% Sharpe floor + degradation
            # cap) and the regime/robustness checks. Parameter selection
            # never touches the test slice (optimizer scores train+val only),
            # so the full-window Sharpe is evaluated once per idea.
            _full_sig2 = self._compute_signals(df, params, symbol=symbol)
            _full_r = self._compute_performance(df, _full_sig2, interval)
            full_window_sharpe_net = _full_r["sharpe_net"]
            psr_test = _psr(full_window_sharpe_net, deflated_hurdle,
                            _full_r["n_obs"], _ann_qc6,
                            _full_r["skew"], _full_r["kurt"])
            # Pooled train+val PSR — reported for diagnostics, not gated
            # (gating it would double-charge the same evidence).
            _tv_df = pd.concat([train_df, val_df])
            _tv_sig = self._compute_signals(_tv_df, params, symbol=symbol)
            _tv_r = self._compute_performance(_tv_df, _tv_sig, interval)
            psr_trainval = _psr(_tv_r["sharpe_net"],
                                deflated_sr_star(n_trials, max(_tv_r["n_obs"], 2), _ann_qc6),
                                _tv_r["n_obs"], _ann_qc6,
                                _tv_r["skew"], _tv_r["kurt"])
            gate2_pass = (
                train_r["max_dd"] <= max_dd_threshold
                and val_r["max_dd"]   <= max_dd_threshold
                and train_val_gap     <= _max_tvg
            )
            gate3_pass = (
                gate2_pass
                and psr_test >= GATE_CONFIG.psr_confidence_test
                and test_r["max_dd"]  <= max_dd_threshold
            )
            if not gate3_pass:
                self.log_daemon(
                    "WARN",
                    f"Backtest [{idea_id}] PSR gate: full-window PSR "
                    f"{psr_test:.2f} (need {GATE_CONFIG.psr_confidence_test}) "
                    f"vs SR*={deflated_hurdle:.2f} ({n_trials} trials/"
                    f"{GATE_CONFIG.deflation_window_days}d), or DD/gap")

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

        # QC3: cost sensitivity gate — the DRAG arm only. (The old
        # "net < 0.4" arm was dominated by the Sharpe gate and is now
        # subsumed by PSR; audit finding, 2026-07-10.)
        cost_pass = True
        cost_note = ""
        if test_sharpe_gross - test_sharpe_net > 0.8:
            cost_pass = False
            cost_note = (f"Strategy is cost-sensitive — gross Sharpe {test_sharpe_gross:.2f} "
                         f"degrades to net {test_sharpe_net:.2f} after transaction costs")
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

        # QC6: multiple-testing deflation — SUBSUMED by the PSR principal rule
        # above (SR* IS the deflated benchmark inside PSR); no separate binary.
        # deflated_hurdle and n_trials were computed there and are persisted
        # for traceability.
        deflation_note = ""

        # QC7: parameter robustness (DSL signals) — a real edge survives ±20%
        # parameter jitter; a knife-edge fit does not. Pure numpy, no LLM cost.
        robustness_score = None
        robustness_pass = True
        robustness_note = ""
        if params.get("signal_type") == "dsl" and test_sharpe_net > 0:
            robustness_score = self._robustness_check(
                test_df, params["dsl"], test_sharpe_net, interval)
            robustness_pass = robustness_score >= GATE_CONFIG.robustness_min_fraction
            if not robustness_pass:
                robustness_note = (
                    f"Parameter fragility: only {robustness_score:.0%} of ±20% "
                    f"parameter perturbations retain >"
                    f"{GATE_CONFIG.robustness_sharpe_ratio:.0%} of base Sharpe "
                    f"(need {GATE_CONFIG.robustness_min_fraction:.0%}) — knife-edge fit"
                )
                self.log_daemon(
                    "WARN", f"Backtest [{idea_id}] robustness gate FAILED: {robustness_note}")

        # ── Benchmark: excess performance vs the market index (profile symbol) ─
        strat_ann = float(test_r.get("ann_return", 0.0))
        benchmark_sharpe, excess_ann_return = 0.0, 0.0
        try:
            bench_df = self._fetch_prices(BENCHMARK_SYMBOL, interval, days=1825)
            if not bench_df.empty:
                bench_ret = bench_df["close"].pct_change().reindex(df.index).dropna()
                if len(bench_ret) > 60 and float(np.std(bench_ret)) > 1e-10:
                    benchmark_sharpe = float(
                        np.mean(bench_ret) / np.std(bench_ret) * np.sqrt(_ann_qc6)
                    )
                    excess_ann_return = float(strat_ann - np.mean(bench_ret) * _ann_qc6)
        except Exception as _bench_exc:
            self.log_daemon("WARN", f"Backtest [{idea_id}] benchmark fetch failed: {_bench_exc}")

        # ── Benchmark gate (Phase 3.2, RISK-ADJUSTED since 2026-07-10): the
        # strategy's net Sharpe must beat simply holding the universe equal-
        # weight. The old raw-return comparison punished market-neutral /
        # long-short books in bull markets (a Sharpe-2 neutral strategy would
        # fail against a levered-beta rally) — a category error. Raw ann-return
        # excess is still computed and stored, REPORT-ONLY.
        equal_weight_sharpe, excess_vs_ew_ann_return = 0.0, 0.0
        benchmark_pass = True
        benchmark_note = ""
        try:
            ew_ret = self._equal_weight_klci_returns(interval).reindex(df.index).dropna()
            if len(ew_ret) > 60 and float(np.std(ew_ret)) > 1e-10:
                equal_weight_sharpe = float(
                    np.mean(ew_ret) / np.std(ew_ret) * np.sqrt(_ann_qc6)
                )
                ew_ann = float(np.mean(ew_ret) * _ann_qc6)
                excess_vs_ew_ann_return = float(strat_ann - ew_ann)  # report-only
                if GATE_CONFIG.benchmark_gate_enabled:
                    # Like-for-like evidence: FULL-WINDOW strategy Sharpe vs
                    # full-window EW Sharpe (the test slice alone is too noisy
                    # to compare against a diversified-basket Sharpe).
                    benchmark_pass = (
                        full_window_sharpe_net >= equal_weight_sharpe
                                           + GATE_CONFIG.benchmark_min_excess_ann
                    )
                    if not benchmark_pass:
                        benchmark_note = (
                            f"Benchmark gate: full-window net Sharpe "
                            f"{full_window_sharpe_net:.2f} does not beat "
                            f"equal-weight universe Sharpe {equal_weight_sharpe:.2f} "
                            f"(raw-return excess {excess_vs_ew_ann_return:+.1%}, report-only)"
                        )
            else:
                self.log_daemon(
                    "WARN", f"Backtest [{idea_id}] benchmark gate SKIPPED — "
                            f"insufficient equal-weight data (fail-open, disclosed)")
        except Exception as _ew_exc:
            # Benchmark data unavailable → do not block on it (fail-open, warn).
            self.log_daemon("WARN", f"Backtest [{idea_id}] equal-weight benchmark failed "
                                    f"(fail-open, disclosed): {_ew_exc}")

        # ── Sanity flags (warn but do not auto-reject) ────────────────────────
        sanity_flags = self._detect_sanity_flags(
            test_sharpe_gross, test_r["max_dd"], test_r["win_rate"], actual_trades, interval,
        )
        for flag in sanity_flags:
            self.log_daemon("WARN", f"Backtest [{idea_id}] SANITY FLAG: {flag}")

        overall_pass = (gate3_pass and trade_count_pass and cost_pass
                        and oos_pass and regime_pass
                        and robustness_pass and benchmark_pass and capacity_pass)

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
                f"Active strategy passes all gates: net Sharpe {test_sharpe_net:.2f} "
                f"(PSR {psr_test:.2f} vs SR* {deflated_hurdle:.2f}, {n_trials} trials), "
                f"OOS={sharpe_oos:.2f}, regimes={regimes_positive}/3"
            )
        else:
            verdict = "reject"
            _psr_note = ""
            if psr_trainval is not None and not gate2_pass:
                _psr_note = (f"Gate2: train+val PSR {psr_trainval:.2f} < "
                             f"{GATE_CONFIG.psr_confidence_trainval} vs SR* "
                             f"{deflated_hurdle:.2f} ({n_trials} trials/"
                             f"{GATE_CONFIG.deflation_window_days}d), or DD/gap")
            elif psr_test is not None and not gate3_pass:
                _psr_note = (f"Gate3: test PSR {psr_test:.2f} < "
                             f"{GATE_CONFIG.psr_confidence_test} vs SR* "
                             f"{deflated_hurdle:.2f} — not confidently above "
                             f"the {n_trials}-trial noise benchmark, or DD")
            verdict_reason = " | ".join(filter(None, [
                _psr_note or ("" if gate2_pass else "Gate2 failed (Sharpe or DD)"),
                "" if _psr_note or gate3_pass else "Gate3 failed (test Sharpe or DD)",
                "" if cost_pass    else cost_note,
                "" if oos_pass     else oos_note,
                "" if regime_pass  else regime_note,
                "" if trade_count_pass else trade_count_note,
                "" if robustness_pass else robustness_note,
                "" if benchmark_pass else benchmark_note,
                "" if capacity_pass else capacity_note,
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
                    json.dumps(params),
                    json.dumps({**results, "funding_source": funding_source}),
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
                        psr_test=?, psr_trainval=?
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
                      run_id))

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

                if cur_stage != "stage2":
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

        # Knowledge graph: surviving ideas link to the techniques they used,
        # so the graph accumulates evidence about which knowledge works.
        # (Outside the save transaction — the store opens its own connection.)
        if overall_pass and params.get("signal_type") == "dsl":
            self._link_idea_to_techniques(idea_id, row["title"], params["dsl"])

        # ── Save equity curve to backtest_series ─────────────────────────────
        try:
            sig_full = self._compute_signals(df, params)
            if _has_custom_exit:
                sig_full = self._compute_signal_with_exits(df, sig_full, _exit_profile)
            sig_shifted = sig_full.shift(1).fillna(0)
            bar_returns = df["close"].pct_change().fillna(0)
            _rates      = self._cost_rates(df)
            _deltas     = sig_shifted.diff().fillna(0)
            _cost_bar   = (_deltas.clip(lower=0) * _rates["buy"]
                           + (-_deltas).clip(lower=0) * _rates["sell"])
            net_bar     = sig_shifted * bar_returns - _cost_bar
            equity      = (1 + net_bar.clip(-0.5, 0.5)).cumprod()
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
            with db_session() as conn:
                conn.execute("DELETE FROM backtest_series WHERE idea_id=?", (idea_id,))
                rows_eq = [
                    (idea_id, str(d)[:10],
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
            self.log_daemon("INFO",
                f"Backtest [{idea_id}] saved {len(rows_eq)} equity curve points to backtest_series")
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
                _stamp_versions(conn)
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

        # ── Save equity curve to backtest_series ─────────────────────────────
        try:
            oos_start = int(len(nav) * 0.70)
            peak      = nav.expanding().max()
            dd_series = (nav - peak) / peak.clip(lower=1e-9)
            bench_curve = None
            try:
                _bdf = self._fetch_prices(BENCHMARK_SYMBOL, "1d", days=1825)
                if not _bdf.empty:
                    _bret = _bdf["close"].pct_change().reindex(nav.index).fillna(0)
                    bench_curve = (1 + _bret).cumprod()
            except Exception:
                bench_curve = None
            with db_session() as conn:
                conn.execute("DELETE FROM backtest_series WHERE idea_id=?", (idea_id,))
                rows_eq = [
                    (idea_id, str(d)[:10],
                     float(v) - 1.0,
                     float(bench_curve.iloc[i]) - 1.0 if bench_curve is not None else 0.0,
                     float(dd_series.iloc[i]), 1 if i >= oos_start else 0)
                    for i, (d, v) in enumerate(zip(nav.index, nav.values))
                ]
                conn.executemany(
                    "INSERT INTO backtest_series "
                    "(idea_id, date, strategy_pct, benchmark_pct, drawdown_pct, is_oos) "
                    "VALUES (?,?,?,?,?,?)", rows_eq,
                )
            self.log_daemon("INFO",
                f"FundScreen [{idea_id}] saved {len(rows_eq)} equity curve points to backtest_series")
        except Exception as _eq_exc:
            self.log_daemon("WARN",
                f"FundScreen [{idea_id}] could not save equity series: {_eq_exc}")

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
            "train_val_gap_tol":   round(_max_tvg, 3),
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
