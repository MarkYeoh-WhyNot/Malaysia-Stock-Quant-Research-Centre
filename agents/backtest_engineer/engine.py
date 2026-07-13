"""Backtest computation engine: vectorized signal evaluation and PnL math.

This module contains the pure computational functions that drive backtesting:
signal generation from parameters, per-bar return calculation (the single
source of truth for net PnL), trade reconstruction, performance metrics
(Sharpe, regimes, sanity flags), and exit-logic post-processing.

All functions that were instance methods (taking `self`) are refactored to
accept `engine` as the first parameter for dependency injection.
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from config.settings import (
    DEFAULT_LEVERAGE, MAX_LEVERAGE, LIQUIDATION_BUFFER,
    FUNDING_INTERVAL_HOURS, AVG_FUNDING_RATE_PER_INTERVAL,
)
from data.market_data import BARS_PER_YEAR
from config.settings import bars_per_day as _bars_per_day

logger = logging.getLogger(__name__)


def _is_subdaily_interval(interval: str) -> bool:
    """True for hour/minute/second bars, False for day/week/month.

    Parses the interval STRING (unit letters, digits stripped) rather than
    looking up BARS_PER_YEAR — the profile table only carries the active
    market's intervals (bursa has no "4h"), so a lookup would mis-classify a
    crypto interval as daily whenever this runs under a different profile.
    """
    unit = "".join(c for c in (interval or "").lower() if c.isalpha())
    return unit in ("h", "hr", "hrs", "hour", "hours",
                    "m", "min", "mins", "minute", "minutes", "s", "sec", "secs")


def series_date_key(ts, interval: str) -> str:
    """Row key for a backtest_series point.

    Daily-and-slower intervals key on the calendar date (YYYY-MM-DD).
    Sub-daily intervals (4h/1h/15m…) MUST keep the intraday time, or every
    bar on the same calendar day collapses to one key and collides on
    backtest_series' UNIQUE(idea_id, date) — which raised, aborted the whole
    persist block, and silently lost BOTH the equity curve and the trade
    blotter for every sub-daily (crypto) backtest (idea 232, 2026-07-14).
    Weekly/daily stay date-only so Bursa output is byte-identical.
    """
    return str(ts)[:16] if _is_subdaily_interval(interval) else str(ts)[:10]


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


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, 1e-9)
    return 100 - (100 / (1 + rs))


def _compute_signals(engine, df: pd.DataFrame, params: dict,
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
            frame["cpo_close"] = engine._fetch_cpo_series(df.index)
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
            frame["funding_rate"] = engine._fetch_funding_column(symbol, df.index)
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
        rsi      = _rsi(close, int(params.get("rsi_period", 14)))
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


def _net_return_series(engine, df: pd.DataFrame, signals: pd.Series, interval: str,
                       leverage: float | None = None, lag: int = 1,
                       extra_cost_per_side: float = 0.0) -> dict:
    """SINGLE SOURCE OF TRUTH for per-bar net returns.

    Every consumer of "what did this strategy earn each bar" — the gated
    metrics (_compute_performance), the regime stress test
    (_compute_regimes), and the persisted equity curve — MUST route
    through here, so the drawdown curve shown on the dashboard is the same
    series behind the gated Sharpe (previously they were re-derived
    independently and omitted funding/leverage/liquidation → diverged on
    crypto). Returns date-indexed Series (bar 0 = 0.0, no position at
    start) so callers keep date alignment.

    Encapsulates the full model: QC1 one-bar signal lag, QC3 per-side
    costs on position deltas, WS3 signed positions + real/modeled funding
    + leverage + bounded per-bar liquidation. Each WS3 term is a
    documented no-op on Bursa (FUNDING_INTERVAL_HOURS=None, leverage 1.0).

    Returns {net, gross, signal_shifted (all indexed Series),
    leverage_used, funding_drag_pct}.
    """
    close = df["close"]
    sig   = signals.fillna(0)

    # QC1: strict signal delay (signal_shifted[t] = signal[t-lag]). lag=1 is
    # the research convention (fill at the signal-generating close); lag=2
    # is the conservative fill-robustness variant (cede a full bar — if the
    # edge dies at lag=2 it was living on the signal bar's own move).
    signal_shifted = sig.shift(lag).fillna(0)
    assert float(signal_shifted.iloc[0]) == 0.0, \
        "Lookahead guard failure: signal_shifted[0] != 0"

    bar_returns = close.pct_change().fillna(0)
    gross_bar   = signal_shifted * bar_returns   # sign-correct for shorts

    # QC3: per-side costs on every position change (bar 0 cost = 0).
    rates  = engine._cost_rates(df, interval)
    deltas = np.diff(signal_shifted.values)
    cost   = pd.Series(0.0, index=df.index)
    # extra_cost_per_side: optional size-aware market-impact haircut for the
    # REPORTED capacity-adjusted variant — 0.0 in the gated path so the
    # gated Sharpe is never touched.
    cost.iloc[1:] = (np.clip(deltas, 0, None) * rates["buy"]
                     + np.clip(-deltas, 0, None) * rates["sell"]
                     + np.abs(deltas) * extra_cost_per_side)

    # WS3: funding accrual (crypto only; charged on the LAGGED position,
    # never used to form the signal — no lookahead). Real per-bar
    # settlements when the caller attached funding_bar_sum, else the
    # disclosed modeled constant scaled by settlements-per-bar.
    leverage = min(leverage if leverage else DEFAULT_LEVERAGE, MAX_LEVERAGE)
    bar_days = 365.0 / BARS_PER_YEAR.get(interval, 252)
    settlements_per_bar = (bar_days * 24.0 / FUNDING_INTERVAL_HOURS) if FUNDING_INTERVAL_HOURS else 0.0
    if "funding_bar_sum" in df.columns and FUNDING_INTERVAL_HOURS:
        funding = signal_shifted * df["funding_bar_sum"].fillna(0.0)
    else:
        funding = signal_shifted * AVG_FUNDING_RATE_PER_INTERVAL * settlements_per_bar

    # WS3: leverage + bounded per-bar liquidation (inert at leverage 1.0).
    net = (gross_bar - cost - funding) * leverage
    if leverage > 1.0:
        liq_floor = -(1.0 / leverage) * (1.0 - LIQUIDATION_BUFFER)
        net = net.clip(lower=liq_floor)
    # Funding PnL contribution sign: funding is SUBTRACTED above, so its
    # contribution is the negative of the summed drag.
    funding_drag_pct = float(-(funding * leverage).sum())

    return {
        "net": net, "gross": gross_bar, "signal_shifted": signal_shifted,
        "cost": cost, "funding": (funding if isinstance(funding, pd.Series)
                                  else pd.Series(funding, index=df.index)),
        "leverage_used": leverage, "funding_drag_pct": funding_drag_pct,
    }


def _reconstruct_trades(engine, df: pd.DataFrame, signals: pd.Series,
                        interval: str) -> list[dict]:
    """Reconstruct the discrete trade blotter from the vectorized position
    series. A trade is a maximal run of constant non-zero (lagged)
    position; PnL is attributed from the SAME net-return series behind the
    gated Sharpe, so summed net_pct reconciles to the backtest return.
    The backtest never places orders — this is faithful to the math, not
    an independent order log."""
    r = _net_return_series(engine, df, signals, interval)
    pos     = r["signal_shifted"].values
    gross   = r["gross"].values
    funding = r["funding"].values
    cost    = r["cost"].values
    close   = df["close"].values
    dates   = df.index
    n = len(pos)
    oos_start = int(n * 0.70)

    # Split each transition cost (charged at the bar the position CHANGES)
    # into the portion that CLOSES the prior position and the portion that
    # OPENS the new one, proportional to the units on each side. So every
    # bar's gross/funding and every cost fragment is attributed to exactly
    # one trade → summed trade net reconciles to the net-return series.
    open_cost  = np.zeros(n)
    close_cost = np.zeros(n)
    prev = 0.0
    for b in range(1, n):
        c = cost[b]
        if c:
            a_prev, a_new = abs(prev), abs(pos[b])
            denom = a_prev + a_new
            if denom > 0:
                close_cost[b] = c * a_prev / denom
                open_cost[b]  = c * a_new / denom
        prev = pos[b]

    trades: list[dict] = []
    i, seq = 1, 0   # bar 0 is always flat
    while i < n:
        if pos[i] == 0:
            i += 1
            continue
        sign = pos[i]
        j = i
        while j < n and pos[j] == sign:
            j += 1
        # Held bars i..j-1 (entry filled at close[i-1], exit at close[j-1]).
        gseg = float(np.sum(gross[i:j]))
        fseg = float(np.sum(funding[i:j]))
        entry_cost = float(open_cost[i])
        exit_cost  = float(close_cost[j]) if j < n else 0.0
        cseg = entry_cost + exit_cost
        nseg = gseg - fseg - cseg
        seq += 1
        trades.append({
            "seq": seq,
            "direction": "long" if sign > 0 else "short",
            "entry_date": str(dates[i - 1])[:19],
            "exit_date":  str(dates[j - 1])[:19],
            "entry_price": round(float(close[i - 1]), 6),
            "exit_price":  round(float(close[j - 1]), 6),
            "bars_held":   int(j - i),
            "gross_pct":   round(gseg * 100.0, 4),
            "cost_pct":    round(cseg * 100.0, 4),
            "net_pct":     round(nseg * 100.0, 4),
            "is_oos":      1 if i >= oos_start else 0,
        })
        i = j
    return trades


def _compute_performance(engine, df: pd.DataFrame, signals: pd.Series, interval: str,
                         leverage: float | None = None, lag: int = 1,
                         extra_cost_per_side: float = 0.0) -> dict:
    """Compute performance with QC1 lookahead guard and QC3 realistic costs.

    QC1 — Lookahead bias guard:
      The position at bar t earns the return from bar t. `signals` is
      delayed `lag` bars (default 1 — the signal computed on close t-1 is
      traded at that close, earning close t-1 -> close t; close-to-close,
      no lookahead). lag=2 is the conservative fill-robustness variant.
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
    # Route through the single source of truth so the gated Sharpe, the
    # regime test, and the persisted equity curve are the SAME series.
    r = _net_return_series(engine, df, signals, interval, leverage, lag, extra_cost_per_side)
    net_returns      = r["net"].values[1:]     # drop bar 0 (always 0)
    gross_returns    = r["gross"].values[1:]
    signal_changes   = np.abs(np.diff(r["signal_shifted"].values))
    leverage         = r["leverage_used"]
    funding_drag_pct = r["funding_drag_pct"]

    n = len(net_returns)
    _empty = {
        "sharpe": 0.0, "sharpe_gross": 0.0, "sharpe_net": 0.0,
        "max_dd": 1.0, "win_rate": 0.0, "profit_factor": 0.0,
        "total_trades": 0, "ann_return": 0.0, "cagr": 0.0,
        "ulcer_index": 0.0, "avg_drawdown": 0.0, "dd_duration_bars": 0,
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

    # CAGR — compounded (geometric) annual return, unlike the arithmetic
    # ann_return below. This is the figure an investor actually realizes;
    # ann_return is kept because gates / benchmark excess read it.
    cagr = float(cum[-1] ** (ann / n) - 1.0) if cum[-1] > 0 else -1.0

    # Drawdown QUALITY (two books at the same max_dd are not equal):
    #   ulcer_index   = RMS of the drawdown path (penalises deep + long)
    #   avg_drawdown  = mean depth while underwater
    #   dd_duration   = longest consecutive underwater run, in bars
    ulcer_index  = float(np.sqrt(np.mean(dd ** 2)))
    _underwater  = dd > 1e-9
    avg_drawdown = float(dd[_underwater].mean()) if _underwater.any() else 0.0
    _max_run = _run = 0
    for _u in _underwater:
        _run = _run + 1 if _u else 0
        if _run > _max_run:
            _max_run = _run
    dd_duration_bars = int(_max_run)

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
        "cagr":          round(cagr, 4),
        "ulcer_index":   round(ulcer_index, 4),
        "avg_drawdown":  round(avg_drawdown, 4),
        "dd_duration_bars": dd_duration_bars,
        "leverage_used": leverage,
        "funding_drag_pct": round(funding_drag_pct, 4),
        "n_obs":         n,
        "skew":          round(ret_skew, 3),
        "kurt":          round(ret_kurt, 3),
    }

# ── QC2: Walk-forward IS / OOS validation ────────────────────────────────


def _compute_walk_forward(engine, df: pd.DataFrame, params: dict, interval: str) -> dict:
    """Split full dataset 70/30 and compute Sharpe separately for IS and OOS periods.

    Returns sharpe_is, sharpe_oos, oos_degradation (fraction drop from IS to OOS).
    """
    n        = len(df)
    split_at = int(n * 0.70)
    is_df    = df.iloc[:split_at]
    oos_df   = df.iloc[split_at:]

    if len(is_df) < 60 or len(oos_df) < 20:
        return {"sharpe_is": 0.0, "sharpe_oos": 0.0, "oos_degradation": 0.0}

    is_perf  = _compute_performance(engine, is_df,  _compute_signals(engine, is_df,  params), interval)
    oos_perf = _compute_performance(engine, oos_df, _compute_signals(engine, oos_df, params), interval)

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


def _compute_regimes(engine, df: pd.DataFrame, params: dict, interval: str) -> dict:
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
    rolling_vol = close.pct_change().rolling(60).std() * np.sqrt(BARS_PER_YEAR.get(interval, 252))  # annualised

    # Signals on full series (so MAs etc. have full context), then per-bar
    # NET return via the single source of truth — regime attribution now
    # uses the same funding/leverage-aware series as the gated Sharpe
    # (previously it re-derived returns and omitted funding on crypto).
    sig     = _compute_signals(engine, df, params)
    net_bar = _net_return_series(engine, df, sig, interval)["net"]

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


def _get_exit_profile_by_key(engine, strategy_key: str) -> dict:
    """Return the exit profile for a given strategy_key, or the default profile."""
    return engine.EXIT_PROFILES.get(strategy_key or "", engine._DEFAULT_EXIT_PROFILE)


def _apply_exit_logic(
    engine,
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


