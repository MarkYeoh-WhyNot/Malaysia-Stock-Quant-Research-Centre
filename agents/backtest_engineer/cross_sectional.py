"""Cross-sectional strategy validation and basket backtesting.

Two related functions that test whether a factor (or a single-name idea's
parsed signal, legacy path) generalises across the full universe rather
than working on one lucky stock:

  * ``cross_sectional_test`` — computes cross-sectional IC (Spearman rank
    correlation between factor score and next-bar return, at each date)
    plus a quintile-portfolio Sharpe. This is both a standalone IC check
    AND the IC gate consumed by ``run_cross_sectional_backtest`` below.
  * ``run_cross_sectional_backtest`` — the gated long-top/short-bottom
    basket backtest across the universe (~20 majors), running the NAV
    through the standard gate stack (PSR, DD caps, OOS walk-forward,
    regime stress, benchmark) plus the IC gate.

Both take `engine` (a BacktestEngineer instance) as their first argument,
to reach its data-fetch/cost/logging helpers and (for the legacy path)
signal parsing. `run_cross_sectional_backtest` calls `cross_sectional_test`
as a plain sibling function in this module (not `engine.cross_sectional_test`)
since both moved out of the class together.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from config.settings import (
    GATE_CONFIG, KLCI_BY_SYMBOL, DEFAULT_SYMBOLS,
    FUNDING_INTERVAL_HOURS,
)
from config.settings import FETCH_DAYS_BY_INTERVAL as _FETCH_DAYS
from data.database import db_session
from data.market_data import BARS_PER_YEAR
from agents.backtest_engineer import stats
from agents.backtest_engineer import engine as engine_mod
from agents.backtest_engineer.engine import _funding_bar_sum


def _ic_series(sig_panel: pd.DataFrame, ret_panel: pd.DataFrame,
               spread_ok: bool) -> tuple:
    """Per-date cross-sectional IC + top-quintile (+ top-minus-bottom, when
    ``spread_ok``) portfolio returns. Shared by ``cross_sectional_test`` (full
    window) and the xs-factor sweep's train+val screening (optimizer.py) so
    both compute IC identically — extracted, not duplicated.

    Returns ``(ic_series, portfolio_rets, spread_rets)``, each a plain list.
    """
    ic_series: list[float] = []
    portfolio_rets: list[float] = []
    spread_rets: list[float] = []   # top-minus-bottom (factor mode, shorts allowed)

    for date in sig_panel.index:
        sig_row = sig_panel.loc[date].dropna()
        ret_row = ret_panel.loc[date].dropna()
        common_stocks = sig_row.index.intersection(ret_row.index)
        if len(common_stocks) < 5:
            continue

        sv = sig_row[common_stocks].values
        rv = ret_row[common_stocks].values

        ic = stats.spearman(sv, rv)
        if not np.isnan(ic):
            ic_series.append(ic)

        # Top-quintile portfolio (top ~20% of available stocks that day)
        n_q = max(1, len(common_stocks) // 5)
        top_idx = np.argsort(sv)[-n_q:]
        if len(top_idx) > 0:
            portfolio_rets.append(float(np.mean(rv[top_idx])))
        if spread_ok:
            bot_idx = np.argsort(sv)[:n_q]
            spread_rets.append(float(np.mean(rv[top_idx]) - np.mean(rv[bot_idx])))

    return ic_series, portfolio_rets, spread_rets


def cross_sectional_test(engine, factor_formula: str, idea_id: int,
                         factor: dict | None = None,
                         interval: str = "1d", days: int = 730,
                         persist: bool = True) -> dict:
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

    ``persist`` controls whether the IC columns are UPDATEd onto the idea's
    latest backtest_runs row. That UPDATE assumes the DSL/legacy backtest
    has already INSERTed its row (true for the research_daemon legacy-veto
    call site). ``run_cross_sectional_backtest`` calls this BEFORE its own
    INSERT, so it passes ``persist=False`` and folds the IC values into its
    own INSERT instead — otherwise the UPDATE silently matches no row and
    the queryable mean_ic/ic_tstat/stocks_positive_ic/best_stocks columns
    stay NULL (the numbers still land in result_data JSON either way).
    """
    with db_session() as conn:
        row = conn.execute("SELECT * FROM alpha_ideas WHERE id=?", (idea_id,)).fetchone()
    if not row:
        return {"error": f"Idea {idea_id} not found", "factor_is_real": False}

    params = None
    if factor is None:
        formula = factor_formula or row["factor_formula"] or ""
        params  = engine._parse_factor(formula, row["title"], row["hypothesis"] or "")
        if not params or "error" in params:
            params = {"signal_type": "momentum", "momentum_period": 20, "long_only": True}
    else:
        from agents.backtest_engineer import factors as factor_registry
        interval = interval or "1d"
        days = _FETCH_DAYS.get(interval, days)

    engine.log_daemon(
        "INFO", f"CrossSect [{idea_id}]: {interval} data for "
                f"{len(DEFAULT_SYMBOLS)} names "
                f"({'factor ' + factor['name'] if factor else 'legacy signal'})")

    # ── Build score + forward-return panels ──────────────────────────────
    signal_series: dict[str, pd.Series] = {}
    return_series: dict[str, pd.Series] = {}

    for symbol in DEFAULT_SYMBOLS:
        try:
            df = engine._fetch_prices(symbol, interval if factor else "1d",
                                    days=days)
            if df.empty or len(df) < 60:
                continue
            if factor is not None:
                for _col in factor_registry.required_columns(factor["name"]):
                    if _col == "funding_rate" and _col not in df.columns:
                        df[_col] = engine._fetch_funding_column(symbol, df.index)
                sig = factor_registry.compute_factor(
                    factor["name"], df, factor.get("params"))
                fwd_ret = df["close"].pct_change().shift(-1)
                # continuous scores: 0 is a legitimate value — keep it
                valid = sig.notna() & fwd_ret.notna()
            else:
                sig     = engine_mod._compute_signals(engine, df, params)
                fwd_ret = df["close"].pct_change().shift(-1)   # next-bar return
                valid   = sig.notna() & fwd_ret.notna() & (sig != 0)
            if valid.sum() < 20:
                continue
            signal_series[symbol] = sig[valid]
            return_series[symbol] = fwd_ret[valid]
        except Exception as e:
            engine.log_daemon("WARN", f"CrossSect: skipped {symbol}: {e}")

    n_stocks = len(signal_series)
    if n_stocks < 5:
        engine.log_daemon("WARN", f"CrossSect [{idea_id}]: only {n_stocks} stocks have data")
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
    from config.settings import ALLOW_SHORT as _allow_short
    _spread_ok = factor is not None and _allow_short
    ic_series, portfolio_rets, spread_rets = _ic_series(sig_panel, ret_panel, _spread_ok)

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
    ic_tstat = stats.nw_tstat(ic_arr)

    # ── Per-stock IC (how predictive is the factor within each ticker) ────
    stock_ics: dict[str, float] = {}
    for sym in sig_panel.columns:
        sig_ts = sig_panel[sym].dropna()
        ret_ts = ret_panel[sym].dropna()
        overlap = sig_ts.index.intersection(ret_ts.index)
        if len(overlap) < 20:
            continue
        ic = stats.spearman(sig_ts[overlap].values, ret_ts[overlap].values)
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
    # (skipped when the caller will fold these into its own INSERT — see
    # the ``persist`` docstring note above)
    if persist:
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
            engine.log_daemon("WARN", f"CrossSect: failed to save IC stats for [{idea_id}]: {e}")

    engine.log_daemon(
        "INFO" if factor_is_real else "WARN",
        f"CrossSect [{idea_id}] {'REAL' if factor_is_real else 'WEAK'} "
        f"mean_IC={mean_ic:.3f} t={ic_tstat:.2f} pos_stocks={stocks_positive_ic}/{n_stocks} "
        f"q_sharpe={quintile_sharpe:.2f}",
    )
    return result


def run_cross_sectional_backtest(engine, idea_id: int, row: dict,
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
        return engine._reject_idea(idea_id, row, "xs_factor",
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

    engine.log_daemon(
        "INFO", f"XSect backtest [{idea_id}]: factor={fname}{fparams} "
                f"top{top_n}/bottom{bottom_n} rebal={rebalance_bars} bars "
                f"{interval} across {len(DEFAULT_SYMBOLS)} names")
    engine._log_progress(idea_id, 15, f"Building {len(DEFAULT_SYMBOLS)}-name factor panel")

    # ── Panel build ───────────────────────────────────────────────────────
    closes: dict[str, pd.Series] = {}
    scores: dict[str, pd.Series] = {}
    fundmap: dict[str, pd.Series] = {}
    side_rate: dict[str, float] = {}
    coverage_notes: list[str] = []
    for symbol in DEFAULT_SYMBOLS:
        try:
            df = engine._fetch_prices(symbol, interval, days=days)
            if df.empty or len(df) < 100:
                coverage_notes.append(f"{symbol}: {0 if df.empty else len(df)} bars — excluded")
                continue
            if needs_funding and "funding_rate" not in df.columns:
                df["funding_rate"] = engine._fetch_funding_column(symbol, df.index)
            scores[symbol] = factor_registry.compute_factor(fname, df, fparams)
            closes[symbol] = df["close"]
            if FUNDING_INTERVAL_HOURS:
                _f = engine._fetch_funding_history(symbol)
                fundmap[symbol] = _funding_bar_sum(_f, df.index)
            _r = engine._cost_rates(df, interval)
            side_rate[symbol] = (_r["buy"] + _r["sell"]) / 2.0
        except Exception as exc:
            coverage_notes.append(f"{symbol}: {exc}")

    if len(scores) < max(5, top_n + bottom_n):
        return engine._reject_idea(
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
        return engine._reject_idea(idea_id, row, "xs_history",
                                 f"only {n_bars} common bars (need ≥252)",
                                 reason_category="data")

    # ── Rebalance loop: ranks at bar close → weights from the NEXT bar ────
    # Rows between rebalances stay NaN and forward-fill from the last
    # rebalance row; the rebalance rows themselves are taken LITERALLY
    # (fix 2026-07-12: the old replace(0→NaN).ffill() also forward-filled a
    # dropped name's stale weight over its legitimate 0 — names could enter
    # but never exit, gross leverage crept up every membership change and
    # exit turnover was never costed).
    weights = pd.DataFrame(np.nan, index=close_p.index, columns=close_p.columns)
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
    weights = weights.ffill().fillna(0.0)
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
    engine._log_progress(idea_id, 55, "Gating basket NAV")
    train_df, val_df, test_df = engine._split(port_df)
    one = lambda d: pd.Series(1.0, index=d.index)
    train_r = engine_mod._compute_performance(engine, train_df, one(train_df), interval)
    val_r   = engine_mod._compute_performance(engine, val_df,   one(val_df),   interval)
    test_r  = engine_mod._compute_performance(engine, test_df,  one(test_df),  interval)
    test_sharpe_net   = test_r["sharpe_net"]
    test_sharpe_gross = test_r["sharpe_gross"]
    train_val_gap = abs(train_r["sharpe"] - val_r["sharpe"])
    _ann = BARS_PER_YEAR.get(interval, 252)
    _max_tvg = stats.train_val_gap_tolerance(
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
    ic = cross_sectional_test(engine, row["factor_formula"] or fname, idea_id,
                                   factor={"name": fname, "params": fparams},
                                   interval=interval, persist=False)
    ic_pass = bool(ic.get("factor_is_real"))

    # Benchmark gate — RISK-ADJUSTED (2026-07-10): the basket's net Sharpe
    # must beat holding the universe equal-weight. Raw ann returns are
    # computed for the report only — comparing raw return punished
    # market-neutral books in bull markets (category error).
    _ew = engine._equal_weight_klci_returns(interval)
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
    min_rebals = engine._MIN_TRADES.get(hp_class, 30)
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

    from agents.backtest_engineer.backtest_engineer import _stamp_versions

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
               verdict, verdict_reason,
               mean_ic, ic_tstat, stocks_positive_ic, best_stocks)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
            ic.get("mean_ic"), ic.get("ic_tstat"), ic.get("stocks_positive_ic"),
            json.dumps(ic.get("best_stocks")) if ic.get("best_stocks") is not None else None,
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

    engine._clear_progress(idea_id)
    engine.log_daemon(
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
