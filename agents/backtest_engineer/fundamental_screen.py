"""Fundamental-screen portfolio backtest.

A proper equal-weight portfolio backtest for fundamental screening
strategies (ROE/PB/PE/DY ranking across a 5+ ticker universe), as opposed
to the single-stock constant-signal path in `_run_backtest`. Builds a
quarterly-rebalanced (Mar/Jun/Sep/Dec) weekly portfolio from the top-33%
(min 3, max 10) composite-scored tickers, then runs it through the same
train/val/test + IS/OOS + regime gate shape as the single-name path, using
the FUNDAMENTAL_SCREEN_THRESHOLDS relaxed thresholds (quarterly rebalance
strategies are not comparable to active daily-signal trading).

Takes `engine` (a BacktestEngineer instance) to reach its data-fetch,
progress-logging, and split helpers, and calls into `agents.backtest_engineer
.engine` (aliased `engine_mod`, avoiding the name collision with the
`engine` instance parameter) for `_compute_performance`.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from config.settings import BENCHMARK_SYMBOL
from data.database import db_session
from agents.backtest_engineer import engine as engine_mod


def run_fundamental_screen_backtest(engine, idea_id: int, row: dict) -> dict:
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
    from agents.backtest_engineer.backtest_engineer import _stamp_versions

    ticker_raw = row.get("ticker", "") or ""
    universe   = [t.strip() for t in ticker_raw.split(",")
                  if t.strip() and ".KL" in t.strip()]

    # ── Universe size gate ────────────────────────────────────────────────
    if len(universe) < 5:
        msg = (f"Universe too small for factor ranking ({len(universe)} stocks). "
               f"Minimum 5 required.")
        engine.log_daemon("WARN", f"FundScreen [{idea_id}] {msg}")
        with db_session() as conn:
            conn.execute(
                "UPDATE alpha_ideas SET status='rejected', rejection_reason=? WHERE id=?",
                (msg, idea_id),
            )
        return {"error": msg, "idea_id": idea_id,
                "gate2_pass": False, "gate3_pass": False}

    engine.log_daemon("INFO",
        f"FundScreen [{idea_id}] universe={len(universe)} tickers, "
        f"first 5: {universe[:5]}")
    engine._log_progress(idea_id, 15,
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
        engine.log_daemon("WARN", f"FundScreen [{idea_id}] {msg}")
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

    engine.log_daemon("INFO",
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

    engine.log_daemon("INFO",
        f"FundScreen [{idea_id}] selected {n_select}/{len(tickers_w_data)}: {selected}")

    engine._log_progress(idea_id, 25,
        f"Fetching weekly prices for {len(selected)} stocks")

    # ── Fetch weekly prices ───────────────────────────────────────────────
    prices: dict[str, pd.Series] = {}
    for ticker in selected:
        try:
            df = engine._fetch_prices(ticker, "1wk", days=1825)
            if not df.empty and len(df) >= 52:
                prices[ticker] = df["close"]
                engine.log_daemon("INFO",
                    f"FundScreen [{idea_id}] {ticker}: {len(df)} weekly bars")
        except Exception as exc:
            engine.log_daemon("WARN",
                f"FundScreen [{idea_id}] price fetch failed for {ticker}: {exc}")

    if len(prices) < 3:
        msg = (f"Price data unavailable for sufficient stocks: "
               f"{len(prices)}/{len(selected)} fetched")
        engine.log_daemon("WARN", f"FundScreen [{idea_id}] {msg}")
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
            engine.log_daemon("INFO",
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

    engine.log_daemon("INFO",
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
    engine._log_progress(idea_id, 50, "Running train/val/test split")

    train_df, val_df, test_df = engine._split(port_df)

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
        results[sname] = engine_mod._compute_performance(engine, sdf, sig, interval)

    # ── IS / OOS walk-forward ─────────────────────────────────────────────
    engine._log_progress(idea_id, 65, "Walk-forward IS/OOS")
    n         = len(port_df)
    split_at  = int(n * 0.70)
    is_perf   = engine_mod._compute_performance(
        engine,
        port_df.iloc[:split_at],
        pd.Series(1.0, index=port_df.index[:split_at]),
        interval,
    ) if split_at >= 52 else {"sharpe_net": 0.0}
    oos_perf  = engine_mod._compute_performance(
        engine,
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
    engine._log_progress(idea_id, 75, "Regime stress test")
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

    _fs              = engine.FUNDAMENTAL_SCREEN_THRESHOLDS
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

    engine._log_progress(idea_id, 90, "Saving results")

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
        engine.log_daemon("INFO",
            f"FundScreen [{idea_id}] saved — overall_pass={overall_pass}")
    except Exception as exc:
        engine.log_daemon("ERROR",
            f"FundScreen [{idea_id}] DB save FAILED: {exc}")
        raise

    # ── Save equity curve to backtest_series ─────────────────────────────
    try:
        oos_start = int(len(nav) * 0.70)
        peak      = nav.expanding().max()
        dd_series = (nav - peak) / peak.clip(lower=1e-9)
        bench_curve = None
        try:
            _bdf = engine._fetch_prices(BENCHMARK_SYMBOL, "1d", days=1825)
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
        engine.log_daemon("INFO",
            f"FundScreen [{idea_id}] saved {len(rows_eq)} equity curve points to backtest_series")
    except Exception as _eq_exc:
        engine.log_daemon("WARN",
            f"FundScreen [{idea_id}] could not save equity series: {_eq_exc}")

    engine._log_progress(idea_id, 100, "Complete")
    engine._clear_progress(idea_id)
    engine.log_daemon(
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
