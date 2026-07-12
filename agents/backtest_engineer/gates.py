"""Gate enforcement: the PSR principal rule plus orthogonal guards.

``evaluate_gates`` is the decision core of a single-name backtest — it takes
the ALREADY-COMPUTED train/val/test performance, walk-forward, and regime
metrics and applies every stage2/3 gate: the deflated-PSR principal rule
(replacing the old fixed per-class Sharpe thresholds), DD caps, the
noise-aware train/val gap tolerance, OOS walk-forward degradation, regime
robustness, parameter-perturbation robustness (QC7), cost-drag sensitivity,
minimum trade-count floors, and the risk-adjusted benchmark gate.

SACRED: no threshold, formula, or pass/fail condition here may be changed
without an explicit human-approval task (see CLAUDE.md — GATE_CONFIG /
GATE_OVERRIDES are human-only). This module only relocates the existing
gate-enforcement code out of `_run_backtest`; the logic is byte-identical.

Liquidity floor and capacity are computed earlier in `_run_backtest`'s
orchestration (a name too illiquid never reaches this function at all); the
already-computed `capacity_pass`/`capacity_note` are threaded through here
only because they fold into `overall_pass` and `verdict_reason`. The IC gate
lives in cross_sectional.py (basket path only, not applicable single-name).

Takes `engine` (a BacktestEngineer instance) to reach its cost-model,
data-fetch, and logging helpers, and calls into `agents.backtest_engineer
.engine` (aliased `engine_mod` to avoid the name collision with the
`engine` instance parameter) for `_compute_signals`/`_compute_performance`
/`_detect_sanity_flags`, and into `stats` for the PSR/deflation/robustness
math.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import GATE_CONFIG, BENCHMARK_SYMBOL
from data.database import db_session
from data.market_data import BARS_PER_YEAR
from agents.backtest_engineer import stats
from agents.backtest_engineer import engine as engine_mod


def recent_trial_count(conn, window_days: int) -> int:
    """Distinct ideas backtested inside the rolling deflation window.

    Synthetic calibration probes (slug 'calib-%', inserted by
    scripts/calibration_harness.py) are excluded: they measure the gate
    stack itself and must not raise the deflated hurdle for real ideas.
    LEFT JOIN keeps orphaned backtest_runs rows counted (this DB
    legitimately carries synthetic rows without an alpha_ideas parent) —
    only a positively identified calib probe is dropped, so the count is
    byte-identical to the old query whenever no harness has run.
    """
    return conn.execute(
        "SELECT COUNT(DISTINCT br.idea_id) AS n FROM backtest_runs br "
        "LEFT JOIN alpha_ideas ai ON ai.id = br.idea_id "
        "WHERE br.created_at >= datetime('now', ?) "
        "AND (ai.slug IS NULL OR ai.slug NOT LIKE 'calib-%')",
        (f"-{int(window_days)} days",),
    ).fetchone()["n"]


def regime_gate_decision(params: dict, regimes_positive: int,
                         regime_sharpes: dict | None) -> tuple[bool, str, bool]:
    """QC5 decision for both candidate types. Returns (pass, note, is_scoped).

    unscoped:      positive in >= 2/3 volatility terciles — byte-identical to
                   the original rule (pinned by tests).
    regime-scoped: (DSL "regime_filter" key) flat outside its declared
                   terciles BY CONSTRUCTION, so 2/3 would be unfailable
                   theatre — instead EVERY declared tercile must be positive.
    """
    _dsl = params.get("dsl")
    _rf = _dsl.get("regime_filter") if isinstance(_dsl, dict) else None
    if _rf:
        active = list(_rf.get("active") or [])
        ok = bool(active) and all(
            (regime_sharpes or {}).get(f"sharpe_{a}", 0.0) > 0 for a in active)
        note = "" if ok else (f"Regime-scoped strategy not positive in every "
                              f"declared tercile {active}")
        return ok, note, True
    ok = regimes_positive >= 2
    note = "" if ok else (f"Strategy only works in {regimes_positive}/3 "
                          f"volatility regimes — not robust enough")
    return ok, note, False


def evaluate_gates(engine, *, idea_id, params, hp_class, interval, df, symbol,
                    train_df, val_df, test_df, train_r, val_r, test_r,
                    sharpe_is, sharpe_oos, oos_deg, regimes_positive,
                    capacity_pass, capacity_note,
                    regime_sharpes: dict | None = None) -> dict:
    """Apply the full stage2/3 gate stack. Returns a dict with every pass/fail
    boolean, diagnostic note, and computed metric the caller needs for DB
    persistence and the final result payload (see the return statement for
    the exact key set — this is a literal, mechanical relocation of the
    original `_run_backtest` gate-enforcement block, so every key name
    matches the local variable it used to be)."""
    train_val_gap = abs(train_r["sharpe"] - val_r["sharpe"])

    # Extract gross / net Sharpe for the test period
    test_sharpe_net   = test_r["sharpe_net"]
    test_sharpe_gross = test_r["sharpe_gross"]

    # Per holding-period-class thresholds
    sharpe_threshold = engine._SHARPE_THRESHOLDS.get(hp_class, GATE_CONFIG.stage3_min_sharpe)
    max_dd_threshold = GATE_CONFIG.stage3_max_drawdown
    min_trades       = engine._MIN_TRADES.get(hp_class, 30)
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
        _fs = engine.FUNDAMENTAL_SCREEN_THRESHOLDS
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
        _max_tvg = engine.FUNDAMENTAL_SCREEN_THRESHOLDS["max_train_val_gap"]
    else:
        _max_tvg = stats.train_val_gap_tolerance(
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
        n_trials = recent_trial_count(conn, int(GATE_CONFIG.deflation_window_days)) + 1
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
        _full_sig2 = engine_mod._compute_signals(engine, df, params, symbol=symbol)
        _full_r = engine_mod._compute_performance(engine, df, _full_sig2, interval)
        full_window_sharpe_net = _full_r["sharpe_net"]
        psr_test = _psr(full_window_sharpe_net, deflated_hurdle,
                        _full_r["n_obs"], _ann_qc6,
                        _full_r["skew"], _full_r["kurt"])
        # Pooled train+val PSR — reported for diagnostics, not gated
        # (gating it would double-charge the same evidence).
        _tv_df = pd.concat([train_df, val_df])
        _tv_sig = engine_mod._compute_signals(engine, _tv_df, params, symbol=symbol)
        _tv_r = engine_mod._compute_performance(engine, _tv_df, _tv_sig, interval)
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
            engine.log_daemon(
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
        engine.log_daemon(
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
        engine.log_daemon("WARN", f"Backtest [{idea_id}] cost gate FAILED: {cost_note}")

    # QC2: OOS degradation gate — relaxed for fundamental screens
    oos_pass = True
    oos_note = ""
    _is_fund_screen = params.get("signal_type") == "fundamental_screen"
    _max_oos_deg    = (engine.FUNDAMENTAL_SCREEN_THRESHOLDS["max_oos_degradation"]
                       if _is_fund_screen else 0.50)
    _min_oos_sharpe = (engine.FUNDAMENTAL_SCREEN_THRESHOLDS["min_oos_sharpe"]
                       if _is_fund_screen else 0.30)
    if sharpe_is > 0 and oos_deg > _max_oos_deg:
        oos_pass = False
        oos_note = (f"OOS Sharpe degradation: IS={sharpe_is:.2f} OOS={sharpe_oos:.2f} "
                    f"deg={oos_deg:.2f} > {_max_oos_deg:.2f} — likely overfitted")
    if sharpe_oos < _min_oos_sharpe:
        oos_pass = False
        oos_note = oos_note or f"OOS Sharpe {sharpe_oos:.2f} < {_min_oos_sharpe:.2f} floor"
    if not oos_pass:
        engine.log_daemon("WARN", f"Backtest [{idea_id}] OOS gate FAILED: {oos_note}")

    # QC5: regime robustness gate — NOT a relaxation for scoped candidates:
    # the composite (flat segments included) still faces PSR/DD/OOS/
    # trade-count on the full history, and the regime choice is charged as
    # >= 6 trials via its optimizer_runs row (WARNed below if missing).
    regime_pass, regime_note, _regime_scoped = regime_gate_decision(
        params, regimes_positive, regime_sharpes)
    if not regime_pass:
        engine.log_daemon("WARN", f"Backtest [{idea_id}] regime gate FAILED: {regime_note}")
    if _regime_scoped:
        with db_session() as conn:
            _dof = conn.execute(
                "SELECT 1 FROM optimizer_runs WHERE idea_id=? AND status='done' "
                "AND n_configs >= 6", (idea_id,)).fetchone()
        if not _dof:
            engine.log_daemon(
                "WARN", f"Backtest [{idea_id}] regime-scoped idea missing its "
                        f">=6-config optimizer_runs DOF charge — the regime "
                        f"choice is an uncounted trial")

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
        robustness_score = stats.robustness_check(
            engine, test_df, params["dsl"], test_sharpe_net, interval, GATE_CONFIG)
        robustness_pass = robustness_score >= GATE_CONFIG.robustness_min_fraction
        if not robustness_pass:
            robustness_note = (
                f"Parameter fragility: only {robustness_score:.0%} of ±20% "
                f"parameter perturbations retain >"
                f"{GATE_CONFIG.robustness_sharpe_ratio:.0%} of base Sharpe "
                f"(need {GATE_CONFIG.robustness_min_fraction:.0%}) — knife-edge fit"
            )
            engine.log_daemon(
                "WARN", f"Backtest [{idea_id}] robustness gate FAILED: {robustness_note}")

    # ── Benchmark: excess performance vs the market index (profile symbol) ─
    strat_ann = float(test_r.get("ann_return", 0.0))
    benchmark_sharpe, excess_ann_return = 0.0, 0.0
    try:
        bench_df = engine._fetch_prices(BENCHMARK_SYMBOL, interval, days=1825)
        if not bench_df.empty:
            bench_ret = bench_df["close"].pct_change().reindex(df.index).dropna()
            if len(bench_ret) > 60 and float(np.std(bench_ret)) > 1e-10:
                benchmark_sharpe = float(
                    np.mean(bench_ret) / np.std(bench_ret) * np.sqrt(_ann_qc6)
                )
                excess_ann_return = float(strat_ann - np.mean(bench_ret) * _ann_qc6)
    except Exception as _bench_exc:
        engine.log_daemon("WARN", f"Backtest [{idea_id}] benchmark fetch failed: {_bench_exc}")

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
        ew_ret = engine._equal_weight_klci_returns(interval).reindex(df.index).dropna()
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
            engine.log_daemon(
                "WARN", f"Backtest [{idea_id}] benchmark gate SKIPPED — "
                        f"insufficient equal-weight data (fail-open, disclosed)")
    except Exception as _ew_exc:
        # Benchmark data unavailable → do not block on it (fail-open, warn).
        engine.log_daemon("WARN", f"Backtest [{idea_id}] equal-weight benchmark failed "
                                f"(fail-open, disclosed): {_ew_exc}")

    # ── Sanity flags (warn but do not auto-reject) ────────────────────────
    sanity_flags = engine_mod._detect_sanity_flags(
        test_sharpe_gross, test_r["max_dd"], test_r["win_rate"], actual_trades, interval,
    )
    for flag in sanity_flags:
        engine.log_daemon("WARN", f"Backtest [{idea_id}] SANITY FLAG: {flag}")

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

    return {
        "train_val_gap": train_val_gap,
        "test_sharpe_net": test_sharpe_net,
        "test_sharpe_gross": test_sharpe_gross,
        "sharpe_threshold": sharpe_threshold,
        "max_dd_threshold": max_dd_threshold,
        "min_trades": min_trades,
        "actual_trades": actual_trades,
        "_max_tvg": _max_tvg,
        "n_trials": n_trials,
        "deflated_hurdle": deflated_hurdle,
        "psr_test": psr_test,
        "psr_trainval": psr_trainval,
        "full_window_sharpe_net": full_window_sharpe_net,
        "gate2_pass": gate2_pass,
        "gate3_pass": gate3_pass,
        "trade_count_pass": trade_count_pass,
        "trade_count_note": trade_count_note,
        "cost_pass": cost_pass,
        "cost_note": cost_note,
        "oos_pass": oos_pass,
        "oos_note": oos_note,
        "regime_pass": regime_pass,
        "regime_note": regime_note,
        "deflation_note": deflation_note,
        "robustness_score": robustness_score,
        "robustness_pass": robustness_pass,
        "robustness_note": robustness_note,
        "benchmark_sharpe": benchmark_sharpe,
        "excess_ann_return": excess_ann_return,
        "equal_weight_sharpe": equal_weight_sharpe,
        "excess_vs_ew_ann_return": excess_vs_ew_ann_return,
        "benchmark_pass": benchmark_pass,
        "benchmark_note": benchmark_note,
        "sanity_flags": sanity_flags,
        "overall_pass": overall_pass,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "_is_fund_screen": _is_fund_screen,
    }
