"""Alpha hunt campaign — crypto sub-daily focus (Mark-approved scope, 2026-07-10).

Goal: find at least ONE genuine, gate-passing strategy. Two-stage design with
honest multiple-testing accounting:

  Stage A (screen): every (pair × timeframe × config) candidate is evaluated
    in-memory on the TRAIN+VAL slices only — vectorized `signal_from_dsl` +
    `_compute_performance`, no LLM, no DB writes, test slice never touched.
    EVERY screen is counted as a trial.

  Stage B (gate): the top candidates run through the real `backtest_idea()`
    pipeline (all gates: DQ, split, walk-forward, cost, OOS, regime, deflated
    Sharpe, benchmark, capacity). Each finalist carries the FULL Stage-A trial
    count into the deflated-Sharpe hurdle via an `optimizer_runs` row — the
    winner must beat the expected max Sharpe of that many noise trials. A
    finalist that only looked good because we searched hard gets rejected
    here, correctly.

Honesty notes baked in:
  * The deflated hurdle rises with ln(trials) and with bar frequency — an
    hourly strategy must clear a visibly higher annualized-Sharpe bar than a
    daily one, because hourly noise produces bigger annualized Sharpes. 15m is
    excluded by default (--include-15m to add it): with only ~400 days of 15m
    history the honest hurdle is ~3.5+ net Sharpe, which borders on
    unpassable, and constant-slippage costs at 15m turnover are understated.
  * Zero passes is a valid outcome and is reported as such.

Run (crypto mode only):
  MARKET_MODE=crypto PYTHONPATH=. ./venv/bin/python scripts/alpha_hunt.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd


# ── Curated candidate configs ────────────────────────────────────────────────
# Classic, pre-registered formulations — NOT tuned to the data. Each entry:
# (name, family, tree_builder(short: bool) -> dict)

def _t_sma(fast, slow, short):
    t = {"entry": {"leaf": "sma_cross", "fast": fast, "slow": slow, "direction": "above"},
         "exit":  {"leaf": "sma_cross", "fast": fast, "slow": slow, "direction": "below"}}
    if short:
        t["short_entry"] = {"leaf": "sma_cross", "fast": fast, "slow": slow, "direction": "below"}
        t["short_exit"]  = {"leaf": "sma_cross", "fast": fast, "slow": slow, "direction": "above"}
    return t


def _t_ema(fast, slow, short):
    t = {"entry": {"leaf": "ema_cross", "fast": fast, "slow": slow, "direction": "above"},
         "exit":  {"leaf": "ema_cross", "fast": fast, "slow": slow, "direction": "below"}}
    if short:
        t["short_entry"] = {"leaf": "ema_cross", "fast": fast, "slow": slow, "direction": "below"}
        t["short_exit"]  = {"leaf": "ema_cross", "fast": fast, "slow": slow, "direction": "above"}
    return t


def _t_rsi(period, lo, hi, short):
    t = {"entry": {"leaf": "rsi", "period": period, "below": lo},
         "exit":  {"leaf": "rsi", "period": period, "above": hi}}
    if short:
        t["short_entry"] = {"leaf": "rsi", "period": period, "above": hi}
        t["short_exit"]  = {"leaf": "rsi", "period": period, "below": 50}
    return t


def _t_boll(period, std, short):
    t = {"entry": {"leaf": "bollinger", "period": period, "std": std, "band": "below_lower"},
         "exit":  {"leaf": "bollinger", "period": period, "std": std, "band": "above_upper"}}
    if short:
        t["short_entry"] = {"leaf": "bollinger", "period": period, "std": std, "band": "above_upper"}
        t["short_exit"]  = {"leaf": "bollinger", "period": period, "std": std, "band": "below_lower"}
    return t


def _t_macd(fast, slow, sig, short):
    t = {"entry": {"leaf": "macd", "fast": fast, "slow": slow, "signal": sig, "condition": "bullish"},
         "exit":  {"leaf": "macd", "fast": fast, "slow": slow, "signal": sig, "condition": "bearish"}}
    if short:
        t["short_entry"] = {"leaf": "macd", "fast": fast, "slow": slow, "signal": sig, "condition": "bearish"}
        t["short_exit"]  = {"leaf": "macd", "fast": fast, "slow": slow, "signal": sig, "condition": "bullish"}
    return t


def _t_mom(period, min_ret, short):
    # Time-series momentum: enter after strong trailing return, exit on pullback.
    t = {"entry": {"leaf": "momentum", "period": period, "min_return": min_ret},
         "exit":  {"leaf": "reversal", "period": 5, "max_return": -0.02}}
    return t  # no natural short mirror without a negative-momentum leaf combo


def _t_zscore(period, thr, short):
    t = {"entry": {"leaf": "zscore", "period": period, "below": -thr},
         "exit":  {"leaf": "zscore", "period": period, "above": 0.0}}
    if short:
        t["short_entry"] = {"leaf": "zscore", "period": period, "above": thr}
        t["short_exit"]  = {"leaf": "zscore", "period": period, "below": 0.0}
    return t


def _t_volratio(period, ratio, short):
    # Volume-confirmation breakout: volume surge + positive short momentum.
    return {"entry": {"op": "AND", "children": [
                {"leaf": "volume_ratio", "period": period, "min_ratio": ratio},
                {"leaf": "momentum", "period": 5, "min_return": 0.01}]},
            "exit": {"leaf": "reversal", "period": 3, "max_return": -0.02}}


def _t_rank(short):
    # 6-month formation rolling-rank momentum (skip 10 bars), exit mid-rank.
    return {"entry": {"leaf": "rolling_rank", "formation": 126, "skip": 10,
                      "window": 252, "min_pct": 0.8},
            "exit":  {"leaf": "rolling_rank", "formation": 126, "skip": 10,
                      "window": 252, "max_pct": 0.5}}


def _t_fund_level(thr, short):
    # Funding carry: hold the side the crowd pays. Long when shorts pay
    # (funding meaningfully negative per 8h), exit once longs start paying.
    t = {"entry": {"leaf": "funding_level", "below": -thr},
         "exit":  {"leaf": "funding_level", "above": 0.0}}
    if short:
        t["short_entry"] = {"leaf": "funding_level", "above": thr}
        t["short_exit"]  = {"leaf": "funding_level", "below": 0.0}
    return t


def _t_fund_z(period, z, short):
    # Contrarian funding-extreme reversion: fade crowded positioning vs the
    # rate's own rolling history, exit when funding normalizes to its mean.
    t = {"entry": {"leaf": "funding_zscore", "period": period, "below": -z},
         "exit":  {"leaf": "funding_zscore", "period": period, "above": 0.0}}
    if short:
        t["short_entry"] = {"leaf": "funding_zscore", "period": period, "above": z}
        t["short_exit"]  = {"leaf": "funding_zscore", "period": period, "below": 0.0}
    return t


CONFIGS = [
    ("sma_10_30",       "trend",      lambda s: _t_sma(10, 30, s)),
    ("sma_20_50",       "trend",      lambda s: _t_sma(20, 50, s)),
    ("sma_50_200",      "trend",      lambda s: _t_sma(50, 200, s)),
    ("ema_9_21",        "trend",      lambda s: _t_ema(9, 21, s)),
    ("ema_20_50",       "trend",      lambda s: _t_ema(20, 50, s)),
    ("rsi_14_30_70",    "reversion",  lambda s: _t_rsi(14, 30, 70, s)),
    ("rsi_7_20_80",     "reversion",  lambda s: _t_rsi(7, 20, 80, s)),
    ("rsi_21_35_65",    "reversion",  lambda s: _t_rsi(21, 35, 65, s)),
    ("boll_20_2.0",     "reversion",  lambda s: _t_boll(20, 2.0, s)),
    ("boll_50_2.0",     "reversion",  lambda s: _t_boll(50, 2.0, s)),
    ("macd_12_26_9",    "trend",      lambda s: _t_macd(12, 26, 9, s)),
    ("macd_8_17_9",     "trend",      lambda s: _t_macd(8, 17, 9, s)),
    ("mom_20_3pct",     "momentum",   lambda s: _t_mom(20, 0.03, s)),
    ("mom_60_5pct",     "momentum",   lambda s: _t_mom(60, 0.05, s)),
    ("mom_120_10pct",   "momentum",   lambda s: _t_mom(120, 0.10, s)),
    ("zscore_20_2.0",   "reversion",  lambda s: _t_zscore(20, 2.0, s)),
    ("zscore_20_1.5",   "reversion",  lambda s: _t_zscore(20, 1.5, s)),
    ("zscore_60_1.5",   "reversion",  lambda s: _t_zscore(60, 1.5, s)),
    ("volratio_20_2.5", "breakout",   lambda s: _t_volratio(20, 2.5, s)),
    ("rank_126_top20",  "momentum",   lambda s: _t_rank(s)),
    # Funding carry — the one direction the 2026-07 campaign confirmed
    # (finding-campaign-funding-carry-2026-07-ic-real). Perp-only data;
    # pairs without funding history screen 0 trades and drop out honestly.
    ("fund_lvl_1bp",    "carry",      lambda s: _t_fund_level(0.0001, s)),
    ("fund_lvl_3bp",    "carry",      lambda s: _t_fund_level(0.0003, s)),
    ("fund_z_60_2.0",   "carry",      lambda s: _t_fund_z(60, 2.0, s)),
    ("fund_z_120_1.5",  "carry",      lambda s: _t_fund_z(120, 1.5, s)),
]


def _class_min_trades(interval: str) -> int:
    """Full-window minimum trade count by holding-period class (mirrors
    BacktestEngineer: sub-daily bars → SUBDAILY=100; our daily configs carry
    no intraday keywords → MEDIUM_TERM=30)."""
    return 100 if interval in ("15m", "1h", "4h") else 30


def stage_a(pairs, timeframes, include_short=True, verbose=True):
    """Screen every candidate on TRAIN+VAL only. Returns (records, n_trials)."""
    from agents.backtest_engineer.backtest_engineer import BacktestEngineer
    from agents.backtest_engineer import engine
    from agents.backtest_engineer.signal_dsl import signal_from_dsl
    from agents.data_engineer.data_engineer import DataEngineer
    from config.settings import FETCH_DAYS_BY_INTERVAL

    eng = BacktestEngineer()
    de = DataEngineer()
    records = []
    n_trials = 0

    for tf in timeframes:
        days = FETCH_DAYS_BY_INTERVAL.get(tf, 1825)
        for pair in pairs:
            try:
                df = de.fetch_prices(pair, tf, days, use_cache=True)
            except Exception as exc:
                print(f"  [skip] {pair} {tf}: fetch failed {exc}", file=sys.stderr)
                continue
            if df is None or len(df) < 300:
                print(f"  [skip] {pair} {tf}: only {0 if df is None else len(df)} bars",
                      file=sys.stderr)
                continue
            # Carry configs need funding_rate, which the OHLCV fetch doesn't
            # carry — attach once per (pair, tf), mirroring the Stage B seam
            # (engine.py signal path). No history → all-NaN → the leaf never
            # fires → 0 trades → ineligible, the honest outcome.
            if any(fam == "carry" for _n, fam, _b in CONFIGS):
                df = df.copy()
                df["funding_rate"] = eng._fetch_funding_column(pair, df.index)
                if df["funding_rate"].isna().all():
                    print(f"  [note] {pair} {tf}: no funding history — carry "
                          f"configs will screen 0 trades", file=sys.stderr)
            train_df, val_df, _test_df = eng._split(df)

            for name, family, build in CONFIGS:
                for short in ([False, True] if include_short else [False]):
                    tree = build(short)
                    if short and "short_entry" not in tree:
                        continue  # family has no short mirror; skip duplicate
                    n_trials += 1
                    try:
                        sig = signal_from_dsl(df, tree)
                        p_tr = engine._compute_performance(
                            eng, train_df, sig.loc[train_df.index], tf)
                        p_va = engine._compute_performance(
                            eng, val_df, sig.loc[val_df.index], tf)
                    except Exception as exc:
                        records.append({"pair": pair, "tf": tf, "config": name,
                                        "short": short, "error": str(exc)[:80]})
                        continue
                    rec = {
                        "pair": pair, "tf": tf, "config": name, "family": family,
                        "short": short,
                        "train_sharpe_net": round(p_tr["sharpe_net"], 3),
                        "val_sharpe_net":   round(p_va["sharpe_net"], 3),
                        "trades_trainval":  int(p_tr["total_trades"] + p_va["total_trades"]),
                        "train_dd": round(p_tr["max_dd"], 3),
                        "val_dd":   round(p_va["max_dd"], 3),
                    }
                    # Eligibility: positive on both slices, enough trades that
                    # the full window will clear the class minimum (train+val
                    # is 80% of the window → demand 80% of the minimum).
                    rec["eligible"] = (
                        rec["train_sharpe_net"] > 0.0
                        and rec["val_sharpe_net"] > 0.0
                        and rec["trades_trainval"] >= int(0.8 * _class_min_trades(tf))
                    )
                    records.append(rec)
            if verbose:
                done = [r for r in records if r["pair"] == pair and r["tf"] == tf]
                elig = sum(1 for r in done if r.get("eligible"))
                print(f"  screened {pair} {tf}: {len(done)} configs, {elig} eligible",
                      file=sys.stderr)
    return records, n_trials


def stage_b(finalists, n_trials_total):
    """Run the real gated pipeline for each finalist, charging the FULL
    Stage-A trial count to the deflated-Sharpe hurdle via optimizer_runs."""
    from agents.backtest_engineer.backtest_engineer import BacktestEngineer
    from data.database import db_session

    results = []
    for f in finalists:
        eng = BacktestEngineer()
        with db_session() as conn:
            conn.execute("UPDATE alpha_ideas SET status='rejected' "
                         "WHERE slug LIKE 'hunt-%' AND status IN ('processing','pending')")
            cur = conn.execute(
                """INSERT INTO alpha_ideas
                     (slug, title, hypothesis, ticker, timeframe, factor_formula,
                      stage, status, novelty_score, logic_score, feasibility_score)
                   VALUES (?,?,?,?,?,?, 'stage2', 'processing', 0.8, 0.8, 0.8)""",
                (f"hunt-{f['config']}-{f['pair'].replace('/', '')}-{f['tf']}"
                 f"-{'ls' if f['short'] else 'lo'}",
                 f"hunt: {f['config']} {f['pair']} {f['tf']}"
                 f" {'long/short' if f['short'] else 'long-only'}",
                 f"alpha-hunt finalist ({f['family']}); screened val net Sharpe "
                 f"{f['val_sharpe_net']}", f["pair"], f["tf"], f["config"]))
            idea_id = cur.lastrowid
            conn.execute(
                """INSERT INTO optimizer_runs
                     (idea_id, status, seed, n_configs, started_at, finished_at,
                      summary_json, winner_json)
                   VALUES (?, 'done', 0, ?, datetime('now'), datetime('now'), ?, ?)""",
                (idea_id, n_trials_total,
                 json.dumps({"note": "alpha_hunt stage A screen",
                             "total_trials": n_trials_total}),
                 json.dumps({"dsl": f["tree"], "instrument": f["pair"],
                             "timeframe": f["tf"]})))
        r = eng.backtest_idea(idea_id)
        passed = bool(r.get("gate3_pass") or r.get("overall_pass"))
        out = {
            "idea_id": idea_id, **{k: f[k] for k in
                                    ("pair", "tf", "config", "family", "short",
                                     "val_sharpe_net")},
            "passed": passed,
            "test_sharpe_net": r.get("test_sharpe_net"),
            "sharpe_is": r.get("sharpe_is"), "sharpe_oos": r.get("sharpe_oos"),
            "trades_full_window": r.get("actual_trades"),
            "deflated_hurdle": r.get("deflated_hurdle"),
            "n_trials": r.get("n_trials"),
            "train_val_gap": r.get("train_val_gap"),
            "train_val_gap_tol": r.get("train_val_gap_tol"),
            "gates": {k: r.get(k) for k in
                      ("gate2_pass", "gate3_pass", "trade_count_pass", "cost_pass",
                       "oos_pass", "regime_pass", "deflation_pass",
                       "benchmark_pass", "capacity_pass") if k in r},
            "verdict_reason": (r.get("verdict_reason") or r.get("error") or "")[:200],
            "dsl": f["tree"],
        }
        results.append(out)
        tag = "*** PASS ***" if passed else "reject"
        print(f"[{tag}] {f['pair']} {f['tf']} {f['config']} "
              f"({'L/S' if f['short'] else 'LO'}) test_net="
              f"{out['test_sharpe_net']} hurdle={out['deflated_hurdle']} "
              f"trades={out['trades_full_window']}", file=sys.stderr)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-15m", action="store_true",
                    help="add 15m bars (honest deflated hurdle ~3.5+ — near-unpassable)")
    ap.add_argument("--top", type=int, default=8, help="Stage B finalist count")
    ap.add_argument("--pairs", type=str, default="",
                    help="comma-separated subset, default full universe")
    args = ap.parse_args()

    from config.settings import MARKET_MODE, DEFAULT_SYMBOLS
    if MARKET_MODE != "crypto":
        sys.exit("alpha_hunt is crypto-scope (Mark's decision) — set MARKET_MODE=crypto")

    from data.database import init_db
    init_db()

    pairs = ([p.strip() for p in args.pairs.split(",") if p.strip()]
             or list(DEFAULT_SYMBOLS))
    timeframes = (["15m"] if args.include_15m else []) + ["1h", "4h", "1d"]

    t0 = time.time()
    print(f"Stage A: {len(pairs)} pairs × {timeframes} × {len(CONFIGS)} configs "
          f"(+short mirrors), train+val only …", file=sys.stderr)
    records, n_trials = stage_a(pairs, timeframes)
    eligible = [r for r in records if r.get("eligible")]
    eligible.sort(key=lambda r: r["val_sharpe_net"], reverse=True)

    # Dedupe finalists: max 2 per family and 1 per (pair, config) so Stage B
    # spends its slots on genuinely different hypotheses.
    finalists, fam_count, seen = [], {}, set()
    for r in eligible:
        key = (r["pair"], r["config"])
        if key in seen or fam_count.get(r["family"], 0) >= 2:
            continue
        seen.add(key)
        fam_count[r["family"]] = fam_count.get(r["family"], 0) + 1
        name = r["config"]
        build = next(b for n, _f, b in CONFIGS if n == name)
        r["tree"] = build(r["short"])
        finalists.append(r)
        if len(finalists) >= args.top:
            break

    print(f"Stage A done in {time.time()-t0:.0f}s: {n_trials} trials, "
          f"{len(eligible)} eligible, {len(finalists)} finalists → Stage B",
          file=sys.stderr)

    results = stage_b(finalists, n_trials) if finalists else []
    survivors = [r for r in results if r["passed"]]

    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "market_mode": MARKET_MODE,
        "pairs": pairs, "timeframes": timeframes,
        "stage_a_trials": n_trials,
        "stage_a_eligible": len(eligible),
        "stage_b_finalists": len(finalists),
        "survivors": survivors,
        "finalist_results": results,
        "top_screens": eligible[:20],
    }
    out_path = os.path.join(os.environ.get("OPENCLAW_RUNTIME_DIR", "."),
                            "alpha_hunt_report.json")
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=1, default=str)

    # Campaign verdicts belong in the knowledge graph — falsified directions
    # and confirmed signals both — so the generator and red/blue retrieve
    # them next cycle. Never fatal: a graph hiccup must not void the hunt.
    try:
        from knowledge.ingestion.campaign_findings import emit_alpha_hunt_findings
        print(f"KG findings: {emit_alpha_hunt_findings(report)}", file=sys.stderr)
    except Exception as exc:
        print(f"KG findings write failed (non-fatal): {exc}", file=sys.stderr)

    print(json.dumps({"trials": n_trials, "eligible": len(eligible),
                      "finalists": len(finalists),
                      "survivors": len(survivors),
                      "report": out_path}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
