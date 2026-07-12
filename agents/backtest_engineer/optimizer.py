"""Parameter-sweep optimizer — seeded random search over a strategy family.

Given an idea whose factor_formula parses to a DSL tree, sweep ~N configs
(random parameter draws from each leaf's declared range × allowed timeframes ×
instruments), score each on TRAIN+VAL only, then evaluate the single winner on
the TEST slice exactly once. The sweep's config count is fed into the
deflated-Sharpe hurdle of the winner's subsequent gated backtest run (via the
optimizer_runs table), so trying 300 variants honestly raises the bar the
winner must clear — "we searched" is disclosed to the multiple-testing gate,
never hidden from it.

Pure numpy/pandas after one Haiku parse of the base formula; per-config cost is
an in-memory signal evaluation + performance calc (the _robustness_check
pattern), no LLM and no per-config data fetch.

``run_xs_sweep`` is the cross-sectional counterpart (2026-07-12,
docs/funding_carry_sweep_design.md): same honest protocol over an xs FACTOR
spec (period × top_n × bottom_n × rebalance_bars) instead of a DSL tree.
Configs are scored by running the actual long/short basket rebalance loop on
the TRAIN+VAL window only (net of turnover costs and real funding drag) and
ranked by VAL net Sharpe — the metric every swept dimension affects, and the
one deflated-PSR later corrects for. Stricter than run_sweep in one deliberate
way: the sweep NEVER touches the test slice, not even for the winner — the
winner's only test-slice contact is its single subsequent gated
run_cross_sectional_backtest, with this sweep's full n_configs charged to its
deflation hurdle via optimizer_runs.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd

from config.settings import ALLOWED_TIMEFRAMES, KLCI_STOCKS
from data.database import db_session

logger = logging.getLogger(__name__)

# Sweep defaults. SWEEP_TIMEFRAMES excludes 1wk (too few bars to sweep
# meaningfully) and intersects with the profile's allowed set, so on Bursa
# this degrades to ["1d"] automatically.
DEFAULT_N_CONFIGS = 300
SWEEP_TIMEFRAMES = [tf for tf in ("15m", "1h", "4h", "1d") if tf in ALLOWED_TIMEFRAMES]
MAX_INSTRUMENTS = 5
MIN_VAL_TRADES = 10        # a val Sharpe on <10 trades is noise, not a score
TOP_SUMMARY = 20

# Cross-sectional sweep grid (pre-registered in
# docs/funding_carry_sweep_design.md — the factor period range comes from
# factors.FACTORS at draw time, never hardcoded here). bottom_n is forced to 0
# on markets without shorting, matching run_cross_sectional_backtest.
XS_DEFAULT_N_CONFIGS = 200
XS_TOP_N_CHOICES = (2, 3, 4, 5, 6)
XS_BOTTOM_N_CHOICES = (0, 2, 3, 4, 5, 6)
XS_REBALANCE_CHOICES = (3, 5, 7, 10, 14, 21, 28)
XS_MIN_VAL_REBALANCES = 10   # a val Sharpe on <10 rebalances is noise (cf. MIN_VAL_TRADES)


def randomize_tree(tree: dict, rng: np.random.RandomState) -> dict:
    """Copy of the tree with every numeric param drawn UNIFORMLY from its
    leaf's declared range (full-range random search — unlike perturb_tree's
    ±20% local jitter). Structure, leaf choice, one_of key and choice values
    are preserved: the sweep explores the same strategy FAMILY, not arbitrary
    new strategies."""
    from agents.backtest_engineer.signal_dsl import LEAVES

    def _randomize(node):
        if not isinstance(node, dict):
            return node
        out = dict(node)
        leaf = node.get("leaf")
        if leaf in LEAVES:
            spec = LEAVES[leaf]
            all_params = dict(spec.get("params", {}))
            for name, prange in spec.get("one_of", []):
                all_params[name] = prange
            for pname, (ptype, lo, hi) in all_params.items():
                if pname in out:
                    if ptype == "int":
                        out[pname] = int(rng.randint(int(lo), int(hi) + 1))
                    else:
                        out[pname] = round(float(rng.uniform(lo, hi)), 6)
            if leaf in ("sma_cross", "ema_cross", "macd"):
                if float(out.get("fast", 0)) >= float(out.get("slow", 1e9)):
                    out["fast"] = max(2, int(out["slow"]) - 1)
        if "children" in out:
            out["children"] = [_randomize(c) for c in out["children"]]
        if "child" in out:
            out["child"] = _randomize(out["child"])
        return out

    out = {
        "entry": _randomize(tree["entry"]) if tree.get("entry") else None,
        "exit": _randomize(tree["exit"]) if tree.get("exit") else None,
    }
    if tree.get("short_entry"):
        out["short_entry"] = _randomize(tree["short_entry"])
    if tree.get("short_exit"):
        out["short_exit"] = _randomize(tree["short_exit"])
    return out


def _instruments_for(idea_ticker: str) -> list[str]:
    """The sweep's instrument set: the idea's own tickers if given, else the
    first MAX_INSTRUMENTS of the universe (the universe list is authored in
    rough liquidity order — BTC/ETH/BNB/SOL/XRP lead the crypto profile)."""
    from config.settings import TICKER_REGEX
    found = TICKER_REGEX.findall(idea_ticker or "")
    if found:
        seen: set = set()
        return [t for t in found if not (t in seen or seen.add(t))][:MAX_INSTRUMENTS]
    return [s["symbol"] for s in KLCI_STOCKS[:MAX_INSTRUMENTS]]


def generate_configs(base_dsl: dict, idea_ticker: str, seed: int,
                     n_total: int = DEFAULT_N_CONFIGS) -> list[dict]:
    """Deterministic (seeded) list of sweep configs, spread evenly across
    (instrument × timeframe) cells; each config is a full-range random draw of
    the base tree's numeric parameters."""
    rng = np.random.RandomState(seed)
    instruments = _instruments_for(idea_ticker)
    cells = [(inst, tf) for inst in instruments for tf in SWEEP_TIMEFRAMES]
    per_cell = max(1, n_total // len(cells))
    configs = []
    for inst, tf in cells:
        for _ in range(per_cell):
            configs.append({"instrument": inst, "timeframe": tf,
                            "dsl": randomize_tree(base_dsl, rng)})
    return configs


def run_sweep(idea_id: int, seed: int = 42,
              n_configs: int = DEFAULT_N_CONFIGS) -> dict:
    """Execute a full sweep for one idea. Returns the summary dict that the
    daemon persists to optimizer_runs (top configs by VAL score, winner's
    single TEST evaluation, and the honest trial count).

    Selection protocol: configs are scored on train+val only (val net Sharpe,
    subject to a minimum val trade count and a positive train Sharpe). The
    TEST slice is evaluated exactly once, for the chosen winner — never during
    the search.
    """
    from agents.backtest_engineer.backtest_engineer import BacktestEngineer
    from agents.backtest_engineer import engine

    eng = BacktestEngineer()

    with db_session() as conn:
        row = conn.execute("SELECT * FROM alpha_ideas WHERE id=?", (idea_id,)).fetchone()
    if not row:
        return {"error": f"idea {idea_id} not found"}

    base = eng._parse_factor(row["factor_formula"] or "", row["title"],
                             row["hypothesis"] or "")
    if not base.get("representable") or not base.get("dsl"):
        return {"error": f"factor not DSL-representable: {base.get('reason', '?')}"}

    configs = generate_configs(base["dsl"], row["ticker"] or "", seed, n_configs)

    # One fetch + split per (instrument, timeframe) cell; every config in the
    # cell re-uses the in-memory frames.
    from config.settings import FETCH_DAYS_BY_INTERVAL
    frames: dict[tuple, tuple] = {}
    scored = []
    for cfg in configs:
        key = (cfg["instrument"], cfg["timeframe"])
        if key not in frames:
            df = eng._fetch_prices(cfg["instrument"], cfg["timeframe"],
                                   days=FETCH_DAYS_BY_INTERVAL.get(cfg["timeframe"], 1825))
            if df is None or df.empty or len(df) < 252:
                frames[key] = None
                logger.warning(f"sweep[{idea_id}]: insufficient data for {key} "
                               f"({0 if df is None else len(df)} bars)")
            else:
                frames[key] = eng._split(df)
        splits = frames[key]
        if splits is None:
            continue
        train_df, val_df, _test_df = splits
        try:
            params = {"signal_type": "dsl", "dsl": cfg["dsl"]}
            train_perf = engine._compute_performance(
                eng, train_df, engine._compute_signals(eng, train_df, params), cfg["timeframe"])
            val_perf = engine._compute_performance(
                eng, val_df, engine._compute_signals(eng, val_df, params), cfg["timeframe"])
        except Exception as e:
            logger.warning(f"sweep[{idea_id}]: config failed: {e}")
            continue
        eligible = (val_perf["total_trades"] >= MIN_VAL_TRADES
                    and train_perf["sharpe_net"] > 0)
        scored.append({
            "instrument": cfg["instrument"], "timeframe": cfg["timeframe"],
            "dsl": cfg["dsl"],
            "train_sharpe": train_perf["sharpe_net"],
            "val_sharpe": val_perf["sharpe_net"],
            "val_trades": val_perf["total_trades"],
            "val_max_dd": val_perf["max_dd"],
            "eligible": eligible,
        })

    n_evaluated = len(scored)
    eligible = [s for s in scored if s["eligible"]]
    ranked = sorted(eligible, key=lambda s: s["val_sharpe"], reverse=True)

    winner = None
    if ranked:
        best = ranked[0]
        _, _, test_df = frames[(best["instrument"], best["timeframe"])]
        try:
            params = {"signal_type": "dsl", "dsl": best["dsl"]}
            test_perf = engine._compute_performance(
                eng, test_df, engine._compute_signals(eng, test_df, params), best["timeframe"])
            winner = {**best,
                      "test_sharpe": test_perf["sharpe_net"],
                      "test_max_dd": test_perf["max_dd"],
                      "test_trades": test_perf["total_trades"]}
        except Exception as e:
            logger.warning(f"sweep[{idea_id}]: winner test eval failed: {e}")

    def _slim(s):  # summary rows don't need the full tree
        return {k: v for k, v in s.items() if k != "dsl"}

    return {
        "idea_id": idea_id,
        "seed": seed,
        "n_configs": len(configs),
        "n_evaluated": n_evaluated,
        "n_eligible": len(eligible),
        "top": [_slim(s) for s in ranked[:TOP_SUMMARY]],
        "winner": winner,          # includes the winning dsl tree
        "finished_at": datetime.utcnow().isoformat(),
    }


# ── Cross-sectional factor sweep ──────────────────────────────────────────────

def randomize_xs_config(fname: str, rng: np.random.RandomState) -> dict:
    """One random xs config: every factor param drawn uniformly from its
    registry-declared range (the exact counterpart of randomize_tree pulling
    from signal_dsl.LEAVES), basket/rebalance params from the pre-registered
    choice grids. bottom_n is structurally 0 without shorting — same rule as
    run_cross_sectional_backtest."""
    from agents.backtest_engineer.factors import FACTORS
    from config.settings import ALLOW_SHORT

    params: dict = {}
    for pname, (ptype, lo, hi) in FACTORS[fname]["params"].items():
        if ptype == "int":
            params[pname] = int(rng.randint(int(lo), int(hi) + 1))
        else:
            params[pname] = round(float(rng.uniform(lo, hi)), 6)
    return {
        "factor": {"name": fname, "params": params},
        "top_n": int(rng.choice(XS_TOP_N_CHOICES)),
        "bottom_n": int(rng.choice(XS_BOTTOM_N_CHOICES)) if ALLOW_SHORT else 0,
        "rebalance_bars": int(rng.choice(XS_REBALANCE_CHOICES)),
    }


def _fetch_xs_panel(eng, fname: str, interval: str, days: int) -> tuple:
    """SWEEP-ONLY panel fetch: per-name frames (with the factor's funding
    column merged when needed), per-bar settlement-summed funding drag, and
    per-name side cost rates. Mirrors run_cross_sectional_backtest's panel
    loop but is deliberately NOT shared with it — the gated path stays
    byte-stable. Two distinct funding series exist on purpose: the frame's
    ffill'd ``funding_rate`` column is the FACTOR input; ``fundmap`` holds
    per-bar settlement SUMS for PnL drag."""
    from agents.backtest_engineer import factors as factor_registry
    from agents.backtest_engineer.engine import _funding_bar_sum
    from config.settings import DEFAULT_SYMBOLS, FUNDING_INTERVAL_HOURS

    needs_funding = "funding_rate" in factor_registry.required_columns(fname)
    frames: dict[str, pd.DataFrame] = {}
    fundmap: dict[str, pd.Series] = {}
    side_rate: dict[str, float] = {}
    coverage_notes: list[str] = []
    for symbol in DEFAULT_SYMBOLS:
        try:
            df = eng._fetch_prices(symbol, interval, days=days)
            if df is None or df.empty or len(df) < 100:
                coverage_notes.append(
                    f"{symbol}: {0 if df is None or df.empty else len(df)} bars — excluded")
                continue
            if needs_funding and "funding_rate" not in df.columns:
                df["funding_rate"] = eng._fetch_funding_column(symbol, df.index)
            frames[symbol] = df
            if FUNDING_INTERVAL_HOURS:
                fundmap[symbol] = _funding_bar_sum(
                    eng._fetch_funding_history(symbol), df.index)
            _r = eng._cost_rates(df, interval)
            side_rate[symbol] = (_r["buy"] + _r["sell"]) / 2.0
        except Exception as exc:
            coverage_notes.append(f"{symbol}: {exc}")
    return frames, fundmap, side_rate, coverage_notes


def _xs_basket_score(close_p: pd.DataFrame, score_p: pd.DataFrame,
                     fund_p: pd.DataFrame, rate_vec: pd.Series,
                     top_n: int, bottom_n: int, rebalance_bars: int,
                     t: int, ann: float) -> tuple:
    """Net long/short basket returns over the (train+val) panel, mirroring
    run_cross_sectional_backtest's rebalance semantics exactly: ranks at bar
    close, weights effective the NEXT bar, equal-weight legs, per-side costs
    on turnover, funding drag on held weights.

    Returns (train_sharpe, val_sharpe, n_val_rebalances, port_ret) where
    val = rows from ``t`` onward and port_ret is the full net return series
    (returned for tests/reporting — selection uses only the Sharpes)."""
    n_bars = len(close_p)
    ret_p = close_p.pct_change().fillna(0.0)
    # NaN rows between rebalances ffill from the last rebalance row; rebalance
    # rows are literal (same 2026-07-12 exit-ffill fix as the gated engine).
    weights = pd.DataFrame(np.nan, index=close_p.index, columns=close_p.columns)
    current = pd.Series(0.0, index=close_p.columns)
    rebalance_rows: list[int] = []
    for i in range(0, n_bars, rebalance_bars):
        sv = score_p.iloc[i].dropna()
        if len(sv) >= max(5, top_n + bottom_n):
            new_w = pd.Series(0.0, index=close_p.columns)
            ranked = sv.sort_values()
            new_w[ranked.index[-top_n:]] = 1.0 / top_n
            if bottom_n > 0:
                new_w[ranked.index[:bottom_n]] = -1.0 / bottom_n
            if not new_w.equals(current):
                rebalance_rows.append(i)
            current = new_w
        weights.iloc[i] = current
    weights = weights.ffill().fillna(0.0)
    w_held = weights.shift(1).fillna(0.0)   # one-bar execution delay
    gross = (w_held * ret_p).sum(axis=1)
    turnover = (weights - weights.shift(1)).abs().fillna(0.0)
    costs = (turnover * rate_vec).sum(axis=1)
    funding = (w_held * fund_p).sum(axis=1)  # long pays +funding, short receives
    port_ret = gross - costs - funding

    def _sh(r):
        return (float(np.mean(r) / np.std(r) * np.sqrt(ann))
                if len(r) > 20 and np.std(r) > 1e-12 else 0.0)

    train_sharpe = _sh(port_ret.iloc[:t].values)
    val_sharpe = _sh(port_ret.iloc[t:].values)
    n_val_rebalances = sum(1 for i in rebalance_rows if i >= t)
    return train_sharpe, val_sharpe, n_val_rebalances, port_ret


def run_xs_sweep(idea_id: int, seed: int = 42,
                 n_configs: int = XS_DEFAULT_N_CONFIGS) -> dict:
    """Cross-sectional factor sweep for one "xs:"-spec idea. Same honest
    protocol as run_sweep — and stricter on one point: the TEST slice is
    never evaluated here at all. The panel is truncated to train+val before
    any config is scored; the winner's single test-slice exposure is its
    subsequent gated run_cross_sectional_backtest, whose deflated-PSR hurdle
    absorbs this sweep's full n_configs via optimizer_runs.

    Selection: eligible = train net Sharpe > 0 AND ≥ XS_MIN_VAL_REBALANCES
    rebalance events in the val window; ranked by VAL net basket Sharpe
    (the one metric every swept dimension affects). Per-config val mean IC is
    recorded for the report, never used for selection.
    """
    from agents.backtest_engineer.backtest_engineer import BacktestEngineer
    from agents.backtest_engineer import factors as factor_registry
    from agents.backtest_engineer.cross_sectional import _ic_series
    from config.settings import GATE_CONFIG, FETCH_DAYS_BY_INTERVAL
    from data.market_data import BARS_PER_YEAR

    eng = BacktestEngineer()

    with db_session() as conn:
        row = conn.execute("SELECT * FROM alpha_ideas WHERE id=?", (idea_id,)).fetchone()
    if not row:
        return {"error": f"idea {idea_id} not found"}

    _ff = (row["factor_formula"] or "").strip()
    if not _ff.startswith("xs:"):
        return {"error": "not a cross-sectional idea (factor_formula has no xs: spec)"}
    try:
        spec = json.loads(_ff[3:])
        assert spec.get("signal_type") == "cross_sectional"
    except Exception:
        return {"error": "malformed cross-sectional spec (xs: prefix but invalid JSON)"}

    fname = (spec.get("factor") or {}).get("name", "")
    if fname not in factor_registry.FACTORS:
        return {"error": f"unknown factor '{fname}' — available: "
                         f"{sorted(factor_registry.FACTORS)}"}
    interval = spec.get("interval") or row["timeframe"] or "1d"
    days = FETCH_DAYS_BY_INTERVAL.get(interval, 1825)

    frames, fundmap, side_rate, coverage_notes = _fetch_xs_panel(
        eng, fname, interval, days)
    if len(frames) < 5:
        return {"error": f"only {len(frames)} names usable for the sweep panel "
                         f"(need ≥5); coverage: {coverage_notes[:5]}"}

    close_p = pd.DataFrame({s: f["close"] for s, f in frames.items()}).sort_index()
    n_full = len(close_p)
    if n_full < 252:
        return {"error": f"only {n_full} common bars (need ≥252)"}

    # Truncate to train+val BEFORE anything is scored — the test slice is
    # untouched by construction, not by discipline.
    t = int(n_full * GATE_CONFIG.stage3_data_split_train)
    v = int(n_full * GATE_CONFIG.stage3_data_split_val)
    close_tv = close_p.iloc[:t + v]
    cutoff_ts = close_tv.index[-1]
    frames_tv = {s: f[f.index <= cutoff_ts] for s, f in frames.items()}
    fund_p = (pd.DataFrame(fundmap).reindex(close_tv.index).fillna(0.0)
              if fundmap else pd.DataFrame(0.0, index=close_tv.index,
                                           columns=close_tv.columns))
    rate_vec = pd.Series(side_rate).reindex(close_tv.columns).fillna(
        float(np.mean(list(side_rate.values()))) if side_rate else 0.0)
    ann = BARS_PER_YEAR.get(interval, 252)
    fwd_ret_tv = close_tv.pct_change().shift(-1)

    rng = np.random.RandomState(seed)
    configs = [randomize_xs_config(fname, rng) for _ in range(n_configs)]

    # Factor scores depend only on the factor params — cache per param set so
    # repeated draws of the same period don't recompute 20 rolling windows.
    score_cache: dict[tuple, tuple] = {}

    def _score_panel(fparams: dict) -> tuple:
        key = tuple(sorted(fparams.items()))
        if key not in score_cache:
            sp = pd.DataFrame({
                s: factor_registry.compute_factor(fname, f, fparams)
                for s, f in frames_tv.items()}).reindex(close_tv.index)
            ics, _, _ = _ic_series(sp.iloc[t:], fwd_ret_tv.iloc[t:], False)
            val_ic = float(np.mean(ics)) if ics else 0.0
            score_cache[key] = (sp, val_ic)
        return score_cache[key]

    scored = []
    for cfg in configs:
        try:
            score_p, val_mean_ic = _score_panel(cfg["factor"]["params"])
            train_sh, val_sh, n_val_rebals, _ = _xs_basket_score(
                close_tv, score_p, fund_p, rate_vec,
                cfg["top_n"], cfg["bottom_n"], cfg["rebalance_bars"], t, ann)
        except Exception as e:
            logger.warning(f"xs-sweep[{idea_id}]: config failed: {e}")
            continue
        eligible = (train_sh > 0
                    and n_val_rebals >= XS_MIN_VAL_REBALANCES)
        scored.append({
            "factor": cfg["factor"],
            "top_n": cfg["top_n"], "bottom_n": cfg["bottom_n"],
            "rebalance_bars": cfg["rebalance_bars"], "interval": interval,
            "train_sharpe": round(train_sh, 3),
            "val_sharpe": round(val_sh, 3),
            "val_rebalances": n_val_rebals,
            "val_mean_ic": round(val_mean_ic, 4),
            "eligible": eligible,
        })

    n_evaluated = len(scored)
    eligible = [s for s in scored if s["eligible"]]
    ranked = sorted(eligible, key=lambda s: s["val_sharpe"], reverse=True)
    winner = dict(ranked[0]) if ranked else None

    # Report-only disclosures (pre-registered): the selection landscape, so
    # the winner's advantage over the field is visible after the fact.
    def _dist(key):
        vals = sorted(s[key] for s in eligible)
        return ({"max": vals[-1], "median": vals[len(vals) // 2],
                 "min": vals[0]} if vals else None)

    return {
        "idea_id": idea_id,
        "seed": seed,
        "run_type": "cross_sectional",
        "factor_name": fname,
        "n_configs": len(configs),
        "n_evaluated": n_evaluated,
        "n_eligible": len(eligible),
        "top": ranked[:TOP_SUMMARY],
        "winner": winner,
        "val_sharpe_dist": _dist("val_sharpe"),
        "val_ic_dist": _dist("val_mean_ic"),
        "coverage_notes": coverage_notes[:10],
        "finished_at": datetime.utcnow().isoformat(),
    }
