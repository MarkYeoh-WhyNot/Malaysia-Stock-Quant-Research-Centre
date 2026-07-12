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
"""
from __future__ import annotations

import json
import logging
from datetime import datetime

import numpy as np

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
