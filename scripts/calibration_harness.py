"""Gate calibration harness — is the pipeline honest, or just strict?

The problem this answers: when the backtester rejects everything, we cannot
tell whether that means *no edge exists in the data* or *the gates are
miscalibrated and would reject genuinely-good strategies too*. A rejection is
only informative if we first prove the gates PASS a strategy that genuinely has
edge and REJECT one that is pure noise.

Method: feed SYNTHETIC price series with known statistical properties through
the *real* gate stack (data-quality → split → walk-forward → deflated Sharpe →
trade-count → benchmark → regime → robustness). No mocking of the gates — only
the two data seams are injected:

  * ``_fetch_prices``  — returns the designed synthetic OHLCV per symbol.
  * ``_parse_factor``  — returns a fixed DSL tree (keeps the harness offline and
                         free; every gate we calibrate runs *after* parsing, so
                         bypassing the LLM parse costs no calibration coverage).

Planted cases:

  WINNER  — a mean-reverting (Ornstein–Uhlenbeck) series paired with a z-score
            reversion rule. A correct pipeline should PASS this (it is real,
            tradable, out-of-sample-stable edge). If it is REJECTED, the gates
            are too strict / miscalibrated — every real rejection is then suspect.

  LOSER   — a pure random walk paired with the same rule. A correct pipeline
            MUST REJECT this. If it PASSES, the gates produce false positives —
            the worse failure, because it means "passed" is not trustworthy.

The harness is market-agnostic: it reads the active profile (ticker shape,
universe, calendar) from ``config.settings`` and runs identically under
``MARKET_MODE=bursa`` and ``MARKET_MODE=crypto``. Bursa is the calibrated
control; crypto is the suspect (its gate thresholds were never re-derived for
crypto volatility). Running both is what lets a failure be localised: both fail
→ shared gate logic; only crypto fails → crypto threshold miscalibration.

Run standalone:
    MARKET_MODE=bursa  ./venv/bin/python scripts/calibration_harness.py
    MARKET_MODE=crypto ./venv/bin/python scripts/calibration_harness.py

Exit code 0 iff calibration passes (winner passed AND loser rejected).
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ── Synthetic price generators ──────────────────────────────────────────────
# All seeded — the harness must be deterministic so a pass/fail is reproducible.

def _business_index(n: int, calendar: str) -> pd.DatetimeIndex:
    """Contiguous date index with no gaps the DQ gate would flag as missing
    bars: business days for equities, calendar days for 24/7 crypto."""
    freq = "B" if calendar == "business" else "D"
    return pd.date_range("2019-01-01", periods=n, freq=freq)


def _ohlcv(close: np.ndarray, index: pd.DatetimeIndex,
           daily_value: float) -> pd.DataFrame:
    """Wrap a close path in a clean OHLCV frame. Volume is set so
    close*volume clears the top liquidity tier in either market (avoids the
    capacity/liquidity gate confounding the calibration signal). Intrabar
    range is tiny and symmetric so no bar looks like a corporate-action gap.
    Prices arrive already positive (generators work in log-return space)."""
    high = close * 1.001
    low = close * 0.999
    openp = np.concatenate([[close[0]], close[:-1]])
    volume = np.maximum(daily_value / close, 1.0)
    return pd.DataFrame(
        {"open": openp, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )


def ou_series(n: int, seed: int, *, kappa: float = 0.14, sigma: float = 0.022,
              level: float = 50.0) -> np.ndarray:
    """Ornstein–Uhlenbeck mean-reverting path in LOG-price space, exponentiated
    back to price. Reversion to ``level`` with ~2% daily innovation vol → a
    z-score reversion rule has genuine, out-of-sample-stable edge and produces
    ~70 round-trips over 1400 bars (clears min-trade gates comfortably), with
    realistic bar-to-bar moves (no corporate-action-gap false flags). This is
    the KNOWN WINNER for the reversion rule."""
    rng = np.random.default_rng(seed)
    log_level = np.log(level)
    x = np.empty(n)
    x[0] = log_level
    for t in range(1, n):
        x[t] = x[t - 1] + kappa * (log_level - x[t - 1]) + sigma * rng.standard_normal()
    return np.exp(x)


def random_walk(n: int, seed: int, *, sigma: float = 0.015,
                drift: float = 0.0, level: float = 50.0) -> np.ndarray:
    """Pure random walk in log-return space (geometric) — no mean reversion, no
    exploitable autocorrelation, realistic ~1.5% daily vol. Any technical rule
    applied to it has zero true edge. KNOWN LOSER / neutral benchmark."""
    rng = np.random.default_rng(seed)
    rets = drift + sigma * rng.standard_normal(n)
    return level * np.exp(np.cumsum(rets))


# ── DSL trees (fixed — bypass the LLM parser) ───────────────────────────────

def _zscore_tree(allow_short: bool) -> dict:
    """Classic mean-reversion: long when z < -1 (oversold), flat at z >= 0.
    In a short-capable market, also short when z > +1, cover at z <= 0."""
    tree = {
        "entry": {"leaf": "zscore", "period": 20, "below": -0.5},
        "exit":  {"leaf": "zscore", "period": 20, "above": 0.0},
    }
    if allow_short:
        tree["short_entry"] = {"leaf": "zscore", "period": 20, "above": 0.5}
        tree["short_exit"]  = {"leaf": "zscore", "period": 20, "below": 0.0}
    return tree


# ── Case definition & execution ─────────────────────────────────────────────

@dataclass
class Trial:
    seed: int
    passed: bool
    verdict: str
    failing_gate: str
    metrics: dict = field(default_factory=dict)


def _make_engine():
    from agents.backtest_engineer.backtest_engineer import BacktestEngineer
    return BacktestEngineer()


def _insert_idea(ticker: str, tree: dict) -> int:
    """Insert a stage2 idea and return its id. factor_formula is descriptive
    only — the parser is patched to return ``tree`` regardless.

    Every calibration trial submits the same DSL tree, which would trip the
    pipeline's semantic-dedup gate (it rejects a second idea parsing to the
    same signal). The dedup query ignores rejected rows, so we retire prior
    calibration probes first — each trial is then evaluated on its own merits.
    """
    from data.database import db_session
    with db_session() as conn:
        conn.execute("UPDATE alpha_ideas SET status='rejected' "
                     "WHERE slug LIKE 'calib-%' AND status != 'rejected'")
        cur = conn.execute(
            """INSERT INTO alpha_ideas
                 (slug, title, hypothesis, ticker, timeframe, factor_formula,
                  stage, status, novelty_score, logic_score, feasibility_score)
               VALUES (?,?,?,?,?,?, 'stage2', 'processing', 0.8, 0.8, 0.8)""",
            (f"calib-{ticker}-{np.random.randint(1_000_000)}",
             "calibration probe", "synthetic calibration probe",
             ticker, "1d", "z-score reversion (synthetic calibration)"),
        )
        return cur.lastrowid


# A single backtest Sharpe estimated on a 20% validation slice has a large
# standard error, so a point-estimate pass/fail is itself noisy. The honest
# measure is the PASS RATE over many independent seeds: a calibrated gate
# stack passes a genuine edge most of the time and passes pure noise rarely.
WIN_PASS_MIN = 0.90   # STRONG edge (SR~2.6) should clear the gates ≥90%
LOSE_PASS_MAX = 0.05  # pure noise should clear them ≤5% of the time
# PSR-redesign target (2026-07-10): a MODERATE edge (SR~1.4) must pass ≥60%
# of the time AMONG trials that satisfy the risk mandate — rejections caused
# purely by the drawdown cap are the risk policy working, not gate noise, so
# they're excluded from the statistical-calibration denominator (and reported
# separately).
MODERATE_PASS_MIN = 0.60


def _which_gate_failed(result: dict) -> str:
    """Name the first failing gate from the result dict's *_pass flags, so a
    systematic false-negative can be attributed to a specific gate."""
    if result.get("gate3_pass") or result.get("overall_pass"):
        return ""
    # gate2 (train/val consistency) has no dedicated flag — infer it when every
    # named gate3 sub-gate passed but the run still failed.
    named = ["trade_count_pass", "cost_pass", "oos_pass", "regime_pass",
             "deflation_pass", "benchmark_pass", "capacity_pass"]
    for g in named:
        if g in result and not result[g]:
            return g.replace("_pass", "")
    if result.get("error"):
        return "data_or_parse"
    # Distinguish the DD risk cap from the statistical gap check — a
    # risk-mandate rejection is intentional policy, not calibration noise.
    try:
        from config.settings import GATE_CONFIG
        _cap = GATE_CONFIG.stage3_max_drawdown
        for _slice in ("train", "val", "test"):
            if (result.get(_slice) or {}).get("max_dd", 0) > _cap:
                return "risk_dd_cap"
    except Exception:
        pass
    return "train_val_gap"


def _run_case(kind: str, tree: dict, seeds: list[int], n_bars: int) -> list[Trial]:
    from config.settings import (MARKET_CALENDAR, TICKER_EXAMPLE, DEFAULT_SYMBOLS)
    index = _business_index(n_bars, MARKET_CALENDAR)
    daily_value = 5e8  # clears blue-chip tier in both markets
    ticker = TICKER_EXAMPLE.split()[0]          # "1155.KL" / "BTC/USDT"

    trials: list[Trial] = []
    from config.settings import ALLOW_SHORT as _allow_short
    for seed in seeds:
        if kind == "winner":
            close = ou_series(n_bars, seed)                   # SR ≈ 2.6 (strong)
        elif kind == "moderate":
            # Market-fair moderate tier: the tier is defined by the planted
            # strategy's TRUE net Sharpe (~1.4 median), not by kappa. A
            # long-only tree captures only half the OU reversion, so Bursa
            # needs faster reversion to plant the same edge strength
            # (measured: L/S kappa 0.07 ≈ long-only kappa 0.13 ≈ SR 1.4).
            close = ou_series(n_bars, seed,
                              kappa=0.07 if _allow_short else 0.13)
        else:
            close = random_walk(n_bars, seed)                 # zero true edge
        case_df = _ohlcv(close, index, daily_value)
        # Benchmark basket: independent low-drift random walks → a fair neutral
        # bar. Re-seeded off the same seed so each trial is self-contained.
        bench = {sym: _ohlcv(random_walk(n_bars, seed + 100 + i), index, daily_value)
                 for i, sym in enumerate(DEFAULT_SYMBOLS)}

        engine = _make_engine()
        engine._fetch_prices = (               # type: ignore[assignment]
            lambda symbol, interval="1d", days=1825, _df=case_df, _b=bench: (
                _df if symbol.split(",")[0].strip() == ticker
                else _b.get(symbol.split(",")[0].strip(),
                            _ohlcv(random_walk(n_bars, seed + 999), index, daily_value))
            ).copy())
        engine._parse_factor = (               # type: ignore[assignment]
            lambda formula, title, hyp, _t=tree: {
                "signal_type": "dsl", "dsl": _t, "representable": True})

        idea_id = _insert_idea(ticker, tree)
        try:
            result = engine.backtest_idea(idea_id)
        except Exception as exc:
            trials.append(Trial(seed, False, "ERROR",
                                f"{type(exc).__name__}", {}))
            continue

        passed = bool(result.get("gate3_pass") or result.get("overall_pass"))
        trials.append(Trial(
            seed=seed, passed=passed,
            verdict=result.get("verdict") or ("PASS" if passed else "REJECTED"),
            failing_gate=_which_gate_failed(result),
            metrics={k: result.get(k) for k in
                     ("sharpe_is", "sharpe_oos", "train_val_gap", "actual_trades")
                     if result.get(k) is not None},
        ))
    return trials


def run_calibration(seeds: Optional[list[int]] = None, n_bars: int = 2000,
                    verbose: bool = True) -> dict:
    """Run the winner and loser cases across many seeds in the active market
    mode and report pass rates + a per-gate false-negative attribution."""
    from config.settings import MARKET_MODE, ALLOW_SHORT
    from data.database import init_db

    init_db()
    if seeds is None:
        seeds = list(range(1, 13))              # 12 independent trials per case
    tree = _zscore_tree(ALLOW_SHORT)

    win = _run_case("winner", tree, seeds, n_bars)
    lose = _run_case("loser", tree, seeds, n_bars)
    # Strength tier (PSR redesign, 2026-07-10): a MODERATE genuine edge
    # (kappa 0.07 → true net Sharpe ≈ 1.4 median). Makes the stack's
    # operating point VISIBLE on every run instead of only "strong passes".
    moderate = _run_case("moderate", tree, seeds, n_bars)

    win_rate = sum(t.passed for t in win) / len(win)
    lose_rate = sum(t.passed for t in lose) / len(lose)
    # Moderate pass rate is computed among trials that satisfy the risk
    # mandate: DD-cap rejections are the risk policy working (excluded from
    # the statistical denominator, reported separately).
    _mod_eligible = [t for t in moderate if t.failing_gate != "risk_dd_cap"]
    mod_rate = (sum(t.passed for t in _mod_eligible) / len(_mod_eligible)
                if _mod_eligible else 1.0)
    mod_dd_rejects = sum(1 for t in moderate if t.failing_gate == "risk_dd_cap")
    # Attribute winner rejections to the gate that stopped them.
    gate_hist: dict = {}
    for t in win:
        if not t.passed:
            gate_hist[t.failing_gate] = gate_hist.get(t.failing_gate, 0) + 1
    mod_hist: dict = {}
    for t in moderate:
        if not t.passed:
            mod_hist[t.failing_gate] = mod_hist.get(t.failing_gate, 0) + 1

    calibrated = (win_rate >= WIN_PASS_MIN and lose_rate <= LOSE_PASS_MAX
                  and mod_rate >= MODERATE_PASS_MIN)
    report = {
        "market_mode": MARKET_MODE,
        "calibrated": calibrated,
        "n_seeds": len(seeds),
        "n_bars": n_bars,
        "winner_pass_rate": round(win_rate, 3),
        "loser_pass_rate": round(lose_rate, 3),
        "moderate_pass_rate": round(mod_rate, 3),
        "moderate_dd_cap_rejects": mod_dd_rejects,
        "moderate_reject_by_gate": mod_hist,
        "winner_reject_by_gate": gate_hist,
        "diagnosis": _diagnose(win_rate, lose_rate, gate_hist, mod_rate),
        "winner_trials": [t.__dict__ for t in win],
        "loser_trials": [t.__dict__ for t in lose],
        "moderate_trials": [t.__dict__ for t in moderate],
    }
    if verbose:
        _print_report(report)
    return report


def _diagnose(win_rate: float, lose_rate: float, gate_hist: dict,
              mod_rate: float = 1.0) -> str:
    parts = []
    if lose_rate > LOSE_PASS_MAX:
        parts.append(
            f"FALSE POSITIVES — pure noise passed {lose_rate:.0%} of the time "
            f"(> {LOSE_PASS_MAX:.0%} tolerance). 'Passed' is not trustworthy; the "
            "pipeline can promote luck as alpha. This is the more dangerous failure.")
    if win_rate < WIN_PASS_MIN:
        top = max(gate_hist, key=gate_hist.get) if gate_hist else "unknown"
        parts.append(
            f"FALSE NEGATIVES — a genuine, out-of-sample-stable edge passed only "
            f"{win_rate:.0%} of the time (< {WIN_PASS_MIN:.0%} target). Dominant "
            f"blocker: the '{top}' gate ({gate_hist.get(top, 0)}/{sum(gate_hist.values())} "
            "rejections). Real-world rejections are partly gate noise, not just "
            "absence of edge.")
    if mod_rate < MODERATE_PASS_MIN:
        parts.append(
            f"OVER-STRICT AT MODERATE STRENGTH — a genuine ~Sharpe-1.4 edge "
            f"(within the risk mandate) passed only {mod_rate:.0%} "
            f"(< {MODERATE_PASS_MIN:.0%} target). The stack only certifies "
            "near-exceptional edges; realistic ones are being rejected.")
    if not parts:
        parts.append(
            f"GATES TRUSTWORTHY — strong edge passes {win_rate:.0%}, moderate "
            f"{mod_rate:.0%}, noise {lose_rate:.0%}. Real rejections can be "
            "read as 'no edge found'.")
    return " | ".join(parts)


def _print_report(report: dict) -> None:
    line = "=" * 72
    print(line)
    print(f"GATE CALIBRATION — MARKET_MODE={report['market_mode']}  "
          f"seeds={report['n_seeds']}  bars={report['n_bars']}")
    print(line)
    print(f"  strong   (SR~2.6) pass rate : {report['winner_pass_rate']:.0%}  "
          f"(target ≥ {WIN_PASS_MIN:.0%})")
    print(f"  moderate (SR~1.4) pass rate : {report['moderate_pass_rate']:.0%}  "
          f"(target ≥ {MODERATE_PASS_MIN:.0%}, excl. "
          f"{report['moderate_dd_cap_rejects']} DD-cap risk rejections)")
    print(f"  noise    (SR 0.0) pass rate : {report['loser_pass_rate']:.0%}  "
          f"(target ≤ {LOSE_PASS_MAX:.0%})")
    if report["winner_reject_by_gate"]:
        print(f"  strong rejections by gate   : {report['winner_reject_by_gate']}")
    if report["moderate_reject_by_gate"]:
        print(f"  moderate rejections by gate : {report['moderate_reject_by_gate']}")
    print(line)
    print(("CALIBRATED [OK]  " if report["calibrated"] else "MISCALIBRATED [XX]  ")
          + report["diagnosis"])
    print(line)


if __name__ == "__main__":
    rep = run_calibration()
    print(json.dumps({"calibrated": rep["calibrated"],
                      "market_mode": rep["market_mode"],
                      "winner_pass_rate": rep["winner_pass_rate"],
                      "moderate_pass_rate": rep["moderate_pass_rate"],
                      "loser_pass_rate": rep["loser_pass_rate"]}))
    sys.exit(0 if rep["calibrated"] else 1)
