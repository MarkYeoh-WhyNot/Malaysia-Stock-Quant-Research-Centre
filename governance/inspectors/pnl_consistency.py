"""PnL Consistency Inspector — validates single-source-of-truth net returns.

The keystone invariant (see agents/backtest_engineer/engine.py's
`_net_return_series` docstring): every consumer of "what did this strategy
earn each bar" — the persisted equity curve, the regime stress test, and the
gated performance metrics — MUST derive from the SAME per-bar net-return
array for a given backtest run/params. Before the PnL engine was
consolidated, these were re-derived independently and could diverge (e.g. one
consumer omitting funding/leverage on crypto, or using different transaction
costs), silently showing a different drawdown curve than the one behind the
gated Sharpe.

This inspector recomputes the canonical net-return series via
`agents.backtest_engineer.engine._net_return_series` and diffs it against
whatever arrays were actually used to build each downstream artifact for the
run. Any numeric divergence beyond float tolerance means a consumer
re-derived PnL independently instead of routing through the single source of
truth.
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from governance.base import Inspector
from governance.schemas import Finding
from agents.backtest_engineer.engine import _net_return_series

# Float-equality tolerance for comparing recomputed vs. reported net-return
# arrays. These are float64 computations over the SAME inputs when the code
# path is correct, so any real divergence (different cost rate, different
# lag, different leverage) is orders of magnitude larger than this.
_ATOL = 1e-9
_RTOL = 1e-9

# ctx["reported"] keys this inspector knows how to check, and the
# human-readable name of the consumer each one represents.
_CONSUMERS = {
    "equity_curve": "persisted equity curve",
    "regime": "regime stress test (_compute_regimes)",
    "gate_metrics": "gated performance metrics (_compute_performance)",
}


def _to_array(value: Any) -> Optional[np.ndarray]:
    """Coerce a reported/canonical net-return value to a float ndarray."""
    if value is None:
        return None
    if isinstance(value, dict):
        # Accept the raw _net_return_series() return dict too.
        value = value.get("net")
        if value is None:
            return None
    if isinstance(value, pd.Series):
        return value.to_numpy(dtype=float)
    return np.asarray(value, dtype=float)


class PnLConsistencyInspector(Inspector):
    """L0 deterministic auditor: one net-return series per backtest run.

    Validates that the equity curve, regime attribution, and gated metrics
    for a given run/params all derive from the same per-bar net-return
    array as `engine._net_return_series`.
    """

    name = "PnLConsistencyInspector"
    level = "L0"

    def inspect(self, scope: str, ctx: Dict[str, Any]) -> Optional[Finding]:
        """Validate that downstream consumers match the canonical net-return series.

        Args:
            scope: Context identifier (e.g. "backtest_run:1234")
            ctx: Dictionary containing:
                - "engine": a BacktestEngineer instance (for `_cost_rates`)
                - "df": OHLCV DataFrame for the run
                - "signals": raw (unshifted) signal Series for the run
                - "interval": bar interval string (default "1d")
                - "lag" / "leverage" / "extra_cost_per_side": optional
                  overrides matching how the run's canonical series was
                  produced (defaults match `_net_return_series` defaults)
                - "params": optional dict, echoed into evidence for
                  traceability (not used in the recompute itself, since the
                  canonical call takes `signals` directly)
                - "reported": dict of consumer_name -> net-return array-like
                  (pd.Series / list / ndarray / `_net_return_series()`
                  result dict) actually used to build that consumer's
                  output for this run. Recognized keys: "equity_curve",
                  "regime", "gate_metrics". Missing keys are skipped.

        Returns:
            None if there isn't enough context to run the check (no engine/
            df/signals, or no reported arrays to compare). Otherwise a
            Finding: BLOCKER if any reported consumer diverges from the
            canonical series, PASS if all checked consumers match.
        """
        engine = ctx.get("engine")
        df = ctx.get("df")
        signals = ctx.get("signals")
        reported = ctx.get("reported") or {}

        if engine is None or df is None or signals is None or not reported:
            return None

        interval = ctx.get("interval", "1d")
        canonical = _net_return_series(
            engine, df, signals, interval,
            leverage=ctx.get("leverage"),
            lag=ctx.get("lag", 1),
            extra_cost_per_side=ctx.get("extra_cost_per_side", 0.0),
        )["net"]
        canon_arr = _to_array(canonical)

        checked: list[str] = []
        mismatches: list[dict] = []
        for key, label in _CONSUMERS.items():
            if key not in reported:
                continue
            rep_arr = _to_array(reported[key])
            if rep_arr is None:
                continue
            checked.append(key)

            if rep_arr.shape != canon_arr.shape:
                mismatches.append({
                    "consumer": key, "label": label,
                    "reason": "shape_mismatch",
                    "reported_shape": rep_arr.shape,
                    "canonical_shape": canon_arr.shape,
                })
                continue

            if not np.allclose(rep_arr, canon_arr, atol=_ATOL, rtol=_RTOL, equal_nan=True):
                max_diff = float(np.nanmax(np.abs(rep_arr - canon_arr)))
                mismatches.append({
                    "consumer": key, "label": label,
                    "reason": "value_mismatch",
                    "max_abs_diff": max_diff,
                })

        if not checked:
            return None

        evidence = {
            "checked_consumers": checked,
            "mismatches": mismatches,
        }
        if "params" in ctx:
            evidence["params"] = ctx["params"]

        if mismatches:
            return Finding(
                agent=self.name,
                level=self.level,
                scope=scope,
                status="FAIL",
                severity="BLOCKER",
                evidence=evidence,
                local_recommendation=(
                    "Consumer(s) "
                    + ", ".join(m["consumer"] for m in mismatches)
                    + " diverge from the canonical _net_return_series output for this "
                    "run — one or more of the equity curve, regime attribution, or "
                    "gated metrics re-derived PnL independently instead of routing "
                    "through agents.backtest_engineer.engine._net_return_series. "
                    "Check for a divergent cost_rate, lag, leverage, or funding "
                    "assumption in the offending consumer."
                ),
                escalate_to="BacktestEngineer",
            )

        return Finding(
            agent=self.name,
            level=self.level,
            scope=scope,
            status="PASS",
            severity="INFO",
            evidence=evidence,
            local_recommendation=(
                f"All checked consumers ({', '.join(checked)}) match the canonical "
                "net-return series."
            ),
        )
