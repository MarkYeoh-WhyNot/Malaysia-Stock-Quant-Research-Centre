"""Funding Cost Auditor — validates crypto perp funding is genuinely applied.

Crypto perpetual backtests accrue funding via `engine._net_return_series`
(see `agents/backtest_engineer/engine.py`): the lagged position pays/receives
`df["funding_bar_sum"]` (real per-bar settlements) when present, else a
disclosed modeled average (`AVG_FUNDING_RATE_PER_INTERVAL`) scaled by
settlements-per-bar. Bursa has no perp funding
(`FUNDING_INTERVAL_HOURS is None`), so this is a documented no-op there.

The bug class this catches: a call path that computes net returns from a
frame stripped of `funding_bar_sum` (or otherwise bypasses the funding term)
while genuinely nonzero funding data exists for the run. That path silently
overstates net Sharpe by omitting a real cost/income stream. This inspector
recomputes the net-return series WITH funding and compares it against a
funding-disabled baseline; if they are numerically identical despite nonzero
funding data, funding was not actually applied anywhere in the path under
test.

Bursa has no funding mechanism at all, so this check is CRYPTO ONLY — it
returns None (not applicable) whenever `MARKET_MODE != "crypto"`.
"""

from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd

from governance.base import Inspector
from governance.schemas import Finding

# Float-equality tolerance for comparing the WITH-funding and WITHOUT-funding
# net-return arrays. Both are float64 computations over the same price/signal
# inputs when funding is genuinely the only difference, so a real funding
# term produces a divergence many orders of magnitude above this.
_ATOL = 1e-12


@contextmanager
def _funding_disabled():
    """Temporarily zero out the engine module's funding constants.

    Mirrors the Bursa no-op state (`FUNDING_INTERVAL_HOURS=None`,
    `AVG_FUNDING_RATE_PER_INTERVAL=0.0`) so a recompute under this context is
    a true funding-free baseline, regardless of what `funding_bar_sum` column
    the caller's df carries.
    """
    from agents.backtest_engineer import engine as engine_mod

    orig_hours = engine_mod.FUNDING_INTERVAL_HOURS
    orig_rate = engine_mod.AVG_FUNDING_RATE_PER_INTERVAL
    engine_mod.FUNDING_INTERVAL_HOURS = None
    engine_mod.AVG_FUNDING_RATE_PER_INTERVAL = 0.0
    try:
        yield
    finally:
        engine_mod.FUNDING_INTERVAL_HOURS = orig_hours
        engine_mod.AVG_FUNDING_RATE_PER_INTERVAL = orig_rate


def _default_synthetic_case():
    """Build a small synthetic crypto run with genuinely nonzero funding.

    Used when the caller doesn't supply engine/df/signals in ctx — lets this
    inspector run as a standalone regression guard, not only as a per-run
    audit.
    """
    from agents.backtest_engineer.backtest_engineer import BacktestEngineer

    n = 400
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    rng = np.random.default_rng(7)
    close = 100 * np.exp(np.cumsum(0.01 * rng.standard_normal(n)))
    df = pd.DataFrame({
        "open": close, "high": close * 1.001, "low": close * 0.999,
        "close": close, "volume": np.full(n, 1e9),
    }, index=idx)
    df["funding_bar_sum"] = 0.0009  # brutal but deterministic nonzero rate
    signals = pd.Series(1.0, index=idx)  # permanently long — pays funding
    return BacktestEngineer(), df, signals, "1d"


def _to_array(net_return_result: Any) -> np.ndarray:
    net = net_return_result["net"] if isinstance(net_return_result, dict) else net_return_result
    if isinstance(net, pd.Series):
        return net.to_numpy(dtype=float)
    return np.asarray(net, dtype=float)


class FundingCostAuditor(Inspector):
    """L0 deterministic auditor: crypto perp funding must move net returns.

    Recomputes the canonical net-return series (`engine._net_return_series`)
    with funding applied and compares it against a funding-disabled
    recompute of the same inputs. If both are identical despite nonzero
    funding data for the run, the code path under test silently drops
    funding — BLOCKER. If they diverge as expected, PASS.
    """

    name = "FundingCostAuditor"
    level = "L0"

    def inspect(self, scope: str, ctx: Dict[str, Any]) -> Optional[Finding]:
        """Validate that funding genuinely moves the net-return series.

        Args:
            scope: Context identifier (e.g. "backtest_run:1234")
            ctx: Dictionary containing (all optional):
                - "engine": a BacktestEngineer instance
                - "df": OHLCV DataFrame with a `funding_bar_sum` column
                - "signals": raw (unshifted) signal Series
                - "interval": bar interval string (default "1d")
                - "net_return_with_funding_fn": override callable
                  `(engine, df, signals, interval) -> dict` used to produce
                  the "WITH funding" series — defaults to
                  `engine._net_return_series` called directly on `df`. Lets a
                  caller reproduce a specific "skips funding" call path
                  (e.g. one that drops `funding_bar_sum` before computing)
                  under audit.
                If engine/df/signals are omitted, a small synthetic crypto
                run with nonzero funding is used so this inspector also
                works as a standalone regression guard.

        Returns:
            None if MARKET_MODE != "crypto" (Bursa has no funding — not
            applicable). Otherwise a Finding: BLOCKER if the WITH-funding
            and funding-disabled series are numerically identical despite
            nonzero funding data, PASS if they diverge as expected.
        """
        from config.settings import MARKET_MODE
        if MARKET_MODE != "crypto":
            return None  # Bursa has no perp funding — not applicable

        from agents.backtest_engineer import engine as engine_mod

        engine = ctx.get("engine")
        df = ctx.get("df")
        signals = ctx.get("signals")
        interval = ctx.get("interval", "1d")
        if engine is None or df is None or signals is None:
            engine, df, signals, interval = _default_synthetic_case()

        funding_col = df["funding_bar_sum"] if "funding_bar_sum" in df.columns else pd.Series(dtype=float)
        funding_present = bool((funding_col.fillna(0.0) != 0.0).any())

        with_fn: Callable = ctx.get("net_return_with_funding_fn") or engine_mod._net_return_series
        r_with = with_fn(engine, df, signals, interval)

        # Funding-disabled baseline: same inputs, funding term forced off.
        # Strip funding_bar_sum too so a stray fallback rate can't leak in.
        df_stripped = df.drop(columns=["funding_bar_sum"], errors="ignore")
        with _funding_disabled():
            r_without = engine_mod._net_return_series(engine, df_stripped, signals, interval)

        net_with = _to_array(r_with)
        net_without = _to_array(r_without)
        identical = (
            net_with.shape == net_without.shape
            and np.allclose(net_with, net_without, atol=_ATOL, equal_nan=True)
        )

        evidence = {
            "funding_present": funding_present,
            "identical_with_without": identical,
            "n_bars": int(net_with.shape[0]) if net_with.ndim else 0,
            "max_abs_diff": (
                float(np.nanmax(np.abs(net_with - net_without)))
                if net_with.shape == net_without.shape else None
            ),
        }
        if isinstance(r_with, dict) and "funding_drag_pct" in r_with:
            evidence["funding_drag_pct_with"] = r_with["funding_drag_pct"]
        if isinstance(r_without, dict) and "funding_drag_pct" in r_without:
            evidence["funding_drag_pct_without"] = r_without["funding_drag_pct"]

        if funding_present and identical:
            return Finding(
                agent=self.name,
                level=self.level,
                scope=scope,
                status="FAIL",
                severity="BLOCKER",
                evidence=evidence,
                local_recommendation=(
                    "Net returns are numerically IDENTICAL with and without funding "
                    "despite nonzero funding_bar_sum data for this run — the audited "
                    "call path is silently dropping the funding term (real settlements "
                    "or the modeled fallback) from agents.backtest_engineer.engine."
                    "_net_return_series. This overstates net Sharpe on crypto perps by "
                    "omitting a real cost/income stream. Check that the caller attaches "
                    "funding_bar_sum before computing net returns and does not bypass "
                    "_net_return_series."
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
                "Funding genuinely moves net returns for this crypto run "
                "(WITH-funding and funding-disabled series diverge as expected)."
                if funding_present else
                "No nonzero funding data present for this run — nothing to audit."
            ),
        )
