"""Regime Attribution Auditor — validates regime-split returns match the gate.

The keystone invariant: the per-bar net-return series that
`agents.backtest_engineer.engine._compute_regimes` slices into volatility
terciles MUST be the identical series that
`agents.backtest_engineer.engine._compute_performance` uses to compute the
gated Sharpe (sharpe_net / PSR). Both currently route through the single
source of truth `_net_return_series(...)["net"]` (see engine.py, line ~539
and ~401), which includes QC3 per-side costs and, on crypto, WS3
funding/leverage. If regime attribution were ever re-derived independently
(e.g. from GROSS returns, or a net series that omits funding on crypto),
the regime Sharpes shown to a human reviewer would silently diverge from
the number the gate actually passed/failed on — exactly the crypto-funding
gap flagged in B2.

This auditor does not re-run the backtest; it compares two return series
handed to it in ctx (the series `_compute_regimes` split on, and the series
`_compute_performance` gated on) and flags any divergence beyond float
tolerance.
"""

from typing import Optional, Dict, Any
from governance.base import Inspector
from governance.schemas import Finding


class RegimeAttributionAuditor(Inspector):
    """L0 deterministic auditor for regime-split / gate return-series consistency.

    Validates that the per-bar return series behind the regime stress test
    (_compute_regimes) is identical to the per-bar return series behind the
    gated Sharpe (_compute_performance) — including funding on crypto. A
    mismatch (e.g. regime split on gross returns while the gate uses net)
    means the regime attribution shown to a reviewer does not describe the
    series that was actually gated.
    """

    name = "RegimeAttributionAuditor"
    level = "L0"

    def inspect(
        self, scope: str, ctx: Dict[str, Any]
    ) -> Optional[Finding]:
        """Validate regime-split returns match the gated return series.

        Args:
            scope: Context identifier (e.g. "backtest_run:1234")
            ctx: Dictionary containing:
                - "regime_net_returns": per-bar return series (list/array of
                  float) that _compute_regimes sliced into volatility
                  terciles.
                - "gate_net_returns": per-bar return series (list/array of
                  float) that _compute_performance used to compute the
                  gated Sharpe / PSR.
                - "tolerance": optional float tolerance (default 1e-9 —
                  these should be the SAME object/values, not merely
                  close).

        Returns:
            Finding with status PASS if the two series match within
            tolerance, FAIL/BLOCKER if they diverge, or None if either
            series is missing from ctx (check not applicable).
        """
        regime_returns = ctx.get("regime_net_returns")
        gate_returns = ctx.get("gate_net_returns")
        tolerance = ctx.get("tolerance", 1e-9)

        if regime_returns is None or gate_returns is None:
            return None

        len_regime = len(regime_returns)
        len_gate = len(gate_returns)

        if len_regime != len_gate:
            evidence = {
                "regime_series_len": len_regime,
                "gate_series_len": len_gate,
                "reason": "length_mismatch",
            }
            return Finding(
                agent=self.name,
                level=self.level,
                scope=scope,
                status="FAIL",
                severity="BLOCKER",
                evidence=evidence,
                local_recommendation=(
                    f"Regime-split return series (len={len_regime}) and gated "
                    f"return series (len={len_gate}) have different lengths — "
                    f"they cannot be the same series. Regime attribution and the "
                    f"gate decision are being computed from different data."
                ),
                escalate_to="BacktestEngineer",
            )

        diffs = [abs(float(a) - float(b)) for a, b in zip(regime_returns, gate_returns)]
        max_diff = max(diffs) if diffs else 0.0
        n_diverging = sum(1 for d in diffs if d > tolerance)
        consistent = max_diff <= tolerance

        evidence = {
            "n_bars": len_regime,
            "max_abs_diff": round(max_diff, 10),
            "n_bars_diverging": n_diverging,
            "tolerance": tolerance,
            "consistent": consistent,
        }

        if consistent:
            return Finding(
                agent=self.name,
                level=self.level,
                scope=scope,
                status="PASS",
                severity="INFO",
                evidence=evidence,
                local_recommendation=(
                    "Regime-split return series matches the gated return series "
                    "bar-for-bar (same net series, including funding/leverage "
                    "on crypto)."
                ),
            )

        return Finding(
            agent=self.name,
            level=self.level,
            scope=scope,
            status="FAIL",
            severity="BLOCKER",
            evidence=evidence,
            local_recommendation=(
                f"Regime attribution diverges from the gated return series on "
                f"{n_diverging}/{len_regime} bars (max abs diff "
                f"{max_diff:.8f}). The regime stress test is not describing the "
                f"same series the gate passed/failed on — likely GROSS vs NET "
                f"(costs and/or crypto funding omitted from one side). Route "
                f"both through _net_return_series(...)['net'] in "
                f"agents/backtest_engineer/engine.py."
            ),
            escalate_to="BacktestEngineer",
        )
