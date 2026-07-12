"""Concentration/Correlation Inspector — hidden-portfolio-risk auditor.

Two INDEPENDENT active paper strategies can look diversified on paper (own
idea_id, own hypothesis, own gate history) while actually being one bet
wearing two hats:

  a. Same-symbol overlap — two or more strategies hold positions in the
     same underlying ticker/symbol at the same time. Concentration risk
     that position-level sizing (per idea) cannot see, because each idea's
     own risk checks only look at ITS OWN book.

  b. Return correlation — even on different symbols, two strategies whose
     daily NAV/return series move together are not diversifying anything;
     a shared drawdown will hit both at once. Pairwise Pearson correlation
     above CORRELATION_ESCALATION_THRESHOLD is treated as a single hidden
     bet, not two independent ones, and escalates to BLOCKER.

Both checks are genuinely new — no existing inspector or risk_monitor
method computes cross-strategy return correlation, and while
risk_monitor.portfolio_risk_snapshot() tracks same-symbol overlap NOTIONAL
for its own overlap_risk_score heuristic, it does not raise a governance
Finding for it. This inspector is ctx-driven (no direct DB reads inside
inspect()) so it can be exercised deterministically in tests; real callers
assemble ctx from paper_trades / paper_equity.
"""

from typing import Optional, Dict, Any, List, Sequence

import numpy as np

from governance.base import Inspector
from governance.schemas import Finding
from config.settings import CORRELATION_ESCALATION_THRESHOLD


def _symbol_of(position: Dict[str, Any]) -> Optional[str]:
    return position.get("symbol") or position.get("ticker") or position.get("pair")


def _returns_from_curve(curve: Sequence[float]) -> Optional[np.ndarray]:
    """Convert a NAV/equity curve into a simple daily-return series.

    Returns None if the curve is too short or degenerate to diff.
    """
    arr = np.asarray(curve, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return None
    denom = arr[:-1]
    # Guard against div-by-zero on a zeroed-out NAV bar.
    if np.any(denom == 0):
        return None
    return np.diff(arr) / denom


class ConcentrationCorrelationInspector(Inspector):
    """L0 deterministic auditor for cross-strategy concentration/correlation.

    Runs two checks against the set of currently-active sandbox/paper
    strategies and rolls them into a single Finding whose severity is the
    worse of the two:

      a. same-symbol overlap (2+ strategies holding the same ticker at
         once) -> WARNING
      b. pairwise return correlation above threshold -> BLOCKER
    """

    name = "ConcentrationCorrelationInspector"
    level = "L0"

    def inspect(self, scope: str, ctx: Dict[str, Any]) -> Optional[Finding]:
        """Check same-symbol overlap and pairwise return correlation.

        Args:
            scope: Context identifier (e.g. "portfolio:paper", "date:2026-07-12")
            ctx: Dictionary containing:
                - "positions": list of dicts, each with "idea_id" and a
                  ticker key ("symbol"/"ticker"/"pair"), for currently OPEN
                  positions across active strategies.
                - "equity_curves": dict of idea_id -> sequence of NAV values
                  in chronological order (e.g. paper_equity.nav per idea),
                  used to derive daily returns. Ignored for any idea_id
                  present in "returns".
                - "returns": optional dict of idea_id -> sequence of
                  pre-computed daily returns (takes precedence over
                  equity_curves for that idea_id).
                - "correlation_threshold": optional float override of
                  CORRELATION_ESCALATION_THRESHOLD.

        Returns:
            Finding with severity INFO/PASS if diversified, WARNING if
            same-symbol overlap is present, or BLOCKER if any pair of
            active strategies' daily returns correlate above threshold
            (BLOCKER wins if both conditions are present).
        """
        positions: List[Dict[str, Any]] = ctx.get("positions", []) or []
        equity_curves: Dict[Any, Sequence[float]] = ctx.get("equity_curves", {}) or {}
        precomputed_returns: Dict[Any, Sequence[float]] = ctx.get("returns", {}) or {}
        threshold = ctx.get("correlation_threshold", CORRELATION_ESCALATION_THRESHOLD)

        # ── Check A: same-symbol overlap ─────────────────────────────────
        symbol_to_ideas: Dict[str, set] = {}
        for pos in positions:
            sym = _symbol_of(pos)
            idea_id = pos.get("idea_id")
            if sym is None or idea_id is None:
                continue
            symbol_to_ideas.setdefault(sym, set()).add(idea_id)

        overlaps = {
            sym: sorted(ids) for sym, ids in symbol_to_ideas.items() if len(ids) > 1
        }

        # ── Check B: pairwise Pearson correlation of daily returns ──────
        returns_by_idea: Dict[Any, np.ndarray] = {}
        for idea_id, series in precomputed_returns.items():
            arr = np.asarray(series, dtype=float)
            if len(arr) >= 2:
                returns_by_idea[idea_id] = arr
        for idea_id, curve in equity_curves.items():
            if idea_id in returns_by_idea:
                continue  # precomputed returns take precedence
            r = _returns_from_curve(curve)
            if r is not None:
                returns_by_idea[idea_id] = r

        pairwise_correlations = []
        idea_ids = sorted(returns_by_idea.keys(), key=lambda x: str(x))
        for i in range(len(idea_ids)):
            for j in range(i + 1, len(idea_ids)):
                a, b = returns_by_idea[idea_ids[i]], returns_by_idea[idea_ids[j]]
                n = min(len(a), len(b))
                if n < 2:
                    continue
                a_n, b_n = a[:n], b[:n]
                if np.std(a_n) == 0 or np.std(b_n) == 0:
                    continue  # flat series — correlation undefined, skip
                corr = float(np.corrcoef(a_n, b_n)[0, 1])
                pairwise_correlations.append(
                    {
                        "pair": [idea_ids[i], idea_ids[j]],
                        "correlation": round(corr, 4),
                        "n_obs": int(n),
                    }
                )

        high_corr_pairs = [
            p for p in pairwise_correlations if abs(p["correlation"]) > threshold
        ]
        max_abs_corr = (
            max((abs(p["correlation"]) for p in pairwise_correlations), default=None)
        )

        # ── Roll up severity: BLOCKER > WARNING > INFO ───────────────────
        if high_corr_pairs:
            status, severity = "FAIL", "BLOCKER"
        elif overlaps:
            status, severity = "WARN", "WARNING"
        else:
            status, severity = "PASS", "INFO"

        evidence = {
            "same_symbol_overlaps": overlaps,
            "same_symbol_overlap_count": len(overlaps),
            "correlation_threshold": threshold,
            "pairwise_correlations": pairwise_correlations,
            "high_correlation_pairs": high_corr_pairs,
            "max_abs_correlation": (
                round(max_abs_corr, 4) if max_abs_corr is not None else None
            ),
            "active_strategy_count": len(
                set(returns_by_idea.keys())
                | {p.get("idea_id") for p in positions if p.get("idea_id") is not None}
            ),
        }

        if severity == "INFO":
            return Finding(
                agent=self.name,
                level=self.level,
                scope=scope,
                status=status,
                severity=severity,
                evidence=evidence,
                local_recommendation=(
                    "No same-symbol overlap and no pair of active strategies "
                    f"correlates above {threshold:.2f}. Strategies are genuinely diversified."
                ),
            )

        rec_parts = []
        if overlaps:
            overlap_desc = "; ".join(
                f"{sym} held by ideas {ids}" for sym, ids in overlaps.items()
            )
            rec_parts.append(
                f"Same-symbol overlap detected: {overlap_desc}. Size these as ONE "
                "combined position for risk purposes, not N independent ones."
            )
        if high_corr_pairs:
            pairs_desc = "; ".join(
                f"ideas {p['pair']} r={p['correlation']}" for p in high_corr_pairs
            )
            rec_parts.append(
                f"Pairwise return correlation exceeds {threshold:.2f}: {pairs_desc}. "
                "These strategies are not diversifying capital — treat as a single "
                "combined bet and reduce joint sizing before adding further overlap."
            )

        return Finding(
            agent=self.name,
            level=self.level,
            scope=scope,
            status=status,
            severity=severity,
            evidence=evidence,
            local_recommendation=" ".join(rec_parts),
            escalate_to="RiskMonitor" if severity == "WARNING" else "PortfolioExecutor",
        )
