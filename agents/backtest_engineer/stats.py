"""Statistical helpers for gate decisions — pure functions, no I/O.

The principal pass rule (gate redesign, 2026-07-10): instead of fixed Sharpe
thresholds, an idea passes when we are statistically confident its TRUE net
Sharpe exceeds the noise-implied benchmark, given how much evidence backs the
estimate. That is the Probabilistic Sharpe Ratio (Bailey & López de Prado,
2012), evaluated against the Deflated benchmark SR* (expected max Sharpe of N
noise trials on this sample length).

Why this replaces fixed thresholds: a fixed bar (e.g. net ≥ 1.1) ignores
evidence length — it passes a lucky 6-month Sharpe-3 fluke and rejects a
five-year Sharpe-1.3 with tight error bars. PSR scales the requirement with
the sample: more bars/trades → tighter confidence → a moderate true edge can
qualify on sufficient evidence, while a strong-looking short sample cannot.
"""
from __future__ import annotations

import math

import numpy as np


def norm_cdf(x: float) -> float:
    """Standard normal CDF via erf — avoids a scipy dependency."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def moments(returns: np.ndarray) -> tuple[float, float]:
    """(skewness, kurtosis) of a return series. Kurtosis is NON-excess
    (normal = 3.0), the convention the PSR formula expects. Degenerate
    series → normal moments (0, 3) so PSR falls back to the Gaussian case."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 4:
        return 0.0, 3.0
    std = r.std()
    if std < 1e-12:
        return 0.0, 3.0
    z = (r - r.mean()) / std
    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4))
    # Clamp pathological estimates from fat-tailed small samples — beyond
    # these bounds the PSR variance term can go negative/absurd.
    return float(np.clip(skew, -3.0, 3.0)), float(np.clip(kurt, 1.0, 30.0))


def psr(sr_obs_ann: float, sr_star_ann: float, n_obs: int, ann: float,
        skew: float = 0.0, kurt: float = 3.0) -> float:
    """Probabilistic Sharpe Ratio: P(true Sharpe > SR*) given the estimate.

    Args:
      sr_obs_ann:  observed ANNUALIZED net Sharpe of the slice
      sr_star_ann: benchmark ANNUALIZED Sharpe to beat (deflated noise max)
      n_obs:       number of return observations in the slice
      ann:         bars per year (annualization factor used for the Sharpes)
      skew, kurt:  return-series moments (kurt non-excess; normal = 3)

    Closed form (Bailey–López de Prado 2012), computed on PER-BAR Sharpes:
      PSR = Φ( (SR̂ − SR*) · √(n−1) / √(1 − γ₃·SR̂ + (γ₄−1)/4 · SR̂²) )
    """
    if n_obs < 4 or ann <= 0:
        return 0.0
    sr_hat = sr_obs_ann / math.sqrt(ann)     # per-bar
    sr_star = sr_star_ann / math.sqrt(ann)
    var_term = 1.0 - skew * sr_hat + (kurt - 1.0) / 4.0 * sr_hat ** 2
    if var_term <= 1e-12:
        # Degenerate variance estimate — be conservative, not generous.
        return 0.0
    stat = (sr_hat - sr_star) * math.sqrt(n_obs - 1) / math.sqrt(var_term)
    return norm_cdf(stat)


def deflated_sr_star(n_trials: int, n_obs: int, ann: float) -> float:
    """Annualized expected max Sharpe of `n_trials` iid noise strategies on a
    sample of `n_obs` bars — the benchmark a real edge must beat (E[max SR of
    N trials] ≈ √(2·ln N / T), annualized). Same formula the deflation gate
    used; now it feeds PSR as SR* instead of acting as a separate binary."""
    if n_obs <= 0:
        return float("inf")
    return float(math.sqrt(2.0 * math.log(max(n_trials, 2)) / n_obs)
                 * math.sqrt(ann))
