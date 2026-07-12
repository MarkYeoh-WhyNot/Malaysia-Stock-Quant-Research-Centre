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

import logging
import math

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Train-val gap tolerance: allow gaps up to this many σ of sampling noise before
# flagging potential overfitting (permits genuine stationary edges on short data).
_TVG_SIGMA_K = 2.0


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


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation via pandas rank (handles ties, no scipy needed)."""
    n = len(x)
    if n < 4:
        return np.nan
    rx = pd.Series(x).rank(method="average").values.astype(float)
    ry = pd.Series(y).rank(method="average").values.astype(float)
    mx, my = rx.mean(), ry.mean()
    num   = np.mean((rx - mx) * (ry - my))
    denom = rx.std(ddof=0) * ry.std(ddof=0)
    return float(num / denom) if denom > 1e-10 else np.nan


def nw_tstat(series: np.ndarray, max_lag: int | None = None) -> float:
    """t-stat of the series mean using Newey-West (Bartlett kernel) standard
    errors. Daily IC observations are autocorrelated, so the iid t-stat
    (mean / (std/√n)) overstates significance; this corrects for it."""
    n = len(series)
    if n < 3:
        return 0.0
    x = series - series.mean()
    if max_lag is None:
        max_lag = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    max_lag = max(0, min(max_lag, n - 1))
    gamma0 = float(np.dot(x, x)) / n
    lrv = gamma0
    for lag in range(1, max_lag + 1):
        w = 1.0 - lag / (max_lag + 1.0)
        lrv += 2.0 * w * (float(np.dot(x[lag:], x[:-lag])) / n)
    if lrv <= 1e-12:
        return 0.0
    return float(series.mean() / np.sqrt(lrv / n))


def sharpe_stderr(sharpe_ann: float, n_bars: int, ann: float) -> float:
    """Standard error of an annualised Sharpe estimate (Lo 2002, IID form):
    se(SR_per_bar) = sqrt((1 + 0.5·SR_per_bar²)/n); annualise by sqrt(ann)."""
    if n_bars < 2 or ann <= 0:
        return float("inf")
    sr_pb = sharpe_ann / np.sqrt(ann)
    return float(np.sqrt((1.0 + 0.5 * sr_pb * sr_pb) / n_bars) * np.sqrt(ann))


def train_val_gap_tolerance(train_sharpe: float, val_sharpe: float,
                            n_train: int, n_val: int, ann: float,
                            floor: float) -> float:
    """Max allowable |train − val| Sharpe before it counts as overfitting.
    Never below ``floor`` (with lots of data a real 0.30 gap still matters),
    but widened to a k-sigma band of the gap's sampling noise for short
    slices — so genuine stationary edges are not rejected on Sharpe noise."""
    se_gap = float(np.hypot(sharpe_stderr(train_sharpe, n_train, ann),
                            sharpe_stderr(val_sharpe, n_val, ann)))
    return max(float(floor), _TVG_SIGMA_K * se_gap)


def robustness_check(engine_instance, test_df: pd.DataFrame, dsl_tree: dict,
                     base_sharpe: float, interval: str, gate_config) -> float:
    """QC7: fraction of ±20% parameter perturbations whose test-split net
    Sharpe stays above robustness_sharpe_ratio × base. Seeded for
    reproducibility; vectorized, no LLM cost.

    Args:
        engine_instance: BacktestEngineer instance, threaded through to the
            engine module's _compute_signals/_compute_performance functions
        test_df: test data
        dsl_tree: DSL parse tree
        base_sharpe: baseline Sharpe to measure robustness against
        interval: bar interval (e.g., "1d")
        gate_config: GATE_CONFIG with robustness_draws, robustness_sharpe_ratio, etc.
    """
    from agents.backtest_engineer import signal_dsl
    from agents.backtest_engineer import engine as engine_mod
    rng = np.random.RandomState(1234)
    ok, valid = 0, 0
    for _ in range(gate_config.robustness_draws):
        perturbed = signal_dsl.perturb_tree(dsl_tree, rng)
        try:
            sig = engine_mod._compute_signals(
                engine_instance, test_df, {"signal_type": "dsl", "dsl": perturbed})
            perf = engine_mod._compute_performance(engine_instance, test_df, sig, interval)
            valid += 1
            if perf["sharpe_net"] > gate_config.robustness_sharpe_ratio * base_sharpe:
                ok += 1
        except Exception as e:
            logger.warning(f"Robustness draw failed: {e}")
    return ok / max(valid, 1)
