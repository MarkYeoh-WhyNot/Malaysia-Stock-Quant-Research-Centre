"""PSR helper pins (gate redesign, 2026-07-10) — closed form sanity + a Monte
Carlo cross-check that the confidence statement is actually calibrated."""
import numpy as np
import pytest

from agents.backtest_engineer.stats import psr, moments, deflated_sr_star, norm_cdf


def test_norm_cdf_basics():
    assert abs(norm_cdf(0.0) - 0.5) < 1e-12
    assert abs(norm_cdf(1.645) - 0.95) < 1e-3
    assert norm_cdf(-8) < 1e-10 and norm_cdf(8) > 1 - 1e-10


def test_psr_monotonic_in_evidence_and_edge():
    # More bars → more confidence for the same observed Sharpe
    p_short = psr(1.4, 0.6, n_obs=252, ann=252)
    p_long = psr(1.4, 0.6, n_obs=1260, ann=252)
    assert p_long > p_short > 0.5
    # Higher observed Sharpe → more confidence
    assert psr(2.0, 0.6, 1260, 252) > psr(1.0, 0.6, 1260, 252)
    # Observed == benchmark → exactly 50%
    assert abs(psr(0.8, 0.8, 1260, 252) - 0.5) < 1e-9


def test_psr_known_case_five_years():
    """SR=1.4 observed over 5yr daily (1260 bars) vs SR*=0.6, Gaussian:
    stat = (1.4-0.6)/√252 · √1259 / √(1+(3-1)/4·(1.4/√252)²) ≈ 1.784
    → PSR ≈ Φ(1.784) ≈ 0.963."""
    p = psr(1.4, 0.6, n_obs=1260, ann=252, skew=0.0, kurt=3.0)
    assert 0.955 < p < 0.972, p


def test_psr_fat_tails_reduce_confidence():
    gaussian = psr(1.4, 0.6, 1260, 252, skew=0.0, kurt=3.0)
    fat = psr(1.4, 0.6, 1260, 252, skew=-1.0, kurt=8.0)
    assert fat < gaussian


def test_psr_monte_carlo_calibration():
    """The calibration claim itself: if true Sharpe == SR*, PSR ≥ 0.95 should
    fire ~5% of the time (false positives ≈ 1 − confidence)."""
    rng = np.random.default_rng(7)
    ann, n = 252, 1260
    true_sr_ann = 0.6
    mu = true_sr_ann / np.sqrt(ann) * 0.01   # per-bar mean at 1% vol
    fires = 0
    trials = 400
    for _ in range(trials):
        r = rng.normal(mu, 0.01, n)
        sr_obs = r.mean() / r.std() * np.sqrt(ann)
        sk, ku = moments(r)
        if psr(sr_obs, true_sr_ann, n, ann, sk, ku) >= 0.95:
            fires += 1
    rate = fires / trials
    assert 0.01 < rate < 0.11, f"false-positive rate {rate} not ≈5%"


def test_deflated_sr_star_scaling():
    # More trials → higher benchmark; more bars → lower
    assert deflated_sr_star(100, 1260, 252) > deflated_sr_star(10, 1260, 252)
    assert deflated_sr_star(50, 2520, 252) < deflated_sr_star(50, 1260, 252)
    # Sanity magnitude: 50 trials on 5yr daily ≈ sqrt(2·ln50/1260)·15.87 ≈ 1.25
    assert 1.1 < deflated_sr_star(50, 1260, 252) < 1.4


def test_moments_degenerate_and_clamps():
    assert moments(np.array([0.01] * 100)) == (0.0, 3.0)   # zero variance
    assert moments(np.array([1.0, 2.0])) == (0.0, 3.0)     # too short
    sk, ku = moments(np.concatenate([np.zeros(500), np.array([50.0])]))
    assert -3.0 <= sk <= 3.0 and 1.0 <= ku <= 30.0          # clamped
