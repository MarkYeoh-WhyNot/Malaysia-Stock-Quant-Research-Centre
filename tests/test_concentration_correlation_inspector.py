"""Tests for ConcentrationCorrelationInspector — cross-strategy hidden
concentration/correlation risk (D2).

Two checks, both new logic:
  a. same-symbol overlap across active strategies -> WARNING
  b. pairwise Pearson correlation of daily returns above threshold -> BLOCKER
"""

import numpy as np
import pytest

from config.settings import CORRELATION_ESCALATION_THRESHOLD
from governance.inspectors.concentration_correlation import (
    ConcentrationCorrelationInspector,
)


def _flat_curve(start, n, drift=0.0, seed=None, noise=0.0):
    """Build a synthetic NAV curve with a constant drift + optional noise."""
    rng = np.random.RandomState(seed) if seed is not None else None
    curve = [start]
    for _ in range(n - 1):
        step = drift
        if noise and rng is not None:
            step += rng.normal(0, noise)
        curve.append(curve[-1] * (1 + step))
    return curve


def test_no_positions_no_curves_is_trivial_pass():
    """No active strategies at all -> PASS/INFO."""
    inspector = ConcentrationCorrelationInspector()
    finding = inspector.inspect(scope="portfolio:paper", ctx={})
    assert finding is not None
    assert finding.status == "PASS"
    assert finding.severity == "INFO"


def test_diversified_low_correlation_is_pass():
    """GOOD case: different symbols, near-zero-correlation return streams -> PASS/INFO."""
    inspector = ConcentrationCorrelationInspector()

    positions = [
        {"idea_id": 1, "symbol": "1155.KL"},
        {"idea_id": 2, "symbol": "5347.KL"},
        {"idea_id": 3, "symbol": "6033.KL"},
    ]

    # Independent random-walk-ish return streams (different seeds), long
    # enough that spurious high correlation is very unlikely.
    rng = np.random.RandomState(0)
    returns = {
        1: rng.normal(0.0005, 0.01, 250).tolist(),
        2: np.random.RandomState(1).normal(0.0003, 0.012, 250).tolist(),
        3: np.random.RandomState(2).normal(-0.0002, 0.009, 250).tolist(),
    }

    finding = inspector.inspect(
        scope="portfolio:paper",
        ctx={"positions": positions, "returns": returns},
    )

    assert finding is not None
    assert finding.status == "PASS"
    assert finding.severity == "INFO"
    assert finding.evidence["same_symbol_overlap_count"] == 0
    assert finding.evidence["high_correlation_pairs"] == []
    assert finding.evidence["max_abs_correlation"] < CORRELATION_ESCALATION_THRESHOLD


def test_same_symbol_overlap_triggers_warning():
    """BAD case A: two strategies hold the same underlying ticker -> WARNING."""
    inspector = ConcentrationCorrelationInspector()

    positions = [
        {"idea_id": 10, "symbol": "1155.KL"},
        {"idea_id": 11, "symbol": "1155.KL"},  # same symbol as idea 10
        {"idea_id": 12, "symbol": "5347.KL"},
    ]

    # Uncorrelated returns so ONLY the overlap check fires.
    returns = {
        10: np.random.RandomState(10).normal(0, 0.01, 100).tolist(),
        11: np.random.RandomState(11).normal(0, 0.01, 100).tolist(),
        12: np.random.RandomState(12).normal(0, 0.01, 100).tolist(),
    }

    finding = inspector.inspect(
        scope="portfolio:paper",
        ctx={"positions": positions, "returns": returns},
    )

    assert finding is not None
    assert finding.status == "WARN"
    assert finding.severity == "WARNING"
    assert "1155.KL" in finding.evidence["same_symbol_overlaps"]
    assert sorted(finding.evidence["same_symbol_overlaps"]["1155.KL"]) == [10, 11]
    assert finding.escalate_to == "RiskMonitor"


def test_high_correlation_triggers_blocker():
    """BAD case B: two strategies' daily returns correlate above threshold -> BLOCKER."""
    inspector = ConcentrationCorrelationInspector()

    positions = [
        {"idea_id": 20, "symbol": "1155.KL"},
        {"idea_id": 21, "symbol": "5347.KL"},  # different symbol, but correlated returns
    ]

    rng = np.random.RandomState(42)
    base = rng.normal(0.0004, 0.015, 200)
    # idea 21's returns are base + tiny independent noise -> correlation ~ near 1.0
    noise = np.random.RandomState(43).normal(0, 0.0005, 200)
    returns = {
        20: base.tolist(),
        21: (base + noise).tolist(),
    }

    finding = inspector.inspect(
        scope="portfolio:paper",
        ctx={"positions": positions, "returns": returns},
    )

    assert finding is not None
    assert finding.status == "FAIL"
    assert finding.severity == "BLOCKER"
    assert len(finding.evidence["high_correlation_pairs"]) == 1
    pair = finding.evidence["high_correlation_pairs"][0]
    assert sorted(pair["pair"]) == [20, 21]
    assert abs(pair["correlation"]) > CORRELATION_ESCALATION_THRESHOLD
    assert finding.escalate_to == "PortfolioExecutor"


def test_blocker_wins_over_warning_when_both_present():
    """When both same-symbol overlap AND high correlation are present,
    overall severity must be the worse one (BLOCKER), not WARNING."""
    inspector = ConcentrationCorrelationInspector()

    positions = [
        {"idea_id": 30, "symbol": "1155.KL"},
        {"idea_id": 31, "symbol": "1155.KL"},  # overlap
    ]

    rng = np.random.RandomState(7)
    base = rng.normal(0.0, 0.01, 150)
    returns = {
        30: base.tolist(),
        31: (base * 1.0).tolist(),  # perfectly correlated
    }

    finding = inspector.inspect(
        scope="portfolio:paper",
        ctx={"positions": positions, "returns": returns},
    )

    assert finding is not None
    assert finding.severity == "BLOCKER"
    assert finding.evidence["same_symbol_overlap_count"] == 1
    assert len(finding.evidence["high_correlation_pairs"]) == 1


def test_equity_curves_are_converted_to_returns():
    """Equity curves (NAV series) should be diffed into returns when
    'returns' isn't provided directly, and correlated curves still BLOCKER."""
    inspector = ConcentrationCorrelationInspector()

    curve_a = _flat_curve(100_000.0, 120, drift=0.001, seed=1, noise=0.005)
    # curve_b tracks curve_a's percentage moves almost exactly (scaled NAV base)
    ret_a = np.diff(curve_a) / np.asarray(curve_a[:-1])
    curve_b = [50_000.0]
    for r in ret_a:
        curve_b.append(curve_b[-1] * (1 + r))

    finding = inspector.inspect(
        scope="portfolio:paper",
        ctx={
            "positions": [],
            "equity_curves": {40: curve_a, 41: curve_b},
        },
    )

    assert finding is not None
    assert finding.severity == "BLOCKER"
    assert len(finding.evidence["high_correlation_pairs"]) == 1


def test_custom_correlation_threshold_override():
    """A ctx-provided correlation_threshold should override the config default."""
    inspector = ConcentrationCorrelationInspector()

    rng = np.random.RandomState(5)
    a = rng.normal(0, 0.01, 100)
    # Moderate correlation: mix of a and independent noise.
    b = 0.5 * a + 0.5 * np.random.RandomState(6).normal(0, 0.01, 100)
    returns = {50: a.tolist(), 51: b.tolist()}

    # With the default threshold (0.75) this should likely PASS (moderate corr).
    lenient = inspector.inspect(
        scope="portfolio:paper",
        ctx={"positions": [], "returns": returns},
    )
    # With a very low threshold, the same data must escalate to BLOCKER.
    strict = inspector.inspect(
        scope="portfolio:paper",
        ctx={"positions": [], "returns": returns, "correlation_threshold": 0.05},
    )

    assert strict.severity == "BLOCKER"
    # Sanity: strict finding's threshold reflects the override.
    assert strict.evidence["correlation_threshold"] == 0.05
    assert lenient.evidence["correlation_threshold"] == CORRELATION_ESCALATION_THRESHOLD


def test_finding_can_be_recorded():
    """Findings from this inspector persist via the base record() method."""
    inspector = ConcentrationCorrelationInspector()
    finding = inspector.inspect(scope="portfolio:paper", ctx={})
    row_id = inspector.record(finding)
    assert isinstance(row_id, int)
    assert row_id > 0


def test_flat_return_series_skipped_without_crash():
    """Zero-variance return series (no trades yet) should be skipped, not
    raise (correlation is undefined for a constant series)."""
    inspector = ConcentrationCorrelationInspector()
    returns = {
        60: [0.0] * 50,
        61: [0.0] * 50,
    }
    finding = inspector.inspect(
        scope="portfolio:paper",
        ctx={"positions": [], "returns": returns},
    )
    assert finding is not None
    assert finding.evidence["pairwise_correlations"] == []
    assert finding.status == "PASS"
