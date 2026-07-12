"""Tests for the MetricConsistencyAuditor governance inspector."""

import pytest
from governance.inspectors.metric_consistency import MetricConsistencyAuditor
from governance.schemas import Finding


class TestMetricConsistencyAuditor:
    """Test suite for MetricConsistencyAuditor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.auditor = MetricConsistencyAuditor()

    def test_auditor_metadata(self):
        """Test that auditor has correct metadata."""
        assert self.auditor.name == "MetricConsistencyAuditor"
        assert self.auditor.level == "L0"

    def test_pass_consistent_metrics(self):
        """Test PASS verdict when metrics are consistent and reasonable."""
        # Simulate a typical backtest with ~500 trading days
        perf = {
            "ann_return": 0.1500,  # 15% arithmetic annual return
            "cagr": 0.1420,        # ~14.2% compounded
            "sharpe_net": 1.2,
            "max_dd": 0.15,
        }
        ctx = {
            "performance": perf,
            "n_obs": 500,
            "leverage_used": 1.0,
        }

        finding = self.auditor.inspect("backtest_run:100", ctx)

        assert finding is not None
        assert finding.status == "PASS"
        assert finding.severity == "INFO"
        assert finding.evidence["ann_return"] == 0.15
        assert finding.evidence["cagr"] == 0.142
        assert finding.evidence["n_obs"] == 500

    def test_pass_zero_returns(self):
        """Test PASS verdict when both metrics are close to zero."""
        perf = {
            "ann_return": 0.0001,
            "cagr": 0.0002,
            "sharpe_net": 0.05,
        }
        ctx = {
            "performance": perf,
            "n_obs": 300,
        }

        finding = self.auditor.inspect("backtest_run:101", ctx)

        assert finding is not None
        assert finding.status == "PASS"
        assert finding.severity == "INFO"

    def test_fail_missing_ann_return(self):
        """Test FAIL verdict when ann_return is missing."""
        perf = {
            "cagr": 0.12,
            "sharpe_net": 1.1,
        }
        ctx = {"performance": perf}

        finding = self.auditor.inspect("backtest_run:102", ctx)

        assert finding is not None
        assert finding.status == "FAIL"
        assert finding.severity == "BLOCKER"
        assert finding.evidence["missing_field"] == "ann_return"
        assert finding.escalate_to == "BacktestEngineer"

    def test_fail_missing_cagr(self):
        """Test FAIL verdict when cagr is missing."""
        perf = {
            "ann_return": 0.15,
            "sharpe_net": 1.1,
        }
        ctx = {"performance": perf}

        finding = self.auditor.inspect("backtest_run:103", ctx)

        assert finding is not None
        assert finding.status == "FAIL"
        assert finding.severity == "BLOCKER"
        assert finding.evidence["missing_field"] == "cagr"

    def test_fail_non_finite_ann_return(self):
        """Test FAIL verdict when ann_return is NaN or Inf."""
        perf = {
            "ann_return": float("nan"),
            "cagr": 0.12,
        }
        ctx = {"performance": perf}

        finding = self.auditor.inspect("backtest_run:104", ctx)

        # NaN comparison: float("nan") < 1e6 is False, not (-1e6 < float("nan") < 1e6)
        assert finding is not None
        assert finding.status == "FAIL"
        assert finding.severity == "BLOCKER"

    def test_fail_non_finite_cagr(self):
        """Test FAIL verdict when cagr is infinity."""
        perf = {
            "ann_return": 0.15,
            "cagr": float("inf"),
        }
        ctx = {"performance": perf}

        finding = self.auditor.inspect("backtest_run:105", ctx)

        assert finding is not None
        assert finding.status == "FAIL"
        assert finding.severity == "BLOCKER"
        assert "cagr" in finding.evidence
        assert finding.evidence["issue"] == "cagr is not finite"

    def test_fail_opposite_sign_returns(self):
        """Test FAIL verdict when ann_return and cagr have opposite signs (500+ obs).

        This is the key BAD case: equity curve endpoint is inverted or corrupted.
        """
        perf = {
            "ann_return": 0.15,    # positive arithmetic return
            "cagr": -0.08,         # negative CAGR (equity went down)
            "sharpe_net": -0.5,
        }
        ctx = {
            "performance": perf,
            "n_obs": 600,  # long enough to trigger the sanity check
        }

        finding = self.auditor.inspect("backtest_run:106", ctx)

        assert finding is not None
        assert finding.status == "FAIL"
        assert finding.severity == "BLOCKER"
        assert "Opposite signs" in finding.evidence["issue"]
        assert finding.evidence["ann_return"] == 0.15
        assert finding.evidence["cagr"] == -0.08

    def test_fail_ratio_too_high(self):
        """Test FAIL verdict when CAGR >> ann_return (ratio > 5.0)."""
        perf = {
            "ann_return": 0.02,    # small arithmetic return
            "cagr": 0.15,          # 7.5x larger CAGR (should not happen if consistent)
            "sharpe_net": 0.8,
        }
        ctx = {
            "performance": perf,
            "n_obs": 700,  # long enough backtest
        }

        finding = self.auditor.inspect("backtest_run:107", ctx)

        assert finding is not None
        assert finding.status == "FAIL"
        assert finding.severity == "BLOCKER"
        assert "ratio" in finding.evidence or "inconsistency" in finding.evidence["issue"].lower()

    def test_fail_ratio_too_low(self):
        """Test FAIL verdict when ann_return >> CAGR (ratio < 0.2)."""
        perf = {
            "ann_return": 0.50,    # large arithmetic return
            "cagr": 0.02,          # 25x smaller CAGR (unlikely if consistent)
            "sharpe_net": 1.2,
        }
        ctx = {
            "performance": perf,
            "n_obs": 800,  # long enough backtest
        }

        finding = self.auditor.inspect("backtest_run:108", ctx)

        assert finding is not None
        assert finding.status == "FAIL"
        assert finding.severity == "BLOCKER"
        assert "ratio" in finding.evidence or "inconsistency" in finding.evidence["issue"].lower()

    def test_pass_short_backtest_high_ratio(self):
        """Test PASS verdict for short backtest even with high ratio.

        Ratio checks only apply to n_obs > 500 to avoid false positives on short tests.
        """
        perf = {
            "ann_return": 0.02,    # small return
            "cagr": 0.15,          # high ratio, but in a short backtest this is OK
            "sharpe_net": 0.5,
        }
        ctx = {
            "performance": perf,
            "n_obs": 50,  # only 50 observations
        }

        finding = self.auditor.inspect("backtest_run:109", ctx)

        # Should PASS because n_obs is too small to enforce ratio checks
        assert finding is not None
        assert finding.status == "PASS"

    def test_pass_both_metrics_small_ratio(self):
        """Test PASS verdict when both metrics are small (< 0.01) regardless of ratio."""
        perf = {
            "ann_return": 0.005,   # 0.5%
            "cagr": 0.0001,        # 0.01%
            "sharpe_net": 0.1,
        }
        ctx = {
            "performance": perf,
            "n_obs": 600,
        }

        finding = self.auditor.inspect("backtest_run:110", ctx)

        # Should PASS because both are small (< 0.01) so ratio check skipped
        assert finding is not None
        assert finding.status == "PASS"

    def test_no_finding_when_no_performance_dict(self):
        """Test that inspect returns None when no performance dict in context."""
        ctx = {
            "n_obs": 500,
        }

        finding = self.auditor.inspect("backtest_run:111", ctx)

        assert finding is None

    def test_no_finding_when_performance_not_dict(self):
        """Test that inspect returns None when performance is not a dict."""
        ctx = {
            "performance": "not a dict",
            "n_obs": 500,
        }

        finding = self.auditor.inspect("backtest_run:112", ctx)

        assert finding is None

    def test_finding_structure(self):
        """Test that FAIL finding has required fields."""
        perf = {
            "cagr": 0.12,
            # missing ann_return
        }
        ctx = {"performance": perf}

        finding = self.auditor.inspect("backtest_run:113", ctx)

        assert isinstance(finding, Finding)
        assert finding.agent == "MetricConsistencyAuditor"
        assert finding.level == "L0"
        assert finding.scope == "backtest_run:113"
        assert finding.status in ["PASS", "FAIL"]
        assert finding.severity in ["INFO", "BLOCKER"]

    def test_planted_bad_case_equity_curve_mismatch(self):
        """Planted BAD case: CAGR doesn't match what equity curve would imply.

        Scenario: a performance dict where CAGR reports 60% but the daily returns
        only support ~10% (inconsistent endpoint with extreme ratio > 5x).
        """
        perf = {
            "ann_return": 0.10,    # daily returns average to 10% annual
            "cagr": 0.60,          # but CAGR claims 60% (endpoint is badly inflated)
            "sharpe_net": 1.5,
        }
        ctx = {
            "performance": perf,
            "n_obs": 600,
        }

        finding = self.auditor.inspect("backtest_run:999", ctx)

        # Should FAIL due to ratio mismatch (60% / 10% = 6.0x, exceeds 5.0 threshold)
        # This is our planted bad case
        assert finding is not None
        assert finding.status == "FAIL"
        assert finding.severity == "BLOCKER"
        assert "ratio" in finding.evidence or "inconsistency" in finding.evidence["issue"].lower()

    def test_planted_good_case_consistent_metrics(self):
        """Planted GOOD case: metrics are internally consistent."""
        perf = {
            "ann_return": 0.18,    # 18% arithmetic annual return
            "cagr": 0.17,          # ~17% compounded (ratio ≈ 0.94)
            "sharpe_net": 1.4,
            "max_dd": 0.12,
        }
        ctx = {
            "performance": perf,
            "n_obs": 750,  # long backtest
            "leverage_used": 1.0,
        }

        finding = self.auditor.inspect("backtest_run:888", ctx)

        # Should PASS because metrics are consistent (ratio within 0.2–5.0)
        assert finding is not None
        assert finding.status == "PASS"
        assert finding.severity == "INFO"
        assert finding.evidence["ann_return"] == 0.18
        assert finding.evidence["cagr"] == 0.17
