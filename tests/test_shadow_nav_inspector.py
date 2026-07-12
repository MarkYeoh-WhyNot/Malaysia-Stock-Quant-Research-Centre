"""Tests for ShadowNAVInspector — shadow portfolio NAV accounting.

Tests validate that the inspector correctly detects:
1. BAD: Naive summing of independent sandbox NAVs (double-counting shared capital)
2. GOOD: Correct shared-book NAV accounting
3. Edge cases: no active strategies, no reported NAV, etc.
"""

import pytest
from governance.inspectors.shadow_nav import ShadowNAVInspector
from governance.schemas import Finding
from config.settings import PAPER_CAPITAL_MYR


class TestShadowNAVInspector:
    """Test suite for ShadowNAVInspector."""

    def setup_method(self):
        """Initialize inspector for each test."""
        self.inspector = ShadowNAVInspector()

    def test_inspector_metadata(self):
        """Verify inspector name and level."""
        assert self.inspector.name == "ShadowNAVInspector"
        assert self.inspector.level == "L0"

    def test_no_active_strategies_passes(self):
        """No active strategies should pass trivially."""
        ctx = {
            "active_strategy_count": 0,
            "gross_exposure_myr": 0.0,
        }
        finding = self.inspector.inspect("portfolio_risk_snapshot", ctx)

        assert finding is not None
        assert finding.status == "PASS"
        assert finding.severity == "INFO"
        assert finding.evidence["active_strategy_count"] == 0

    def test_bad_case_naive_sum_detected_blocker(self):
        """BAD CASE: Naive sum of independent sandboxes detected as reported NAV.

        Scenario:
        - 3 active strategies, each with RM100k capital
        - All strategies are neutral (no PnL)
        - Someone reports total NAV = 3 * RM100k = RM300k (naive sum)
        - This is DOUBLE-COUNTING shared capital
        - Expected: BLOCKER severity
        """
        paper_capital = PAPER_CAPITAL_MYR
        num_strategies = 3
        naive_sum = num_strategies * paper_capital  # RM300k if capital is RM100k

        ctx = {
            "reported_total_nav": naive_sum,  # Reported as RM300k
            "gross_exposure_myr": 50_000.0,  # RM50k exposure (neutral positions)
            "active_strategy_count": num_strategies,
            "sandbox_navs": [
                paper_capital,
                paper_capital,
                paper_capital,
            ],  # Three neutral sandboxes
        }

        finding = self.inspector.inspect("bad_nav_accounting", ctx)

        assert finding is not None
        assert finding.status == "FAIL"
        assert finding.severity == "BLOCKER"
        assert "double-count" in finding.local_recommendation.lower()
        assert "naive sum" in finding.local_recommendation.lower()
        assert finding.escalate_to == "RiskMonitor"

        # Verify evidence shows the discrepancy
        evidence = finding.evidence
        assert evidence["reported_total_nav"] == naive_sum
        assert evidence["naive_sum_of_sandboxes"] == naive_sum
        assert (
            evidence["active_strategy_count"] == num_strategies
        )

    def test_good_case_shared_book_model_passes(self):
        """GOOD CASE: Portfolio uses correct shared-book NAV accounting.

        Scenario:
        - 3 active strategies, each with RM100k capital
        - Total deployable capital (shared book): RM100k
        - Reported total NAV ≈ RM100k (correct shared-book model)
        - Expected: PASS
        """
        paper_capital = PAPER_CAPITAL_MYR
        num_strategies = 3
        shared_book_nav = paper_capital  # RM100k (shared capital, not naive sum)
        gross_exposure = 45_000.0  # RM45k exposure (4.5x leverage on shared book)

        ctx = {
            "reported_total_nav": shared_book_nav,  # Correctly reported as RM100k
            "gross_exposure_myr": gross_exposure,
            "active_strategy_count": num_strategies,
            "sandbox_navs": [
                paper_capital,
                paper_capital,
                paper_capital,
            ],  # Three sandboxes, but NAV correctly not summed
        }

        finding = self.inspector.inspect("good_nav_accounting", ctx)

        assert finding is not None
        assert finding.status == "PASS"
        assert finding.severity == "INFO"
        assert "shared-book" in finding.local_recommendation.lower()

        # Verify evidence shows paper_capital_multiplier
        evidence = finding.evidence
        assert evidence["model"] == "shared-book"
        expected_multiplier = gross_exposure / paper_capital
        assert abs(evidence["paper_capital_multiplier"] - expected_multiplier) < 0.01

    def test_good_case_with_small_pnl(self):
        """GOOD CASE: Shared-book model with small PnL variations.

        Scenario:
        - 2 active strategies
        - Shared capital: RM100k
        - Small accumulated PnL: +RM3k (within tolerance)
        - Reported NAV ≈ RM103k
        - Expected: PASS (within tolerance)
        """
        paper_capital = PAPER_CAPITAL_MYR
        small_pnl = 3_000.0  # RM3k profit
        reported_nav = paper_capital + small_pnl

        ctx = {
            "reported_total_nav": reported_nav,
            "gross_exposure_myr": 55_000.0,
            "active_strategy_count": 2,
        }

        finding = self.inspector.inspect("nav_with_pnl", ctx)

        assert finding is not None
        assert finding.status == "PASS"
        assert finding.severity == "INFO"

    def test_no_reported_nav_passes_with_note(self):
        """No reported NAV in context should pass with recommendation to provide it."""
        ctx = {
            "active_strategy_count": 2,
            "gross_exposure_myr": 40_000.0,
            # No reported_total_nav
        }

        finding = self.inspector.inspect("nav_report_incomplete", ctx)

        assert finding is not None
        assert finding.status == "PASS"
        assert finding.severity == "INFO"
        assert (
            "reported_total_nav" in finding.local_recommendation.lower()
        )

    def test_naive_sum_with_large_pnl(self):
        """Naive sum detection should still work with PnL accumulated.

        Scenario:
        - 2 strategies with RM100k capital each
        - Each has accumulated RM5k PnL
        - Naive sum would report: 2 * (100k + 5k) = RM210k
        - Correct shared-book: RM100k + 10k = RM110k (shared PnL)
        """
        paper_capital = PAPER_CAPITAL_MYR
        pnl_per_sandbox = 5_000.0
        num_strategies = 2

        naive_sum_with_pnl = num_strategies * (paper_capital + pnl_per_sandbox)
        correct_nav = paper_capital + (pnl_per_sandbox * num_strategies)

        ctx = {
            "reported_total_nav": naive_sum_with_pnl,  # Reporting naive sum
            "gross_exposure_myr": 55_000.0,
            "active_strategy_count": num_strategies,
            "sandbox_navs": [
                paper_capital + pnl_per_sandbox,
                paper_capital + pnl_per_sandbox,
            ],
        }

        finding = self.inspector.inspect("nav_with_pnl_naive", ctx)

        assert finding is not None
        # Naive sum (210k) significantly differs from shared-book (110k), should warn or fail
        assert finding.status in ["FAIL", "WARN"]
        if finding.status == "FAIL":
            assert finding.severity == "BLOCKER"
            assert "double-count" in finding.local_recommendation.lower()

    def test_correct_nav_with_large_exposure(self):
        """GOOD CASE: High leverage but correct NAV accounting.

        Scenario:
        - 1 active strategy
        - Shared capital: RM100k
        - Gross exposure: RM480k (4.8x leverage)
        - Reported NAV: RM100k (correct)
        - Expected: PASS with 4.8x multiplier noted
        """
        paper_capital = PAPER_CAPITAL_MYR
        gross_exposure = 480_000.0  # 4.8x leverage

        ctx = {
            "reported_total_nav": paper_capital,
            "gross_exposure_myr": gross_exposure,
            "active_strategy_count": 1,
        }

        finding = self.inspector.inspect("high_leverage_correct_nav", ctx)

        assert finding is not None
        assert finding.status == "PASS"
        assert finding.severity == "INFO"
        assert finding.evidence["paper_capital_multiplier"] == 4.8

    def test_edge_case_single_strategy_neutral(self):
        """Single strategy, neutral position, correct NAV."""
        paper_capital = PAPER_CAPITAL_MYR

        ctx = {
            "reported_total_nav": paper_capital,
            "gross_exposure_myr": 0.0,
            "active_strategy_count": 1,
            "sandbox_navs": [paper_capital],
        }

        finding = self.inspector.inspect("single_strategy_flat", ctx)

        assert finding is not None
        # Single sandbox with matching NAV should pass (no naive sum issue for 1 strategy)
        assert finding.status == "PASS"
        assert finding.evidence["paper_capital_multiplier"] == 0.0

    def test_reported_nav_significantly_below_capital(self):
        """Reported NAV well below capital (e.g., due to drawdown)."""
        paper_capital = PAPER_CAPITAL_MYR
        drawdown_nav = paper_capital * 0.75  # RM75k (25% drawdown)

        ctx = {
            "reported_total_nav": drawdown_nav,
            "gross_exposure_myr": 30_000.0,
            "active_strategy_count": 2,
        }

        finding = self.inspector.inspect("drawdown_scenario", ctx)

        assert finding is not None
        assert finding.status == "PASS"  # Still using shared-book model correctly
        assert finding.severity == "INFO"

    def test_blockers_have_escalation_path(self):
        """BLOCKER findings should escalate to RiskMonitor."""
        paper_capital = PAPER_CAPITAL_MYR
        num_strategies = 5
        naive_sum = num_strategies * paper_capital

        ctx = {
            "reported_total_nav": naive_sum,
            "gross_exposure_myr": 200_000.0,
            "active_strategy_count": num_strategies,
            "sandbox_navs": [paper_capital] * num_strategies,
        }

        finding = self.inspector.inspect("portfolio_blocker", ctx)

        assert finding is not None
        assert finding.severity == "BLOCKER"
        assert finding.escalate_to == "RiskMonitor"

    def test_inspector_produces_valid_finding(self):
        """Ensure all fields of Finding are populated in PASS case."""
        ctx = {
            "reported_total_nav": PAPER_CAPITAL_MYR,
            "gross_exposure_myr": 50_000.0,
            "active_strategy_count": 1,
        }

        finding = self.inspector.inspect("test_scope", ctx)

        assert finding is not None
        assert isinstance(finding, Finding)
        assert finding.agent == "ShadowNAVInspector"
        assert finding.level == "L0"
        assert finding.scope == "test_scope"
        assert finding.status in ["PASS", "FAIL"]
        assert finding.severity in ["INFO", "WARNING", "BLOCKER"]
        assert finding.evidence is not None
        assert finding.local_recommendation is not None

    def test_tolerance_handles_float_precision(self):
        """Tolerance should handle floating-point precision issues."""
        paper_capital = PAPER_CAPITAL_MYR
        # Reported NAV with tiny floating-point error
        reported_nav = paper_capital + 0.001  # RM100,000.001

        ctx = {
            "reported_total_nav": reported_nav,
            "gross_exposure_myr": 45_000.0,
            "active_strategy_count": 1,
        }

        finding = self.inspector.inspect("float_precision", ctx)

        assert finding is not None
        assert finding.status == "PASS"  # Should tolerate tiny float error

    def test_multiple_strategies_different_navs(self):
        """Sandbox NAVs can differ (some have losses, some have gains)."""
        paper_capital = PAPER_CAPITAL_MYR

        # Sandboxes with different PnL
        sandbox_navs = [
            paper_capital + 5_000.0,  # +RM5k
            paper_capital - 2_000.0,  # -RM2k
            paper_capital + 1_000.0,  # +RM1k
        ]
        naive_sum = sum(sandbox_navs)  # RM305k (incorrect)
        correct_shared_nav = paper_capital + sum([n - paper_capital for n in sandbox_navs])

        ctx = {
            "reported_total_nav": naive_sum,  # Reporting naive sum
            "gross_exposure_myr": 150_000.0,
            "active_strategy_count": 3,
            "sandbox_navs": sandbox_navs,
        }

        finding = self.inspector.inspect("mixed_pnl_naive", ctx)

        # Even with different PnL, naive sum is a significant deviation from shared-book
        # Should be either FAIL (BLOCKER) or WARN depending on tolerance
        assert finding is not None
        assert finding.status in ["FAIL", "WARN"]
        if finding.status == "FAIL":
            assert finding.severity == "BLOCKER"


class TestShadowNAVInspectorIntegration:
    """Integration tests with database and risk_monitor patterns."""

    def test_matches_risk_monitor_pattern(self):
        """Test context matches the pattern from risk_monitor.portfolio_risk_snapshot()."""
        inspector = ShadowNAVInspector()

        # Simulating context from risk_monitor.portfolio_risk_snapshot()
        paper_capital = PAPER_CAPITAL_MYR
        gross_exposure = 125_000.0
        paper_capital_multiplier = gross_exposure / paper_capital

        ctx = {
            "reported_total_nav": paper_capital,  # Correct shared-book
            "gross_exposure_myr": gross_exposure,
            "active_strategy_count": 4,
            "paper_capital_multiplier": paper_capital_multiplier,
        }

        finding = inspector.inspect("risk_monitor_pattern", ctx)

        assert finding is not None
        assert finding.status == "PASS"
        assert finding.evidence["paper_capital_multiplier"] == paper_capital_multiplier

    def test_inspector_records_to_db_via_base_class(self):
        """Verify the inspector has the record() method from Inspector base."""
        inspector = ShadowNAVInspector()

        # Should have inherited record() method
        assert hasattr(inspector, "record")
        assert callable(inspector.record)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
