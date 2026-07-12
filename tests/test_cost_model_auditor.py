"""Test the CostModelAuditor governance inspector.

Validates that the auditor correctly catches cost-rate mismatches,
particularly the critical bug: sub-daily backtests silently using
daily cost defaults.
"""

import pytest
from governance.inspectors.cost_model import CostModelAuditor


class TestCostModelAuditor:
    """Test suite for CostModelAuditor."""

    @pytest.fixture
    def auditor(self):
        """Fixture: instantiate the auditor."""
        return CostModelAuditor()

    # ── GOOD CASE: daily backtest with matching costs ──────────────────────

    def test_daily_backtest_passes_with_correct_costs(self, auditor):
        """Daily (1d) backtest with correct cost rates should PASS."""
        ctx = {
            "interval": "1d",
            "cost_rates": {
                "buy": 0.0026,  # Commission + clearing + stamp + slippage
                "sell": 0.0016,
            },
            "expected_cost_rates": {
                "buy": 0.0026,
                "sell": 0.0016,
            },
            "adv_value_myr": 10_000_000,
            "bars_per_day": 1.0,
        }
        finding = auditor.inspect("backtest_run:1", ctx)
        assert finding is not None
        assert finding.status == "PASS"
        assert finding.severity == "INFO"

    # ── BAD CASE 1: sub-daily with daily cost defaults (the critical bug) ──

    def test_subdaily_with_daily_cost_defaults_fails(self, auditor):
        """Sub-daily (1h) backtest using daily costs should FAIL with BLOCKER."""
        # In a 1h backtest, costs should be HIGHER per-bar because the position
        # is held for fewer bars per day (about 6.5 bars in Bursa 09:00-17:00).
        # If the auditor computes costs as if it were daily, we get artificially
        # low per-bar costs → overstated Sharpe.

        # Correct costs for 1h: daily cost / bars_per_day(1h)
        daily_buy = 0.0026
        daily_sell = 0.0016
        bars_per_hour = 6.5  # Bursa trades ~6.5 hours per day
        expected_buy = daily_buy / bars_per_hour
        expected_sell = daily_sell / bars_per_hour

        # But the run actually used the daily costs (the bug)
        ctx = {
            "interval": "1h",
            "cost_rates": {
                "buy": daily_buy,  # WRONG: using daily default
                "sell": daily_sell,  # WRONG: using daily default
            },
            "expected_cost_rates": {
                "buy": expected_buy,
                "sell": expected_sell,
            },
            "adv_value_myr": 10_000_000,
            "bars_per_day": bars_per_hour,
        }
        finding = auditor.inspect("backtest_run:2", ctx)
        assert finding is not None
        assert finding.status == "FAIL"
        assert finding.severity == "BLOCKER"
        # Should mention the sub-daily interval and using daily defaults
        assert "sub-daily" in finding.local_recommendation.lower()
        assert "silent fallback" in finding.local_recommendation.lower()
        assert "above expected" in finding.local_recommendation.lower()

    # ── BAD CASE 2: 15m interval with daily costs ────────────────────────────

    def test_15m_with_daily_costs_fails(self, auditor):
        """15-minute backtest using daily costs should FAIL."""
        daily_buy = 0.0026
        daily_sell = 0.0016
        bars_per_day = 26  # ~26 15-min bars in a 6.5-hour trading day

        # Correct costs for 15m
        expected_buy = daily_buy / bars_per_day
        expected_sell = daily_sell / bars_per_day

        # But the run used daily costs
        ctx = {
            "interval": "15m",
            "cost_rates": {
                "buy": daily_buy,  # WRONG
                "sell": daily_sell,  # WRONG
            },
            "expected_cost_rates": {
                "buy": expected_buy,
                "sell": expected_sell,
            },
            "adv_value_myr": 10_000_000,
            "bars_per_day": bars_per_day,
        }
        finding = auditor.inspect("backtest_run:3", ctx)
        assert finding is not None
        assert finding.status == "FAIL"
        assert finding.severity == "BLOCKER"

    # ── BAD CASE 3: missing cost_rates fields ────────────────────────────────

    def test_missing_buy_rate_fails(self, auditor):
        """Missing 'buy' field in cost_rates should FAIL."""
        ctx = {
            "interval": "1d",
            "cost_rates": {
                "sell": 0.0016,
                # 'buy' is missing
            },
            "expected_cost_rates": {
                "buy": 0.0026,
                "sell": 0.0016,
            },
        }
        finding = auditor.inspect("backtest_run:4", ctx)
        assert finding is not None
        assert finding.status == "FAIL"
        assert finding.severity == "BLOCKER"
        assert "incomplete" in finding.local_recommendation.lower()

    def test_missing_expected_rates_fails(self, auditor):
        """Missing 'buy' field in expected_cost_rates should FAIL."""
        ctx = {
            "interval": "1d",
            "cost_rates": {
                "buy": 0.0026,
                "sell": 0.0016,
            },
            "expected_cost_rates": {
                "sell": 0.0016,
                # 'buy' is missing
            },
        }
        finding = auditor.inspect("backtest_run:5", ctx)
        assert finding is not None
        assert finding.status == "FAIL"
        assert finding.severity == "BLOCKER"

    # ── BAD CASE 4: cost mismatch on daily interval ─────────────────────────

    def test_daily_with_mismatched_costs_fails(self, auditor):
        """Daily interval with cost mismatch should FAIL."""
        ctx = {
            "interval": "1d",
            "cost_rates": {
                "buy": 0.0020,  # Doesn't match
                "sell": 0.0016,
            },
            "expected_cost_rates": {
                "buy": 0.0026,
                "sell": 0.0016,
            },
            "adv_value_myr": 10_000_000,
            "bars_per_day": 1.0,
        }
        finding = auditor.inspect("backtest_run:6", ctx)
        assert finding is not None
        assert finding.status == "FAIL"
        assert finding.severity == "BLOCKER"
        assert "diverge" in finding.local_recommendation.lower()

    # ── EDGE CASE: tolerance for small float rounding ──────────────────────

    def test_small_rounding_differences_pass(self, auditor):
        """Small float rounding differences (< tolerance) should PASS."""
        base_buy = 0.0026
        base_sell = 0.0016
        # Add tiny rounding error that's within 0.01% tolerance
        ctx = {
            "interval": "1d",
            "cost_rates": {
                "buy": base_buy + 1e-8,  # Add tiny rounding error
                "sell": base_sell + 1e-8,
            },
            "expected_cost_rates": {
                "buy": base_buy,
                "sell": base_sell,
            },
            "adv_value_myr": 10_000_000,
            "bars_per_day": 1.0,
        }
        finding = auditor.inspect("backtest_run:7", ctx)
        assert finding is not None
        assert finding.status == "PASS"

    # ── WEEKLY and monthly backtests (should use daily rates, no scaling) ─────

    def test_weekly_backtest_passes_with_daily_rates(self, auditor):
        """Weekly (1w) backtest is NOT sub-daily, so daily rates are appropriate."""
        ctx = {
            "interval": "1w",
            "cost_rates": {
                "buy": 0.0026,
                "sell": 0.0016,
            },
            "expected_cost_rates": {
                "buy": 0.0026,
                "sell": 0.0016,
            },
            "adv_value_myr": 10_000_000,
            "bars_per_day": 1.0 / 5.0,  # ~0.2 bars per day
        }
        finding = auditor.inspect("backtest_run:8", ctx)
        assert finding is not None
        assert finding.status == "PASS"

    # ── Integration: realistic scenario ───────────────────────────────────────

    def test_realistic_1h_backtest_with_correct_costs(self, auditor):
        """Realistic 1h backtest on high-ADV stock with interval-correct costs."""
        # Blue-chip stock: RM25M ADV
        adv_value = 25_000_000
        bars_per_day = 6.5

        # Daily rates (what a 1d backtest would use):
        # commission 0.08%, clearing 0.03%, stamp 0.10% (buy), slippage 0.05%
        daily_buy = (25_000_000 * (0.0008 + 0.0003 + 0.0010 + 0.0005)) / 25_000_000
        daily_sell = (25_000_000 * (0.0008 + 0.0003 + 0.0005)) / 25_000_000

        # Correct rates for 1h (per-bar)
        hourly_buy = daily_buy / bars_per_day
        hourly_sell = daily_sell / bars_per_day

        ctx = {
            "interval": "1h",
            "cost_rates": {
                "buy": hourly_buy,
                "sell": hourly_sell,
            },
            "expected_cost_rates": {
                "buy": hourly_buy,
                "sell": hourly_sell,
            },
            "adv_value_myr": adv_value,
            "bars_per_day": bars_per_day,
        }
        finding = auditor.inspect("backtest_run:9", ctx)
        assert finding is not None
        assert finding.status == "PASS"
        assert "correct for interval '1h'" in finding.local_recommendation.lower()

    def test_realistic_1h_backtest_with_daily_cost_bug(self, auditor):
        """Realistic 1h backtest that accidentally uses daily costs (the bug)."""
        adv_value = 25_000_000
        bars_per_day = 6.5

        daily_buy = 0.00260
        daily_sell = 0.00160

        hourly_buy = daily_buy / bars_per_day
        hourly_sell = daily_sell / bars_per_day

        # The bug: run actually applied daily costs
        ctx = {
            "interval": "1h",
            "cost_rates": {
                "buy": daily_buy,  # BUG: should be hourly_buy
                "sell": daily_sell,  # BUG: should be hourly_sell
            },
            "expected_cost_rates": {
                "buy": hourly_buy,
                "sell": hourly_sell,
            },
            "adv_value_myr": adv_value,
            "bars_per_day": bars_per_day,
        }
        finding = auditor.inspect("backtest_run:10", ctx)
        assert finding is not None
        assert finding.status == "FAIL"
        assert finding.severity == "BLOCKER"
        # The delta should be roughly 6.5x (since costs are ~6.5x too high)
        buy_ratio = daily_buy / hourly_buy
        assert buy_ratio > 6.0  # About 6.5x too high
