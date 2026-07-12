"""Tests for DSLRepresentabilityChecker inspector.

Validates that the DSL parser adheres to the honesty contract:
- Unknown leaves are NEVER silently substituted (BLOCKER if detected)
- Representable strategies parse to valid trees with known leaves only
- Unrepresentable strategies correctly return representable=false with reason
"""

import pytest
from governance.inspectors.dsl_representability import DSLRepresentabilityChecker
from agents.backtest_engineer import signal_dsl


@pytest.fixture
def checker():
    """Instantiate the DSLRepresentabilityChecker."""
    return DSLRepresentabilityChecker()


class TestHonestRepresentableParse:
    """Tests for strategies that CAN be represented honestly."""

    def test_honest_rsi_parse(self, checker):
        """A strategy using only known leaves (rsi) should PASS."""
        parse_result = {
            "representable": True,
            "dsl": {
                "entry": {"leaf": "rsi", "period": 14, "above": 50},
                "exit": None,
            },
            "notes": "RSI above 50 entry",
        }
        ctx = {
            "parse_result": parse_result,
            "leaf_registry": signal_dsl.LEAVES,
            "factor_formula": "RSI above 50",
            "title": "RSI Momentum",
            "expected_representable": True,
        }

        finding = checker.inspect("parse_test:1", ctx)
        assert finding is not None
        assert finding.status == "PASS"
        assert finding.severity == "INFO"
        assert "honest representable parse" in finding.evidence.get("issue", "")

    def test_honest_ma_cross_parse(self, checker):
        """A strategy using sma_cross with valid parameters should PASS."""
        parse_result = {
            "representable": True,
            "dsl": {
                "entry": {
                    "op": "AND",
                    "children": [
                        {"leaf": "sma_cross", "fast": 10, "slow": 50, "direction": "above"},
                        {"leaf": "volume_ratio", "period": 20, "min_ratio": 1.5},
                    ],
                },
                "exit": None,
            },
            "notes": "Golden cross with volume confirmation",
        }
        ctx = {
            "parse_result": parse_result,
            "leaf_registry": signal_dsl.LEAVES,
            "factor_formula": "10-day SMA crosses above 50-day SMA with volume spike",
            "title": "SMA Golden Cross",
            "expected_representable": True,
        }

        finding = checker.inspect("parse_test:2", ctx)
        assert finding is not None
        assert finding.status == "PASS"
        assert "sma_cross" in str(finding.evidence.get("leaves_used", []))
        assert "volume_ratio" in str(finding.evidence.get("leaves_used", []))

    def test_honest_complex_tree_parse(self, checker):
        """A strategy using multiple known leaves with nesting should PASS."""
        parse_result = {
            "representable": True,
            "dsl": {
                "entry": {
                    "op": "OR",
                    "children": [
                        {"leaf": "rsi", "period": 14, "below": 30},
                        {
                            "op": "AND",
                            "children": [
                                {"leaf": "bollinger", "period": 20, "std": 2.0, "band": "below_lower"},
                                {"leaf": "volume_ratio", "period": 10, "min_ratio": 2.0},
                            ],
                        },
                    ],
                },
                "exit": {"leaf": "rsi", "period": 14, "above": 70},
            },
            "notes": "Mean reversion with volume confirmation",
        }
        ctx = {
            "parse_result": parse_result,
            "leaf_registry": signal_dsl.LEAVES,
            "factor_formula": "RSI below 30 OR (Bollinger lower + volume spike), exit on RSI above 70",
            "title": "Mean Reversion",
            "expected_representable": True,
        }

        finding = checker.inspect("parse_test:3", ctx)
        assert finding is not None
        assert finding.status == "PASS"


class TestSilentSubstitutionBLOCKER:
    """Tests for the BLOCKER case: silent substitution of unknown leaves."""

    def test_unknown_leaf_substitution_is_blocker(self, checker):
        """Parser returning true with an UNKNOWN leaf is BLOCKER (silent substitution)."""
        parse_result = {
            "representable": True,
            "dsl": {
                "entry": {
                    "leaf": "sentiment_score",  # UNKNOWN leaf — not in signal_dsl.LEAVES
                    "min_sentiment": 0.7,
                },
                "exit": None,
            },
            "notes": "Sentiment above 0.7",
        }
        ctx = {
            "parse_result": parse_result,
            "leaf_registry": signal_dsl.LEAVES,
            "factor_formula": "Buy when sentiment score > 0.7",
            "title": "Sentiment Trade",
            "used_unknown_leaf": False,  # We'll rely on the checker to find it
        }

        finding = checker.inspect("parse_test:blocker_1", ctx)
        assert finding is not None
        assert finding.status == "FAIL"
        assert finding.severity == "BLOCKER"
        assert "silent substitution" in finding.evidence.get("issue", "").lower()
        assert "sentiment_score" in str(finding.evidence.get("unknown_leaves", []))

    def test_nested_unknown_leaf_in_tree_is_blocker(self, checker):
        """Parser returning true with unknown leaf nested in AND/OR is BLOCKER."""
        parse_result = {
            "representable": True,
            "dsl": {
                "entry": {
                    "op": "AND",
                    "children": [
                        {"leaf": "rsi", "period": 14, "above": 50},
                        {
                            "leaf": "earnings_surprise",  # UNKNOWN leaf
                            "min_surprise": 0.05,
                        },
                    ],
                },
                "exit": None,
            },
            "notes": "RSI + earnings surprise",
        }
        ctx = {
            "parse_result": parse_result,
            "leaf_registry": signal_dsl.LEAVES,
            "factor_formula": "RSI above 50 AND earnings surprise > 5%",
            "title": "RSI + Earnings",
        }

        finding = checker.inspect("parse_test:blocker_2", ctx)
        assert finding is not None
        assert finding.status == "FAIL"
        assert finding.severity == "BLOCKER"
        assert "earnings_surprise" in str(finding.evidence.get("unknown_leaves", []))

    def test_used_unknown_leaf_flag_is_blocker(self, checker):
        """Parser reporting unknown leaf via used_unknown_leaf flag is BLOCKER."""
        parse_result = {
            "representable": True,
            "dsl": {
                "entry": {"leaf": "rsi", "period": 14, "above": 50},
                "exit": None,
            },
            "notes": "RSI only",
        }
        ctx = {
            "parse_result": parse_result,
            "leaf_registry": signal_dsl.LEAVES,
            "factor_formula": "Some formula",
            "title": "Test",
            "used_unknown_leaf": True,  # Flag set explicitly
        }

        finding = checker.inspect("parse_test:blocker_3", ctx)
        assert finding is not None
        assert finding.status == "FAIL"
        assert finding.severity == "BLOCKER"


class TestHonestUnrepresentableParse:
    """Tests for strategies that CANNOT be represented (honest rejection)."""

    def test_honest_unrepresentable_earnings_based(self, checker):
        """Parser correctly rejecting earnings-based strategy should PASS."""
        parse_result = {
            "representable": False,
            "reason": "Earnings surprise data is unavailable; only price/volume/dividend leaves are supported",
        }
        ctx = {
            "parse_result": parse_result,
            "leaf_registry": signal_dsl.LEAVES,
            "factor_formula": "Buy on positive earnings surprise",
            "title": "Earnings Strategy",
            "expected_representable": False,
        }

        finding = checker.inspect("parse_test:unrep_1", ctx)
        assert finding is not None
        assert finding.status == "PASS"
        assert "honest unrepresentable rejection" in finding.evidence.get("issue", "").lower()
        assert "earnings surprise data is unavailable" in finding.evidence.get("reason", "").lower()

    def test_honest_unrepresentable_analyst_coverage(self, checker):
        """Parser correctly rejecting analyst coverage strategy should PASS."""
        parse_result = {
            "representable": False,
            "reason": "Analyst coverage changes require external data source not available",
        }
        ctx = {
            "parse_result": parse_result,
            "leaf_registry": signal_dsl.LEAVES,
            "factor_formula": "Buy when analyst coverage increases",
            "title": "Analyst Coverage",
            "expected_representable": False,
        }

        finding = checker.inspect("parse_test:unrep_2", ctx)
        assert finding is not None
        assert finding.status == "PASS"
        assert "honest unrepresentable rejection" in finding.evidence.get("issue", "").lower()

    def test_honest_unrepresentable_sentiment(self, checker):
        """Parser correctly rejecting sentiment-based strategy should PASS."""
        parse_result = {
            "representable": False,
            "reason": "Sentiment scoring from social media requires external data integration",
        }
        ctx = {
            "parse_result": parse_result,
            "leaf_registry": signal_dsl.LEAVES,
            "factor_formula": "Buy on positive sentiment shift",
            "title": "Sentiment Strategy",
        }

        finding = checker.inspect("parse_test:unrep_3", ctx)
        assert finding is not None
        assert finding.status == "PASS"


class TestExpectedVsActualMismatches:
    """Tests for cases where expected representability doesn't match actual."""

    def test_expected_unrepresentable_but_got_representable(self, checker):
        """Test reports WARNING when parser unexpectedly found a representation."""
        parse_result = {
            "representable": True,
            "dsl": {
                "entry": {"leaf": "rsi", "period": 14, "above": 50},
                "exit": None,
            },
            "notes": "RSI above 50",
        }
        ctx = {
            "parse_result": parse_result,
            "leaf_registry": signal_dsl.LEAVES,
            "factor_formula": "RSI above 50",
            "title": "RSI Test",
            "expected_representable": False,  # We expected unrepresentable
        }

        finding = checker.inspect("parse_test:mismatch_1", ctx)
        assert finding is not None
        assert finding.status == "WARN"
        assert finding.severity == "WARNING"
        assert "representable=true when test expected false" in finding.evidence.get("issue", "").lower()

    def test_expected_representable_but_got_unrepresentable(self, checker):
        """Test reports WARNING when parser unexpectedly rejected as unrepresentable."""
        parse_result = {
            "representable": False,
            "reason": "Some reason",
        }
        ctx = {
            "parse_result": parse_result,
            "leaf_registry": signal_dsl.LEAVES,
            "factor_formula": "RSI above 50",
            "title": "RSI Test",
            "expected_representable": True,  # We expected representable
        }

        finding = checker.inspect("parse_test:mismatch_2", ctx)
        assert finding is not None
        assert finding.status == "WARN"
        assert finding.severity == "WARNING"
        assert "returned representable=false when test expected true" in finding.evidence.get("issue", "").lower()


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_parse_result(self, checker):
        """Empty parse result defaults to unrepresentable."""
        parse_result = {}
        ctx = {
            "parse_result": parse_result,
            "leaf_registry": signal_dsl.LEAVES,
            "factor_formula": "Something",
            "title": "Empty Test",
        }

        finding = checker.inspect("parse_test:edge_1", ctx)
        assert finding is not None
        # Should be treated as representable=false
        assert finding.status == "PASS"
        assert "honest unrepresentable rejection" in finding.evidence.get("issue", "").lower()

    def test_no_expected_representability_provided(self, checker):
        """When expected_representable is None, should still validate honesty."""
        parse_result = {
            "representable": True,
            "dsl": {
                "entry": {"leaf": "rsi", "period": 14, "above": 50},
                "exit": None,
            },
            "notes": "RSI above 50",
        }
        ctx = {
            "parse_result": parse_result,
            "leaf_registry": signal_dsl.LEAVES,
            "factor_formula": "RSI above 50",
            "title": "RSI Test",
            # expected_representable not provided
        }

        finding = checker.inspect("parse_test:edge_2", ctx)
        assert finding is not None
        # Should still PASS because the tree is honest
        assert finding.status == "PASS"

    def test_multiple_unknown_leaves_in_tree(self, checker):
        """Parser with multiple unknown leaves should report all in BLOCKER."""
        parse_result = {
            "representable": True,
            "dsl": {
                "entry": {
                    "op": "AND",
                    "children": [
                        {"leaf": "sentiment_score", "min_sentiment": 0.7},
                        {"leaf": "earnings_surprise", "min_surprise": 0.05},
                    ],
                },
                "exit": None,
            },
            "notes": "Sentiment and earnings",
        }
        ctx = {
            "parse_result": parse_result,
            "leaf_registry": signal_dsl.LEAVES,
            "factor_formula": "Sentiment + earnings",
            "title": "Multi Unknown",
        }

        finding = checker.inspect("parse_test:edge_3", ctx)
        assert finding is not None
        assert finding.status == "FAIL"
        assert finding.severity == "BLOCKER"
        unknown_leaves = finding.evidence.get("unknown_leaves", [])
        assert "sentiment_score" in unknown_leaves
        assert "earnings_surprise" in unknown_leaves

    def test_short_leg_in_crypto_parse(self, checker):
        """Crypto strategies with short legs should be validated for unknown leaves."""
        parse_result = {
            "representable": True,
            "dsl": {
                "entry": {"leaf": "rsi", "period": 14, "above": 50},
                "exit": None,
                "short_entry": {"leaf": "rsi", "period": 14, "below": 30},
                "short_exit": None,
            },
            "notes": "Long on RSI overbought, short on RSI oversold",
        }
        ctx = {
            "parse_result": parse_result,
            "leaf_registry": signal_dsl.LEAVES,
            "factor_formula": "RSI-based mean reversion (both legs)",
            "title": "RSI Mean Reversion Crypto",
            "expected_representable": True,
        }

        finding = checker.inspect("parse_test:edge_4", ctx)
        assert finding is not None
        assert finding.status == "PASS"


class TestLeafExtraction:
    """Tests for correct extraction of leaves used in trees."""

    def test_extract_single_leaf(self, checker):
        """Correctly extract single leaf from tree."""
        parse_result = {
            "representable": True,
            "dsl": {
                "entry": {"leaf": "momentum", "period": 20, "min_return": 0.02},
                "exit": None,
            },
            "notes": "Momentum entry",
        }
        ctx = {
            "parse_result": parse_result,
            "leaf_registry": signal_dsl.LEAVES,
            "factor_formula": "20-day momentum > 2%",
            "title": "Momentum Test",
        }

        finding = checker.inspect("parse_test:extract_1", ctx)
        assert finding is not None
        assert "momentum" in finding.evidence.get("leaves_used", [])

    def test_extract_multiple_leaves_from_tree(self, checker):
        """Correctly extract multiple leaves from complex tree."""
        parse_result = {
            "representable": True,
            "dsl": {
                "entry": {
                    "op": "AND",
                    "children": [
                        {"leaf": "rsi", "period": 14, "above": 50},
                        {
                            "op": "OR",
                            "children": [
                                {"leaf": "macd", "fast": 12, "slow": 26, "signal": 9, "condition": "bullish"},
                                {"leaf": "momentum", "period": 10, "min_return": 0.01},
                            ],
                        },
                    ],
                },
                "exit": {"leaf": "bollinger", "period": 20, "std": 2.0, "band": "above_upper"},
            },
            "notes": "Complex entry/exit",
        }
        ctx = {
            "parse_result": parse_result,
            "leaf_registry": signal_dsl.LEAVES,
            "factor_formula": "Complex RSI+MACD/momentum",
            "title": "Complex Extraction",
        }

        finding = checker.inspect("parse_test:extract_2", ctx)
        assert finding is not None
        leaves_used = finding.evidence.get("leaves_used", [])
        assert "rsi" in leaves_used
        assert "macd" in leaves_used
        assert "momentum" in leaves_used
        assert "bollinger" in leaves_used


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
