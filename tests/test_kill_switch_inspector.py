"""Tests for KillSwitchInspector — paper-trading kill-switch governance."""

import pytest
from governance.inspectors.kill_switch import KillSwitchInspector


def test_kill_switch_no_triggers():
    """Good case: no kill switches triggered — strategies are healthy."""
    inspector = KillSwitchInspector()
    finding = inspector.inspect(
        scope="portfolio_risk",
        ctx={
            "kill_switches": {
                "triggered": [],
                "count": 0,
            }
        }
    )
    assert finding is not None
    assert finding.agent == "KillSwitchInspector"
    assert finding.level == "L0"
    assert finding.scope == "portfolio_risk"
    assert finding.status == "PASS"
    assert finding.severity == "INFO"
    assert finding.evidence["triggered_count"] == 0
    assert finding.evidence["active_strategies_healthy"] is True


def test_kill_switch_drawdown_breach():
    """Bad case: drawdown kill switch triggered."""
    inspector = KillSwitchInspector()
    finding = inspector.inspect(
        scope="portfolio_risk",
        ctx={
            "kill_switches": {
                "triggered": [
                    {
                        "idea_id": 567,
                        "trigger": "drawdown",
                        "detail": "DD 27.5%",
                    }
                ],
                "count": 1,
            }
        }
    )
    assert finding is not None
    assert finding.status == "FAIL"
    assert finding.severity == "BLOCKER"
    assert finding.evidence["triggered_count"] == 1
    assert len(finding.evidence["triggers"]) == 1
    assert finding.evidence["triggers"][0]["idea_id"] == 567
    assert finding.evidence["triggers"][0]["trigger"] == "drawdown"
    assert "drawdown-breached" in finding.local_recommendation.lower()
    assert finding.escalate_to == "RiskMonitor"


def test_kill_switch_data_confidence_failure():
    """Bad case: data confidence kill switch triggered."""
    inspector = KillSwitchInspector()
    finding = inspector.inspect(
        scope="portfolio_risk",
        ctx={
            "kill_switches": {
                "triggered": [
                    {
                        "idea_id": 123,
                        "trigger": "data_confidence",
                        "detail": "65/100",
                    }
                ],
                "count": 1,
            }
        }
    )
    assert finding is not None
    assert finding.status == "FAIL"
    assert finding.severity == "BLOCKER"
    assert finding.evidence["triggered_count"] == 1
    assert finding.evidence["triggers"][0]["trigger"] == "data_confidence"
    assert "data quality" in finding.local_recommendation.lower()


def test_kill_switch_corporate_action_unresolved():
    """Bad case: unresolved corporate action kill switch triggered."""
    inspector = KillSwitchInspector()
    finding = inspector.inspect(
        scope="portfolio_risk",
        ctx={
            "kill_switches": {
                "triggered": [
                    {
                        "idea_id": 999,
                        "trigger": "corporate_action",
                        "detail": "2 unresolved",
                    }
                ],
                "count": 1,
            }
        }
    )
    assert finding is not None
    assert finding.status == "FAIL"
    assert finding.severity == "BLOCKER"
    assert finding.evidence["triggers"][0]["trigger"] == "corporate_action"
    assert "corporate action" in finding.local_recommendation.lower()


def test_kill_switch_multiple_triggers():
    """Bad case: multiple kill switches triggered (escalate to PortfolioExecutor)."""
    inspector = KillSwitchInspector()
    finding = inspector.inspect(
        scope="portfolio_risk",
        ctx={
            "kill_switches": {
                "triggered": [
                    {
                        "idea_id": 567,
                        "trigger": "drawdown",
                        "detail": "DD 28.0%",
                    },
                    {
                        "idea_id": 123,
                        "trigger": "data_confidence",
                        "detail": "60/100",
                    },
                    {
                        "idea_id": 999,
                        "trigger": "corporate_action",
                        "detail": "1 unresolved",
                    },
                ],
                "count": 3,
            }
        }
    )
    assert finding is not None
    assert finding.status == "FAIL"
    assert finding.severity == "BLOCKER"
    assert finding.evidence["triggered_count"] == 3
    assert len(finding.evidence["triggers"]) == 3
    # Multiple triggers (count > 1) should escalate to PortfolioExecutor
    assert finding.escalate_to == "PortfolioExecutor"
    # Evidence should include all trigger types
    assert set(finding.evidence["trigger_types"]) == {
        "drawdown",
        "data_confidence",
        "corporate_action",
    }
    # All three recommendations should be present
    assert "drawdown-breached" in finding.local_recommendation.lower()
    assert "data quality" in finding.local_recommendation.lower()
    assert "corporate action" in finding.local_recommendation.lower()


def test_kill_switch_missing_kill_switches_context():
    """Edge case: missing kill_switches in context (default to empty)."""
    inspector = KillSwitchInspector()
    finding = inspector.inspect(
        scope="portfolio_risk",
        ctx={}  # No kill_switches key
    )
    assert finding is not None
    assert finding.status == "PASS"
    assert finding.severity == "INFO"
    assert finding.evidence["triggered_count"] == 0


def test_kill_switch_finding_recorded():
    """Test that findings can be persisted via the record() method."""
    inspector = KillSwitchInspector()
    finding = inspector.inspect(
        scope="portfolio_risk",
        ctx={
            "kill_switches": {
                "triggered": [],
                "count": 0,
            }
        }
    )
    # This will write to the DB
    row_id = inspector.record(finding)
    assert isinstance(row_id, int)
    assert row_id > 0


def test_kill_switch_finding_recorded_with_blocker():
    """Test that BLOCKER findings can be persisted."""
    inspector = KillSwitchInspector()
    finding = inspector.inspect(
        scope="portfolio_risk",
        ctx={
            "kill_switches": {
                "triggered": [
                    {
                        "idea_id": 567,
                        "trigger": "drawdown",
                        "detail": "DD 30.0%",
                    }
                ],
                "count": 1,
            }
        }
    )
    row_id = inspector.record(finding)
    assert isinstance(row_id, int)
    assert row_id > 0


def test_kill_switch_single_trigger_escalation():
    """Test that single trigger escalates to RiskMonitor (not PortfolioExecutor)."""
    inspector = KillSwitchInspector()
    finding = inspector.inspect(
        scope="portfolio_risk",
        ctx={
            "kill_switches": {
                "triggered": [
                    {
                        "idea_id": 567,
                        "trigger": "drawdown",
                        "detail": "DD 25.1%",
                    }
                ],
                "count": 1,
            }
        }
    )
    assert finding.escalate_to == "RiskMonitor"


def test_kill_switch_two_triggers_escalation():
    """Test that 2+ triggers escalate to PortfolioExecutor."""
    inspector = KillSwitchInspector()
    finding = inspector.inspect(
        scope="portfolio_risk",
        ctx={
            "kill_switches": {
                "triggered": [
                    {
                        "idea_id": 567,
                        "trigger": "drawdown",
                        "detail": "DD 26.0%",
                    },
                    {
                        "idea_id": 123,
                        "trigger": "data_confidence",
                        "detail": "70/100",
                    },
                ],
                "count": 2,
            }
        }
    )
    assert finding.escalate_to == "PortfolioExecutor"


def test_kill_switch_duplicate_trigger_types():
    """Test multiple triggers of the same type (e.g., two strategies' drawdowns)."""
    inspector = KillSwitchInspector()
    finding = inspector.inspect(
        scope="portfolio_risk",
        ctx={
            "kill_switches": {
                "triggered": [
                    {
                        "idea_id": 567,
                        "trigger": "drawdown",
                        "detail": "DD 26.0%",
                    },
                    {
                        "idea_id": 890,
                        "trigger": "drawdown",
                        "detail": "DD 28.5%",
                    },
                ],
                "count": 2,
            }
        }
    )
    assert finding is not None
    assert finding.status == "FAIL"
    assert finding.severity == "BLOCKER"
    assert finding.evidence["triggered_count"] == 2
    # Even though both are drawdown, they're for different ideas
    assert len(finding.evidence["triggers"]) == 2
    # trigger_types should only contain "drawdown" once (it's a set)
    assert finding.evidence["trigger_types"] == ["drawdown"]
    assert finding.escalate_to == "PortfolioExecutor"
