"""Tests for governance department managers and rollup logic."""

import pytest
from datetime import datetime, timedelta
import json

from governance.managers import (
    DEPARTMENT_AGENT_MAP,
    summarize_department,
    summarize_all_departments,
    get_all_inspector_names,
)
from governance.schemas import DepartmentSummary
from governance.inspectors import (
    DSLRepresentabilityChecker,
    LeafSemanticsAuditor,
    NegativeMappingGuard,
    PnLConsistencyInspector,
    FundingCostAuditor,
    FillConventionAuditor,
    CostModelAuditor,
    MetricConsistencyAuditor,
    RegimeAttributionAuditor,
    ShadowNAVInspector,
    ConcentrationCorrelationInspector,
    CapacityAggregationInspector,
    KillSwitchInspector,
    SourceHealthInspector,
)
from data.database import db_session, init_db
from governance.schemas import Finding


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Create a temporary test database."""
    db_path = tmp_path / "test_governance.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    monkeypatch.setenv("MARKET_MODE", "bursa")

    # Force reimport of config to pick up the env var
    import importlib
    import config.settings
    importlib.reload(config.settings)

    from config.settings import DB_PATH
    # Create the DB at the new path
    init_db()

    # Clear any existing governance findings
    with db_session() as conn:
        conn.execute("DELETE FROM governance_findings")

    yield db_path
    # Cleanup happens automatically when tmp_path is removed


@pytest.fixture(autouse=True)
def cleanup_findings(temp_db):
    """Automatically clean up governance findings before each test."""
    with db_session() as conn:
        conn.execute("DELETE FROM governance_findings")
    yield
    # Cleanup after test is optional since each test gets fresh DB


class TestDepartmentAgentMap:
    """Test the DEPARTMENT_AGENT_MAP structure."""

    def test_all_departments_have_agents(self):
        """Each department should have at least one agent (except Paper Trading
        which is a placeholder)."""
        for dept, agents in DEPARTMENT_AGENT_MAP.items():
            if dept != "Paper Trading":
                assert len(agents) > 0, f"Department {dept} has no agents"

    def test_all_agents_are_strings(self):
        """All agent names should be strings."""
        for agents in DEPARTMENT_AGENT_MAP.values():
            for agent in agents:
                assert isinstance(agent, str), f"Agent name {agent} is not a string"

    def test_no_duplicate_agents_across_departments(self):
        """Agents should not be assigned to multiple departments."""
        all_agents = []
        for dept, agents in DEPARTMENT_AGENT_MAP.items():
            all_agents.extend(agents)

        duplicates = [a for a in all_agents if all_agents.count(a) > 1]
        assert not duplicates, f"Agents assigned to multiple departments: {duplicates}"

    def test_department_names_are_consistent(self):
        """Department names should be capitalized and consistent."""
        assert all(isinstance(dept, str) for dept in DEPARTMENT_AGENT_MAP.keys())
        # All department names should be non-empty
        assert all(len(dept) > 0 for dept in DEPARTMENT_AGENT_MAP.keys())

    def test_expected_departments_exist(self):
        """Check that all expected departments exist."""
        expected_depts = {
            "Backtest Fidelity",
            "Parser Honesty",
            "Portfolio Risk",
            "Data Integrity",
            "Paper Trading",
        }
        actual_depts = set(DEPARTMENT_AGENT_MAP.keys())
        assert actual_depts == expected_depts, f"Department mismatch. Expected: {expected_depts}, Got: {actual_depts}"

    def test_expected_inspectors_are_mapped(self):
        """All 14 inspector class names should be in DEPARTMENT_AGENT_MAP."""
        expected_inspectors = {
            "PnLConsistencyInspector",
            "FundingCostAuditor",
            "FillConventionAuditor",
            "CostModelAuditor",
            "MetricConsistencyAuditor",
            "RegimeAttributionAuditor",
            "DSLRepresentabilityChecker",
            "LeafSemanticsAuditor",
            "NegativeMappingGuard",
            "ShadowNAVInspector",
            "ConcentrationCorrelationInspector",
            "CapacityAggregationInspector",
            "KillSwitchInspector",
            "SourceHealthInspector",
        }
        actual_inspectors = set(get_all_inspector_names())
        assert actual_inspectors == expected_inspectors, (
            f"Inspector mismatch. Expected: {expected_inspectors}, "
            f"Got: {actual_inspectors}"
        )


class TestSummarizeDepartment:
    """Test the summarize_department function."""

    def test_empty_findings_returns_green(self, temp_db):
        """Department with no findings should be GREEN/HEALTHY."""
        summary = summarize_department("Data Integrity", lookback_hours=24)

        assert summary.department == "Data Integrity"
        assert summary.status == "HEALTHY"
        assert summary.findings_count == 0
        assert summary.blocking_issues == 0
        assert summary.recommendations == []

    def test_pass_findings_returns_green(self, temp_db):
        """Department with only PASS findings should be GREEN/HEALTHY."""
        # Directly insert a PASS finding (avoid calling live inspector)
        with db_session() as conn:
            conn.execute(
                """INSERT INTO governance_findings
                   (agent, level, scope, status, severity, evidence, local_recommendation)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                ("SourceHealthInspector", "L0", "data_source:yahoo", "PASS", "INFO",
                 '{"status": "healthy"}', "Yahoo Finance API is responsive."),
            )

        summary = summarize_department("Data Integrity", lookback_hours=24)

        assert summary.status == "HEALTHY"
        assert summary.findings_count == 1
        assert summary.blocking_issues == 0

    def test_warning_findings_returns_amber(self, temp_db):
        """Department with WARNING findings should be AMBER/WARNING."""
        with db_session() as conn:
            conn.execute(
                """INSERT INTO governance_findings
                   (agent, level, scope, status, severity, evidence, local_recommendation)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                ("SourceHealthInspector", "L0", "data_source:yahoo", "WARN", "WARNING",
                 '{"status": "degraded"}', "Yahoo Finance API is temporarily unavailable."),
            )

        summary = summarize_department("Data Integrity", lookback_hours=24)

        assert summary.status == "WARNING"
        assert summary.findings_count == 1
        assert summary.blocking_issues == 0
        assert len(summary.recommendations) > 0

    def test_blocker_findings_returns_red(self, temp_db):
        """Department with BLOCKER findings should be RED/ALERT."""
        with db_session() as conn:
            conn.execute(
                """INSERT INTO governance_findings
                   (agent, level, scope, status, severity, evidence, local_recommendation, escalate_to)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                ("SourceHealthInspector", "L0", "data_source:yahoo", "FAIL", "BLOCKER",
                 '{"error": "API timeout"}', "Yahoo Finance is unavailable; check API health.",
                 "DataEngineer"),
            )

        summary = summarize_department("Data Integrity", lookback_hours=24)

        assert summary.status == "ALERT"
        assert summary.findings_count == 1
        assert summary.blocking_issues == 1
        assert len(summary.recommendations) > 0

    def test_mixed_findings_worst_severity_wins(self, temp_db):
        """With mixed findings, worst severity should determine status.
        BLOCKER > WARNING > INFO."""
        with db_session() as conn:
            # Insert PASS/INFO finding
            conn.execute(
                """INSERT INTO governance_findings
                   (agent, level, scope, status, severity)
                   VALUES (?, ?, ?, ?, ?)""",
                ("SourceHealthInspector", "L0", "data_source:yahoo", "PASS", "INFO"),
            )

            # Insert WARNING finding
            conn.execute(
                """INSERT INTO governance_findings
                   (agent, level, scope, status, severity, local_recommendation)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                ("SourceHealthInspector", "L0", "data_source:klse_screener", "WARN", "WARNING",
                 "KLSE Screener is degraded."),
            )

            # Insert BLOCKER finding
            conn.execute(
                """INSERT INTO governance_findings
                   (agent, level, scope, status, severity, local_recommendation, escalate_to)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                ("SourceHealthInspector", "L0", "data_source:binance", "FAIL", "BLOCKER",
                 "Binance API is down.", "DataEngineer"),
            )

        summary = summarize_department("Data Integrity", lookback_hours=24)

        # BLOCKER is the worst, so status should be ALERT
        assert summary.status == "ALERT"
        assert summary.findings_count == 3
        assert summary.blocking_issues == 1

    def test_recommendations_deduped_and_capped(self, temp_db):
        """Recommendations should be deduplicated and capped at ~5."""
        with db_session() as conn:
            # Insert 10 warnings with 2 unique recommendations
            for i in range(10):
                rec = "Check source health." if i < 5 else "Investigate connectivity."
                conn.execute(
                    """INSERT INTO governance_findings
                       (agent, level, scope, status, severity, local_recommendation)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    ("SourceHealthInspector", "L0", f"data_source:source_{i}", "WARN", "WARNING", rec),
                )

        summary = summarize_department("Data Integrity", lookback_hours=24)

        # Should have 2 unique recommendations, capped at 5
        assert len(summary.recommendations) == 2

    def test_lookback_window_respected(self, temp_db):
        """Findings outside the lookback window should not be counted."""
        # Insert an old finding (30 hours ago)
        with db_session() as conn:
            old_time = (datetime.utcnow() - timedelta(hours=30)).isoformat()
            conn.execute(
                """INSERT INTO governance_findings
                   (agent, level, scope, status, severity, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                ("SourceHealthInspector", "L0", "data_source:yahoo",
                 "PASS", "INFO", old_time),
            )

            # Insert a recent finding (1 hour ago via default timestamp)
            conn.execute(
                """INSERT INTO governance_findings
                   (agent, level, scope, status, severity)
                   VALUES (?, ?, ?, ?, ?)""",
                ("SourceHealthInspector", "L0", "data_source:klse_screener", "PASS", "INFO"),
            )

        # Query with 24-hour lookback
        summary = summarize_department("Data Integrity", lookback_hours=24)

        # Should only count the recent finding
        assert summary.findings_count == 1

    def test_unknown_department(self, temp_db):
        """Unknown department should return HEALTHY with zero findings."""
        summary = summarize_department("Nonexistent Department", lookback_hours=24)

        assert summary.department == "Nonexistent Department"
        assert summary.status == "HEALTHY"
        assert summary.findings_count == 0
        assert summary.blocking_issues == 0


class TestSummarizeAllDepartments:
    """Test the summarize_all_departments function."""

    def test_all_departments_returned(self, temp_db):
        """All departments should be returned, even if empty."""
        summaries = summarize_all_departments(lookback_hours=24)

        depts = {s.department for s in summaries}
        expected = set(DEPARTMENT_AGENT_MAP.keys())
        assert depts == expected

    def test_paper_trading_gracefully_empty(self, temp_db):
        """Paper Trading department with no inspectors should be HEALTHY."""
        summaries = summarize_all_departments(lookback_hours=24)

        paper_trading = next(s for s in summaries if s.department == "Paper Trading")
        assert paper_trading.status == "HEALTHY"
        assert paper_trading.findings_count == 0
        assert paper_trading.blocking_issues == 0

    def test_sorted_by_department_name(self, temp_db):
        """Departments should be returned sorted by name."""
        summaries = summarize_all_departments(lookback_hours=24)
        depts = [s.department for s in summaries]

        assert depts == sorted(depts)

    def test_mixed_statuses_across_departments(self, temp_db):
        """Departments should independently reflect their own findings."""
        with db_session() as conn:
            # Insert a PASS finding for Data Integrity
            conn.execute(
                """INSERT INTO governance_findings
                   (agent, level, scope, status, severity)
                   VALUES (?, ?, ?, ?, ?)""",
                ("SourceHealthInspector", "L0", "data_source:yahoo", "PASS", "INFO"),
            )

            # Insert a BLOCKER finding for Parser Honesty (DSL)
            conn.execute(
                """INSERT INTO governance_findings
                   (agent, level, scope, status, severity, evidence, local_recommendation)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                ("DSLRepresentabilityChecker", "L0", "dsl_registry", "FAIL", "BLOCKER",
                 '{"issue": "unknown leaf used"}', "Fix DSL registry."),
            )

        summaries = summarize_all_departments(lookback_hours=24)

        di_summary = next(s for s in summaries if s.department == "Data Integrity")
        ph_summary = next(s for s in summaries if s.department == "Parser Honesty")

        assert di_summary.status == "HEALTHY"  # Only PASS finding
        assert ph_summary.status == "ALERT"    # BLOCKER finding


class TestGetAllInspectorNames:
    """Test the get_all_inspector_names function."""

    def test_returns_all_inspectors(self):
        """Should return all 14 inspector names."""
        names = get_all_inspector_names()

        expected = {
            "PnLConsistencyInspector",
            "FundingCostAuditor",
            "FillConventionAuditor",
            "CostModelAuditor",
            "MetricConsistencyAuditor",
            "RegimeAttributionAuditor",
            "DSLRepresentabilityChecker",
            "LeafSemanticsAuditor",
            "NegativeMappingGuard",
            "ShadowNAVInspector",
            "ConcentrationCorrelationInspector",
            "CapacityAggregationInspector",
            "KillSwitchInspector",
            "SourceHealthInspector",
        }

        assert set(names) == expected
        assert len(names) == 14
