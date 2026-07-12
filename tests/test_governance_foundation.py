"""Governance foundation: table schema, inspector base class, and schemas."""

import json
import pytest
from data.database import db_session, init_db
from governance.base import Inspector
from governance.schemas import Finding


@pytest.fixture(autouse=True)
def _setup():
    """Initialize database before each test and clean up after."""
    init_db()
    yield


def test_governance_findings_table_created():
    """Verify governance_findings table is created by init_db()."""
    with db_session() as conn:
        # Query the table to ensure it exists and has the expected schema
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='governance_findings'")
        table = cursor.fetchone()
        assert table is not None, "governance_findings table was not created"

        # Check that all required columns exist
        cursor = conn.execute("PRAGMA table_info(governance_findings)")
        columns = {row["name"] for row in cursor.fetchall()}
        required_columns = {
            "id", "agent", "level", "scope", "status", "severity",
            "evidence", "local_recommendation", "escalate_to", "created_at"
        }
        assert required_columns.issubset(columns), \
            f"Missing columns: {required_columns - columns}"


def test_governance_findings_indices_created():
    """Verify indices on governance_findings are created."""
    with db_session() as conn:
        # Check for idx_gov_scope
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_gov_scope'"
        )
        assert cursor.fetchone() is not None, "idx_gov_scope index not found"

        # Check for idx_gov_status
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_gov_status'"
        )
        assert cursor.fetchone() is not None, "idx_gov_status index not found"


def test_inspector_record_writes_finding():
    """Verify Inspector.record() writes a Finding to the database."""

    class TestInspector(Inspector):
        name = "TestInspector"
        level = "L0"

        def inspect(self, scope, ctx):
            return None  # Not tested here

    inspector = TestInspector()

    # Create a Finding
    finding = Finding(
        agent="TestInspector",
        level="L0",
        scope="test:1",
        status="PASS",
        severity="INFO",
        evidence=["data_point_1", "data_point_2"],
        local_recommendation="All checks passed",
        escalate_to=None,
    )

    # Record it
    row_id = inspector.record(finding)
    assert row_id > 0, "record() should return a positive row ID"

    # Read it back
    with db_session() as conn:
        cursor = conn.execute(
            "SELECT * FROM governance_findings WHERE id = ?", (row_id,)
        )
        row = cursor.fetchone()

    assert row is not None, "Finding was not written to database"
    assert row["agent"] == "TestInspector"
    assert row["level"] == "L0"
    assert row["scope"] == "test:1"
    assert row["status"] == "PASS"
    assert row["severity"] == "INFO"
    assert row["local_recommendation"] == "All checks passed"
    assert row["escalate_to"] is None

    # Verify evidence was JSON-encoded
    evidence = json.loads(row["evidence"])
    assert evidence == ["data_point_1", "data_point_2"]


def test_inspector_record_with_dict_evidence():
    """Verify Inspector.record() handles dict evidence."""

    class TestInspector(Inspector):
        name = "TestInspector"
        level = "L1"

        def inspect(self, scope, ctx):
            return None

    inspector = TestInspector()

    finding = Finding(
        agent="TestInspector",
        level="L1",
        scope="backtest_run:42",
        status="WARN",
        severity="WARNING",
        evidence={"sharpe": 0.65, "threshold": 0.80, "gap": -0.15},
        local_recommendation="Sharpe below threshold",
        escalate_to="L2",
    )

    row_id = inspector.record(finding)

    with db_session() as conn:
        cursor = conn.execute(
            "SELECT evidence, escalate_to FROM governance_findings WHERE id = ?",
            (row_id,)
        )
        row = cursor.fetchone()

    assert row is not None
    evidence = json.loads(row["evidence"])
    assert evidence["sharpe"] == 0.65
    assert evidence["threshold"] == 0.80
    assert row["escalate_to"] == "L2"


def test_inspector_record_with_none_evidence():
    """Verify Inspector.record() handles None evidence."""

    class TestInspector(Inspector):
        name = "TestInspector"
        level = "L0"

        def inspect(self, scope, ctx):
            return None

    inspector = TestInspector()

    finding = Finding(
        agent="TestInspector",
        level="L0",
        scope="idea:99",
        status="PASS",
        severity="INFO",
        evidence=None,
        local_recommendation="No issues",
        escalate_to=None,
    )

    row_id = inspector.record(finding)

    with db_session() as conn:
        cursor = conn.execute(
            "SELECT evidence FROM governance_findings WHERE id = ?", (row_id,)
        )
        row = cursor.fetchone()

    assert row is not None
    assert row["evidence"] is None


def test_finding_dataclass_creation():
    """Verify Finding dataclass can be instantiated with all fields."""
    finding = Finding(
        agent="DemoInspector",
        level="L2",
        scope="portfolio:main",
        status="FAIL",
        severity="BLOCKER",
        evidence=["violation_1", "violation_2"],
        local_recommendation="Halt trading",
        escalate_to="CTO",
    )

    assert finding.agent == "DemoInspector"
    assert finding.level == "L2"
    assert finding.scope == "portfolio:main"
    assert finding.status == "FAIL"
    assert finding.severity == "BLOCKER"
    assert finding.escalate_to == "CTO"
