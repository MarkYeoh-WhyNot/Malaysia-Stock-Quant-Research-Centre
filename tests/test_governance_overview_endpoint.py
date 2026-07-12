"""Tests for GET /api/governance/overview — the Command Centre governance
health strip (F2). Separate from /api/departments/overview (pipeline
activity); this rolls up L0 inspector findings per L1 department via
governance.managers.summarize_all_departments()."""

from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from governance.managers import DEPARTMENT_AGENT_MAP
from data.database import db_session, init_db
from dashboard.api.server import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_findings():
    """Governance findings tests share the project DB (same pattern as
    tests/test_governance_managers.py) — clean up before and after so this
    test's rows never leak into other tests or linger in the DB."""
    init_db()
    with db_session() as conn:
        conn.execute("DELETE FROM governance_findings")
    yield
    with db_session() as conn:
        conn.execute("DELETE FROM governance_findings")


def _insert(agent, status, severity, evidence=None, local_recommendation=None, created_at=None):
    with db_session() as conn:
        if created_at is not None:
            conn.execute(
                """INSERT INTO governance_findings
                   (agent, level, scope, status, severity, evidence, local_recommendation, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (agent, "L0", "test:scope", status, severity, evidence, local_recommendation, created_at),
            )
        else:
            conn.execute(
                """INSERT INTO governance_findings
                   (agent, level, scope, status, severity, evidence, local_recommendation)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (agent, "L0", "test:scope", status, severity, evidence, local_recommendation),
            )


def test_all_five_departments_present_even_with_zero_findings():
    resp = client.get("/api/governance/overview")
    assert resp.status_code == 200
    data = resp.json()
    assert "departments" in data
    assert "as_of" in data
    depts = {d["department"] for d in data["departments"]}
    assert depts == set(DEPARTMENT_AGENT_MAP.keys())
    assert len(data["departments"]) == 5

    # Zero findings everywhere -> every department HEALTHY, no top_blocker.
    for d in data["departments"]:
        assert d["status"] == "HEALTHY"
        assert d["findings_count"] == 0
        assert d["blocking_issues"] == 0
        assert d["top_blocker"] is None
        assert d["recommendations"] == []


def test_response_shape_per_department():
    resp = client.get("/api/governance/overview")
    data = resp.json()
    expected_keys = {
        "department", "status", "findings_count", "blocking_issues",
        "top_blocker", "recommendations",
    }
    for d in data["departments"]:
        assert set(d.keys()) == expected_keys


def test_mixed_severities_produce_correct_status_and_top_blocker():
    # Data Integrity: PASS only -> HEALTHY, no blocker
    _insert("SourceHealthInspector", "PASS", "INFO")

    # Backtest Fidelity: WARNING only -> WARNING, no blocker (blocking_issues == 0)
    _insert("PnLConsistencyInspector", "WARN", "WARNING",
            local_recommendation="Investigate PnL drift.")

    # Parser Honesty: BLOCKER with a recommendation -> ALERT, top_blocker == recommendation
    _insert("DSLRepresentabilityChecker", "FAIL", "BLOCKER",
            evidence='{"issue": "unknown leaf"}',
            local_recommendation="Register the missing DSL leaf before re-running.")

    resp = client.get("/api/governance/overview")
    assert resp.status_code == 200
    data = resp.json()
    by_name = {d["department"]: d for d in data["departments"]}

    di = by_name["Data Integrity"]
    assert di["status"] == "HEALTHY"
    assert di["findings_count"] == 1
    assert di["blocking_issues"] == 0
    assert di["top_blocker"] is None

    bf = by_name["Backtest Fidelity"]
    assert bf["status"] == "WARNING"
    assert bf["findings_count"] == 1
    assert bf["blocking_issues"] == 0
    assert bf["top_blocker"] is None

    ph = by_name["Parser Honesty"]
    assert ph["status"] == "ALERT"
    assert ph["findings_count"] == 1
    assert ph["blocking_issues"] == 1
    assert ph["top_blocker"] == "Register the missing DSL leaf before re-running."

    # Untouched departments stay HEALTHY/empty.
    pr = by_name["Portfolio Risk"]
    assert pr["status"] == "HEALTHY"
    assert pr["findings_count"] == 0
    assert pr["top_blocker"] is None

    pt = by_name["Paper Trading"]
    assert pt["status"] == "HEALTHY"
    assert pt["findings_count"] == 0
    assert pt["top_blocker"] is None


def test_blocker_without_recommendation_falls_back_to_truncated_evidence():
    long_evidence = "x" * 300
    _insert("ShadowNAVInspector", "FAIL", "BLOCKER", evidence=long_evidence,
            local_recommendation=None)

    resp = client.get("/api/governance/overview")
    data = resp.json()
    pr = next(d for d in data["departments"] if d["department"] == "Portfolio Risk")

    assert pr["status"] == "ALERT"
    assert pr["blocking_issues"] == 1
    assert pr["top_blocker"] is not None
    assert pr["top_blocker"].endswith("…")
    assert len(pr["top_blocker"]) == 161  # 160 chars + ellipsis


def test_most_recent_blocker_wins_when_multiple_exist():
    # Explicit, distinct timestamps (well within the 24h lookback, computed
    # relative to "now") so ORDER BY created_at DESC is deterministic — two
    # inserts in the same wall-clock second would otherwise tie.
    now = datetime.utcnow()
    _insert("KillSwitchInspector", "FAIL", "BLOCKER",
            local_recommendation="Older blocker — should not be surfaced.",
            created_at=(now - timedelta(hours=2)).isoformat())
    _insert("ConcentrationCorrelationInspector", "FAIL", "BLOCKER",
            local_recommendation="Newest blocker — should be surfaced.",
            created_at=(now - timedelta(hours=1)).isoformat())

    resp = client.get("/api/governance/overview")
    data = resp.json()
    pr = next(d for d in data["departments"] if d["department"] == "Portfolio Risk")

    assert pr["blocking_issues"] == 2
    assert pr["top_blocker"] == "Newest blocker — should be surfaced."


def test_departments_overview_endpoint_still_works_independently():
    """Guard against regressions: the existing pipeline-activity endpoint
    must be untouched by the new governance endpoint's addition."""
    resp = client.get("/api/departments/overview")
    assert resp.status_code == 200
    data = resp.json()
    assert "departments" in data
    ids = {d["id"] for d in data["departments"]}
    assert ids == {
        "alpha_research", "data_engineering", "quant_research", "red_blue",
        "execution", "market_intelligence", "knowledge_base",
    }
