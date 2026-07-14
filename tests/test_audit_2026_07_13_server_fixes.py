"""Regression tests for the 2026-07-13 weekly-audit-corner fixes to
dashboard/api/server.py (self-audit follow-up task 5; see docs/audit_log.md).

Covers: gate_thresholds drift (was frozen at pre-2026-07-10 values), the
red_blue advances/conditionals double-count, the /advance endpoint writing a
fabricated audit-log entry for an already-terminal stage, and the
market_intelligence ticker_overlap column-name bug.
"""
from fastapi.testclient import TestClient

from config.settings import GATE_CONFIG
from data.database import db_session, init_db
from dashboard.api.server import app

client = TestClient(app)

_PREFIX = "test-audit0713-"


def _purge():
    with db_session() as conn:
        ids = [r["id"] for r in conn.execute(
            "SELECT id FROM alpha_ideas WHERE slug LIKE ?", (_PREFIX + "%",)).fetchall()]
        for iid in ids:
            conn.execute("DELETE FROM pipeline_events WHERE idea_id=?", (iid,))
            conn.execute("DELETE FROM gate_decisions WHERE idea_id=?", (iid,))
            conn.execute("DELETE FROM alpha_ideas WHERE id=?", (iid,))
        conn.execute("DELETE FROM market_events WHERE source=?", (_PREFIX + "src",))


import pytest


@pytest.fixture(autouse=True)
def _setup():
    init_db()
    _purge()
    yield
    _purge()


def _insert_idea(slug, stage, status="active"):
    with db_session() as conn:
        cur = conn.execute(
            "INSERT INTO alpha_ideas (slug, title, ticker, stage, status) "
            "VALUES (?, ?, '1155.KL', ?, ?)",
            (slug, slug, stage, status))
        return cur.lastrowid


def test_gate_thresholds_reflect_live_gate_config():
    resp = client.get("/api/system/direction")
    assert resp.status_code == 200
    gt = resp.json()["gate_thresholds"]

    assert gt["gate0"]["logic"] == GATE_CONFIG.gate0_min_logic_score
    assert gt["gate0"]["data_quality"] == GATE_CONFIG.gate0_min_data_quality
    assert gt["gate0"]["overfitting_risk_max"] == GATE_CONFIG.gate0_max_overfitting_risk
    assert gt["psr_confidence_test"] == GATE_CONFIG.psr_confidence_test
    assert gt["train_val_gap_max"] == GATE_CONFIG.stage3_max_train_val_gap
    assert gt["stage4a_min_sharpe"] == GATE_CONFIG.stage4a_min_sharpe
    assert gt["stage4a_max_drawdown"] == GATE_CONFIG.stage4a_max_drawdown
    # The old dict's dead fields (pre-2026-07-10 fixed-Sharpe gates) must be gone.
    assert "stage2_sharpe" not in gt
    assert "stage4a_sharpe" not in gt
    assert "stage4a_max_dd" not in gt


def test_advance_endpoint_rejects_already_terminal_stage():
    idea_id = _insert_idea(_PREFIX + "terminal", stage="stage5")
    resp = client.post(f"/api/pipeline/ideas/{idea_id}/advance", json={"action": "advance"})
    assert resp.status_code == 400

    # No fabricated audit-trail entries should have been written.
    with db_session() as conn:
        pe = conn.execute(
            "SELECT COUNT(*) AS n FROM pipeline_events WHERE idea_id=?", (idea_id,)
        ).fetchone()["n"]
        gd = conn.execute(
            "SELECT COUNT(*) AS n FROM gate_decisions WHERE idea_id=?", (idea_id,)
        ).fetchone()["n"]
    assert pe == 0
    assert gd == 0


def test_advance_endpoint_still_advances_a_normal_stage():
    idea_id = _insert_idea(_PREFIX + "normal", stage="stage2")
    resp = client.post(f"/api/pipeline/ideas/{idea_id}/advance", json={"action": "advance"})
    assert resp.status_code == 200
    with db_session() as conn:
        row = conn.execute("SELECT stage FROM alpha_ideas WHERE id=?", (idea_id,)).fetchone()
    assert row["stage"] == "stage3"


def test_red_blue_conditional_advance_not_double_counted():
    idea_id = _insert_idea(_PREFIX + "condadv", stage="stage3")
    with db_session() as conn:
        conn.execute(
            "INSERT INTO pipeline_events (idea_id,stage,event_type,agent,notes) "
            "VALUES (?, 'stage3', 'advanced', 'RedBlueTeam', ?)",
            (idea_id, '{"verdict": "conditional"}'))

    resp = client.get(f"/api/departments/red_blue/{idea_id}")
    assert resp.status_code == 200
    summary = resp.json()["summary"]
    assert summary["total_debates"] == 1
    assert summary["conditionals"] == 1
    # A conditional-advance must NOT also be counted as a plain advance.
    assert summary["advances"] == 0


def test_market_intelligence_ticker_overlap_matches_affected_tickers():
    idea_id = _insert_idea(_PREFIX + "ticker-overlap", stage="stage2")
    with db_session() as conn:
        conn.execute(
            "INSERT INTO market_events (event_id, source, ticker, event_type, headline, "
            "affected_tickers) VALUES (?, ?, '1155.KL', 'news', 'Test multi-ticker event', ?)",
            (_PREFIX + "evt1", _PREFIX + "src", "5347.KL,1155.KL"))

    resp = client.get(f"/api/departments/market_intelligence/{idea_id}")
    assert resp.status_code == 200
    overlap = resp.json()["ticker_overlap"]
    assert any("Test multi-ticker event" in s for s in overlap)
