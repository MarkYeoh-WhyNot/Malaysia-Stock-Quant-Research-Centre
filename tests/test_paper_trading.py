"""Paper trading end-to-end: signal-driven entry, mark-to-market, NAV-based evaluation.

Uses the local dev database (data/openclaw.db) with a sentinel idea id and
cleans up after itself. Price data is stubbed — no network access.
"""
import asyncio

import numpy as np
import pandas as pd
import pytest

from config.settings import PAPER_CAPITAL_MYR, BURSA_BOARD_LOT
from data.database import db_session, init_db
from agents.portfolio_executor.portfolio_executor import PortfolioExecutor

TEST_IDEA_ID = 987_654_321


@pytest.fixture()
def executor(monkeypatch):
    init_db()
    _cleanup()
    with db_session() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO alpha_ideas (id, slug, title, ticker, stage, status) "
            "VALUES (?, 'test-paper-trading-e2e', 'Paper trading E2E test', "
            "'1155.KL', 'stage4a', 'active')",
            (TEST_IDEA_ID,))
    ex = PortfolioExecutor()
    state = {"close": 10.00}

    def fake_latest_bar(self, ticker):
        return {"close": state["close"], "date": "2026-07-08",
                "adv_value": 30_000_000.0}

    monkeypatch.setattr(PortfolioExecutor, "_latest_bar", fake_latest_bar)
    yield ex, state
    _cleanup()


def _cleanup():
    with db_session() as conn:
        conn.execute("DELETE FROM paper_trades WHERE idea_id=?", (TEST_IDEA_ID,))
        conn.execute("DELETE FROM paper_equity WHERE idea_id=?", (TEST_IDEA_ID,))
        conn.execute("DELETE FROM pipeline_events WHERE idea_id=?", (TEST_IDEA_ID,))
        conn.execute("DELETE FROM gate_decisions WHERE idea_id=?", (TEST_IDEA_ID,))
        conn.execute("DELETE FROM alpha_ideas WHERE id=?", (TEST_IDEA_ID,))


def test_entry_fills_at_close_with_board_lots(executor):
    ex, state = executor
    result = asyncio.run(ex.paper_entry(TEST_IDEA_ID, "1155.KL"))
    assert "error" not in result
    assert result["entry_price"] == 10.00
    assert result["units"] % BURSA_BOARD_LOT == 0
    assert result["units"] == 9500  # 95% of RM100k at RM10
    assert result["entry_cost"] > 0

    # duplicate entry is refused
    dup = asyncio.run(ex.paper_entry(TEST_IDEA_ID, "1155.KL"))
    assert "error" in dup


def test_short_entries_rejected(executor):
    ex, _ = executor
    result = asyncio.run(ex.paper_entry(TEST_IDEA_ID, "1155.KL", direction="short"))
    assert "error" in result


def test_mark_to_market_and_exit_pnl(executor):
    ex, state = executor
    entry = asyncio.run(ex.paper_entry(TEST_IDEA_ID, "1155.KL"))

    # price rises 5% → NAV should rise accordingly
    state["close"] = 10.50
    mtm = ex.mark_to_market(TEST_IDEA_ID)
    expected_nav = (PAPER_CAPITAL_MYR - entry["units"] * 10.0 - entry["entry_cost"]
                    + entry["units"] * 10.50)
    assert mtm["nav"] == pytest.approx(expected_nav, abs=0.01)

    exit_res = asyncio.run(ex.paper_exit(entry["trade_id"]))
    assert "error" not in exit_res
    gross = entry["units"] * (10.50 - 10.00)
    assert exit_res["pnl"] == pytest.approx(
        gross - entry["entry_cost"] - exit_res["exit_cost"], abs=0.01)
    assert exit_res["pnl"] < gross  # costs always bite


def test_evaluation_uses_nav_series(executor):
    ex, state = executor
    asyncio.run(ex.paper_entry(TEST_IDEA_ID, "1155.KL"))

    # Fabricate a 40-day NAV history with a gentle uptrend
    rng = np.random.RandomState(1)
    navs = PAPER_CAPITAL_MYR * np.cumprod(1 + rng.randn(40) * 0.003 + 0.002)
    dates = pd.date_range("2026-05-20", periods=40, freq="D")
    with db_session() as conn:
        for d, nav in zip(dates, navs):
            conn.execute(
                "INSERT OR REPLACE INTO paper_equity (idea_id, date, nav) VALUES (?,?,?)",
                (TEST_IDEA_ID, d.strftime("%Y-%m-%d"), float(nav)))

    result = asyncio.run(ex.evaluate_paper_performance(TEST_IDEA_ID))
    assert result["days_tracked"] >= 40
    assert result["total_days"] >= 30
    assert result["sharpe"] != 0.0
    assert 0 <= result["max_drawdown"] < 1
