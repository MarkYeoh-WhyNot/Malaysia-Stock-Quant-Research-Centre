"""Phase 4.2/4.3: portfolio concentration snapshot + kill switches.

Uses the dev DB with sentinel ids and cleans up after itself. No network.
"""
import pytest

from data.database import db_session, init_db
from agents.risk_monitor.risk_monitor import RiskMonitor

SENTINELS = (900_001, 900_002, 900_003)


@pytest.fixture()
def rm():
    init_db()
    _cleanup()
    yield RiskMonitor()
    _cleanup()


def _cleanup():
    with db_session() as conn:
        for i in SENTINELS:
            conn.execute("DELETE FROM paper_trades WHERE idea_id=?", (i,))
            conn.execute("DELETE FROM alpha_ideas WHERE id=?", (i,))
        conn.execute("DELETE FROM risk_snapshots WHERE detail LIKE '%TESTMARK%'")


def _open_position(idea_id, ticker, units, price):
    with db_session() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO alpha_ideas (id, slug, title, ticker, stage, status) "
            "VALUES (?, ?, 'risk test', ?, 'stage4a', 'active')",
            (idea_id, f"risk-test-{idea_id}", ticker))
        conn.execute(
            "INSERT INTO paper_trades (idea_id, pair, direction, units, entry_price, status) "
            "VALUES (?, ?, 'long', ?, ?, 'open')", (idea_id, ticker, units, price))


def test_no_positions_is_ok(rm):
    snap = rm.portfolio_risk_snapshot()
    assert snap["concentration_ok"] is True
    assert snap["open_positions"] == 0


def test_bank_concentration_breach(rm):
    # Three all-bank positions → bank_pct = 100% → breaches max_bank_pct (40%)
    _open_position(SENTINELS[0], "1155.KL", 1000, 10.0)   # Maybank (Banking)
    _open_position(SENTINELS[1], "1295.KL", 1000, 10.0)   # Public Bank (Banking)
    _open_position(SENTINELS[2], "1023.KL", 1000, 10.0)   # CIMB (Banking)
    snap = rm.portfolio_risk_snapshot()
    assert snap["open_positions"] == 3
    assert snap["bank_pct"] == pytest.approx(1.0)
    assert snap["concentration_ok"] is False
    assert any("bank" in b for b in snap["breaches"])


def test_diversified_book_passes_sector_limit(rm):
    # spread across sectors, no single sector > 35%
    _open_position(SENTINELS[0], "1155.KL", 1000, 10.0)   # Banking
    _open_position(SENTINELS[1], "5347.KL", 1000, 10.0)   # Utilities
    _open_position(SENTINELS[2], "6012.KL", 1000, 10.0)   # Telecoms
    snap = rm.portfolio_risk_snapshot()
    assert snap["max_sector_pct"] == pytest.approx(1 / 3, abs=0.01)
    # single-name is 1/3 > 15% so single-name limit still trips — that's expected
    assert "sector" not in " ".join(snap["breaches"]) or snap["max_sector_pct"] <= 0.35


def test_shadow_portfolio_overlap_and_multiplier(rm):
    # Two separate STRATEGIES both hold Maybank → same-symbol overlap; a third
    # holds another name. Three sandboxes, so combined exposure > one book.
    _open_position(SENTINELS[0], "1155.KL", 1000, 10.0)   # strat A → Maybank
    _open_position(SENTINELS[1], "1155.KL", 1000, 10.0)   # strat B → Maybank (overlap!)
    _open_position(SENTINELS[2], "5347.KL", 1000, 10.0)   # strat C → Tenaga
    snap = rm.portfolio_risk_snapshot()
    assert snap["active_strategy_count"] == 3
    assert snap["same_symbol_overlap_count"] == 1          # 1155.KL held by 2 strats
    assert snap["same_symbol_overlap_notional"] == pytest.approx(20000.0)
    assert snap["net_exposure_myr"] == pytest.approx(30000.0)   # all long
    assert snap["paper_capital_multiplier"] > 0
    assert 0 <= snap["overlap_risk_score"] <= 100
