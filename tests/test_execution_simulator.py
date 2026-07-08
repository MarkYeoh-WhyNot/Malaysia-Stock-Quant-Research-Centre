"""Phase 6.1/6.2: execution simulator + pre-trade checks. Pure functions, no DB."""
from agents.portfolio_executor.execution_simulator import simulate_fill, pre_trade_check
from config.settings import BURSA_BOARD_LOT


def test_full_fill_when_capacity_ample():
    # RM100k NAV, RM10 stock, deep liquidity (RM30M ADV) → full 95% fill
    r = simulate_fill(100_000, 10.0, 30_000_000, 0.95)
    assert r["status"] == "FILLED"
    assert r["units"] == 9500


def test_partial_fill_when_capacity_binds():
    # Thin stock: ADV RM200k → 5% capacity/day = RM10k, well under the RM95k
    # NAV-sized order → partial fill capped by capacity, still lot-rounded.
    r = simulate_fill(100_000, 10.0, 200_000, 0.95)
    assert r["status"] == "PARTIAL_FILL"
    assert r["units"] < r["requested_units"]
    assert r["units"] % BURSA_BOARD_LOT == 0


def test_failed_fill_when_nav_too_small():
    r = simulate_fill(500, 100.0, 30_000_000, 0.95)
    assert r["status"] == "FAILED"
    assert r["units"] == 0


def test_failed_fill_on_invalid_price():
    assert simulate_fill(100_000, 0, 30_000_000, 0.95)["status"] == "FAILED"


def test_pre_trade_check_passes_clean_order():
    bar = {"close": 10.0, "adv_value": 30_000_000, "date": "2026-06-01"}
    check = pre_trade_check("1155.KL", 100_000, bar, dq_confidence=95.0,
                            unresolved_corp_actions=0)
    assert check["passed"] is True
    assert check["reasons"] == []


def test_pre_trade_check_rejects_invalid_ticker():
    bar = {"close": 10.0, "adv_value": 30_000_000}
    check = pre_trade_check("AAPL", 100_000, bar, None, 0)
    assert check["passed"] is False
    assert any("ticker" in r for r in check["reasons"])


def test_pre_trade_check_rejects_illiquid_and_low_confidence():
    bar = {"close": 10.0, "adv_value": 100_000}  # below RM500k floor
    check = pre_trade_check("1155.KL", 100_000, bar, dq_confidence=50.0,
                            unresolved_corp_actions=2)
    assert check["passed"] is False
    assert any("liquidity" in r for r in check["reasons"])
    assert any("confidence" in r for r in check["reasons"])
    assert any("corporate action" in r for r in check["reasons"])


def test_pre_trade_check_rejects_missing_price_data():
    check = pre_trade_check("1155.KL", 100_000, None, None, 0)
    assert check["passed"] is False
    assert any("price data" in r for r in check["reasons"])
