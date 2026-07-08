"""Bursa transaction cost model — config/settings.py is the single source of truth."""
import numpy as np
import pandas as pd
import pytest

from config.settings import (
    bursa_trade_cost, bursa_slippage_tier,
    BURSA_STAMP_DUTY_CAP_MYR, BURSA_CLEARING_CAP_MYR, BURSA_BOARD_LOT,
)
from agents.portfolio_executor.portfolio_executor import PortfolioExecutor


def test_stamp_duty_caps_at_rm200():
    # RM1M buy: 0.15% would be RM1,500 — must cap at RM200
    cost = bursa_trade_cost(1_000_000, "buy", "BLUE_CHIP")
    # commission 800 + clearing 300 + stamp 200 (capped) + slippage 500
    assert cost == pytest.approx(1800.0)


def test_sell_side_has_no_stamp_duty():
    buy = bursa_trade_cost(100_000, "buy", "BLUE_CHIP")
    sell = bursa_trade_cost(100_000, "sell", "BLUE_CHIP")
    assert buy - sell == pytest.approx(min(100_000 * 0.0015, BURSA_STAMP_DUTY_CAP_MYR))


def test_stamp_duty_uncapped_below_threshold():
    # RM100k buy: 0.15% = RM150 < RM200 cap → uncapped
    buy = bursa_trade_cost(100_000, "buy", "BLUE_CHIP")
    # 80 comm + 30 clearing + 150 stamp + 50 slippage
    assert buy == pytest.approx(310.0)


def test_clearing_caps_at_rm1000():
    # RM5M trade: clearing 0.03% = RM1,500 — must cap at RM1,000
    sell = bursa_trade_cost(5_000_000, "sell", "BLUE_CHIP")
    # 4000 comm + 1000 clearing (capped) + 2500 slippage
    assert sell == pytest.approx(7500.0)


def test_slippage_tiers_ordered():
    small = bursa_trade_cost(50_000, "sell", "SMALL_CAP")
    mid = bursa_trade_cost(50_000, "sell", "MID_CAP")
    blue = bursa_trade_cost(50_000, "sell", "BLUE_CHIP")
    assert small > mid > blue


def test_tier_classification():
    assert bursa_slippage_tier(25_000_000) == "BLUE_CHIP"
    assert bursa_slippage_tier(5_000_000) == "MID_CAP"
    assert bursa_slippage_tier(100_000) == "SMALL_CAP"


def test_board_lot_sizing():
    # RM100k NAV, RM10 stock, 95% alloc → 9,500 shares (95 lots)
    assert PortfolioExecutor.size_shares(100_000, 10.0) == 9500
    # never a partial lot
    assert PortfolioExecutor.size_shares(100_000, 7.77) % BURSA_BOARD_LOT == 0
    # too expensive for one lot → 0
    assert PortfolioExecutor.size_shares(1_000, 100.0) == 0


def test_backtester_uses_asymmetric_rates():
    from agents.backtest_engineer.backtest_engineer import BacktestEngineer
    be = BacktestEngineer()
    idx = pd.date_range("2021-01-01", periods=300, freq="B")
    df = pd.DataFrame({"close": np.linspace(10, 12, 300), "volume": 3_000_000}, index=idx)
    rates = be._cost_rates(df)
    assert rates["tier"] == "BLUE_CHIP"
    assert rates["buy"] > rates["sell"] > 0
