"""Phase 1.1: date-aware Bursa fee schedule.

Costs are date-dependent (stamp-duty remission 0.15%→0.10% from 2023-07-13). The
versioned fee_schedules table lets a backtest apply the rate in force at the time
rather than a single hardcoded constant.
"""
from data.database import init_db
from data.fee_schedule import get_fee_schedule, bursa_trade_cost_asof
from config.settings import bursa_trade_cost


def setup_module(_):
    init_db()  # ensures fee_schedules exists + seeded on the dev DB


def test_pre_remission_schedule():
    s = get_fee_schedule("2020-01-01")
    assert s["stamp_duty_rate"] == 0.0015
    assert s["stamp_duty_cap"] == 200.0
    assert s["source"].startswith("fee_schedules")


def test_post_remission_schedule():
    s = get_fee_schedule("2024-01-01")
    assert s["stamp_duty_rate"] == 0.0010
    assert s["stamp_duty_cap"] == 1000.0


def test_cost_differs_across_remission_boundary():
    # RM100k buy: pre = 0.15% stamp (RM150), post = 0.10% (RM100) → RM50 cheaper
    pre = bursa_trade_cost_asof(100_000, "buy", "BLUE_CHIP", "2020-01-01")
    post = bursa_trade_cost_asof(100_000, "buy", "BLUE_CHIP", "2024-01-01")
    assert pre - post == 50.0


def test_asof_none_matches_current_constants():
    # No date → current schedule, identical to the constant-based model
    assert bursa_trade_cost_asof(100_000, "buy", "BLUE_CHIP", None) == \
        bursa_trade_cost(100_000, "buy", "BLUE_CHIP")
