"""Regression: sub-daily backtests lost their equity curve + trade blotter.

2026-07-14 (idea 232, ETH 4h): backtest_series rows were keyed on
`str(d)[:10]` (calendar date only). On a sub-daily interval, the 6 bars of a
4h day all collapse to the same YYYY-MM-DD, colliding on
backtest_series' UNIQUE(idea_id, date). The insert raised inside the persist
try-block, which ALSO holds the backtest_trades insert — so both the equity
curve and the trade blotter were silently lost for every sub-daily (crypto)
backtest. Fix: engine.series_date_key keeps the intraday time for sub-daily
intervals and stays date-only (byte-identical) for daily/weekly.
"""
import pandas as pd
import pytest

from agents.backtest_engineer.engine import series_date_key
from data.database import db_session, init_db

_TEST_IDEA_ID = 999_777_014


def test_subdaily_keys_are_distinct_within_a_day():
    # Six 4h bars on the SAME calendar day must produce six distinct keys.
    day = pd.date_range("2024-01-15", periods=6, freq="4h")
    keys = [series_date_key(t, "4h") for t in day]
    assert len(set(keys)) == 6, keys
    # ...and daily/weekly stay date-only (Bursa output unchanged).
    assert series_date_key(pd.Timestamp("2024-01-15 00:00:00"), "1d") == "2024-01-15"
    assert series_date_key(pd.Timestamp("2024-01-15 00:00:00"), "1wk") == "2024-01-15"


def test_subdaily_rows_insert_without_unique_collision():
    """The real failure was a UNIQUE(idea_id, date) violation on insert — prove
    a full sub-daily day's worth of rows now persists."""
    init_db()
    with db_session() as conn:
        conn.execute("DELETE FROM backtest_series WHERE idea_id=?", (_TEST_IDEA_ID,))
    try:
        bars = pd.date_range("2024-01-15", periods=12, freq="4h")  # 2 days × 6
        rows = [(_TEST_IDEA_ID, series_date_key(d, "4h"), 0.01 * i, 0.0, 0.0,
                 1 if i >= 8 else 0) for i, d in enumerate(bars)]
        with db_session() as conn:
            conn.executemany(
                "INSERT INTO backtest_series "
                "(idea_id, date, strategy_pct, benchmark_pct, drawdown_pct, is_oos) "
                "VALUES (?,?,?,?,?,?)", rows)
            n = conn.execute(
                "SELECT COUNT(*) AS n FROM backtest_series WHERE idea_id=?",
                (_TEST_IDEA_ID,)).fetchone()["n"]
        assert n == 12
    finally:
        with db_session() as conn:
            conn.execute("DELETE FROM backtest_series WHERE idea_id=?", (_TEST_IDEA_ID,))
