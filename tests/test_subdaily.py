"""Sub-daily timeframes (crypto 15m floor): profile constants, gates, engine
scaling, DQ, paper-trading slots. Bursa parity is pinned throughout."""
import json

import numpy as np
import pandas as pd
import pytest

from tests.test_crypto_mode import run_crypto


# ── Profile constants / parity ────────────────────────────────────────────────

def test_bursa_profile_reproduces_daily_only():
    import config.settings as s
    assert s.ALLOWED_TIMEFRAMES == ["1d", "1wk"]
    assert s.bars_per_day("1d") == 1.0
    assert s.FETCH_DAYS_BY_INTERVAL == {"1d": 1825, "1wk": 1825}
    assert s.CACHE_STALENESS_HOURS_BY_INTERVAL["1d"] == 12.0
    assert "intraday" in s.BLOCKED_MODES          # Bursa keeps the hard block
    assert "hourly" in s.FEASIBILITY_DOCK_KEYWORDS


def test_crypto_profile_allows_subdaily():
    out = run_crypto("""
import json
import config.settings as s
print(json.dumps({"tfs": s.ALLOWED_TIMEFRAMES,
                  "bpd_1h": s.bars_per_day("1h"), "bpd_15m": s.bars_per_day("15m"),
                  "blocked_intraday": "intraday" in s.BLOCKED_MODES,
                  "blocked_1m": "1 minute" in s.BLOCKED_MODES,
                  "dock": s.FEASIBILITY_DOCK_KEYWORDS}))
""")
    r = json.loads(out.strip().splitlines()[-1])
    assert r["tfs"] == ["15m", "1h", "4h", "1d", "1wk"]
    assert r["bpd_1h"] == 24.0 and r["bpd_15m"] == 96.0
    assert r["blocked_intraday"] is False and r["blocked_1m"] is True
    assert "hourly" not in r["dock"]


# ── Sandbox timeframe validation ─────────────────────────────────────────────

def test_bursa_sandbox_rejects_subdaily_timeframe():
    from data.database import init_db
    init_db()
    from pipeline.sandbox import submit_sandbox_idea
    r = submit_sandbox_idea({"title": "SBX tf test", "hypothesis": "hourly momentum",
                             "ticker": "1155.KL", "timeframe": "1h",
                             "factor_formula": "close above sma(50)"})
    assert r["ok"] is False
    assert "1d, 1wk" in r["error"]


def test_crypto_sandbox_accepts_1h_rejects_1m():
    out = run_crypto("""
import json
from data.database import init_db
init_db()
from pipeline.sandbox import submit_sandbox_idea
ok_1h = submit_sandbox_idea({"title": "CRX 1h zscore", "hypothesis": "hourly z-score mean reversion on Bitcoin",
                             "ticker": "BTC/USDT", "timeframe": "1h",
                             "factor_formula": "enter long when z-score(20) < -2, exit when z-score > 0"})
bad_1m = submit_sandbox_idea({"title": "CRX 1m", "hypothesis": "minute momentum",
                              "ticker": "BTC/USDT", "timeframe": "1m",
                              "factor_formula": "close above sma(50)"})
print(json.dumps({"ok_1h": ok_1h.get("ok"), "bad_1m": bad_1m.get("ok"),
                  "err_1m": bad_1m.get("error", "")}))
""")
    r = json.loads(out.strip().splitlines()[-1])
    assert r["ok_1h"] is True
    assert r["bad_1m"] is False and "15m" in r["err_1m"]


# ── Engine scaling ────────────────────────────────────────────────────────────

def _synthetic_df(n=600, freq="D", seed=0):
    idx = pd.date_range("2023-01-01", periods=n, freq=freq)
    rng = np.random.RandomState(seed)
    close = pd.Series(100 * np.cumprod(1 + rng.randn(n) * 0.01), index=idx)
    return pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99,
                         "close": close, "volume": 2_000_000.0})


def test_cost_rates_daily_equivalent_adv_neutral_at_1d():
    from agents.backtest_engineer.backtest_engineer import BacktestEngineer
    eng = BacktestEngineer()
    df = _synthetic_df()
    r_default = eng._cost_rates(df)
    r_1d = eng._cost_rates(df, "1d")
    assert r_default["adv_value_myr"] == r_1d["adv_value_myr"]  # parity
    assert r_default["tier"] == r_1d["tier"]


def test_cost_rates_scales_subdaily_adv_in_crypto():
    out = run_crypto("""
import json
import numpy as np, pandas as pd
from agents.backtest_engineer.backtest_engineer import BacktestEngineer
eng = BacktestEngineer()
idx = pd.date_range("2024-01-01", periods=3000, freq="h")
close = pd.Series(100.0, index=idx)
# $10M traded per 1h bar → $240M/day: BLUE_CHIP daily, SMALL_CAP if unscaled
df = pd.DataFrame({"close": close, "volume": 100_000.0})
r = eng._cost_rates(df, "1h")
print(json.dumps({"tier": r["tier"], "adv": r["adv_value_myr"]}))
""")
    r = json.loads(out.strip().splitlines()[-1])
    assert r["tier"] == "BLUE_CHIP"
    assert r["adv"] == pytest.approx(240_000_000.0, rel=0.01)


def test_subdaily_classification_and_min_trades():
    from agents.backtest_engineer.backtest_engineer import BacktestEngineer
    # In Bursa mode there IS no sub-daily interval, but the classifier logic
    # itself is numeric-first and shared:
    assert BacktestEngineer._MIN_TRADES["SUBDAILY"] == 100
    assert BacktestEngineer._SHARPE_THRESHOLDS["SUBDAILY"] == 1.1
    # Keyword-based INTRADAY behavior unchanged for daily-bar text:
    assert BacktestEngineer.classify_holding_period(
        "1d", "intraday scalping", "") == "INTRADAY"
    out = run_crypto("""
from agents.backtest_engineer.backtest_engineer import BacktestEngineer
print(BacktestEngineer.classify_holding_period("15m", "zscore reversion", ""))
print(BacktestEngineer.classify_holding_period("4h", "", ""))
""")
    lines = out.strip().splitlines()
    assert lines[-2] == "SUBDAILY" and lines[-1] == "SUBDAILY"


# ── Data quality: sub-daily missing bars ─────────────────────────────────────

def test_dq_missing_bar_detection_subdaily_crypto():
    out = run_crypto("""
import json
import numpy as np, pandas as pd
from data.data_quality import compute_data_confidence
idx = pd.date_range("2024-01-01", periods=1000, freq="h")
close = pd.Series(100 + np.cumsum(np.random.RandomState(0).randn(1000)), index=idx)
full = pd.DataFrame({"close": close, "volume": 1e6})
gapped = full.iloc[::2]  # drop every other bar → ~50% missing
dq_full = compute_data_confidence(full, "1h")
dq_gap = compute_data_confidence(gapped, "1h")
print(json.dumps({"full": dq_full["missing_day_frac"], "gap": dq_gap["missing_day_frac"]}))
""")
    r = json.loads(out.strip().splitlines()[-1])
    assert r["full"] < 0.02
    assert r["gap"] > 0.4


def test_dq_bursa_daily_path_unchanged():
    from data.data_quality import compute_data_confidence
    df = _synthetic_df(300, "B")   # clean business-day series
    dq = compute_data_confidence(df, "1d")
    assert dq["missing_day_frac"] == 0.0
    # weekly still skips missing-day scoring entirely (historical behavior)
    dqw = compute_data_confidence(_synthetic_df(100, "W"), "1wk")
    assert dqw["missing_day_frac"] == 0.0


# ── Paper trading slots + funding pro-ration ─────────────────────────────────

def test_equity_slot_formats():
    from datetime import datetime
    from agents.portfolio_executor.portfolio_executor import equity_slot
    ts = datetime(2026, 7, 9, 13, 47)
    assert equity_slot("1d", ts) == "2026-07-09"
    assert equity_slot("1wk", ts) == "2026-07-09"
    assert equity_slot("1h", ts) == "2026-07-09T13:00"
    assert equity_slot("15m", ts) == "2026-07-09T13:45"
    assert equity_slot("4h", ts) == "2026-07-09T12:00"


def test_funding_proration_subdaily():
    out = run_crypto("""
import json
from data.database import init_db, db_session
init_db()
from agents.portfolio_executor.portfolio_executor import PortfolioExecutor
with db_session() as conn:
    conn.execute("INSERT OR IGNORE INTO alpha_ideas (id, slug, title, ticker, stage, status) VALUES (999901,'fund-test','FUND test','BTC/USDT','stage4a','active')")
    conn.execute("INSERT INTO paper_trades (idea_id, pair, direction, units, entry_price, entry_cost, funding_paid, status) VALUES (999901,'BTC/USDT','long',1.0,50000,50,0,'open')")
    tid = conn.execute("SELECT id FROM paper_trades WHERE idea_id=999901").fetchone()["id"]
ex = PortfolioExecutor()
ex._accrue_funding(tid, 1.0, 50000, hours=24.0)
with db_session() as conn:
    day = conn.execute("SELECT funding_paid FROM paper_trades WHERE id=?", (tid,)).fetchone()["funding_paid"]
    conn.execute("UPDATE paper_trades SET funding_paid=0 WHERE id=?", (tid,))
ex._accrue_funding(tid, 1.0, 50000, hours=1.0)
with db_session() as conn:
    hour = conn.execute("SELECT funding_paid FROM paper_trades WHERE id=?", (tid,)).fetchone()["funding_paid"]
    conn.execute("DELETE FROM paper_trades WHERE id=?", (tid,))
print(json.dumps({"day": day, "hour": hour}))
""")
    r = json.loads(out.strip().splitlines()[-1])
    # 24h = 3 settlements × 0.0001 × 50000 = 15; 1h = 15/24
    assert r["day"] == pytest.approx(15.0, rel=0.01)
    assert r["hour"] == pytest.approx(15.0 / 24.0, rel=0.02)
