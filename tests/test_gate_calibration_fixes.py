"""Pins for the two calibration-driven gate fixes (2026-07-10, Mark-approved).

1. Trade-count gate counts trades over the FULL backtest window
   (train+val+test), not the test slice alone — the old test-slice count made
   MEDIUM_TERM (10-60d holds) structurally unpassable: max 25 trades fit in a
   ~252-bar test slice, below the 30-trade minimum.

2. The corp-action DQ penalty only applies where corporate actions exist
   (HAS_CORPORATE_ACTIONS profile constant). Crypto has no splits/dividends —
   a >25% bar on a volatile alt is a real move, not a data anomaly; the
   penalty was hard-rejecting whole pairs (SOL/USDT: 59.9/100) at the DQ door.
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run_mode(market_mode: str, code: str) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        env = {**os.environ, "MARKET_MODE": market_mode,
               "OPENCLAW_RUNTIME_DIR": tmp, "PYTHONPATH": REPO}
        r = subprocess.run([sys.executable, "-c", code], env=env, cwd=REPO,
                           capture_output=True, text=True, timeout=180)
        assert r.returncode == 0, f"stderr:\n{r.stderr[-2000:]}"
        return r.stdout


# ── Fix 1: full-window trade counting ────────────────────────────────────────

_TRADE_COUNT_SNIPPET = """
import json
import numpy as np
import pandas as pd
from data.database import init_db, db_session
init_db()
from agents.backtest_engineer.backtest_engineer import BacktestEngineer
from scripts.calibration_harness import (ou_series, _ohlcv, _business_index,
                                          _insert_idea)
from config.settings import MARKET_CALENDAR, TICKER_EXAMPLE, DEFAULT_SYMBOLS

# A slow mean-reverter: enough trades over 5yr, but too few in the test slice
# alone. period=60 z-score on a slow OU → sparse round trips.
n = 2000
idx = _business_index(n, MARKET_CALENDAR)
ticker = TICKER_EXAMPLE.split()[0]
win = _ohlcv(ou_series(n, 3, kappa=0.05, sigma=0.02), idx, 5e8)
bench = {s: _ohlcv(ou_series(n, 100 + i, kappa=0.01, sigma=0.015), idx, 5e8)
         for i, s in enumerate(DEFAULT_SYMBOLS)}

tree = {"entry": {"leaf": "zscore", "period": 60, "below": -1.2},
        "exit":  {"leaf": "zscore", "period": 60, "above": 0.0}}

eng = BacktestEngineer()
eng._fetch_prices = lambda sym, interval="1d", days=1825: (
    win if sym.split(",")[0].strip() == ticker
    else bench.get(sym.split(",")[0].strip(), win)).copy()
eng._parse_factor = lambda f, t, h: {"signal_type": "dsl", "dsl": tree,
                                      "representable": True}
idea_id = _insert_idea(ticker, tree)
r = eng.backtest_idea(idea_id)
with db_session() as conn:
    row = conn.execute(
        "SELECT total_trades, trade_count, holding_period_class "
        "FROM backtest_runs WHERE idea_id=? ORDER BY id DESC LIMIT 1",
        (idea_id,)).fetchone()
print("RESULT " + json.dumps({
    "full_window_trades": r.get("actual_trades"),
    "test_slice_trades": row["total_trades"] if row else None,
    "db_trade_count": row["trade_count"] if row else None,
    "hp_class": row["holding_period_class"] if row else None,
    "trade_count_pass": r.get("trade_count_pass"),
}))
"""


def test_trade_count_uses_full_window():
    """actual_trades must be the whole-backtest sum: strictly greater than the
    test-slice count for a strategy that trades throughout, and the gate
    decision must follow the full-window figure."""
    out = _run_mode("bursa", _TRADE_COUNT_SNIPPET)
    line = [l for l in out.splitlines() if l.startswith("RESULT ")][-1]
    res = json.loads(line[len("RESULT "):])
    assert res["full_window_trades"] is not None, res
    # Full-window count strictly exceeds the test slice's — proves summation.
    assert res["full_window_trades"] > res["test_slice_trades"], res
    # DB trade_count column stores the full-window figure the gate used.
    assert res["db_trade_count"] == res["full_window_trades"], res
    # Gate decision consistent with the full-window count and class minimum.
    min_by_class = {"INTRADAY": 100, "SUBDAILY": 100, "SHORT_TERM": 50,
                    "MEDIUM_TERM": 30, "LONG_TERM": 15}
    expected = res["full_window_trades"] >= min_by_class.get(res["hp_class"], 30)
    assert res["trade_count_pass"] == expected, res


# ── Fix 2: corp-action DQ penalty is market-aware ────────────────────────────

_DQ_SNIPPET = """
import json
import numpy as np
import pandas as pd
from data.database import init_db
init_db()
from agents.backtest_engineer.backtest_engineer import BacktestEngineer

# Clean 24/7 daily series with one genuine 35% crash bar (crypto reality).
n = 900
idx = pd.date_range("2022-01-01", periods=n, freq="D")
rng = np.random.default_rng(5)
close = 100 * np.exp(np.cumsum(0.01 * rng.standard_normal(n)))
close[450:] *= 0.65   # single -35% gap, real move
df = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99,
                   "close": close, "volume": np.full(n, 1e9)}, index=idx)

eng = BacktestEngineer()
res = eng._data_quality_gate(0, "TEST/USDT", df, "1d")
print("RESULT " + json.dumps({"score": res["confidence_score"],
                               "passed": res["passed"],
                               "notes": res.get("notes", "")}))
"""


def test_crypto_dq_ignores_large_real_moves():
    """In crypto mode a 35% bar must NOT be penalised as a corp action."""
    out = _run_mode("crypto", _DQ_SNIPPET)
    line = [l for l in out.splitlines() if l.startswith("RESULT ")][-1]
    res = json.loads(line[len("RESULT "):])
    assert "corp-action" not in (res["notes"] or ""), res
    assert res["score"] >= 80, f"clean crypto series should score high: {res}"
    assert res["passed"] is True, res


def test_bursa_dq_still_flags_gaps():
    """Bursa keeps the corp-action penalty — same series must be dinged."""
    out = _run_mode("bursa", _DQ_SNIPPET)
    line = [l for l in out.splitlines() if l.startswith("RESULT ")][-1]
    res = json.loads(line[len("RESULT "):])
    assert "corp-action" in (res["notes"] or ""), res


def test_profiles_define_has_corporate_actions():
    from config.markets import bursa, crypto
    assert bursa.HAS_CORPORATE_ACTIONS is True
    assert crypto.HAS_CORPORATE_ACTIONS is False
