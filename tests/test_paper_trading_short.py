"""WS3: paper trading short positions + funding accrual (crypto perps).

settings binds one market per process (module-level `from config.settings
import ...` copies each name at import time in portfolio_executor.py AND
execution_simulator.py), so — same as test_crypto_mode.py — these run tiny
subprocess snippets with MARKET_MODE=crypto and a scratch runtime dir rather
than in-process monkeypatching, which can't reach constants already copied
into other modules' namespaces.
"""
import json
import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_crypto(code: str) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        env = {**os.environ, "MARKET_MODE": "crypto",
               "OPENCLAW_RUNTIME_DIR": tmp, "PYTHONPATH": REPO}
        r = subprocess.run([sys.executable, "-c", code], env=env, cwd=REPO,
                           capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, f"stderr:\n{r.stderr[-2000:]}"
        return r.stdout


_SETUP = """
import asyncio, json
from data.database import init_db
init_db()
from agents.portfolio_executor.portfolio_executor import PortfolioExecutor
ex = PortfolioExecutor()
IDEA_ID = 1
import sqlite3
from data.database import db_session
with db_session() as conn:
    conn.execute(
        "INSERT OR IGNORE INTO alpha_ideas (id, slug, title, ticker, stage, status) "
        "VALUES (?, 'test-short', 'Short test', 'BTC/USDT', 'stage4a', 'active')", (IDEA_ID,))
_price = {"v": 100_000.0}
def fake_latest_bar(self, ticker, interval="1d"):
    return {"close": _price["v"], "date": "2026-07-08", "adv_value": 5_000_000_000.0}
PortfolioExecutor._latest_bar = fake_latest_bar
"""


def test_bursa_rejects_short_direction_control():
    """Control (no subprocess needed — CURRENT process is bursa by default)."""
    import asyncio
    from data.database import init_db
    from agents.portfolio_executor.portfolio_executor import PortfolioExecutor
    init_db()
    ex = PortfolioExecutor()
    result = asyncio.run(ex.paper_entry(999_888_777, "1155.KL", "short"))
    assert "error" in result and "long-only" in result["error"]


def test_short_entry_stores_negative_units_and_pnl_signs():
    out = run_crypto(_SETUP + """
async def main():
    entry = await ex.paper_entry(IDEA_ID, "BTC/USDT", "short")
    assert "error" not in entry, entry
    units_negative = entry["units"] < 0

    # price falls 10% -> short profits
    _price["v"] = 90_000.0
    fall = await ex.paper_exit(entry["trade_id"])

    entry2 = await ex.paper_entry(IDEA_ID, "BTC/USDT", "short")
    _price["v"] = 110_000.0  # price rises 10% -> short loses
    rise = await ex.paper_exit(entry2["trade_id"])

    print(json.dumps({
        "direction": entry["direction"], "units_negative": units_negative,
        "profit_on_fall": fall["pnl"] > 0, "loss_on_rise": rise["pnl"] < 0,
    }))
asyncio.run(main())
""")
    r = json.loads(out.strip().splitlines()[-1])
    assert r["direction"] == "short"
    assert r["units_negative"] is True
    assert r["profit_on_fall"] is True
    assert r["loss_on_rise"] is True


def test_idea_cash_sign_agnostic_after_short_round_trip():
    out = run_crypto(_SETUP + """
async def main():
    from config.settings import PAPER_CAPITAL_MYR
    entry = await ex.paper_entry(IDEA_ID, "BTC/USDT", "short")
    _price["v"] = 95_000.0
    exit_res = await ex.paper_exit(entry["trade_id"])
    cash = ex._idea_cash(IDEA_ID)
    print(json.dumps({"cash": cash, "expected": PAPER_CAPITAL_MYR + exit_res["pnl"]}))
asyncio.run(main())
""")
    r = json.loads(out.strip().splitlines()[-1])
    assert abs(r["cash"] - r["expected"]) < 0.02


def test_funding_accrues_and_signs_correctly():
    out = run_crypto(_SETUP + """
async def main():
    long_entry = await ex.paper_entry(IDEA_ID, "BTC/USDT", "long")
    ex._accrue_funding(long_entry["trade_id"], long_entry["units"], long_entry["entry_price"])
    with db_session() as conn:
        row = conn.execute("SELECT funding_paid FROM paper_trades WHERE id=?",
                           (long_entry["trade_id"],)).fetchone()
    print(json.dumps({"long_funding_paid": row["funding_paid"]}))
asyncio.run(main())
""")
    r = json.loads(out.strip().splitlines()[-1])
    # Default AVG_FUNDING_RATE_PER_INTERVAL is positive -> a long PAYS (cost, positive charge).
    assert r["long_funding_paid"] > 0


def test_no_funding_accrual_when_interval_none():
    out = run_crypto(_SETUP + """
import agents.portfolio_executor.portfolio_executor as pe_mod
pe_mod.FUNDING_INTERVAL_HOURS = None
async def main():
    entry = await ex.paper_entry(IDEA_ID, "BTC/USDT", "long")
    r = await ex.daily_update(IDEA_ID, "BTC/USDT",
        {"signal_type": "dsl", "dsl": {"entry": {"leaf": "rsi", "period": 14, "below": 100}}})
    with db_session() as conn:
        row = conn.execute("SELECT funding_paid FROM paper_trades WHERE id=?",
                           (entry["trade_id"],)).fetchone()
    print(json.dumps({"funding_paid": row["funding_paid"] or 0}))
asyncio.run(main())
""")
    r = json.loads(out.strip().splitlines()[-1])
    assert r["funding_paid"] == 0
