"""Crypto-mode behavior tests (dual-market Phase C).

settings binds one market per process, so these run tiny subprocess snippets
with MARKET_MODE=crypto and a SCRATCH runtime dir (never the dev DB). Slower
than in-process tests, so kept few and high-value.
"""
import json
import os
import subprocess
import sys
import tempfile

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_crypto(code: str) -> str:
    """Run a snippet under MARKET_MODE=crypto with an isolated runtime dir."""
    with tempfile.TemporaryDirectory() as tmp:
        env = {**os.environ, "MARKET_MODE": "crypto",
               "OPENCLAW_RUNTIME_DIR": tmp, "PYTHONPATH": REPO}
        r = subprocess.run([sys.executable, "-c", code], env=env, cwd=REPO,
                           capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, f"stderr:\n{r.stderr[-2000:]}"
        return r.stdout


def test_sandbox_accepts_usdt_rejects_kl_and_perps():
    out = run_crypto("""
import json
from data.database import init_db
init_db()
from pipeline.sandbox import submit_sandbox_idea
ok = submit_sandbox_idea({"title": "CRX btc ma", "hypothesis": "50-day MA cross on Bitcoin, hold weeks",
                          "ticker": "BTC/USDT", "factor_formula": "close crosses above sma(50)"})
kl = submit_sandbox_idea({"title": "CRX kl", "hypothesis": "momentum",
                          "ticker": "1155.KL", "factor_formula": "close above sma(50) uptrend"})
perp = submit_sandbox_idea({"title": "CRX perp", "hypothesis": "long perpetual futures basis",
                            "ticker": "ETH/USDT", "factor_formula": "funding rate positive for days"})
print(json.dumps({"ok": ok["ok"], "status": ok.get("status"),
                  "kl_ok": kl["ok"], "kl_err": kl.get("error", "")[:40],
                  "perp_ok": perp["ok"]}))
""")
    r = json.loads(out.strip().splitlines()[-1])
    assert r["ok"] is True and r["status"] == "pending"
    assert r["kl_ok"] is False and "No valid ticker" in r["kl_err"]
    assert r["perp_ok"] is False


def test_data_quality_does_not_flag_crypto_weekends():
    out = run_crypto("""
import json, numpy as np, pandas as pd
from data.data_quality import compute_data_confidence
idx = pd.date_range("2026-01-01", periods=200, freq="D")   # includes weekends
df = pd.DataFrame({"close": np.linspace(100, 130, 200), "volume": 5_000_000}, index=idx)
dq = compute_data_confidence(df)
print(json.dumps({"missing": dq["missing_day_frac"], "score": dq["confidence_score"]}))
""")
    r = json.loads(out.strip().splitlines()[-1])
    assert r["missing"] == 0.0          # weekends are trading days in crypto
    assert r["score"] >= 90


def test_bursa_mode_still_flags_weekend_gaps_as_ok():
    """Inverse control in the CURRENT (bursa) process: a business-day series has
    no missing days under the business calendar."""
    import numpy as np
    import pandas as pd
    from data.data_quality import compute_data_confidence
    idx = pd.bdate_range("2026-01-01", periods=200)
    df = pd.DataFrame({"close": np.linspace(10, 13, 200), "volume": 3_000_000}, index=idx)
    dq = compute_data_confidence(df)
    assert dq["missing_day_frac"] == 0.0


def test_feasibility_and_prompts_in_crypto_mode():
    out = run_crypto("""
import json
from agents.researcher.strategy_researcher import StrategyResearcher, SYSTEM, GATE0_SYSTEM
from agents.red_blue_team.red_blue_team import RED_SYSTEM, JUDGE_SYSTEM
feas = StrategyResearcher._compute_feasibility(
    {"hypothesis": "50-day breakout on Bitcoin, hold for weeks"}, "BTC/USDT",
    "close crosses above sma(50)")
feas_onchain = StrategyResearcher._compute_feasibility(
    {"hypothesis": "buy when on-chain whale wallets accumulate"}, "BTC/USDT",
    "on-chain flow indicator positive")
print(json.dumps({
    "feas": feas, "feas_onchain": feas_onchain,
    "sys_crypto": "CRYPTO SPOT" in SYSTEM and "BTC/USDT" in SYSTEM,
    "sys_no_bursa": "BURSA MALAYSIA SPECIALIST" not in SYSTEM,
    "gate0_crypto": "crypto" in GATE0_SYSTEM.lower(),
    "red_crypto": "BTC-beta" in RED_SYSTEM,
    "judge_crypto": "perpetuals" in JUDGE_SYSTEM,
}))
""")
    r = json.loads(out.strip().splitlines()[-1])
    assert r["feas"] >= 0.8              # clean daily-OHLCV idea on a valid pair
    assert r["feas_onchain"] < r["feas"]  # unavailable data docks the score
    assert all(r[k] for k in ("sys_crypto", "sys_no_bursa", "gate0_crypto",
                              "red_crypto", "judge_crypto"))


def test_concierge_resolves_crypto_names_and_prompt_switches():
    out = run_crypto("""
import json
from data.database import init_db
init_db()
from agents.concierge.concierge_agent import ConciergeAgent, TOOLS, _system_prompt
c = ConciergeAgent()
res = c._tool_resolve_tickers(["bitcoin", "DeFi"])
prompt = _system_prompt()
submit_desc = next(t for t in TOOLS if t["name"] == "submit_strategy_idea")["description"]
print(json.dumps({
    "btc": res["matches"]["bitcoin"],
    "defi_has_uni": "UNI/USDT" in res["matches"]["DeFi"],
    "prompt_crypto": "BTC/USDT" in prompt and "CRYPTO SPOT MARKET" in prompt,
    "prompt_no_klci": "1155.KL" not in prompt,
    "tools_crypto": "BTC/USDT" in submit_desc,
    "guardrail": "human-only decision" in prompt,
}))
""")
    r = json.loads(out.strip().splitlines()[-1])
    assert r["btc"] == ["BTC/USDT"]
    assert r["defi_has_uni"] is True
    assert all(r[k] for k in ("prompt_crypto", "prompt_no_klci", "tools_crypto", "guardrail"))


def test_fractional_fill_simulation_in_crypto_mode():
    out = run_crypto("""
import json
from agents.portfolio_executor.execution_simulator import simulate_fill, pre_trade_check
fill = simulate_fill(100_000, 100_000.0, 5_000_000_000, 0.95)   # $100k NAV, $100k BTC
check_ok = pre_trade_check("BTC/USDT", 100_000,
                           {"close": 100_000.0, "adv_value": 5_000_000_000}, 95.0, 0)
check_kl = pre_trade_check("1155.KL", 100_000,
                           {"close": 10.0, "adv_value": 5_000_000_000}, 95.0, 0)
print(json.dumps({"units": fill["units"], "status": fill["status"],
                  "ok": check_ok["passed"], "kl": check_kl["passed"]}))
""")
    r = json.loads(out.strip().splitlines()[-1])
    assert r["units"] == pytest.approx(0.95, abs=0.001)   # fractional BTC, not 0
    assert r["status"] == "FILLED"
    assert r["ok"] is True
    assert r["kl"] is False               # .KL is not a crypto instrument
