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


def test_sandbox_accepts_usdt_rejects_kl_and_bad_perps():
    out = run_crypto("""
import json
from data.database import init_db
init_db()
from pipeline.sandbox import submit_sandbox_idea
ok = submit_sandbox_idea({"title": "CRX btc ma", "hypothesis": "50-day MA cross on Bitcoin, hold weeks",
                          "ticker": "BTC/USDT", "factor_formula": "close crosses above sma(50)"})
kl = submit_sandbox_idea({"title": "CRX kl", "hypothesis": "momentum",
                          "ticker": "1155.KL", "factor_formula": "close above sma(50) uptrend"})
# WS3: a clean short thesis on a perp is now ACCEPTED (long/short is the point).
short_ok = submit_sandbox_idea({"title": "CRX eth short", "hypothesis": "short ETH on breakdown below sma(50)",
                                "ticker": "ETH/USDT", "factor_formula": "close crosses below sma(50)"})
# a multi-leg spread/pairs structure is still hard-blocked (BLOCKED_MODES) —
# unlike the funding-rate case below, this IS an outright rejection.
spread_bad = submit_sandbox_idea({"title": "CRX spread", "hypothesis": "spread trade between two majors",
                                  "ticker": "SOL/USDT", "factor_formula": "close above sma(50)"})
print(json.dumps({"ok": ok["ok"], "status": ok.get("status"),
                  "kl_ok": kl["ok"], "kl_err": kl.get("error", "")[:40],
                  "short_ok": short_ok["ok"], "spread_bad": spread_bad["ok"]}))
""")
    r = json.loads(out.strip().splitlines()[-1])
    assert r["ok"] is True and r["status"] == "pending"
    assert r["kl_ok"] is False and "No valid ticker" in r["kl_err"]
    assert r["short_ok"] is True
    assert r["spread_bad"] is False


def test_crypto_refusal_wording_is_long_short():
    """The blocked-mode refusal must not claim 'long-only' in crypto mode
    (ALLOW_SHORT=True) — it should state the long/short capability."""
    out = run_crypto("""
import json
from data.database import init_db
init_db()
from pipeline.sandbox import submit_sandbox_idea, _INFEASIBLE_HINT
spread = submit_sandbox_idea({"title": "CRX spread word", "hypothesis": "spread trade between two majors",
                              "ticker": "SOL/USDT", "factor_formula": "close above sma(50)"})
print(json.dumps({"err": spread.get("error", ""), "hint": _INFEASIBLE_HINT}))
""")
    r = json.loads(out.strip().splitlines()[-1])
    assert "long-only" not in r["err"]
    assert "long/short" in r["err"]
    assert "short-selling" not in r["hint"]


def test_crypto_templates_are_long_short():
    """WS3 follow-through: crypto-mode event→formula templates must express
    both directions (short_entry-able phrasing), in price/indicator terms."""
    out = run_crypto("""
import json
import scripts.event_watcher as ew
t = ew.FACTOR_FORMULA_TEMPLATES
print(json.dumps({
    "funding_short": "enter short" in t["funding_spike"],
    "funding_long": "enter long" in t["funding_spike"],
    "unlock_short": "short" in t["unlock"].lower(),
    "basis_directional": "enter short" in t["basis_dislocation"] and "enter long" in t["basis_dislocation"],
    "btc_symmetric": "short" in t["btc_move"],
    "listing_long_only": "short" not in t["listing"].lower(),
    "price_based_entry": "close < sma(10)" in t["funding_spike"],
}))
""")
    r = json.loads(out.strip().splitlines()[-1])
    assert all(r.values()), r


def test_crypto_concierge_prompt_lists_crypto_techniques():
    """The concierge system prompt's arsenal index must carry the CRYPTO
    technique set (and its new tool must return detail for a crypto key)."""
    out = run_crypto("""
import json
from agents.concierge.concierge_agent import _system_prompt, ConciergeAgent
p = _system_prompt()
c = ConciergeAgent.__new__(ConciergeAgent)  # tool method needs no client
detail = c._tool_suggest_techniques({"key": "funding_rate_carry"})
print(json.dumps({"arsenal": "TECHNIQUE ARSENAL" in p,
                  "crypto_key": "funding_rate_carry" in p,
                  "bursa_key": "epf_flow" in p,
                  "detail_ok": "TECHNIQUE:" in detail.get("techniques", "")}))
""")
    r = json.loads(out.strip().splitlines()[-1])
    assert r["arsenal"] is True
    assert r["crypto_key"] is True
    assert r["bursa_key"] is False
    assert r["detail_ok"] is True


def test_funding_rate_formula_docked_not_hard_blocked_at_sandbox():
    """A formula needing a HISTORICAL funding-rate series (not backtestable —
    no such DSL leaf/data column exists) is docked by the deterministic
    feasibility score but not hard-blocked at the shallow sandbox pre-check;
    real enforcement is at DSL-parse time (signal_dsl: unrepresentable ->
    REJECTED, never silently genericized). This test documents that gap
    explicitly rather than asserting an outcome the code doesn't produce."""
    out = run_crypto("""
import json
from agents.researcher.strategy_researcher import StrategyResearcher
clean = StrategyResearcher._compute_feasibility(
    {"hypothesis": "50-day MA cross on Bitcoin"}, "BTC/USDT", "close crosses above sma(50)")
funding = StrategyResearcher._compute_feasibility(
    {"hypothesis": "long perpetual futures basis"}, "SOL/USDT", "funding rate positive for days")
print(json.dumps({"clean": clean, "funding": funding}))
""")
    r = json.loads(out.strip().splitlines()[-1])
    assert r["funding"] < r["clean"]  # docked relative to a clean formula
    assert r["funding"] >= 0.6         # NOT hard-blocked at this stage (documented gap)


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
    # WS3: SYSTEM now frames crypto as a perpetuals (long/short) specialist,
    # not spot-only.
    "sys_crypto": "CRYPTO PERPETUALS" in SYSTEM and "BTC/USDT" in SYSTEM,
    "sys_no_bursa": "BURSA MALAYSIA SPECIALIST" not in SYSTEM,
    "gate0_crypto": "crypto" in GATE0_SYSTEM.lower(),
    "red_crypto": "BTC-beta" in RED_SYSTEM,
    "judge_crypto": "leverage" in JUDGE_SYSTEM and "funding" in JUDGE_SYSTEM.lower(),
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
    # WS3: the concierge prompt now frames crypto as a perpetual long/short
    # market (MARKET_BRIEF text changed from "CRYPTO SPOT MARKET" to
    # "CRYPTO PERPETUAL MARKET").
    "prompt_crypto": "BTC/USDT" in prompt and "CRYPTO PERPETUAL MARKET" in prompt,
    "prompt_no_klci": "1155.KL" not in prompt,
    "prompt_allows_short": "Long AND short are both supported" in prompt,
    "tools_crypto": "BTC/USDT" in submit_desc,
    "tools_allow_short": "LONG OR SHORT" in submit_desc,
    "guardrail": "human-only decision" in prompt,
}))
""")
    r = json.loads(out.strip().splitlines()[-1])
    assert r["btc"] == ["BTC/USDT"]
    assert r["defi_has_uni"] is True
    assert all(r[k] for k in ("prompt_crypto", "prompt_no_klci", "prompt_allows_short",
                              "tools_crypto", "tools_allow_short", "guardrail"))


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
