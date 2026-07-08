"""Dual-market research-content fix (2026-07-09): DiversityEngine, ResearchHunter,
KBIngester, and AlphaSeedGenerator previously hardcoded Bursa Malaysia content
regardless of MARKET_MODE — verified live when the crypto daemon ingested
"Web Document Analysis for Companies Listed in Bursa Malaysia" into its own KB.

Bursa checks run in-process (default mode). Crypto checks that need module-level
wiring (settings-bound prompts) run as subprocess snippets, following the
established pattern in tests/test_crypto_mode.py — settings binds one market
per process, so there's no in-process way to exercise crypto content.
"""
import os
import subprocess
import sys
import tempfile

import pytest

from config.markets import load_market_profile
from config import settings

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_crypto(code: str) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        env = {**os.environ, "MARKET_MODE": "crypto",
               "OPENCLAW_RUNTIME_DIR": tmp, "PYTHONPATH": REPO}
        r = subprocess.run([sys.executable, "-c", code], env=env, cwd=REPO,
                           capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, f"stderr:\n{r.stderr[-2000:]}"
        return r.stdout


# ── Profile content: same 9 taxonomy keys, market-appropriate text ───────────

SHARED_ANGLE_KEYS = {
    "price_action", "fundamental", "event_driven", "institutional", "macro",
    "commodity", "sector_rotation", "behavioural", "statistical_modelling",
}


def test_bursa_research_angles_content():
    assert set(settings.RESEARCH_ANGLES.keys()) == SHARED_ANGLE_KEYS
    for angle, data in settings.RESEARCH_ANGLES.items():
        assert data["description"]
        assert len(data["queries"]) >= 2
    assert "Bursa Malaysia equity trading" == settings.RELEVANCE_TARGET


def test_crypto_research_angles_same_keys_different_content():
    crypto = load_market_profile("crypto")
    assert set(crypto.RESEARCH_ANGLES.keys()) == SHARED_ANGLE_KEYS
    for angle, data in crypto.RESEARCH_ANGLES.items():
        assert data["description"]
        assert len(data["queries"]) >= 2
    assert crypto.RELEVANCE_TARGET == "crypto spot trading"


def test_crypto_content_has_no_bursa_wording():
    crypto = load_market_profile("crypto")
    blob = " ".join(
        [crypto.RESEARCH_QUERY_PERSONA, crypto.ALPHA_SEED_SYSTEM,
         crypto.RELEVANCE_TARGET, crypto.RELEVANCE_SCALE] +
        [d["description"] for d in crypto.RESEARCH_ANGLES.values()] +
        [q for d in crypto.RESEARCH_ANGLES.values() for q in d["queries"]]
    ).lower()
    for banned in ("bursa", "klse", "malaysia", "epf", "cpo", "opr"):
        assert banned not in blob, f"crypto research content still mentions '{banned}'"


def test_bursa_content_has_no_crypto_wording():
    blob = " ".join(
        [settings.RESEARCH_QUERY_PERSONA, settings.ALPHA_SEED_SYSTEM,
         settings.RELEVANCE_TARGET, settings.RELEVANCE_SCALE] +
        [d["description"] for d in settings.RESEARCH_ANGLES.values()]
    ).lower()
    # "crypto" appears legitimately in Bursa's irrelevant-tier example
    # ("Australian CFD trading, cryptocurrency, forex pairs" as a rejection
    # example) — that's correct, not a leak. Check for on-topic crypto terms
    # that would indicate crypto content bled into Bursa's angles instead.
    for banned in ("binance", "btc/usdt", "on-chain", "tokenomics", "halving"):
        assert banned not in blob


# ── Angle keyword taxonomy stays the same 9 keys, per-market content ─────────

def test_angle_keywords_share_taxonomy_keys():
    crypto = load_market_profile("crypto")
    assert set(settings.ANGLE_KEYWORDS.keys()) == SHARED_ANGLE_KEYS
    assert set(crypto.ANGLE_KEYWORDS.keys()) == SHARED_ANGLE_KEYS
    # institutional angle: EPF/GLC language for Bursa, whale/ETF for crypto
    assert "epf" in settings.ANGLE_KEYWORDS["institutional"]
    assert "whale wallet" in crypto.ANGLE_KEYWORDS["institutional"]
    assert "epf" not in crypto.ANGLE_KEYWORDS["institutional"]


# ── Consumer wiring: DiversityEngine reads the active profile ────────────────

def test_diversity_engine_uses_bursa_profile_in_process():
    from knowledge.ingestion.diversity_engine import DiversityEngine
    balance = DiversityEngine().check_balance()
    assert set(balance["all_angles"]) == SHARED_ANGLE_KEYS
    assert balance["total_docs"] >= 0  # doesn't crash, reads settings correctly


def test_diversity_engine_and_kb_ingester_crypto_wiring():
    out = run_crypto("""
import json
from data.database import init_db
init_db()
from knowledge.ingestion.diversity_engine import DiversityEngine
from knowledge.ingestion.kb_ingester import KBIngester, SYSTEM as KB_SYSTEM
from knowledge.ingestion.research_hunter import ResearchHunter
from config import settings

balance = DiversityEngine().check_balance()
angles_ok = set(balance["all_angles"]) == {
    "price_action", "fundamental", "event_driven", "institutional", "macro",
    "commodity", "sector_rotation", "behavioural", "statistical_modelling",
}

# fill_angle looks up settings.RESEARCH_ANGLES for a known crypto-only angle
data = settings.RESEARCH_ANGLES.get("institutional")
angle_is_crypto = "whale" in data["description"].lower() or "etf" in data["description"].lower()

print(json.dumps({
    "angles_ok": angles_ok,
    "angle_is_crypto": angle_is_crypto,
    "kb_system_has_crypto": "crypto" in KB_SYSTEM.lower(),
    "kb_system_no_bursa": "bursa" not in KB_SYSTEM.lower(),
    "relevance_target": settings.RELEVANCE_TARGET,
}))
""")
    r = __import__("json").loads(out.strip().splitlines()[-1])
    assert r["angles_ok"] is True
    assert r["angle_is_crypto"] is True
    assert r["kb_system_has_crypto"] is True
    assert r["kb_system_no_bursa"] is True
    assert r["relevance_target"] == "crypto spot trading"


def test_alpha_seeds_feasibility_and_kb_hunt_enabled_in_crypto_mode():
    out = run_crypto("""
import json
from knowledge.ingestion.alpha_seeds import is_market_feasible, SYSTEM
from config import settings
usdt_ok, _ = is_market_feasible({"ticker": "BTC/USDT", "title": "", "hypothesis": "", "factor_formula": ""})
kl_bad, kl_reason = is_market_feasible({"ticker": "1155.KL", "title": "", "hypothesis": "", "factor_formula": ""})
perp_bad, perp_reason = is_market_feasible({"ticker": "BTC/USDT", "title": "perpetual leverage strategy",
                                            "hypothesis": "", "factor_formula": ""})
print(json.dumps({
    "usdt_ok": usdt_ok, "kl_bad": kl_bad, "perp_bad": perp_bad,
    "system_no_bursa": "bursa" not in SYSTEM.lower(),
    "kb_hunt_enabled": "kb_hunt" in settings.ENABLED_JOBS,
    "alpha_seeds_enabled": "alpha_seeds" in settings.ENABLED_JOBS,
}))
""")
    r = __import__("json").loads(out.strip().splitlines()[-1])
    assert r["usdt_ok"] is True
    assert r["kl_bad"] is False
    assert r["perp_bad"] is False
    assert r["system_no_bursa"] is True
    assert r["kb_hunt_enabled"] is True
    assert r["alpha_seeds_enabled"] is True


def test_bursa_mode_kb_hunt_still_enabled_control():
    """Inverse control in the current (bursa) process: nothing regressed."""
    assert settings.ENABLED_JOBS is None  # None = all jobs enabled (unchanged)


def test_bursa_alpha_seeds_still_rejects_short_selling():
    """Regression: BLOCKED_MODES reuse didn't loosen the Bursa feasibility filter."""
    from knowledge.ingestion.alpha_seeds import is_market_feasible
    bad, reason = is_market_feasible(
        {"ticker": "1155.KL", "title": "pairs trade strategy", "hypothesis": "", "factor_formula": ""})
    assert bad is False
    assert "infeasible phrase" in reason
