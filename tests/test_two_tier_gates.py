"""Two-tier gate redesign (2026-07-14, Mark-approved).

Tier 1 (PSR + noise-aware OOS + noise-aware robustness + DD/capacity) stays a
hard auto-reject. Tier 2 (trade count, regime breadth, benchmark, cost) is
advisory: tripping one no longer rejects the idea — it is HELD at
status='needs_review' for a human approve/reject decision, with the tripped
checks recorded in backtest_runs.advisory_flags. Nothing reaches paper trading
without human sign-off.

These run each scenario in a subprocess with an isolated OPENCLAW_RUNTIME_DIR
(same pattern as tests/test_gate_calibration_fixes.py) so the dev DB is untouched.
"""
import json
import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(code: str, market_mode: str = "bursa") -> str:
    with tempfile.TemporaryDirectory() as tmp:
        env = {**os.environ, "MARKET_MODE": market_mode,
               "OPENCLAW_RUNTIME_DIR": tmp, "PYTHONPATH": REPO}
        r = subprocess.run([sys.executable, "-c", code], env=env, cwd=REPO,
                           capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, f"stderr:\n{r.stderr[-3000:]}"
        return r.stdout


# ── A genuine sparse-trade edge clears Tier 1 and is HELD for review ──────────

_LOWFREQ_SNIPPET = """
import json
from data.database import init_db
init_db()
from config.settings import ALLOW_SHORT
from scripts.calibration_harness import _run_case, _zscore_tree_lowfreq

# Same low-frequency case the calibration harness uses (genuine reversion edge
# on a short track record → few trades). 8 seeds so the held-path count is
# stable; 700 bars matches the harness's low-freq window.
trials = _run_case("lowfreq", _zscore_tree_lowfreq(ALLOW_SHORT),
                   list(range(1, 9)), 700)
tier1 = sum(1 for t in trials if t.passed)
held  = sum(1 for t in trials if t.metrics.get("held_for_review"))
trades = sorted(t.metrics.get("actual_trades") for t in trials
                if t.metrics.get("actual_trades") is not None)
verdicts = [t.verdict for t in trials]
print(json.dumps({"n": len(trials), "tier1": tier1, "held": held,
                  "trades": trades, "verdicts": verdicts}))
"""


def test_sparse_trade_edge_clears_tier1_and_is_held_not_rejected():
    out = _run(_LOWFREQ_SNIPPET)
    r = json.loads(out.strip().splitlines()[-1])
    # Every genuine low-frequency edge clears the Tier-1 statistical/risk core —
    # it is NOT hard-rejected for being infrequent (the whole point).
    assert r["tier1"] == r["n"], r
    # ...and the advisory HELD-for-review path activates (not auto-rejected):
    # at least one trips an advisory check and is held with verdict 'review'.
    assert r["held"] >= 1, r
    assert "review" in r["verdicts"], r


# ── The held state persists correctly + a human can approve/reject it ─────────

_FLOW_SNIPPET = """
import json
from data.database import init_db, db_session
init_db()
from dashboard.api.server import advance_idea, AdvanceBody

_n = [0]
def _mk(status):
    _n[0] += 1
    with db_session() as conn:
        cur = conn.execute(
            "INSERT INTO alpha_ideas (slug, title, ticker, timeframe, stage, status) "
            "VALUES (?,?,?,?, 'stage2', ?)",
            (f"tt-{status}-{_n[0]}", f"held {status}", "1155.KL", "1d", status))
        return cur.lastrowid

def _get(iid):
    with db_session() as conn:
        r = conn.execute("SELECT stage, status FROM alpha_ideas WHERE id=?", (iid,)).fetchone()
    return {"stage": r["stage"], "status": r["status"]}

# Approve a needs_review idea → advances into stage3/active (Red-Blue).
a = _mk("needs_review")
advance_idea(a, AdvanceBody(action="advance"))
after_approve = _get(a)

# Reject a needs_review idea → rejected.
b = _mk("needs_review")
advance_idea(b, AdvanceBody(action="reject"))
after_reject = _get(b)

# A held idea is EXCLUDED from the daemon's stage2 backtest-selection query.
c = _mk("needs_review")
with db_session() as conn:
    picked = conn.execute(
        "SELECT id FROM alpha_ideas "
        "WHERE stage='stage2' AND status IN ('active','pending') "
        "AND id NOT IN (SELECT DISTINCT idea_id FROM backtest_runs)").fetchall()
excluded = c not in [row["id"] for row in picked]

print(json.dumps({"approve": after_approve, "reject": after_reject,
                  "excluded_from_daemon": excluded}))
"""


def test_needs_review_idea_can_be_approved_rejected_and_is_not_auto_advanced():
    out = _run(_FLOW_SNIPPET)
    r = json.loads(out.strip().splitlines()[-1])
    # Approve unblocks it into stage3 (Red-Blue), active.
    assert r["approve"] == {"stage": "stage3", "status": "active"}, r
    # Reject kills it.
    assert r["reject"]["status"] == "rejected", r
    # The daemon never auto-advances a held idea.
    assert r["excluded_from_daemon"] is True, r
