"""Global daily cap on the two deterministic, zero-LLM-cost auto-ideation
mechanisms: revisit_scan and finding_driven_candidates (2026-07-13
self-audit, P1-4). Each runs 4x/day (MAX_REVISITS_PER_CYCLE=3 in
pipeline/revisit.py, MAX_CANDIDATES_PER_CYCLE=4 in
pipeline/finding_candidates.py — the latter covers BOTH the plain
`auto-finding-*` and the regime-scoped `rg-*` variants against the same
cap) — worst case 12 + 16 = 28 ideas/day combined. Both insert directly at
stage2/pending with hardcoded novelty=logic=feasibility=0.7, BYPASSING
gate0 entirely, so unlike organic ideas nearly every one reaches a real
backtest. More backtested ideas raises recent_trial_count() (see
agents/backtest_engineer/gates.py) -> n_trials -> the deflated-Sharpe
hurdle SR* for EVERY idea in the system, not just these — a real (if
logarithmically dampened) cost nobody had priced before this pass. See
docs/auto_ideation_throughput.md for the full arithmetic.

Deliberately excludes alpha_seeds/screener_ideas/organic generation, which
have their own, separate, much lower per-run limits and still go through
gate0 — see the design doc for why those aren't part of this cap.
"""
from config.settings import AUTO_IDEAS_DAILY_CAP
from data.database import db_session

_AUTO_SLUG_PATTERNS = ("revisit-%", "auto-finding-%", "rg-%")


def auto_submissions_today() -> int:
    """Count of today's alpha_ideas rows from the mechanisms this cap
    governs (revisit / finding-driven / regime-scoped), UTC calendar day —
    matches get_agent_daily_spend's day boundary convention elsewhere."""
    with db_session() as conn:
        clause = " OR ".join("slug LIKE ?" for _ in _AUTO_SLUG_PATTERNS)
        row = conn.execute(
            f"SELECT COUNT(*) AS n FROM alpha_ideas "
            f"WHERE date(created_at) = date('now') AND ({clause})",
            _AUTO_SLUG_PATTERNS).fetchone()
        return row["n"]


def auto_ideation_cap_reached() -> bool:
    return auto_submissions_today() >= AUTO_IDEAS_DAILY_CAP
