"""pipeline/throughput_guard.py: the global daily cap on revisit_scan +
finding_driven_candidates (P1-4, 2026-07-13 self-audit) — these two
mechanisms insert directly at stage2/pending, bypassing gate0, so an
unbounded rate raises the deflated-Sharpe hurdle for every idea in the
system (see docs/auto_ideation_throughput.md). Also covers the daemon-level
skip behavior and the funnel report's new summary section.

House pattern: share the local DB, isolate by a distinctive slug prefix,
purge before + after.
"""
import asyncio

import pytest

from data.database import db_session, init_db
from pipeline import throughput_guard

_PREFIX = "test-tg-999"


def _purge():
    with db_session() as conn:
        conn.execute("DELETE FROM alpha_ideas WHERE slug LIKE ?", (f"%{_PREFIX}%",))


@pytest.fixture(autouse=True)
def _setup():
    init_db()
    _purge()
    yield
    _purge()


def _insert_idea(slug):
    with db_session() as conn:
        conn.execute(
            "INSERT INTO alpha_ideas (slug, title, stage, status) "
            "VALUES (?, 'test', 'stage2', 'pending')", (slug,))


def _run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ── auto_submissions_today ──────────────────────────────────────────────────

def test_counts_revisit_and_finding_driven_and_regime_scoped_slugs():
    _insert_idea(f"revisit-{_PREFIX}-1-20260713000001")
    _insert_idea(f"auto-finding-{_PREFIX}-2-rsi")
    _insert_idea(f"rg-{_PREFIX}-bull-3")
    assert throughput_guard.auto_submissions_today() >= 3


def test_does_not_count_organic_or_seed_ideas():
    before = throughput_guard.auto_submissions_today()
    _insert_idea(f"seed-2026-07-13-{_PREFIX}-unrelated")
    _insert_idea(f"2026-07-13-{_PREFIX}-organic-idea")
    after = throughput_guard.auto_submissions_today()
    assert after == before


def test_cap_reached_flips_once_threshold_hit(monkeypatch):
    # The local dev DB may already carry real revisit-/auto-finding-/rg-
    # rows from actual daemon activity today — set the cap relative to the
    # CURRENT count rather than assuming a clean slate of zero.
    baseline = throughput_guard.auto_submissions_today()
    monkeypatch.setattr(throughput_guard, "AUTO_IDEAS_DAILY_CAP", baseline + 2)
    assert not throughput_guard.auto_ideation_cap_reached()
    _insert_idea(f"revisit-{_PREFIX}-a-1")
    _insert_idea(f"revisit-{_PREFIX}-b-2")
    assert throughput_guard.auto_ideation_cap_reached()


# ── daemon integration: jobs skip cleanly when the cap is reached ──────────

def test_revisit_scan_job_skips_when_cap_reached(monkeypatch):
    from scripts.research_daemon import ResearchDaemon
    monkeypatch.setattr(throughput_guard, "AUTO_IDEAS_DAILY_CAP", 1)
    _insert_idea(f"revisit-{_PREFIX}-cap-1")

    def fail_if_called():
        raise AssertionError("run_revisit_scan should not be called when the cap is reached")
    monkeypatch.setattr("pipeline.revisit.run_revisit_scan", fail_if_called)

    daemon = ResearchDaemon(scan_interval=60)
    _run_async(daemon._process_revisit_scan())
    assert "revisit_scan" in daemon._job_last_run  # still marked, so it doesn't spin every cycle


def test_finding_driven_candidates_job_skips_when_cap_reached(monkeypatch):
    from scripts.research_daemon import ResearchDaemon
    monkeypatch.setattr(throughput_guard, "AUTO_IDEAS_DAILY_CAP", 1)
    _insert_idea(f"auto-finding-{_PREFIX}-cap-1")

    def fail_if_called():
        raise AssertionError(
            "run_finding_driven_candidates should not be called when the cap is reached")
    monkeypatch.setattr("pipeline.finding_candidates.run_finding_driven_candidates",
                        fail_if_called)

    daemon = ResearchDaemon(scan_interval=60)
    _run_async(daemon._process_finding_driven_candidates())
    assert "finding_driven_candidates" in daemon._job_last_run


# ── funnel report includes the new auto-mechanism summary ──────────────────

def test_funnel_report_includes_auto_mechanism_summary():
    from scripts.research_daemon import ResearchDaemon
    _insert_idea(f"revisit-{_PREFIX}-report-1")
    _insert_idea(f"auto-finding-{_PREFIX}-report-2")

    daemon = ResearchDaemon(scan_interval=60)
    counts = daemon._auto_mechanism_counts(24)
    assert counts["revisit"] >= 1
    assert counts["finding_driven"] >= 1
    assert "quota_cap" in counts and "quota_used_today" in counts
    assert "leaf_attempts" in counts and "leaf_approved" in counts and "leaf_cost" in counts
