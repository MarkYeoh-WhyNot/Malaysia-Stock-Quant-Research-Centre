"""Event-driven revisit job (Phase 5 of the ideation-loop wiring): old
rejected ideas re-open when a regime flips, a new data source lands, or a
new finding contradicts the pattern that killed them.

House pattern: share the local DB, isolate by a slug/title prefix, purge
before + after. Network-touching trigger detection (_current_vol_tercile /
_current_macro_regime) is monkeypatched — these tests exercise the
selection/enqueue/accounting logic, not live data fetches.
"""
import json

import pytest

from data.database import db_session, init_db
from knowledge.graph import store
from pipeline import revisit

_PREFIX = "test-revisit-"


def _purge():
    with db_session() as conn:
        ids = [r["id"] for r in conn.execute(
            "SELECT id FROM alpha_ideas WHERE slug LIKE ? OR title LIKE ?",
            (_PREFIX + "%", "TESTREVISIT%"))]
        for iid in ids:
            for tbl in ("backtest_runs", "optimizer_runs", "gate_decisions",
                       "pipeline_events", "paper_trades"):
                conn.execute(f"DELETE FROM {tbl} WHERE idea_id=?", (iid,))
            conn.execute("DELETE FROM strategy_cemetery WHERE idea_id=?", (iid,))
            conn.execute("DELETE FROM alpha_ideas WHERE id=?", (iid,))
        conn.execute("DELETE FROM revisit_state WHERE key LIKE ? OR key IN "
                     "('vol_regime:benchmark', 'macro_regime:benchmark', "
                     "'finding_scan:last_node_id')", (_PREFIX + "%",))
        conn.execute("DELETE FROM data_source_events WHERE source_name LIKE ?",
                     (_PREFIX + "%",))
        # Substring match (not just prefix) — P2-6 tests need slugs like
        # "finding-campaign-{_PREFIX}real" where the required
        # 'finding-campaign-' literal must come BEFORE _PREFIX.
        kb_ids = [r["id"] for r in conn.execute(
            "SELECT id FROM kb_nodes WHERE slug LIKE ?", (f"%{_PREFIX}%",))]
        for nid in kb_ids:
            conn.execute("DELETE FROM kb_edges WHERE source_id=? OR target_id=?",
                         (nid, nid))
            conn.execute("DELETE FROM kb_fts WHERE node_id=?", (nid,))
            conn.execute("DELETE FROM kb_nodes WHERE id=?", (nid,))


@pytest.fixture(autouse=True)
def _setup():
    init_db()
    _purge()
    yield
    _purge()


def _insert_idea(slug, ticker="BTC/USDT", timeframe="1d", extra=None):
    with db_session() as conn:
        cur = conn.execute(
            "INSERT INTO alpha_ideas (slug, title, hypothesis, ticker, "
            "timeframe, factor_formula, stage, status) "
            "VALUES (?,?,?,?,?,?, 'stage2', 'rejected')",
            (slug, f"TESTREVISIT {slug}", "test hypothesis", ticker, timeframe,
             "test formula"))
        idea_id = cur.lastrowid
    return idea_id


def _insert_backtest_run(idea_id, **regime_cols):
    with db_session() as conn:
        cols = ", ".join(regime_cols.keys())
        placeholders = ", ".join("?" for _ in regime_cols)
        conn.execute(
            f"INSERT INTO backtest_runs (idea_id, {cols}) VALUES (?, {placeholders})",
            (idea_id, *regime_cols.values()))


def _insert_cemetery(idea_id, revival_conditions="generic", rejection_reason="generic",
                     rejected_at_stage="stage2"):
    with db_session() as conn:
        conn.execute(
            "INSERT INTO strategy_cemetery (idea_id, strategy_name, factor_type, "
            "sector, rejected_at_stage, rejection_reason, revival_conditions) "
            "VALUES (?, ?, 'momentum', 'general', ?, ?, ?)",
            (idea_id, f"TESTREVISIT {idea_id}", rejected_at_stage,
             rejection_reason, revival_conditions))


def _set_parent(idea_id, parent_id):
    with db_session() as conn:
        conn.execute("UPDATE alpha_ideas SET parent_idea_id=? WHERE id=?",
                    (parent_id, idea_id))


# ── select_candidates: regime_change ─────────────────────────────────────────

def test_regime_change_selects_matching_tercile_only():
    strong_id = _insert_idea(_PREFIX + "high-strong")
    _insert_backtest_run(strong_id, regimes_positive=1, sharpe_high_vol=1.2,
                         sharpe_low_vol=-0.5, sharpe_mid_vol=-0.1)
    wrong_tercile_id = _insert_idea(_PREFIX + "low-strong")
    _insert_backtest_run(wrong_tercile_id, regimes_positive=1, sharpe_low_vol=1.2,
                         sharpe_high_vol=-0.5, sharpe_mid_vol=-0.1)
    robust_id = _insert_idea(_PREFIX + "already-robust")
    _insert_backtest_run(robust_id, regimes_positive=2, sharpe_high_vol=1.0,
                         sharpe_mid_vol=0.5, sharpe_low_vol=-0.2)

    triggers = [{"type": "regime_change", "kind": "vol_tercile",
                "from": "low_vol", "to": "high_vol"}]
    candidates = revisit.select_candidates(triggers)
    ids = {c["idea_id"] for c in candidates}
    assert strong_id in ids
    assert wrong_tercile_id not in ids
    assert robust_id not in ids


def test_regime_change_respects_cooldown_and_calib_exclusion():
    idea_id = _insert_idea(_PREFIX + "cooldown-test")
    _insert_backtest_run(idea_id, regimes_positive=1, sharpe_high_vol=1.0,
                         sharpe_low_vol=-0.1, sharpe_mid_vol=-0.1)
    calib_id = _insert_idea("calib-" + _PREFIX + "probe")
    _insert_backtest_run(calib_id, regimes_positive=1, sharpe_high_vol=1.0,
                         sharpe_low_vol=-0.1, sharpe_mid_vol=-0.1)

    triggers = [{"type": "regime_change", "kind": "vol_tercile",
                "from": "low_vol", "to": "high_vol"}]
    first = revisit.select_candidates(triggers, limit=10)
    assert idea_id in {c["idea_id"] for c in first}
    assert calib_id not in {c["idea_id"] for c in first}

    revisit.enqueue_revisit({"idea_id": idea_id, "reason": "test"})
    second = revisit.select_candidates(triggers, limit=10)
    assert idea_id not in {c["idea_id"] for c in second}


def test_never_chain_revives_a_revisit_of_a_revisit():
    """Caught live 2026-07-12: idea 218 (a revisit of 174) got revisited
    AGAIN as idea 220 by an unrelated contradicting-finding trigger, titled
    'revisit: revisit: ...' — a structurally blocked idea can loop forever,
    burning a trial slot each cycle for zero new information. Only the
    idea's OWN offspring are excluded; it stays revivable itself."""
    root_id = _insert_idea(_PREFIX + "root")
    _insert_backtest_run(root_id, regimes_positive=1, sharpe_high_vol=1.0,
                         sharpe_low_vol=-0.1, sharpe_mid_vol=-0.1)
    already_revisited_id = _insert_idea(_PREFIX + "already-revisited")
    _set_parent(already_revisited_id, root_id)
    # Backdate past the cooldown window so root_id's exclusion below is
    # attributable ONLY to the new "own-offspring" guard, not the
    # pre-existing recent-cooldown check (a separate, already-tested rule).
    with db_session() as conn:
        conn.execute(
            "UPDATE alpha_ideas SET created_at=datetime('now', '-200 days') WHERE id=?",
            (already_revisited_id,))
    _insert_backtest_run(already_revisited_id, regimes_positive=1, sharpe_high_vol=1.0,
                         sharpe_low_vol=-0.1, sharpe_mid_vol=-0.1)

    triggers = [{"type": "regime_change", "kind": "vol_tercile",
                "from": "low_vol", "to": "high_vol"}]
    candidates = revisit.select_candidates(triggers, limit=10)
    ids = {c["idea_id"] for c in candidates}
    assert root_id in ids
    assert already_revisited_id not in ids


def test_never_revives_a_structurally_unrepresentable_idea():
    idea_id = _insert_idea(_PREFIX + "unrepresentable")
    _insert_backtest_run(idea_id, regimes_positive=1, sharpe_high_vol=1.0,
                         sharpe_low_vol=-0.1, sharpe_mid_vol=-0.1)
    _insert_cemetery(idea_id, rejected_at_stage="unrepresentable",
                     rejection_reason="custom cross-asset ratio not in the condition set")

    triggers = [{"type": "regime_change", "kind": "vol_tercile",
                "from": "low_vol", "to": "high_vol"}]
    candidates = revisit.select_candidates(triggers, limit=10)
    assert idea_id not in {c["idea_id"] for c in candidates}


# ── select_candidates: data_source ───────────────────────────────────────────

def test_data_source_matches_rejection_reason():
    idea_id = _insert_idea(_PREFIX + "funding-gap")
    _insert_cemetery(idea_id, revival_conditions="generic boilerplate",
                     rejection_reason="no edge: funding_history unavailable at the time")
    unrelated_id = _insert_idea(_PREFIX + "unrelated")
    _insert_cemetery(unrelated_id, revival_conditions="generic", rejection_reason="low sharpe")

    triggers = [{"type": "data_source", "source_name": "funding_history",
                "description": "backfilled"}]
    candidates = revisit.select_candidates(triggers)
    ids = {c["idea_id"] for c in candidates}
    assert idea_id in ids
    assert unrelated_id not in ids


# ── select_candidates: contradicting_finding ─────────────────────────────────

def test_contradicting_finding_matches_by_factor_sector():
    idea_id = _insert_idea(_PREFIX + "contradicted")
    with db_session() as conn:
        conn.execute(
            "INSERT INTO strategy_cemetery (idea_id, strategy_name, factor_type, "
            "sector, rejected_at_stage, rejection_reason, revival_conditions) "
            "VALUES (?, ?, 'momentum', 'plantation', 'stage2', 'no edge', 'generic')",
            (idea_id, f"TESTREVISIT {idea_id}"))

    triggers = [{"type": "contradicting_finding", "finding_id": 1,
                "pattern_slug": "reject-momentum-plantation-no_edge"}]
    candidates = revisit.select_candidates(triggers)
    assert idea_id in {c["idea_id"] for c in candidates}


def test_contradicting_finding_ignores_malformed_slug():
    triggers = [{"type": "contradicting_finding", "finding_id": 1,
                "pattern_slug": "not-enough-parts"}]
    assert revisit.select_candidates(triggers) == []


# ── enqueue_revisit ───────────────────────────────────────────────────────────

def test_enqueue_creates_lineage_and_reuses_winner_dsl():
    parent_id = _insert_idea(_PREFIX + "parent")
    with db_session() as conn:
        conn.execute(
            """INSERT INTO optimizer_runs
                 (idea_id, status, seed, n_configs, started_at, finished_at,
                  summary_json, winner_json)
               VALUES (?, 'done', 0, 5, datetime('now'), datetime('now'), '{}', ?)""",
            (parent_id, json.dumps({"dsl": {"entry": {"leaf": "rsi", "period": 14,
                                                       "below": 30}}})))

    res = revisit.enqueue_revisit({"idea_id": parent_id, "reason": "test reason"})
    assert res["ok"] and res["parent_idea_id"] == parent_id

    with db_session() as conn:
        row = conn.execute(
            "SELECT parent_idea_id, stage, status, hypothesis FROM alpha_ideas "
            "WHERE id=?", (res["idea_id"],)).fetchone()
        assert row["parent_idea_id"] == parent_id
        assert row["stage"] == "stage2" and row["status"] == "pending"
        assert "test reason" in row["hypothesis"]

        opt = conn.execute(
            "SELECT winner_json FROM optimizer_runs WHERE idea_id=?",
            (res["idea_id"],)).fetchone()
        assert json.loads(opt["winner_json"])["dsl"]["entry"]["leaf"] == "rsi"


def test_enqueue_missing_parent_fails_cleanly():
    res = revisit.enqueue_revisit({"idea_id": 999999999, "reason": "x"})
    assert not res["ok"]


# ── detect_triggers: data_source_events + contradicting finding wiring ──────

def test_detect_triggers_consumes_data_source_event_once():
    with db_session() as conn:
        conn.execute(
            "INSERT INTO data_source_events (source_name, description) VALUES (?, ?)",
            (_PREFIX + "new_source", "test event"))

    import unittest.mock as mock
    with mock.patch.object(revisit, "_current_vol_tercile", return_value=None), \
         mock.patch.object(revisit, "_current_macro_regime", return_value=None):
        first = revisit.detect_triggers()
        second = revisit.detect_triggers()

    assert any(t["type"] == "data_source" for t in first)
    assert not any(t["type"] == "data_source" for t in second)


def test_detect_triggers_reports_regime_flip_only_on_change():
    import unittest.mock as mock
    with mock.patch.object(revisit, "_current_vol_tercile", return_value="low_vol"), \
         mock.patch.object(revisit, "_current_macro_regime", return_value=None):
        first = revisit.detect_triggers()  # no prior snapshot -> no flip reported
    assert not any(t["type"] == "regime_change" for t in first)

    with mock.patch.object(revisit, "_current_vol_tercile", return_value="high_vol"), \
         mock.patch.object(revisit, "_current_macro_regime", return_value=None):
        second = revisit.detect_triggers()
    flips = [t for t in second if t["type"] == "regime_change"]
    assert flips and flips[0]["from"] == "low_vol" and flips[0]["to"] == "high_vol"


def test_detect_triggers_finding_contradicts_rejection_pattern():
    """Positive case: a genuine finding-campaign-* node's heuristic-origin
    contradicts edge fires the trigger normally."""
    pattern_id = store.upsert_node(
        "rejection_pattern", slug=_PREFIX + "pattern", title="test pattern")
    finding_id = store.upsert_node(
        "finding", slug=f"finding-campaign-{_PREFIX}real", title="x")
    store.add_edge(finding_id, pattern_id, "contradicts", origin="heuristic")

    import unittest.mock as mock
    with mock.patch.object(revisit, "_current_vol_tercile", return_value=None), \
         mock.patch.object(revisit, "_current_macro_regime", return_value=None):
        triggers = revisit.detect_triggers()
        again = revisit.detect_triggers()

    hits = [t for t in triggers if t["type"] == "contradicting_finding"
           and t["finding_id"] == finding_id]
    assert len(hits) == 1
    assert not any(t["type"] == "contradicting_finding" and t["finding_id"] == finding_id
                  for t in again)


# ── P2-6 (2026-07-13 audit): contradicts edges from the graph extractor are
# unreliable (LLM-origin, frequently backwards even at 0.88-0.95 weight) —
# the trigger must ignore them regardless of confidence, and must ignore a
# genuine finding node that isn't the deterministic campaign-findings kind. ──

def test_detect_triggers_ignores_llm_origin_contradicts_even_at_high_weight():
    pattern_id = store.upsert_node(
        "rejection_pattern", slug=_PREFIX + "pattern2", title="test pattern 2")
    finding_id = store.upsert_node(
        "finding", slug=f"finding-campaign-{_PREFIX}llm", title="y")
    store.add_edge(finding_id, pattern_id, "contradicts", weight=0.95, origin="llm")

    import unittest.mock as mock
    with mock.patch.object(revisit, "_current_vol_tercile", return_value=None), \
         mock.patch.object(revisit, "_current_macro_regime", return_value=None):
        triggers = revisit.detect_triggers()

    assert not any(t["type"] == "contradicting_finding" and t["finding_id"] == finding_id
                  for t in triggers)


def test_detect_triggers_ignores_non_finding_source_node():
    """Regression: the query previously never checked the source node's
    type at all — a plain idea's contradicts edge to a rejection_pattern
    must not fire this trigger, even with heuristic origin."""
    pattern_id = store.upsert_node(
        "rejection_pattern", slug=_PREFIX + "pattern3", title="test pattern 3")
    idea_node_id = store.upsert_node("idea", slug=_PREFIX + "not-a-finding", title="z")
    store.add_edge(idea_node_id, pattern_id, "contradicts", origin="heuristic")

    import unittest.mock as mock
    with mock.patch.object(revisit, "_current_vol_tercile", return_value=None), \
         mock.patch.object(revisit, "_current_macro_regime", return_value=None):
        triggers = revisit.detect_triggers()

    assert not any(t["type"] == "contradicting_finding" and t["finding_id"] == idea_node_id
                  for t in triggers)


def test_detect_triggers_ignores_finding_node_outside_campaign_namespace():
    """A node_type='finding' that isn't from campaign_findings.py's
    deterministic finding-campaign-* namespace (e.g. one the LLM extractor
    itself created) must not qualify either."""
    pattern_id = store.upsert_node(
        "rejection_pattern", slug=_PREFIX + "pattern4", title="test pattern 4")
    finding_id = store.upsert_node("finding", slug=_PREFIX + "not-campaign-slug", title="w")
    store.add_edge(finding_id, pattern_id, "contradicts", origin="heuristic")

    import unittest.mock as mock
    with mock.patch.object(revisit, "_current_vol_tercile", return_value=None), \
         mock.patch.object(revisit, "_current_macro_regime", return_value=None):
        triggers = revisit.detect_triggers()

    assert not any(t["type"] == "contradicting_finding" and t["finding_id"] == finding_id
                  for t in triggers)
