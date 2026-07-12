"""Finding-driven candidate generation: closes the loop from "a confirmed/
watch direction landed in the knowledge graph" to "a gated candidate is
automatically submitted to test it further."

House pattern: share the local DB, isolate by slug prefix, purge before +
after.
"""
import pytest

from data.database import db_session, init_db
from knowledge.graph import store
from pipeline import finding_candidates as fc

_PREFIX = "test-fc-"


def _purge():
    # Submitted ideas' slugs carry the numeric finding-node id, not our slug
    # prefix — but their hypothesis text embeds the finding's slug (which
    # DOES carry the prefix), so match on that instead.
    with db_session() as conn:
        ids = [r["id"] for r in conn.execute(
            "SELECT id FROM alpha_ideas WHERE hypothesis LIKE ? OR slug LIKE ?",
            (f"%{_PREFIX}%", _PREFIX + "%"))]
        for iid in ids:
            for tbl in ("backtest_runs", "optimizer_runs", "gate_decisions",
                       "pipeline_events", "paper_trades"):
                conn.execute(f"DELETE FROM {tbl} WHERE idea_id=?", (iid,))
            conn.execute("DELETE FROM alpha_ideas WHERE id=?", (iid,))
        node_ids = [r["id"] for r in conn.execute(
            "SELECT id FROM kb_nodes WHERE slug LIKE ?", (_PREFIX + "%",))]
        for nid in node_ids:
            conn.execute("DELETE FROM kb_edges WHERE source_id=? OR target_id=?",
                         (nid, nid))
            conn.execute("DELETE FROM kb_fts WHERE node_id=?", (nid,))
            conn.execute("DELETE FROM kb_nodes WHERE id=?", (nid,))
        conn.execute("DELETE FROM revisit_state WHERE key="
                     "'finding_candidates:last_node_id'")


@pytest.fixture(autouse=True)
def _setup():
    init_db()
    _purge()
    yield
    _purge()


def _make_finding(slug, direction, leaf_names):
    fnode = store.upsert_node(
        "finding", slug=_PREFIX + slug, title=f"test finding {slug}",
        domain="research-record", summary="test", tags=[direction])
    for name in leaf_names:
        leaf = store.upsert_node("leaf", slug=f"leaf-{name}", title=name, domain="dsl")
        store.add_edge(fnode, leaf, "uses_leaf", weight=1.0, origin="heuristic")
    return fnode


# ── builder sanity ───────────────────────────────────────────────────────────

def test_all_default_builders_produce_valid_trees():
    from agents.backtest_engineer.signal_dsl import validate
    for name, build in fc.LEAF_DEFAULT_BUILDERS.items():
        for short in (False, True):
            errs = validate(build(short))
            assert not errs, f"{name} short={short}: {errs}"


# ── submit_leaf_candidate ────────────────────────────────────────────────────

def test_submit_leaf_candidate_inserts_winner_json_and_dedupes():
    import json
    tree = fc.LEAF_DEFAULT_BUILDERS["rsi"](False)
    res1 = fc.submit_leaf_candidate(
        tree, slug=_PREFIX + "rsi1", title="t", hypothesis="h",
        ticker="BTC/USDT")
    assert res1["ok"]
    with db_session() as conn:
        row = conn.execute(
            "SELECT stage, status, family FROM alpha_ideas WHERE id=?",
            (res1["idea_id"],)).fetchone()
        opt = conn.execute(
            "SELECT n_configs, winner_json FROM optimizer_runs WHERE idea_id=?",
            (res1["idea_id"],)).fetchone()
    assert row["stage"] == "stage2" and row["status"] == "pending"
    assert row["family"] == "finding_driven"
    assert opt["n_configs"] == 1
    assert json.loads(opt["winner_json"])["dsl"]["entry"]["leaf"] == "rsi"

    res2 = fc.submit_leaf_candidate(
        tree, slug=_PREFIX + "rsi2", title="t", hypothesis="h",
        ticker="BTC/USDT")
    assert not res2["ok"] and "duplicate" in res2["error"]


def test_submit_leaf_candidate_rejects_invalid_tree():
    res = fc.submit_leaf_candidate(
        {"entry": {"leaf": "not_a_real_leaf"}},
        slug=_PREFIX + "bad", title="t", hypothesis="h", ticker="BTC/USDT")
    assert not res["ok"] and "invalid tree" in res["error"]


# ── finding scan ─────────────────────────────────────────────────────────────

def test_new_actionable_findings_filters_direction_and_id():
    confirmed_id = _make_finding("confirmed1", "confirmed", ["rsi"])
    falsified_id = _make_finding("falsified1", "falsified", ["sma_cross"])
    watch_id = _make_finding("watch1", "watch", ["macd"])
    with db_session() as conn:
        found = fc._new_actionable_findings(conn, since_id=0)
    slugs = {f["slug"] for f in found}
    assert _PREFIX + "confirmed1" in slugs
    assert _PREFIX + "watch1" in slugs
    assert _PREFIX + "falsified1" not in slugs

    with db_session() as conn:
        found_after = fc._new_actionable_findings(conn, since_id=confirmed_id)
    assert all(f["id"] > confirmed_id for f in found_after)


def test_leaves_for_finding_returns_only_uses_leaf_edges():
    fid = _make_finding("multi", "confirmed", ["rsi", "funding_level"])
    with db_session() as conn:
        leaves = fc._leaves_for_finding(conn, fid)
    assert set(leaves) == {"rsi", "funding_level"}


# ── end-to-end cycle ─────────────────────────────────────────────────────────

def test_run_cycle_submits_plain_and_regime_scoped_and_advances_snapshot():
    _make_finding("e2e", "confirmed", ["rsi", "macd"])
    result = fc.run_finding_driven_candidates()
    assert result["findings_scanned"] == 1
    assert result["leaf_matches"] == 2
    # 2 leaves x (plain + regime-scoped) = 4, exactly at the cap
    assert result["submitted"] == 4
    assert len(result["ideas"]) == 4

    with db_session() as conn:
        families = [r["family"] for r in conn.execute(
            "SELECT family FROM alpha_ideas WHERE id IN ({})".format(
                ",".join("?" * len(result["ideas"]))), result["ideas"])]
    assert families.count("finding_driven") == 2
    assert families.count("regime_scoped") == 2

    # second run sees nothing new (snapshot advanced past this finding)
    result2 = fc.run_finding_driven_candidates()
    assert result2["findings_scanned"] == 0
    assert result2["submitted"] == 0


def test_run_cycle_skips_unrecognized_leaves_silently():
    _make_finding("unrecognized", "confirmed", ["gap", "div_days_to_ex"])
    result = fc.run_finding_driven_candidates()
    assert result["findings_scanned"] == 1
    assert result["leaf_matches"] == 0
    assert result["submitted"] == 0


def test_run_cycle_respects_max_candidates_per_cycle():
    _make_finding("many", "confirmed",
                 ["rsi", "sma_cross", "ema_cross", "ma_level", "macd"])
    result = fc.run_finding_driven_candidates()
    assert result["leaf_matches"] == 5
    assert result["submitted"] == fc.MAX_CANDIDATES_PER_CYCLE
