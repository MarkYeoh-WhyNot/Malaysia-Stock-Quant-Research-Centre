"""Pin gates.recent_trial_count: calibration probes (slug 'calib-%') must not
inflate the deflated-Sharpe trial window, while real ideas AND orphaned
backtest_runs rows (no alpha_ideas parent — legitimate synthetic rows) keep
counting exactly as before.

House pattern: share the local DB, isolate by slug/id prefix, purge
before + after.
"""
import pytest

from agents.backtest_engineer.gates import recent_trial_count
from data.database import db_session, init_db

_SLUG_PREFIX = "test-tce-"
_ORPHAN_IDEA_ID = 998000001  # never inserted into alpha_ideas


def _purge():
    with db_session() as conn:
        ids = [r["id"] for r in conn.execute(
            "SELECT id FROM alpha_ideas WHERE slug LIKE ? OR slug LIKE ?",
            (_SLUG_PREFIX + "%", "calib-test-tce-%"))]
        for iid in ids + [_ORPHAN_IDEA_ID]:
            conn.execute("DELETE FROM backtest_runs WHERE idea_id=?", (iid,))
            conn.execute("DELETE FROM alpha_ideas WHERE id=?", (iid,))


@pytest.fixture(autouse=True)
def _setup():
    init_db()
    _purge()
    yield
    _purge()


def _insert_idea(slug):
    with db_session() as conn:
        cur = conn.execute(
            "INSERT INTO alpha_ideas (slug, title, stage, status) "
            "VALUES (?, ?, 'stage2', 'rejected')", (slug, slug))
        return cur.lastrowid


def _insert_run(idea_id):
    with db_session() as conn:
        conn.execute(
            "INSERT INTO backtest_runs (idea_id, created_at) "
            "VALUES (?, datetime('now'))", (idea_id,))


def test_calib_probes_excluded_real_and_orphan_counted():
    with db_session() as conn:
        base = recent_trial_count(conn, 90)

    real_id = _insert_idea(_SLUG_PREFIX + "real")
    calib_id = _insert_idea("calib-test-tce-probe")
    _insert_run(real_id)
    _insert_run(calib_id)
    # Orphan rows predate the FK (legacy synthetic rows in the live DB) —
    # simulate one the way it actually exists there.
    with db_session() as conn:
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute(
            "INSERT INTO backtest_runs (idea_id, created_at) "
            "VALUES (?, datetime('now'))", (_ORPHAN_IDEA_ID,))
        conn.execute("PRAGMA foreign_keys=ON")

    with db_session() as conn:
        n = recent_trial_count(conn, 90)
    # +2: the real idea and the orphan count; the calib probe does not.
    assert n == base + 2


def test_window_respected():
    real_id = _insert_idea(_SLUG_PREFIX + "old")
    with db_session() as conn:
        conn.execute(
            "INSERT INTO backtest_runs (idea_id, created_at) "
            "VALUES (?, datetime('now', '-120 days'))", (real_id,))
        base = recent_trial_count(conn, 90)
        wide = recent_trial_count(conn, 365)
    assert wide == base + 1
