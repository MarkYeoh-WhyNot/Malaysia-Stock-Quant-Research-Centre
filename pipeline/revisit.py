#!/usr/bin/env python3
"""Event-driven revisit — "temu gu zhi xin" (温故而知新): re-open old rejected
ideas when the conditions that killed them may no longer hold.

Three triggers, each answering a different question:
  (i)   regime change   — the benchmark's volatility tercile (or, on Bursa,
                          the macro regime_label) just flipped. Ideas that
                          were strong in exactly ONE tercile and rejected for
                          it (regimes_positive=1) are candidates for revival
                          IF the tercile that just started is the one they
                          were strong in.
  (ii)  new data source — a source landed that the rejected idea's family
                          couldn't use before (e.g. funding history). No
                          reliable auto-detector exists; data_source_events
                          rows are a manual convention (insert one, the scan
                          consumes it once).
  (iii) contradicting finding — a new `finding` node (campaign_findings.py)
                          carries a `contradicts` edge to a `rejection_pattern`
                          node — the verdict that killed a whole factor/sector/
                          reason class may itself be wrong now.

Every revival is a NEW alpha_ideas row (parent_idea_id lineage) entering at
stage2/pending and flowing through the ordinary backtest_idea path — so it is
counted by gates.recent_trial_count exactly like any other idea. No special
accounting is needed; re-opening a case is not free scrutiny.
"""
from __future__ import annotations

import json
import logging

from data.database import db_session

logger = logging.getLogger(__name__)

COOLDOWN_DAYS = 90
MAX_REVISITS_PER_CYCLE = 3
REGIME_STATES = ("low_vol", "mid_vol", "high_vol")


# ── Trigger detection ────────────────────────────────────────────────────────

def _current_vol_tercile(conn) -> str | None:
    """Which vol tercile the benchmark symbol is in RIGHT NOW, using the same
    ex-ante mask the regime-scoped candidates trade on (no lookahead)."""
    from config.settings import BENCHMARK_SYMBOL, FETCH_DAYS_BY_INTERVAL
    from agents.data_engineer.data_engineer import DataEngineer
    from agents.backtest_engineer.signal_dsl import _regime_mask

    try:
        de = DataEngineer()
        df = de.fetch_prices(BENCHMARK_SYMBOL, "1d",
                             FETCH_DAYS_BY_INTERVAL.get("1d", 1825), use_cache=True)
    except Exception as exc:
        logger.warning(f"[Revisit] benchmark fetch failed: {exc}")
        return None
    if df is None or len(df) < 260:
        return None
    for state in REGIME_STATES:
        mask = _regime_mask(df, [state])
        if bool(mask.iloc[-1]):
            return state
    return None


def _current_macro_regime(conn) -> str | None:
    row = conn.execute(
        "SELECT regime_label FROM macro_features ORDER BY as_of_date DESC LIMIT 1"
    ).fetchone()
    return row["regime_label"] if row and row["regime_label"] else None


def _snapshot(conn, key: str) -> str | None:
    row = conn.execute("SELECT value FROM revisit_state WHERE key=?", (key,)).fetchone()
    return row["value"] if row else None


def _update_snapshot(conn, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO revisit_state (key, value, updated_at) VALUES (?, ?, datetime('now')) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
        (key, value))


def detect_triggers() -> list[dict]:
    """Returns a list of {"type": ..., ...} trigger events fired since the
    last scan. Always advances the stored snapshots/consumed flags, even when
    nothing else about a trigger is used, so a flip is only reported once."""
    triggers: list[dict] = []
    with db_session() as conn:
        vol_now = _current_vol_tercile(conn)
        if vol_now:
            vol_prev = _snapshot(conn, "vol_regime:benchmark")
            if vol_prev and vol_prev != vol_now:
                triggers.append({"type": "regime_change", "kind": "vol_tercile",
                                 "from": vol_prev, "to": vol_now})
            _update_snapshot(conn, "vol_regime:benchmark", vol_now)

        macro_now = _current_macro_regime(conn)
        if macro_now:
            macro_prev = _snapshot(conn, "macro_regime:benchmark")
            if macro_prev and macro_prev != macro_now:
                triggers.append({"type": "regime_change", "kind": "macro",
                                 "from": macro_prev, "to": macro_now})
            _update_snapshot(conn, "macro_regime:benchmark", macro_now)

        for r in conn.execute(
                "SELECT id, source_name, description FROM data_source_events "
                "WHERE consumed=0"):
            triggers.append({"type": "data_source", "id": r["id"],
                             "source_name": r["source_name"],
                             "description": r["description"]})
            conn.execute("UPDATE data_source_events SET consumed=1 WHERE id=?", (r["id"],))

        last_seen = int(_snapshot(conn, "finding_scan:last_node_id") or 0)
        max_seen = last_seen
        for r in conn.execute(
                "SELECT e.source_id AS finding_id, n.id AS pattern_id, n.slug AS pattern_slug "
                "FROM kb_edges e JOIN kb_nodes n ON n.id = e.target_id "
                "WHERE e.relation='contradicts' AND e.source_id > ? "
                "AND n.node_type='rejection_pattern'", (last_seen,)):
            triggers.append({"type": "contradicting_finding",
                             "finding_id": r["finding_id"],
                             "pattern_slug": r["pattern_slug"]})
            max_seen = max(max_seen, r["finding_id"])
        if max_seen > last_seen:
            _update_snapshot(conn, "finding_scan:last_node_id", str(max_seen))

    if triggers:
        logger.info(f"[Revisit] {len(triggers)} trigger(s): "
                    f"{[t['type'] for t in triggers]}")
    return triggers


# ── Candidate selection ──────────────────────────────────────────────────────

def _already_revisited_recently(conn, parent_id: int) -> bool:
    row = conn.execute(
        "SELECT 1 FROM alpha_ideas WHERE parent_idea_id=? "
        "AND created_at >= datetime('now', ?) LIMIT 1",
        (parent_id, f"-{COOLDOWN_DAYS} days")).fetchone()
    return row is not None


def _eligible(conn, idea_id: int) -> bool:
    row = conn.execute(
        "SELECT slug, parent_idea_id FROM alpha_ideas WHERE id=?", (idea_id,)).fetchone()
    if not row or (row["slug"] or "").startswith("calib-"):
        return False
    if row["parent_idea_id"] is not None:
        # Never chain-revive a revisit of a revisit — without this, a
        # structurally blocked idea (see next check) loops "revisit:
        # revisit: revisit: ..." forever, burning a trial slot each time
        # for zero new information (caught live 2026-07-12: idea 218, a
        # revisit of 174, got revisited AGAIN as idea 220 by an unrelated
        # contradicting-finding trigger). The ROOT idea stays revivable —
        # only its own offspring are excluded.
        return False
    cemetery = conn.execute(
        "SELECT rejected_at_stage FROM strategy_cemetery WHERE idea_id=? "
        "ORDER BY id DESC LIMIT 1", (idea_id,)).fetchone()
    if cemetery and cemetery["rejected_at_stage"] == "unrepresentable":
        # Permanent structural block: no DSL leaf can express this formula
        # regardless of market regime, new data, or contradicting KG
        # findings — reason_category classification (rejection_memory.py's
        # coarse keyword buckets) can mislabel the TRUE cause, but this
        # column is set directly by the unrepresentable-DSL gate itself
        # (backtest_engineer.py) and is reliable. Only a code change (a
        # new leaf) fixes this, not fresh evidence.
        return False
    return not _already_revisited_recently(conn, idea_id)


def select_candidates(triggers: list[dict], limit: int = MAX_REVISITS_PER_CYCLE) -> list[dict]:
    """Map fired triggers to concrete (idea_id, reason) revival candidates."""
    candidates: list[dict] = []
    with db_session() as conn:
        for t in triggers:
            if len(candidates) >= limit:
                break
            if t["type"] == "regime_change" and t["kind"] == "vol_tercile":
                col = f"sharpe_{t['to']}"
                for r in conn.execute(
                        f"""SELECT br.idea_id AS idea_id FROM backtest_runs br
                            WHERE br.regimes_positive = 1 AND br.{col} > 0
                            ORDER BY br.id DESC LIMIT 20"""):
                    if _eligible(conn, r["idea_id"]):
                        candidates.append({
                            "idea_id": r["idea_id"],
                            "reason": (f"Regime revisit: benchmark just entered "
                                      f"{t['to']}, and this idea was strong ONLY "
                                      f"in {t['to']} when rejected (regimes_positive=1)"),
                        })
                    if len(candidates) >= limit:
                        break

            elif t["type"] == "data_source":
                # The generic per-category revival_conditions template rarely
                # names a data source; the free-text rejection_reason (what
                # the idea was ACTUALLY rejected for) is the far likelier
                # match — e.g. "no funding history available".
                pattern = f"%{t['source_name']}%"
                for r in conn.execute(
                        "SELECT idea_id, revival_conditions, rejection_reason "
                        "FROM strategy_cemetery "
                        "WHERE revival_conditions LIKE ? OR rejection_reason LIKE ? "
                        "ORDER BY id DESC LIMIT 20", (pattern, pattern)):
                    if r["idea_id"] and _eligible(conn, r["idea_id"]):
                        candidates.append({
                            "idea_id": r["idea_id"],
                            "reason": (f"Data-source revisit: '{t['source_name']}' "
                                      f"landed ({t['description'] or ''}); revival "
                                      f"condition was: {r['revival_conditions']}"),
                        })
                    if len(candidates) >= limit:
                        break

            elif t["type"] == "contradicting_finding":
                slug = t["pattern_slug"]
                # reject-{factor_type}-{sector}-{reason_category}
                parts = slug.split("-", 3)
                if len(parts) < 4:
                    continue
                _prefix, factor_type, sector, reason_category = parts
                for r in conn.execute(
                        "SELECT idea_id FROM strategy_cemetery WHERE factor_type=? "
                        "AND sector=? ORDER BY id DESC LIMIT 20",
                        (factor_type, sector)):
                    if r["idea_id"] and _eligible(conn, r["idea_id"]):
                        candidates.append({
                            "idea_id": r["idea_id"],
                            "reason": (f"Finding revisit: a new finding contradicts "
                                      f"the rejection pattern {slug} "
                                      f"(reason_category={reason_category})"),
                        })
                    if len(candidates) >= limit:
                        break
    return candidates[:limit]


# ── Enqueue ───────────────────────────────────────────────────────────────────

def enqueue_revisit(candidate: dict) -> dict:
    """Insert a new stage2/pending idea carrying parent lineage. Reuses the
    parent's optimizer winner DSL when available (same signal, fresh scrutiny)
    so backtest_idea's winner_json preference (backtest_engineer.py:913)
    replays it exactly rather than re-parsing free text."""
    idea_id = candidate["idea_id"]
    with db_session() as conn:
        parent = conn.execute(
            "SELECT slug, title, ticker, timeframe, factor_formula FROM alpha_ideas "
            "WHERE id=?", (idea_id,)).fetchone()
        if not parent:
            return {"ok": False, "error": f"parent idea {idea_id} not found"}

        winner = conn.execute(
            "SELECT winner_json FROM optimizer_runs WHERE idea_id=? AND status='done' "
            "ORDER BY id DESC LIMIT 1", (idea_id,)).fetchone()

        slug = f"revisit-{idea_id}-{__import__('time').strftime('%Y%m%d%H%M%S')}"
        cur = conn.execute(
            """INSERT INTO alpha_ideas
                 (slug, title, hypothesis, ticker, timeframe, factor_formula,
                  stage, status, novelty_score, logic_score, feasibility_score,
                  parent_idea_id)
               VALUES (?,?,?,?,?,?, 'stage2', 'pending', 0.7, 0.7, 0.7, ?)""",
            (slug, f"revisit: {parent['title']}",
             f"{candidate['reason']} [original idea {idea_id}]",
             parent["ticker"], parent["timeframe"], parent["factor_formula"],
             idea_id))
        new_id = cur.lastrowid

        if winner and winner["winner_json"]:
            conn.execute(
                """INSERT INTO optimizer_runs
                     (idea_id, status, seed, n_configs, started_at, finished_at,
                      summary_json, winner_json)
                   VALUES (?, 'done', 0, 1, datetime('now'), datetime('now'), ?, ?)""",
                (new_id, json.dumps({"note": "revisit — reused parent DSL"}),
                 winner["winner_json"]))

    logger.info(f"[Revisit] enqueued idea {new_id} (parent {idea_id}) slug={slug}")
    return {"ok": True, "idea_id": new_id, "parent_idea_id": idea_id, "slug": slug}


def run_revisit_scan() -> dict:
    """One full cycle: detect -> select -> enqueue. Returns a summary dict."""
    triggers = detect_triggers()
    if not triggers:
        return {"triggers": 0, "enqueued": 0}
    candidates = select_candidates(triggers)
    enqueued = [enqueue_revisit(c) for c in candidates]
    ok = [e for e in enqueued if e.get("ok")]
    return {"triggers": len(triggers), "candidates": len(candidates),
            "enqueued": len(ok), "ideas": [e["idea_id"] for e in ok]}
