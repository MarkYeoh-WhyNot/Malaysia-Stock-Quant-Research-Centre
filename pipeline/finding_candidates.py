#!/usr/bin/env python3
"""Finding-driven candidate generation — closes the ideation loop from "a
confirmed/watch direction landed in the knowledge graph" to "a gated
candidate is automatically submitted to validate it further."

Scans NEW finding nodes tagged confirmed/watch (campaign_findings.py's
direction tag) since the last scan, reads which DSL leaves each one used
(uses_leaf edges), and for every RECOGNIZED leaf submits:
  - one plain single-name candidate using that leaf with sensible default
    parameters, and
  - one regime-scoped variant of the SAME base tree (active=["high_vol"] by
    default — carry/extremity edges are commonly regime-dependent; Phase 4's
    submit_regime_scoped_idea charges its own >=6-config DOF via
    optimizer_runs, independent of this job's own accounting).

Both enter the pipeline via the SAME winner_json convention alpha_hunt and
regime_candidates.py use: backtest_idea() prefers optimizer_runs.winner_json
over the LLM factor-formula parser (backtest_engineer.py:913-928), so the
exact tree submitted is the tree that gets backtested — no re-parsing
ambiguity. Each base candidate is charged exactly 1 trial (n_configs=1): a
deterministic construction, not a sweep — alpha_hunt.py remains the tool for
parameter sweeps.

Bursa-inapplicable leaves (funding_level/funding_zscore need funding_rate,
crypto-only) simply produce zero candidates there — no market-mode branch
needed. Event/commodity leaves (gap, div_days_to_ex, cpo_change) need domain
context this deterministic job doesn't have — deliberately absent from
LEAF_DEFAULT_BUILDERS, not a placeholder.

Deterministic, zero LLM cost.
"""
from __future__ import annotations

import json
import logging

from data.database import db_session

logger = logging.getLogger(__name__)

MAX_CANDIDATES_PER_CYCLE = 4
DEFAULT_REGIME_ACTIVE = ["high_vol"]


# ── Default single-leaf tree builders ───────────────────────────────────────

def _t_rsi(short: bool) -> dict:
    t = {"entry": {"leaf": "rsi", "period": 14, "below": 30},
         "exit":  {"leaf": "rsi", "period": 14, "above": 70}}
    if short:
        t["short_entry"] = {"leaf": "rsi", "period": 14, "above": 70}
        t["short_exit"]  = {"leaf": "rsi", "period": 14, "below": 50}
    return t


def _t_sma_cross(short: bool) -> dict:
    t = {"entry": {"leaf": "sma_cross", "fast": 20, "slow": 50, "direction": "above"},
         "exit":  {"leaf": "sma_cross", "fast": 20, "slow": 50, "direction": "below"}}
    if short:
        t["short_entry"] = {"leaf": "sma_cross", "fast": 20, "slow": 50, "direction": "below"}
        t["short_exit"]  = {"leaf": "sma_cross", "fast": 20, "slow": 50, "direction": "above"}
    return t


def _t_ema_cross(short: bool) -> dict:
    t = {"entry": {"leaf": "ema_cross", "fast": 12, "slow": 26, "direction": "above"},
         "exit":  {"leaf": "ema_cross", "fast": 12, "slow": 26, "direction": "below"}}
    if short:
        t["short_entry"] = {"leaf": "ema_cross", "fast": 12, "slow": 26, "direction": "below"}
        t["short_exit"]  = {"leaf": "ema_cross", "fast": 12, "slow": 26, "direction": "above"}
    return t


def _t_ma_level(short: bool) -> dict:
    t = {"entry": {"leaf": "ma_level", "period": 50, "ma_type": "sma", "direction": "above"},
         "exit":  {"leaf": "ma_level", "period": 50, "ma_type": "sma", "direction": "below"}}
    if short:
        t["short_entry"] = {"leaf": "ma_level", "period": 50, "ma_type": "sma", "direction": "below"}
        t["short_exit"]  = {"leaf": "ma_level", "period": 50, "ma_type": "sma", "direction": "above"}
    return t


def _t_momentum(short: bool) -> dict:
    # No natural short mirror without a negative-momentum leaf combo (mirrors
    # alpha_hunt.py's _t_mom convention).
    return {"entry": {"leaf": "momentum", "period": 20, "min_return": 0.03},
            "exit":  {"leaf": "reversal", "period": 5, "max_return": -0.02}}


def _t_bollinger(short: bool) -> dict:
    t = {"entry": {"leaf": "bollinger", "period": 20, "std": 2.0, "band": "below_lower"},
         "exit":  {"leaf": "bollinger", "period": 20, "std": 2.0, "band": "above_upper"}}
    if short:
        t["short_entry"] = {"leaf": "bollinger", "period": 20, "std": 2.0, "band": "above_upper"}
        t["short_exit"]  = {"leaf": "bollinger", "period": 20, "std": 2.0, "band": "below_lower"}
    return t


def _t_macd(short: bool) -> dict:
    t = {"entry": {"leaf": "macd", "fast": 12, "slow": 26, "signal": 9, "condition": "bullish"},
         "exit":  {"leaf": "macd", "fast": 12, "slow": 26, "signal": 9, "condition": "bearish"}}
    if short:
        t["short_entry"] = {"leaf": "macd", "fast": 12, "slow": 26, "signal": 9, "condition": "bearish"}
        t["short_exit"]  = {"leaf": "macd", "fast": 12, "slow": 26, "signal": 9, "condition": "bullish"}
    return t


def _t_volume_ratio(short: bool) -> dict:
    return {"entry": {"op": "AND", "children": [
                {"leaf": "volume_ratio", "period": 20, "min_ratio": 2.5},
                {"leaf": "momentum", "period": 5, "min_return": 0.01}]},
            "exit": {"leaf": "reversal", "period": 3, "max_return": -0.02}}


def _t_rolling_rank(short: bool) -> dict:
    return {"entry": {"leaf": "rolling_rank", "formation": 126, "skip": 10,
                      "window": 252, "min_pct": 0.8},
            "exit":  {"leaf": "rolling_rank", "formation": 126, "skip": 10,
                      "window": 252, "max_pct": 0.5}}


def _t_zscore(short: bool) -> dict:
    t = {"entry": {"leaf": "zscore", "period": 20, "below": -2.0},
         "exit":  {"leaf": "zscore", "period": 20, "above": 0.0}}
    if short:
        t["short_entry"] = {"leaf": "zscore", "period": 20, "above": 2.0}
        t["short_exit"]  = {"leaf": "zscore", "period": 20, "below": 0.0}
    return t


def _t_funding_level(short: bool) -> dict:
    t = {"entry": {"leaf": "funding_level", "below": -0.0001},
         "exit":  {"leaf": "funding_level", "above": 0.0}}
    if short:
        t["short_entry"] = {"leaf": "funding_level", "above": 0.0001}
        t["short_exit"]  = {"leaf": "funding_level", "below": 0.0}
    return t


def _t_funding_zscore(short: bool) -> dict:
    t = {"entry": {"leaf": "funding_zscore", "period": 60, "below": -2.0},
         "exit":  {"leaf": "funding_zscore", "period": 60, "above": 0.0}}
    if short:
        t["short_entry"] = {"leaf": "funding_zscore", "period": 60, "above": 2.0}
        t["short_exit"]  = {"leaf": "funding_zscore", "period": 60, "below": 0.0}
    return t


LEAF_DEFAULT_BUILDERS = {
    "rsi": _t_rsi, "sma_cross": _t_sma_cross, "ema_cross": _t_ema_cross,
    "ma_level": _t_ma_level, "momentum": _t_momentum, "reversal": _t_momentum,
    "bollinger": _t_bollinger, "macd": _t_macd, "volume_ratio": _t_volume_ratio,
    "rolling_rank": _t_rolling_rank, "zscore": _t_zscore,
    "funding_level": _t_funding_level, "funding_zscore": _t_funding_zscore,
}


# ── Finding scan ─────────────────────────────────────────────────────────────

def _new_actionable_findings(conn, since_id: int) -> list[dict]:
    """Finding nodes tagged confirmed or watch, id > since_id (tags is a JSON
    array string — see knowledge/graph/store.py:upsert_node)."""
    return [dict(r) for r in conn.execute(
        "SELECT id, slug, title FROM kb_nodes WHERE node_type='finding' "
        "AND id > ? AND (tags LIKE '%\"confirmed\"%' OR tags LIKE '%\"watch\"%') "
        "ORDER BY id ASC", (since_id,))]


def _leaves_for_finding(conn, finding_id: int) -> list[str]:
    return [r["leaf_name"] for r in conn.execute(
        "SELECT l.title AS leaf_name FROM kb_edges e "
        "JOIN kb_nodes l ON l.id = e.target_id "
        "WHERE e.source_id=? AND e.relation='uses_leaf' AND l.node_type='leaf'",
        (finding_id,))]


# ── Submission ───────────────────────────────────────────────────────────────

def submit_leaf_candidate(tree: dict, *, slug: str, title: str, hypothesis: str,
                          ticker: str, timeframe: str = "1d") -> dict:
    """Insert a deterministic, single-trial candidate via the winner_json
    convention (backtest_idea prefers it over the LLM factor-formula parser).
    Returns {"ok": False, "error": ...} on invalid tree or signature collision.
    """
    from agents.backtest_engineer.signal_dsl import canonical_signature, validate
    from pipeline.idea_text import ensure_description
    errors = validate(tree)
    if errors:
        return {"ok": False, "error": f"invalid tree: {errors}"}
    # Defensive: this is an unguarded param boundary — never store an empty
    # description even if a future caller passes one (today's callers don't).
    hypothesis = ensure_description(title, hypothesis, json.dumps(tree))
    signature = canonical_signature(tree, ticker)
    with db_session() as conn:
        dup = conn.execute(
            "SELECT id FROM alpha_ideas WHERE signal_signature=? "
            "AND status != 'rejected' LIMIT 1", (signature,)).fetchone()
        if dup:
            return {"ok": False, "error": f"duplicate of idea {dup['id']}"}
        cur = conn.execute(
            """INSERT INTO alpha_ideas
                 (slug, title, hypothesis, ticker, timeframe, factor_formula,
                  stage, status, novelty_score, logic_score, feasibility_score,
                  signal_signature, family)
               VALUES (?,?,?,?,?,?, 'stage2','pending',0.7,0.7,0.7,?,?)""",
            (slug, title, hypothesis, ticker, timeframe, json.dumps(tree),
             signature, "finding_driven"))
        idea_id = cur.lastrowid
        conn.execute(
            """INSERT INTO optimizer_runs
                 (idea_id, status, seed, n_configs, started_at, finished_at,
                  summary_json, winner_json)
               VALUES (?, 'done', 0, 1, datetime('now'), datetime('now'), ?, ?)""",
            (idea_id, json.dumps({"note": "finding-driven deterministic candidate"}),
             json.dumps({"dsl": tree, "instrument": ticker, "timeframe": timeframe})))
    return {"ok": True, "idea_id": idea_id, "slug": slug, "signature": signature}


def run_finding_driven_candidates() -> dict:
    """One scan cycle: new confirmed/watch findings -> leaf-driven candidates
    (plain + regime-scoped). Returns a summary dict. Query-heavy, no LLM."""
    from config.settings import DEFAULT_SYMBOLS
    from pipeline.regime_candidates import submit_regime_scoped_idea
    from pipeline.revisit import _snapshot, _update_snapshot

    ticker = DEFAULT_SYMBOLS[0] if DEFAULT_SYMBOLS else None
    if not ticker:
        return {"findings_scanned": 0, "leaf_matches": 0, "submitted": 0, "ideas": []}

    with db_session() as conn:
        since_id = int(_snapshot(conn, "finding_candidates:last_node_id") or 0)
        findings = _new_actionable_findings(conn, since_id)
        max_seen = since_id
        leaf_queue: list[tuple[int, str, str]] = []  # (finding_id, slug, leaf_name)
        for f in findings:
            max_seen = max(max_seen, f["id"])
            for leaf_name in _leaves_for_finding(conn, f["id"]):
                if leaf_name in LEAF_DEFAULT_BUILDERS:
                    leaf_queue.append((f["id"], f["slug"], leaf_name))
        if max_seen > since_id:
            _update_snapshot(conn, "finding_candidates:last_node_id", str(max_seen))

    submitted = []
    for finding_id, finding_slug, leaf_name in leaf_queue:
        if len(submitted) >= MAX_CANDIDATES_PER_CYCLE:
            break
        tree = LEAF_DEFAULT_BUILDERS[leaf_name](False)

        res = submit_leaf_candidate(
            tree, slug=f"auto-finding-{finding_id}-{leaf_name}",
            title=f"Auto candidate from {finding_slug}: {leaf_name}",
            hypothesis=(f"Deterministic default-parameter candidate built from "
                       f"the '{leaf_name}' leaf used in confirmed/watch finding "
                       f"'{finding_slug}' — closes the loop from a new "
                       f"knowledge-graph direction to a gated test."),
            ticker=ticker)
        if res.get("ok"):
            submitted.append(res)
            logger.info(f"[FindingCandidates] submitted {res['slug']} "
                       f"(idea {res['idea_id']}) from finding {finding_slug}")
        if len(submitted) >= MAX_CANDIDATES_PER_CYCLE:
            break

        rg = submit_regime_scoped_idea(
            tree, DEFAULT_REGIME_ACTIVE,
            title=f"Auto candidate from {finding_slug}: {leaf_name} (regime-scoped)",
            hypothesis=(f"Regime-scoped variant of the {leaf_name}-based candidate "
                       f"from '{finding_slug}' — active only in "
                       f"{DEFAULT_REGIME_ACTIVE} vol terciles."),
            ticker=ticker, timeframe="1d")
        if rg.get("ok"):
            submitted.append(rg)
            logger.info(f"[FindingCandidates] submitted regime-scoped "
                       f"{rg['slug']} (idea {rg['idea_id']}) from finding {finding_slug}")

    return {"findings_scanned": len(findings), "leaf_matches": len(leaf_queue),
            "submitted": len(submitted), "ideas": [s["idea_id"] for s in submitted]}
