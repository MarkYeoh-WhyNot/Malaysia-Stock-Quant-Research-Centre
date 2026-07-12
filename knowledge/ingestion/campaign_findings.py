#!/usr/bin/env python3
"""Campaign-level research findings → knowledge graph.

Alpha-hunt campaigns produce two kinds of knowledge that individual
alpha_ideas rows can't carry: FALSIFIED directions ("N trials across this
family found nothing — stop digging here") and CONFIRMED signals ("this
survived every gate"). Both belong in the graph as `finding` nodes so the
generator's GraphRAG retrieval (strategy_researcher.generate_ideas) and the
red/blue team's attack-ammunition search surface them automatically.

Slug namespace is `finding-campaign-*` — disjoint from the governance
ingester's `finding-{int}` (evidence_graph.ingest_findings). Zero LLM budget;
idempotent via upsert_node's content_hash.

  finding --uses_leaf-->     leaf        (which DSL vocabulary the campaign swept)
  finding --contradicts-->   <any node>  (verdict conflicts with target's claim)
  finding --supports-->      <any node>
  finding --refines-->       <any node>  (narrows/redirects an earlier finding)
"""
import json
import logging
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data.database import db_session
from knowledge.graph import store

logger = logging.getLogger(__name__)

INGESTION_VERSION = "kg-v1"

DIRECTIONS = ("falsified", "confirmed", "watch")

# A zero-survivor campaign only earns a falsified-direction node when the
# search was large enough that "we found nothing" is evidence rather than
# an underpowered look. 500 ≈ a multi-pair multi-timeframe sweep.
MIN_TRIALS_FOR_FALSIFIED = 500


def _node_id_by_slug(slug: str) -> int | None:
    with db_session() as conn:
        row = conn.execute("SELECT id FROM kb_nodes WHERE slug=?", (slug,)).fetchone()
    return row["id"] if row else None


def record_campaign_finding(slug: str, title: str, summary: str, direction: str,
                            tags: list | None = None, content: str = "",
                            leaf_names: tuple = (), contradicts_slugs: tuple = (),
                            supports_slugs: tuple = (),
                            refines_slugs: tuple = ()) -> int:
    """Upsert one campaign finding node plus its typed edges. Returns node id.

    `leaf_names` are bare DSL leaf names (e.g. "funding_level") — the leaf
    nodes are upserted on demand so ordering vs evidence_graph.ingest_leaves
    doesn't matter. `*_slugs` must name EXISTING nodes; missing targets are
    logged and skipped (a finding never fabricates its counterparty).
    """
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")

    fnode = store.upsert_node(
        "finding", slug=f"finding-campaign-{slug}", title=title,
        domain="research-record", summary=summary[:2000],
        tags=[direction] + [t for t in (tags or []) if t],
        content=content, ingestion_version=INGESTION_VERSION,
    )

    for name in leaf_names:
        leaf = store.upsert_node(
            "leaf", slug=f"leaf-{name}", title=name, domain="dsl",
            ingestion_version=INGESTION_VERSION,
        )
        store.add_edge(fnode, leaf, "uses_leaf", weight=1.0, origin="heuristic")

    for relation, slugs in (("contradicts", contradicts_slugs),
                            ("supports", supports_slugs),
                            ("refines", refines_slugs)):
        for target_slug in slugs:
            tid = _node_id_by_slug(target_slug)
            if tid is None:
                logger.warning(f"[CampaignFindings] {relation} target missing, "
                               f"skipped: {target_slug}")
                continue
            store.add_edge(fnode, tid, relation, weight=1.0, origin="heuristic")

    return fnode


def _leaves_in_tree(tree: dict) -> set:
    """Collect every {"leaf": name} in a DSL tree (entry/exit/short_* + AND/OR)."""
    leaves: set = set()

    def _walk(node):
        if isinstance(node, dict):
            if "leaf" in node:
                leaves.add(node["leaf"])
            for v in node.values():
                _walk(v)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(tree)
    return leaves


def emit_alpha_hunt_findings(report: dict) -> dict:
    """Turn an alpha_hunt report dict into finding nodes.

    Zero survivors over a big-enough sweep → ONE falsified-direction node
    covering the whole campaign (leaf vocabulary from the finalist trees).
    Each survivor → one confirmed node. Small zero-survivor runs (e.g. a
    --pairs smoke test) record nothing — absence of evidence at low N.
    """
    stats = {"falsified": 0, "confirmed": 0}
    trials = int(report.get("stage_a_trials") or 0)
    survivors = report.get("survivors") or []
    results = report.get("finalist_results") or []
    month = str(report.get("generated_at") or "")[:7] or "undated"

    if not survivors:
        if trials < MIN_TRIALS_FOR_FALSIFIED:
            logger.info(f"[CampaignFindings] zero survivors at {trials} trials "
                        f"(< {MIN_TRIALS_FOR_FALSIFIED}) — not evidence, skipped")
            return stats
        leaf_names = set()
        for r in results:
            leaf_names |= _leaves_in_tree(r.get("dsl") or {})
        record_campaign_finding(
            slug=f"alpha-hunt-{month}-no-edge",
            title=f"Alpha hunt {month}: no gate-passing edge found",
            summary=(f"Campaign over pairs={report.get('pairs')} "
                     f"timeframes={report.get('timeframes')}: {trials} Stage-A "
                     f"trials, {report.get('stage_b_finalists', 0)} finalists "
                     f"through the full gate stack, zero survivors. This "
                     f"direction is falsified at this search size — do not "
                     f"re-propose these formulations on the same universe "
                     f"without new information."),
            direction="falsified",
            tags=["alpha-hunt", str(report.get("market_mode") or "")],
            content=json.dumps({"stage_a_trials": trials,
                                "finalists": report.get("stage_b_finalists"),
                                "pairs": report.get("pairs"),
                                "timeframes": report.get("timeframes")}),
            leaf_names=tuple(sorted(leaf_names)),
        )
        stats["falsified"] = 1
        return stats

    for s in survivors:
        cfg, pair, tf = s.get("config"), s.get("pair"), s.get("tf")
        record_campaign_finding(
            slug=f"alpha-hunt-{month}-{cfg}-{str(pair).replace('/', '')}-{tf}",
            title=f"Alpha hunt {month}: {cfg} survived all gates on {pair} {tf}",
            summary=(f"Gate-passing strategy from the {month} campaign: {cfg} "
                     f"({s.get('family')}) on {pair} {tf}, test net Sharpe "
                     f"{s.get('test_sharpe_net')} vs deflated hurdle "
                     f"{s.get('deflated_hurdle')} at n_trials={s.get('n_trials')}."),
            direction="confirmed",
            tags=["alpha-hunt", str(s.get("family") or ""), str(pair or "")],
            content=json.dumps({k: s.get(k) for k in
                                ("idea_id", "test_sharpe_net", "deflated_hurdle",
                                 "n_trials", "trades_full_window")}),
            leaf_names=tuple(sorted(_leaves_in_tree(s.get("dsl") or {}))),
        )
        stats["confirmed"] += 1
    return stats
