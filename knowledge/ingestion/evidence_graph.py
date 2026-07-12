#!/usr/bin/env python3
"""Deterministic evidence-graph ingestion — turns operational rows into a truth graph.

This is Slice 1 + 1.5 + 3 of docs/knowledge_graph_evolution_design.md. It promotes
existing rows (NOT LLM output) into first-class kb_nodes and wires typed kb_edges,
so the graph can answer "which strategies were tried, which died, and why". Zero LLM
budget; idempotent (upsert_node's content_hash handles repeats).

  idea  --compiled_to-->  strategy  --shares_signature-->  signature
                          strategy  --uses_leaf-->         leaf
                          strategy  --produced-->          backtest_run
                          strategy  --failed|passed-->     gate_decision
  finding --reported_by-->  agent
  finding --exposed_to-->   risk (e.g. parser_approximation)

An idea is promoted to a `strategy` node only when it has a signature, a backtest,
a gate decision, or has advanced past gate0 (design §5.5) — raw candidates stay
`idea` nodes. Every operational row that references a non-existent idea is skipped
(this DB legitimately carries orphaned synthetic rows).
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

# Canonical named risks (identity-defined nodes; created on demand).
RISKS = {
    "parser_approximation": "A strategy compiled to an approximate DSL tree instead of an exact one (parser-honesty violation).",
    "cost_drag": "Alpha eroded by transaction costs / fees / funding.",
    "liquidation_regime_fragility": "Strategy fragile in high-volatility / liquidation regimes.",
    "btc_beta_overlap": "Return dominated by BTC beta rather than idiosyncratic edge.",
    "funding_data_gap": "Missing / distorted funding data undermines the signal.",
}

# Substrings that map a governance finding to a named risk.
_FINDING_RISK_HINTS = {
    "parser_approximation": ("parser", "approxim", "representab", "ma_level", "ema_cross"),
    "cost_drag": ("cost", "fee", "slippage", "funding drag"),
    "liquidation_regime_fragility": ("liquidation", "regime", "volatil"),
    "btc_beta_overlap": ("btc beta", "beta overlap", "market beta"),
}


def _slug_hash(value: str) -> str:
    return store.content_hash(value)[:12]


def _risk_node(name: str) -> int:
    """Upsert a named risk node (identity-defined; no source ref)."""
    return store.upsert_node(
        "risk", slug=f"risk-{name}", title=name.replace("_", " "),
        domain="risk", summary=RISKS.get(name, name),
        ingestion_version=INGESTION_VERSION,
    )


# ── Slice 1.5: leaves ────────────────────────────────────────────────────────

def ingest_leaves() -> int:
    """Register every executable DSL leaf as a `leaf` node (parser-honesty spine)."""
    try:
        from agents.backtest_engineer.signal_dsl import LEAVES
    except Exception as e:
        logger.warning(f"[EvidenceGraph] Could not import DSL leaves: {e}")
        return 0
    n = 0
    for name, spec in LEAVES.items():
        card = spec.get("shape_card") if isinstance(spec, dict) else None
        summary = (card or f"Executable DSL leaf '{name}'.")
        if isinstance(summary, (dict, list)):
            summary = json.dumps(summary)
        store.upsert_node(
            "leaf", slug=f"leaf-{name}", title=name, domain="dsl",
            summary=str(summary)[:500], ingestion_version=INGESTION_VERSION,
        )
        n += 1
    return n


def _leaves_used_by(idea: dict) -> set:
    """Best-effort deterministic extraction of leaf names a strategy uses, by
    scanning its formula/proxy/context text for {"leaf": "<name>"} or bare names."""
    from agents.backtest_engineer.signal_dsl import LEAVES
    blob = " ".join(str(idea.get(k) or "") for k in
                    ("factor_formula", "price_based_proxy", "kb_context", "signal_signature"))
    used = set()
    # explicit {"leaf": "x"} occurrences
    for name in LEAVES:
        if f'"{name}"' in blob or f"'{name}'" in blob or f'leaf": "{name}' in blob:
            used.add(name)
    return used


# ── Slice 1: strategies + evidence ───────────────────────────────────────────

def _promote_idea_to_strategy(idea: dict, reason: str) -> int:
    """Emit a `strategy` node for an evaluated idea and link idea --compiled_to--> strategy.

    strategy identity is the slug `strategy-<idea_id>` (the underlying alpha_ideas
    row is already claimed by the idea node via UNIQUE(ref_table,ref_id), so the
    strategy node stays ref-free; traceability is the slug + the compiled_to edge)."""
    iid = idea["id"]
    tags = [t for t in (idea.get("family"), idea.get("ticker"), idea.get("stage")) if t]
    strat = store.upsert_node(
        "strategy", slug=f"strategy-{iid}",
        title=idea.get("title") or f"strategy {iid}",
        domain=idea.get("family") or "",
        summary=(idea.get("hypothesis") or idea.get("title") or "")[:500],
        tags=tags, ingestion_version=INGESTION_VERSION,
    )
    # link the raw idea node (if one exists) to its compiled strategy
    with db_session() as conn:
        idea_node = conn.execute(
            "SELECT id FROM kb_nodes WHERE node_type='idea' AND ref_table='alpha_ideas' AND ref_id=?",
            (iid,),
        ).fetchone()
    if idea_node:
        store.add_edge(idea_node["id"], strat, "compiled_to", weight=1.0, origin="heuristic")
    return strat


def _qualifies(idea: dict, has_backtest: bool, has_gate: bool) -> str | None:
    if idea.get("signal_signature"):
        return "has_signature"
    if has_backtest:
        return "has_backtest"
    if has_gate:
        return "has_gate_decision"
    stage = (idea.get("stage") or "").lower()
    if stage and stage not in ("gate0", "gate-0", "screen"):
        return f"stage:{stage}"
    return None


def ingest_strategies_and_evidence() -> dict:
    """Promote qualifying ideas → strategy nodes; register signatures, backtests,
    gate decisions; wire the evidence chain. Orphan-safe."""
    stats = {"strategies": 0, "signatures": 0, "backtests": 0, "gate_decisions": 0,
             "leaf_edges": 0, "skipped_orphans": 0}

    with db_session() as conn:
        ideas = [dict(r) for r in conn.execute("SELECT * FROM alpha_ideas").fetchall()]
        bt_by_idea = {}
        for r in conn.execute("SELECT * FROM backtest_runs").fetchall():
            bt_by_idea.setdefault(r["idea_id"], []).append(dict(r))
        gd_by_idea = {}
        for r in conn.execute("SELECT * FROM gate_decisions").fetchall():
            gd_by_idea.setdefault(r["idea_id"], []).append(dict(r))

    live_idea_ids = {i["id"] for i in ideas}
    strat_by_idea: dict[int, int] = {}

    # 1) promote ideas
    for idea in ideas:
        iid = idea["id"]
        reason = _qualifies(idea, iid in bt_by_idea, iid in gd_by_idea)
        if not reason:
            continue
        strat = _promote_idea_to_strategy(idea, reason)
        strat_by_idea[iid] = strat
        stats["strategies"] += 1

        # signature
        sig = idea.get("signal_signature")
        if sig:
            sig_node = store.upsert_node(
                "signature", slug=f"signature-{_slug_hash(sig)}",
                title=(sig[:80]), domain="signature", summary=sig[:500],
                ingestion_version=INGESTION_VERSION,
            )
            store.add_edge(strat, sig_node, "shares_signature", weight=1.0,
                           origin="heuristic", count_evidence=True)
            stats["signatures"] += 1

        # leaves used (parser-honesty visibility)
        for leaf_name in _leaves_used_by(idea):
            leaf_node = store.upsert_node(
                "leaf", slug=f"leaf-{leaf_name}", title=leaf_name, domain="dsl",
                ingestion_version=INGESTION_VERSION)
            if store.add_edge(strat, leaf_node, "uses_leaf", weight=1.0, origin="heuristic"):
                stats["leaf_edges"] += 1

    # 2) backtests → produced
    for iid, runs in bt_by_idea.items():
        if iid not in live_idea_ids:
            stats["skipped_orphans"] += len(runs)
            continue
        strat = strat_by_idea.get(iid)
        if strat is None:
            continue
        for run in runs:
            passed = run.get("passed")
            sharpe = run.get("net_sharpe") if run.get("net_sharpe") is not None else run.get("test_sharpe")
            bt = store.upsert_node(
                "backtest_run", slug=f"backtest-{run['id']}",
                title=f"{run.get('run_type') or 'backtest'} {run.get('pair') or ''} {run.get('timeframe') or ''}".strip(),
                domain="evidence",
                summary=f"net_sharpe={sharpe} max_dd={run.get('max_dd')} trades={run.get('trades')} passed={passed} verdict={run.get('verdict')}",
                ref=("backtest_runs", run["id"]), ingestion_version=INGESTION_VERSION,
            )
            store.add_edge(strat, bt, "produced", weight=1.0, origin="heuristic")
            stats["backtests"] += 1

    # 3) gate decisions → failed|passed
    for iid, decisions in gd_by_idea.items():
        if iid not in live_idea_ids:
            stats["skipped_orphans"] += len(decisions)
            continue
        strat = strat_by_idea.get(iid)
        if strat is None:
            continue
        for gd in decisions:
            decision = (gd.get("decision") or "").lower()
            rel = "passed" if decision in ("approve", "pass", "passed", "promote") else "failed"
            gnode = store.upsert_node(
                "gate_decision", slug=f"gate-{gd['id']}",
                title=f"{gd.get('gate') or 'gate'}: {gd.get('decision')}",
                domain="evidence",
                summary=(gd.get("rationale") or "")[:500],
                ref=("gate_decisions", gd["id"]), ingestion_version=INGESTION_VERSION,
            )
            store.add_edge(strat, gnode, rel, weight=1.0, origin="heuristic")
            stats["gate_decisions"] += 1

    return stats


# ── Slice 3: findings ────────────────────────────────────────────────────────

def ingest_findings() -> dict:
    """Promote governance_findings → finding nodes + reported_by(agent) /
    exposed_to(risk) edges.

    Dedupes by (agent, level, scope, severity, status) rather than by row id:
    the inspectors' record() path now only inserts a new governance_findings
    row on an actual state change (see governance/base.py), but this ingester
    stays defense-in-depth against any inspector that bypasses record() or
    against pre-existing duplicate rows — same (agent, level, scope, severity,
    status) always maps to the same finding node instead of minting one node
    per row."""
    stats = {"findings": 0, "agents": 0, "risk_edges": 0}
    with db_session() as conn:
        findings = [dict(r) for r in conn.execute("SELECT * FROM governance_findings").fetchall()]

    seen_agents: dict[str, int] = {}
    seen_finding_slugs: set[str] = set()
    for f in findings:
        fid = f["id"]
        agent = (f.get("agent") or "unknown_agent")
        level = f.get("level") or ""
        scope = f.get("scope") or ""
        severity = f.get("severity") or ""
        status = f.get("status") or ""
        blob = " ".join(str(f.get(k) or "") for k in
                        ("scope", "evidence", "local_recommendation", "escalate_to")).lower()
        fslug = f"finding-{_slug_hash('|'.join((agent, level, scope, severity, status)))}"
        fnode = store.upsert_node(
            "finding", slug=fslug,
            title=f"{agent}: {severity} ({status})".strip(),
            domain="governance",
            summary=(f.get("evidence") or f.get("local_recommendation") or "")[:500],
            tags=[t for t in (level, severity, status) if t],
            ref=("governance_findings", fid), ingestion_version=INGESTION_VERSION,
        )
        if fslug not in seen_finding_slugs:
            seen_finding_slugs.add(fslug)
            stats["findings"] += 1

        if agent not in seen_agents:
            seen_agents[agent] = store.upsert_node(
                "agent", slug=f"agent-{_slug_hash(agent)}", title=agent,
                domain="governance", ingestion_version=INGESTION_VERSION)
            stats["agents"] += 1
        store.add_edge(fnode, seen_agents[agent], "reported_by", weight=1.0, origin="heuristic")

        for risk_name, hints in _FINDING_RISK_HINTS.items():
            if any(h in blob for h in hints):
                store.add_edge(fnode, _risk_node(risk_name), "exposed_to",
                               weight=1.0, origin="heuristic")
                stats["risk_edges"] += 1
    return stats


def ingest_all() -> dict:
    """Run every deterministic channel. Safe to call repeatedly."""
    result = {"leaves": ingest_leaves()}
    result.update(ingest_strategies_and_evidence())
    result.update(ingest_findings())
    logger.info(f"[EvidenceGraph] {result}")
    return result


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(ingest_all())


if __name__ == "__main__":
    main()
