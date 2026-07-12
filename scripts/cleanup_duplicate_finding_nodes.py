#!/usr/bin/env python3
"""One-time cleanup: collapse duplicate `finding` kb_nodes down to distinct
governance states (state-transitions only).

Context: governance_findings previously got a new row every daemon cycle
(~60s) even when the verdict didn't change, and knowledge/ingestion/
evidence_graph.py's ingest_findings() used to key each finding node's slug
off the row id (always unique) instead of its content — so the graph
accumulated one node per cycle instead of one node per distinct verdict.

Two upstream fixes now prevent new duplicates:
  - governance/base.py Inspector.record() only inserts a new
    governance_findings row on an actual state change.
  - evidence_graph.ingest_findings() now keys the finding-node slug off
    (agent, level, scope, severity, status) content, not row id.

This script cleans up the *existing* backlog: finding nodes are entirely
derived from governance_findings (not LLM output, no human edits), so it is
safe to wipe the finding-node namespace and rebuild it from scratch — the
rebuild naturally collapses to one node per distinct (agent, level, scope,
severity, status) combination that ever occurred, i.e. state transitions
only, using the same ingest_findings() path the daemon already runs.

Does NOT touch campaign_findings.py / the finding-campaign-* slug namespace
(a separate, unrelated feature).

Usage:
    python scripts/cleanup_duplicate_finding_nodes.py           # dry run (default)
    python scripts/cleanup_duplicate_finding_nodes.py --apply   # actually delete + rebuild
"""
import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data.database import db_session
from knowledge.ingestion import evidence_graph


def _finding_node_count() -> int:
    with db_session() as conn:
        return conn.execute(
            "SELECT COUNT(*) AS c FROM kb_nodes WHERE node_type='finding' "
            "AND slug NOT LIKE 'finding-campaign-%'"
        ).fetchone()["c"]


def _wipe_finding_nodes() -> int:
    # Governance-derived nodes only (slug 'finding-<hash>', ref_table=
    # 'governance_findings') — finding-campaign-* (campaign_findings.py,
    # alpha_hunt verdicts) is a disjoint namespace with no governance_findings
    # row behind it; ingest_findings() never rebuilds it, so wiping it here
    # would delete it permanently.
    with db_session() as conn:
        ids = [r["id"] for r in conn.execute(
            "SELECT id FROM kb_nodes WHERE node_type='finding' "
            "AND slug NOT LIKE 'finding-campaign-%'"
        ).fetchall()]
        for nid in ids:
            conn.execute("DELETE FROM kb_edges WHERE source_id=? OR target_id=?", (nid, nid))
            conn.execute("DELETE FROM kb_fts WHERE node_id=?", (nid,))
            conn.execute("DELETE FROM kb_nodes WHERE id=?", (nid,))
    return len(ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true",
                        help="Actually delete + rebuild. Without this flag, dry-run only.")
    args = parser.parse_args()

    before = _finding_node_count()
    print(f"Existing finding nodes: {before}")

    if not args.apply:
        print("Dry run — no changes made. Re-run with --apply to execute.")
        return

    deleted = _wipe_finding_nodes()
    print(f"Deleted {deleted} finding nodes (+ their edges/fts rows).")

    stats = evidence_graph.ingest_findings()
    after = _finding_node_count()
    print(f"Rebuilt from governance_findings: {stats}")
    print(f"Finding nodes after cleanup: {after} (was {before})")


if __name__ == "__main__":
    main()
