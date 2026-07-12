#!/usr/bin/env python3
"""Knowledge-graph health check — the anti-garbage governance job (design §8).

Enforces the "no claim without a source / stay disciplined" rules as assertions
and reports violations (logged, and optionally written back to governance_findings
so the graph audits itself). Deterministic, read-mostly, cheap; meant to run daily.

Checks:
  - every node_type is registered in kb_node_type_registry              [BLOCKER]
  - every edge relation is in store.RELATIONS                           [BLOCKER]
  - evidence nodes (backtest_run/gate_decision/finding) carry a ref     [WARN]
  - duplicate node titles within a type → alias-resolution candidates   [INFO]
  - active signatures with < MIN_SIG_EDGES supporting edges (orphans)   [INFO]

Usage: python scripts/graph_health_check.py [--record]
"""
import logging
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data.database import db_session
from knowledge.graph import store

logger = logging.getLogger(__name__)

MIN_SIG_EDGES = 1  # active signature with fewer supporting edges is an orphan
EVIDENCE_TYPES = ("backtest_run", "gate_decision", "finding")


def run_health_check(record: bool = False) -> dict:
    findings: list[dict] = []

    def flag(severity, check, detail):
        findings.append({"severity": severity, "check": check, "detail": detail})

    with db_session() as conn:
        registry = {r["node_type"] for r in conn.execute(
            "SELECT node_type FROM kb_node_type_registry")}

        # 1) unregistered node types (BLOCKER)
        for r in conn.execute(
            "SELECT node_type, COUNT(*) n FROM kb_nodes GROUP BY node_type"):
            if r["node_type"] not in registry:
                flag("BLOCKER", "unregistered_node_type",
                     f"{r['n']} node(s) of type {r['node_type']!r} not in registry")

        # 2) unknown edge relations (BLOCKER)
        for r in conn.execute(
            "SELECT relation, COUNT(*) n FROM kb_edges GROUP BY relation"):
            if r["relation"] not in store.RELATIONS:
                flag("BLOCKER", "unknown_relation",
                     f"{r['n']} edge(s) with relation {r['relation']!r} not in RELATIONS")

        # 3) evidence nodes missing a source ref (WARN)
        marks = ",".join("?" * len(EVIDENCE_TYPES))
        row = conn.execute(
            f"SELECT COUNT(*) n FROM kb_nodes WHERE node_type IN ({marks}) "
            f"AND (ref_table IS NULL OR ref_id IS NULL)", EVIDENCE_TYPES).fetchone()
        if row["n"]:
            flag("WARN", "evidence_without_source",
                 f"{row['n']} evidence node(s) lack ref_table/ref_id")

        # 4) duplicate titles within a type (INFO → alias candidates)
        dups = conn.execute("""
            SELECT node_type, title, COUNT(*) n FROM kb_nodes
            WHERE title IS NOT NULL AND title <> ''
            GROUP BY node_type, title HAVING COUNT(*) > 1
        """).fetchall()
        if dups:
            flag("INFO", "duplicate_titles",
                 f"{len(dups)} (type,title) groups duplicated → alias candidates")

        # 5) orphan signatures (INFO)
        orphans = conn.execute(f"""
            SELECT COUNT(*) n FROM kb_nodes s
            WHERE s.node_type='signature'
              AND (SELECT COUNT(*) FROM kb_edges e
                   WHERE e.source_id=s.id OR e.target_id=s.id) < {MIN_SIG_EDGES}
        """).fetchone()
        if orphans["n"]:
            flag("INFO", "orphan_signatures",
                 f"{orphans['n']} signature node(s) with < {MIN_SIG_EDGES} edges")

    blockers = [f for f in findings if f["severity"] == "BLOCKER"]
    for f in findings:
        logf = logger.error if f["severity"] == "BLOCKER" else logger.info
        logf(f"[GraphHealth] {f['severity']} {f['check']}: {f['detail']}")

    if record and findings:
        with db_session() as conn:
            for f in findings:
                conn.execute("""
                    INSERT INTO governance_findings
                        (agent, level, scope, status, severity, evidence, local_recommendation)
                    VALUES ('GraphHealthCheck','L0','knowledge_graph',?,?,?,?)
                """, ("FAIL" if f["severity"] == "BLOCKER" else "PASS",
                      "BLOCKER" if f["severity"] == "BLOCKER" else "INFO",
                      f["detail"], f"resolve {f['check']}"))

    result = {"findings": len(findings), "blockers": len(blockers),
              "detail": findings}
    logger.info(f"[GraphHealth] {result['findings']} findings, {result['blockers']} blockers")
    return result


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    record = "--record" in sys.argv
    out = run_health_check(record=record)
    print({"findings": out["findings"], "blockers": out["blockers"]})
    sys.exit(1 if out["blockers"] else 0)


if __name__ == "__main__":
    main()
