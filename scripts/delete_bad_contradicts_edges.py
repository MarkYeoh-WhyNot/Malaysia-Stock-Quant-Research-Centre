"""One-off cleanup (self-audit follow-up task 4, 2026-07-13, Mark approved):
delete every LLM-origin `contradicts` edge from the knowledge graph.

The 2026-07-13 audit (docs/audit_log.md) hand-checked a representative sample
of every live `contradicts` edge in both markets and found 100% are
origin='llm', with confidence weight uncorrelated with correctness — wrong at
0.88-0.95 weight just as often as low weight. `pipeline/revisit.py`'s
contradicting-finding trigger was already fixed (P2-6) to never read
origin='llm' edges regardless of weight, so these edges are functionally
inert — this cleanup just removes them from the KG explorer/dashboard view
where they'd otherwise still show up as noise.

Only deletes relation='contradicts' AND origin='llm'. Heuristic-origin edges
(knowledge/ingestion/campaign_findings.py's deterministic output) and every
other relation type are untouched.

Usage:
    python scripts/delete_bad_contradicts_edges.py            # dry run, report only
    python scripts/delete_bad_contradicts_edges.py --apply    # write changes
"""
import argparse

from data.database import db_session


def run(apply: bool):
    with db_session() as conn:
        rows = conn.execute(
            "SELECT e.id, e.weight, s.slug AS src, t.slug AS tgt "
            "FROM kb_edges e JOIN kb_nodes s ON s.id=e.source_id "
            "JOIN kb_nodes t ON t.id=e.target_id "
            "WHERE e.relation='contradicts' AND e.origin='llm'"
        ).fetchall()

    print(f"LLM-origin contradicts edges: {len(rows)}")
    for r in rows[:20]:
        print(f"  [edge {r['id']}, weight={r['weight']}] {r['src']} -> {r['tgt']}")
    if len(rows) > 20:
        print(f"  ... and {len(rows) - 20} more")

    if not apply:
        print("\nDry run only — pass --apply to delete.")
        return

    with db_session() as conn:
        conn.execute("DELETE FROM kb_edges WHERE relation='contradicts' AND origin='llm'")
    print(f"\nDeleted {len(rows)} edge(s).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="delete (default: dry run)")
    args = parser.parse_args()
    run(args.apply)
