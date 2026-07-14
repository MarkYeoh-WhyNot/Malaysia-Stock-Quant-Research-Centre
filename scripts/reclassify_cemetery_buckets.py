"""One-off backfill (self-audit follow-up task 4, 2026-07-13, Mark approved):
re-run the FIXED _classify() against every strategy_cemetery row's stored
rejection_reason and update revival_conditions/classified_by to match.

Scope, deliberately limited: only touches rows NOT already classified_by
'explicit:*' — those went through the new score-based
ResearchDaemon._gate0_reason_category() path (P2-5/P2-6 fix) and are already
correct. Rows with classified_by NULL or a bare keyword predate that fix and
carry the ~88% (Bursa) / ~18% (crypto) "overfitting" over-capture documented
in docs/audit_log.md. This is a keyword-fallback correction only — it cannot
fully reproduce the new logic since old rows never stored the structured
gate0 scores (data_quality_score, overfitting_risk, feasibility) the new path
reads. It only changes revival_conditions text + classified_by trace column;
rejection_patterns aggregate counts are NOT touched (separate table, out of
scope for this pass).

Usage:
    python scripts/reclassify_cemetery_buckets.py            # dry run, report only
    python scripts/reclassify_cemetery_buckets.py --apply    # write changes
"""
import argparse
from collections import defaultdict

from data.database import db_session
from knowledge.ingestion.rejection_memory import (
    _REASON_CATEGORY_KEYWORDS, _REVIVAL_CONDITIONS, _classify,
)


def _eligible_rows(conn):
    return conn.execute(
        "SELECT id, idea_id, rejection_reason, revival_conditions, classified_by "
        "FROM strategy_cemetery "
        "WHERE rejection_reason IS NOT NULL AND rejection_reason != '' "
        "AND (classified_by IS NULL OR classified_by NOT LIKE 'explicit:%') "
        "ORDER BY id"
    ).fetchall()


def run(apply: bool):
    with db_session() as conn:
        rows = _eligible_rows(conn)

    before_counts = defaultdict(int)
    after_counts = defaultdict(int)
    changes = []

    for row in rows:
        old_label = row["classified_by"] or "other"
        # revival_conditions is a free-text template value — recover the old
        # label by reverse-matching against _REVIVAL_CONDITIONS instead of
        # re-deriving it (classified_by only stores the matched KEYWORD, not
        # the resulting label, when a keyword fired pre-fix).
        old_derived_label = next(
            (lbl for lbl, txt in _REVIVAL_CONDITIONS.items() if txt == row["revival_conditions"]),
            "other",
        )
        before_counts[old_derived_label] += 1

        new_label, matched_kw = _classify(row["rejection_reason"], _REASON_CATEGORY_KEYWORDS, "other")
        after_counts[new_label] += 1

        if new_label != old_derived_label:
            changes.append({
                "cemetery_id": row["id"], "idea_id": row["idea_id"],
                "old_label": old_derived_label, "new_label": new_label,
                "matched_kw": matched_kw, "text": row["rejection_reason"][:160],
            })

    print(f"Eligible rows (not already gate0-explicit): {len(rows)}")
    print(f"Would reclassify: {len(changes)}\n")
    print("Before:", dict(before_counts))
    print("After: ", dict(after_counts))
    print()
    for c in changes[:30]:
        print(f"  [cemetery {c['cemetery_id']}, idea {c['idea_id']}] "
              f"{c['old_label']} -> {c['new_label']} (kw={c['matched_kw']!r}) {c['text']}")
    if len(changes) > 30:
        print(f"  ... and {len(changes) - 30} more")

    if not apply:
        print("\nDry run only — pass --apply to write changes.")
        return

    with db_session() as conn:
        for c in changes:
            new_revival = _REVIVAL_CONDITIONS.get(c["new_label"], _REVIVAL_CONDITIONS["other"])
            classified_by = c["matched_kw"] if c["matched_kw"] else None
            conn.execute(
                "UPDATE strategy_cemetery SET revival_conditions=?, classified_by=? WHERE id=?",
                (new_revival, classified_by, c["cemetery_id"]),
            )
    print(f"\nApplied {len(changes)} update(s).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    args = parser.parse_args()
    run(args.apply)
