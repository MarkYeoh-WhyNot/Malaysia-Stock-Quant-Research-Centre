"""Read-only audit of RejectionMemory's _classify() keyword buckets (P2-5,
2026-07-13 self-audit). Idea #218's "irrelevant" mislabeling (fixed earlier
today) was the ONE bucket bug we happened to trip over — this pulls every
strategy_cemetery.rejection_reason and re-runs the SAME classifier so the
other buckets get the same scrutiny, instead of staying unaudited until
something else breaks by accident.

Usage:
    python scripts/audit_rejection_buckets.py            # group + sample
    python scripts/audit_rejection_buckets.py --full      # print every row, not just a sample
    python scripts/audit_rejection_buckets.py --bucket low_sharpe   # one bucket only

Never writes to the DB — pulls strategy_cemetery.rejection_reason and
re-classifies it in memory only.
"""
import argparse
import sys
from collections import defaultdict

from data.database import db_session
from knowledge.ingestion.rejection_memory import _REASON_CATEGORY_KEYWORDS, _classify


def audit(sample_size: int = 8) -> dict:
    with db_session() as conn:
        rows = conn.execute(
            "SELECT id, idea_id, rejection_reason FROM strategy_cemetery "
            "WHERE rejection_reason IS NOT NULL AND rejection_reason != '' "
            "ORDER BY id"
        ).fetchall()

    buckets = defaultdict(list)
    matched_kw_counts = defaultdict(lambda: defaultdict(int))
    for row in rows:
        text = row["rejection_reason"]
        label, matched_kw = _classify(text, _REASON_CATEGORY_KEYWORDS, "other")
        buckets[label].append({"id": row["id"], "idea_id": row["idea_id"], "text": text,
                               "matched_kw": matched_kw})
        if matched_kw:
            matched_kw_counts[label][matched_kw] += 1

    return {"total": len(rows), "buckets": buckets, "matched_kw_counts": matched_kw_counts}


def _print_report(result: dict, sample_size: int, only_bucket: str | None, full: bool):
    print(f"Total strategy_cemetery rows with rejection_reason: {result['total']}\n")
    buckets = result["buckets"]
    order = sorted(buckets, key=lambda b: -len(buckets[b]))
    for label in order:
        if only_bucket and label != only_bucket:
            continue
        entries = buckets[label]
        print(f"=== {label} ({len(entries)} rows) ===")
        kw_counts = result["matched_kw_counts"].get(label, {})
        if kw_counts:
            top_kw = sorted(kw_counts.items(), key=lambda kv: -kv[1])
            print("  matched keyword breakdown: " +
                  ", ".join(f"{kw!r}={n}" for kw, n in top_kw))
        shown = entries if full else entries[:sample_size]
        for e in shown:
            print(f"  [idea {e['idea_id']}, cemetery id {e['id']}, "
                  f"kw={e['matched_kw']!r}] {e['text'][:220]}")
        if not full and len(entries) > sample_size:
            print(f"  ... and {len(entries) - sample_size} more (use --full to see all)")
        print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="print every row, not just a sample")
    parser.add_argument("--bucket", default=None, help="restrict output to one bucket label")
    parser.add_argument("--sample-size", type=int, default=8)
    args = parser.parse_args()

    result = audit(args.sample_size)
    if result["total"] == 0:
        print("No strategy_cemetery rows with rejection_reason found in this DB.")
        return
    _print_report(result, args.sample_size, args.bucket, args.full)


if __name__ == "__main__":
    main()
