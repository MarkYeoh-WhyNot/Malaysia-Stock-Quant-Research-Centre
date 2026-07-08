#!/usr/bin/env python3
"""
Reclassify all kb_documents using Claude Haiku.

Processes docs in batches of 5 — classifies each using title + first 200 chars
of summary, then updates domain in DB.  Prints before/after counts.

Usage:
    python scripts/reclassify_kb_domains.py   (run from the repo root)
"""

import sys
import os
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
os.environ.setdefault("PYTHONPATH", _REPO_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(_REPO_ROOT, ".env"))

from data.database import db_session
from knowledge.ingestion.kb_ingester import KBIngester, VALID_DOMAINS

BATCH_SIZE = 5


def main():
    ingester = KBIngester()

    # ── Snapshot before ─────────────────────────────────────────────────────
    with db_session() as conn:
        before_rows = conn.execute(
            "SELECT domain, COUNT(*) as n FROM kb_documents GROUP BY domain ORDER BY n DESC"
        ).fetchall()
        all_docs = conn.execute(
            "SELECT id, title, summary FROM kb_documents ORDER BY id"
        ).fetchall()

    print("=== BEFORE reclassification ===")
    for r in before_rows:
        print(f"  {r['domain']}: {r['n']}")
    print(f"  TOTAL: {len(all_docs)}")
    print()

    # ── Reclassify in batches ────────────────────────────────────────────────
    updated = 0
    errors = 0
    changes = []

    for i in range(0, len(all_docs), BATCH_SIZE):
        batch = all_docs[i: i + BATCH_SIZE]
        print(f"Processing batch {i // BATCH_SIZE + 1} "
              f"(docs {i + 1}–{min(i + BATCH_SIZE, len(all_docs))}) ...")

        for doc in batch:
            doc_id = doc["id"]
            title = doc["title"] or ""
            summary = (doc["summary"] or "")[:200]

            try:
                # Fetch old domain for logging
                with db_session() as conn:
                    old_row = conn.execute(
                        "SELECT domain FROM kb_documents WHERE id=?", (doc_id,)
                    ).fetchone()
                old_domain = old_row["domain"] if old_row else "unknown"

                new_domain = ingester.classify_domain(doc_id, title, summary)
                updated += 1
                status = "CHANGED" if new_domain != old_domain else "same"
                print(f"    [{doc_id}] {status}: {old_domain!r} → {new_domain!r} | {title[:55]}")
                if new_domain != old_domain:
                    changes.append((doc_id, old_domain, new_domain, title[:55]))

            except Exception as e:
                errors += 1
                print(f"    [{doc_id}] ERROR: {e}")

    # ── Snapshot after ───────────────────────────────────────────────────────
    with db_session() as conn:
        after_rows = conn.execute(
            "SELECT domain, COUNT(*) as n FROM kb_documents GROUP BY domain ORDER BY n DESC"
        ).fetchall()

    print()
    print("=== AFTER reclassification ===")
    for r in after_rows:
        print(f"  {r['domain']}: {r['n']}")
    print(f"  TOTAL: {sum(r['n'] for r in after_rows)}")
    print()
    print(f"Updated: {updated}  |  Errors: {errors}  |  Domain changes: {len(changes)}")
    if changes:
        print()
        print("=== Changed documents ===")
        for doc_id, old, new, title in changes:
            print(f"  [{doc_id}] {old!r} → {new!r}  {title}")


if __name__ == "__main__":
    main()
