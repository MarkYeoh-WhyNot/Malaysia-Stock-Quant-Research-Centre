#!/usr/bin/env python3
"""
One-off migration: extract valid .KL tickers from malformed ticker strings in alpha_ideas.

Fixes rows where the ticker field contains description strings like:
  "Healthcare sector (e.g., IHH Healthcare 5225.KL, KPJ Healthcare 5878.KL)"
  → "5225.KL,5878.KL,7081.KL"

Also resets stage2/failed ideas whose failure was caused by bad tickers back to
status='pending' so the daemon retries them with the correct ticker.

Run once:
  /opt/openclaw/venv/bin/python scripts/fix_ticker_strings.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.database import db_session, init_db
from data.yahoo.client import extract_tickers


def fix_malformed_tickers():
    init_db()

    # Detect rows that clearly contain description strings (not clean .KL codes)
    with db_session() as conn:
        rows = conn.execute(
            "SELECT id, title, ticker, stage, status FROM alpha_ideas "
            "WHERE ticker IS NOT NULL AND LENGTH(ticker) > 8 AND ("
            "  ticker LIKE '%sector%' OR ticker LIKE '%e.g.%' "
            "  OR ticker LIKE '% (%' OR ticker LIKE '%Healthcare%' "
            "  OR ticker LIKE '%Banking%' OR ticker LIKE '%Plantation%' "
            "  OR ticker LIKE '%Utility%' OR ticker LIKE '%Telecom%' "
            "  OR (ticker NOT LIKE '%.KL' AND ticker NOT LIKE '%,.KL%' AND ticker LIKE '% %')"
            ")"
        ).fetchall()

    if not rows:
        print("No malformed tickers found — nothing to do.")
        return

    print(f"Found {len(rows)} ideas with potentially malformed tickers:\n")
    updated = 0
    skipped = 0

    for row in rows:
        old_ticker = row["ticker"]
        candidates = extract_tickers(old_ticker)

        # If extract_tickers fell back to the raw string, no .KL codes were found
        if len(candidates) == 1 and candidates[0] == old_ticker.strip():
            print(f"  [SKIP] id={row['id']:3d} — no .KL found in: '{old_ticker[:70]}'")
            skipped += 1
            continue

        new_ticker = ",".join(candidates[:5])  # max 5 tickers

        if new_ticker == old_ticker:
            continue

        print(f"  [FIX]  id={row['id']:3d} '{row['title'][:55]}'")
        print(f"         OLD: {old_ticker[:80]}")
        print(f"         NEW: {new_ticker}")
        print(f"         stage={row['stage']} status={row['status']}")

        with db_session() as conn:
            conn.execute(
                "UPDATE alpha_ideas SET ticker=?, updated_at=datetime('now') WHERE id=?",
                (new_ticker, row["id"]),
            )
        updated += 1

    print(f"\nTicker fix: {updated} updated, {skipped} skipped.\n")

    # Reset ideas that failed purely due to bad ticker → retry them
    with db_session() as conn:
        cur = conn.execute(
            "UPDATE alpha_ideas "
            "SET status='pending', rejection_reason=NULL, updated_at=datetime('now') "
            "WHERE stage='stage2' AND status='failed' "
            "AND (rejection_reason LIKE '%Insufficient historical data%' "
            "  OR rejection_reason LIKE '%0 bars%' "
            "  OR rejection_reason LIKE '%Backtest returned error%')"
        )
        reset_count = cur.rowcount

    if reset_count:
        print(f"Reset {reset_count} stage2/failed idea(s) to pending (bad-ticker failures → retry).")
    else:
        print("No stage2/failed ideas needed resetting.")


if __name__ == "__main__":
    fix_malformed_tickers()
