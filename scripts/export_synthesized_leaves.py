"""Recover LeafSynthesizer-approved leaves for manual git commit.

Production containers have no git binary, so approved leaves are never
actually committed there (2026-07-13 self-audit) — they only exist on the
runtime volume and in leaf_synthesis_attempts.module_source. This is the
Mac-side recovery tool: pull an approved leaf + its test out of the
database (the guaranteed-to-survive copy) and write them to the repo's real
paths, ready for `git add` + commit.

Usage:
    python scripts/export_synthesized_leaves.py                # all approved, not yet exported
    python scripts/export_synthesized_leaves.py --leaf-name foo # one specific leaf
    python scripts/export_synthesized_leaves.py --all           # re-export every approved leaf
"""
import argparse
import logging
import os

from config.settings import BASE_DIR
from data.database import db_session

logger = logging.getLogger(__name__)

_GENERATED_DIR = os.path.join(str(BASE_DIR), "agents", "backtest_engineer", "leaves_generated")
_TESTS_DIR = os.path.join(str(BASE_DIR), "tests")


def _approved_rows(leaf_name: str | None, include_all: bool) -> list:
    with db_session() as conn:
        if leaf_name:
            rows = conn.execute(
                "SELECT * FROM leaf_synthesis_attempts WHERE status='approved' "
                "AND leaf_name=? ORDER BY id DESC LIMIT 1", (leaf_name,)).fetchall()
        elif include_all:
            rows = conn.execute(
                "SELECT * FROM leaf_synthesis_attempts WHERE status='approved' "
                "AND module_source IS NOT NULL ORDER BY id").fetchall()
        else:
            # Only leaves missing from the local checkout — the common case
            # after a fresh clone or a VPS rebuild that dropped the image layer.
            rows = conn.execute(
                "SELECT * FROM leaf_synthesis_attempts WHERE status='approved' "
                "AND module_source IS NOT NULL ORDER BY id").fetchall()
            rows = [r for r in rows if not os.path.exists(
                os.path.join(_GENERATED_DIR, f"{r['leaf_name']}.py"))]
    return rows


def export_leaves(leaf_name: str | None = None, include_all: bool = False) -> list[str]:
    os.makedirs(_GENERATED_DIR, exist_ok=True)
    os.makedirs(_TESTS_DIR, exist_ok=True)
    written = []
    for row in _approved_rows(leaf_name, include_all):
        name = row["leaf_name"]
        if not row["module_source"]:
            logger.warning(f"[Export] '{name}' has no module_source recorded — skipped")
            continue
        leaf_path = os.path.join(_GENERATED_DIR, f"{name}.py")
        with open(leaf_path, "w") as fh:
            fh.write(row["module_source"])
        written.append(leaf_path)
        logger.info(f"[Export] Wrote {leaf_path}")
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leaf-name", default=None, help="Export only this leaf")
    parser.add_argument("--all", action="store_true", dest="include_all",
                        help="Re-export every approved leaf, not just missing ones")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    written = export_leaves(args.leaf_name, args.include_all)
    if not written:
        print("Nothing to export — no approved leaves missing from the local checkout.")
        return
    print(f"Exported {len(written)} leaf file(s):")
    for path in written:
        print(f"  {path}")
    print("\nNote: this recovers the compute leaf only (test files aren't stored in "
          "module_source — pull test_code from leaf_synthesis_attempts.spec_json's "
          "worked_example if you need to regenerate the test, or copy from the "
          "runtime volume's leaves_generated/tests/ if it's still reachable).")
    print("Review, then: git add agents/backtest_engineer/leaves_generated/ && git commit")


if __name__ == "__main__":
    main()
