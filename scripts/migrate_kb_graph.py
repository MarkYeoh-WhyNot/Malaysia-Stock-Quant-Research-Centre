#!/usr/bin/env python3
"""Run the KB → GraphRAG migration (idempotent; safe to re-run).

Usage: python scripts/migrate_kb_graph.py
"""
import logging
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(_REPO_ROOT, ".env"))

from data.database import init_db
from knowledge.graph.migrate import migrate_kb_graph


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    init_db()
    counts = migrate_kb_graph()
    print(f"Migration complete: {counts}")


if __name__ == "__main__":
    main()
