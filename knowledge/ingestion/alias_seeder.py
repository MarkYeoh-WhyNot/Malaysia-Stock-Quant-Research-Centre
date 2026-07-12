#!/usr/bin/env python3
"""Deterministic alias seeding for entity resolution (Slice 2).

Maps variant spellings to a canonical form so retrieval treats "Bitcoin", "XBT",
and "BTCUSDT" as BTC, or "DPSR" as the deflated PSR. Deterministic sources only;
LLM-suggested aliases (origin='llm') are added elsewhere as candidates and never
auto-trusted.

Aliases carry a canonical STRING (useful for query expansion even before a
canonical node exists) and an optional node_id (resolved when a matching node
is present). Idempotent: re-seeding overwrites the same alias rows.
"""
import logging
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data.database import db_session

logger = logging.getLogger(__name__)

# Hardcoded metric / concept abbreviations (authoritative).
METRIC_ALIASES = {
    "dpsr": "deflated_probabilistic_sharpe_ratio",
    "deflated psr": "deflated_probabilistic_sharpe_ratio",
    "psr": "probabilistic_sharpe_ratio",
    "ic": "information_coefficient",
    "adv": "average_daily_value",
    "cagr": "compound_annual_growth_rate",
    "dd": "max_drawdown",
    "mdd": "max_drawdown",
    "cas": "capacity_adjusted_sharpe",
    "oos": "out_of_sample",
    "is": "in_sample",
}

# Crypto symbol variants → canonical base symbol.
CRYPTO_ALIASES = {
    "bitcoin": "BTC", "xbt": "BTC", "btcusdt": "BTC", "btc perp": "BTC", "btc-perp": "BTC",
    "ethereum": "ETH", "ethusdt": "ETH", "eth perp": "ETH",
    "solana": "SOL", "solusdt": "SOL",
}


def _upsert_alias(conn, alias: str, canonical: str, alias_type: str,
                  origin: str = "human", confidence: float = 1.0, node_id=None):
    conn.execute("""
        INSERT INTO kb_aliases (alias, canonical, node_id, alias_type, confidence, origin)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(alias) DO UPDATE SET
            canonical=excluded.canonical, alias_type=excluded.alias_type,
            confidence=excluded.confidence, origin=excluded.origin,
            node_id=COALESCE(excluded.node_id, kb_aliases.node_id)
    """, (alias.strip().lower(), canonical, node_id, alias_type, confidence, origin))


def seed_aliases() -> dict:
    stats = {"metric": 0, "crypto": 0, "bursa": 0}
    with db_session() as conn:
        for alias, canon in METRIC_ALIASES.items():
            _upsert_alias(conn, alias, canon, "metric"); stats["metric"] += 1
        for alias, canon in CRYPTO_ALIASES.items():
            _upsert_alias(conn, alias, canon, "ticker"); stats["crypto"] += 1

        # Bursa: name → ticker, and ticker without the .KL suffix → ticker.
        try:
            rows = conn.execute("SELECT ticker, name FROM stock_universe").fetchall()
        except Exception:
            rows = []
        for r in rows:
            ticker = (r["ticker"] or "").strip()
            if not ticker:
                continue
            if r["name"]:
                _upsert_alias(conn, r["name"], ticker, "ticker"); stats["bursa"] += 1
            if ticker.upper().endswith(".KL"):
                _upsert_alias(conn, ticker[:-3], ticker, "ticker"); stats["bursa"] += 1

    logger.info(f"[AliasSeeder] {stats}")
    return stats


def resolve(term: str) -> str:
    """Return the canonical form for a term, or the term unchanged."""
    if not term:
        return term
    with db_session() as conn:
        row = conn.execute("SELECT canonical FROM kb_aliases WHERE alias=?",
                           (term.strip().lower(),)).fetchone()
    return row["canonical"] if row and row["canonical"] else term


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(seed_aliases())


if __name__ == "__main__":
    main()
