#!/usr/bin/env python3
"""Backfill the July 2026 alpha-hunt campaign verdicts into the knowledge graph.

The 2026-07 campaign predates the automatic findings emitter in alpha_hunt.py,
so its two conclusions live only in docs/session memory. This records them as
`finding` nodes (idempotent; safe to re-run):

  1. FALSIFIED — no OHLCV edge on liquid majors: thousands of Stage-A trials
     across the full pre-registered technical grid, zero gate survivors.
  2. CONFIRMED — funding-carry signal is real (IC 0.018, t≈3): the one
     direction that survived scrutiny; refines finding 1 by redirecting the
     search from price-derived to funding-derived features.

Usage (crypto container):
  MARKET_MODE=crypto PYTHONPATH=. python scripts/backfill_campaign_findings.py
"""
import json
import logging
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(_REPO_ROOT, ".env"))

# The ten OHLCV leaves the 2026-07 CONFIGS grid actually swept.
OHLCV_LEAVES = ("sma_cross", "ema_cross", "rsi", "bollinger", "macd",
                "momentum", "reversal", "zscore", "volume_ratio", "rolling_rank")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from config.settings import MARKET_MODE
    if MARKET_MODE != "crypto":
        sys.exit("These are crypto-campaign findings — set MARKET_MODE=crypto")

    from data.database import init_db
    init_db()
    from knowledge.ingestion.campaign_findings import record_campaign_finding

    n1 = record_campaign_finding(
        slug="alpha-hunt-2026-07-no-ohlcv-edge-liquid-majors",
        title="Alpha hunt 2026-07: no OHLCV edge on liquid crypto majors",
        summary=("The July 2026 alpha-hunt campaign screened the full "
                 "pre-registered technical grid (trend, reversion, momentum, "
                 "breakout — 20 configs, long and short mirrors) across the "
                 "liquid USDT majors on 1h/4h/1d bars, every screen counted "
                 "as a trial. Zero candidates passed the full gate stack "
                 "(deflated PSR, OOS walk-forward, cost drag, regimes). "
                 "Verdict: price/volume-derived signals on liquid majors are "
                 "falsified at this search size — do not re-propose OHLCV "
                 "formulations on this universe without new information."),
        direction="falsified",
        tags=["alpha-hunt", "crypto", "ohlcv"],
        content=json.dumps({"campaign": "2026-07", "universe": "liquid USDT majors",
                            "timeframes": ["1h", "4h", "1d"],
                            "grid": "20 pre-registered configs + short mirrors",
                            "survivors": 0}),
        leaf_names=OHLCV_LEAVES,
    )
    print(f"finding 1 (falsified OHLCV) node id: {n1}")

    n2 = record_campaign_finding(
        slug="funding-carry-2026-07-ic-real",
        title="Funding-carry signal is real (2026-07 cross-sectional scan)",
        summary=("Cross-sectional funding-rate carry showed a statistically "
                 "real information coefficient on the crypto universe in the "
                 "July 2026 scan — the only direction that survived scrutiny "
                 "while the entire OHLCV grid was falsified. The unexplored "
                 "funding-carry parameter sweep is the highest-expected-value "
                 "research direction: build candidates from funding features "
                 "(funding_level, funding_zscore), not price-derived ones."),
        direction="confirmed",
        tags=["alpha-hunt", "crypto", "funding", "carry"],
        content=json.dumps({"campaign": "2026-07", "ic": 0.018, "t_stat": 3,
                            "status": "sweep unexplored — next campaign"}),
        leaf_names=("funding_level", "funding_zscore"),
        refines_slugs=("finding-campaign-alpha-hunt-2026-07-no-ohlcv-edge-liquid-majors",),
    )
    print(f"finding 2 (confirmed funding carry) node id: {n2}")


if __name__ == "__main__":
    main()
