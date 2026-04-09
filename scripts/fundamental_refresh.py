#!/usr/bin/env python
"""
fundamental_refresh.py — one-shot KLCI fundamental data refresh.

Scrapes klsescreener.com stock pages for all SLUG_MAP stocks and upserts
into fundamental_data, quarterly_history, and dividend_history tables.

Usage:
    PYTHONPATH=/opt/openclaw/app /opt/openclaw/venv/bin/python scripts/fundamental_refresh.py
"""
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fundamental_refresh")

from data.database import init_db
from data.klse_screener.fundamental_scraper import KLSEFundamentalScraper

if __name__ == "__main__":
    init_db()
    logger.info("Starting KLCI fundamental refresh...")
    result = KLSEFundamentalScraper().refresh_all_klci()
    logger.info(
        f"Done — success={result['success']} failed={result['failed']}"
    )
    for s in result["stocks"]:
        status = "✓" if s["ok"] else "✗"
        print(f"  {status} {s['ticker']}")
