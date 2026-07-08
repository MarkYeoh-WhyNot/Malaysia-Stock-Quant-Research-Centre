#!/usr/bin/env python3
"""
cpo_daily.py — CPO price fetch + plantation lag signal scan

Designed to run via cron at 23:00 UTC = 07:00 MYT (before market open):
    0 23 * * * /opt/openclaw/venv/bin/python /opt/openclaw/app/scripts/cpo_daily.py

Steps:
  1. Fetch latest CPO spot price (MPOB → cpo.com.my → yfinance)
  2. Refresh historical CPO data in cpo_prices table
  3. Compute Spearman lag correlations for all plantation tickers
  4. Auto-create Gate 0 ideas for significant signals (|corr| > 0.35)
  5. Seed KB document about CPO-plantation relationship (once only)
  6. Log all results to daemon_logs
"""
import logging
import os
import sys
from datetime import datetime

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("openclaw.cpo_daily")

from data.database import init_db, db_session
from data.cpo.mpob_scraper import MPOBScraper, PLANTATION_TICKERS, PLANTATION_NAMES
from data.cpo.cpo_signal import CPOSignalGenerator

# ── KB seed text ──────────────────────────────────────────────────────────────
_KB_SEED_TITLE = "CPO Price Lead-Lag Effect on Bursa Plantation Stocks"
_KB_SEED_CONTENT = """
CPO Price Lead-Lag Effect on Bursa Plantation Stocks

Crude palm oil (CPO) spot price is the primary revenue driver for Malaysian plantation
companies listed on Bursa Malaysia. Due to institutional rebalancing delays and retail
investor information processing lags, plantation stock prices typically follow CPO spot
price movements with a 2-5 day lag. This creates a systematic lead-lag signal opportunity.

MECHANISM:
When CPO spot price surges (e.g. +1-2% in a day), plantation companies' forward earnings
expectations improve immediately. However, fund managers take 1-3 days to update their
models, receive internal approval, and execute trades. Retail investors take even longer.
This creates a predictable window during which the stock has not fully priced in the CPO move.

STRONGEST EFFECT:
Pure-play plantation companies show the strongest CPO correlation lag:
- Sime Darby Plantation (5285.KL): ~80% CPO revenue exposure
- Kuala Lumpur Kepong (2445.KL): ~70% plantation revenues

WEAKER EFFECT (due to diversification):
- IOI Corporation (1961.KL): significant downstream/property businesses
- PPB Group (4065.KL): consumer food, cinema — CPO is secondary

OPTIMAL LAG VARIES:
- Bull markets (rising CPO trend): shorter lags (1-2 days) as institutions act faster
- Bear markets / uncertain conditions: longer lags (3-5 days)
- Typical median lag across KLCI plantation stocks: 2-3 trading days

SIGNAL CONSTRUCTION:
1. Compute CPO_1d_return = (CPO_today - CPO_yesterday) / CPO_yesterday
2. If CPO_1d_return > +1.0%, expect plantation stock to rise in {lag} days
3. Entry: buy at open on day T+{lag}
4. Exit: T+{lag+1} close or CPO reversal < -1.5%
5. Stop-loss: -6% from entry

TRANSACTION COST CONSIDERATION:
Bursa stamp duty 0.10% (buy-side) + brokerage ~0.10% = ~0.20% per side = ~0.40% round trip.
Strategy only viable when |CPO_1d_return| > 1.5% to overcome frictional costs.

CAVEATS:
- CPO futures roll dates can distort spot prices (last trading day effects)
- Ringgit/USD moves can decouple plantation stocks from CPO temporarily
- EPF rebalancing (semi-annual) overwhelms CPO signal in index heavyweights
- Earnings season (Feb/May/Aug/Nov): stock-specific news dominates

REFERENCES:
- Bursa Malaysia CPO futures contract specification (FCPO)
- MPOB monthly palm oil statistics
- Academic: "Commodity Price Transmission to Stock Returns in Emerging ASEAN Markets"
""".strip()


def _seed_kb_document():
    """Ingest the CPO-plantation KB seed document (once only — checks by title)."""
    try:
        with db_session() as conn:
            exists = conn.execute(
                "SELECT id FROM kb_documents WHERE title=?",
                (_KB_SEED_TITLE,),
            ).fetchone()
        if exists:
            logger.info(f"KB seed already exists (doc_id={exists['id']}) — skipping")
            return

        from knowledge.ingestion.kb_ingester import KBIngester
        kb     = KBIngester()
        result = kb.ingest_text(
            content=_KB_SEED_CONTENT,
            title=_KB_SEED_TITLE,
            domain="commodity",
            source_url="cpo_daily:seed",
        )
        logger.info(
            f"KB seed ingested: doc_id={result.get('doc_id')} "
            f"relevance={result.get('relevance_score', 0):.2f} "
            f"({result.get('relevance_category', '?')})"
        )
    except Exception as e:
        logger.warning(f"KB seed ingestion failed (non-blocking): {e}")


def main():
    run_ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    logger.info(f"=== CPO Daily starting — {run_ts} ===")

    init_db()

    if os.getenv("CPO_ENABLED", "true").lower() not in ("true", "1", "yes"):
        logger.info("CPO_ENABLED=false — nothing to do")
        return

    scraper   = MPOBScraper()
    generator = CPOSignalGenerator()

    # ── Step 1: Latest CPO price ──────────────────────────────────────────────
    cpo_price_info = None
    try:
        cpo_price_info = scraper.fetch_daily_cpo_price()
        logger.info(
            f"[CPO] Price: {cpo_price_info['price_myr_per_tonne']:.2f} MYR/t "
            f"({cpo_price_info['date']}) [{cpo_price_info['source']}]"
        )
        with db_session() as conn:
            conn.execute(
                "INSERT INTO daemon_logs (level, source, message) VALUES ('INFO', 'cpo_daily', ?)",
                (f"CPO price: {cpo_price_info['price_myr_per_tonne']:.2f} MYR/t "
                 f"date={cpo_price_info['date']} src={cpo_price_info['source']}",),
            )
    except Exception as e:
        logger.warning(f"[CPO] Price fetch failed: {e} — continuing with cached data")
        with db_session() as conn:
            conn.execute(
                "INSERT INTO daemon_logs (level, source, message) VALUES ('WARN', 'cpo_daily', ?)",
                (f"CPO price fetch failed: {e}",),
            )

    # ── Step 2: Refresh historical data ──────────────────────────────────────
    try:
        hist_df = scraper.get_historical_cpo(days=365)
        logger.info(f"[CPO] Historical: {len(hist_df)} rows in DB cache")
    except Exception as e:
        logger.warning(f"[CPO] Historical refresh failed: {e}")
        hist_df = None

    # ── Step 3: Lag analysis on all plantation tickers ───────────────────────
    logger.info(f"[CPO] Running lag analysis on {len(PLANTATION_TICKERS)} tickers...")
    signals = generator.daily_scan()

    # ── Step 4: Log results + auto-generate Gate 0 ideas ─────────────────────
    ideas_created = 0
    print("\n" + "=" * 72)
    print(f"  CPO Lag Signal Report — {run_ts}")
    print("=" * 72)
    if cpo_price_info:
        print(f"  CPO spot:  {cpo_price_info['price_myr_per_tonne']:.2f} MYR/t  ({cpo_price_info['source']})")
    if hist_df is not None:
        print(f"  DB cache:  {len(hist_df)} daily price rows")
    print("-" * 72)
    print(f"  {'Ticker':<10} {'Company':<28} {'BestLag':>7} {'Corr':>7} {'Sig':>5} {'Signal':>8}  {'Direction'}")
    print("-" * 72)

    for sig in signals:
        ticker    = sig.get("ticker", "?")
        company   = sig.get("company", ticker)
        lag       = sig.get("best_lag_days", 0)
        corr      = sig.get("best_lag_corr", 0.0)
        sig_flag  = "YES" if sig.get("is_significant") else "no"
        signal_v  = sig.get("signal_today", 0.0)
        direction = sig.get("predicted_direction", "neutral")
        n_obs     = sig.get("n_observations", 0)

        if sig.get("error"):
            print(f"  {ticker:<10} {company[:27]:<28} {'ERROR':<7} {sig['error'][:30]}")
            continue

        print(
            f"  {ticker:<10} {company[:27]:<28} {lag:>5}d  "
            f"{corr:>+6.3f}  {sig_flag:>5}  {signal_v:>+7.4f}  → {direction}"
            f"  (n={n_obs})"
        )

        # Show per-lag correlations
        lag_corrs = sig.get("lag_correlations", {})
        if lag_corrs:
            details = "  " + "  ".join(
                f"lag{k}:{v:+.3f}" for k, v in sorted(lag_corrs.items())
            )
            print(f"           {details}")

        # Auto-generate Gate 0 idea for strong signals
        if sig.get("is_significant") and abs(corr) > 0.35:
            idea_result = generator.generate_idea_if_significant(sig)
            if idea_result.get("created"):
                print(f"           ✓ Gate 0 idea [{idea_result['idea_id']}] created")
                ideas_created += 1
            else:
                print(f"           · Skipped: {idea_result['reason']}")

    # Summary row
    sig_count = sum(1 for s in signals if s.get("is_significant"))
    print("-" * 72)
    print(
        f"  Summary: {len(signals)} tickers | {sig_count} significant | "
        f"{ideas_created} Gate 0 ideas created"
    )
    print("=" * 72 + "\n")

    # ── Step 5: Seed KB document (once only) ──────────────────────────────────
    _seed_kb_document()

    # ── Final daemon_logs summary ─────────────────────────────────────────────
    with db_session() as conn:
        conn.execute(
            "INSERT INTO daemon_logs (level, source, message) VALUES ('INFO', 'cpo_daily', ?)",
            (f"CPO daily complete: {len(signals)} tickers scanned, "
             f"{sig_count} significant, {ideas_created} ideas created",),
        )

    logger.info(
        f"=== CPO Daily done: {len(signals)} tickers, "
        f"{sig_count} significant, {ideas_created} ideas ==="
    )


if __name__ == "__main__":
    main()
