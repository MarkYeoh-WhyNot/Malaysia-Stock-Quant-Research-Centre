"""
EPFSignalGenerator — daily EPF accumulation scan and pipeline idea injection.

Runs EPFScraper on all 30 KLCI stocks, identifies systematic accumulation
patterns, and auto-creates Gate 0 alpha ideas for strong signals.
"""
import json
import logging
import re
from datetime import datetime
from typing import List

from config.settings import KLCI_STOCKS
from data.epf.epf_scraper import EPFScraper, _TICKER_TO_NAME
from data.database import db_session

logger = logging.getLogger(__name__)

# KB seed — ingested once on first run
_KB_SEED_TITLE   = "EPF as a Systematic Alpha Signal on Bursa Malaysia"
_KB_SEED_CONTENT = """
EPF as a Systematic Alpha Signal on Bursa Malaysia

The Employees Provident Fund (EPF) is Malaysia's mandatory retirement fund managing
~MYR 1 trillion in assets. As a price-insensitive, mandate-driven institutional buyer,
EPF accumulation patterns create predictable price support.

KEY CHARACTERISTICS:
EPF receives mandatory 23% payroll contributions monthly regardless of market conditions,
creating consistent buying pressure in their holdings. Substantial shareholder disclosures
(triggered at 5%+ ownership or ±1% change above 5%) are public on Bursa Malaysia.
Stocks where EPF increases ownership for 3+ consecutive quarters show statistically
significant outperformance over the following 6-12 months.

THE MECHANISM:
EPF accumulation signals legitimacy to other institutional investors, reducing information
asymmetry and triggering sympathetic buying from KWAP, PNB, and foreign funds. This
creates a reflexive loop: EPF buys → price stable/rising → other institutions follow
→ further price support.

KNOWN EPF HEAVY WEIGHTS (typical holdings):
- CIMB Group (1023.KL): ~22% EPF ownership
- Tenaga Nasional (5347.KL): ~24% EPF ownership
- Telekom Malaysia (4863.KL): ~22% EPF ownership
- MISC Berhad (3816.KL): ~15% EPF ownership
- Maybank (1155.KL): ~11% EPF ownership

SIGNAL CONSTRUCTION:
1. Track EPF % ownership per quarter from Bursa substantial shareholder disclosures
2. If EPF ownership increases for 3 of last 4 quarters → STRONG accumulation signal
3. If increases for 2 of last 4 quarters → MODERATE signal
4. Entry: buy at open after disclosure date; hold 30-90 days
5. Stop-loss: -8% from entry, exit if EPF stake drops by >0.5%

CAVEATS:
- EPF disclosures lag actual trades by 3-5 business days (mandatory filing window)
- Large market cap stocks (Maybank, CIMB) have lower price impact per % change
- EPF occasionally sells for liquidity — distinguish from strategic distribution
- Quarterly window is a proxy; actual accumulation may be faster/slower

BACKTESTING NOTES:
- Use Bursa Malaysia CDSPI disclosure database for historical data
- Control for market beta (KLCI returns) to isolate EPF-specific alpha
- Account for price drift between EPF purchase date and disclosure date
- Minimum 15 events required for statistical significance

REFERENCES:
- Bursa Malaysia Substantial Shareholder filings (Company Announcements)
- Securities Commission Malaysia Act 1993 (disclosure thresholds)
- EPF Annual Report (asset allocation breakdown)
""".strip()


class EPFSignalGenerator:
    """Orchestrates EPF accumulation signal scanning and Gate 0 idea injection."""

    def __init__(self):
        self.scraper = EPFScraper()

    # ── KB seed ───────────────────────────────────────────────────────────────

    def _seed_kb_document(self):
        """Ingest EPF KB seed document (once only — checks by title)."""
        try:
            with db_session() as conn:
                exists = conn.execute(
                    "SELECT id FROM kb_documents WHERE title=?",
                    (_KB_SEED_TITLE,),
                ).fetchone()
            if exists:
                logger.info(f"EPF KB seed already exists (doc_id={exists['id']}) — skipping")
                return

            from knowledge.ingestion.kb_ingester import KBIngester
            kb     = KBIngester()
            result = kb.ingest_text(
                content=_KB_SEED_CONTENT,
                title=_KB_SEED_TITLE,
                domain="institutional",
                source_url="epf_signal:seed",
            )
            logger.info(
                f"EPF KB seed ingested: doc_id={result.get('doc_id')} "
                f"relevance={result.get('relevance_score', 0):.2f} "
                f"({result.get('relevance_category', '?')})"
            )
        except Exception as e:
            logger.warning(f"EPF KB seed ingestion failed (non-blocking): {e}")

    # ── Gate 0 idea creation ──────────────────────────────────────────────────

    def _create_idea(self, signal: dict) -> dict:
        """Create a Gate 0 alpha idea for a strong EPF accumulation signal."""
        ticker  = signal["ticker"]
        company = signal.get("company", _TICKER_TO_NAME.get(ticker, ticker))
        qi      = signal["quarters_increasing"]
        total   = signal["total_change_4q"]
        pct     = signal["current_pct"]

        today_str = datetime.utcnow().strftime("%Y-%m-%d")
        slug = (
            f"epf-accum-{re.sub(r'[^a-z0-9]', '', ticker.lower())}-{today_str}"
        )

        with db_session() as conn:
            existing = conn.execute(
                "SELECT id FROM alpha_ideas WHERE slug=?", (slug,)
            ).fetchone()
        if existing:
            return {
                "created":  False,
                "idea_id":  existing["id"],
                "reason":   "EPF idea already exists for today",
            }

        title = f"EPF Accumulation Signal — {company} ({ticker})"
        hypothesis = (
            f"EPF has increased its ownership in {company} ({ticker}) "
            f"for {qi} of the last 4 reporting periods "
            f"(cumulative change: {total:+.2f}%, current holding: {pct:.2f}%). "
            f"As a mandate-driven, price-insensitive institutional buyer with ~MYR 1 trillion "
            f"in AUM, sustained EPF accumulation creates predictable upward price pressure "
            f"and signals legitimacy to KWAP, PNB, and foreign funds — triggering sympathetic "
            f"buying that amplifies the effect. EPF receives mandatory 23% payroll "
            f"contributions monthly regardless of market conditions, creating consistent "
            f"structural buying demand. "
            f"Strategy: buy {ticker} following EPF disclosure and hold 30-90 days."
        )
        formula = (
            f"Signal trigger: EPF ownership % increases in {qi}/4 recent reporting periods. "
            f"Data source: Bursa Malaysia substantial shareholder disclosures (EPF/KWSP). "
            f"Entry: buy {ticker} at open on the business day following disclosure. "
            f"Position size: 5% of portfolio NAV (equal-weighted). "
            f"Exit: 90 calendar days from entry OR EPF stake drops >0.5% in next disclosure. "
            f"Stop-loss: -8% from entry price. "
            f"Cross-sectional extension: apply to all KLCI stocks simultaneously for "
            f"factor-level backtesting (long top-quintile EPF accumulators)."
        )

        with db_session() as conn:
            conn.execute("""
                INSERT OR IGNORE INTO alpha_ideas
                  (slug, title, hypothesis, ticker, timeframe, factor_formula,
                   data_sources, stage, status, novelty_score, logic_score)
                VALUES (?, ?, ?, ?, '90d', ?, ?, 'gate0', 'pending', ?, ?)
            """, (
                slug, title, hypothesis, ticker, formula,
                json.dumps([
                    "Bursa Malaysia EPF substantial shareholder disclosures",
                    f"Yahoo Finance {ticker} daily OHLCV",
                    "i3investor EPF tracker",
                ]),
                0.62,   # novelty: documented effect, but ticker-specific signal adds value
                0.78,   # logic: strong mechanism — price-insensitive mandate buyer
            ))
            row = conn.execute(
                "SELECT id FROM alpha_ideas WHERE slug=?", (slug,)
            ).fetchone()
            idea_id = row["id"]

            conn.execute("""
                INSERT INTO pipeline_events
                  (idea_id, stage, event_type, agent, notes)
                VALUES (?, 'gate0', 'created', 'EPFSignalGenerator', ?)
            """, (
                idea_id,
                f"Auto-created EPF accumulation idea: {qi}/4 quarters increasing, "
                f"total={total:+.2f}% current={pct:.2f}%",
            ))
            conn.execute(
                "INSERT INTO daemon_logs (level, source, message) VALUES "
                "('INFO', 'EPFSignalGenerator', ?)",
                (f"Created Gate 0 idea [{idea_id}] '{title}'",),
            )

        logger.info(f"EPFSignalGenerator: created idea [{idea_id}] '{title}'")
        return {"created": True, "idea_id": idea_id, "reason": "Strong EPF accumulation signal"}

    # ── Daily scan ────────────────────────────────────────────────────────────

    def daily_scan(self) -> List[dict]:
        """Fetch disclosures + compute signals for all KLCI stocks.

        Auto-creates Gate 0 ideas for stocks with signal_strength == 'strong'.
        Returns list of signal dicts.
        """
        # Seed KB on first run
        self._seed_kb_document()

        # Refresh from web sources
        disclosures = self.scraper.fetch_epf_disclosures()
        logger.info(f"EPF daily scan: {len(disclosures)} disclosures fetched")

        signals: List[dict] = []
        ideas_created = 0

        for stock in KLCI_STOCKS:
            ticker  = stock["symbol"]
            company = stock["name"]

            signal = self.scraper.compute_accumulation_signal(ticker)
            signal["company"] = company

            signals.append(signal)

            if signal.get("signal_strength") == "strong" and "error" not in signal:
                result = self._create_idea(signal)
                if result.get("created"):
                    ideas_created += 1

        logger.info(
            f"EPF daily scan complete: {len(signals)} tickers, "
            f"{sum(1 for s in signals if 'error' not in s)} with data, "
            f"{ideas_created} ideas created"
        )
        return signals

    # ── Weekly report (for Telegram /epf) ────────────────────────────────────

    def weekly_report(self) -> dict:
        """Generate an EPF movement summary across all KLCI stocks.

        Fetches fresh disclosures, then categorises every KLCI ticker into
        accumulating / distributing / stable / no_data.

        Returns:
            {generated_at, disclosures_fetched,
             accumulating, distributing, stable, no_data}
        """
        # Refresh disclosures (non-fatal if sources down)
        try:
            disclosures = self.scraper.fetch_epf_disclosures()
        except Exception as e:
            logger.warning(f"EPF weekly_report: fetch failed ({e}) — using cached data")
            disclosures = []

        accumulating: List[dict] = []
        distributing: List[dict] = []
        stable:       List[dict] = []
        no_data:      List[dict] = []

        for stock in KLCI_STOCKS:
            ticker  = stock["symbol"]
            company = stock["name"]
            signal  = self.scraper.compute_accumulation_signal(ticker)

            if "error" in signal:
                no_data.append({"ticker": ticker, "company": company})
                continue

            entry = {
                "ticker":              ticker,
                "company":             company,
                "current_pct":         signal["current_pct"],
                "total_change_4q":     signal["total_change_4q"],
                "signal_strength":     signal["signal_strength"],
                "quarters_increasing": signal["quarters_increasing"],
            }

            trend = signal.get("trend", "stable")
            if trend == "accumulating":
                accumulating.append(entry)
            elif trend == "distributing":
                distributing.append(entry)
            else:
                stable.append(entry)

        # Sort for readability
        accumulating.sort(
            key=lambda x: (x["signal_strength"] == "strong", x["total_change_4q"]),
            reverse=True,
        )
        distributing.sort(key=lambda x: x["total_change_4q"])

        return {
            "generated_at":       datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            "disclosures_fetched": len(disclosures),
            "accumulating":       accumulating,
            "distributing":       distributing,
            "stable":             stable,
            "no_data":            no_data,
        }
