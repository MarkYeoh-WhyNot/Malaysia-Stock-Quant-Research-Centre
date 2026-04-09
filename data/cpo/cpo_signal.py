"""
CPOSignalGenerator — daily CPO-plantation lag scan and Gate 0 idea creation.

Runs MPOBScraper.compute_lag_signal() on every plantation ticker, logs results,
and auto-creates Gate 0 ideas in alpha_ideas for signals with |corr| > 0.35.
"""
import json
import logging
import re
from datetime import datetime
from typing import List

from data.cpo.mpob_scraper import MPOBScraper, PLANTATION_TICKERS, PLANTATION_NAMES
from data.database import db_session

logger = logging.getLogger(__name__)


class CPOSignalGenerator:
    """Orchestrates daily CPO lag scan and pipeline idea injection."""

    def __init__(self):
        self.scraper = MPOBScraper()

    # ── Daily scan ────────────────────────────────────────────────────────────

    def daily_scan(self) -> List[dict]:
        """Run compute_lag_signal() on all PLANTATION_TICKERS.

        Returns a list of signal dicts (one per ticker, errors included).
        """
        results = []
        for ticker in PLANTATION_TICKERS:
            try:
                signal = self.scraper.compute_lag_signal(ticker)
                signal["company"] = PLANTATION_NAMES.get(ticker, ticker)
                results.append(signal)
            except Exception as e:
                logger.warning(f"CPO scan failed for {ticker}: {e}")
                results.append({
                    "ticker":              ticker,
                    "company":             PLANTATION_NAMES.get(ticker, ticker),
                    "error":               str(e),
                    "is_significant":      False,
                    "predicted_direction": "neutral",
                    "best_lag_days":       0,
                    "best_lag_corr":       0.0,
                    "signal_today":        0.0,
                })
        return results

    # ── Gate 0 idea creation ──────────────────────────────────────────────────

    def generate_idea_if_significant(self, signal: dict) -> dict:
        """Create a Gate 0 alpha idea if the CPO lag signal is strong enough.

        Threshold: is_significant=True AND |best_lag_corr| > 0.35

        Skips if:
          - Signal has an error
          - A CPO lag idea for this ticker on today's date already exists

        Returns:
            {"created": bool, "idea_id": int|None, "reason": str}
        """
        ticker = signal.get("ticker", "")
        corr   = float(signal.get("best_lag_corr", 0.0))
        lag    = int(signal.get("best_lag_days", 1))

        if signal.get("error"):
            return {
                "created": False, "idea_id": None,
                "reason": f"Signal error: {signal['error']}",
            }

        if not signal.get("is_significant") or abs(corr) < 0.35:
            return {
                "created": False, "idea_id": None,
                "reason": f"Correlation too weak: |{corr:.3f}| < 0.35",
            }

        company   = signal.get("company", ticker)
        direction = "positive" if corr > 0 else "negative"
        today_str = datetime.utcnow().strftime("%Y-%m-%d")
        slug      = f"cpo-lag-{re.sub(r'[^a-z0-9]','', ticker.lower())}-{lag}d-{today_str}"

        # Idempotency: skip if already created today
        with db_session() as conn:
            existing = conn.execute(
                "SELECT id FROM alpha_ideas WHERE slug=?", (slug,)
            ).fetchone()
        if existing:
            return {
                "created": False, "idea_id": existing["id"],
                "reason": "CPO lag idea already exists for today",
            }

        signal_today = signal.get("signal_today", 0.0)
        n_obs        = signal.get("n_observations", 0)

        title = f"CPO Lag Signal — {company} {lag}d lag"

        hypothesis = (
            f"Crude Palm Oil (CPO) spot price moves predict {company} ({ticker}) "
            f"stock returns with a {lag}-day lag "
            f"(Spearman r={corr:+.2f}, {direction} correlation, n={n_obs} obs). "
            f"The lag exists because institutional fund managers rebalance plantation "
            f"holdings 1-5 days after CPO price signals, while retail investors process "
            f"the information even later. "
            f"Pure-play planters show the strongest effect; conglomerates are dampened "
            f"by revenue diversification. "
            f"Today's CPO 1d return: {signal_today:+.3%}."
        )

        formula = (
            f"Signal source: MPOB BEPI CPO spot price (daily). "
            f"Entry trigger: CPO_1d_return = (CPO_today - CPO_yesterday) / CPO_yesterday > +0.010 (>+1.0%). "
            f"Action: Buy {ticker} at market open on day T+{lag} after the CPO surge. "
            f"Position size: 5% of portfolio NAV (equal weight). "
            f"Exit rule: Close position at T+{lag+1} close OR if CPO_1d_return < -0.015 (-1.5%). "
            f"Stop-loss: -6% from entry price. "
            f"Data required: MPOB daily CPO price, Yahoo Finance daily OHLCV {ticker}."
        )

        with db_session() as conn:
            conn.execute("""
                INSERT OR IGNORE INTO alpha_ideas
                  (slug, title, hypothesis, ticker, timeframe, factor_formula,
                   data_sources, stage, status, novelty_score, logic_score)
                VALUES (?, ?, ?, ?, '1d', ?, ?, 'gate0', 'pending', ?, ?)
            """, (
                slug, title, hypothesis, ticker, formula,
                json.dumps([
                    "MPOB BEPI CPO spot price (daily)",
                    f"Yahoo Finance daily OHLCV {ticker}",
                ]),
                0.65,   # novelty: lag signals are known; ticker-specific tuning adds value
                0.72,   # logic: CPO is primary revenue driver — sound mechanism
            ))
            row = conn.execute(
                "SELECT id FROM alpha_ideas WHERE slug=?", (slug,)
            ).fetchone()
            idea_id = row["id"]

            conn.execute("""
                INSERT INTO pipeline_events
                  (idea_id, stage, event_type, agent, notes)
                VALUES (?, 'gate0', 'created', 'CPOSignalGenerator', ?)
            """, (
                idea_id,
                f"Auto-created CPO lag idea: "
                f"lag={lag}d corr={corr:+.3f} "
                f"signal_today={signal_today:+.4f} → {signal.get('predicted_direction')}",
            ))
            conn.execute(
                "INSERT INTO daemon_logs (level, source, message) VALUES ('INFO', 'CPOSignalGenerator', ?)",
                (f"Created Gate 0 idea [{idea_id}] '{title}' "
                 f"(lag={lag}d corr={corr:+.3f})",),
            )

        logger.info(
            f"CPOSignalGenerator: created idea [{idea_id}] '{title}'"
        )
        return {"created": True, "idea_id": idea_id, "reason": "Signal is significant"}

    # ── Batch run ─────────────────────────────────────────────────────────────

    def run_full_scan(self) -> dict:
        """Run daily_scan + generate_idea_if_significant for all tickers.

        Returns a summary dict with counts.
        """
        signals       = self.daily_scan()
        ideas_created = 0
        ideas_skipped = 0

        for sig in signals:
            if not sig.get("is_significant"):
                continue
            result = self.generate_idea_if_significant(sig)
            if result["created"]:
                ideas_created += 1
            else:
                ideas_skipped += 1

        return {
            "tickers_scanned": len(signals),
            "significant":     sum(1 for s in signals if s.get("is_significant")),
            "ideas_created":   ideas_created,
            "ideas_skipped":   ideas_skipped,
            "signals":         signals,
        }
