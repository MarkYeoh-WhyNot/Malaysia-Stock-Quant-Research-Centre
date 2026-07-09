"""
EventWatcher — standalone daemon that monitors market events every 5 minutes
during trading hours (09:00–17:00 MYT) and 30 minutes after hours.

Sources:
  - Bursa announcements (KLSE Screener)
  - RSS feeds (Reuters, The Edge, Bernama, Yahoo Finance)
  - Commodity price moves (CPO, Brent crude, gold)
  - Economic calendar (Finnhub + hardcoded BNM dates)

Actions per event:
  - gate0_idea  → creates a Gate 0 alpha idea
  - alert       → sends Telegram alert
  - kb_only     → ingests to knowledge base
  - ignore      → discarded
"""
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

import requests

from data.database import db_session, init_db
from data.events.rss_client import RSSClient
from data.events.bursa_scraper import BursaScraper
from data.events.finnhub_client import FinnhubClient
from data.events.commodity_monitor import CommodityMonitor
from data.events.crypto_monitor import CryptoMonitor
from agents.event_classifier import EventClassifier, HISTORICAL_EDGES
from agents.researcher.strategy_researcher import StrategyResearcher
from knowledge.ingestion.kb_ingester import KBIngester

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("openclaw.event_watcher")

MYT = timezone(timedelta(hours=8))

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

EVENT_TYPE_EMOJIS = {
    "earnings_beat": "[GREEN]",
    "earnings_miss": "[RED]",
    "dividend_declared": "[PURPLE]",
    "bonus_issue": "[PURPLE]",
    "contract_win": "[BLUE]",
    "opr_hike": "[ORANGE]",
    "opr_cut": "[ORANGE]",
    "opr_hold": "[ORANGE]",
    "bnm_opr": "[ORANGE]",
    "cpo_move": "[YELLOW]",
    "crude_oil_move": "[YELLOW]",
    "gold_move": "[YELLOW]",
    "china_pmi_strong": "[GRAY]",
    "china_pmi_weak": "[GRAY]",
    "fed_decision": "[GRAY]",
    "macro_context": "[GRAY]",
    "analyst_upgrade": "[BLUE]",
    "analyst_downgrade": "[RED]",
}

EVENT_DOMAIN_MAP = {
    "earnings_beat": "event_driven",
    "earnings_miss": "event_driven",
    "earnings": "event_driven",
    "dividend_declared": "event_driven",
    "contract_win": "event_driven",
    "bonus_issue": "event_driven",
    "rights_issue": "event_driven",
    "mou": "event_driven",
    "acquisition": "event_driven",
    "disposal": "event_driven",
    "cpo_move": "event_driven",
    "crude_oil_move": "commodity",
    "gold_move": "commodity",
    "opr_hike": "macro",
    "opr_cut": "macro",
    "opr_hold": "macro",
    "bnm_opr": "macro",
    "china_pmi_strong": "macro",
    "china_pmi_weak": "macro",
    "fed_decision": "macro",
    "fed_hike": "macro",
    "fed_cut": "macro",
    "fed_hold": "macro",
    "ecb_decision": "macro",
    "macro_context": "macro",
    "macro_event": "macro",
    # Crypto event types (WS2)
    "funding_spike": "event_driven",
    "oi_surge": "behavioural",
    "basis_dislocation": "statistical_modelling",
    "btc_move": "commodity",       # crypto's "commodity" angle = BTC-dominance/beta
    "eth_move": "commodity",
    "listing": "event_driven",
    "unlock": "event_driven",
    "depeg": "event_driven",
    "btc_dominance_shift": "commodity",
    "dxy_move": "macro",
    "yield_move": "macro",
    "regulatory": "macro",
}

FACTOR_FORMULA_TEMPLATES = {
    "earnings_beat": (
        "Enter long when RSI(14) > 45 the day after an earnings beat announcement is confirmed. "
        "Hold for 20 trading days. Exit on RSI(14) > 70 or day 20, whichever comes first."
    ),
    "earnings_miss": (
        "Avoid holding on earnings miss. For pairs, go long the sector benchmark and reduce "
        "exposure to the missing stock for 15 trading days."
    ),
    "dividend_declared": (
        "Enter long 15 trading days before the ex-dividend date. "
        "Exit 1 trading day before ex-date to capture pre-dividend drift."
    ),
    "contract_win": (
        "Enter long on the trading day after the contract win announcement. "
        "Hold for 10 trading days or until RSI(14) > 65, whichever comes first."
    ),
    "bonus_issue": (
        "Enter long 10 trading days before the bonus issue ex-date. "
        "Exit on the ex-date or when RSI(14) > 70."
    ),
    "cpo_move": (
        "Enter long on CPO-correlated plantation stocks 3 trading days after CPO futures "
        "move exceeds +2%. Exit after 5 trading days or on mean reversion."
    ),
    "crude_oil_move": (
        "Enter long on O&G stocks 3 trading days after Brent crude moves >+3%. "
        "Exit after 5 trading days or on price target."
    ),
    "opr_hike": (
        "Enter long on banking stocks (Maybank, CIMB, Public Bank, RHB, HLBANK) on BNM OPR hike day. "
        "Hold for 10 trading days to capture NIM expansion re-rating."
    ),
    "opr_cut": (
        "Reduce exposure to banking stocks on BNM OPR cut announcement. "
        "Rotate into rate-sensitive growth sectors for 15 trading days."
    ),
    "analyst_upgrade": (
        "Enter long the day after analyst upgrade initiation/buy call. "
        "Hold for 10 trading days. Exit if price targets are hit or RSI > 70."
    ),
    "china_pmi_weak": (
        "On China PMI release below 50, reduce exposure to export-sensitive industrials. "
        "Monitor for 10 trading days before re-entry."
    ),
    # Crypto event types (WS2) — long-only spot phrasing (perp long/short lands
    # in Workstream 3); these are directional entry rules on the affected pair.
    "funding_spike": (
        "When perp funding rate exceeds +0.05% (crowded long), reduce/avoid new long entries "
        "for 2-3 days pending mean reversion. When funding is very negative, treat as a "
        "contrarian long-entry signal."
    ),
    "oi_surge": (
        "After a >15% open-interest jump alongside a price move, wait for the immediate "
        "session to close before entering — avoid chasing into a potential liquidation cascade. "
        "Re-evaluate 1-2 days later."
    ),
    "basis_dislocation": (
        "When perp basis exceeds 0.5% above index, treat as a short-term overextension signal "
        "on the spot side; avoid fresh long entries until basis normalises."
    ),
    "btc_move": (
        "On a BTC move >5%, expect alts to follow with a 0-2 day lag and amplified beta. "
        "Enter long the lagging correlated pair after BTC's move, exit after 3-5 trading days "
        "or on convergence."
    ),
    "eth_move": (
        "On an ETH move >6%, expect smart-contract/L2/DeFi tokens to follow with a 0-2 day lag. "
        "Enter long the lagging correlated pair, exit after 3-5 trading days or on convergence."
    ),
    "listing": (
        "Enter long on the trading day after a major-exchange listing announcement is confirmed. "
        "Hold for 3-5 trading days or exit on RSI(14) > 75, whichever comes first."
    ),
    "unlock": (
        "Reduce exposure ahead of a confirmed large token-unlock date. Re-evaluate for re-entry "
        "5 trading days after the unlock once sell pressure is absorbed."
    ),
    "dxy_move": (
        "On DXY strength >0.6%, treat as a mild headwind — avoid fresh long entries on majors "
        "for 1-2 trading days pending confirmation of risk-off continuation."
    ),
}


def _send_telegram(message: str):
    """Send a plain-text Telegram message."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.debug("Telegram not configured — skipping alert")
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
        }, timeout=10)
    except Exception as exc:
        logger.warning(f"EventWatcher: Telegram send failed: {exc}")


def _log_daemon(level: str, message: str):
    """Log to daemon_logs table and Python logger."""
    try:
        with db_session() as conn:
            conn.execute(
                "INSERT INTO daemon_logs (level, source, message) VALUES (?, ?, ?)",
                (level.upper(), "EventWatcher", message),
            )
    except Exception:
        pass
    getattr(logger, level.lower(), logger.info)(message)


def _save_raw_event(event: dict) -> bool:
    """
    Save raw event to market_events table.
    Returns True if inserted (new), False if duplicate.
    """
    try:
        with db_session() as conn:
            conn.execute("""
                INSERT OR IGNORE INTO market_events
                  (event_id, source, ticker, company, event_type, headline, body,
                   raw_url, published_at, affected_sectors, affected_tickers)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.get("event_id"),
                event.get("source", "unknown"),
                event.get("ticker"),
                event.get("company"),
                event.get("event_type", "general"),
                event.get("headline", ""),
                event.get("body"),
                event.get("raw_url"),
                event.get("published_at"),
                json.dumps(event.get("affected_sectors", [])),
                json.dumps(event.get("affected_tickers", [])),
            ))
        return True
    except Exception as exc:
        logger.debug(f"EventWatcher: save_raw_event failed: {exc}")
        return False


def _update_classified_event(event_id: str, classified: dict, action: str, idea_id=None):
    """Update market_events row with classifier output."""
    try:
        with db_session() as conn:
            conn.execute("""
                UPDATE market_events SET
                  event_type=?, confidence=?, sentiment=?, magnitude=?,
                  is_actionable=?, historical_edge=?, action_taken=?,
                  idea_id=?, classified_at=?,
                  affected_tickers=?
                WHERE event_id=?
            """, (
                classified.get("event_type"),
                classified.get("confidence"),
                classified.get("sentiment"),
                classified.get("magnitude"),
                1 if classified.get("is_actionable") else 0,
                classified.get("historical_edge"),
                action,
                idea_id,
                classified.get("classified_at"),
                json.dumps(classified.get("affected_tickers", [])),
                event_id,
            ))
    except Exception as exc:
        logger.debug(f"EventWatcher: update_classified failed: {exc}")


class EventWatcher:
    """Main event monitoring daemon."""

    def __init__(self):
        self.rss = RSSClient()
        self.bursa = BursaScraper()
        self.finnhub = FinnhubClient()
        self.commodities = CommodityMonitor()
        self.crypto = CryptoMonitor()
        self.classifier = EventClassifier()
        self.researcher = StrategyResearcher()
        self.kb = KBIngester()
        self._cycle_count = 0
        self._rss_cycle = 0  # fetch RSS every 3rd cycle

    def run_cycle(self):
        """Single scan cycle.

        Dual-market: sources 1, 2 and 4 (Bursa announcements, Malaysian news
        RSS, Bursa economic calendar) are Bursa-only and are skipped in crypto
        mode. Source 3 (the price-move monitor) is market-aware — its
        watchlist is profile-conditional (commodities for Bursa, BTC/ETH
        majors for crypto).
        """
        from config.settings import MARKET_MODE
        _is_bursa = MARKET_MODE == "bursa"
        raw_events = []

        # 1. Bursa announcements — every cycle (Bursa only)
        if _is_bursa:
            try:
                ann = self.bursa.fetch_announcements(hours_back=1)
                raw_events.extend(ann)
                logger.debug(f"Bursa announcements: {len(ann)}")
            except Exception as exc:
                _log_daemon("WARN", f"Bursa scraper error: {exc}")

        # 2. RSS feeds — every 3rd cycle (15 min cadence; Bursa news sources)
        if _is_bursa:
            self._rss_cycle += 1
            if self._rss_cycle >= 3:
                try:
                    rss = self.rss.fetch_all_feeds()
                    raw_events.extend(rss)
                    logger.debug(f"RSS: {len(rss)} entries")
                except Exception as exc:
                    _log_daemon("WARN", f"RSS error: {exc}")
                self._rss_cycle = 0

        # 3. Market price-move monitor — every cycle (watchlist per profile)
        try:
            comm = self.commodities.check_moves()
            raw_events.extend(comm)
            logger.debug(f"Price-move events: {len(comm)}")
        except Exception as exc:
            _log_daemon("WARN", f"Price-move monitor error: {exc}")

        # 4. Economic calendar check (Bursa only)
        if _is_bursa:
            try:
                self.check_economic_calendar()
            except Exception as exc:
                _log_daemon("WARN", f"Calendar check error: {exc}")

        # 5. Perp funding/OI/basis monitor — every cycle (crypto only)
        if not _is_bursa:
            try:
                crypto_events = self.crypto.check_moves()
                raw_events.extend(crypto_events)
                logger.debug(f"Crypto perp events: {len(crypto_events)}")
            except Exception as exc:
                _log_daemon("WARN", f"Crypto monitor error: {exc}")

        # Process all new events
        ideas_created = 0
        alerts_sent = 0
        kb_ingested = 0
        ignored = 0

        for event in raw_events:
            try:
                # Save raw event first
                _save_raw_event(event)

                # Classify
                classified = self.classifier.classify(event)
                action = self.classifier.determine_action(classified)

                idea_id = None

                if action == "gate0_idea":
                    idea_id = self.create_gate0_idea(event, classified)
                    ideas_created += 1
                elif action == "alert":
                    self.send_telegram_alert(event, classified)
                    alerts_sent += 1
                elif action == "kb_only":
                    self.ingest_to_kb(event, classified)
                    kb_ingested += 1
                else:
                    ignored += 1

                _update_classified_event(
                    event.get("event_id", ""),
                    classified,
                    action,
                    idea_id,
                )

            except Exception as exc:
                _log_daemon("WARN", f"Event processing error: {exc}")

        summary = (
            f"Cycle {self._cycle_count}: {len(raw_events)} events — "
            f"{ideas_created} ideas, {alerts_sent} alerts, {kb_ingested} KB, {ignored} ignored"
        )
        _log_daemon("INFO", summary)

    def create_gate0_idea(self, event: dict, classified: dict) -> int | None:
        """Create a Gate 0 alpha idea from a classified event."""
        try:
            event_type = classified.get("event_type", "general")
            ticker = classified.get("ticker") or event.get("ticker")
            affected = classified.get("affected_tickers", [])

            # For commodity events with no single ticker, use first affected
            if not ticker and affected:
                ticker = affected[0]
            if not ticker:
                # Can't create idea without a valid ticker
                _log_daemon("WARN", f"create_gate0_idea: no ticker for event {event.get('event_id')}")
                return None

            # Use primary ticker — validated against the active market's ticker
            # shape (.KL for Bursa, BASE/USDT for crypto), not hardcoded .KL.
            # (Fix: this previously rejected every crypto event outright, so
            # BTC/ETH price-move and funding/OI events could never become ideas.)
            from config.settings import TICKER_REGEX, DATA_BACKEND

            primary = ticker.split(",")[0].strip()
            if not TICKER_REGEX.fullmatch(primary):
                return None

            title = classified.get("suggested_idea_title") or (
                f"{event_type.replace('_', ' ').title()} — {event.get('company') or primary}"
            )
            hypothesis = classified.get("suggested_hypothesis") or (
                f"Exploit {event_type.replace('_', ' ')} signal in {primary}. "
                f"Historical edge: {classified.get('historical_edge', 'unknown')}."
            )
            factor_formula = FACTOR_FORMULA_TEMPLATES.get(
                event_type,
                f"Enter {primary} following {event_type.replace('_', ' ')} signal. "
                f"Hold 10 trading days. Exit on RSI(14) > 65 or day 10.",
            )

            idea = {
                "title": title[:200],
                "hypothesis": hypothesis,
                "ticker": primary,
                "timeframe": "1d",
                "factor_formula": factor_formula,
                "data_sources": [DATA_BACKEND, event.get("source", "event")],
                "novelty_score": 0.70,
                "logic_score": 0.75,
                "screen_source": f"event_{event_type}",
            }

            idea_id = self.researcher.save_idea(idea)
            _log_daemon("INFO", f"Gate0 idea created: [{idea_id}] {title[:60]}")
            return idea_id

        except Exception as exc:
            _log_daemon("WARN", f"create_gate0_idea failed: {exc}")
            return None

    def send_telegram_alert(self, event: dict, classified: dict):
        """Send Telegram alert for medium-confidence actionable events."""
        event_type = classified.get("event_type", "general")
        emoji = EVENT_TYPE_EMOJIS.get(event_type, "[INFO]")
        ticker = classified.get("ticker") or event.get("ticker") or "SECTOR"
        company = classified.get("company") or event.get("company") or ""
        confidence = float(classified.get("confidence", 0.0))
        edge = classified.get("historical_edge") or "Unknown"
        sentiment = classified.get("sentiment", "neutral").upper()
        reasoning = classified.get("reasoning") or ""
        headline = event.get("headline", "")

        lines = [
            f"EVENT ALERT — {emoji} {event_type.upper().replace('_', ' ')}",
            "",
            f"{ticker}  {company}",
            f"{headline}",
            "",
            f"Sentiment: {sentiment}  |  Confidence: {confidence*100:.0f}%",
            f"Historical edge: {edge}",
        ]
        if reasoning:
            lines += ["", f"Reasoning: {reasoning}"]
        lines += [
            "",
            "Action: Human review recommended.",
            "Use /events to see full event feed.",
        ]

        _send_telegram("\n".join(lines))

    def ingest_to_kb(self, event: dict, classified: dict):
        """Ingest event into knowledge base as market context."""
        headline = event.get("headline", "")
        body = event.get("body") or ""
        event_type = classified.get("event_type", "macro_context")
        affected = classified.get("affected_tickers", [])
        sentiment = classified.get("sentiment", "neutral")
        reasoning = classified.get("reasoning") or ""

        content = (
            f"{body}\n\n"
            f"Event type: {event_type}\n"
            f"Affected tickers: {', '.join(affected) if affected else 'none'}\n"
            f"Sentiment: {sentiment}\n"
            f"Reasoning: {reasoning}"
        ).strip()

        if len(content) < 50:
            return  # too short to be useful

        domain = EVENT_DOMAIN_MAP.get(event_type, "macro")

        try:
            self.kb.ingest_text(
                title=headline[:200],
                content=content,
                domain=domain,
            )
        except Exception as exc:
            logger.debug(f"EventWatcher: KB ingest failed: {exc}")

    def check_economic_calendar(self):
        """Check for high-importance macro events scheduled today and alert if unprocessed."""
        today = datetime.utcnow().isoformat()[:10]
        try:
            with db_session() as conn:
                rows = conn.execute("""
                    SELECT * FROM economic_calendar
                    WHERE scheduled_date = ? AND importance = 'high' AND processed = 0
                    ORDER BY scheduled_time
                """, (today,)).fetchall()
        except Exception:
            return

        for row in rows:
            row = dict(row)
            msg_lines = [
                f"MACRO CALENDAR — {row['event_name']} TODAY",
                "",
                f"Country: {row.get('country', '?')}",
                f"Time: {row.get('scheduled_time', 'TBC')} MYT",
            ]
            if row.get("forecast_value"):
                msg_lines.append(f"Forecast: {row['forecast_value']}")
            if row.get("previous_value"):
                msg_lines.append(f"Previous: {row['previous_value']}")
            msg_lines += ["", "Monitor for trading signal after release."]

            _send_telegram("\n".join(msg_lines))

            # Mark as processed so we don't re-alert
            try:
                with db_session() as conn:
                    conn.execute(
                        "UPDATE economic_calendar SET processed=1 WHERE id=?", (row["id"],)
                    )
            except Exception:
                pass

    def run(self):
        """Main loop — runs indefinitely, sleeping between cycles."""
        init_db()
        _log_daemon("INFO", "EventWatcher daemon starting")

        # Refresh Finnhub economic calendar on startup
        try:
            count = self.finnhub.fetch_economic_calendar(days_ahead=30)
            _log_daemon("INFO", f"Economic calendar refreshed: {count} events")
        except Exception as exc:
            _log_daemon("WARN", f"Calendar refresh failed: {exc}")

        while True:
            try:
                self._cycle_count += 1
                self.run_cycle()
            except Exception as exc:
                _log_daemon("ERROR", f"Cycle {self._cycle_count} error: {exc}")

            # Sleep based on market hours (MYT = UTC+8)
            now = datetime.now(tz=MYT)
            if 9 <= now.hour < 17 and now.weekday() < 5:
                time.sleep(300)   # 5 min during market hours
            else:
                time.sleep(1800)  # 30 min after hours / weekends


if __name__ == "__main__":
    watcher = EventWatcher()
    watcher.run()
