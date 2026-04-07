"""
Morning Briefing — generates and sends a daily 8am KL time briefing via Telegram.

Includes pipeline status, overnight i3investor research, upcoming dividends,
today's research angle, and yesterday's AI spend.

Usage (direct):
    /opt/openclaw/venv/bin/python scripts/morning_briefing.py

Supervisor service: openclaw-briefing
Daemon integration: _process_morning_briefing() in research_daemon.py
"""
import logging
import os
import sys
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

import requests

from agents.risk_monitor.risk_monitor import RiskMonitor
from data.i3investor.scraper import I3investorScraper
from data.klse.fundamental_scanner import FundamentalScanner
from data.database import db_session, init_db
from knowledge.ingestion.diversity_engine import DiversityEngine
from knowledge.ingestion.kb_ingester import KBIngester

logger = logging.getLogger("openclaw.morning_briefing")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

# Major brokerages whose research we auto-ingest
MAJOR_BROKERAGES = {
    "RHB", "Kenanga", "Maybank", "CIMB", "PublicBank",
    "AmInvest", "PhillipCapital", "Hong Leong", "UOB", "Affin",
}

# KL is UTC+8
_KL_TZ = timezone(timedelta(hours=8))


class MorningBriefing:
    """
    Generates and distributes the daily morning briefing.
    """

    def __init__(self):
        self.risk_monitor = RiskMonitor()
        self.scraper      = I3investorScraper()
        self.scanner      = FundamentalScanner()
        self.kb           = KBIngester()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def generate_briefing(self) -> dict:
        """
        Compile and send the morning briefing.

        Returns a dict with all components so callers can inspect without
        re-running.
        """
        logger.info("Morning briefing: starting generation...")
        now_kl = datetime.now(_KL_TZ)

        # a) Pipeline status
        health = self.risk_monitor.pipeline_health_report()

        # b) Overnight research articles (top 5)
        articles = []
        try:
            all_articles = self.scraper.get_research_articles(max_articles=20)
            articles = all_articles[:5]
        except Exception as e:
            logger.warning(f"Morning briefing: research articles fetch failed: {e}")

        # c) Dividend announcements in next 7 days
        dividends = []
        try:
            dividends = self.scanner.scan_dividend_calendar(days_ahead=7)
        except Exception as e:
            logger.warning(f"Morning briefing: dividend calendar failed: {e}")

        # d) Today's research angle from DiversityEngine
        research_angle = ""
        try:
            de      = DiversityEngine()
            balance = de.check_balance()
            research_angle = balance.get("least_covered", "")
        except Exception as e:
            logger.warning(f"Morning briefing: diversity engine failed: {e}")

        # e) Yesterday's AI spend
        yesterday = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
        with db_session() as conn:
            yesterday_spend = conn.execute(
                "SELECT COALESCE(SUM(cost_usd),0) as t FROM ai_usage WHERE created_at LIKE ?",
                (f"{yesterday}%",),
            ).fetchone()["t"]
            today_spend = conn.execute(
                "SELECT COALESCE(SUM(cost_usd),0) as t FROM ai_usage WHERE created_at >= date('now')",
            ).fetchone()["t"]

        # f) Cross-reference articles with pipeline ideas
        pipeline_ideas = self._get_pipeline_ideas()
        flagged_articles = [
            a for a in articles if self.check_article_relevance(a, pipeline_ideas)
        ]

        # g) Format message
        message = self._format_message(
            now_kl          = now_kl,
            health          = health,
            articles        = articles,
            flagged_articles= flagged_articles,
            dividends       = dividends,
            research_angle  = research_angle,
            yesterday_spend = float(yesterday_spend),
            today_spend     = float(today_spend),
        )

        # h) Send to Telegram
        sent = self._send_telegram(message)

        # i) Auto-ingest major brokerage research in background
        try:
            self.auto_ingest_research(articles)
        except Exception as e:
            logger.warning(f"Morning briefing: auto-ingest failed: {e}")

        result = {
            "sent":            sent,
            "articles":        len(articles),
            "dividends":       len(dividends),
            "research_angle":  research_angle,
            "yesterday_spend": float(yesterday_spend),
            "health":          health["health"],
            "message_len":     len(message),
            "generated_at":    now_kl.isoformat(),
        }
        logger.info(f"Morning briefing: done — sent={sent} articles={len(articles)} dividends={len(dividends)}")
        return result

    # ------------------------------------------------------------------
    # Article relevance cross-check
    # ------------------------------------------------------------------

    def check_article_relevance(self, article: dict, pipeline_ideas: list) -> bool:
        """
        Return True if the article mentions a stock already in the pipeline.
        """
        article_tickers = set(article.get("tickers", []))
        if not article_tickers:
            return False
        for idea in pipeline_ideas:
            idea_ticker = idea.get("pair", "")
            if idea_ticker and idea_ticker in article_tickers:
                return True
        return False

    # ------------------------------------------------------------------
    # Auto-ingest brokerage research
    # ------------------------------------------------------------------

    def auto_ingest_research(self, articles: list) -> int:
        """
        For articles from major brokerages, fetch full content and ingest
        into the KB with domain='fundamental' and brokerage tag.

        Returns number of articles successfully ingested.
        """
        ingested = 0
        for article in articles:
            brokerage = article.get("brokerage", "")
            if not brokerage or brokerage not in MAJOR_BROKERAGES:
                continue
            url = article.get("url", "")
            if not url:
                continue
            try:
                # Fetch full content using the scraper
                content = self.scraper.get_article_content(url)
                if not content or len(content) < 100:
                    continue

                # Build tags
                tickers = article.get("tickers", [])
                tags    = [brokerage] + tickers
                title   = article.get("title", url[:60])

                # Ingest synchronously (kb_ingester has async ingest_url, use ingest_text)
                import asyncio
                result = asyncio.run(
                    self.kb.ingest_text(
                        content,
                        title=title,
                        domain="fundamental",
                        tags=tags,
                    )
                ) if hasattr(self.kb, "ingest_text") else None

                if result:
                    ingested += 1
                    logger.info(f"Auto-ingested: {title[:50]} [{brokerage}]")

            except Exception as e:
                logger.debug(f"Auto-ingest failed for {article.get('title', url)}: {e}")

        logger.info(f"Morning briefing: auto-ingested {ingested} brokerage articles")
        return ingested

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_pipeline_ideas(self) -> list:
        """Fetch active pipeline ideas for relevance cross-check."""
        try:
            with db_session() as conn:
                return conn.execute(
                    "SELECT id, title, pair, stage FROM alpha_ideas WHERE status='active' ORDER BY id DESC LIMIT 50"
                ).fetchall()
        except Exception:
            return []

    def _format_message(
        self,
        now_kl,
        health,
        articles,
        flagged_articles,
        dividends,
        research_angle,
        yesterday_spend,
        today_spend,
    ) -> str:
        lines = [
            f"🌅 *OpenClaw Morning Briefing* — {now_kl.strftime('%A, %d %b %Y')}",
            f"",
        ]

        # Pipeline health
        health_icon = {"healthy": "✅", "degraded": "⚠️", "critical": "🚨"}.get(
            health["health"], "ℹ️"
        )
        lines += [
            f"*Pipeline* {health_icon}",
            f"  Status: `{health['health'].upper()}`  |  Ideas: `{health['total_ideas']}`",
            f"  Errors (1h): `{health['errors_1h']}`",
            f"  Spend yesterday: `${yesterday_spend:.3f}`  |  Today so far: `${today_spend:.3f}`",
            f"",
        ]

        # Research articles
        if articles:
            lines.append(f"*Overnight Research ({len(articles)} articles)*")
            for a in articles[:5]:
                flag = " ⭐" if a in flagged_articles else ""
                broker = f" _[{a['brokerage']}]_" if a.get("brokerage") else ""
                tickers = ", ".join(a.get("tickers", [])[:3])
                ticker_str = f" `{tickers}`" if tickers else ""
                lines.append(f"  • {a['title'][:55]}{broker}{ticker_str}{flag}")
            if flagged_articles:
                lines.append(f"  ⭐ = matches active pipeline idea")
            lines.append("")

        # Dividends
        if dividends:
            lines.append(f"*Upcoming Dividends (next 7 days)*")
            for d in dividends[:5]:
                amt = f" — {d['dividend_amount']} MYR" if d.get("dividend_amount") else ""
                yld = f" ({d['current_yield_pct']:.1f}%)" if d.get("current_yield_pct") else ""
                lines.append(f"  • `{d['symbol']}` {d['name'][:20]} | Ex: {d['ex_date']}{amt}{yld}")
            lines.append("")

        # Research angle
        if research_angle:
            lines.append(f"*Today's Research Focus*")
            lines.append(f"  Least-covered angle: `{research_angle}`")
            lines.append("")

        lines.append(f"_OpenClaw v1 | {now_kl.strftime('%H:%M')} KLT_")
        return "\n".join(lines)

    def _send_telegram(self, text: str) -> bool:
        """Send message to the configured Telegram chat."""
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            logger.warning("Morning briefing: Telegram not configured — skipping send")
            return False
        try:
            resp = requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                json={
                    "chat_id":    TELEGRAM_CHAT_ID,
                    "text":       text,
                    "parse_mode": "Markdown",
                },
                timeout=15,
            )
            resp.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Morning briefing: Telegram send failed: {e}")
            return False


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    init_db()
    result = MorningBriefing().generate_briefing()
    print(f"\nBriefing result: {result}")
