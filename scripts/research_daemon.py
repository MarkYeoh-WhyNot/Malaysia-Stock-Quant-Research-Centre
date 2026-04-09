import asyncio
import json
import logging
import signal
import time
from datetime import datetime, timedelta

from agents.backtest_engineer.backtest_engineer import BacktestEngineer
from agents.data_engineer.data_engineer import DataEngineer
from agents.portfolio_executor.portfolio_executor import PortfolioExecutor
from agents.red_blue_team.red_blue_team import RedBlueTeam
from agents.researcher.strategy_researcher import StrategyResearcher
from agents.risk_monitor.risk_monitor import RiskMonitor
from config.settings import key_health_check
from data.database import db_session, init_db
from knowledge.ingestion.diversity_engine import DiversityEngine
from knowledge.ingestion.alpha_seeds import AlphaSeedGenerator
from scripts.morning_briefing import MorningBriefing

logger = logging.getLogger("openclaw.daemon")


class ResearchDaemon:
    def __init__(self, scan_interval=60):
        self.scan_interval      = scan_interval
        self.running            = False
        self.cycle_count        = 0
        self.researcher         = StrategyResearcher()
        self.risk_monitor       = RiskMonitor()
        self.data_engineer      = DataEngineer()
        self.backtest_engineer  = BacktestEngineer()
        self.red_blue_team      = RedBlueTeam()
        self.portfolio_executor = PortfolioExecutor()
        self.diversity_engine   = DiversityEngine()
        self._last_kb_hunt: datetime | None = None
        self._last_briefing: datetime | None = None
        self._last_alpha_seeds: datetime | None = None
        self._last_klse_refresh: datetime | None = None
        self._last_screener_ideas: datetime | None = None

    def start(self):
        logger.info("OpenClaw Research Daemon starting...")
        init_db()
        # Security check — log key health without exposing the actual key
        kh = key_health_check()
        if kh["issues"]:
            for issue in kh["issues"]:
                logger.warning(f"[Security] {issue}")
        else:
            logger.info(f"[Security] Key health check passed (key={kh['key_preview']})")
        self.running = True
        signal.signal(signal.SIGINT,  self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)
        asyncio.run(self._main_loop())

    def _shutdown(self, *args):
        logger.info("Daemon shutting down...")
        self.running = False

    async def _main_loop(self):
        while self.running:
            self.cycle_count += 1
            start = time.time()
            logger.info(f"[Daemon] Scan cycle #{self.cycle_count}")
            try:
                await self._scan_and_dispatch()
            except Exception as e:
                logger.error(f"[Daemon] Cycle error: {e}", exc_info=True)
            elapsed = time.time() - start
            await asyncio.sleep(max(0, self.scan_interval - elapsed))

    async def _scan_and_dispatch(self):
        # Write scan heartbeat to daemon_logs so the dashboard can track liveness
        try:
            with db_session() as conn:
                conn.execute(
                    "INSERT INTO daemon_logs (level, source, message) VALUES ('INFO', 'ResearchDaemon', ?)",
                    (f"Scan cycle #{self.cycle_count}",)
                )
        except Exception:
            pass

        health = self.risk_monitor.pipeline_health_report()
        logger.info(
            f"[Daemon] health={health['health']} ideas={health['total_ideas']} "
            f"spend=${health['daily_spend']:.2f}"
        )
        await self._process_gate0()
        await self._process_stage1()
        await self._process_stage2()
        await self._process_red_blue()
        await self._process_stage3()
        await self._process_paper_trading()
        await self._daily_knowledge_hunt()
        await self._process_alpha_seeds()
        await self._process_morning_briefing()
        await self._process_klse_refresh()
        await self._process_screener_ideas()

    # ── Stage 0 — novelty / logic screen ─────────────────────────────────────

    async def _process_gate0(self):
        with db_session() as conn:
            pending = conn.execute(
                "SELECT id, title FROM alpha_ideas "
                "WHERE stage='gate0' AND status='pending' LIMIT 5"
            ).fetchall()
        for row in pending:
            try:
                result = self.researcher.score_gate0(row["id"])
                logger.info(
                    f"[Gate0] {'PASS' if result.get('pass') else 'FAIL'}: {row['title'][:50]}"
                )
                if not result.get("pass"):
                    try:
                        from knowledge.ingestion.rejection_memory import RejectionMemory
                        RejectionMemory().record_rejection(
                            row["id"], result.get("rationale", ""), "gate0"
                        )
                    except Exception as re:
                        logger.warning(f"[Gate0] RejectionMemory failed: {re}")
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"[Gate0] Error {row['id']}: {e}")

    # ── Stage 1 — deep research ───────────────────────────────────────────────

    async def _process_stage1(self):
        with db_session() as conn:
            pending = conn.execute(
                "SELECT id, title FROM alpha_ideas "
                "WHERE stage='stage1' AND status='active' AND research_score IS NULL LIMIT 3"
            ).fetchall()
        for row in pending:
            try:
                result = self.researcher.research_idea(row["id"])
                logger.info(
                    f"[Stage1] {row['title'][:50]} score={result.get('research_score')}"
                )
                await asyncio.sleep(2)
            except Exception as e:
                logger.error(f"[Stage1] Error {row['id']}: {e}")

    # ── Stage 2 — backtest (Gates 2+3) ───────────────────────────────────────

    async def _process_stage2(self):
        """Run BacktestEngineer on stage2/active ideas that have no backtest run yet.

        DataEngineer pre-caches the price data; BacktestEngineer reads from that cache.
        After a single-stock backtest pass (Gates 2+3), cross_sectional_test() validates
        that the factor generalises across the full KLCI universe.  Only ideas that pass
        BOTH gates advance to stage3.
        On fail at either gate: idea is rejected.
        Ideas are set to status='processing' before the backtest starts to prevent
        re-queueing if the daemon cycles while a long backtest is running.
        """
        # ── Stuck idea detector ───────────────────────────────────────────────
        # Reset ideas stuck in 'processing' (daemon restart mid-backtest) or
        # 'active' (stale, >30 min untouched) back to 'pending' so they are retried.
        with db_session() as conn:
            cur = conn.execute(
                "UPDATE alpha_ideas SET status='pending', updated_at=datetime('now') "
                "WHERE stage='stage2' AND status IN ('processing', 'active') "
                "AND updated_at < datetime('now', '-30 minutes')"
            )
            stuck_count = cur.rowcount
        if stuck_count:
            logger.warning(f"[Stage2] Unstuck {stuck_count} idea(s) older than 30 min → reset to pending")
            try:
                with db_session() as conn:
                    conn.execute(
                        "INSERT INTO daemon_logs (level, source, message) VALUES ('WARN', 'ResearchDaemon', ?)",
                        (f"[Stage2] Unstuck {stuck_count} idea(s) in processing/active > 30 min — reset to pending",)
                    )
            except Exception:
                pass

        with db_session() as conn:
            pending = conn.execute(
                "SELECT id, title, ticker, factor_formula FROM alpha_ideas "
                "WHERE stage='stage2' AND status IN ('active', 'pending') "
                "AND id NOT IN (SELECT DISTINCT idea_id FROM backtest_runs) "
                "LIMIT 3"
            ).fetchall()

        for row in pending:
            try:
                # Mark as processing so the next daemon cycle skips this idea
                with db_session() as conn:
                    conn.execute(
                        "UPDATE alpha_ideas SET status='processing', updated_at=datetime('now') WHERE id=?",
                        (row["id"],),
                    )
                ticker = row["ticker"] or "1155.KL"
                # Pre-cache 5yr daily bars so BacktestEngineer hits the cache
                self.data_engineer.fetch_prices(ticker, days=1825, use_cache=True)
                result = self.backtest_engineer.backtest_idea(row["id"])
                # Surface error returns from backtest as exceptions so the except block handles them
                if result.get("error"):
                    raise RuntimeError(f"Backtest returned error: {result['error']}")
                logger.info(
                    f"[Stage2] {'PASS' if result.get('gate3_pass') else 'FAIL'}: "
                    f"{row['title'][:50]} "
                    f"train={result.get('train', {}).get('sharpe', 0):.2f} "
                    f"test={result.get('test', {}).get('sharpe', 0):.2f}"
                )
                if not result.get("gate3_pass") and not result.get("error"):
                    try:
                        from knowledge.ingestion.rejection_memory import RejectionMemory
                        reason = (
                            f"Backtest failed G2/G3 — "
                            f"train_sharpe={result.get('train', {}).get('sharpe', 0):.2f} "
                            f"val_sharpe={result.get('val', {}).get('sharpe', 0):.2f} "
                            f"test_sharpe={result.get('test', {}).get('sharpe', 0):.2f}"
                        )
                        RejectionMemory().record_rejection(row["id"], reason, "stage2")
                    except Exception as re:
                        logger.warning(f"[Stage2] RejectionMemory failed: {re}")

                # ── Cross-sectional validation gate ───────────────────────────
                if result.get("gate3_pass"):
                    await asyncio.sleep(1)
                    cs = self.backtest_engineer.cross_sectional_test(
                        result.get("factor_formula", ""), row["id"]
                    )
                    logger.info(
                        f"[Stage2-CS] idea={row['id']} "
                        f"mean_IC={cs.get('mean_ic', 0):.3f} "
                        f"t={cs.get('ic_tstat', 0):.2f} "
                        f"pos_stocks={cs.get('stocks_positive_ic', 0)}/{cs.get('stocks_tested', 0)} "
                        f"real={cs.get('factor_is_real')}"
                    )
                    if not cs.get("factor_is_real"):
                        # Factor fails cross-sectional breadth — reverse stage3 promotion
                        best = cs.get("best_stocks", [])
                        best_names = ", ".join(s["symbol"] for s in best[:3]) if best else "none"
                        reason = (
                            f"Factor does not generalise across KLCI universe — "
                            f"mean_IC={cs.get('mean_ic', 0):.3f} "
                            f"t-stat={cs.get('ic_tstat', 0):.2f} "
                            f"positive_stocks={cs.get('stocks_positive_ic', 0)}/30"
                        )
                        with db_session() as conn:
                            conn.execute(
                                "UPDATE alpha_ideas SET stage='stage2', status='rejected', "
                                "updated_at=datetime('now') WHERE id=?",
                                (row["id"],),
                            )
                            conn.execute(
                                "INSERT INTO pipeline_events "
                                "(idea_id, stage, event_type, agent, notes) "
                                "VALUES (?, 'stage2', 'rejected', 'BacktestEngineer', ?)",
                                (row["id"], reason),
                            )
                            conn.execute(
                                "INSERT INTO gate_decisions "
                                "(idea_id, gate, decision, decided_by, rationale) "
                                "VALUES (?, 'gate_cs', 'reject', 'BacktestEngineer', ?)",
                                (row["id"], reason),
                            )
                        logger.warning(
                            f"[Stage2-CS] REJECTED [{row['id']}] {row['title'][:50]} — {reason}"
                        )
                        try:
                            from knowledge.ingestion.rejection_memory import RejectionMemory
                            RejectionMemory().record_rejection(row["id"], reason, "stage2_cs")
                        except Exception as re:
                            logger.warning(f"[Stage2-CS] RejectionMemory failed: {re}")
                    else:
                        # Factor is real — save best_stocks in pipeline event and confirm stage3
                        best = cs.get("best_stocks", [])
                        best_names = ", ".join(
                            f"{s['symbol']}({s['ic']:.3f})" for s in best[:5]
                        )
                        with db_session() as conn:
                            conn.execute(
                                "INSERT INTO pipeline_events "
                                "(idea_id, stage, event_type, agent, notes) "
                                "VALUES (?, 'stage2', 'cs_passed', 'BacktestEngineer', ?)",
                                (row["id"],
                                 f"CS PASS mean_IC={cs.get('mean_ic', 0):.3f} "
                                 f"t={cs.get('ic_tstat', 0):.2f} "
                                 f"best_stocks=[{best_names}]"),
                            )
                        logger.info(
                            f"[Stage2-CS] ADVANCED [{row['id']}] to stage3 "
                            f"best_stocks=[{best_names}]"
                        )

                await asyncio.sleep(2)
            except Exception as e:
                import traceback as _tb
                err_detail = _tb.format_exc()
                logger.error(f"[Stage2] Error idea={row['id']}: {e}\n{err_detail}")
                # Log to daemon_logs for dashboard visibility
                try:
                    with db_session() as conn:
                        conn.execute(
                            "INSERT INTO daemon_logs (level, source, message) VALUES ('ERROR', 'ResearchDaemon', ?)",
                            (f"[Stage2] Backtest failed idea={row['id']} '{row['title'][:60]}': {str(e)[:400]}",)
                        )
                except Exception:
                    pass
                # Mark idea as failed so it doesn't loop endlessly
                with db_session() as conn:
                    conn.execute(
                        "UPDATE alpha_ideas "
                        "SET status='failed', rejection_reason=?, updated_at=datetime('now') "
                        "WHERE id=? AND status='processing'",
                        (f"Backtest exception: {str(e)[:500]}", row["id"]),
                    )

    # ── Red-Blue adversarial review ───────────────────────────────────────────

    async def _process_red_blue(self):
        """Run RedBlueTeam stress-test on stage3/active ideas with no red-blue decision yet.

        On advance/conditional verdict: idea moves to stage4a.
        On reject: idea is archived.
        """
        with db_session() as conn:
            pending = conn.execute(
                "SELECT id, title FROM alpha_ideas "
                "WHERE stage='stage3' AND status='active' "
                "AND id NOT IN ("
                "  SELECT idea_id FROM gate_decisions WHERE gate='gate3_rb'"
                ") LIMIT 2"
            ).fetchall()

        for row in pending:
            try:
                result = self.red_blue_team.stress_test(row["id"])
                logger.info(
                    f"[RedBlue] {row['title'][:50]} "
                    f"verdict={result.get('verdict')} "
                    f"advanced={result.get('advanced')}"
                )
                await asyncio.sleep(3)
            except Exception as e:
                logger.error(f"[RedBlue] Error {row['id']}: {e}")

    # ── Stage 3 — recovery / final validation ────────────────────────────────

    async def _process_stage3(self):
        """Safety-net for stage3/active ideas in anomalous states.

        Case A: has red-blue approval but still in stage3 → promote to stage4a.
        Case B: no backtest run at all (shouldn't normally happen) → run backtest.
        """
        with db_session() as conn:
            # Case A: approved by red-blue but not yet promoted
            stuck = conn.execute(
                "SELECT id, title FROM alpha_ideas "
                "WHERE stage='stage3' AND status='active' "
                "AND id IN ("
                "  SELECT idea_id FROM gate_decisions "
                "  WHERE gate='gate3_rb' AND decision='approve'"
                ") LIMIT 5"
            ).fetchall()

            # Case B: no backtest run at all
            no_bt = conn.execute(
                "SELECT id, title, ticker FROM alpha_ideas "
                "WHERE stage='stage3' AND status='active' "
                "AND id NOT IN (SELECT DISTINCT idea_id FROM backtest_runs) "
                "LIMIT 2"
            ).fetchall()

        for row in stuck:
            try:
                with db_session() as conn:
                    conn.execute(
                        "UPDATE alpha_ideas SET stage='stage4a', updated_at=datetime('now') "
                        "WHERE id=? AND stage='stage3'",
                        (row["id"],),
                    )
                logger.info(f"[Stage3] Promoted stuck idea [{row['id']}] to stage4a: {row['title'][:50]}")
            except Exception as e:
                logger.error(f"[Stage3] Promotion error {row['id']}: {e}")

        for row in no_bt:
            try:
                ticker = row["ticker"] or "1155.KL"
                self.data_engineer.fetch_prices(ticker, days=1825, use_cache=True)
                result = self.backtest_engineer.backtest_idea(row["id"])
                logger.info(
                    f"[Stage3-BT] Recovery backtest [{row['id']}] "
                    f"pass={result.get('gate3_pass')}"
                )
                await asyncio.sleep(2)
            except Exception as e:
                logger.error(f"[Stage3-BT] Error {row['id']}: {e}")

    # ── Stage 4a — paper trading monitoring ──────────────────────────────────

    async def _process_paper_trading(self):
        """Monitor stage4a/active ideas for drawdown breaches and Gate 4a evaluation.

        Uses RiskMonitor to check per-idea drawdown from closed paper trades.
        If drawdown is breached, the idea is rejected.
        If Gate 4a criteria are met (30+ days, Sharpe ≥ 0.80, DD ≤ 20%), the idea
        advances to stage4b via PortfolioExecutor.evaluate_paper_performance().
        """
        with db_session() as conn:
            ideas = conn.execute(
                "SELECT id, title FROM alpha_ideas "
                "WHERE stage='stage4a' AND status='active' LIMIT 5"
            ).fetchall()

        for row in ideas:
            try:
                dd_result = self.risk_monitor.check_strategy_drawdown(row["id"])
                if dd_result.get("breached"):
                    with db_session() as conn:
                        conn.execute(
                            "UPDATE alpha_ideas SET status='rejected', updated_at=datetime('now') "
                            "WHERE id=?",
                            (row["id"],),
                        )
                    logger.warning(
                        f"[Stage4a] Drawdown breach — rejecting [{row['id']}]: "
                        f"{row['title'][:40]} dd={dd_result.get('drawdown', 0):.1%}"
                    )
                elif dd_result.get("trade_count", 0) > 0:
                    # Evaluate Gate 4a if there are closed paper trades
                    eval_result = await self.portfolio_executor.evaluate_paper_performance(row["id"])
                    if eval_result.get("gate4a_pass"):
                        logger.info(
                            f"[Stage4a] GATE4A PASS [{row['id']}]: {row['title'][:40]} "
                            f"sharpe={eval_result.get('sharpe', 0):.2f} "
                            f"days={eval_result.get('total_days', 0)}"
                        )
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"[Stage4a] Error {row['id']}: {e}")

    # ── Hourly alpha seed processing ─────────────────────────────────────────

    async def _process_alpha_seeds(self):
        """Run AlphaSeedGenerator.process_undigested() once per hour.

        Guarded by a 55-minute minimum gap so it fires at most once per daemon hour.
        Processes up to 5 undigested KB documents per run.
        """
        now = datetime.utcnow()
        if self._last_alpha_seeds and (now - self._last_alpha_seeds) < timedelta(minutes=55):
            return

        with db_session() as conn:
            undigested = conn.execute(
                "SELECT COUNT(*) as n FROM kb_documents WHERE seeded=0"
            ).fetchone()["n"]

        if undigested == 0:
            return

        logger.info(f"[AlphaSeeds] {undigested} undigested docs — processing up to 5...")
        try:
            result = AlphaSeedGenerator().process_undigested(limit=5)
            self._last_alpha_seeds = now
            logger.info(
                f"[AlphaSeeds] processed={result['processed']} "
                f"ideas_created={result['total_ideas_created']} "
                f"skipped={result['skipped']}"
            )
        except Exception as e:
            logger.error(f"[AlphaSeeds] Error: {e}", exc_info=True)

    # ── Daily knowledge diversity hunt ────────────────────────────────────────

    async def _daily_knowledge_hunt(self):
        """Run DiversityEngine.daily_hunt() once per day at ~22:00 UTC (06:00 KL time).

        Guards against multiple runs in the same UTC-22 hour window by tracking
        _last_kb_hunt. A 20-hour minimum gap ensures one run per calendar day.
        """
        now = datetime.utcnow()
        if now.hour != 22:
            return
        if self._last_kb_hunt and (now - self._last_kb_hunt) < timedelta(hours=20):
            return

        logger.info("[DailyHunt] Starting daily knowledge diversity hunt...")
        try:
            result = self.diversity_engine.daily_hunt()
            self._last_kb_hunt = now
            logger.info(
                f"[DailyHunt] Complete — angle='{result.get('target_angle')}' "
                f"found={result.get('papers_found', 0)} "
                f"ingested={result.get('papers_ingested', 0)}"
            )
        except Exception as e:
            logger.error(f"[DailyHunt] Error: {e}", exc_info=True)


    # ── Morning briefing — 00:00 UTC (08:00 KL time) ─────────────────────────

    async def _process_morning_briefing(self):
        """Send the daily morning briefing once per day at 00:00 UTC (08:00 KL).

        Guarded by a 20-hour minimum gap to prevent duplicate sends.
        """
        now = datetime.utcnow()
        if now.hour != 0:
            return
        if self._last_briefing and (now - self._last_briefing) < timedelta(hours=20):
            return

        logger.info("[MorningBriefing] Generating daily briefing...")
        try:
            result = MorningBriefing().generate_briefing()
            self._last_briefing = now
            logger.info(
                f"[MorningBriefing] Done — sent={result.get('sent')} "
                f"articles={result.get('articles')} dividends={result.get('dividends')}"
            )
        except Exception as e:
            logger.error(f"[MorningBriefing] Error: {e}", exc_info=True)


    # ── KLSE fundamental refresh — 10:00 UTC (18:00 MYT, after market close) ──

    async def _process_klse_refresh(self):
        """Refresh KLCI fundamental data once per day at 10:00 UTC.

        Scrapes klsescreener.com stock pages for all SLUG_MAP stocks and upserts
        into fundamental_data, quarterly_history, and dividend_history tables.
        """
        now = datetime.utcnow()
        if now.hour != 10:
            return
        if self._last_klse_refresh and (now - self._last_klse_refresh) < timedelta(hours=20):
            return

        logger.info("[KLSERefresh] Starting KLCI fundamental data refresh...")
        try:
            from data.klse_screener.fundamental_scraper import KLSEFundamentalScraper
            result = KLSEFundamentalScraper().refresh_all_klci()
            self._last_klse_refresh = now
            logger.info(
                f"[KLSERefresh] Complete — success={result['success']} "
                f"failed={result['failed']}"
            )
        except Exception as e:
            logger.error(f"[KLSERefresh] Error: {e}", exc_info=True)

    # ── Screener-driven idea generation — 11:00 UTC (19:00 MYT) ─────────────

    async def _process_screener_ideas(self):
        """Run KLSEProactiveScreener and generate ideas once per day at 11:00 UTC."""
        now = datetime.utcnow()
        if now.hour != 11:
            return
        if self._last_screener_ideas and (now - self._last_screener_ideas) < timedelta(hours=20):
            return

        logger.info("[ScreenerIdeas] Running 8-screen KLSE idea generation...")
        try:
            generated = self.researcher.generate_screener_ideas()
            self._last_screener_ideas = now
            logger.info(f"[ScreenerIdeas] Complete — {generated} ideas created")
        except Exception as e:
            logger.error(f"[ScreenerIdeas] Error: {e}", exc_info=True)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    ResearchDaemon(scan_interval=60).start()
