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
from scripts.alerts import send_alert

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
        # last-run timestamps per scheduled job, persisted in job_state so
        # daily jobs catch up after downtime instead of being skipped
        self._job_last_run: dict[str, datetime | None] = {}

    # ── Scheduler helpers (persisted, catch-up aware) ─────────────────────────

    def _load_job_state(self):
        try:
            with db_session() as conn:
                rows = conn.execute("SELECT job_name, last_run_utc FROM job_state").fetchall()
            for r in rows:
                try:
                    self._job_last_run[r["job_name"]] = datetime.fromisoformat(r["last_run_utc"])
                except (ValueError, TypeError):
                    pass
            if self._job_last_run:
                logger.info(f"[Scheduler] Restored last-run state for {len(self._job_last_run)} jobs")
        except Exception as e:
            logger.warning(f"[Scheduler] Could not load job state: {e}")

    def _job_due(self, name: str, daily_at_hour: int | None = None,
                 min_gap: timedelta | None = None) -> bool:
        """Whether a scheduled job should run now (UTC).

        Daily jobs are due from `last_run.date() + 1 day at daily_at_hour`
        onward — if the daemon was down or busy during the target hour, the job
        runs on the next cycle instead of being skipped for the day. A job that
        has never run is due immediately.
        """
        now = datetime.utcnow()
        last = self._job_last_run.get(name)
        if last is None:
            return True
        if daily_at_hour is not None:
            from datetime import time as dt_time
            due = datetime.combine(last.date() + timedelta(days=1),
                                   dt_time(hour=daily_at_hour))
            return now >= due
        if min_gap is not None:
            return (now - last) >= min_gap
        return True

    def _mark_job_run(self, name: str):
        now = datetime.utcnow()
        self._job_last_run[name] = now
        try:
            with db_session() as conn:
                conn.execute(
                    "INSERT INTO job_state (job_name, last_run_utc) VALUES (?, ?) "
                    "ON CONFLICT(job_name) DO UPDATE SET last_run_utc=excluded.last_run_utc",
                    (name, now.isoformat()),
                )
        except Exception as e:
            logger.warning(f"[Scheduler] Could not persist job state for {name}: {e}")

    def start(self):
        logger.info("OpenClaw Research Daemon starting...")
        init_db()
        self._load_job_state()
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
        # Doubles as the "daemon restarted" notice — a fresh alert here after an
        # unexpected exit tells the operator the container's restart policy
        # kicked in, without needing separate crash-vs-restart bookkeeping.
        send_alert("daemon started")
        try:
            asyncio.run(self._main_loop())
        except Exception as e:
            logger.error(f"[Daemon] Fatal error: {e}", exc_info=True)
            send_alert(f"daemon crashed: {e}")
            raise

    def _shutdown(self, *args):
        logger.info("Daemon shutting down...")
        self.running = False

    def _touch_heartbeat(self):
        """Freshness file for the Docker healthcheck (docker/healthcheck_daemon.sh)."""
        try:
            from config.settings import DB_PATH
            (DB_PATH.parent / "daemon_heartbeat").touch()
        except Exception:
            pass

    async def _main_loop(self):
        while self.running:
            self.cycle_count += 1
            start = time.time()
            logger.info(f"[Daemon] Scan cycle #{self.cycle_count}")
            self._touch_heartbeat()
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
        await self._process_cpo_daily()
        await self._process_analyst_monitor()
        await self._process_db_maintenance()

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
                if result.get("retry"):
                    logger.warning(f"[Gate0] RETRY (parse failure): {row['title'][:50]}")
                    continue
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
        """Drive stage4a/active ideas: signal-driven entries/exits, daily NAV
        mark-to-market, drawdown breach rejection, and Gate 4a evaluation.

        Each idea's strategy signal is recomputed daily from the params stored
        by its passing backtest run (no LLM cost). Positions are opened/closed
        at the latest cached KLSE close via PortfolioExecutor.daily_update().
        """
        import json as _json

        with db_session() as conn:
            ideas = conn.execute(
                "SELECT id, title, ticker FROM alpha_ideas "
                "WHERE stage='stage4a' AND status='active' LIMIT 5"
            ).fetchall()

        today = datetime.utcnow().strftime("%Y-%m-%d")
        for row in ideas:
            try:
                with db_session() as conn:
                    done_today = conn.execute(
                        "SELECT 1 FROM paper_equity WHERE idea_id=? AND date=?",
                        (row["id"], today),
                    ).fetchone()
                if done_today:
                    continue

                with db_session() as conn:
                    bt = conn.execute(
                        "SELECT params, pair FROM backtest_runs "
                        "WHERE idea_id=? AND passed=1 ORDER BY id DESC LIMIT 1",
                        (row["id"],),
                    ).fetchone()
                if not bt or not bt["params"]:
                    logger.warning(
                        f"[Stage4a] No passed backtest params for [{row['id']}] — cannot paper trade"
                    )
                    continue
                params = _json.loads(bt["params"])
                ticker = row["ticker"] or bt["pair"]

                update = await self.portfolio_executor.daily_update(row["id"], ticker, params)
                logger.info(
                    f"[Stage4a] [{row['id']}] {ticker} signal={update.get('signal')} "
                    f"action={update.get('action')} nav={update.get('nav')}"
                )

                dd_result = self.risk_monitor.check_strategy_drawdown(row["id"])
                if dd_result.get("breached"):
                    open_trade = self.portfolio_executor._open_trade(row["id"])
                    if open_trade:
                        await self.portfolio_executor.paper_exit(open_trade["id"])
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
                    continue

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
        if not self._job_due("alpha_seeds", min_gap=timedelta(minutes=55)):
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
            self._mark_job_run("alpha_seeds")
            logger.info(
                f"[AlphaSeeds] processed={result['processed']} "
                f"ideas_created={result['total_ideas_created']} "
                f"skipped={result['skipped']}"
            )
        except Exception as e:
            logger.error(f"[AlphaSeeds] Error: {e}", exc_info=True)

    # ── Daily knowledge diversity hunt ────────────────────────────────────────

    async def _daily_knowledge_hunt(self):
        """Run DiversityEngine.daily_hunt() once per day, due from 22:00 UTC
        (06:00 KL). Catch-up aware: if the daemon was down at 22:00, the hunt
        runs on the next cycle instead of skipping the day."""
        if not self._job_due("kb_hunt", daily_at_hour=22):
            return

        logger.info("[DailyHunt] Starting daily knowledge diversity hunt...")
        try:
            result = self.diversity_engine.daily_hunt()
            self._mark_job_run("kb_hunt")
            logger.info(
                f"[DailyHunt] Complete — angle='{result.get('target_angle')}' "
                f"found={result.get('papers_found', 0)} "
                f"ingested={result.get('papers_ingested', 0)}"
            )
        except Exception as e:
            logger.error(f"[DailyHunt] Error: {e}", exc_info=True)


    # ── Morning briefing — 00:00 UTC (08:00 KL time) ─────────────────────────

    async def _process_morning_briefing(self):
        """Send the daily morning briefing once per day, due from 00:00 UTC
        (08:00 KL). Catch-up aware."""
        if not self._job_due("morning_briefing", daily_at_hour=0):
            return

        logger.info("[MorningBriefing] Generating daily briefing...")
        try:
            result = MorningBriefing().generate_briefing()
            self._mark_job_run("morning_briefing")
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
        if not self._job_due("klse_refresh", daily_at_hour=10):
            return

        logger.info("[KLSERefresh] Starting KLCI fundamental data refresh...")
        try:
            from data.klse_screener.fundamental_scraper import KLSEFundamentalScraper
            result = KLSEFundamentalScraper().refresh_all_klci()
            self._mark_job_run("klse_refresh")
            logger.info(
                f"[KLSERefresh] Complete — success={result['success']} "
                f"failed={result['failed']}"
            )
        except Exception as e:
            logger.error(f"[KLSERefresh] Error: {e}", exc_info=True)

    # ── Screener-driven idea generation — 11:00 UTC (19:00 MYT) ─────────────

    async def _process_screener_ideas(self):
        """Run KLSEProactiveScreener and generate ideas once per day, due from
        11:00 UTC (19:00 MYT). Catch-up aware."""
        if not self._job_due("screener_ideas", daily_at_hour=11):
            return

        logger.info("[ScreenerIdeas] Running 8-screen KLSE idea generation...")
        try:
            generated = self.researcher.generate_screener_ideas()
            self._mark_job_run("screener_ideas")
            logger.info(f"[ScreenerIdeas] Complete — {generated} ideas created")
        except Exception as e:
            logger.error(f"[ScreenerIdeas] Error: {e}", exc_info=True)


    # ── CPO daily signal — 01:00 UTC (09:00 MYT) ──────────────────────────────

    async def _process_cpo_daily(self):
        """Run the CPO/palm-oil daily signal once per day (folded in from
        scripts/cpo_daily.py so it gets scheduler catch-up and supervision)."""
        if not self._job_due("cpo_daily", daily_at_hour=1):
            return
        logger.info("[CPODaily] Running CPO daily signal...")
        try:
            from scripts.cpo_daily import main as cpo_main
            await asyncio.get_event_loop().run_in_executor(None, cpo_main)
            self._mark_job_run("cpo_daily")
            logger.info("[CPODaily] Complete")
        except Exception as e:
            logger.error(f"[CPODaily] Error: {e}", exc_info=True)

    # ── Analyst coverage monitor — 02:00 UTC (10:00 MYT) ─────────────────────

    async def _process_analyst_monitor(self):
        """Run the analyst coverage-initiation tracker once per day (folded in
        from scripts/analyst_monitor.py)."""
        if not self._job_due("analyst_monitor", daily_at_hour=2):
            return
        logger.info("[AnalystMonitor] Running analyst coverage monitor...")
        try:
            from scripts.analyst_monitor import main as analyst_main
            await asyncio.get_event_loop().run_in_executor(None, analyst_main)
            self._mark_job_run("analyst_monitor")
            logger.info("[AnalystMonitor] Complete")
        except Exception as e:
            logger.error(f"[AnalystMonitor] Error: {e}", exc_info=True)


    # ── Nightly DB maintenance — 03:00 UTC (11:00 MYT) ───────────────────────

    async def _process_db_maintenance(self):
        """Prune unbounded log/usage tables and take a compressed DB backup,
        once per day. Both are cheap, local, no-LLM-cost operations."""
        if not self._job_due("db_maintenance", daily_at_hour=3):
            return
        logger.info("[DBMaintenance] Pruning old logs and backing up database...")
        try:
            with db_session() as conn:
                logs_deleted = conn.execute(
                    "DELETE FROM daemon_logs WHERE created_at < datetime('now', '-30 days')"
                ).rowcount
                usage_deleted = conn.execute(
                    "DELETE FROM ai_usage WHERE created_at < datetime('now', '-90 days')"
                ).rowcount
            # Separate connection/transaction: TRUNCATE checkpoint needs exclusive
            # WAL access and can't run inside the delete transaction above.
            try:
                with db_session() as conn:
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except Exception as _wal_exc:
                logger.warning(f"[DBMaintenance] wal_checkpoint skipped: {_wal_exc}")
            logger.info(f"[DBMaintenance] Pruned daemon_logs={logs_deleted} ai_usage={usage_deleted}")

            from scripts.backup_db import run_backup
            backup_result = await asyncio.get_event_loop().run_in_executor(None, run_backup)
            logger.info(f"[DBMaintenance] Backup: {backup_result['file']}")

            self._mark_job_run("db_maintenance")
        except Exception as e:
            logger.error(f"[DBMaintenance] Error: {e}", exc_info=True)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    ResearchDaemon(scan_interval=60).start()
