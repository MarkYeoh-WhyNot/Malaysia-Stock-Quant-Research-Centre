import asyncio, logging, os, sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

from telegram import Update, BotCommand
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from data.database import db_session, init_db
from agents.researcher.strategy_researcher import StrategyResearcher
from agents.risk_monitor.risk_monitor import RiskMonitor
from knowledge.ingestion.kb_ingester import KBIngester
from knowledge.ingestion.research_hunter import ResearchHunter
from knowledge.ingestion.diversity_engine import DiversityEngine
from knowledge.ingestion.alpha_seeds import AlphaSeedGenerator
from data.i3investor.scraper import I3investorScraper
from data.klse.fundamental_scanner import FundamentalScanner
from scripts.morning_briefing import MorningBriefing

logger = logging.getLogger("openclaw.telegram")

BOT_TOKEN     = os.getenv("TELEGRAM_BOT_TOKEN", "")
ADMIN_CHAT    = os.getenv("TELEGRAM_CHAT_ID", "")
ALLOWED_CHATS = set(c.strip() for c in os.getenv("TELEGRAM_ALLOWED_CHATS", ADMIN_CHAT).split(",") if c.strip())

def is_allowed(update: Update) -> bool:
    return str(update.effective_chat.id) in ALLOWED_CHATS or not ALLOWED_CHATS

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🐋 *Mark's Research Centre — Bursa Malaysia KLSE*\n"
        "_Quantitative equity research pipeline_\n\n"
        "*Pipeline*\n"
        "/status — health report: stages, spend, errors\n"
        "/ideas — last 8 active ideas with stage & Sharpe\n"
        "/spend — AI cost breakdown by model today\n\n"
        "*Research*\n"
        "/generate `[topic]` — generate KLCI equity ideas via Claude\n"
        "/screen — scrape live KLSE data + i3investor coverage\n"
        "/research `<idea_id>` — hunt academic papers for an idea\n\n"
        "*Market Data*\n"
        "/briefing — send morning briefing now\n"
        "/dividends — ex-dividend dates in next 14 days\n"
        "/fundamentals `<ticker>` — full fundamentals + quarters + dividends\n"
        "/dividend_calendar — upcoming ex-dividend dates from screener data\n"
        "/epf — EPF ownership tracker: accumulating/distributing stocks\n"
        "/analyst — analyst coverage initiations last 7 days\n"
        "/cpo — CPO lag signal report for plantation stocks\n\n"
        "*Knowledge Base*\n"
        "/kb `<url|text>` — ingest a URL or text into knowledge base\n"
        "/search `<query>` — full-text search across KB documents\n"
        "/diversity — KB coverage by research angle\n"
        "/digest `<doc_id|all>` — generate alpha ideas from KB documents\n\n"
        "*Event Engine*\n"
        "/events — event feed (last 24h): earnings, dividends, contracts, macro\n"
        "/calendar — upcoming macro events: BNM OPR, Fed, China PMI\n"
        "/event_stats — event engine health & stats\n\n"
        "*Arsenal*\n"
        "/arsenal — all quantitative techniques with implemented status\n"
        "/arsenal `<key>` — full detail for one technique\n\n"
        "/start — show this help",
        parse_mode="Markdown"
    )

async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update): return
    monitor = RiskMonitor()
    health  = monitor.pipeline_health_report()

    today = datetime.utcnow().strftime("%Y-%m-%d")
    with db_session() as conn:
        today_count = conn.execute(
            "SELECT COUNT(*) as c FROM alpha_ideas WHERE created_at LIKE ?",
            (f"{today}%",)
        ).fetchone()["c"]
        stage_counts = conn.execute(
            "SELECT stage, COUNT(*) as c FROM alpha_ideas "
            "WHERE status='active' GROUP BY stage ORDER BY stage"
        ).fetchall()
        last_scan = conn.execute(
            "SELECT created_at FROM daemon_logs "
            "WHERE source='ResearchDaemon' ORDER BY id DESC LIMIT 1"
        ).fetchone()

    last_scan_str = last_scan["created_at"][:16] if last_scan else "never"

    lines = [
        f"🐋 *Pipeline Status* — {datetime.utcnow().strftime('%H:%M UTC')}",
        f"",
        f"Health: `{health['health'].upper()}`",
        f"Total ideas: `{health['total_ideas']}`",
        f"Ideas today: `{today_count}`",
        f"Today's spend: `${health['daily_spend']:.4f}`",
        f"Errors (1h): `{health['errors_1h']}`",
        f"Last scan: `{last_scan_str}`",
    ]
    if stage_counts:
        lines.append(f"")
        lines.append(f"*Active ideas by stage:*")
        for row in stage_counts:
            lines.append(f"  `{row['stage']}`: {row['c']}")

    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

async def cmd_ideas(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update): return
    with db_session() as conn:
        ideas = conn.execute("""
            SELECT id, title, ticker, stage, novelty_score, logic_score, backtest_sharpe
            FROM alpha_ideas WHERE status='active' ORDER BY id DESC LIMIT 8
        """).fetchall()
    if not ideas:
        await update.message.reply_text("No active ideas in pipeline yet.")
        return
    lines = ["📊 *Recent KLSE Alpha Ideas*\n"]
    for i in ideas:
        sharpe = f"Sharpe={i['backtest_sharpe']:.2f}" if i['backtest_sharpe'] else ""
        ticker = i['ticker'] or '—'
        lines.append(f"[{i['id']}] `{i['stage']}` *{i['title'][:40]}*\n    {ticker} {sharpe}")
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

async def cmd_spend(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update): return
    today = datetime.utcnow().strftime("%Y-%m-%d")
    with db_session() as conn:
        rows  = conn.execute("SELECT model, SUM(cost_usd) as cost, COUNT(*) as calls FROM ai_usage WHERE created_at LIKE ? GROUP BY model ORDER BY cost DESC", (f"{today}%",)).fetchall()
        total = conn.execute("SELECT COALESCE(SUM(cost_usd),0) as t FROM ai_usage WHERE created_at LIKE ?", (f"{today}%",)).fetchone()["t"]
    budget = float(os.getenv("AI_DAILY_BUDGET_USD", "50"))
    lines  = [f"💰 *AI Spend Today*\n", f"Total: `${total:.4f}` / `${budget:.0f}`\n"]
    for r in rows:
        lines.append(f"  `{r['model'].split('-')[1]}`: ${float(r['cost']):.4f} ({r['calls']} calls)")
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

async def cmd_generate(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update): return
    topic = " ".join(ctx.args) if ctx.args else None
    await update.message.reply_text(
        f"🔬 Generating ideas{f' on: _{topic}_' if topic else ''}...\n_(this takes ~30 seconds)_",
        parse_mode="Markdown"
    )
    try:
        researcher = StrategyResearcher()
        result     = researcher.run({"action": "generate", "topic": topic, "count": 3})
        lines      = [f"✅ Created {result['ideas_created']} ideas:\n"]
        for r in result.get("results", []):
            gate = r.get("gate0", {})
            icon = "✓" if gate.get("pass") else "✗"
            lines.append(f"{icon} [{r['id']}] {r['title'][:50]}")
        await update.message.reply_text("\n".join(lines))
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")

async def cmd_screen(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update): return
    await update.message.reply_text(
        "📡 Running 8 KLSE screens on klsescreener.com...\n_(~2 minutes, polite rate limit)_",
        parse_mode="Markdown"
    )
    try:
        from data.klse_screener.screener import KLSEProactiveScreener
        screener = KLSEProactiveScreener()
        results  = screener.run_all_screens()

        total = sum(r["count"] for r in results.values())
        lines = [f"📊 *KLSE Screen Results* — {total} total matches\n"]
        for name, r in results.items():
            if r["count"] == 0:
                continue
            tickers = ", ".join(
                f"`{s['ticker']}`" for s in r["stocks"][:4]
            )
            lines.append(
                f"*{name}* ({r['count']}) — _{r['description']}_\n  {tickers}"
            )

        # Also check i3investor for recent analyst coverage
        try:
            scraper  = I3investorScraper()
            articles = scraper.get_research_articles(max_articles=5)
            if articles:
                lines.append(f"\n📰 *Recent i3investor Coverage*")
                for a in articles[:3]:
                    broker = f" [{a['brokerage']}]" if a.get("brokerage") else ""
                    tickers_str = ", ".join(a.get("tickers", [])[:2])
                    t_str = f" `{tickers_str}`" if tickers_str else ""
                    lines.append(f"  • {a['title'][:50]}{broker}{t_str}")
        except Exception as e2:
            logger.warning(f"i3investor coverage check failed: {e2}")

        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")


async def cmd_fundamentals(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Fetch full fundamental data for a single ticker from klsescreener.com."""
    if not is_allowed(update): return
    text = update.message.text.strip()
    parts = text.split()
    if len(parts) < 2:
        await update.message.reply_text(
            "Usage: /fundamentals 1023.KL"
        )
        return
    ticker = parts[1].upper()
    if not ticker.endswith(".KL"):
        ticker = ticker + ".KL"

    await update.message.reply_text(
        f"📊 Fetching fundamental data for `{ticker}`...\n_(~5 seconds)_",
        parse_mode="Markdown"
    )
    try:
        from data.klse_screener.fundamental_scraper import KLSEFundamentalScraper
        data = KLSEFundamentalScraper().fetch_all(ticker)

        if "error" in data:
            await update.message.reply_text(f"❌ {data['error']}")
            return

        fund = data["fundamentals"]
        lines = [
            f"📊 *{fund.get('name') or ticker}* `{ticker}`\n",
            f"Price:  `{fund.get('price') or '—'} MYR`",
        ]
        for label, key in [
            ("DY",   "dy"),
            ("DPS",  "dps_ttm"),
            ("EPS",  "eps_ttm"),
            ("P/E",  "pe"),
            ("P/B",  "pb"),
            ("ROE",  "roe"),
            ("NTA",  "nta"),
            ("Mkt Cap", "market_cap_b"),
        ]:
            val = fund.get(key)
            if val is not None:
                lines.append(f"{label:<8} `{val}`")
        rsi = fund.get("rsi_14")
        if rsi is not None:
            lines.append(f"RSI(14)  `{rsi}`")

        quarters = data["quarterly_history"][:4]
        if quarters:
            lines.append(f"\n*Last {len(quarters)} Quarters:*")
            for q in quarters:
                qd = q.get("q_date", "?")
                eps_v = q.get("eps")
                dps_v = q.get("dps")
                yoy_v = q.get("yoy_pct")
                parts = [f"`Q{q.get('quarter','')} {qd}`"]
                if eps_v is not None:
                    parts.append(f"EPS={eps_v}")
                if dps_v is not None:
                    parts.append(f"DPS={dps_v}")
                if yoy_v is not None:
                    parts.append(f"YoY={yoy_v:+.1f}%")
                lines.append("  " + "  ".join(parts))

        divs = data["dividend_history"][:4]
        if divs:
            lines.append(f"\n*Last {len(divs)} Dividends:*")
            for d in divs:
                dtype = d.get("dividend_type", "")
                sen = d.get("dps_sen")
                exd = d.get("ex_date", "?")
                lines.append(
                    f"  `{exd}` {d.get('subject', dtype)[:30]} "
                    f"{'— ' + str(sen) + ' sen' if sen else ''}"
                )

        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")


async def cmd_dividend_calendar(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Show confirmed + predicted ex-dividend dates with imminent alerts."""
    if not is_allowed(update): return

    from datetime import date as _date, timedelta as _td
    from collections import defaultdict

    def _short_name(raw: str, ticker: str) -> str:
        """Strip Bursa code suffix e.g. 'MAYBANK (1155)' → 'Maybank'."""
        if not raw:
            return ticker
        # Remove trailing " (NNNN)" Bursa code
        if ' (' in raw:
            raw = raw.split(' (')[0]
        return raw.strip()[:18]

    try:
        today = _date.today()
        today_str = today.isoformat()

        # ── Name lookup from fundamental_data ─────────────────────────────
        name_map: dict = {}
        try:
            with db_session() as conn:
                for r in conn.execute(
                    "SELECT ticker, name FROM fundamental_data "
                    "GROUP BY ticker HAVING MAX(fetched_at)"
                ).fetchall():
                    if r['name']:
                        name_map[r['ticker']] = r['name']
        except Exception:
            pass
        # KLCI short names as fallback
        try:
            from config.settings import KLCI_BY_SYMBOL
            for sym, info in KLCI_BY_SYMBOL.items():
                if sym not in name_map:
                    name_map[sym] = info['name']
        except Exception:
            pass

        # ── SECTION A: Confirmed upcoming ex-dates ─────────────────────────
        with db_session() as conn:
            confirmed = conn.execute("""
                SELECT ticker, ex_date, dps_sen, dividend_type, subject
                FROM dividend_history
                WHERE ex_date >= ?
                ORDER BY ex_date ASC
            """, (today_str,)).fetchall()

        confirmed_tickers = {r['ticker'] for r in confirmed}

        # ── SECTION B: Predicted next ex-dates ────────────────────────────
        cutoff_90 = today + _td(days=90)

        with db_session() as conn:
            all_hist = conn.execute(
                "SELECT ticker, ex_date, dps_sen "
                "FROM dividend_history ORDER BY ticker, ex_date ASC"
            ).fetchall()

        # Group history by ticker
        ticker_dates: dict = defaultdict(list)
        ticker_last_sen: dict = {}
        for r in all_hist:
            ticker_dates[r['ticker']].append(r['ex_date'])
            ticker_last_sen[r['ticker']] = r['dps_sen']  # last in ASC = most recent

        predictions = []
        for ticker, dates in ticker_dates.items():
            if ticker in confirmed_tickers:
                continue  # already confirmed — skip prediction
            if len(dates) < 2:
                continue
            date_objs = sorted(_date.fromisoformat(d) for d in dates)
            intervals = [
                (date_objs[i + 1] - date_objs[i]).days
                for i in range(len(date_objs) - 1)
            ]
            recent = intervals[-3:]  # weight recency
            avg_interval = sum(recent) / len(recent)
            predicted = date_objs[-1] + _td(days=int(avg_interval))
            if predicted < today or predicted > cutoff_90:
                continue
            predictions.append({
                'ticker':         ticker,
                'predicted_date': predicted,
                'est_sen':        ticker_last_sen.get(ticker),
                'days_until':     (predicted - today).days,
            })

        predictions.sort(key=lambda x: x['predicted_date'])
        predictions = predictions[:5]

        # ── Build response ─────────────────────────────────────────────────
        lines = ["📅 *Dividend Calendar*\n"]

        lines.append("✅ *Confirmed upcoming ex-dates:*")
        if confirmed:
            for r in confirmed:
                dt = _date.fromisoformat(r['ex_date'])
                name = _short_name(name_map.get(r['ticker'], ''), r['ticker'])
                date_fmt = dt.strftime("%b %-d")
                days_left = (dt - today).days
                sen_str = f" — {r['dps_sen']:.1f} sen" if r['dps_sen'] else ""
                dtype_str = f" ({r['dividend_type']})" if r['dividend_type'] else ""
                warn = " ⚠️" if days_left <= 7 else ""
                lines.append(
                    f"  `{r['ticker']}` *{name}*  {date_fmt}{sen_str}{dtype_str}{warn}"
                )
        else:
            lines.append("  _No confirmed ex-dates in database._")
            lines.append("  _Run /screen to refresh from KLSE Screener._")

        lines.append("")
        lines.append("🔮 *Predicted next ex-dates (based on history):*")
        if predictions:
            for p in predictions:
                name = _short_name(name_map.get(p['ticker'], ''), p['ticker'])
                date_fmt = p['predicted_date'].strftime("%b %-d")
                est_str = f" — est. ~{p['est_sen']:.0f} sen" if p.get('est_sen') else ""
                lines.append(f"  `{p['ticker']}` *{name}*  ~{date_fmt}{est_str}")
        else:
            lines.append("  _Not enough history to predict._")

        lines.append("")
        lines.append(
            "_⚠️ Predictions are estimates based on historical patterns. "
            "Always verify on Bursa announcements._"
        )

        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

        # ── ISSUE 3: Alert for imminent ex-dates (≤ 7 days) ───────────────
        imminent = [
            r for r in confirmed
            if 0 <= (_date.fromisoformat(r['ex_date']) - today).days <= 7
        ]

        for r in imminent:
            dt = _date.fromisoformat(r['ex_date'])
            days_left = (dt - today).days
            name = _short_name(name_map.get(r['ticker'], ''), r['ticker'])
            date_fmt = dt.strftime("%b %-d, %Y")
            sen_str = f"{r['dps_sen']:.1f} sen" if r['dps_sen'] else "amount TBC"
            dtype_str = r['dividend_type'] or 'dividend'
            window_start = (dt - _td(days=5)).strftime("%b %-d")
            window_end   = (dt - _td(days=1)).strftime("%b %-d")

            alert_lines = [
                f"⚠️ *Ex-dividend alert:*",
                f"  `{r['ticker']}` ({name}) goes ex-dividend in "
                f"*{days_left} day{'s' if days_left != 1 else ''}*",
                f"  ({date_fmt}) — {sen_str} {dtype_str} dividend.",
                f"",
                f"  Pre-ex-dividend drift window: {window_start}–{window_end}",
                f"  Historical pattern: price tends to rise",
                f"  in final 3–5 days before ex-date.",
            ]
            await update.message.reply_text("\n".join(alert_lines), parse_mode="Markdown")

            # Auto-generate Gate 0 idea for the pre-ex-dividend capture strategy
            topic = (
                f"pre-ex-dividend price drift capture for {r['ticker']} {name} "
                f"ex-date {date_fmt} {sen_str} {dtype_str} dividend Bursa Malaysia"
            )
            await update.message.reply_text(
                f"🔬 Auto-generating pre-ex-dividend idea for `{r['ticker']}`...\n"
                f"_(will appear in /ideas shortly)_",
                parse_mode="Markdown",
            )
            try:
                researcher = StrategyResearcher()
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda t=topic: researcher.run(
                        {"action": "generate", "topic": t, "count": 1}
                    ),
                )
                created = result.get('ideas_created', 0)
                idea_lines = [f"✅ Auto-generated `{created}` pre-ex-dividend idea(s) for `{r['ticker']}`."]
                for res in result.get("results", [])[:2]:
                    gate = res.get("gate0", {})
                    icon = "✓" if gate.get("pass") else "✗"
                    idea_lines.append(f"  {icon} [{res['id']}] {res['title'][:55]}")
                await update.message.reply_text("\n".join(idea_lines), parse_mode="Markdown")
            except Exception as e_gen:
                logger.warning(f"Auto-generate idea for {r['ticker']} failed: {e_gen}")
                await update.message.reply_text(
                    f"⚠️ Auto-generate idea failed: {e_gen}"
                )

    except Exception as e:
        logger.exception("cmd_dividend_calendar failed")
        await update.message.reply_text(
            f"❌ Dividend calendar error: {e}\n_Check /status for system health._",
            parse_mode="Markdown",
        )


async def cmd_kb(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update): return
    if not ctx.args:
        await update.message.reply_text(
            "Usage:\n"
            "/kb https\\://some-url.com — ingest URL (Brave fallback on failure)\n"
            "/kb topic words — search Brave for articles on topic, ingest top 3"
        )
        return

    arg = " ".join(ctx.args)
    kb  = KBIngester()

    _CAT_ICON = {
        "irrelevant": "🟥",
        "generic":    "🟨",
        "partial":    "🟧",
        "relevant":   "🟩",
        "direct":     "🟦",
    }
    _CAT_ACTION = {
        "irrelevant": "saved, NOT seeded — wrong market/asset class",
        "generic":    "saved, NOT seeded — transferable concepts only",
        "partial":    "saved, seeded with confidence cap 0.65 — ASEAN/EM context",
        "relevant":   "saved, seeded — Bursa-specific",
        "direct":     "saved, seeded (priority) — actionable KLSE intelligence",
    }

    # ── Non-URL: Brave topic search ──────────────────────────────────────────
    if not arg.startswith("http"):
        from knowledge.ingestion.kb_ingester import BraveSearchFetcher
        from config.settings import BRAVE_SEARCH_API_KEY

        if not BRAVE_SEARCH_API_KEY:
            # No Brave key — fall back to raw text ingest
            await update.message.reply_text(
                "📥 Ingesting text into knowledge base...\n_(~20 seconds)_",
                parse_mode="Markdown"
            )
            try:
                result = kb.ingest_text(arg, title="Telegram input")
            except Exception as e:
                await update.message.reply_text(f"❌ Error: {e}")
                return
            # Fall through to the standard single-result display
        else:
            await update.message.reply_text(
                f"🔍 Searching Brave for: _{arg[:60]}_\n_(finding and ingesting top 3 articles, ~30–45s)_",
                parse_mode="Markdown",
            )
            fetcher     = BraveSearchFetcher()
            brave_items = fetcher.search_and_extract(arg, num_results=3)

            if not brave_items:
                await update.message.reply_text(
                    f"❌ Brave Search found no results for: _{arg[:60]}_\n"
                    f"Try a more specific query or provide a direct URL.",
                    parse_mode="Markdown",
                )
                return

            ingested = []
            for item in brave_items:
                try:
                    res = kb.ingest_text(item["content"], item["title"], source_url=item["url"])
                    ingested.append((item["title"], item["url"], res))
                except Exception as e:
                    logger.warning(f"Brave ingest failed for '{item['title']}': {e}")

            lines = [f"✅ *Brave Search → KB*: _{arg[:50]}_\n"]
            lines.append(f"Found `{len(brave_items)}` results, ingested `{len(ingested)}`:\n")
            for title, url, res in ingested:
                cat  = res.get("relevance_category", "?")
                icon = _CAT_ICON.get(cat, "⬜")
                lines.append(f"{icon} [{res.get('doc_id')}] {title[:55]}")
            await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
            return

    # ── URL ingestion (with automatic Brave fallback) ────────────────────────
    else:
        await update.message.reply_text(
            f"📥 Fetching and ingesting URL...\n`{arg[:80]}`\n_(~20–30 seconds)_",
            parse_mode="Markdown"
        )
        try:
            result = await kb.ingest_url(arg)
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
            return

    # ── Handle Brave fallback notification ───────────────────────────────────
    brave_fallback = result.get("brave_fallback", False)
    if brave_fallback and "error" in result:
        await update.message.reply_text(
            f"❌ Direct fetch failed and Brave Search fallback also returned no results.\n"
            f"URL: `{arg[:80]}`",
            parse_mode="Markdown",
        )
        return

    if "error" in result and not brave_fallback:
        await update.message.reply_text(f"❌ Ingest failed: {result['error']}")
        return

    # Auto-infer domain if ingestion defaulted to "other"
    domain = result.get("domain", "other")
    domain_inferred = False
    if domain == "other":
        try:
            inferred = kb.classify_domain(
                result["doc_id"],
                result["title"],
                result.get("summary", ""),
            )
            if inferred != "other":
                domain = inferred
                domain_inferred = True
        except Exception as e:
            logger.warning(f"Domain inference failed: {e}")

    summary = result.get("summary", "") or ""
    snippet = (summary[:200] + "…") if len(summary) > 200 else summary
    tags    = result.get("tags", [])
    domain_label = f"`{domain}`" + (" _(auto-classified)_" if domain_inferred else "")

    relevance_score    = result.get("relevance_score")
    relevance_category = result.get("relevance_category", "")
    relevance_reason   = result.get("relevance_reason", "")

    if relevance_score is not None:
        icon   = _CAT_ICON.get(relevance_category, "⬜")
        action = _CAT_ACTION.get(relevance_category, "")
        rel_label = (
            f"{icon} `{relevance_score:.2f}` — *{relevance_category}*\n"
            f"_{relevance_reason[:100]}_\n"
            f"↳ _{action}_"
        )
    else:
        rel_label = "`n/a`"

    header = "✅ *KB Ingestion Complete*"
    if brave_fallback:
        header += " _(via Brave Search — direct fetch failed)_"
    lines = [
        f"{header}\n",
        f"Doc ID:     `{result['doc_id']}`",
        f"Title:      *{result['title'][:60]}*",
        f"Domain:     {domain_label}",
        f"Relevance:  {rel_label}",
        f"Concepts:   `{result['concepts_extracted']}`",
        f"Tags:       `{', '.join(tags[:6]) or 'none'}`",
    ]
    if snippet:
        lines.append(f"\n_{snippet}_")

    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def cmd_search(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update): return
    if not ctx.args:
        await update.message.reply_text("Usage: /search <query>")
        return
    query = " ".join(ctx.args)
    kb   = KBIngester()
    docs = kb.search(query, limit=5)
    if not docs:
        await update.message.reply_text(f"No results found for: {query}")
        return
    lines = [f"🔍 *Search: {query}*\n"]
    for d in docs:
        lines.append(f"• [{d['id']}] *{d['title'][:50]}* `{d['domain']}`")
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def cmd_vault(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Export the knowledge graph to the Obsidian vault on demand."""
    if not is_allowed(update): return
    await update.message.reply_text("📚 Exporting knowledge graph to Obsidian vault...")
    try:
        from scripts.export_obsidian import export_vault
        import asyncio as _aio
        result = await _aio.get_event_loop().run_in_executor(None, export_vault)
        await update.message.reply_text(
            f"✅ Vault exported: {result['notes']} notes, {result['edges']} links\n"
            f"`{result['path']}`", parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ Vault export failed: {e}")


async def cmd_research(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update): return
    if not ctx.args:
        await update.message.reply_text("Usage: /research <idea_id>")
        return
    try:
        idea_id = int(ctx.args[0])
    except ValueError:
        await update.message.reply_text("idea_id must be a number")
        return

    with db_session() as conn:
        row = conn.execute(
            "SELECT id, title, hypothesis FROM alpha_ideas WHERE id=?", (idea_id,)
        ).fetchone()
    if not row:
        await update.message.reply_text(f"Idea {idea_id} not found.")
        return

    await update.message.reply_text(
        f"🔬 Hunting papers for idea [{idea_id}]: _{row['title'][:50]}_\n_(~30s)_",
        parse_mode="Markdown"
    )
    try:
        hunter = ResearchHunter()
        result = hunter.hunt(row["title"], row["hypothesis"] or "")
        lines  = [
            f"📚 *Research Hunt — Idea [{idea_id}]*\n",
            f"Papers found:    `{result['papers_found']}`",
            f"Papers ingested: `{result['papers_ingested']}`\n",
        ]
        for title in result.get("titles", [])[:8]:
            lines.append(f"• {title[:60]}")
        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")


async def cmd_briefing(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update): return
    await update.message.reply_text(
        "🌅 Generating morning briefing...\n_(fetches pipeline status, i3investor, dividends, ~30s)_",
        parse_mode="Markdown"
    )
    try:
        result = MorningBriefing().generate_briefing()
        if result.get("sent"):
            await update.message.reply_text(
                f"✅ Briefing sent!\n"
                f"Articles: `{result['articles']}`  Dividends: `{result['dividends']}`\n"
                f"Research focus: `{result['research_angle'] or 'n/a'}`",
                parse_mode="Markdown"
            )
        else:
            await update.message.reply_text(
                "⚠️ Briefing generated but Telegram send failed — check bot token/chat ID."
            )
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")


async def cmd_dividends(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update): return
    await update.message.reply_text(
        "📅 Fetching upcoming ex-dividend dates (next 14 days)...",
        parse_mode="Markdown"
    )
    try:
        scanner = FundamentalScanner()
        divs    = scanner.scan_dividend_calendar(days_ahead=14)
        if not divs:
            await update.message.reply_text("No ex-dividend dates found in the next 14 days.")
            return
        lines = [f"📅 *Upcoming Ex-Dividend Dates (next 14 days)*\n"]
        for d in divs[:10]:
            amt = f" — {d['dividend_amount']} MYR" if d.get("dividend_amount") else ""
            yld = f" ({d['current_yield_pct']:.1f}%)" if d.get("current_yield_pct") else ""
            lines.append(f"`{d['symbol']}` *{d['name'][:18]}* | {d['ex_date']}{amt}{yld}")
        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")


async def cmd_digest(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update): return
    if not ctx.args:
        await update.message.reply_text(
            "Usage:\n/digest <doc_id> — digest a specific document\n/digest all — process all undigested (limit 20)"
        )
        return

    arg = ctx.args[0].lower()
    seeder = AlphaSeedGenerator()

    if arg == "all":
        await update.message.reply_text(
            "🌱 Processing all undigested KB documents (limit 20)...\n_(may take a few minutes)_",
            parse_mode="Markdown"
        )
        try:
            result = seeder.process_undigested(limit=20)
            await update.message.reply_text(
                f"✅ *Alpha Seed Batch Complete*\n\n"
                f"Documents processed: `{result['processed']}`\n"
                f"Documents skipped:   `{result['skipped']}`\n"
                f"Ideas created:       `{result['total_ideas_created']}`",
                parse_mode="Markdown"
            )
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
        return

    try:
        doc_id = int(arg)
    except ValueError:
        await update.message.reply_text("doc_id must be a number or 'all'")
        return

    # Look up title for display
    with db_session() as conn:
        doc = conn.execute(
            "SELECT id, title FROM kb_documents WHERE id=?", (doc_id,)
        ).fetchone()
    if not doc:
        await update.message.reply_text(f"Document {doc_id} not found.")
        return

    await update.message.reply_text(
        f"🌱 Digesting doc [{doc_id}]: _{doc['title'][:60]}_\n_(~20–30 seconds)_",
        parse_mode="Markdown"
    )
    try:
        result = seeder.digest(doc_id)
        if result.get("skipped"):
            reason = result.get("reason", "unknown")
            await update.message.reply_text(
                f"⚠️ Skipped doc [{doc_id}] — reason: `{reason}`",
                parse_mode="Markdown"
            )
            return

        # Fetch the newly created ideas for display
        today = datetime.utcnow().strftime("%Y-%m-%d")
        with db_session() as conn:
            ideas = conn.execute(
                "SELECT id, title, ticker FROM alpha_ideas "
                "WHERE slug LIKE 'seed-%' AND created_at LIKE ? ORDER BY id DESC LIMIT 10",
                (f"{today}%",),
            ).fetchall()

        lines = [
            f"🌱 *Alpha Seeds from:* _{result['title'][:50]}_\n",
            f"Core insight: _{result['core_insight'][:150]}_\n",
            f"Hypotheses generated: `{result['hypotheses_generated']}`",
            f"Ideas created: `{result['ideas_saved']}`",
        ]
        if ideas:
            lines.append("\n*Ideas:*")
            for idea in ideas[:result['ideas_saved']]:
                ticker = idea['ticker'] or '—'
                lines.append(f"• [{idea['id']}] {idea['title'][:50]} (`{ticker}`)")

        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")


async def cmd_arsenal(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update): return
    try:
        from knowledge.ingestion.technique_library import TechniqueLibrary
        lib = TechniqueLibrary()
        key = " ".join(ctx.args).strip().lower().replace(" ", "_") if ctx.args else None
        text = lib.format_telegram_summary(key)
        # Strip Markdown markers so plain text renders cleanly
        for ch in ("*", "_", "`"):
            text = text.replace(ch, "")
        # Send as plain text in 4000-char chunks (avoids all Markdown parse errors)
        for i in range(0, len(text), 4000):
            await update.message.reply_text(text[i:i+4000])
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")


async def handle_pdf_document(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update): return
    doc = update.message.document
    if not doc or doc.mime_type != "application/pdf":
        await update.message.reply_text("Only PDF files are supported for KB ingestion.")
        return

    filename = doc.file_name or "upload.pdf"
    await update.message.reply_text(
        f"📄 Downloading `{filename}`…\n_(extracting text and ingesting, ~20–30 seconds)_",
        parse_mode="Markdown",
    )

    try:
        import io
        import pdfplumber

        tg_file = await ctx.bot.get_file(doc.file_id)
        buf = io.BytesIO()
        await tg_file.download_to_memory(buf)
        buf.seek(0)

        text_parts = []
        title = ""
        with pdfplumber.open(buf) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""
                text_parts.append(page_text)
                if i == 0 and not title:
                    for line in page_text.splitlines():
                        line = line.strip()
                        if len(line) > 5:
                            title = line[:120]
                            break

        full_text = "\n\n".join(text_parts)[:50000]
        if not full_text.strip():
            await update.message.reply_text("❌ Could not extract any text from the PDF.")
            return

        if not title:
            title = filename.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").title()

        kb = KBIngester()
        result = kb.ingest_text(content=full_text, title=title, source_url=f"pdf_upload:{filename}")

    except Exception as e:
        await update.message.reply_text(f"❌ Error processing PDF: {e}")
        return

    if "error" in result:
        await update.message.reply_text(f"❌ Ingest failed: {result['error']}")
        return

    # Auto-classify domain if needed
    domain = result.get("domain", "other")
    domain_inferred = False
    if domain == "other":
        try:
            inferred = kb.classify_domain(result["doc_id"], result["title"], result.get("summary", ""))
            if inferred != "other":
                domain = inferred
                domain_inferred = True
        except Exception as e:
            logger.warning(f"Domain inference failed: {e}")

    summary = result.get("summary", "") or ""
    snippet = (summary[:200] + "…") if len(summary) > 200 else summary
    tags = result.get("tags", [])
    domain_label = f"`{domain}`" + (" _(auto-classified)_" if domain_inferred else "")

    relevance_score    = result.get("relevance_score")
    relevance_category = result.get("relevance_category", "")
    relevance_reason   = result.get("relevance_reason", "")

    _CAT_ICON = {"irrelevant":"🟥","generic":"🟨","partial":"🟧","relevant":"🟩","direct":"🟦"}
    _CAT_ACTION = {
        "irrelevant": "saved, NOT seeded — wrong market/asset class",
        "generic":    "saved, NOT seeded — transferable concepts only",
        "partial":    "saved, seeded with confidence cap 0.65 — ASEAN/EM context",
        "relevant":   "saved, seeded — Bursa-specific",
        "direct":     "saved, seeded (priority) — actionable KLSE intelligence",
    }

    if relevance_score is not None:
        icon   = _CAT_ICON.get(relevance_category, "⬜")
        action = _CAT_ACTION.get(relevance_category, "")
        rel_label = (
            f"{icon} `{relevance_score:.2f}` — *{relevance_category}*\n"
            f"_{relevance_reason[:100]}_\n"
            f"↳ _{action}_"
        )
    else:
        rel_label = "`n/a`"

    lines = [
        "✅ *PDF Ingested into KB*\n",
        f"File:       `{filename}`",
        f"Doc ID:     `{result['doc_id']}`",
        f"Title:      *{result['title'][:60]}*",
        f"Domain:     {domain_label}",
        f"Relevance:  {rel_label}",
        f"Concepts:   `{result['concepts_extracted']}`",
        f"Tags:       `{', '.join(tags[:6]) or 'none'}`",
    ]
    if snippet:
        lines.append(f"\n_{snippet}_")

    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def cmd_analyst(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Analyst coverage initiation monitor — last 7 days of coverage events."""
    if not is_allowed(update): return
    await update.message.reply_text("🔬 Fetching analyst coverage events... (may take 30s)")
    try:
        from data.analyst.coverage_monitor import AnalystCoverageMonitor
        mon    = AnalystCoverageMonitor()
        # Refresh data then build report
        mon.fetch_new_reports(days_back=1)
        events = mon.recent_events(days=7)

        lines = [
            "🔬 *Analyst Coverage Monitor* (last 7 days)",
            f"_{events['cutoff_date']} → now · {events['total']} records_\n",
        ]

        if events["first_coverage"]:
            lines.append("🆕 *FIRST COVERAGE:*")
            for e in events["first_coverage"][:6]:
                tp_str = f" TP:RM{e['target_price']:.2f}" if e.get("target_price") else ""
                lines.append(
                    f"  • *{e.get('company', e['ticker'])}* `{e['ticker']}` — "
                    f"{e['analyst_house']}{tp_str} _{e['date']}_"
                )
            lines.append("")

        if events["upgrades"]:
            lines.append("📈 *UPGRADES:*")
            for e in events["upgrades"][:5]:
                tp_str = f" TP:RM{e['target_price']:.2f}" if e.get("target_price") else ""
                lines.append(
                    f"  • *{e.get('company', e['ticker'])}* — "
                    f"{e['analyst_house']}{tp_str}"
                )
            lines.append("")

        if events["downgrades"]:
            lines.append("📉 *DOWNGRADES:*")
            for e in events["downgrades"][:5]:
                lines.append(
                    f"  • *{e.get('company', e['ticker'])}* — {e['analyst_house']}"
                )
            lines.append("")

        if events["maintains"]:
            maint_names = ", ".join(
                f"`{e['ticker']}`" for e in events["maintains"][:6]
            )
            lines.append(f"→ *Maintained:* {maint_names}")
            lines.append("")

        if events["total"] == 0:
            lines.append(
                "_No analyst coverage events in DB yet. Data accumulates over time "
                "as the monitor scrapes Brave Search and i3investor._\n"
                "_Run /analyst again in a few hours, or use /kb to manually ingest "
                "a Bursa analyst report URL._"
            )

        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
    except Exception as e:
        logger.exception("cmd_analyst failed")
        await update.message.reply_text(f"❌ Analyst monitor error: {e}")


async def cmd_cpo(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """CPO lag signal report for plantation stocks."""
    if not is_allowed(update): return
    await update.message.reply_text("🌴 Running CPO lag analysis... (may take 30s)")
    try:
        from data.cpo.cpo_signal import CPOSignalGenerator
        gen     = CPOSignalGenerator()
        signals = gen.daily_scan()

        from data.cpo.mpob_scraper import MPOBScraper
        scraper = MPOBScraper()
        try:
            price_info = scraper.fetch_daily_cpo_price()
            cpo_line   = f"CPO spot: *{price_info['price_myr_per_tonne']:.2f} MYR/t* ({price_info['date']})"
        except Exception:
            cpo_line = "_CPO price unavailable_"

        lines = ["🌴 *CPO Plantation Lag Signal Report*\n", cpo_line, ""]

        for sig in signals:
            ticker    = sig.get("ticker", "")
            company   = sig.get("company", ticker)
            lag       = sig.get("best_lag_days", 0)
            corr      = sig.get("best_lag_corr", 0.0)
            direction = sig.get("predicted_direction", "neutral")
            sig_flag  = "✅" if sig.get("is_significant") else "·"

            if sig.get("error"):
                lines.append(f"{sig_flag} `{ticker}` _{sig['error'][:40]}_")
                continue

            dir_icon = {"up": "📈", "down": "📉", "neutral": "→"}.get(direction, "→")
            lines.append(
                f"{sig_flag} *{company}* `{ticker}` — lag {lag}d corr {corr:+.3f} "
                f"{dir_icon} {direction}"
            )

        sig_count = sum(1 for s in signals if s.get("is_significant"))
        lines.append(f"\n_{len(signals)} tickers · {sig_count} significant signals_")

        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
    except Exception as e:
        logger.exception("cmd_cpo failed")
        await update.message.reply_text(f"❌ CPO signal error: {e}")


async def cmd_diversity(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update): return
    try:
        de      = DiversityEngine()
        balance = de.check_balance()
        coverage = balance["coverage"]
        least   = balance["least_covered"]

        lines = ["📊 *KB Diversity Balance*\n"]
        for angle, count in sorted(coverage.items(), key=lambda x: x[1]):
            marker = " ← needs research" if angle == least else ""
            lines.append(f"`{angle:<20}` {count:>3} docs{marker}")
        lines.append(f"\nTotal docs tracked: `{balance['total_docs']}`")
        lines.append(f"Most under-researched: `{least}`")
        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")


async def cmd_epf(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """EPF Accumulation Tracker — shows EPF ownership movements across KLCI."""
    if not is_allowed(update): return
    await update.message.reply_text("🏛️ Fetching EPF ownership data... (may take 30s)")
    try:
        from data.epf.epf_signal import EPFSignalGenerator
        gen    = EPFSignalGenerator()
        report = gen.weekly_report()

        acc  = report["accumulating"]
        dist = report["distributing"]
        stbl = report["stable"]
        nd   = report["no_data"]

        lines = [
            f"🏛️ *EPF Accumulation Tracker*",
            f"_{report['generated_at']} · {report['disclosures_fetched']} disclosures fetched_\n",
        ]

        if acc:
            acc_names = []
            for e in acc[:8]:
                strength_icon = "🔥" if e["signal_strength"] == "strong" else "📈"
                pct_str = f"{e['current_pct']:.1f}%" if e["current_pct"] else ""
                chg_str = f"{e['total_change_4q']:+.2f}%" if e["total_change_4q"] else ""
                acc_names.append(
                    f"  {strength_icon} *{e['company']}* `{e['ticker']}`"
                    + (f" {pct_str} ({chg_str})" if pct_str else "")
                )
            lines.append(f"▲ *Accumulating* ({len(acc)} stocks):")
            lines.extend(acc_names)
            lines.append("")

        if dist:
            dist_names = []
            for e in dist[:6]:
                pct_str = f"{e['current_pct']:.1f}%" if e["current_pct"] else ""
                chg_str = f"{e['total_change_4q']:+.2f}%" if e["total_change_4q"] else ""
                dist_names.append(
                    f"  📉 *{e['company']}* `{e['ticker']}`"
                    + (f" {pct_str} ({chg_str})" if pct_str else "")
                )
            lines.append(f"▼ *Distributing* ({len(dist)} stocks):")
            lines.extend(dist_names)
            lines.append("")

        if stbl:
            stbl_names = ", ".join(f"`{e['ticker']}`" for e in stbl[:8])
            lines.append(f"→ *Stable:* {stbl_names}")
            if len(stbl) > 8:
                lines.append(f"  _(+{len(stbl)-8} more)_")
            lines.append("")

        if nd:
            lines.append(
                f"_No EPF disclosure data for {len(nd)} stocks — "
                f"run /epf again to refresh from Bursa_"
            )

        if not acc and not dist and not stbl:
            lines.append(
                "_No EPF data in database yet. The tracker scrapes Bursa disclosures "
                "via Brave Search — data accumulates over time._\n"
                "_Try /kb with a Bursa EPF announcement URL to seed data manually._"
            )

        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
    except Exception as e:
        logger.exception("cmd_epf failed")
        await update.message.reply_text(f"❌ EPF tracker error: {e}")

async def cmd_events(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Show last 10 events from market_events."""
    if not is_allowed(update): return
    today = datetime.utcnow().strftime("%Y-%m-%d")
    with db_session() as conn:
        rows = conn.execute("""
            SELECT event_type, ticker, company, headline, action_taken,
                   confidence, detected_at
            FROM market_events
            WHERE detected_at >= datetime('now','-24 hours')
            ORDER BY detected_at DESC LIMIT 10
        """).fetchall()

        stats = conn.execute("""
            SELECT
              COUNT(*) as total,
              SUM(CASE WHEN action_taken='gate0_idea' THEN 1 ELSE 0 END) as ideas,
              SUM(CASE WHEN action_taken='alert' THEN 1 ELSE 0 END) as alerts,
              SUM(CASE WHEN action_taken='kb_only' THEN 1 ELSE 0 END) as kb
            FROM market_events WHERE detected_at LIKE ?
        """, (f"{today}%",)).fetchone()

    if not rows:
        await update.message.reply_text(
            "No events in the last 24 hours.\n"
            "Is the EventWatcher running? Check /event_stats"
        )
        return

    lines = [f"*Event Feed* (last 24h)\n"]
    ACTION_ICONS = {
        "gate0_idea": "🟢", "alert": "🟡", "kb_only": "⚫", "ignore": "⬛"
    }
    for r in rows:
        icon = ACTION_ICONS.get(r["action_taken"], "⚪")
        ticker_str = f"`{r['ticker']}`" if r["ticker"] else ""
        company_str = r["company"] or ""
        ts = (r["detected_at"] or "")[:16].replace("T", " ")
        etype = (r["event_type"] or "").upper().replace("_", " ")
        conf = f"{r['confidence']*100:.0f}%" if r["confidence"] else ""
        headline = (r["headline"] or "")[:60]
        lines.append(f"{icon} {ticker_str} {company_str} `{etype}` {conf}")
        lines.append(f"  {headline}")
        lines.append(f"  _{ts}_")
        lines.append("")

    total = stats["total"] or 0
    ideas = stats["ideas"] or 0
    alerts = stats["alerts"] or 0
    kb = stats["kb"] or 0
    lines.append(f"Total: {total} | Ideas: {ideas} | Alerts: {alerts} | KB: {kb}")
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def cmd_calendar(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Show upcoming economic calendar events."""
    if not is_allowed(update): return
    today = datetime.utcnow().strftime("%Y-%m-%d")
    cutoff = (datetime.utcnow() + __import__('datetime').timedelta(days=30)).strftime("%Y-%m-%d")

    with db_session() as conn:
        rows = conn.execute("""
            SELECT event_name, event_type, scheduled_date, scheduled_time,
                   country, importance, forecast_value, previous_value
            FROM economic_calendar
            WHERE scheduled_date >= ? AND scheduled_date <= ?
            ORDER BY scheduled_date, scheduled_time
        """, (today, cutoff)).fetchall()

    if not rows:
        await update.message.reply_text(
            "No upcoming economic events in the calendar.\n"
            "Run: PYTHONPATH=/opt/openclaw/app /opt/openclaw/venv/bin/python scripts/seed_economic_calendar.py"
        )
        return

    high = [r for r in rows if r["importance"] == "high"]
    med  = [r for r in rows if r["importance"] != "high"]

    lines = ["*Economic Calendar* (next 30 days)\n"]
    if high:
        lines.append("*HIGH IMPORTANCE:*")
        for r in high[:8]:
            dt = r["scheduled_date"]
            time_str = r["scheduled_time"] or ""
            country = r["country"] or "??"
            time_label = f" {time_str} MYT" if time_str else ""
            fcast = f" (fcst: {r['forecast_value']})" if r["forecast_value"] else ""
            lines.append(f"  `{dt}` — {r['event_name']} ({country}){time_label}{fcast}")
        lines.append("")
    if med:
        lines.append("*MEDIUM:*")
        for r in med[:5]:
            lines.append(f"  `{r['scheduled_date']}` — {r['event_name']} ({r['country'] or '??'})")

    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def cmd_event_stats(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Show event engine health status."""
    if not is_allowed(update): return
    today = datetime.utcnow().strftime("%Y-%m-%d")

    with db_session() as conn:
        stats = conn.execute("""
            SELECT
              COUNT(*) as total,
              SUM(CASE WHEN action_taken='gate0_idea' THEN 1 ELSE 0 END) as ideas,
              SUM(CASE WHEN action_taken='alert'      THEN 1 ELSE 0 END) as alerts,
              SUM(CASE WHEN action_taken='kb_only'    THEN 1 ELSE 0 END) as kb
            FROM market_events WHERE detected_at LIKE ?
        """, (f"{today}%",)).fetchone()

        last_log = conn.execute("""
            SELECT message, created_at FROM daemon_logs
            WHERE source='EventWatcher' ORDER BY id DESC LIMIT 1
        """).fetchone()

        # Cycle count from log messages
        cycle_row = conn.execute("""
            SELECT message FROM daemon_logs
            WHERE source='EventWatcher' AND message LIKE 'Cycle %'
            ORDER BY id DESC LIMIT 1
        """).fetchone()

        by_source = conn.execute("""
            SELECT source, COUNT(*) as n FROM market_events
            WHERE detected_at LIKE ? GROUP BY source ORDER BY n DESC
        """, (f"{today}%",)).fetchall()

    watcher_status = "STOPPED"
    last_cycle_str = "never"
    cycle_num = "?"

    if last_log:
        last_cycle_str = (last_log["created_at"] or "")[:16]
        # Check if last log was recent (within 10 minutes)
        try:
            from datetime import datetime as _dt
            last_dt = _dt.strptime(last_cycle_str, "%Y-%m-%d %H:%M")
            secs_ago = (_dt.utcnow() - last_dt).total_seconds()
            watcher_status = "RUNNING" if secs_ago < 600 else "POSSIBLY STALLED"
        except Exception:
            pass

    if cycle_row:
        import re
        m = re.search(r"Cycle (\d+)", cycle_row["message"])
        if m:
            cycle_num = m.group(1)

    total  = stats["total"]  or 0
    ideas  = stats["ideas"]  or 0
    alerts = stats["alerts"] or 0
    kb     = stats["kb"]     or 0

    source_str = " ".join(f"`{r['source']}({r['n']})`" for r in by_source[:5])

    lines = [
        f"*Event Engine Status*",
        f"",
        f"Watcher: `{watcher_status}` (cycle #{cycle_num})",
        f"Last cycle: `{last_cycle_str}`",
        f"",
        f"Today ({today}):",
        f"  `{total}` events processed",
        f"  `{ideas}` gate0 ideas created",
        f"  `{alerts}` alerts sent",
        f"  `{kb}` KB documents added",
        f"",
        f"Sources: {source_str or 'none yet'}",
    ]
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


def main():
    if not BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN not set — bot cannot start")
        sys.exit(1)
    init_db()
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start",     cmd_start))
    app.add_handler(CommandHandler("status",    cmd_status))
    app.add_handler(CommandHandler("ideas",     cmd_ideas))
    app.add_handler(CommandHandler("spend",     cmd_spend))
    app.add_handler(CommandHandler("generate",  cmd_generate))
    app.add_handler(CommandHandler("screen",             cmd_screen))
    app.add_handler(CommandHandler("fundamentals",       cmd_fundamentals))
    app.add_handler(CommandHandler("dividend_calendar",  cmd_dividend_calendar))
    app.add_handler(CommandHandler("briefing",           cmd_briefing))
    app.add_handler(CommandHandler("dividends",          cmd_dividends))
    app.add_handler(CommandHandler("kb",        cmd_kb))
    app.add_handler(CommandHandler("search",    cmd_search))
    app.add_handler(CommandHandler("vault",     cmd_vault))
    app.add_handler(CommandHandler("research",  cmd_research))
    app.add_handler(CommandHandler("arsenal",   cmd_arsenal))
    app.add_handler(CommandHandler("diversity", cmd_diversity))
    app.add_handler(CommandHandler("digest",    cmd_digest))
    app.add_handler(CommandHandler("epf",         cmd_epf))
    app.add_handler(CommandHandler("analyst",     cmd_analyst))
    app.add_handler(CommandHandler("cpo",         cmd_cpo))
    app.add_handler(CommandHandler("events",      cmd_events))
    app.add_handler(CommandHandler("calendar",    cmd_calendar))
    app.add_handler(CommandHandler("event_stats", cmd_event_stats))
    app.add_handler(MessageHandler(filters.Document.PDF, handle_pdf_document))

    # Register bot command menu (shows up in Telegram UI autocomplete)
    async def _set_commands(application):
        await application.bot.set_my_commands([
            BotCommand("start",     "Help & command list"),
            BotCommand("status",    "Pipeline health report"),
            BotCommand("ideas",     "Last 8 active alpha ideas"),
            BotCommand("spend",     "AI cost breakdown today"),
            BotCommand("generate",  "Generate KLCI equity ideas"),
            BotCommand("screen",             "8-screen KLSE fundamental scan"),
            BotCommand("fundamentals",       "Full fundamentals for a ticker"),
            BotCommand("dividend_calendar",  "Upcoming ex-dividend dates"),
            BotCommand("briefing",  "Send morning briefing now"),
            BotCommand("dividends", "Ex-dividend dates next 14 days"),
            BotCommand("epf",       "EPF ownership tracker"),
            BotCommand("analyst",   "Analyst coverage initiations"),
            BotCommand("cpo",       "CPO plantation lag signals"),
            BotCommand("kb",        "Ingest URL or text into KB"),
            BotCommand("search",    "Search knowledge base"),
            BotCommand("diversity", "KB coverage by research angle"),
            BotCommand("research",  "Hunt academic papers for an idea"),
            BotCommand("digest",    "Generate ideas from KB documents"),
            BotCommand("arsenal",      "Quantitative techniques list"),
            BotCommand("events",       "Event feed (last 24h)"),
            BotCommand("calendar",     "Upcoming macro events"),
            BotCommand("event_stats",  "Event engine health"),
        ])
        logger.info("Telegram bot command menu registered")

    app.post_init = _set_commands
    logger.info(f"Telegram bot starting (admin={ADMIN_CHAT})")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s", datefmt="%H:%M:%S")
    main()
