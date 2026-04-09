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
        "🐋 *OpenClaw — Bursa Malaysia KLSE*\n"
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
    stages  = health.get("stages", {})
    lines = [
        f"🐋 *Pipeline Status* — {datetime.utcnow().strftime('%H:%M UTC')}",
        f"",
        f"Health: `{health['health'].upper()}`",
        f"Total ideas: `{health['total_ideas']}`",
        f"Today's spend: `${health['daily_spend']:.4f}`",
        f"Errors (1h): `{health['errors_1h']}`",
    ]
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

async def cmd_ideas(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update): return
    with db_session() as conn:
        ideas = conn.execute("""
            SELECT id, title, pair, stage, novelty_score, logic_score, backtest_sharpe
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
    """Show upcoming ex-dividend dates from dividend_history table."""
    if not is_allowed(update): return
    from datetime import date as _date
    today = _date.today().isoformat()
    with db_session() as conn:
        rows = conn.execute(
            """
            SELECT dh.ticker, fd.name, dh.ex_date, dh.payment_date,
                   dh.dps_sen, dh.subject, dh.dividend_type
            FROM dividend_history dh
            LEFT JOIN (
                SELECT ticker, name,
                       ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY fetched_at DESC) rn
                FROM fundamental_data
            ) fd ON fd.ticker = dh.ticker AND fd.rn = 1
            WHERE dh.ex_date >= ?
            ORDER BY dh.ex_date ASC
            LIMIT 15
            """,
            (today,),
        ).fetchall()

    if not rows:
        await update.message.reply_text(
            "No upcoming ex-dividend dates found in dividend_history.\n"
            "Run /screen or wait for the 18:00 MYT refresh to populate data.",
        )
        return

    lines = ["📅 *Upcoming Ex-Dividend Dates*\n"]
    for r in rows:
        name_str = (r["name"] or r["ticker"])[:20]
        sen_str = f" — {r['dps_sen']:.2f} sen" if r.get("dps_sen") else ""
        lines.append(
            f"`{r['ex_date']}` *{name_str}* `{r['ticker']}`{sen_str}"
        )
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


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
        # Telegram message limit is 4096 chars; split if needed
        if len(text) <= 4096:
            await update.message.reply_text(text, parse_mode="Markdown")
        else:
            for i in range(0, len(text), 4000):
                await update.message.reply_text(text[i:i+4000], parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")


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
    app.add_handler(CommandHandler("research",  cmd_research))
    app.add_handler(CommandHandler("arsenal",   cmd_arsenal))
    app.add_handler(CommandHandler("diversity", cmd_diversity))
    app.add_handler(CommandHandler("digest",    cmd_digest))
    app.add_handler(CommandHandler("epf",       cmd_epf))
    app.add_handler(CommandHandler("analyst",   cmd_analyst))
    app.add_handler(CommandHandler("cpo",       cmd_cpo))
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
            BotCommand("arsenal",   "Quantitative techniques list"),
        ])
        logger.info("Telegram bot command menu registered")

    app.post_init = _set_commands
    logger.info(f"Telegram bot starting (admin={ADMIN_CHAT})")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s", datefmt="%H:%M:%S")
    main()
