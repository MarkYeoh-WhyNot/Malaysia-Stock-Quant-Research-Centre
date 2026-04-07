import asyncio, logging, os, sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

from telegram import Update
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
        "/dividends — ex-dividend dates in next 14 days\n\n"
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
        "📡 Screening KLSE live data and generating ideas...\n_(fetches klsescreener.com + i3investor, ~45s)_",
        parse_mode="Markdown"
    )
    try:
        researcher = StrategyResearcher()
        result     = researcher.run({"action": "screen_generate", "count": 3})
        lines      = [f"✅ Created {result['ideas_created']} KLSE ideas from live screen:\n"]
        for r in result.get("results", []):
            gate = r.get("gate0", {})
            icon = "✓" if gate.get("pass") else "✗"
            lines.append(f"{icon} [{r['id']}] {r['title'][:50]}")

        # Also check i3investor for recent analyst coverage
        try:
            scraper  = I3investorScraper()
            articles = scraper.get_research_articles(max_articles=5)
            if articles:
                lines.append(f"\n📰 *Recent i3investor Analyst Coverage*")
                for a in articles[:3]:
                    broker = f" [{a['brokerage']}]" if a.get("brokerage") else ""
                    tickers = ", ".join(a.get("tickers", [])[:2])
                    t_str   = f" `{tickers}`" if tickers else ""
                    lines.append(f"  • {a['title'][:50]}{broker}{t_str}")
        except Exception as e2:
            logger.warning(f"i3investor coverage check failed: {e2}")

        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")


async def cmd_kb(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update): return
    if not ctx.args:
        await update.message.reply_text(
            "Usage:\n/kb https\\://some-url.com\n/kb Some text to ingest into knowledge base"
        )
        return

    arg = " ".join(ctx.args)
    kb  = KBIngester()

    if arg.startswith("http"):
        await update.message.reply_text(
            f"📥 Fetching and ingesting URL...\n`{arg[:80]}`\n_(~20–30 seconds)_",
            parse_mode="Markdown"
        )
        try:
            result = await kb.ingest_url(arg)
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
            return
    else:
        await update.message.reply_text(
            "📥 Ingesting text into knowledge base...\n_(~20 seconds)_",
            parse_mode="Markdown"
        )
        try:
            result = kb.ingest_text(arg, title="Telegram input")
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
            return

    if "error" in result:
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

    # 5-tier colour coding
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
        "✅ *KB Ingestion Complete*\n",
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
    app.add_handler(CommandHandler("screen",    cmd_screen))
    app.add_handler(CommandHandler("briefing",  cmd_briefing))
    app.add_handler(CommandHandler("dividends", cmd_dividends))
    app.add_handler(CommandHandler("kb",        cmd_kb))
    app.add_handler(CommandHandler("search",    cmd_search))
    app.add_handler(CommandHandler("research",  cmd_research))
    app.add_handler(CommandHandler("arsenal",   cmd_arsenal))
    app.add_handler(CommandHandler("diversity", cmd_diversity))
    app.add_handler(CommandHandler("digest",    cmd_digest))
    logger.info(f"Telegram bot starting (admin={ADMIN_CHAT})")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s", datefmt="%H:%M:%S")
    main()
