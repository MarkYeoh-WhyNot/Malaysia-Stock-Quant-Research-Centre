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
        "/search `<query>` — full-text search across KB documents\n"
        "/diversity — KB coverage by research angle\n\n"
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
        ticker = i['pair'] or '—'
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
    app.add_handler(CommandHandler("search",    cmd_search))
    app.add_handler(CommandHandler("research",  cmd_research))
    app.add_handler(CommandHandler("diversity", cmd_diversity))
    logger.info(f"Telegram bot starting (admin={ADMIN_CHAT})")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s", datefmt="%H:%M:%S")
    main()
