# OpenClaw — Project Briefing

## Purpose

OpenClaw is a Claude-powered quantitative equity research and backtesting pipeline for **Bursa Malaysia (KLSE/FBM KLCI)** equities. It automatically generates alpha ideas, screens them through quality gates, backtests them, and paper-trades survivors — all driven by Claude AI agents with Telegram and REST API interfaces.

**Key constraints:**
- Bursa Malaysia equities ONLY (strict guards prevent FX contamination)
- Long-only strategies (short-selling restricted on most KLSE stocks)
- Daily AI budget cap: $50 (configurable via `AI_DAILY_BUDGET_USD` in `.env`)
- Paper trading only for now (no live Bursa broker integration yet)

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.12.3 |
| AI | Anthropic Claude (Haiku 4.5, Sonnet 4.6, Opus 4.6) via `anthropic>=0.40` |
| API server | FastAPI + uvicorn (port 8001) |
| Data | yfinance, pandas/numpy/pyarrow, beautifulsoup4 |
| Database | SQLite (`data/openclaw.db`) |
| Cache | Parquet files in `data/cache/` (12-hour staleness) |
| Telegram | python-telegram-bot[job-queue] |
| Scheduling | APScheduler |
| Process mgmt | Supervisor (native) or Docker Compose |
| Config | python-dotenv (`.env` file) |

---

## File Structure

```
/opt/openclaw/app/
├── agents/
│   ├── base_agent.py               # Base class: Claude API, cost tracking, SQLite logging
│   ├── researcher/
│   │   ├── strategy_researcher.py  # Gate 0 novelty/logic screen + Stage 1 deep research
│   │   └── red_blue_team.py        # Adversarial review of strategies
│   ├── data_engineer/
│   │   └── data_engineer.py        # Data fetch, 50+ feature engineering, cache management
│   ├── backtest_engineer/
│   │   └── backtest_engineer.py    # Gates 2-3: vectorized NumPy backtesting, K-fold
│   ├── portfolio_executor/
│   │   └── portfolio_executor.py   # Paper trading, position sizing, exit management
│   └── risk_monitor/
│       └── risk_monitor.py         # Health checks, drawdown monitoring, pipeline status
├── config/
│   ├── settings.py                 # KLCI universe (30 stocks), gate thresholds, model selection
│   └── __init__.py
├── data/
│   ├── database.py                 # SQLite schema + session management (11 tables)
│   ├── yahoo/client.py             # yfinance wrapper (OHLCV, fundamentals, bulk fetch)
│   ├── klse/screener.py            # Web scraper (klsescreener.com → i3investor → hardcoded)
│   ├── oanda/client.py             # Legacy FX broker client (not used for KLSE)
│   ├── cache/                      # Parquet cache files (auto-created)
│   └── openclaw.db                 # SQLite database (auto-created)
├── knowledge/
│   ├── ingestion/kb_ingester.py    # Document/URL ingestion, concept extraction
│   ├── search/                     # Search stub
│   └── graph/                      # Knowledge graph stub
├── dashboard/
│   ├── api/server.py               # FastAPI: /api/health, /api/mission-control, /api/analytics, etc.
│   └── ui/                         # Frontend stub
├── data/
│   ├── i3investor/
│   │   └── scraper.py              # I3investorScraper: research articles, news, dividends, forums
│   └── klse/
│       ├── screener.py             # KLSEScreener: KLCI constituents, fundamental screen
│       └── fundamental_scanner.py  # FundamentalScanner: value, momentum, dividend/earnings calendar
├── scripts/
│   ├── research_daemon.py          # Main event loop: scans Gate 0 queue every 60s; 8am briefing
│   ├── telegram_bot.py             # Telegram: /status /ideas /spend /generate /screen /briefing /dividends
│   └── morning_briefing.py         # MorningBriefing: daily 8am KL digest via Telegram
├── pipeline/                       # Stub (gates are embedded in agents)
├── requirements.txt
├── docker-compose.yml              # 2 containers: openclaw-api + openclaw-daemon
├── .env                            # Live secrets (git-ignored)
└── .env.example                    # Secrets template
```

---

## Pipeline Flow

```
[/screen or daemon tick]
        ↓
[Generate Ideas] — StrategyResearcher (Sonnet)
        ↓
[Gate 0: Novelty + Logic screen]
  novelty ≥ 0.60, logic ≥ 0.70  → Stage 1
  fail                           → Archived
        ↓
[Stage 1: Deep Research]
  research_score ≥ 0.65          → Stage 2
  fail                           → Archived
        ↓
[Gates 2-3: Backtest] — BacktestEngineer (Haiku)
  Train/Val Sharpe ≥ 0.80, DD ≤ 25%, train-val gap ≤ 35%  → Stage 4a
  + Test Sharpe ≥ 0.70
  fail                           → Archived
        ↓
[Stage 4a: Paper Trade] — PortfolioExecutor
  30+ days, Sharpe ≥ 0.80, DD ≤ 20%  → Stage 4b (live — not yet wired)
  fail                                → Archived
```

**All state tracked in `alpha_ideas` table; outcomes in `gate_decisions` + `pipeline_events`.**

---

## SQLite Schema (data/openclaw.db)

| Table | Purpose |
|-------|---------|
| `alpha_ideas` | Ideas with stage, status, scores, backtest metrics |
| `pipeline_events` | Audit log of every stage transition |
| `gate_decisions` | Gate pass/fail with rationale |
| `backtest_runs` | Train/val/test metrics + raw params JSON |
| `paper_trades` | Paper trade open/close/PnL |
| `live_trades` | OANDA order IDs (reserved for future use) |
| `ai_usage` | Token counts + cost per model per agent per task |
| `daemon_logs` | Daemon cycle logs (level, source, message) |
| `kb_documents` | Knowledge base articles with summaries |
| `kb_concepts` | Extracted concepts with domain tags |
| `kb_links` | Document graph edges |

---

## Services & Supervisor

**Supervisor config:** `/etc/supervisor/conf.d/openclaw.conf`

| Service | Command | Port | Log |
|---------|---------|------|-----|
| `openclaw-api` | `uvicorn dashboard.api.server:app --host 0.0.0.0 --port 8001` | 8001 | `logs/api.log` |
| `openclaw-daemon` | `python scripts/research_daemon.py` | — | `logs/daemon.log` |
| `openclaw-telegram` | `python scripts/telegram_bot.py` | — | `logs/telegram.log` |
| `openclaw-briefing` | `python scripts/morning_briefing.py` | — | `logs/briefing.log` |

All services auto-restart on failure. `PYTHONPATH=/opt/openclaw/app` injected by supervisor.

**Supervisor commands:**
```bash
supervisorctl status                      # view all service states
supervisorctl restart openclaw-api        # restart API
supervisorctl restart openclaw-daemon     # restart daemon
supervisorctl restart openclaw-telegram   # restart Telegram bot
supervisorctl restart all                 # restart everything
tail -f /opt/openclaw/app/logs/daemon.log # watch daemon output
```

---

## Python Venv

```
/opt/openclaw/venv/          # virtual environment root
/opt/openclaw/venv/bin/python
/opt/openclaw/venv/bin/pip
```

**Activate:**
```bash
source /opt/openclaw/venv/bin/activate
```

**Run scripts directly (without activating):**
```bash
/opt/openclaw/venv/bin/python scripts/research_daemon.py
```

**Key installed packages:**
```
anthropic==0.89.0
fastapi==0.135.3, uvicorn==0.44.0
pandas==3.0.2, numpy==2.4.4, pyarrow==23.0.1
yfinance==1.2.0
beautifulsoup4==4.14.3
python-telegram-bot==22.7
APScheduler==3.11.2
pydantic==2.12.5
```

---

## Environment Variables (.env)

```bash
ANTHROPIC_API_KEY=            # Required — Claude API key
TELEGRAM_BOT_TOKEN=           # Telegram bot token
TELEGRAM_CHAT_ID=             # Admin chat ID
TELEGRAM_ALLOWED_CHATS=       # Comma-separated whitelist
AI_DAILY_BUDGET_USD=50        # Daily cost cap (default $50)
LOG_LEVEL=INFO
# OANDA_API_KEY / ACCOUNT_ID — legacy FX, not used for KLSE
```

---

## Model Selection (config/settings.py)

| Role | Model | Use |
|------|-------|-----|
| Fast/cheap | `claude-haiku-4-5-20251001` | Backtest signal parsing, routine tasks |
| Main | `claude-sonnet-4-6` | Idea generation, deep research |
| Heavy | `claude-opus-4-6` | Available but avoided (cost) |

Cost is tracked per model/agent/task in the `ai_usage` table and enforced before every Claude call in `BaseAgent.call_claude()`.

---

## KLSE Market Parameters

- **Universe:** FBM KLCI 30 stocks (hardcoded in `config/settings.py` + live scrape)
- **Tickers:** Yahoo Finance format (`1155.KL` = Maybank, `5347.KL` = Tenaga, etc.)
- **Currency:** MYR
- **Trading hours:** 09:00–12:30, 14:30–17:00 MYT (Mon–Fri, ~6h/day)
- **Trading days/year:** 252
- **Transaction cost:** 0.13% round-trip (0.10% brokerage + 0.03% stamp duty)
- **Annualization:** √252 for Sharpe ratio

---

## API Endpoints (port 8001)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Version + timestamp |
| GET | `/api/mission-control` | Pipeline overview, spend, recent logs |
| GET | `/api/analytics` | Funnel rates, daily ideas/spend, agent stats |
| GET | `/api/pipeline/ideas` | Ideas list (filter by `stage`, `status`) |
| GET | `/api/pipeline/ideas/{id}` | Idea detail + events + backtest runs + trades |
| GET | `/api/pipeline/gate-queue` | Pending Gate 0, Stage 1, Stage 2 + recent decisions |

CORS is open (allow all origins).

---

## Telegram Commands

```
/start               Help text
/status              Pipeline health report
/ideas               Last 8 active ideas
/spend               Today's AI cost by model
/generate [topic]    Generate ideas on a topic
/screen              Scrape live KLSE data + i3investor coverage + generate ideas
/briefing            Trigger morning briefing on demand
/dividends           Ex-dividend dates in next 14 days
/search <query>      Search knowledge base
```

---

## Development Status (as of 2026-04-07)

**Built and working:**
- All 5 agents: strategy_researcher, data_engineer, backtest_engineer, portfolio_executor, risk_monitor
- Red/blue team adversarial reviewer
- KLSE screener (klsescreener.com scraper)
- Yahoo Finance data client
- Knowledge base ingester
- Research daemon (background loop)
- Telegram bot (+ /briefing, /dividends commands)
- FastAPI dashboard
- I3investor scraper (`data/i3investor/scraper.py`) — research articles, news, dividends, forum posts
- Fundamental scanner (`data/klse/fundamental_scanner.py`) — value/momentum/dividend/earnings screens
- Morning briefing (`scripts/morning_briefing.py`) — daily 8am KL digest, auto-KB-ingest
- **[Fix 2]** RejectionMemory (`knowledge/ingestion/rejection_memory.py`) — accumulates failure patterns, injects avoidance rules into idea generation
- **[Fix 3]** Feasibility scoring in Gate 0 — 3-dimensional pass: novelty≥0.60, logic≥0.70, feasibility≥0.60
- **[Fix 4]** Red-Blue team grounded in Bursa market structure (T+3, short restrictions, EPF flows, OPR)
- **[Fix 5]** Formula verification in BacktestEngineer — Claude checks signal output against formula description before full backtest
- **[Warn 1]** ResearchHunter pre-ingest relevance filter (< 0.40 = skip)
- **[Warn 2]** KB domain unified to 8 angle names; `check_balance()` uses GROUP BY domain
- **[Warn 3]** i3investor TRUSTED_BROKERAGES whitelist + 200-word minimum content filter
- **[Warn 4]** Holding period classification (INTRADAY/SHORT/MEDIUM/LONG) with per-class Sharpe thresholds
- **[Warn 5]** `_filter_infeasible()` applied to all generate_ideas() and screen_and_generate() output
- **[Warn 6]** Minimum trade count gate per holding period class; failures recorded in RejectionMemory

**Stubs / not yet implemented:**
- Live Bursa Malaysia broker integration (Stage 4b)
- Dashboard frontend UI (`dashboard/ui/`)
- Knowledge base search API (`knowledge/search/`)
- Knowledge graph (`knowledge/graph/`)
- Pipeline gates directory (`pipeline/`) — gates are embedded in agents

**Known architecture notes:**
- OANDA client exists (`data/oanda/client.py`) but is legacy FX — not used for KLSE
- Paper trading runs against SQLite, not a real broker
- `red_blue_team.py` is in `agents/red_blue_team/` (moved from researcher subdir)

---

## Architecture Notes (from 2026-04-07 compatibility audit)

### Column naming mismatch
`alpha_ideas.pair` stores a **stock ticker** (e.g. `1155.KL`), not a currency pair.
This is a historical naming artifact — do not rename (would break all queries).
When reading/writing ideas, treat `pair` as the primary ticker field.

### kb_links — doc-to-doc via shared concepts
`kb_links` has FK constraints on both `source_id` and `target_id` referencing
`kb_documents(id)`. It cannot store concept IDs directly.
Links are created as **doc-to-doc relationships** using `relation='shared_concept'`
when two documents share a concept name (matched via tags/title/summary).
Concept data lives in `kb_concepts` — `kb_links` is for graph traversal only.

### kb_documents.seeded column
`seeded INTEGER DEFAULT 0` — set to `1` after AlphaSeedGenerator processes a document.
**Always filter `WHERE seeded=0`** when selecting documents for seed generation to
avoid re-processing the same document every daemon cycle (budget waste).

### Slug prefix conventions
| Source | Slug format | Example |
|---|---|---|
| `StrategyResearcher.save_idea()` | `YYYY-MM-DD-{title}` | `2026-04-07-maybank-dividend-capture` |
| `AlphaSeedGenerator` (planned) | `seed-YYYY-MM-DD-{title}` | `seed-2026-04-07-momentum-klse` |
| `KBIngester._slug()` | `YYYY-MM-DD-{title}` | `2026-04-07-epf-ownership-dynamics` |

The `seed-` prefix prevents silent `INSERT OR IGNORE` collisions between
organically-generated ideas and KB-seeded ideas on the same day.

### KB domain classification
`VALID_DOMAINS` in `kb_ingester.py` includes both legacy domains (`fx`, `macro`,
`technical`, `fundamental`, `research`) and inference domains used by
`classify_domain()`: `alpha-ideas`, `market-structure`, `analysis-methods`,
`quant-philosophy`, `mental-models`, `factor-data`, `infrastructure`,
`portfolio-management`, `risk-management`, `behavioural`.
The `/kb` Telegram command auto-classifies documents if domain would be `"other"`.

### KB context injection in idea generation
`StrategyResearcher.generate_ideas()` searches `kb_documents` before calling
Claude and injects matching document summaries as context. Failure is non-blocking.
Logs: `"KB context: N documents found for idea generation"` in `daemon_logs`.

### Fix 2 — Rejection feedback loop (2026-04-07)
`rejection_patterns` table (factor_type, sector, reason_category, count, last_seen, example_title)
accumulates Gate 0 and Stage 2 rejection patterns. `RejectionMemory.inject_into_prompt()` returns
an avoidance block injected into `generate_ideas()` for patterns with count ≥ 2.
`alpha_ideas.rejection_reason TEXT` stores why each idea was rejected.

### Fix 3 — Three-dimensional Gate 0 (2026-04-07)
`alpha_ideas.feasibility_score REAL` — computed deterministically (no Claude call) from:
ticker format, long-only flag, Yahoo Finance data availability, holding period vs T+3,
and factor indicators. Gate 0 now requires ALL THREE: novelty ≥ 0.60 AND logic ≥ 0.70
AND feasibility ≥ 0.60. Rejection message lists which dimension(s) failed.

### Fix 4 — Red-Blue Bursa grounding (2026-04-07)
`BURSA_MARKET_BRIEF` constant injected into RED_SYSTEM, BLUE_SYSTEM, and JUDGE_SYSTEM.
Red team is explicitly instructed to attack T+3 settlement, liquidity, EPF flow reversal,
OPR sensitivity, and feasibility for every strategy.

### Fix 5 — Formula verification (2026-04-07)
`BacktestEngineer.verify_formula()` runs the parsed signal on the last 20 bars and asks
Claude (Haiku) to confirm the output matches the formula description.
`backtest_runs.needs_review INTEGER DEFAULT 0` — set to 1 if verified=False or confidence<0.7.
`backtest_runs.verification_note TEXT` — stores the issue description when flagged.

### Warning Fix 1 — ResearchHunter relevance filter (2026-04-07)
`ResearchHunter._is_relevant(title, abstract)` — calls Claude Haiku before ingesting any paper.
Papers scoring < 0.40 are skipped with a log entry: "ResearchHunter: skipped '{title}'...".
`domain` parameter added to `hunt()` so DiversityEngine can set the unified angle directly.

### Warning Fix 2 — KB domain unification (2026-04-07)
`VALID_DOMAINS` in `kb_ingester.py` now contains exactly the 8 DiversityEngine angle names.
`DOMAIN_TO_ANGLE` maps all legacy domain names to their angle equivalent.
`_normalise_domain()` used in `ingest_text()` — all incoming docs get a valid angle domain.
`classify_domain()` now returns angle names and always writes to DB.
`DiversityEngine.check_balance()` now queries `GROUP BY domain` directly — no keyword heuristics.
One-time migration: all existing docs migrated to unified angle names (2026-04-07).
Current distribution: price_action=148, event_driven=2, behavioural=2.

### Warning Fix 3 — i3investor brokerage whitelist (2026-04-07)
`TRUSTED_BROKERAGES` set added to `data/i3investor/scraper.py` (17 trusted publishers).
`_is_trusted_source(author, brokerage)` helper — shared between scraper and morning_briefing.
`get_research_articles()` now filters to trusted sources only before returning.
`auto_ingest_research()` in `morning_briefing.py` uses same whitelist + min 200-word filter.

### Warning Fix 4 — Holding period classification (2026-04-07)
`BacktestEngineer.classify_holding_period(timeframe, factor_formula, hypothesis)` — returns
INTRADAY / SHORT_TERM / MEDIUM_TERM / LONG_TERM.
Per-class Sharpe thresholds: LONG_TERM=0.8, others=1.1 (standard GATE_CONFIG).
`backtest_runs.holding_period_class TEXT` — stored with every run.
INTRADAY strategies automatically flagged needs_review=1 with warning.
SHORT_TERM strategies get "daily bar may overstate performance" warning.

### Warning Fix 5 — _filter_infeasible in generate_ideas (2026-04-07)
`StrategyResearcher._filter_infeasible(ideas)` — checks 4 criteria before any idea is saved:
(1) valid .KL ticker or sector, (2) no infeasible trading modes in hypothesis,
(3) factor_formula > 20 chars, (4) novelty≥0.5 AND logic≥0.6.
Applied in both `generate_ideas()` and `screen_and_generate()`.

### Warning Fix 6 — Minimum trade count gate (2026-04-07)
`BacktestEngineer._MIN_TRADES` — {INTRADAY: 100, SHORT_TERM: 50, MEDIUM_TERM: 30, LONG_TERM: 15}.
Trade count gate applied in `backtest_idea()` — insufficient trades → overall_pass=False.
`backtest_runs.trade_count INTEGER` — actual trade count stored per run.
Failures recorded in `RejectionMemory` with reason_category='insufficient_trades'.
