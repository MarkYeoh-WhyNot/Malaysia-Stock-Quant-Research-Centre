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

## Development Status (as of 2026-04-06)

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

**Stubs / not yet implemented:**
- Live Bursa Malaysia broker integration (Stage 4b)
- Dashboard frontend UI (`dashboard/ui/`)
- Knowledge base search API (`knowledge/search/`)
- Knowledge graph (`knowledge/graph/`)
- Pipeline gates directory (`pipeline/`) — gates are embedded in agents

**Known architecture notes:**
- OANDA client exists (`data/oanda/client.py`) but is legacy FX — not used for KLSE
- Paper trading runs against SQLite, not a real broker
- `red_blue_team.py` is in `agents/researcher/` (not its own subdirectory)
