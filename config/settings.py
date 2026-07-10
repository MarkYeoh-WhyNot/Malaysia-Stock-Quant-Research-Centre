"""Central settings — market-agnostic config + the active MARKET PROFILE.

Dual-market design (2026-07-09): every market-specific value (universe, cost
model, calendar, prompts, ticker rules) lives in a profile module under
config/markets/, selected once per process by the MARKET_MODE env var
(default "bursa" — bit-identical to the original single-market system).

Legacy names (KLCI_STOCKS, BURSA_*, bursa_trade_cost, ...) are re-exported from
the active profile so the ~20 existing import sites keep working unchanged; in
crypto mode those same names simply carry crypto values. New code should prefer
the generic names (MARKET_UNIVERSE, trade_cost, TICKER_REGEX, ...).

One process = one market. Strict pipeline isolation comes from running separate
containers with different MARKET_MODE + OPENCLAW_RUNTIME_DIR (separate DBs).
"""
import os
from pathlib import Path
from dataclasses import dataclass, field, replace
from datetime import datetime
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent.parent
load_dotenv(BASE_DIR / ".env", override=False)

# ── Active market profile ─────────────────────────────────────────────────────
MARKET_MODE = os.getenv("MARKET_MODE", "bursa").strip().lower()

from config.markets import load_market_profile  # noqa: E402

MARKET_PROFILE = load_market_profile(MARKET_MODE)
_P = MARKET_PROFILE

# ── AI ────────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY        = os.getenv("ANTHROPIC_API_KEY", "")
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
BRAVE_SEARCH_API_KEY     = os.getenv("BRAVE_SEARCH_API_KEY", "")
MODEL_FAST               = "claude-haiku-4-5-20251001"
MODEL_MAIN          = "claude-sonnet-4-6"
MODEL_HEAVY         = "claude-opus-4-6"
AI_DAILY_BUDGET_USD = float(os.getenv("AI_DAILY_BUDGET_USD", "50"))

# ── Concierge chat agent ──────────────────────────────────────────────────────
# The dashboard Concierge is a tool-calling agent that lets a human submit ideas
# in natural language and check their status. It has its own daily sub-cap so a
# chatty session can't starve the research pipeline's AI_DAILY_BUDGET_USD.
CONCIERGE_MODEL           = os.getenv("CONCIERGE_MODEL", MODEL_MAIN)
CONCIERGE_DAILY_BUDGET_USD = float(os.getenv("CONCIERGE_DAILY_BUDGET_USD", "5"))
CONCIERGE_MAX_TOOL_ITERS  = int(os.getenv("CONCIERGE_MAX_TOOL_ITERS", "6"))

# ── Market identity (from profile) ────────────────────────────────────────────
MARKET          = _P.MARKET
MARKET_CURRENCY = _P.MARKET_CURRENCY
MARKET_TIMEZONE = _P.MARKET_TIMEZONE
MARKET_NAME     = _P.MARKET_NAME
UNIVERSE_NAME   = _P.UNIVERSE_NAME

# ── Universe (from profile) ───────────────────────────────────────────────────
UNIVERSE_ASOF = _P.UNIVERSE_ASOF

# Legacy name: "KLCI_STOCKS" is the universe of the ACTIVE market. In crypto
# mode it holds the crypto universe — the name is kept only so existing import
# sites don't change. New code: use MARKET_UNIVERSE.
MARKET_UNIVERSE = _P.UNIVERSE
KLCI_STOCKS     = MARKET_UNIVERSE

assert len({_s["symbol"] for _s in KLCI_STOCKS}) == len(KLCI_STOCKS), \
    "Duplicate symbol in market universe"

DEFAULT_SYMBOLS = [s["symbol"] for s in KLCI_STOCKS]

# Backwards-compatible alias used throughout agents / data engineer
DEFAULT_PAIRS   = DEFAULT_SYMBOLS

# Sector lookup
KLCI_BY_SYMBOL  = {s["symbol"]: s for s in KLCI_STOCKS}
KLCI_BY_CODE    = {s["bursa_code"]: s for s in KLCI_STOCKS}
KLCI_SECTORS    = sorted(set(s["sector"] for s in KLCI_STOCKS))

# ── Trading calendar (from profile) ──────────────────────────────────────────
MARKET_OPEN_HOUR      = _P.MARKET_OPEN_HOUR
MARKET_CLOSE_HOUR     = _P.MARKET_CLOSE_HOUR
TRADING_DAYS_PER_YEAR = _P.TRADING_DAYS_PER_YEAR
MARKET_CALENDAR       = _P.CALENDAR      # "business" | "daily"
HAS_CORPORATE_ACTIONS = _P.HAS_CORPORATE_ACTIONS  # False on crypto — no splits/dividends

# ── Market rules & transaction cost model (from profile) ─────────────────────
# MARKET_RULES_VERSION / FEE_MODEL_VERSION are stamped onto every backtest_runs
# row so results are always traceable to the assumptions in force when they ran.
MARKET_RULES_VERSION = _P.MARKET_RULES_VERSION
FEE_MODEL_VERSION    = _P.FEE_MODEL_VERSION

BURSA_SETTLEMENT_CYCLE    = _P.SETTLEMENT_CYCLE
SETTLEMENT_CYCLE          = _P.SETTLEMENT_CYCLE

BURSA_COMMISSION_RATE     = _P.COMMISSION_RATE
BURSA_STAMP_DUTY_RATE     = _P.STAMP_DUTY_RATE
BURSA_STAMP_DUTY_CAP_MYR  = _P.STAMP_DUTY_CAP
BURSA_STAMP_REMISSION_END = _P.STAMP_REMISSION_END
BURSA_CLEARING_RATE       = _P.CLEARING_RATE
BURSA_CLEARING_CAP_MYR    = _P.CLEARING_CAP
BURSA_BOARD_LOT           = _P.BOARD_LOT

BURSA_SLIPPAGE_TIERS      = _P.SLIPPAGE_TIERS
BURSA_TIER_BLUE_CHIP_MYR  = _P.TIER_BLUE_CHIP
BURSA_TIER_MID_CAP_MYR    = _P.TIER_MID_CAP
BURSA_MIN_DAILY_VALUE_MYR = _P.MIN_DAILY_VALUE

# Notional capital allocated to each paper-traded idea (market-agnostic:
# 100k MYR for Bursa, 100k USDT for crypto — same order of magnitude).
PAPER_CAPITAL_MYR = 100_000.0
PAPER_ALLOC_PCT   = 0.95     # fraction of idea NAV deployed per position

# Cost / sizing functions — legacy names bound to the active profile.
bursa_slippage_tier = _P.slippage_tier
bursa_trade_cost    = _P.trade_cost
slippage_tier       = _P.slippage_tier
trade_cost          = _P.trade_cost
size_units          = _P.size_units
funding_cost        = _P.funding_cost

# ── Long/short (WS3) — perpetuals-only; ALLOW_SHORT=False on every other
# market. Everything downstream (DSL, backtester, feasibility, concierge,
# paper trades) branches on this single flag rather than checking DATA_BACKEND.
ALLOW_SHORT            = _P.ALLOW_SHORT
MAX_LEVERAGE           = _P.MAX_LEVERAGE
DEFAULT_LEVERAGE       = _P.DEFAULT_LEVERAGE
LIQUIDATION_BUFFER     = _P.LIQUIDATION_BUFFER
FUNDING_INTERVAL_HOURS = _P.FUNDING_INTERVAL_HOURS
AVG_FUNDING_RATE_PER_INTERVAL = _P.AVG_FUNDING_RATE_PER_INTERVAL

# ── Instruments / prompts / jobs (generic names, from profile) ───────────────
TICKER_REGEX     = _P.TICKER_REGEX
TICKER_EXAMPLE   = _P.TICKER_EXAMPLE
DATA_BACKEND     = _P.DATA_BACKEND
BENCHMARK_SYMBOL = _P.BENCHMARK_SYMBOL
BLOCKED_MODES    = _P.BLOCKED_MODES
UNAVAILABLE_DATA_KEYWORDS = _P.UNAVAILABLE_DATA_KEYWORDS
EXOTIC_KEYWORDS  = _P.EXOTIC_KEYWORDS
FEASIBILITY_DOCK_KEYWORDS = _P.FEASIBILITY_DOCK_KEYWORDS

# ── Timeframes (profile-driven; Bursa values reproduce daily/weekly-only) ────
ALLOWED_TIMEFRAMES = _P.ALLOWED_TIMEFRAMES
FETCH_DAYS_BY_INTERVAL = _P.FETCH_DAYS_BY_INTERVAL
CACHE_STALENESS_HOURS_BY_INTERVAL = _P.CACHE_STALENESS_HOURS_BY_INTERVAL


def bars_per_day(interval: str) -> float:
    """Bars per calendar/trading day for the active market's data backend.
    Exactly 1.0 for '1d', so every formula scaled by this is neutral on the
    daily path (Bursa parity)."""
    if DATA_BACKEND == "binance":
        from data.binance.client import BARS_PER_YEAR
    else:
        from data.yahoo.client import BARS_PER_YEAR
    return BARS_PER_YEAR.get(interval, BARS_PER_YEAR["1d"]) / BARS_PER_YEAR["1d"]
MARKET_BRIEF     = _P.MARKET_BRIEF
RED_TEAM_ATTACKS = _P.RED_TEAM_ATTACKS
BLUE_DEFENSE_NOTES = _P.BLUE_DEFENSE_NOTES
JUDGE_REJECT_RULE  = _P.JUDGE_REJECT_RULE
INSTRUMENT_TYPE  = _P.INSTRUMENT_TYPE
CONCENTRATION_SECTOR = _P.CONCENTRATION_SECTOR
ENABLED_JOBS     = _P.ENABLED_JOBS        # None = all jobs
DIRECTION_DOC    = _P.DIRECTION_DOC        # System Direction dashboard content

# ── Research content (KB hunt / alpha seeds, from profile) ───────────────────
# Fixes the bug where DiversityEngine/ResearchHunter/KBIngester/AlphaSeedGenerator
# hardcoded Bursa Malaysia content regardless of MARKET_MODE (2026-07-09).
RESEARCH_ANGLES        = _P.RESEARCH_ANGLES
ANGLE_KEYWORDS         = _P.ANGLE_KEYWORDS
RESEARCH_QUERY_PERSONA = _P.RESEARCH_QUERY_PERSONA
ALPHA_SEED_SYSTEM      = _P.ALPHA_SEED_SYSTEM
DATA_SOURCES_EXAMPLE   = _P.DATA_SOURCES_EXAMPLE
RELEVANCE_TARGET       = _P.RELEVANCE_TARGET
RELEVANCE_SCALE        = _P.RELEVANCE_SCALE

# ── Runtime state directory ───────────────────────────────────────────────────
# All mutable runtime artifacts (SQLite DB, parquet cache, heartbeat, progress
# file) live here. Defaults to the repo's data/ dir for local development, but
# Docker deployments MUST override via OPENCLAW_RUNTIME_DIR: data/ also holds
# source code (database.py, yahoo/, klse/...), and mounting a named volume
# over /app/data shadowed those modules with first-deploy copies — code
# updates under data/ silently never reached the running containers.
# Each market's containers mount a DIFFERENT volume here → separate DBs.
RUNTIME_DIR = Path(os.getenv("OPENCLAW_RUNTIME_DIR", str(BASE_DIR / "data")))

# ── Database ──────────────────────────────────────────────────────────────────
DB_PATH = RUNTIME_DIR / "openclaw.db"

# ── Pipeline gate / stage thresholds ─────────────────────────────────────────
@dataclass
class GateConfig:
    # Gate 0 — initial idea quality. Values match what score_gate0 ACTUALLY
    # enforces (audit 2026-07-10: the old 0.6/0.7 constants here were DEAD —
    # never read — while the code hardcoded 0.65). Novelty is ADVISORY by
    # design (recorded, never gates: LLM novelty scores are too noisy and a
    # simple genuine alpha always scores low) — the field is kept only as
    # documentation of that intent.
    gate0_min_novelty_score: float      = 0.6   # ADVISORY — not enforced
    gate0_min_logic_score: float        = 0.65
    # Stage 1 — deep research (score is 0.0–10.0; 6.5 ≈ "solid KLSE evidence")
    stage1_min_research_score: float    = 6.5
    # Stage 2/3 — backtesting (slightly relaxed vs FX — equities have lower Sharpe norms)
    # stage3_min_test_sharpe was deleted 2026-07-10 (dead — never read; the
    # PSR principal rule replaced fixed Sharpe thresholds entirely).
    # stage3_min_sharpe survives only as the .get() fallback for unknown
    # holding-period classes in the fundamental-screen path.
    stage3_min_sharpe: float            = 0.8
    stage3_max_train_val_gap: float     = 0.35
    stage3_max_drawdown: float          = 0.25
    stage3_data_split_train: float      = 0.60
    stage3_data_split_val: float        = 0.20
    stage3_data_split_test: float       = 0.20
    # Stage 4a — paper trading
    stage4a_min_days: int               = 30
    stage4a_min_sharpe: float           = 0.8
    stage4a_max_drawdown: float         = 0.20
    # Paper-trade promotion by holding cycle (audit §8.6): 30 calendar days is
    # too short for low-turnover strategies. A strategy may leave paper trading
    # once it satisfies EITHER the day floor OR enough completed trades OR enough
    # rebalance cycles — whichever fits its holding-period class. Keyed off the
    # backtest's holding_period_class.
    stage4a_min_trades: int             = 20     # completed round-trips (alt to days)
    stage4a_min_rebalance_cycles: int   = 3      # full rebalance cycles (alt to days)
    stage4a_min_days_by_class: dict     = field(default_factory=lambda: {
        "INTRADAY":    30,
        "SUBDAILY":    30,    # calendar days — sub-daily marks accumulate fast
        "SHORT_TERM":  30,
        "MEDIUM_TERM": 60,    # ~1 quarter of daily bars
        "LONG_TERM":   120,   # low-turnover needs a longer live look
    })
    # Phase 5.4 — strategy-family quotas (audit §9.3), keyed to the same
    # factor_type taxonomy RejectionMemory uses for rejection patterns. Report-only
    # (surfaced in prompts/dashboard) rather than a hard gate — avoids blocking
    # generation on a heuristic keyword classification.
    family_quota_targets: dict          = field(default_factory=lambda: {
        "momentum": 0.20, "value": 0.20, "quality": 0.15, "mean_reversion": 0.15,
        "event": 0.15, "macro": 0.15,
    })
    # Gate DQ — data-quality gate (audit §6.4). A strategy is not backtested
    # unless its price data clears a minimum Data Confidence Score (0–100). Clean
    # daily blue-chip data scores ~95–100; thin/gappy/stale series score low.
    dq_gate_enabled: bool               = True
    dq_min_confidence: float            = 80.0
    # Suspected-corporate-action gap threshold (unhandled bonus/rights issues):
    # an overnight move beyond this magnitude is flagged and dents confidence.
    dq_corp_action_gap: float           = 0.25
    # Benchmark-relative gate (audit §8.4): a strategy must beat a simple
    # baseline after costs, else its complexity is not justified. Gates on excess
    # annual return vs the equal-weight universe (the harder of the two
    # baselines); the index/benchmark-symbol excess is also stored for reference.
    benchmark_gate_enabled: bool        = True
    benchmark_min_excess_ann: float     = 0.0    # strategy ann_return must exceed EW baseline by this
    # Phase 3.4 — capacity test (audit §8.5). Don't trade more than
    # capacity_max_participation of 20-day ADV per day; a strategy whose position
    # takes longer than capacity_max_days to enter/exit is capacity-constrained.
    capacity_gate_enabled: bool         = True
    capacity_max_participation: float   = 0.05
    capacity_max_days: float            = 5.0
    # Phase 4.2 — portfolio concentration limits (audit §10.2).
    max_single_name_pct: float          = 0.15
    max_sector_pct: float               = 0.35
    max_bank_pct: float                 = 0.40   # watched sector = profile CONCENTRATION_SECTOR
    # QC7 — parameter robustness (DSL signals): fraction of ±20% parameter
    # perturbations that must retain > robustness_sharpe_ratio × base Sharpe
    robustness_min_fraction: float      = 0.6
    robustness_sharpe_ratio: float      = 0.5
    robustness_draws: int               = 8
    # Cross-sectional (IC) gate — North Star metric #1. Defaults are the
    # values previously HARDCODED in cross_sectional_test (backtest_engineer)
    # so Bursa behavior is identical; crypto overrides xs_min_positive_names
    # proportionally to its 20-pair universe.
    xs_min_mean_ic: float               = 0.05
    xs_min_ic_tstat: float              = 1.5
    xs_min_positive_names: int          = 15   # of the 30-name KLCI universe
    # ── Principal pass rule (gate redesign 2026-07-10): deflated PSR ────────
    # Pass iff P(true net Sharpe > SR*) ≥ confidence on the FULL-window
    # evidence, where SR* is the expected max Sharpe of the recent search's
    # noise trials (deflated benchmark — already a high bar: beating the BEST
    # of N noise strategies, not beating zero). Replaces the fixed per-class
    # Sharpe thresholds + the separate deflation binary.
    # 0.70 is CALIBRATED, not arbitrary: harness strength tiers demand noise
    # ≤5%, strong(SR~2.6) ≥90%, moderate(SR~1.4) ≥60% on 2000 bars. Single-
    # strategy noise at 0.70 vs SR*≈1.0 still needs an observed Sharpe ≥1.2
    # (≈0.2% FPR) BEFORE the OOS/regime/robustness/gap guards also fire —
    # joint noise pass rate is pinned at ~0% by the harness.
    psr_confidence_test: float          = 0.70
    # Pooled train+val PSR is DIAGNOSTIC (reported, not gated — gating it
    # would double-charge the same evidence the full-window PSR weighs).
    psr_confidence_trainval: float      = 0.90
    # Deflation trial count = distinct ideas backtested in this window (not
    # all history — an ever-growing N silently raised the bar forever).
    deflation_window_days: int          = 90
    # Gate 0 thresholds — the values score_gate0 ACTUALLY enforces (novelty
    # is advisory by design: LLM novelty scores are too noisy to gate on).
    gate0_min_claude_feasibility: float = 0.70
    gate0_min_data_quality: float       = 0.70
    gate0_max_overfitting_risk: float   = 0.40

# Profile-specific threshold overrides (e.g. crypto's wider drawdown norms).
# Bursa's overrides are {} — defaults ARE the Bursa values.
GATE_CONFIG = replace(GateConfig(), **_P.GATE_OVERRIDES)

# ── Messaging ─────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN  = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID    = os.getenv("TELEGRAM_CHAT_ID", "")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

# ── OANDA (legacy — kept for backward compat, not used) ──────────────────────
OANDA_API_KEY     = os.getenv("OANDA_API_KEY", "")
OANDA_ACCOUNT_ID  = os.getenv("OANDA_ACCOUNT_ID", "")
OANDA_ENVIRONMENT = os.getenv("OANDA_ENVIRONMENT", "practice")
OANDA_BASE_URL    = (
    "https://api-fxtrade.oanda.com"
    if OANDA_ENVIRONMENT == "live"
    else "https://api-fxpractice.oanda.com"
)

# ── Dashboard ─────────────────────────────────────────────────────────────────
DASHBOARD_HOST = "0.0.0.0"
DASHBOARD_PORT = 8001
LOG_DIR        = BASE_DIR / "logs"
LOG_LEVEL      = os.getenv("LOG_LEVEL", "INFO")

# API auth: shared secret required in the X-API-Key header on all /api routes
# (except /api/health). Empty = auth disabled (local development only).
OPENCLAW_API_KEY = os.getenv("OPENCLAW_API_KEY", "")
# Origin allowed by CORS, e.g. "https://openclaw.example.com"
DASHBOARD_ORIGIN = os.getenv("DASHBOARD_ORIGIN", "http://localhost")

# Cross-process progress file written by BacktestEngineer, read by the API
PROGRESS_FILE = RUNTIME_DIR / "openclaw_progress.json"

# Operational alerts (daemon crash/restart, budget exhausted) via Telegram
ALERT_TELEGRAM_CHAT_ID = os.getenv("ALERT_TELEGRAM_CHAT_ID", "") or os.getenv("TELEGRAM_CHAT_ID", "")


# ── Security / Key health ─────────────────────────────────────────────────────

def key_health_check() -> dict:
    """Check API key validity and rotation reminder.

    Never logs the actual key — only the first 8 characters as a preview.
    Creates .key_rotation_date on first run; warns if > 30 days without rotation.
    """
    issues = []

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key or api_key == "sk-ant-your-key-here":
        issues.append("ANTHROPIC_API_KEY is not set or is a placeholder")
    elif not api_key.startswith("sk-ant-"):
        issues.append("ANTHROPIC_API_KEY format looks wrong (expected 'sk-ant-...')")

    rotation_file = BASE_DIR / ".key_rotation_date"
    if rotation_file.exists():
        try:
            rotation_date = datetime.fromisoformat(rotation_file.read_text().strip())
            days_since = (datetime.now() - rotation_date).days
            if days_since > 30:
                issues.append(
                    f"API key not rotated in {days_since} days — "
                    f"consider rotating for security"
                )
        except Exception:
            rotation_file.write_text(datetime.now().isoformat())
    else:
        rotation_file.write_text(datetime.now().isoformat())

    key_preview = api_key[:8] + "..." if api_key else "NOT SET"
    return {
        "key_preview": key_preview,
        "issues": issues,
        "healthy": len(issues) == 0,
    }
