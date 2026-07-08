import os
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent.parent
load_dotenv(BASE_DIR / ".env", override=False)

# ── AI ────────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY        = os.getenv("ANTHROPIC_API_KEY", "")
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
BRAVE_SEARCH_API_KEY     = os.getenv("BRAVE_SEARCH_API_KEY", "")
MODEL_FAST               = "claude-haiku-4-5-20251001"
MODEL_MAIN          = "claude-sonnet-4-6"
MODEL_HEAVY         = "claude-opus-4-6"
AI_DAILY_BUDGET_USD = float(os.getenv("AI_DAILY_BUDGET_USD", "50"))

# ── Market ────────────────────────────────────────────────────────────────────
MARKET          = "KLSE"                    # Bursa Malaysia
MARKET_CURRENCY = "MYR"
MARKET_TIMEZONE = "Asia/Kuala_Lumpur"
MARKET_NAME     = "Bursa Malaysia"

# ── FBM KLCI Top-30 Universe ─────────────────────────────────────────────────
# Yahoo Finance tickers use the .KL suffix for Bursa Malaysia
# bursa_code = the 4-digit Bursa stock code
#
# SURVIVORSHIP BIAS WARNING: this is the constituent list AS OF the date below.
# Backtests over past years use today's members and exclude stocks that were
# dropped from the index, which upward-biases results. Until point-in-time
# constituent history is added, treat all absolute backtest numbers as
# optimistic and rely on the relative gates (cross-sectional IC, OOS
# degradation, deflation hurdle) for idea ranking.
UNIVERSE_ASOF = "2026-04-07"

KLCI_STOCKS = [
    {"symbol": "1155.KL",   "name": "Maybank",                    "sector": "Banking",          "bursa_code": "1155"},
    {"symbol": "1295.KL",   "name": "Public Bank",                "sector": "Banking",          "bursa_code": "1295"},
    {"symbol": "1023.KL",   "name": "CIMB Group",                 "sector": "Banking",          "bursa_code": "1023"},
    {"symbol": "5347.KL",   "name": "Tenaga Nasional",            "sector": "Utilities",        "bursa_code": "5347"},
    {"symbol": "5183.KL",   "name": "Petronas Chemicals",         "sector": "Chemicals",        "bursa_code": "5183"},
    {"symbol": "5225.KL",   "name": "IHH Healthcare",             "sector": "Healthcare",       "bursa_code": "5225"},
    {"symbol": "8869.KL",   "name": "Press Metal Aluminium",      "sector": "Materials",        "bursa_code": "8869"},
    {"symbol": "6947.KL",   "name": "CelcomDigi",                 "sector": "Telecoms",         "bursa_code": "6947"},
    {"symbol": "6012.KL",   "name": "Maxis",                      "sector": "Telecoms",         "bursa_code": "6012"},
    {"symbol": "1066.KL",   "name": "RHB Bank",                   "sector": "Banking",          "bursa_code": "1066"},
    {"symbol": "1961.KL",   "name": "IOI Corporation",            "sector": "Plantations",      "bursa_code": "1961"},
    {"symbol": "5285.KL",   "name": "Sime Darby Plantation",      "sector": "Plantations",      "bursa_code": "5285"},
    {"symbol": "5819.KL",   "name": "Hong Leong Bank",            "sector": "Banking",          "bursa_code": "5819"},
    {"symbol": "3182.KL",   "name": "Genting",                    "sector": "Consumer Disc.",   "bursa_code": "3182"},
    {"symbol": "4715.KL",   "name": "Genting Malaysia",           "sector": "Consumer Disc.",   "bursa_code": "4715"},
    {"symbol": "4863.KL",   "name": "Telekom Malaysia",           "sector": "Telecoms",         "bursa_code": "4863"},
    {"symbol": "4707.KL",   "name": "Nestle Malaysia",            "sector": "Consumer Staples", "bursa_code": "4707"},
    {"symbol": "4065.KL",   "name": "PPB Group",                  "sector": "Consumer Staples", "bursa_code": "4065"},
    {"symbol": "6033.KL",   "name": "Petronas Gas",               "sector": "Energy",           "bursa_code": "6033"},
    {"symbol": "3816.KL",   "name": "MISC Berhad",                "sector": "Transportation",   "bursa_code": "3816"},
    {"symbol": "5168.KL",   "name": "Hartalega",                  "sector": "Healthcare",       "bursa_code": "5168"},
    {"symbol": "2445.KL",   "name": "Kuala Lumpur Kepong",        "sector": "Plantations",      "bursa_code": "2445"},
    {"symbol": "7277.KL",   "name": "Dialog Group",               "sector": "Energy",           "bursa_code": "7277"},
    {"symbol": "1015.KL",   "name": "AmBank Group",               "sector": "Banking",          "bursa_code": "1015"},
    {"symbol": "4197.KL",   "name": "Sime Darby",                 "sector": "Industrial",       "bursa_code": "4197"},
    {"symbol": "1082.KL",   "name": "Hong Leong Financial",       "sector": "Banking",          "bursa_code": "1082"},
    {"symbol": "4677.KL",   "name": "YTL Corporation",            "sector": "Utilities",        "bursa_code": "4677"},
    {"symbol": "5398.KL",   "name": "Gamuda",                     "sector": "Construction",     "bursa_code": "5398"},
    {"symbol": "5296.KL",   "name": "QL Resources",               "sector": "Consumer Staples", "bursa_code": "5296"},
]

assert len({_s["symbol"] for _s in KLCI_STOCKS}) == len(KLCI_STOCKS), \
    "Duplicate symbol in KLCI_STOCKS"

DEFAULT_SYMBOLS = [s["symbol"] for s in KLCI_STOCKS]

# Backwards-compatible alias used throughout agents / data engineer
DEFAULT_PAIRS   = DEFAULT_SYMBOLS

# Sector lookup
KLCI_BY_SYMBOL  = {s["symbol"]: s for s in KLCI_STOCKS}
KLCI_BY_CODE    = {s["bursa_code"]: s for s in KLCI_STOCKS}
KLCI_SECTORS    = sorted(set(s["sector"] for s in KLCI_STOCKS))

# ── Bursa Malaysia trading calendar ──────────────────────────────────────────
# Morning session: 09:00–12:30 MYT, Afternoon: 14:30–17:00 MYT
MARKET_OPEN_HOUR  = 9
MARKET_CLOSE_HOUR = 17
TRADING_DAYS_PER_YEAR = 252

# ── Bursa Malaysia market rules & transaction cost model ──────────────────────
# Single source of truth — the backtester and paper trading must both use these.
#
# MARKET_RULES_VERSION / FEE_MODEL_VERSION are stamped onto every backtest_runs
# row so results are always traceable to the assumptions in force when they ran.
# Bump these whenever a rule or fee below changes.
MARKET_RULES_VERSION      = "2026-07-09"   # T+2 settlement, 100-share board lot, long-only
FEE_MODEL_VERSION         = "2026-07-09"   # 0.10% remitted stamp (cap RM1000), 0.03% clearing

# Settlement: Bursa normal delivery & settlement is T+2 (effective 2019-04-29,
# Bursa Malaysia Securities Clearing). Used in feasibility scoring + red-team
# reasoning; it does not feed the cost math below.
BURSA_SETTLEMENT_CYCLE    = "T+2"

BURSA_COMMISSION_RATE     = 0.0008     # 0.08% per side
# Stamp duty: statutory RM1.50/RM1,000 (0.15%), but REMITTED to an effective
# 0.10% for contract notes executed 2023-07-13 → 2028-07-12, capped at RM1,000
# per contract note (raised from the old RM200 cap). At RM100k paper scale the
# cap rarely binds; the 0.15→0.10 rate cut is the material change (lowers cost).
BURSA_STAMP_DUTY_RATE     = 0.0010     # 0.10% remitted, buy-side only
BURSA_STAMP_DUTY_CAP_MYR  = 1000.0     # capped at RM1,000 per contract note
BURSA_STAMP_REMISSION_END = "2028-07-12"   # revert to 0.15% if not extended
BURSA_CLEARING_RATE       = 0.0003     # 0.03% per side
BURSA_CLEARING_CAP_MYR    = 1000.0     # capped at RM1,000 per side
BURSA_BOARD_LOT           = 100        # minimum lot size (shares)

# Slippage by liquidity tier (fraction of trade value, per side)
BURSA_SLIPPAGE_TIERS = {
    "BLUE_CHIP": 0.0005,   # ADV value ≥ RM20M
    "MID_CAP":   0.0025,   # ADV value ≥ RM2M
    "SMALL_CAP": 0.0075,   # below RM2M
}
BURSA_TIER_BLUE_CHIP_MYR = 20_000_000.0
BURSA_TIER_MID_CAP_MYR   = 2_000_000.0

# Liquidity floor: reject strategies on names below this avg daily traded value
BURSA_MIN_DAILY_VALUE_MYR = 500_000.0

# Notional capital allocated to each paper-traded idea
PAPER_CAPITAL_MYR = 100_000.0
PAPER_ALLOC_PCT   = 0.95     # fraction of idea NAV deployed per position


def bursa_slippage_tier(avg_daily_value_myr: float) -> str:
    """Classify a stock's liquidity tier from average daily traded value (MYR)."""
    if avg_daily_value_myr >= BURSA_TIER_BLUE_CHIP_MYR:
        return "BLUE_CHIP"
    if avg_daily_value_myr >= BURSA_TIER_MID_CAP_MYR:
        return "MID_CAP"
    return "SMALL_CAP"


def bursa_trade_cost(trade_value_myr: float, side: str,
                     slippage_tier: str = "BLUE_CHIP") -> float:
    """Total cost in MYR for one side of a Bursa trade.

    side: 'buy' or 'sell'. Stamp duty applies to the buy side only and is
    capped at RM1,000; clearing is capped at RM1,000 per side.
    """
    value = abs(trade_value_myr)
    cost = value * BURSA_COMMISSION_RATE
    cost += min(value * BURSA_CLEARING_RATE, BURSA_CLEARING_CAP_MYR)
    if side == "buy":
        cost += min(value * BURSA_STAMP_DUTY_RATE, BURSA_STAMP_DUTY_CAP_MYR)
    cost += value * BURSA_SLIPPAGE_TIERS.get(slippage_tier, BURSA_SLIPPAGE_TIERS["MID_CAP"])
    return cost

# ── Runtime state directory ───────────────────────────────────────────────────
# All mutable runtime artifacts (SQLite DB, parquet cache, heartbeat, progress
# file) live here. Defaults to the repo's data/ dir for local development, but
# Docker deployments MUST override via OPENCLAW_RUNTIME_DIR: data/ also holds
# source code (database.py, yahoo/, klse/...), and mounting a named volume
# over /app/data shadowed those modules with first-deploy copies — code
# updates under data/ silently never reached the running containers.
RUNTIME_DIR = Path(os.getenv("OPENCLAW_RUNTIME_DIR", str(BASE_DIR / "data")))

# ── Database ──────────────────────────────────────────────────────────────────
DB_PATH = RUNTIME_DIR / "openclaw.db"

# ── Pipeline gate / stage thresholds ─────────────────────────────────────────
@dataclass
class GateConfig:
    # Gate 0 — initial idea quality
    gate0_min_novelty_score: float      = 0.6
    gate0_min_logic_score: float        = 0.7
    # Stage 1 — deep research (score is 0.0–10.0; 6.5 ≈ "solid KLSE evidence")
    stage1_min_research_score: float    = 6.5
    # Stage 2/3 — backtesting (slightly relaxed vs FX — equities have lower Sharpe norms)
    stage3_min_sharpe: float            = 0.8
    stage3_max_train_val_gap: float     = 0.35
    stage3_min_test_sharpe: float       = 0.7
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
        "SHORT_TERM":  30,
        "MEDIUM_TERM": 60,    # ~1 quarter of daily bars
        "LONG_TERM":   120,   # low-turnover needs a longer live look
    })
    # Gate DQ — data-quality gate (audit §6.4). A strategy is not backtested
    # unless its price data clears a minimum Data Confidence Score (0–100). Clean
    # daily blue-chip data scores ~95–100; thin/gappy/stale series score low.
    dq_gate_enabled: bool               = True
    dq_min_confidence: float            = 80.0
    # Suspected-corporate-action gap threshold (unhandled bonus/rights issues):
    # an overnight move beyond this magnitude is flagged and dents confidence.
    dq_corp_action_gap: float           = 0.25
    # Benchmark-relative gate (audit §8.4): a strategy must beat a simple KLCI
    # baseline after costs, else its complexity is not justified. Gates on excess
    # annual return vs the equal-weight KLCI (the harder of the two baselines);
    # KLCI buy-and-hold excess is also stored for reference.
    benchmark_gate_enabled: bool        = True
    benchmark_min_excess_ann: float     = 0.0    # strategy ann_return must exceed EW-KLCI by this
    # Phase 3.4 — capacity test (audit §8.5). Don't trade more than
    # capacity_max_participation of 20-day ADV per day; a strategy whose position
    # takes longer than capacity_max_days to enter/exit is capacity-constrained.
    # Rarely binds at RM100k paper scale but required for the capital-scaling story.
    capacity_gate_enabled: bool         = True
    capacity_max_participation: float   = 0.05
    capacity_max_days: float            = 5.0
    # Phase 4.2 — portfolio concentration limits (audit §10.2), Malaysia-specific.
    max_single_name_pct: float          = 0.15
    max_sector_pct: float               = 0.35
    max_bank_pct: float                 = 0.40   # KLCI is bank-heavy
    # QC7 — parameter robustness (DSL signals): fraction of ±20% parameter
    # perturbations that must retain > robustness_sharpe_ratio × base Sharpe
    robustness_min_fraction: float      = 0.6
    robustness_sharpe_ratio: float      = 0.5
    robustness_draws: int               = 8

GATE_CONFIG = GateConfig()

# ── Messaging ─────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN  = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID    = os.getenv("TELEGRAM_CHAT_ID", "")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

# ── OANDA (legacy — kept for backward compat, not used for KLSE) ──────────────
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
