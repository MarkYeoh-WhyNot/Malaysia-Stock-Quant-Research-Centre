import os
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent.parent
load_dotenv(BASE_DIR / ".env", override=False)

# ── AI ────────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY        = os.getenv("ANTHROPIC_API_KEY", "")
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
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
    {"symbol": "1082.KL",   "name": "Hong Leong Financial",       "sector": "Banking",          "bursa_code": "1082"},
    {"symbol": "5296.KL",   "name": "QL Resources",               "sector": "Consumer Staples", "bursa_code": "5296"},
]

# Deduplicate and extract just the ticker symbols
_seen = set()
_klci_deduped = []
for _s in KLCI_STOCKS:
    if _s["symbol"] not in _seen:
        _seen.add(_s["symbol"])
        _klci_deduped.append(_s)
KLCI_STOCKS = _klci_deduped

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

# ── Database ──────────────────────────────────────────────────────────────────
DB_PATH = BASE_DIR / "data" / "openclaw.db"

# ── Pipeline gate / stage thresholds ─────────────────────────────────────────
@dataclass
class GateConfig:
    # Gate 0 — initial idea quality
    gate0_min_novelty_score: float      = 0.6
    gate0_min_logic_score: float        = 0.7
    # Stage 1 — deep research
    stage1_min_research_score: float    = 0.65
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
