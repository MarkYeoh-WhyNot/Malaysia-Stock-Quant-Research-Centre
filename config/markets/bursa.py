"""Bursa Malaysia (KLSE) market profile — the system's original market.

Values here were moved VERBATIM from config/settings.py (2026-07-09); a
regression test pins the key numbers so this extraction can never silently
change Bursa behavior. config/settings.py re-exports everything under the
legacy names (KLCI_STOCKS, BURSA_*, bursa_trade_cost, ...).
"""
from __future__ import annotations

import re

MODE            = "bursa"
MARKET          = "KLSE"                    # Bursa Malaysia
MARKET_CURRENCY = "MYR"
MARKET_TIMEZONE = "Asia/Kuala_Lumpur"
MARKET_NAME     = "Bursa Malaysia"
UNIVERSE_NAME   = "FBMKLCI"

# ── FBM KLCI Top-30 Universe ─────────────────────────────────────────────────
# Yahoo Finance tickers use the .KL suffix for Bursa Malaysia
# code = the 4-digit Bursa stock code
#
# SURVIVORSHIP BIAS WARNING: this is the constituent list AS OF the date below.
# Backtests over past years use today's members and exclude stocks that were
# dropped from the index, which upward-biases results. Until point-in-time
# constituent history is added, treat all absolute backtest numbers as
# optimistic and rely on the relative gates (cross-sectional IC, OOS
# degradation, deflation hurdle) for idea ranking.
UNIVERSE_ASOF = "2026-04-07"

UNIVERSE = [
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

# ── Trading calendar ──────────────────────────────────────────────────────────
# Morning session: 09:00–12:30 MYT, Afternoon: 14:30–17:00 MYT
MARKET_OPEN_HOUR      = 9
MARKET_CLOSE_HOUR     = 17
TRADING_DAYS_PER_YEAR = 252
CALENDAR              = "business"   # pd.bdate_range — weekends are non-trading

# ── Market rules & transaction cost model ─────────────────────────────────────
MARKET_RULES_VERSION = "2026-07-09"   # T+2 settlement, 100-share board lot, long-only
FEE_MODEL_VERSION    = "2026-07-09"   # 0.10% remitted stamp (cap RM1000), 0.03% clearing

# Settlement: Bursa normal delivery & settlement is T+2 (effective 2019-04-29,
# Bursa Malaysia Securities Clearing). Used in feasibility scoring + red-team
# reasoning; it does not feed the cost math below.
SETTLEMENT_CYCLE = "T+2"

COMMISSION_RATE     = 0.0008     # 0.08% per side
# Stamp duty: statutory RM1.50/RM1,000 (0.15%), but REMITTED to an effective
# 0.10% for contract notes executed 2023-07-13 → 2028-07-12, capped at RM1,000
# per contract note (raised from the old RM200 cap). At RM100k paper scale the
# cap rarely binds; the 0.15→0.10 rate cut is the material change (lowers cost).
STAMP_DUTY_RATE     = 0.0010     # 0.10% remitted, buy-side only
STAMP_DUTY_CAP      = 1000.0     # capped at RM1,000 per contract note
STAMP_REMISSION_END = "2028-07-12"   # revert to 0.15% if not extended
CLEARING_RATE       = 0.0003     # 0.03% per side
CLEARING_CAP        = 1000.0     # capped at RM1,000 per side
BOARD_LOT           = 100        # minimum lot size (shares); whole board lots only

# Slippage by liquidity tier (fraction of trade value, per side)
SLIPPAGE_TIERS = {
    "BLUE_CHIP": 0.0005,   # ADV value ≥ RM20M
    "MID_CAP":   0.0025,   # ADV value ≥ RM2M
    "SMALL_CAP": 0.0075,   # below RM2M
}
TIER_BLUE_CHIP = 20_000_000.0
TIER_MID_CAP   = 2_000_000.0

# Liquidity floor: reject strategies on names below this avg daily traded value
MIN_DAILY_VALUE = 500_000.0


def slippage_tier(avg_daily_value: float) -> str:
    """Classify a stock's liquidity tier from average daily traded value (MYR)."""
    if avg_daily_value >= TIER_BLUE_CHIP:
        return "BLUE_CHIP"
    if avg_daily_value >= TIER_MID_CAP:
        return "MID_CAP"
    return "SMALL_CAP"


def trade_cost(trade_value: float, side: str,
               tier: str = "BLUE_CHIP") -> float:
    """Total cost in MYR for one side of a Bursa trade.

    side: 'buy' or 'sell'. Stamp duty applies to the buy side only and is
    capped at RM1,000; clearing is capped at RM1,000 per side.
    """
    value = abs(trade_value)
    cost = value * COMMISSION_RATE
    cost += min(value * CLEARING_RATE, CLEARING_CAP)
    if side == "buy":
        cost += min(value * STAMP_DUTY_RATE, STAMP_DUTY_CAP)
    cost += value * SLIPPAGE_TIERS.get(tier, SLIPPAGE_TIERS["MID_CAP"])
    return cost


def size_units(nav: float, price: float, alloc_pct: float) -> float:
    """Units to buy: alloc_pct of NAV, rounded down to a whole 100-share board lot."""
    if price <= 0:
        return 0
    lots = int((nav * alloc_pct) / price / BOARD_LOT)
    return lots * BOARD_LOT


# ── Instruments / data ────────────────────────────────────────────────────────
TICKER_REGEX   = re.compile(r"[\dA-Z]{4,6}\.KL")
TICKER_EXAMPLE = "1155.KL (Maybank)"
DATA_BACKEND   = "yahoo"
BENCHMARK_SYMBOL = "^KLSE"
INSTRUMENT_TYPE  = "listed_equity"   # fee_schedules resolution key

# Hard-blocked trading modes — long-only, daily-bar system (mirrors Gate 0's
# _filter_infeasible so the sandbox path matches the generated-idea path).
BLOCKED_MODES = [
    "short sell", "short-sell", "short-selling", "short selling", "sell short",
    "pairs trade", "pairs trading", "long/short", "long-short", "market neutral",
    "delta neutral", "spread trade", "arbitrage between", "options contract",
    "futures spread", "intraday", "scalp", "hft", "tick data",
]

# Feasibility scoring keyword lists (StrategyResearcher._compute_feasibility)
UNAVAILABLE_DATA_KEYWORDS = [
    "options", "futures contract", "otc", "dark pool",
    "tick data", "level 2", "order book", "forex", "fx rate",
]
EXOTIC_KEYWORDS = [
    "futures price", "options greeks", "cds spread", "credit default",
    "bond yield curve", "repo rate",
]

# ── Red/Blue + researcher market brief ────────────────────────────────────────
MARKET_BRIEF = """
BURSA MALAYSIA MARKET STRUCTURE — MUST KNOW:
- Settlement: T+2 (2 business days, effective 2019-04-29). Affects short-term strategies.
- Short-selling: Bursa operates regulated short-selling (RSS/IDSS) on an
  approved-securities list (~150 names), but this system uses no borrowed-stock
  execution. LONG-ONLY strategies only.
- Trading hours: 9:00-12:30 and 14:30-17:00 MYT. No after-hours.
- Lot size: 100 shares minimum. Affects small-cap liquidity.
- Foreign ownership: EPF owns ~15% of market. KWAP, PNB also large.
  Institutional flows are predictable around rebalancing periods.
- OPR sensitivity: Malaysian banking stocks are highly sensitive
  to BNM Overnight Policy Rate decisions.
- CPO correlation: Plantation stocks (Sime Darby, IOI, KLK) move
  strongly with Crude Palm Oil futures prices.
- Penny stocks: High retail speculation, pump-and-dump risk,
  very wide spreads. Strategies on stocks below RM0.50 are high risk.
- Circuit breakers: Stocks halt if they move >30% in a day.
- Stamp duty: 0.10% on buy side (remitted rate to 2028-07-12), capped at RM1,000. Real cost.
- GLC dynamics: Government-linked companies (Maybank, Tenaga,
  Petronas subsidiaries) have different dynamics — policy-driven.
"""

# Red-team attack lines specific to this market (appended to RED_SYSTEM).
RED_TEAM_ATTACKS = """You MUST specifically attack:
- T+2 settlement risk: does the strategy's holding period interact badly with T+2?
- Liquidity risk: can this be executed in 100-share lots without moving the price?
- EPF flow reversal risk: if EPF rebalances away, does the thesis collapse?
- OPR change risk: for banking strategies, how does a 25bp BNM rate change affect the thesis?
- Penny stock risk: is the ticker a low-liquidity or low-price stock with wide spreads?
- Feasibility: can a real retail or institutional investor in Malaysia actually execute this?"""

BLUE_DEFENSE_NOTES = """When defending, always address Bursa-specific mechanics directly:
- If T+2 is raised: explain how the holding period accommodates settlement.
- If liquidity is raised: cite the stock's average daily volume or lot-size adequacy.
- If EPF flows are raised: explain whether the thesis is EPF-dependent or independent.
- If OPR is raised: quantify the sensitivity and whether the strategy hedges rate risk."""

JUDGE_REJECT_RULE = ("Apply Bursa-specific judgment: reject any strategy that requires "
                     "short-selling unrestricted securities, relies on intraday execution, "
                     "or ignores T+2 settlement constraints.")

# Concentration risk: which sector the max_bank_pct-style limit watches.
CONCENTRATION_SECTOR = "Banking"

# ── Daemon scheduled jobs enabled for this market ─────────────────────────────
# None = all registered jobs run (Bursa is the full original set).
ENABLED_JOBS = None

# GateConfig field overrides for this market (none — defaults ARE Bursa).
GATE_OVERRIDES: dict = {}
