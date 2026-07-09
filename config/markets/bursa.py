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

# Bursa is long-only by design (short-selling heavily restricted, RSS/IDSS
# approved-list only — see MARKET_BRIEF below); crypto's ALLOW_SHORT=True
# (WS3, perpetuals) is the exception, not the default. Leverage/liquidation/
# funding are perp-only concepts with no Bursa equivalent.
ALLOW_SHORT = False
MAX_LEVERAGE = 1.0
DEFAULT_LEVERAGE = 1.0
LIQUIDATION_BUFFER = 0.0
FUNDING_INTERVAL_HOURS = None
AVG_FUNDING_RATE_PER_INTERVAL = 0.0

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


def funding_cost(position: float, funding_rate: float, notional: float) -> float:
    """No-op — funding is a perpetual-futures concept with no Bursa equity
    equivalent. Present only for interface parity with the crypto profile."""
    return 0.0


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

# ── Research content (KB hunt / alpha seeds) ─────────────────────────────────
# Moved verbatim from knowledge/ingestion/diversity_engine.py's ANGLES /
# ANGLE_KEYWORDS (2026-07-09) — same 9 taxonomy keys as kb_ingester.VALID_DOMAINS
# and family_quotas, only the per-market content differs.
RESEARCH_ANGLES = {
    "price_action": {
        "description": "Technical analysis, price momentum, chart patterns on Bursa Malaysia",
        "queries": [
            "price momentum Bursa Malaysia equities",
            "technical analysis KLSE stock returns anomalies",
            "moving average crossover ASEAN equity markets",
        ],
    },
    "fundamental": {
        "description": "Value investing, earnings quality, fundamental factors KLSE",
        "queries": [
            "value investing Bursa Malaysia fundamental factors",
            "earnings quality factor Malaysian equity returns",
            "price-to-book return on equity ASEAN stocks",
        ],
    },
    "event_driven": {
        "description": "Post-earnings drift, dividend capture, corporate events Bursa",
        "queries": [
            "post-earnings announcement drift Malaysia stocks",
            "dividend capture strategy emerging market equities",
            "corporate events stock returns ASEAN",
        ],
    },
    "institutional": {
        "description": "EPF flows, GLC ownership, institutional trading patterns Bursa",
        "queries": [
            "institutional ownership government-linked companies Malaysia stock returns",
            "pension fund investment impact equity prices ASEAN",
            "sovereign wealth fund trading equity market impact",
        ],
    },
    "macro": {
        "description": "OPR cycle, MYR macro impacts on sector returns Bursa Malaysia",
        "queries": [
            "interest rate cycle bank sector returns Malaysia",
            "monetary policy equity sector rotation emerging markets",
            "macroeconomic factors Malaysian stock market performance",
        ],
    },
    "commodity": {
        "description": "CPO price impact on plantation stocks, commodity equity linkages",
        "queries": [
            "palm oil price plantation stock returns Malaysia",
            "commodity equity linkage factor investing",
            "crude oil price energy sector stocks emerging markets",
        ],
    },
    "sector_rotation": {
        "description": "KLSE sector rotation, defensive vs cyclical, sector momentum",
        "queries": [
            "sector rotation strategy emerging market equities",
            "industry momentum returns ASEAN equities",
            "cyclical defensive sector switching Bursa Malaysia",
        ],
    },
    "behavioural": {
        "description": "Investor behaviour biases, market anomalies, sentiment KLSE",
        "queries": [
            "investor sentiment stock market anomalies Malaysia",
            "behavioural biases equity returns emerging markets",
            "market microstructure anomalies ASEAN equities",
        ],
    },
    "statistical_modelling": {
        "description": "Quantitative models: GARCH, HMM, factor models, ML, cointegration for KLSE",
        "queries": [
            "GARCH volatility model Bursa Malaysia equity",
            "hidden markov regime detection ASEAN stock market",
            "random matrix theory portfolio optimization emerging markets",
            "machine learning return prediction Malaysian stocks",
            "factor model Fama French KLSE",
        ],
    },
}

ANGLE_KEYWORDS = {
    "price_action": [
        "momentum", "mean reversion", "mean-reversion", "technical", "moving average",
        "rsi", "breakout", "trend", "macd", "bollinger", "price action", "chart pattern",
        "support", "resistance", "crossover", "oscillator",
    ],
    "fundamental": [
        "value", "earnings", "book value", "p/e", "p/b", "roe", "fundamental",
        "dividend yield", "revenue", "balance sheet", "cash flow", "quality", "valuation",
        "earnings quality", "price-to-book", "return on equity",
    ],
    "event_driven": [
        "pead", "earnings drift", "post-earnings", "dividend capture", "corporate event",
        "announcement", "earnings surprise", "ex-dividend", "rights issue", "bonus issue",
        "earnings announcement", "event study",
    ],
    "institutional": [
        "epf", "kwap", "institutional", "glc", "government-linked", "pension fund",
        "sovereign wealth", "foreign ownership", "msci", "passive fund", "index rebalancing",
        "institutional flows", "ownership structure",
    ],
    "macro": [
        "opr", "bank negara", "bnm", "interest rate", "monetary policy", "macroeconomic",
        "gdp", "inflation", "central bank", "rate cycle", "rate sensitivity", "nim",
        "macroeconomics", "economic cycle",
    ],
    "commodity": [
        "cpo", "palm oil", "crude oil", "commodity", "plantation", "energy sector",
        "tin", "rubber", "commodity equity", "commodity correlation", "resource",
        "crude palm oil", "plantation stock",
    ],
    "sector_rotation": [
        "sector rotation", "sector momentum", "industry momentum", "cyclical", "defensive",
        "banking sector", "telco", "utilities", "reit", "sector switching",
        "sector performance", "industry rotation",
    ],
    "behavioural": [
        "sentiment", "behavioural", "behavioral", "anomaly", "anomalies",
        "investor behaviour", "investor behavior", "bias", "microstructure",
        "calendar effect", "january effect", "overreaction", "herding",
        "market anomaly", "investor sentiment",
    ],
    "statistical_modelling": [
        "garch", "egarch", "arima", "volatility model", "time series",
        "hidden markov", "regime detection", "regime switching",
        "random matrix", "eigenvalue", "minimum spanning tree", "correlation clustering",
        "factor model", "fama french", "pca", "principal component", "ica",
        "machine learning", "regression", "bayesian", "kalman filter",
        "monte carlo", "cointegration", "stationarity", "unit root",
        "statistical arbitrage", "clustering algorithm",
    ],
}

# Research-hunter query-generation persona (moved verbatim from
# knowledge/ingestion/research_hunter.py's QUERY_SYSTEM).
RESEARCH_QUERY_PERSONA = (
    "You are a research librarian generating academic database search queries for "
    "quantitative equity research focused on Bursa Malaysia and ASEAN emerging markets."
)

# Alpha-seed digestion persona (moved verbatim from
# knowledge/ingestion/alpha_seeds.py's SYSTEM).
ALPHA_SEED_SYSTEM = (
    "You are a senior quant researcher specialising in Bursa Malaysia equity markets. "
    "You are comfortable with both discretionary and quantitative approaches including "
    "GARCH/ARIMA time series models, factor models (Fama-French, PCA), Hidden Markov "
    "regime detection, cointegration, Kalman filters, Monte Carlo simulation, Bayesian "
    "inference, machine learning applied to financial data, and statistical arbitrage. "
    "When extracting alpha from statistical modelling papers, translate the quantitative "
    "techniques into concrete, implementable KLSE strategies."
)

# Example data sources for the alpha-seed hypothesis JSON template.
DATA_SOURCES_EXAMPLE = ["Yahoo Finance", "Bursa announcements"]

# Relevance-check target phrase + 5-tier scale, shared verbatim by
# research_hunter._is_relevant and kb_ingester.relevance_check (previously
# two near-duplicate hardcoded copies of the same text).
RELEVANCE_TARGET = "Bursa Malaysia equity trading"
RELEVANCE_SCALE = """  0.00–0.20  irrelevant — completely wrong market or asset class
    Examples: Australian CFD trading, cryptocurrency, forex pairs,
    US options pricing, bond market mechanics, ML for cybersecurity

  0.20–0.40  generic — general finance, transferable concepts only
    Examples: General momentum theory, generic valuation frameworks,
    factor investing with no regional context, portfolio theory

  0.40–0.60  partial — emerging market or Asian market context
    Examples: ASEAN equity research, Southeast Asia fund flows,
    EM factor models, Asian market microstructure, China/India/HK equity

  0.60–0.80  relevant — Bursa Malaysia or Malaysian equity specific
    Examples: KLSE stock returns, Malaysian market anomalies,
    Bursa market microstructure, BNM policy effects, FBM KLCI factors

  0.80–1.00  direct — actionable KLSE intelligence
    Examples: Specific KLSE stock analysis, EPF flow studies,
    CPO-plantation correlation, GLC ownership effects, Bursa volatility"""

# ── Daemon scheduled jobs enabled for this market ─────────────────────────────
# None = all registered jobs run (Bursa is the full original set).
ENABLED_JOBS = None

# GateConfig field overrides for this market (none — defaults ARE Bursa).
GATE_OVERRIDES: dict = {}


# ── System Direction document (dashboard /api/system/direction) ────────────────
# Market-specific "north star" content, moved verbatim out of the endpoint so the
# endpoint is market-agnostic. The endpoint merges this with live KB / idea /
# spend numbers and derives research-angle coverage from RESEARCH_ANGLES.
DIRECTION_DOC = {
    "last_updated": "April 2026",
    "core_purpose": (
        "Find genuine, statistically robust alpha factors in Bursa Malaysia equity markets. "
        "Prove them cross-sectionally. Deploy them safely with human oversight at every "
        "capital decision point."
    ),
    "design_philosophy": (
        "Quality over quantity. 10 robust, well-validated strategies beats 300 hastily "
        "generated noise ideas. Every component must earn its place. The system should get "
        "smarter every day, not just bigger."
    ),
    "success_metrics": [
        {"rank": 1, "metric": "First idea reaches Stage 3 with IC > 0.05 across 15+ stocks"},
        {"rank": 2, "metric": "First idea completes 30-day paper trade with Sharpe >= 1.0"},
        {"rank": 3, "metric": "First live strategy deployed with positive alpha after costs"},
        {"rank": 4, "metric": "KB reaches 50 quality docs across all 9 research angles"},
        {"rank": 5, "metric": "Daily budget stays under $10 while pipeline processes meaningful ideas"},
    ],
    "constraints": [
        "Long-only strategies only (short-selling heavily restricted)",
        "T+2 settlement (effective 2019-04-29) — affects short-term strategy feasibility",
        "Minimum lot size: 100 shares (affects small-cap liquidity)",
        "Stamp duty: 0.10% remitted buy-side, capped RM1,000 (real cost)",
        "Brokerage: ~0.08% per side minimum",
        "Trading hours: 9:00-12:30 and 14:30-17:00 MYT only",
        "Circuit breakers: halt if stock moves >30% in a day",
        "EPF dominates: ~15% of market cap, rebalancing is predictable",
        "OPR sensitivity: banking stocks move with BNM rate decisions",
        "CPO correlation: plantation stocks follow palm oil futures",
    ],
    "transaction_costs": {
        "commission_pct": 0.08,
        "stamp_duty_pct": 0.10,
        "stamp_duty_cap_myr": 1000,
        "clearing_pct": 0.03,
        "clearing_cap_myr": 1000,
        "slippage": {"BLUE_CHIP": 0.05, "MID_CAP": 0.25, "SMALL_CAP": 0.75},
        "min_liquidity_myr": 500000,
        "settlement": "T+2",
    },
}
