"""Crypto (Binance spot) market profile.

Long-only SPOT, daily bars — deliberately mirrors the Bursa pipeline's core
assumptions so the entire gated pipeline (signal DSL, walk-forward, deflated
Sharpe, red/blue, paper trading) runs unchanged. No perps, no margin, no
funding-rate strategies in v1.

Data comes from the exchange's public OHLCV endpoints via ccxt
(data/binance/client.py); the exchange id is configurable (CRYPTO_EXCHANGE_ID)
because binance.com may be geo-restricted in some VPS regions — okx/kraken are
drop-in fallbacks for public market data.
"""
from __future__ import annotations

import os
import re

MODE            = "crypto"
MARKET          = "CRYPTO"
MARKET_CURRENCY = "USDT"
MARKET_TIMEZONE = "UTC"
MARKET_NAME     = "Crypto (Binance spot)"
UNIVERSE_NAME   = "CRYPTO_MAJORS"

# ── Universe: ~20 liquid USDT spot pairs ─────────────────────────────────────
# "sector" is a functional analog used by concentration limits + family quotas;
# "code" is the base asset (fills the bursa_code slot generically).
UNIVERSE_ASOF = "2026-07-09"

UNIVERSE = [
    {"symbol": "BTC/USDT",  "name": "Bitcoin",    "sector": "Store of Value",  "bursa_code": "BTC"},
    {"symbol": "ETH/USDT",  "name": "Ethereum",   "sector": "Smart Contract",  "bursa_code": "ETH"},
    {"symbol": "BNB/USDT",  "name": "BNB",        "sector": "Exchange",        "bursa_code": "BNB"},
    {"symbol": "SOL/USDT",  "name": "Solana",     "sector": "Smart Contract",  "bursa_code": "SOL"},
    {"symbol": "XRP/USDT",  "name": "XRP",        "sector": "Payments",        "bursa_code": "XRP"},
    {"symbol": "ADA/USDT",  "name": "Cardano",    "sector": "Smart Contract",  "bursa_code": "ADA"},
    {"symbol": "DOGE/USDT", "name": "Dogecoin",   "sector": "Meme",            "bursa_code": "DOGE"},
    {"symbol": "AVAX/USDT", "name": "Avalanche",  "sector": "Smart Contract",  "bursa_code": "AVAX"},
    {"symbol": "LINK/USDT", "name": "Chainlink",  "sector": "Oracle",          "bursa_code": "LINK"},
    {"symbol": "DOT/USDT",  "name": "Polkadot",   "sector": "Smart Contract",  "bursa_code": "DOT"},
    {"symbol": "LTC/USDT",  "name": "Litecoin",   "sector": "Payments",        "bursa_code": "LTC"},
    {"symbol": "UNI/USDT",  "name": "Uniswap",    "sector": "DeFi",            "bursa_code": "UNI"},
    {"symbol": "ATOM/USDT", "name": "Cosmos",     "sector": "Smart Contract",  "bursa_code": "ATOM"},
    {"symbol": "NEAR/USDT", "name": "NEAR",       "sector": "Smart Contract",  "bursa_code": "NEAR"},
    {"symbol": "APT/USDT",  "name": "Aptos",      "sector": "Smart Contract",  "bursa_code": "APT"},
    {"symbol": "ARB/USDT",  "name": "Arbitrum",   "sector": "Layer 2",         "bursa_code": "ARB"},
    {"symbol": "OP/USDT",   "name": "Optimism",   "sector": "Layer 2",         "bursa_code": "OP"},
    {"symbol": "FIL/USDT",  "name": "Filecoin",   "sector": "Storage",         "bursa_code": "FIL"},
    {"symbol": "INJ/USDT",  "name": "Injective",  "sector": "DeFi",            "bursa_code": "INJ"},
    {"symbol": "AAVE/USDT", "name": "Aave",       "sector": "DeFi",            "bursa_code": "AAVE"},
]

# ── Trading calendar: 24/7, no sessions ──────────────────────────────────────
MARKET_OPEN_HOUR      = 0
MARKET_CLOSE_HOUR     = 24
TRADING_DAYS_PER_YEAR = 365     # annualization uses √365 for daily bars
CALENDAR              = "daily"  # pd.date_range — weekends ARE trading days

# ── Market rules & transaction cost model ─────────────────────────────────────
MARKET_RULES_VERSION = "2026-07-09"   # 24/7 spot, T+0, long-only, fractional units
FEE_MODEL_VERSION    = "2026-07-09"   # 0.10% taker per side + ADV-tiered slippage

# On-exchange spot settles instantly on fill.
SETTLEMENT_CYCLE = "T+0"

# Binance spot base fee: 0.10% maker / 0.10% taker (no BNB discount assumed —
# conservative). We model every fill as TAKER. No stamp duty, no clearing fee,
# no board lot on crypto spot.
COMMISSION_RATE     = 0.0010    # 0.10% taker, per side
STAMP_DUTY_RATE     = 0.0       # n/a on crypto spot
STAMP_DUTY_CAP      = 0.0
STAMP_REMISSION_END = ""
CLEARING_RATE       = 0.0       # n/a
CLEARING_CAP        = 0.0
BOARD_LOT           = 0.0001    # min quantity step (fractional units allowed)

# Slippage by liquidity tier (fraction of trade value, per side), ADV in USDT.
SLIPPAGE_TIERS = {
    "BLUE_CHIP": 0.0003,   # majors, ADV ≥ $100M (BTC/ETH-class books)
    "MID_CAP":   0.0010,   # ADV ≥ $10M
    "SMALL_CAP": 0.0040,   # below $10M — thin books, weekend gaps
}
TIER_BLUE_CHIP = 100_000_000.0
TIER_MID_CAP   = 10_000_000.0

# Liquidity floor: reject strategies on pairs below this avg daily traded value
MIN_DAILY_VALUE = 1_000_000.0


def slippage_tier(avg_daily_value: float) -> str:
    """Classify a pair's liquidity tier from average daily traded value (USDT)."""
    if avg_daily_value >= TIER_BLUE_CHIP:
        return "BLUE_CHIP"
    if avg_daily_value >= TIER_MID_CAP:
        return "MID_CAP"
    return "SMALL_CAP"


def trade_cost(trade_value: float, side: str,
               tier: str = "BLUE_CHIP") -> float:
    """Total cost in USDT for one side of a spot trade: taker fee + slippage.

    `side` is accepted for interface parity with the Bursa model; crypto spot
    fees are symmetric (no buy-side-only components).
    """
    value = abs(trade_value)
    cost = value * COMMISSION_RATE
    cost += value * SLIPPAGE_TIERS.get(tier, SLIPPAGE_TIERS["MID_CAP"])
    return cost


def size_units(nav: float, price: float, alloc_pct: float) -> float:
    """Units to buy: alloc_pct of NAV, rounded DOWN to the 0.0001 quantity step.

    Fractional units are the norm on crypto spot (0.95 × $100k NAV at a
    $100k BTC price must yield 0.95 BTC, not 0)."""
    if price <= 0:
        return 0.0
    raw = (nav * alloc_pct) / price
    steps = int(raw / BOARD_LOT)
    return round(steps * BOARD_LOT, 8)


# ── Instruments / data ────────────────────────────────────────────────────────
TICKER_REGEX   = re.compile(r"[A-Z0-9]{2,10}/USDT")
TICKER_EXAMPLE = "BTC/USDT (Bitcoin)"
DATA_BACKEND   = "binance"
EXCHANGE_ID    = os.getenv("CRYPTO_EXCHANGE_ID", "binance")
BENCHMARK_SYMBOL = "BTC/USDT"
INSTRUMENT_TYPE  = "spot"            # fee_schedules resolution key

# Hard-blocked trading modes — long-only SPOT, daily bars. Everything
# derivative/levered/short is out of scope by design.
BLOCKED_MODES = [
    "short sell", "short-sell", "short-selling", "short selling", "sell short",
    "pairs trade", "pairs trading", "long/short", "long-short", "market neutral",
    "delta neutral", "spread trade", "arbitrage between",
    "perpetual", "perps", "futures", "margin", "leverage", "leveraged",
    "funding rate arbitrage", "options contract",
    "intraday", "scalp", "hft", "tick data",
]

# Feasibility scoring keyword lists (data we do NOT have in v1)
UNAVAILABLE_DATA_KEYWORDS = [
    "options", "level 2", "order book", "tick data", "dark pool",
    "on-chain", "onchain", "funding rate", "open interest",
    "perpetual", "liquidation data", "whale wallet",
]
EXOTIC_KEYWORDS = [
    "options greeks", "implied volatility surface", "cds spread",
    "credit default", "bond yield curve", "repo rate",
]

# ── Red/Blue + researcher market brief ────────────────────────────────────────
MARKET_BRIEF = """
CRYPTO SPOT MARKET STRUCTURE (BINANCE) — MUST KNOW:
- 24/7/365 trading: no sessions, no gaps-by-closure. Weekend liquidity is
  materially thinner — moves on Sat/Sun often retrace Monday.
- This system is LONG-ONLY SPOT on daily bars: no perps, no margin, no
  shorting, no funding-rate strategies. Reject anything that needs them.
- Settlement: T+0 (instant on-exchange). No board lots — fractional units.
- Fees: ~0.10% taker per side + slippage. Round trip ~0.25-0.30% on majors.
- BTC-beta dominance: most alts are high-beta BTC proxies. An "alt strategy"
  that is just levered BTC exposure has no independent alpha — demand evidence
  of decorrelation from BTC.
- Exchange/counterparty risk: funds live on the exchange; exchange outages and
  withdrawal freezes happen. Not a pricing factor, but a real operational risk.
- Stablecoin risk: USDT is the quote asset; a depeg event distorts every pair
  simultaneously.
- Regulatory shocks: SEC/global actions cause violent regime breaks
  (single-day -20% moves on affected assets).
- Halving/narrative cycles: BTC's ~4-year cycle drives long regimes; strategies
  fit on one regime often die in the next.
- Extreme tails: daily moves of ±10-20% are routine in alts; volatility
  clustering is severe. Sharpe norms differ from equities.
- No fundamentals: no earnings/dividends/book value. "Value" framings need a
  crypto-native metric, and on-chain data is NOT wired into this system in v1.
"""

RED_TEAM_ATTACKS = """You MUST specifically attack:
- BTC-beta risk: is this "alpha" just BTC exposure in disguise? Would BTC
  buy-and-hold beat it after costs?
- Regime dependency: was the edge fit on a single bull/bear regime (halving
  cycle) that has since ended?
- Weekend/thin-liquidity risk: does the signal fire into thin weekend books
  where slippage assumptions break?
- Tail risk: what does a routine -15% overnight move do to the drawdown math?
- Data feasibility: does it secretly need on-chain, funding, or order-book data
  that this system does not have?
- Stablecoin/exchange risk: does the thesis survive a USDT wobble or an
  exchange halt?"""

BLUE_DEFENSE_NOTES = """When defending, always address crypto-specific mechanics directly:
- If BTC-beta is raised: show the strategy's decorrelation from simple BTC exposure.
- If liquidity/weekends are raised: cite the pair's ADV and how fills avoid thin books.
- If regime dependency is raised: show performance across at least two market regimes.
- If data feasibility is raised: confirm the signal uses only daily OHLCV available here."""

JUDGE_REJECT_RULE = ("Apply crypto-specific judgment: reject any strategy that requires "
                     "shorting, perpetuals, margin, or leverage; relies on intraday "
                     "execution; or depends on data this system lacks (on-chain, funding "
                     "rates, order books).")

# Concentration risk: the sector analog whose over-weight the risk monitor flags.
CONCENTRATION_SECTOR = "Smart Contract"

# ── Research content (KB hunt / alpha seeds) ─────────────────────────────────
# Same 9 taxonomy keys as bursa.py (kb_ingester.VALID_DOMAINS / family_quotas
# share this taxonomy across markets) — content authored for crypto spot.
# "commodity" has no literal equivalent in crypto (no CPO/plantation-style
# equity-commodity linkage); reinterpreted as BTC-dominance / altcoin-BTC
# correlation regime, which plays the same "one external price series drives a
# basket of related instruments" role CPO plays for Bursa plantation stocks.
RESEARCH_ANGLES = {
    "price_action": {
        "description": "Technical analysis, momentum, chart patterns on crypto spot markets",
        "queries": [
            "price momentum cryptocurrency spot returns",
            "technical analysis bitcoin ethereum anomalies",
            "moving average crossover cryptocurrency markets",
        ],
    },
    "fundamental": {
        "description": "Tokenomics, network fundamentals, supply schedules as crypto value factors",
        "queries": [
            "tokenomics supply schedule cryptocurrency valuation",
            "network value active addresses crypto fundamental factor",
            "total value locked TVL blockchain protocol valuation",
        ],
    },
    "event_driven": {
        "description": "Halvings, token unlocks, exchange listings, protocol upgrades",
        "queries": [
            "bitcoin halving cycle price impact returns",
            "token unlock vesting schedule price impact",
            "exchange listing announcement cryptocurrency returns",
        ],
    },
    "institutional": {
        "description": "ETF flows, whale-wallet accumulation, exchange-reserve flows",
        "queries": [
            "bitcoin ETF flow impact price returns",
            "whale wallet accumulation cryptocurrency price impact",
            "exchange reserve outflow bitcoin price signal",
        ],
    },
    "macro": {
        "description": "Fed policy, DXY, risk-on/risk-off regime impact on crypto returns",
        "queries": [
            "federal reserve interest rate cryptocurrency market impact",
            "dollar index DXY bitcoin correlation macro regime",
            "risk asset correlation cryptocurrency equity markets",
        ],
    },
    "commodity": {
        "description": "BTC-dominance and altcoin-BTC correlation regime (crypto's commodity-linkage analog)",
        "queries": [
            "bitcoin dominance altcoin correlation regime",
            "altcoin beta bitcoin price co-movement",
            "cryptocurrency market cycle rotation dominance",
        ],
    },
    "sector_rotation": {
        "description": "Rotation between L1s, DeFi, gaming/NFT, and meme-coin narrative sectors",
        "queries": [
            "sector rotation layer1 defi cryptocurrency narrative",
            "narrative cycle cryptocurrency sector momentum",
            "defi layer2 token category performance rotation",
        ],
    },
    "behavioural": {
        "description": "Crypto sentiment, fear-greed index, herding and retail anomalies",
        "queries": [
            "investor sentiment fear greed index cryptocurrency returns",
            "herding behaviour cryptocurrency retail trading",
            "social media sentiment bitcoin price prediction",
        ],
    },
    "statistical_modelling": {
        "description": "Quantitative models: GARCH, HMM, factor models, ML, cointegration for crypto",
        "queries": [
            "GARCH volatility model cryptocurrency bitcoin",
            "hidden markov regime detection cryptocurrency market",
            "random matrix theory portfolio optimization digital assets",
            "machine learning return prediction cryptocurrency",
            "factor model cryptocurrency cross-sectional returns",
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
        "tokenomics", "token supply", "circulating supply", "network value",
        "active addresses", "total value locked", "tvl", "protocol revenue",
        "staking yield", "burn rate", "emission schedule", "fundamental",
    ],
    "event_driven": [
        "halving", "token unlock", "vesting", "exchange listing", "delisting",
        "protocol upgrade", "hard fork", "airdrop", "mainnet launch",
        "governance vote", "event study",
    ],
    "institutional": [
        "etf", "whale wallet", "exchange reserve", "institutional", "custody",
        "grayscale", "spot etf", "futures open interest", "institutional flows",
        "on-chain accumulation",
    ],
    "macro": [
        "federal reserve", "interest rate", "dxy", "dollar index", "monetary policy",
        "risk-on", "risk-off", "macro regime", "inflation", "quantitative easing",
        "macroeconomics", "economic cycle",
    ],
    "commodity": [
        "bitcoin dominance", "btc dominance", "altcoin correlation", "altcoin beta",
        "alt season", "market cycle", "rotation", "dominance chart",
    ],
    "sector_rotation": [
        "sector rotation", "narrative cycle", "layer 1", "layer1", "defi", "gamefi",
        "nft", "meme coin", "sector momentum", "category performance",
    ],
    "behavioural": [
        "sentiment", "fear and greed", "fear greed index", "herding", "anomaly",
        "anomalies", "social media sentiment", "retail trading", "overreaction",
        "market anomaly", "investor sentiment",
    ],
    "statistical_modelling": [
        "garch", "egarch", "arima", "volatility model", "time series",
        "hidden markov", "regime detection", "regime switching",
        "random matrix", "eigenvalue", "minimum spanning tree", "correlation clustering",
        "factor model", "pca", "principal component", "ica",
        "machine learning", "regression", "bayesian", "kalman filter",
        "monte carlo", "cointegration", "stationarity", "unit root",
        "statistical arbitrage", "clustering algorithm",
    ],
}

RESEARCH_QUERY_PERSONA = (
    "You are a research librarian generating academic database search queries for "
    "quantitative cryptocurrency spot-market research focused on major digital assets "
    "(BTC, ETH, and large-cap alts traded on centralized exchanges)."
)

ALPHA_SEED_SYSTEM = (
    "You are a senior quant researcher specialising in cryptocurrency spot markets. "
    "You are comfortable with both discretionary and quantitative approaches including "
    "GARCH/ARIMA time series models, factor models, Hidden Markov regime detection, "
    "cointegration, Kalman filters, Monte Carlo simulation, Bayesian inference, machine "
    "learning applied to financial data, and statistical arbitrage. When extracting alpha "
    "from statistical modelling papers, translate the quantitative techniques into "
    "concrete, implementable long-only spot strategies — this system has no on-chain, "
    "funding-rate, or order-book data in v1, so extracted signals must be expressible "
    "from daily OHLCV alone."
)

DATA_SOURCES_EXAMPLE = ["Binance OHLCV", "Exchange announcements"]

RELEVANCE_TARGET = "crypto spot trading"
RELEVANCE_SCALE = """  0.00–0.20  irrelevant — completely wrong market or asset class
    Examples: equity/stock market research, forex pairs, bond mechanics,
    real estate, ML for cybersecurity, non-financial content

  0.20–0.40  generic — general finance, transferable concepts only
    Examples: General momentum theory, generic valuation frameworks,
    factor investing with no crypto/digital-asset context, portfolio theory

  0.40–0.60  partial — blockchain/digital-asset context but not spot-trading specific
    Examples: general blockchain technology, NFT/gaming ecosystem analysis,
    DeFi protocol mechanics without a trading angle, crypto regulation

  0.60–0.80  relevant — cryptocurrency spot-market trading specific
    Examples: bitcoin/altcoin return studies, crypto market microstructure,
    exchange trading dynamics, crypto factor models

  0.80–1.00  direct — actionable spot-trading intelligence
    Examples: specific BTC/ETH/altcoin price analysis, on-chain flow studies
    with a trading signal, halving-cycle studies, dominance/rotation regimes"""

# ── Daemon scheduled jobs enabled for this market ─────────────────────────────
# Allowlist enforced inside ResearchDaemon._job_due(). Bursa-specific scrapers/
# monitors (klse_refresh, screener_ideas via the KLSE fundamental scanner,
# cpo_daily, analyst_monitor) have no crypto counterpart in v1 and never fire
# in this container.
#
# kb_hunt / alpha_seeds were disabled here (2026-07-09) after a live deploy
# showed the crypto daemon ingesting Bursa-Malaysia content into its own KB —
# DiversityEngine's research angles and ResearchHunter/KBIngester/AlphaSeedGenerator's
# relevance-rating prompts were hardcoded to Bursa regardless of MARKET_MODE.
# Fixed same day: RESEARCH_ANGLES / ANGLE_KEYWORDS / RESEARCH_QUERY_PERSONA /
# ALPHA_SEED_SYSTEM / RELEVANCE_TARGET / RELEVANCE_SCALE / DATA_SOURCES_EXAMPLE
# above are now crypto-authored profile fields consumed by all four modules —
# re-enabled now that they're market-aware.
ENABLED_JOBS = {
    "morning_briefing",
    "kb_hunt",
    "alpha_seeds",
    "graph_maintain",
    "vault_export",
    "funnel_report",
    "db_maintenance",
}

# GateConfig overrides for this market: crypto Sharpe/vol norms differ and the
# universe is one asset class, so single-name concentration is looser while the
# max-drawdown tolerance is slightly wider. Everything else inherits defaults.
GATE_OVERRIDES: dict = {
    "stage3_max_drawdown":  0.35,   # ±10-20% daily tails; 25% DD trips constantly
    "stage4a_max_drawdown": 0.30,
    "max_single_name_pct":  0.30,   # BTC/ETH legitimately dominate a book
    "max_sector_pct":       0.50,   # "Smart Contract" analog covers half the universe
}


# ── System Direction document (dashboard /api/system/direction) ────────────────
# Market-specific "north star" content. The endpoint merges this with live KB /
# idea / spend numbers and derives the research-angle coverage from
# RESEARCH_ANGLES above. Kept honest to current capability: long/short perp
# funding + liquidation constraints are added when Workstream 3 lands.
DIRECTION_DOC = {
    "last_updated": "July 2026",
    "core_purpose": (
        "Find genuine, statistically robust alpha in liquid crypto markets. Prove it "
        "cross-sectionally across majors, survive 24/7 volatility and funding costs, and "
        "paper-trade every strategy before any capital — human oversight at each step."
    ),
    "design_philosophy": (
        "Quality over quantity. A handful of robust, well-validated strategies beats hundreds "
        "of noise ideas. Crypto is faster and rougher than equities — the gates must be "
        "stricter, not looser."
    ),
    "success_metrics": [
        {"rank": 1, "metric": "First idea reaches Stage 3 with IC > 0.05 across 15+ liquid pairs"},
        {"rank": 2, "metric": "First idea completes a 30-day paper trade with Sharpe >= 1.0 after funding + fees"},
        {"rank": 3, "metric": "First strategy paper-proven long AND short across a regime shift"},
        {"rank": 4, "metric": "KB reaches 50 quality crypto-quant docs across all 9 research angles"},
        {"rank": 5, "metric": "Daily budget stays under $10 while the pipeline processes meaningful ideas"},
    ],
    "constraints": [
        "24/7/365 market — no sessions, no closing gaps; weekend liquidity is materially thinner",
        "T+0 settlement (instant on-exchange); fractional sizing, no board lots",
        "Taker fees ~0.10% per side + slippage — round-trip ~0.25-0.30% on majors",
        "BTC-beta dominates: most alts are high-beta BTC proxies — demand independent alpha",
        "High volatility: ±10-20% daily tails are normal; drawdown tolerance is wider by design",
        "Stablecoin depeg + exchange counterparty risk are real tail risks",
        "Paper-only, human-gated to live — no automated capital deployment",
    ],
    "transaction_costs": {
        "taker_pct": 0.10,
        "maker_pct": 0.10,
        "stamp_duty_pct": 0.0,
        "clearing_pct": 0.0,
        "slippage": {"BLUE_CHIP": 0.05, "MID_CAP": 0.15, "SMALL_CAP": 0.40},
        "min_liquidity_usdt": 1_000_000,
        "settlement": "T+0",
    },
}
