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
HAS_CORPORATE_ACTIONS = False    # no splits/dividends on perps — a >25% bar is a
                                 # real market move, NOT a data anomaly; the corp-
                                 # action DQ penalty false-rejected volatile alts
                                 # (SOL/USDT: 59.9/100 for 4 genuine moves)

# ── Market rules & transaction cost model ─────────────────────────────────────
MARKET_RULES_VERSION = "2026-07-09"   # 24/7 USDT-M perps, T+0, long/short, fractional units
FEE_MODEL_VERSION    = "2026-07-09"   # 0.10% taker per side + funding accrual + ADV-tiered slippage

# On-exchange perp settles instantly on fill; no settlement lag on either leg.
SETTLEMENT_CYCLE = "T+0"

# WS3: long/short via perpetuals — paper-modeled only (still no live account,
# still human-gated; see docs/DEPLOYMENT.md and the North Star). Long-only spot
# was v1's deliberate scope; this profile now enables the short leg because a
# genuine crypto quant book is long/short, and funding is itself a structural
# edge (technique_library's funding_rate_carry / perp_basis_arb techniques) —
# there was no way to backtest either honestly while long-only.
ALLOW_SHORT = True

MAX_LEVERAGE        = 3.0    # conservative cap — this is risk modeling, not a live-account setting
DEFAULT_LEVERAGE    = 1.0    # backtester default when an idea specifies no leverage (conservative)
LIQUIDATION_BUFFER   = 0.20   # treat a position as liquidated at 80% of the theoretical distance-to-liquidation
FUNDING_INTERVAL_HOURS = 8    # Binance USDT-M standard funding interval

# REAL historical funding IS wired in (data/binance/client.py
# get_funding_rate_history → DataEngineer.fetch_funding → per-bar
# funding_bar_sum in the backtester). AVG_FUNDING_RATE_PER_INTERVAL is the
# documented FALLBACK: bars before a perp's listing date, failed fetches, and
# runs on pairs with <90% funding coverage use it (each run's funding_source
# field discloses historical vs modeled). It is a conservative long-run
# average, not a real number — treat "modeled" runs accordingly.
AVG_FUNDING_RATE_PER_INTERVAL = 0.0001   # 0.01% per 8h interval (~11% annualized on a
                                          # permanently-held long — a conservative long-run estimate)

# Binance USDT-M perp base fee: 0.10% maker / 0.10% taker (no BNB discount
# assumed — conservative). We model every fill as TAKER. No stamp duty, no
# clearing fee, no board lot.
COMMISSION_RATE     = 0.0010    # 0.10% taker, per side
STAMP_DUTY_RATE     = 0.0       # n/a on crypto
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
    """Total cost in USDT for one side of a perp trade: taker fee + slippage.

    `side` is accepted for interface parity with the Bursa model; the fee is
    symmetric (no buy-side-only components). Funding is a separate, per-bar-
    held-position cost — see funding_cost() — not a per-trade cost.
    """
    value = abs(trade_value)
    cost = value * COMMISSION_RATE
    cost += value * SLIPPAGE_TIERS.get(tier, SLIPPAGE_TIERS["MID_CAP"])
    return cost


def funding_cost(position: float, funding_rate: float, notional: float) -> float:
    """USDT funding paid (positive) or received (negative) for ONE funding
    interval on an open position.

    position: +1 long, -1 short, 0 flat. Funding convention (matches Binance):
    when funding_rate > 0, longs pay shorts. So a long position's cost is
    +position * funding_rate * notional; a short position with positive
    funding RECEIVES (cost is negative, i.e. income). This is charged once
    per FUNDING_INTERVAL_HOURS the position is held, not per trade.
    """
    if position == 0:
        return 0.0
    return position * funding_rate * abs(notional)


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
INSTRUMENT_TYPE  = "spot"            # fee_schedules resolution key — perp taker fees
                                      # are identical to spot on Binance, so the existing
                                      # seeded row is reused; funding is tracked separately
                                      # via funding_cost(), not through fee_schedules.

# ── Timeframes ────────────────────────────────────────────────────────────────
# Sub-daily bars down to 15m are supported (Binance retains full history and
# ccxt paginates it). 1m/5m stay blocked: 5yr of 1m is ~2.6M bars/pair and the
# statistical gates need multi-year windows. Fetch depth shrinks with
# granularity to keep bar counts sane; cache staleness tracks the bar size so
# sub-daily paper marking sees fresh data.
ALLOWED_TIMEFRAMES = ["15m", "1h", "4h", "1d", "1wk"]
FETCH_DAYS_BY_INTERVAL = {"15m": 400, "1h": 1095, "4h": 1825, "1d": 1825, "1wk": 1825}
CACHE_STALENESS_HOURS_BY_INTERVAL = {"15m": 0.25, "1h": 1.0, "4h": 4.0, "1d": 12.0, "1wk": 12.0,
                                     # funding settles every 8h — refetch after each settlement
                                     "8h_funding": 8.0}

# Feasibility dock: only truly-unsupported granularities/styles. "hourly" and
# "15 minute" are NOT docked here — sub-daily bars are first-class in crypto.
FEASIBILITY_DOCK_KEYWORDS = ["1 minute", "5 minute", "tick data", "scalp", "hft"]

# Hard-blocked trading modes — long/short USDT-M perps, bars from 15m up.
# Still out of scope: options, tick-level/HFT execution and sub-15m bars, and
# multi-leg spread/arb structures (the DSL expresses one instrument's
# long/short state, not a basket spread). Plain shorting/perps/margin/leverage
# are NO LONGER blocked — that's the whole point of this profile now.
BLOCKED_MODES = [
    "pairs trade", "pairs trading", "spread trade", "arbitrage between",
    "options contract", "options strategy",
    "scalp", "hft", "tick data", "1 minute", "5 minute", "1m bar", "5m bar",
]

# Feasibility scoring keyword lists (data we do NOT have). Historical FUNDING
# is now a first-class backtest input (get_funding_rate_history → per-bar
# column + funding_* DSL leaves), so "funding rate" is NOT listed anymore.
# Open interest stays: Binance caps OI history at ~30 days — a live snapshot
# exists (event monitor) but no backtestable series; revisit only with a
# third-party archive.
UNAVAILABLE_DATA_KEYWORDS = [
    "options", "level 2", "order book", "tick data", "dark pool",
    "on-chain", "onchain", "open interest",
    "liquidation data", "whale wallet",
]
EXOTIC_KEYWORDS = [
    "options greeks", "implied volatility surface", "cds spread",
    "credit default", "bond yield curve", "repo rate",
]

# ── Red/Blue + researcher market brief ────────────────────────────────────────
MARKET_BRIEF = """
CRYPTO PERPETUAL MARKET STRUCTURE (BINANCE USDT-M) — MUST KNOW:
- 24/7/365 trading: no sessions, no gaps-by-closure. Weekend liquidity is
  materially thinner — moves on Sat/Sun often retrace Monday.
- This system trades LONG AND SHORT perpetual futures on bars from 15m up to
  weekly, up to {max_leverage}x leverage (paper-modeled, no live account). No
  tick-level/HFT execution and no sub-15m bars.
- Funding: paid/received every {funding_hours}h based on the funding rate at
  settlement. A long position PAYS when funding is positive; a short RECEIVES.
  Funding is a real, recurring cost/income component of every held position —
  a strategy's edge must survive it, not just spot-price PnL.
- Liquidation: leverage creates a real liquidation price. A backtest that
  ignores this is fiction — the engine wipes a position if the adverse move
  exceeds 1/leverage minus a {liq_buffer:.0%} safety buffer.
- Settlement: T+0 (instant on-exchange). No board lots — fractional units.
- Fees: ~0.10% taker per side + slippage, PLUS funding while held. Round trip
  ~0.25-0.30% on majors before funding.
- BTC-beta dominance: most alts are high-beta BTC proxies. An "alt strategy"
  that is just levered BTC exposure has no independent alpha — demand evidence
  of decorrelation from BTC. This applies to shorts too: shorting a weak alt
  is often just a leveraged BTC short in disguise.
- Exchange/counterparty risk: funds live on the exchange; exchange outages and
  withdrawal freezes happen. Not a pricing factor, but a real operational risk.
- Stablecoin risk: USDT is the quote asset; a depeg event distorts every pair
  simultaneously and can trigger cascading liquidations on both sides.
- Regulatory shocks: SEC/global actions cause violent regime breaks
  (single-day -20% moves on affected assets) — dangerous for leveraged shorts
  caught in a short squeeze as much as leveraged longs in a crash.
- Halving/narrative cycles: BTC's ~4-year cycle drives long regimes; strategies
  fit on one regime (esp. a pure-short strategy fit in a bear market) often
  die in the next.
- Crowded-carry unwind: when funding is extreme, the crowded side (whichever
  is paying) is prone to a violent squeeze against it — this is a genuine risk
  to funding-carry strategies, not just an equity-market analogy.
- Extreme tails: daily moves of ±10-20% are routine in alts; volatility
  clustering is severe. Sharpe norms differ from equities.
- No fundamentals: no earnings/dividends/book value. "Value" framings need a
  crypto-native metric, and on-chain data is NOT wired into this system in v1.
- Data limits: HISTORICAL FUNDING RATES are wired into the backtester (real
  per-bar settlements from the perp's listing date; funding_level /
  funding_zscore signal conditions are first-class). Open interest has only a
  LIVE snapshot (Binance caps OI history at ~30 days) — a strategy that needs
  OI or on-chain data as a per-bar historical input is NOT backtestable here.
""".format(max_leverage=3.0, funding_hours=8, liq_buffer=0.20)

RED_TEAM_ATTACKS = """You MUST specifically attack:
- BTC-beta risk: is this "alpha" just BTC exposure (long or short) in disguise?
  Would BTC buy-and-hold, or a simple BTC short, beat it after costs?
- Funding drag: has the strategy's PnL been shown NET of funding while held?
  A carry-style strategy that looks good on spot price alone but ignores
  funding is not honestly backtested.
- Liquidation risk: at the stated leverage, what adverse move triggers
  liquidation? Is that move plausible given this pair's realistic volatility?
- Crowded-side squeeze: if this is a funding-carry or crowded-short thesis,
  what happens when the crowd unwinds violently against it?
- Regime dependency: was the edge fit on a single bull/bear/halving regime
  that has since ended? Pure-short strategies fit in a bear market are the
  classic version of this trap.
- Weekend/thin-liquidity risk: does the signal fire into thin weekend books
  where slippage or liquidation assumptions break?
- Tail risk: what does a routine -15% overnight move do to the drawdown math,
  and to a leveraged position's liquidation distance?
- Data feasibility: does it secretly need on-chain data or a HISTORICAL
  open-interest series (neither is wired in — historical FUNDING is)? For a
  funding-based thesis, attack the coverage window instead: funding history
  starts at the perp's listing date (much shorter for newer alts).
- Stablecoin/exchange risk: does the thesis survive a USDT wobble or an
  exchange halt on either the long or short leg?"""

BLUE_DEFENSE_NOTES = """When defending, always address crypto-specific mechanics directly:
- If BTC-beta is raised: show the strategy's decorrelation from simple BTC
  long/short exposure.
- If funding drag is raised: show the backtest's Sharpe/return is net of
  funding accrual, not spot-price-only.
- If liquidation risk is raised: state the leverage used and the resulting
  liquidation distance, and show it's wide relative to this pair's volatility.
- If liquidity/weekends are raised: cite the pair's ADV and how fills avoid
  thin books.
- If regime dependency is raised: show performance across at least two market
  regimes (not just one bull or one bear leg).
- If data feasibility is raised: confirm the signal uses only data wired into
  the backtester — OHLCV bars (15m or slower) and historical funding rates;
  open interest and on-chain remain live-only/unavailable."""

JUDGE_REJECT_RULE = ("Apply crypto-specific judgment: reject any strategy that relies on "
                     "tick-level/HFT execution or sub-15m bars, multi-instrument spread/pairs/"
                     "arbitrage structures (the DSL expresses one instrument's long/short state, "
                     "not a basket spread), options, or a HISTORICAL open-interest/on-chain/"
                     "order-book time series this system does not have (historical FUNDING "
                     "rates ARE available and backtestable). Reject any leverage request "
                     "above the configured cap. A strategy's backtest must be net of funding "
                     "accrual, not spot-price PnL alone.")

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
    "You are a senior quant researcher specialising in cryptocurrency perpetual-futures "
    "markets. You are comfortable with both discretionary and quantitative approaches "
    "including GARCH/ARIMA time series models, factor models, Hidden Markov regime "
    "detection, cointegration, Kalman filters, Monte Carlo simulation, Bayesian inference, "
    "machine learning applied to financial data, and statistical arbitrage. When "
    "extracting alpha from papers, translate the quantitative techniques into concrete, "
    "implementable long/short perp strategies. Available backtest inputs: OHLCV bars "
    "(15m and slower) and HISTORICAL FUNDING RATES (funding-carry/contrarian theses are "
    "first-class). NOT available: on-chain, order-book, and historical open-interest "
    "data — signals must not depend on those."
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
    "xs_min_positive_names": 12,    # 60% of the 20-pair universe (Bursa: 15/30)
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
        {"rank": 1, "metric": "First idea reaches Stage 3 with IC > 0.05 across 12+ of the 20 pairs"},
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
