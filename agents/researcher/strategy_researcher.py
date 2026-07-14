"""
StrategyResearcher — Bursa Malaysia equity alpha research agent.

Scope: Bursa Malaysia / FBM KLCI equities ONLY.
Trade vehicles: individual KLSE stocks identified by Yahoo Finance .KL tickers.
Nothing in this module references, generates, or evaluates foreign-exchange,
currency pairs, forex instruments, or any non-equity asset class.
"""
import json
import re
import logging
from datetime import datetime
from agents.base_agent import BaseAgent, ClaudeJSONError
from config.settings import (
    MODEL_MAIN, MODEL_FAST, GATE_CONFIG,
    KLCI_STOCKS, KLCI_BY_SYMBOL, KLCI_SECTORS,
    TICKER_REGEX, DEFAULT_SYMBOLS,
    UNAVAILABLE_DATA_KEYWORDS, EXOTIC_KEYWORDS,
    MARKET_MODE, MARKET_NAME, MARKET_BRIEF, TICKER_EXAMPLE,
    ALLOW_SHORT,
)
from data.database import db_session

logger = logging.getLogger(__name__)

# ── Full universe reference embedded in every prompt ─────────────────────────
_UNIVERSE_FULL = "\n".join(
    f"  {s['symbol']:12s} {s['name']:35s} ({s['sector']})"
    for s in KLCI_STOCKS
)

_UNIVERSE_BRIEF = " | ".join(
    f"{s['symbol']} {s['name']}" for s in KLCI_STOCKS[:10]
) + f" … +{len(KLCI_STOCKS)-10} more"

# ── System prompt — equity-only, zero FX leakage ─────────────────────────────
SYSTEM = """YOU ARE A BURSA MALAYSIA SPECIALIST.
Every idea you generate MUST:
1. Trade stocks listed on Bursa Malaysia (KLSE)
2. Use ONLY signals derivable from daily OHLCV price and volume data via Yahoo Finance
3. Be implementable by a retail investor with a standard Malaysian brokerage account
4. Reference Malaysian market microstructure (EPF flows, CPO prices, OPR, Bursa rules)

NEVER generate ideas involving:
- US, European, or other non-Malaysian markets
- Macroeconomic indicators requiring external databases (Fed rates, AQI, satellite data)
- Financial statement ratios (P/B, ROE, P/E, DER) unless explicitly using them as a
  SCREENING filter with Yahoo Finance fundamental data
- Machine learning models requiring training infrastructure (HMM, neural networks, SVM)
- Alternative data sources (news sentiment, social media, satellite imagery)
- Commodity prices as primary signals (CPO is acceptable as a secondary context)

════════════════════════════════════════════════════════════════

You are an elite quantitative equity researcher specialising in Bursa Malaysia (KLSE) markets.
You generate, analyse, and score strategies that trade individual Malaysian listed stocks only.
You have zero knowledge of foreign-exchange trading and NEVER produce strategies involving
currency pairs, forex instruments, or spot FX.

BURSA MALAYSIA MARKET MICROSTRUCTURE:
• Trading hours:   09:00–12:30, 14:30–17:00 MYT (UTC+8), Mon–Fri
• Settlement:      T+2 (Bursa CDS), no short-selling for most stocks
• Transaction costs: ~0.25% round trip (brokerage 0.08–0.42% + stamp duty 0.10% remitted)
• Key indices:     FBM KLCI (30 large-caps), FBM70, FBM Small Cap
• Institutional:   EPF (~15% AUM), KWAP, PNB, Permodalan Nasional, foreign funds
• Key sectors:     Banking, Plantation, Telco, Healthcare, Technology, Utilities
• Currency:        MYR — sensitive to USD/MYR rate and CPO/crude commodity prices
• Earnings seasons: February, May, August, November (quarterly reporting)
• Data available:  Yahoo Finance .KL daily OHLCV, dividends, basic fundamentals

Generate ideas SPECIFICALLY implementable on Bursa Malaysia using Yahoo Finance data only.

════════════════════════════════════════════════════════════════
ABSOLUTE CONSTRAINT — READ BEFORE DOING ANYTHING ELSE
════════════════════════════════════════════════════════════════
• Every strategy you produce MUST trade a single Bursa Malaysia stock or a
  basket of Bursa Malaysia stocks.
• The "ticker" field MUST be a Yahoo Finance .KL symbol such as 1155.KL,
  1295.KL, 5347.KL — NEVER a currency pair like EUR_USD or USD_JPY.
• If a prompt somehow references FX, ignore that aspect and produce a
  KLSE equity strategy instead.
• Sector names must come from Bursa Malaysia's official sectors:
  Banking, Utilities, Telecoms, Plantations, Healthcare, Materials,
  Consumer Staples, Consumer Disc., Energy, Transportation,
  Construction, Industrial, Technology, Real Estate.
════════════════════════════════════════════════════════════════

FBM KLCI UNIVERSE (Yahoo Finance tickers, all end in .KL):
{universe}

MARKET MICROSTRUCTURE — BURSA MALAYSIA:
• Exchange:        Bursa Malaysia, Kuala Lumpur
• Index:           FBM KLCI (market-cap weighted, 30 constituents)
• Currency:        Malaysian Ringgit (MYR) — prices quoted in MYR, NOT traded as FX
• Trading hours:   09:00–12:30, 14:30–17:00 MYT (Mon–Fri)
• Settlement:      T+2 (Bursa Depository)
• Lot size:        100 shares minimum (1 board lot)
• Stamp duty:      0.10% remitted per contract (max RM1,000) on purchase
• Brokerage:       typically 0.10%–0.42% per side
• Short-selling:   RESTRICTED — only Approved Securities list, uptick rule applies
• Circuit breaker: ±30% from reference price intraday halt
• Reporting:       Quarterly (Feb/May/Aug/Nov earnings seasons)

EQUITY FACTORS RELEVANT TO KLSE:
Technical:
  • Price momentum (1-month, 3-month, 12-minus-1 month)
  • Moving-average crossovers (20/50 SMA, 50/200 EMA)
  • RSI mean-reversion (oversold <30, overbought >70)
  • Bollinger band squeeze and breakout
  • MACD signal-line crossover
  • Volume surge (2× 20-day avg) as confirmation
  • 52-week high/low breakout

Fundamental:
  • Price-to-Earnings (P/E): KLCI average ~14–18×; value below 10×
  • Price-to-Book (P/B): book value anchor especially for banks
  • Dividend Yield: high-yield KLCI stocks average 4–6%
  • Return on Equity (ROE): quality proxy, strong >15%
  • Earnings growth rate (YoY EPS change)
  • Net profit margin trend
  • Debt/Equity ratio (financial health)
  • Free cash flow yield

Thematic / Event-driven:
  • Crude Palm Oil (CPO) price correlation → Plantation stocks
  • Aluminium LME price → Press Metal (8869.KL)
  • Overnight Policy Rate (OPR) cycle → Banking sector NIM
  • EPF/KWAP rebalancing flows → large-cap GLCs
  • MSCI EM index review → foreign flow events
  • Post-earnings drift (PEAD) on quarterly beats
  • Dividend capture: go long 2 weeks before ex-date, exit on ex-date
  • GLC privatisation / divestiture speculation

WHAT A GOOD KLSE STRATEGY LOOKS LIKE:
- Trades a specific stock (e.g., 1155.KL Maybank) or a sector basket
- Entry based on measurable, reproducible conditions (RSI level, MA crossover,
  EPS beat, CPO price threshold)
- Exit based on price target, stop-loss %, time stop, or reverse signal
- Holding period appropriate for Bursa liquidity (days to months)
- Acknowledges Bursa-specific constraints (lot size, stamp duty, short restrictions)

════════════════════════════════════════════════════════════════
DATA CONSTRAINTS — AVAILABLE DATA SOURCES
════════════════════════════════════════════════════════════════
AVAILABLE — only use these in factor_formula:

Yahoo Finance .KL provides:
  • Daily OHLCV prices (up to 10 years history)
  • Dividend history and ex-dividend dates
  • Basic fundamentals: PE ratio, market cap,
    shares outstanding, book value
  • Volume and average volume
  • 52-week high/low

KLSE Screener provides:
  • Dividend yield, payout ratio
  • ROE, ROA, earnings history
  • Sector classification

Bursa Malaysia announcements provide:
  • Corporate actions, dividend declarations
  • Quarterly results (raw numbers, not consensus)

NOT AVAILABLE — NEVER use these in factor_formula:
  • Bloomberg or Refinitiv consensus estimates
  • Analyst EPS forecasts or SUE calculations
  • Real-time or intraday price data
  • Options data or implied volatility
  • Short interest data
  • Institutional ownership filings
  • Foreign flow data (not real-time)

Every factor_formula MUST be computable using ONLY
the available data sources listed above.
════════════════════════════════════════════════════════════════
""".format(universe=_UNIVERSE_FULL)


# ── Adversarial Gate 0 system prompt — intentionally skeptical ───────────────
GATE0_SYSTEM = """You are a skeptical quantitative researcher at a prop trading firm whose job
is to REJECT weak ideas. You are actively looking for reasons this strategy will NOT work on
Bursa Malaysia. Be demanding — most ideas should fail this gate. Your default stance is rejection.

You know Bursa Malaysia intimately: EPF dominance creates predictable flows that front-runners
exploit; GLC concentration means events move whole sectors; T+2 settlement makes very short-term
strategies expensive; approved-list restrictions mean long-only is the only viable approach for
most stocks; Yahoo Finance .KL data has quality issues (dividend adjustments, split gaps, stale
fundamentals). Score with deep scepticism and give low scores unless the edge is compelling."""


# Timeframe rule for generation prompts — Bursa renders byte-identical to the
# historical "MUST be 1d" wording; crypto lists the profile's allowed set.
from config.settings import ALLOWED_TIMEFRAMES as _ALLOWED_TFS
if MARKET_MODE == "bursa":
    _TIMEFRAME_RULE = '"timeframe" MUST be "1d" (daily bars — KLSE primary timeframe).'
    _TIMEFRAME_JSON = '"1d"'
else:
    _TIMEFRAME_RULE = (f'"timeframe" MUST be one of {"/".join(_ALLOWED_TFS)} — default "1d"; '
                       'use 15m/1h/4h ONLY when the thesis is explicitly about fast '
                       'mean-reversion or short-horizon effects.')
    _TIMEFRAME_JSON = '"1d (or 15m/1h/4h/1wk when the thesis needs it)"'


# ── Crypto mode: swap the generation prompts, leave Bursa's byte-identical ────
# The Bursa SYSTEM/GATE0_SYSTEM above are this pipeline's most battle-tested
# prompts — they are never edited for dual-market support. In crypto mode the
# process simply uses parallel prompts built from the crypto profile.
if MARKET_MODE != "bursa":
    SYSTEM = f"""YOU ARE A CRYPTO PERPETUALS MARKET SPECIALIST (BINANCE USDT-M).
Every idea you generate MUST:
1. Trade perpetual futures quoted in USDT on the exchange (e.g. BTC/USDT)
2. Use ONLY signals derivable from OHLCV price and volume bars, 15m to weekly (plus the live
   funding-rate/open-interest event snapshots this system already monitors — NOT a
   historical funding/OI time series, which is not backtestable here)
3. Be LONG OR SHORT (this system trades both directions on perps, up to the configured
   leverage cap) — state the intended leverage and whether the thesis needs funding income,
   funding cost tolerance, or neither
4. Reference crypto market structure (BTC-beta, weekend liquidity, halving cycles, regime
   shifts, funding dynamics, liquidation risk)

NEVER generate ideas involving:
- Options, or any multi-leg spread/pairs/arbitrage structure (the DSL expresses one
  instrument's long/short state, not a basket spread)
- On-chain data, a HISTORICAL funding-rate/open-interest time series, order books, or
  whale-wallet tracking (NOT available for backtesting in this system)
- Tick-level execution, scalping, or HFT — bars from 15m up to weekly only
- Machine learning models requiring training infrastructure
- News/social sentiment feeds
- Leverage above the configured cap

{MARKET_BRIEF}

TRADABLE UNIVERSE ({{n}} liquid USDT perpetuals):
{{universe}}

WHAT A GOOD CRYPTO PERP STRATEGY LOOKS LIKE:
- Trades a specific pair (e.g. BTC/USDT) or a small basket from the universe, long or short
- Entry from measurable bar-based conditions (MA crossover, RSI level, breakout,
  volume surge, relative strength vs BTC) — a short thesis needs the same rigor as a long
- Exit via price target, stop-loss %, time stop, or reverse signal
- Holding period of days to months (24/7 market, but signals are daily)
- Costs acknowledged: ~0.10% taker per side + slippage + funding while held
- States how the edge differs from simple BTC long/short exposure, and — if leveraged —
  states the leverage and the resulting liquidation distance

Every factor_formula MUST be computable from OHLCV bars (15m to weekly) alone.""".replace(
        "{n}", str(len(KLCI_STOCKS))).replace("{universe}", _UNIVERSE_FULL)

    GATE0_SYSTEM = f"""You are a skeptical quantitative researcher at a crypto fund whose job
is to REJECT weak ideas. You are actively looking for reasons this strategy will NOT work in
crypto perpetual markets. Be demanding — most ideas should fail this gate. Your default stance
is rejection.

You know crypto intimately: most alt "alpha" (long OR short) is disguised BTC beta; edges fit
on one halving-cycle regime die in the next; weekend books are thin and slippage assumptions
break; ±15% overnight moves are routine; exchange and stablecoin risk are real; funding is a
real recurring cost/income that a carry thesis must survive, and leverage creates genuine
liquidation risk that a backtest must account for. This system trades long/short perps on
daily OHLCV — anything needing options, a historical funding/OI series, on-chain data,
multi-leg spreads, or intraday data is automatically infeasible. Score with deep scepticism
and give low scores unless the edge is compelling and decorrelated from simple BTC exposure."""


class StrategyResearcher(BaseAgent):
    name = "StrategyResearcher"
    description = "Bursa Malaysia equity alpha generation, Gate 0 screening, Stage 1 deep research"
    default_model = MODEL_MAIN

    # ── Validation ────────────────────────────────────────────────────────────

    @staticmethod
    def _is_equity_ticker(ticker: str) -> bool:
        """Return True only if ticker matches the active market's format
        (Bursa: 1155.KL / 5235SS.KL; crypto: BTC/USDT)."""
        if not ticker:
            return False
        return bool(TICKER_REGEX.fullmatch(ticker.strip()))

    @staticmethod
    def _reject_if_fx(idea: dict) -> bool:
        """Return True if the idea contains FX contamination."""
        fx_patterns = [
            r'\b(EUR|GBP|USD|JPY|AUD|CAD|NZD|CHF|CNY|SGD)\s*[/_]\s*'
            r'(EUR|GBP|USD|JPY|AUD|CAD|NZD|CHF|CNY|SGD)\b',
            r'\bforex\b', r'\bFX\b', r'\bcurrency pair\b',
            r'\bspot rate\b', r'\bpip\b',
        ]
        blob = json.dumps(idea).upper()
        for pat in fx_patterns:
            if re.search(pat, blob, re.IGNORECASE):
                return True
        return False

    # ── Pre-save infeasibility filter ─────────────────────────────────────────

    def _filter_infeasible(self, ideas: list) -> list:
        """Filter out structurally infeasible ideas before saving to DB.

        Applied after generate_ideas() and before save_idea() to prevent
        garbage from entering the pipeline regardless of the generation path
        (/generate command, /screen command, or daemon auto-generation).

        Checks:
          1. ticker is valid .KL format or a plausible sector name
          2. hypothesis does not contain infeasible trading modes
          3. factor_formula is non-trivial (> 20 chars)
          4. novelty_score >= 0.5 AND logic_score >= 0.6

        Returns the filtered list and logs every rejection.
        """
        _INFEASIBLE_PHRASES = [
            "short sell", "short-sell", "pairs trade", "pairs trading",
            "long/short", "long-short", "options contract", "futures spread",
            "arbitrage between", "delta neutral", "market neutral", "spread trade",
        ]
        valid = []
        for idea in ideas:
            title   = idea.get("title", "")
            ticker  = idea.get("ticker", "")
            hypo    = (idea.get("hypothesis", "") or "").lower()
            formula = (idea.get("factor_formula", "") or "")

            # Check 1: valid .KL ticker (single or comma-separated) or plausible sector
            primary = ticker.split(",")[0].strip()
            if not self._is_equity_ticker(primary):
                sector = (idea.get("sector", "") or "").strip()
                if not sector:
                    self.log_daemon(
                        "INFO",
                        f"generate_ideas: filtered '{title[:50]}' (invalid ticker: {ticker})",
                    )
                    continue

            # Check 2: no infeasible trading modes
            infeasible = next((p for p in _INFEASIBLE_PHRASES if p in hypo), None)
            if infeasible:
                self.log_daemon(
                    "INFO",
                    f"generate_ideas: filtered '{title[:50]}' (infeasible: '{infeasible}' in hypothesis)",
                )
                continue

            # Check 3: non-trivial formula
            if len(formula.strip()) < 20:
                self.log_daemon(
                    "INFO",
                    f"generate_ideas: filtered '{title[:50]}' (trivial factor_formula: '{formula[:30]}')",
                )
                continue

            # NOTE: the old "Check 4" filtered on novelty/logic scores that
            # were produced BY THE SAME LLM CALL that generated the idea —
            # trivially self-serving and never a real filter. Structural
            # checks (1-3) stay; scoring is Gate 0's job (an independent,
            # adversarial call).
            valid.append(idea)

        filtered_count = len(ideas) - len(valid)
        if filtered_count > 0:
            self.log_daemon(
                "INFO",
                f"generate_ideas: filtered {filtered_count}/{len(ideas)} infeasible ideas",
            )
        return valid

    # ── Idea generation ────────────────────────────────────────────────────────

    def generate_ideas(self, topic: str = None, count: int = 5) -> list:
        # ── 1. KB context — GraphRAG retrieval over the knowledge graph ──────
        kb_context = ""
        kb_slugs: list = []   # provenance: which nodes grounded these ideas
        try:
            from knowledge.search.retriever import retrieve, assemble_context

            if topic:
                query = topic
            else:
                # No topic given: target the KB's least-covered research angle
                # instead of a fixed generic query (which just returned the
                # most-recent docs every time).
                query = "Bursa Malaysia equity alpha factor"
                try:
                    from knowledge.ingestion.diversity_engine import DiversityEngine
                    balance = DiversityEngine().check_balance()
                    target = balance.get("least_covered")
                    if target:
                        query = f"{target.replace('_', ' ')} Bursa Malaysia KLSE strategy"
                except Exception:
                    pass

            kb_results = retrieve(query, k=6, hops=2)
            if kb_results:
                kb_slugs = [r["slug"] for r in kb_results]
                kb_context = "\n" + assemble_context(kb_results, max_chars=2500)
                kb_context += (
                    "\n\nGenerate ideas that reference specific techniques and factors "
                    "from the above knowledge-graph context where applicable. Notes "
                    "flagged as CONTRADICTS highlight known counter-evidence — address it.\n"
                )
                self.log_daemon(
                    "INFO",
                    f"KB graph context: {len(kb_results)} nodes for query '{query[:60]}'"
                )
            else:
                self.log_daemon("INFO", "KB graph context: 0 nodes — generating without KB context")
        except Exception as e:
            self.log_daemon("WARN", f"KB context fetch failed (non-blocking): {e}")

        # ── 2. Rejection memory — pattern-level avoidance ────────────────────
        avoidance_context = ""
        try:
            from knowledge.ingestion.rejection_memory import RejectionMemory
            avoidance_context = RejectionMemory().inject_into_prompt()
        except Exception as e:
            self.log_daemon("WARN", f"RejectionMemory unavailable (non-blocking): {e}")

        # ── 3. Last 20 rejected ideas from DB — idea-level avoidance ─────────
        rejected_db_context = ""
        try:
            with db_session() as conn:
                rejected_rows = conn.execute("""
                    SELECT DISTINCT ai.title, gd.rationale
                    FROM alpha_ideas ai
                    JOIN gate_decisions gd ON ai.id = gd.idea_id
                    WHERE ai.status = 'rejected'
                    ORDER BY ai.updated_at DESC LIMIT 20
                """).fetchall()
            if rejected_rows:
                lines = [
                    f"- {r['title']}: {(r['rationale'] or 'rejected')[:120]}"
                    for r in rejected_rows
                ]
                rejected_db_context = (
                    "\nPREVIOUSLY REJECTED IDEAS — DO NOT repeat these or similar concepts:\n"
                    + "\n".join(lines) + "\n"
                )
        except Exception as e:
            self.log_daemon("WARN", f"Rejected ideas DB fetch failed (non-blocking): {e}")

        # ── 4. Active pipeline — enforce uniqueness ───────────────────────────
        pipeline_context = ""
        try:
            with db_session() as conn:
                active_rows = conn.execute("""
                    SELECT title, stage FROM alpha_ideas
                    WHERE status != 'rejected'
                    ORDER BY created_at DESC LIMIT 30
                """).fetchall()
            if active_rows:
                lines = [f"- [{r['stage']}] {r['title']}" for r in active_rows]
                pipeline_context = (
                    "\nALREADY IN PIPELINE — ensure new ideas are meaningfully different:\n"
                    + "\n".join(lines) + "\n"
                )
        except Exception as e:
            self.log_daemon("WARN", f"Pipeline ideas fetch failed (non-blocking): {e}")

        # ── 5. Technique library ──────────────────────────────────────────────
        technique_context = ""
        try:
            from knowledge.ingestion.technique_library import TechniqueLibrary
            technique_context = TechniqueLibrary().get_relevant_techniques(
                strategy_type=topic or "",
                stock_type="blue_chip",
                holding_period="medium_term",
                signal_type="price",
                max_techniques=3,
            )
            if technique_context:
                technique_context = (
                    "\nAvailable quantitative techniques to consider:\n"
                    + technique_context
                    + "\n\nSelect the most appropriate technique for the Bursa Malaysia "
                    "market context. Justify your technique choice briefly in the hypothesis.\n"
                )
        except Exception as e:
            self.log_daemon("WARN", f"TechniqueLibrary unavailable (non-blocking): {e}")

        topic_line = f"Focus exclusively on: {topic}" if topic else (
            "Cover a diverse mix: at least one technical, one fundamental/value, "
            "one event-driven, and one sector-rotation idea."
        )
        diversity_requirement = (
            f"\nDOMAIN DIVERSITY: The {count} ideas MUST span at least 3 different domains from: "
            "momentum, mean_reversion, value, event_driven, sector_rotation, dividend_capture, "
            "earnings, insider_flow, technical, macro. Do NOT generate multiple ideas in the same domain.\n"
        )

        prompt = f"""Generate exactly {count} quantitative equity alpha ideas for Bursa Malaysia stocks.
{kb_context}
{avoidance_context}
{rejected_db_context}
{pipeline_context}
{diversity_requirement}
{technique_context}
{topic_line}

HARD RULES — VIOLATIONS WILL CAUSE THE ENTIRE RESPONSE TO BE DISCARDED:
1. The "ticker" field MUST be a Yahoo Finance .KL symbol (e.g. 1155.KL, 5347.KL).
   DO NOT use currency pair notation (EUR_USD, USD_JPY, etc.) anywhere.
2. The "company" field MUST be the actual company name (e.g. "Maybank", "Tenaga Nasional").
3. The "sector" field MUST be a Bursa Malaysia sector name (Banking, Plantations, etc.).
4. {_TIMEFRAME_RULE}
5. "holding_period" must express weeks or months, NOT pips or ticks.
6. "factor_formula" must describe a STOCK price/fundamental signal, NOT an FX rate signal.

OHLCV DATA CONSTRAINT — MANDATORY FOR BACKTESTING:
The backtest engine can ONLY access daily OHLCV data from Yahoo Finance.
The "factor_formula" field MUST be computable from ONLY:
  ✓ Daily open, high, low, close prices
  ✓ Daily volume
  ✓ Derived: moving averages (SMA/EMA), RSI, MACD, Bollinger Bands, ATR, momentum
  ✓ Derived: rolling returns, rolling volatility, 52-week high/low

Do NOT include ANY of the following in "factor_formula" — ideas with these will be
automatically rejected at the backtest stage (wasting pipeline budget):
  ✗ Dividend yield, TTM yield, payout ratio, yield spread, blended yield
  ✗ Price-to-Book (P/B), ROE, Return on Equity, Debt/Equity ratio (DER)
  ✗ EPS forecasts, net profit margin, earnings yield, free cash flow yield
  ✗ KLCI constituent weights, market-cap-weighted reference yields
  ✗ Book value, BVPS, PE ratio referenced as a computed signal
  ✗ Corporate announcements, ex-dividend dates, dividend declarations
  ✗ Net Interest Margin (NIM), OPR spread calculations
  ✗ Bloomberg/Refinitiv data, analyst consensus estimates
  ✗ Any data not directly derivable from OHLCV price and volume bars

If your alpha thesis requires fundamental data, express the entry/exit timing using
a PRICE-BASED proxy instead:
  Example: Instead of "enter when dividend yield > 5%", write
  "enter when price falls below 200-day SMA and RSI < 35 (proxy for high-yield dip)".

Available tickers (use ONLY these or other confirmed .KL tickers):
{_UNIVERSE_BRIEF}

STRATEGY KEY — classify each idea into one of these strategy types.
The strategy_key determines which exit logic the backtest engine applies:
  cross_sectional_momentum — monthly ranking by 6M return, skip-month rule
  short_term_reversal      — buy sharp drops without fundamental catalyst
  low_volatility_anomaly   — long low-vol quintile quarterly
  rsi_mean_reversion       — RSI < 25 bounce with quality filter
  bollinger_squeeze_breakout — BB compression + volume breakout
  gap_fill                 — gap-down fill within 3 days
  sma_crossover            — 20/50 MA golden cross, unlimited hold
  pead                     — post-earnings announcement drift
  pre_ex_dividend          — accumulate before ex-date
  contract_win             — price response to contract award
  opr_cycle                — BNM OPR sensitivity in banking stocks
  cpo_lag                  — palm oil price lag in plantation stocks
  fundamental_screen       — quarterly fundamental quality screen
  other                    — does not fit any above category

Also include an "exit_quality" field (0.0-1.0) scoring how well-defined the exit is:
  1.0 = clear signal-conditioned exit (RSI level, MA crossover, specific event)
  0.7 = stop + profit target + time fallback
  0.4 = time-based exit only
  0.0 = no exit defined

Return a valid JSON array of exactly {count} objects. Each object:
{{
  "title":          "Concise strategy name (e.g. 'Maybank Dividend Capture Pre-Ex')",
  "hypothesis":     "Why this stock/signal generates alpha on Bursa Malaysia",
  "ticker":         "NNNN.KL  — a valid Bursa .KL symbol, NOT a currency pair",
  "company":        "Full company name",
  "sector":         "Bursa sector (Banking / Plantations / Utilities / etc.)",
  "timeframe":      {_TIMEFRAME_JSON},
  "factor_formula": "Precise signal construction INCLUDING entry AND exit rules. e.g. 'Enter long when 20-day SMA crosses above 50-day SMA and RSI(14) < 65. Exit when death cross (20d below 50d) or -10% stop-loss.'",
  "data_sources":   ["Yahoo Finance daily OHLCV", "Bursa quarterly earnings releases"],
  "strategy_type":  "momentum | value | quality | mean_reversion | event_driven | sector_rotation | technical",
  "strategy_key":   "one of the strategy_key values listed above",
  "exit_quality":   0.7,
  "holding_period": "e.g. 2-6 weeks",
  "concerns":       "Key implementation risks on Bursa (liquidity, lot size, corporate actions)"
}}

Do NOT score your own ideas — Gate 0 evaluates them independently."""

        result = self.call_claude_json(
            SYSTEM,
            [{"role": "user", "content": prompt}],
            max_tokens=4096,
            task_label="generate_ideas",
        )
        raw_ideas = result if isinstance(result, list) else result.get("ideas", [])

        # Filter out any FX-contaminated ideas that slipped through
        clean = []
        for idea in raw_ideas:
            if self._reject_if_fx(idea):
                self.log_daemon("WARN", f"Discarded FX-contaminated idea: {idea.get('title','?')}")
                continue
            ticker = idea.get("ticker", "")
            if not self._is_equity_ticker(ticker):
                # Attempt recovery: extract all .KL codes from anywhere in the idea JSON
                all_text = json.dumps(idea)
                found_raw = re.findall(r'\b\d{4}[A-Z0-9]*\.KL\b', all_text)
                # Deduplicate preserving order
                found = list(dict.fromkeys(found_raw))
                if found:
                    # Sector strategies: store comma-separated list; single-stock: just one ticker
                    idea["ticker"] = ",".join(found[:5])
                    self.log_daemon(
                        "WARN",
                        f"Corrected ticker for '{idea.get('title','?')}' → {idea['ticker']}",
                    )
                else:
                    self.log_daemon("WARN", f"Discarded idea with invalid ticker '{ticker}': {idea.get('title','?')}")
                    continue
            clean.append(idea)

        # Apply pre-save infeasibility filter
        clean = self._filter_infeasible(clean)
        # Provenance: tag every idea with the KB nodes that grounded the prompt
        # (NULL kb_context = ungrounded) so the funnel can measure KB utility.
        if kb_slugs:
            for idea in clean:
                idea.setdefault("kb_context", kb_slugs)
        self.log_daemon("INFO", f"Generated {len(clean)}/{len(raw_ideas)} clean KLSE equity ideas (post-filter)")
        return clean

    # ── Persistence ────────────────────────────────────────────────────────────

    def save_idea(self, idea: dict) -> int:
        title  = idea.get("title", "idea")
        slug   = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
        slug   = f"{datetime.utcnow().strftime('%Y-%m-%d')}-{slug}"
        ticker = idea.get("ticker") or DEFAULT_SYMBOLS[0]

        # Extract all valid tickers for the active market from the field.
        # Handles "1961.KL vs 5012.KL", "Consumer Staples: 4707.KL...", comma
        # lists — and in crypto mode the /USDT equivalents.
        found_tickers = TICKER_REGEX.findall(ticker)
        if not found_tickers:
            raise ValueError(
                f"save_idea() refused: no valid ticker found in '{ticker[:80]}'. "
                f"Title: '{title}'"
            )
        primary_ticker = found_tickers[0]
        # Rewrite ticker field to a clean comma-separated list (deduped, order preserved)
        seen: set = set()
        ticker = ",".join(t for t in found_tickers if not (t in seen or seen.add(t)))

        strategy_key = str(idea.get("strategy_key") or "other").strip()

        # Semantic dedup, layer 1 (text): normalized token signature of the
        # formula + ticker. Catches reworded duplicates ("Maybank golden
        # cross" vs "Maybank 20/50 MA crossover" with the same formula) that
        # the date+title slug never could. Layer 2 (the canonical DSL-tree
        # signature) upgrades this at parse time in the backtest engineer.
        import hashlib as _hashlib
        _formula_tokens = sorted(set(
            re.findall(r"[a-z0-9.]+", (idea.get("factor_formula", "") or "").lower())
        ))
        text_signature = "txt:" + _hashlib.sha256(
            (" ".join(_formula_tokens) + "|" + ticker).encode()
        ).hexdigest()

        with db_session() as conn:
            dup = conn.execute(
                "SELECT id, title FROM alpha_ideas "
                "WHERE signal_signature=? AND status != 'rejected' LIMIT 1",
                (text_signature,),
            ).fetchone()
            if dup:
                self.log_daemon(
                    "INFO",
                    f"save_idea DEDUP: '{title[:50]}' duplicates live idea "
                    f"[{dup['id']}] '{dup['title'][:50]}' — not saved",
                )
                return dup["id"]

            kb_context = idea.get("kb_context")
            from knowledge.ingestion.family_quotas import classify_family
            family = classify_family(
                f"{title} {idea.get('hypothesis', '')} {idea.get('factor_formula', '')}")

            # Never store an empty description — synthesize from title/formula if
            # the LLM produced an idea with no hypothesis (the organic path has
            # its own insert here, separate from the sandbox choke point).
            from pipeline.idea_text import ensure_description
            hypothesis = ensure_description(
                title, idea.get("hypothesis"), idea.get("factor_formula"))

            conn.execute("""
                INSERT OR IGNORE INTO alpha_ideas
                  (slug, title, hypothesis, ticker, timeframe, factor_formula,
                   data_sources, stage, status, novelty_score, logic_score,
                   strategy_key, signal_signature, parent_idea_id, kb_context, family)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'gate0', 'pending', ?, ?, ?, ?, ?, ?, ?)
            """, (
                slug,
                title,
                hypothesis,
                ticker,                               # stored in `ticker` column
                idea.get("timeframe", "1d"),
                idea.get("factor_formula", ""),
                json.dumps(idea.get("data_sources", [])),
                float(idea.get("novelty_score", 0.0)),
                float(idea.get("logic_score", 0.0)),
                strategy_key,
                text_signature,
                idea.get("parent_idea_id"),
                json.dumps(kb_context) if kb_context else None,
                family,
            ))
            row = conn.execute("SELECT id FROM alpha_ideas WHERE slug=?", (slug,)).fetchone()

        self.log_daemon("INFO", f"Saved idea [{row['id']}] {ticker} — {slug}")

        # Phase 5.5: non-blocking cemetery similarity check — informational only.
        try:
            from knowledge.ingestion.rejection_memory import RejectionMemory
            similar = RejectionMemory().find_similar_rejected(title, idea.get("hypothesis", ""))
            if similar:
                self.log_daemon(
                    "INFO",
                    f"Idea [{row['id']}] resembles {len(similar)} past rejection(s), "
                    f"e.g. '{similar[0]['strategy_name'][:50]}' ({similar[0]['similarity']:.0%} "
                    f"overlap): {similar[0]['revival_conditions']}"
                )
        except Exception as e:
            self.log_daemon("WARN", f"Cemetery similarity check failed (non-blocking): {e}")

        return row["id"]

    # ── Gate 0 — initial screening ─────────────────────────────────────────────

    @staticmethod
    def _compute_feasibility(idea_row, ticker: str, factor_formula: str) -> float:
        """Score 0.0-1.0: is the idea actually executable in the active market?

        Ticker format and data-availability keyword lists come from the market
        profile (Bursa: .KL / yfinance limits; crypto: /USDT pairs / no
        on-chain-funding-orderbook data). Scoring structure is mostly identical
        across markets, with one branch: Bursa is long-only (WS3: crypto is
        long/short via perps, ALLOW_SHORT=True) — multi-leg spread/pairs
        structures stay out of scope on both markets (single-instrument DSL).
        """
        score = 0.0
        formula_lower = (factor_formula or "").lower()
        hypothesis_lower = (idea_row["hypothesis"] or "").lower()
        blob = formula_lower + " " + hypothesis_lower

        # +0.3: valid ticker for this market
        if TICKER_REGEX.fullmatch(ticker.strip()):
            score += 0.3

        # +0.2: directionally supported (long-only on Bursa; long/short on
        # crypto perps — WS3). Multi-leg spread/pairs structures are always
        # out of scope (single-instrument DSL, not a basket spread).
        spread_keywords = ["pairs", "spread arbitrage"]
        if ALLOW_SHORT:
            blocked_direction_keywords = spread_keywords
        else:
            blocked_direction_keywords = spread_keywords + ["short", "sell short", "hedge"]
        if not any(kw in blob for kw in blocked_direction_keywords):
            score += 0.2

        # +0.2: data available from this market's backend
        if not any(kw in blob for kw in UNAVAILABLE_DATA_KEYWORDS):
            score += 0.2

        # +0.15: holding period realistic (dockable granularities per market
        # profile — Bursa docks all sub-daily; crypto only sub-15m/tick/HFT)
        from config.settings import FEASIBILITY_DOCK_KEYWORDS
        intraday_keywords = FEASIBILITY_DOCK_KEYWORDS
        short_hold_keywords = ["1 day", "2 day", "3 day", "t+1", "t+2"]
        if any(kw in blob for kw in intraday_keywords):
            score -= 0.3
        elif any(kw in blob for kw in short_hold_keywords):
            score += 0.0   # very short holds are awkward but not impossible
        else:
            score += 0.15   # >5 day holding period — comfortably feasible

        # +0.15: factor uses available indicators (price/volume/fundamental)
        if not any(kw in blob for kw in EXOTIC_KEYWORDS):
            score += 0.15

        return round(min(max(score, 0.0), 1.0), 3)

    def score_gate0(self, idea_id: int) -> dict:
        with db_session() as conn:
            row = conn.execute("SELECT * FROM alpha_ideas WHERE id=?", (idea_id,)).fetchone()
        if not row:
            return {"error": f"Idea {idea_id} not found"}
        row = dict(row)   # convert sqlite3.Row → dict so .get() works everywhere

        ticker         = row["ticker"] or "1155.KL"
        primary_ticker = ticker.split(",")[0].strip()   # handle comma-separated sector tickers
        stock_info     = KLCI_BY_SYMBOL.get(primary_ticker, {})
        company        = stock_info.get("name", primary_ticker)
        sector         = stock_info.get("sector", "Unknown")

        # Compute feasibility before calling Claude
        feasibility = self._compute_feasibility(row, primary_ticker, row["factor_formula"] or "")

        prompt = f"""Evaluate this Bursa Malaysia equity strategy at Gate 0. Reject flawed ideas —
broken logic, unavailable data, severe execution barriers, obvious data-mining.
Simple-but-sound ideas with honest mechanisms SHOULD pass: simplicity is not a
flaw, and the statistical gates downstream (deflated Sharpe hurdle, parameter
robustness, cross-sectional IC) are the arbiters of whether an edge is real.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Stock:      {company} ({ticker}) — {sector}
Title:      {row['title']}
Hypothesis: {row['hypothesis']}
Signal:     {row['factor_formula']}
Timeframe:  {row['timeframe']}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Score each dimension 0.0–1.0:

1. NOVELTY (0–1): Is this differentiated beyond textbook strategies?
   Score honestly — but note this dimension is ADVISORY (used for research
   prioritization), not pass/fail. A simple momentum idea can score 0.2 here
   and still pass Gate 0 if its logic and feasibility are sound.

2. LOGIC (0–1): Is the economic mechanism sound? A clear, honest mechanism
   ("banks lag OPR moves because...") scores well even if simple. Score LOW
   for hand-waving, contradictions, or mechanisms that don't survive a
   moment's scrutiny (e.g. wrong sector for the claimed driver).

3. FEASIBILITY (0–1): Can a real fund execute this on Bursa today?
   Consider: 100-share lot minimum, stamp duty 0.10% (remitted), T+2 settlement impact on
   short holding periods, short-selling restrictions (approved list only),
   thin liquidity in mid/small caps (daily value < RM500k = not viable).
   Score LOW if execution barriers are severe.

4. DATA_QUALITY (0–1): Is ALL required data reliably available on Yahoo Finance .KL?
   Daily OHLCV, dividends, basic fundamentals (PE, book value, market cap) are available.
   Analyst estimates, options data, real-time feeds, tick data are NOT available.
   Score LOW if the signal depends on unavailable data.

5. OVERFITTING_RISK (0–1): Does this look data-mined to fit past returns?
   Score HIGH (bad) if: too many parameters, suspiciously precise thresholds,
   very short lookback, or no intuitive economic rationale for the exact rules.
   Score LOW (good) if: simple signal, clear economic story, robust to parameter changes.
   *** For PASS we need overfitting_risk <= 0.40 (lower is better) ***

6. EXIT_QUALITY (0–1): How well-defined is the exit strategy?
   Score 1.0 if: clear signal-conditioned exit is defined (e.g. 'exit when RSI > 55',
     'exit when 20-day SMA crosses below 50-day SMA', 'exit when gap is filled').
   Score 0.7 if: stop-loss + profit target + time fallback all defined.
   Score 0.4 if: only a fixed time exit is defined (e.g. 'hold for 30 days').
   Score 0.0 if: no exit condition is defined at all.
   *** This dimension is informational only — does NOT affect pass/fail ***

Deterministic pre-score: feasibility={feasibility:.2f}/1.00 (must be >= 0.60 to pass)

Pass requires ALL of (novelty is advisory, NOT a pass condition):
  logic >= 0.65, feasibility >= 0.70,
  data_quality >= 0.70, overfitting_risk <= 0.40, det_feasibility >= 0.60

If you reject this idea due to data unavailability (data_quality_score < 0.70),
include a 'price_based_proxy' field describing how to express a similar hypothesis
using ONLY daily OHLCV data. E.g. instead of P/B ratio, use 52-week price drawdown
as a value proxy. Keep it to one actionable sentence.

Return JSON only:
{{
  "novelty_score":        0.0,
  "logic_score":          0.0,
  "feasibility_score":    0.0,
  "data_quality_score":   0.0,
  "overfitting_risk":     0.5,
  "exit_quality":         0.7,
  "pass":                 false,
  "rationale":            "2-3 sentences — be specific about which dimensions fail and why",
  "key_risks":            ["specific risk 1", "specific risk 2"],
  "data_availability":    "high|medium|low",
  "liquidity_concern":    false,
  "short_selling_required": false,
  "price_based_proxy":    null
}}"""

        try:
            result = self.call_claude_json(
                GATE0_SYSTEM,
                [{"role": "user", "content": prompt}],
                model=MODEL_FAST,
                task_label="gate0_score",
                raise_on_error=True,
            )
        except ClaudeJSONError as exc:
            # A malformed response must NOT be scored as novelty=0/logic=0 and
            # rejected — that was the historical "Gate 0 scored 0.00" bug.
            # Leave the idea pending so the next daemon cycle retries it.
            self.logger.error(
                f"[score_gate0] JSON parse failed for idea {idea_id} ({ticker}) — "
                f"leaving pending for retry.\nRAW RESPONSE:\n{exc.raw[:2000]}"
            )
            return {"error": str(exc), "pass": False, "retry": True}
        except Exception as exc:
            self.logger.error(
                f"[score_gate0] Claude API call raised exception for idea {idea_id}: {exc}",
                exc_info=True,
            )
            return {"error": str(exc), "pass": False, "retry": True}

        novelty      = float(result.get("novelty_score",     result.get("novelty", 0)))
        logic        = float(result.get("logic_score",       result.get("logic",   0)))
        claude_feas  = float(result.get("feasibility_score", 0))
        data_qual    = float(result.get("data_quality_score", 0))
        overfit      = float(result.get("overfitting_risk",  1.0))
        exit_quality = float(result.get("exit_quality",      0.4))  # informational only

        # Log a warning if all scores are suspiciously zero (likely a parse problem)
        if novelty == 0.0 and logic == 0.0 and "error" not in result:
            self.logger.warning(
                f"[score_gate0] Scores all zero for idea {idea_id} — "
                f"possible key mismatch. Keys: {list(result.keys())}\nFull: {result}"
            )

        # Gate 0: ALL FIVE dimensions + deterministic feasibility must pass
        # exit_quality is informational only — does NOT affect pass/fail
        # Novelty is ADVISORY — recorded for prioritization, not pass/fail.
        # A simple genuine alpha always scores low novelty; killing it here
        # contradicted the pipeline's purpose. Redundant/data-mined ideas are
        # punished statistically downstream (deflated Sharpe hurdle,
        # parameter robustness, cross-sectional IC) where the evidence is.
        # Thresholds live in GateConfig (audit hygiene 2026-07-10 — they were
        # hardcoded here while UNRELATED dead constants sat in the config).
        passed = (
            logic       >= GATE_CONFIG.gate0_min_logic_score
            and claude_feas >= GATE_CONFIG.gate0_min_claude_feasibility
            and data_qual   >= GATE_CONFIG.gate0_min_data_quality
            and overfit     <= GATE_CONFIG.gate0_max_overfitting_risk
            and feasibility >= 0.60   # deterministic pre-score (sandbox MIN_FEASIBILITY)
        )

        # Log exit quality with a hint if it's low
        eq_label = {1.0: "excellent", 0.7: "good", 0.4: "weak (time only)", 0.0: "missing"}
        eq_desc  = eq_label.get(exit_quality, f"{exit_quality:.1f}")
        self.log_daemon(
            "INFO" if exit_quality >= 0.7 else "WARN",
            f"Gate0 [{idea_id}] exit_quality={exit_quality:.1f} ({eq_desc})"
        )

        # Build a clear failure message so rejection_memory gets useful context
        if not passed:
            failed_dims = []
            if logic       < 0.65: failed_dims.append(f"logic={logic:.2f}<0.65")
            if claude_feas < 0.70: failed_dims.append(f"feasibility={claude_feas:.2f}<0.70")
            if data_qual   < 0.70: failed_dims.append(f"data_quality={data_qual:.2f}<0.70")
            if overfit     > 0.40: failed_dims.append(f"overfitting_risk={overfit:.2f}>0.40")
            if feasibility < 0.60: failed_dims.append(f"det_feasibility={feasibility:.2f}<0.60")
            rationale = result.get("rationale", "") + f" [FAILED: {', '.join(failed_dims)}]"
        else:
            rationale = result.get("rationale", "")

        with db_session() as conn:
            conn.execute("""
                UPDATE alpha_ideas
                SET novelty_score=?, logic_score=?, feasibility_score=?,
                    stage=?, status=?, updated_at=datetime('now')
                WHERE id=?
            """, (
                novelty,
                logic,
                feasibility,
                "stage1"   if passed else "gate0",
                "active"   if passed else "rejected",
                idea_id,
            ))
            conn.execute("""
                INSERT INTO gate_decisions
                  (idea_id, gate, decision, decided_by, rationale)
                VALUES (?, 'gate0', ?, 'StrategyResearcher', ?)
            """, (idea_id, "approve" if passed else "reject", rationale))
            conn.execute("""
                INSERT INTO pipeline_events
                  (idea_id, stage, event_type, agent, notes)
                VALUES (?, 'gate0', ?, 'StrategyResearcher', ?)
            """, (idea_id, "advanced" if passed else "rejected", rationale))

        result["feasibility_score"]  = feasibility   # deterministic pre-score stored in DB
        result["claude_feasibility"] = claude_feas
        result["data_quality_score"] = data_qual
        result["overfitting_risk"]   = overfit
        result["pass"]               = passed
        result["rationale"]          = rationale

        self.log_daemon(
            "INFO" if passed else "WARN",
            f"Gate 0 {'PASSED' if passed else 'FAILED'}: [{idea_id}] {ticker} — {row['title']} "
            f"(novelty={novelty:.2f} logic={logic:.2f} feas={claude_feas:.2f} "
            f"data_q={data_qual:.2f} overfit={overfit:.2f} det_feas={feasibility:.2f})",
        )

        # ── Price-based proxy redirect + auto-resubmit ────────────────────────
        # Live evidence showed unlimited proxy spawning dominating generation
        # (13 of 15 recent ideas were "Price proxy:" spawns, all rejected).
        # Hard limits: never proxy a proxy, max ONE proxy per original idea
        # ever, and multi-ticker fundamental ideas go to the fundamental
        # screen instead of being degraded into a price proxy.
        if not passed:
            proxy = (result.get("price_based_proxy") or "").strip()
            _is_already_proxy = row["title"].startswith("Price proxy: ")
            _n_tickers = len([t for t in (row["ticker"] or "").split(",")
                              if ".KL" in t])
            if proxy and data_qual < 0.70 and not _is_already_proxy and _n_tickers < 5:
                with db_session() as conn:
                    conn.execute(
                        "UPDATE alpha_ideas SET price_based_proxy=? WHERE id=?",
                        (proxy, idea_id),
                    )
                    already_spawned = conn.execute(
                        "SELECT id FROM alpha_ideas WHERE parent_idea_id=? LIMIT 1",
                        (idea_id,),
                    ).fetchone()
                if already_spawned:
                    self.log_daemon(
                        "INFO",
                        f"Gate 0 REDIRECT: proxy already spawned for [{idea_id}] — skipping",
                    )
                    return result
                self.log_daemon("INFO", f"Gate 0 REDIRECT: Try '{proxy}' instead")
                try:
                    proxy_idea = {
                        "title":          f"Price proxy: {row['title'][:40]}",
                        "hypothesis":     proxy,
                        "ticker":         ticker,
                        "timeframe":      row.get("timeframe") or "1d",
                        "factor_formula": proxy,
                        "data_sources":   ["Yahoo Finance daily OHLCV"],
                        "parent_idea_id": idea_id,
                    }
                    new_id = self.save_idea(proxy_idea)
                    self.log_daemon(
                        "INFO",
                        f"Gate 0 REDIRECT: Auto-created proxy idea [{new_id}] "
                        f"'{proxy_idea['title']}'",
                    )
                except Exception as proxy_err:
                    self.log_daemon(
                        "WARN",
                        f"Gate 0 REDIRECT: Failed to auto-create proxy idea: {proxy_err}",
                    )

        return result

    # ── Stage 1 — deep research ────────────────────────────────────────────────

    def research_idea(self, idea_id: int) -> dict:
        with db_session() as conn:
            _row = conn.execute("SELECT * FROM alpha_ideas WHERE id=?", (idea_id,)).fetchone()
        if not _row:
            return {"error": f"Idea {idea_id} not found"}
        row = dict(_row)   # convert sqlite3.Row → dict so .get() works everywhere

        ticker     = row["ticker"] or "1155.KL"
        stock_info = KLCI_BY_SYMBOL.get(ticker, {})
        company    = stock_info.get("name", ticker)
        sector     = stock_info.get("sector", "Unknown")

        # Hunt for relevant academic papers before calling Claude
        papers_section = ""
        try:
            from knowledge.ingestion.research_hunter import ResearchHunter
            hunter      = ResearchHunter()
            hunt_result = hunter.hunt(row["title"], row["hypothesis"] or "")
            if hunt_result.get("papers_ingested", 0) > 0:
                lines = [
                    f"\nRelevant academic papers found "
                    f"({hunt_result['papers_ingested']} ingested into KB):"
                ]
                for title in hunt_result.get("titles", [])[:5]:
                    lines.append(f"- {title}")
                lines.append(
                    "\nIncorporate insights from these papers where relevant in your analysis."
                )
                papers_section = "\n".join(lines)
        except Exception as e:
            self.log_daemon("WARN", f"ResearchHunter failed for [{idea_id}]: {e}")

        # ── Technique library — inject full detail for any technique referenced ──
        technique_detail = ""
        try:
            from knowledge.ingestion.technique_library import TechniqueLibrary, TECHNIQUE_LIBRARY
            lib = TechniqueLibrary()
            formula_lower = (row["factor_formula"] or "").lower()
            hypothesis_lower = (row["hypothesis"] or "").lower()
            combined = formula_lower + " " + hypothesis_lower

            # Detect which techniques are referenced
            matched = [
                k for k in TECHNIQUE_LIBRARY
                if k.replace("_", " ") in combined or k in combined
            ]
            if matched:
                detail_blocks = [lib.format_full_detail(k) for k in matched[:3]]
                technique_detail = (
                    "\n\nQUANTITATIVE TECHNIQUE REFERENCE (referenced in this strategy):\n"
                    + "\n\n".join(detail_blocks)
                    + "\n\nApply the above technique guidance in your refined signal construction.\n"
                )
            else:
                # No specific technique matched — suggest the most relevant ones
                row_strategy_type = (row.get("strategy_type") or "").lower()
                suggestions = lib.get_relevant_techniques(
                    strategy_type=row_strategy_type,
                    stock_type="blue_chip",
                    holding_period="medium_term",
                    signal_type="price",
                    max_techniques=2,
                )
                if suggestions:
                    technique_detail = (
                        "\n\nSUGGESTED QUANTITATIVE TECHNIQUES for this strategy type:\n"
                        + suggestions
                        + "\n\nConsider whether any of the above would improve signal quality.\n"
                    )
        except Exception as e:
            self.log_daemon("WARN", f"TechniqueLibrary inject failed for [{idea_id}] (non-blocking): {e}")

        # Existing knowledge graph — deep research must build on what the
        # system already knows, and explicitly confront counter-evidence,
        # instead of only hunting fresh papers.
        graph_section, graph_slugs = "", []
        try:
            from knowledge.search.retriever import retrieve
            kb_results = retrieve(
                f"{row['title']} {row['hypothesis'] or ''}"[:300], k=5, hops=2)
            if kb_results:
                graph_slugs = [r["slug"] for r in kb_results]
                supporting = [r for r in kb_results if not r["contradicts"]]
                contra = [r for r in kb_results if r["contradicts"]]
                lines = ["\nEXISTING KNOWLEDGE GRAPH (what the system already knows):"]
                for r in supporting[:4]:
                    lines.append(f"• [{r['node_type']}/{r['domain']}] {r['title']}: "
                                 f"{(r['summary'] or '')[:250]}")
                if contra:
                    lines.append("\nKNOWN COUNTER-EVIDENCE — your research MUST address these:")
                    for r in contra[:3]:
                        lines.append(f"⚠ {r['title']}: {(r['summary'] or '')[:250]}")
                graph_section = "\n".join(lines) + "\n"
                self.log_daemon(
                    "INFO",
                    f"KB graph context: {len(kb_results)} nodes for Stage 1 [{idea_id}] "
                    f"({len(contra)} counter-evidence)",
                )
        except Exception as _g_exc:
            self.log_daemon("WARN", f"KB graph fetch failed for [{idea_id}] (non-blocking): {_g_exc}")

        prompt = f"""Conduct deep research on this Bursa Malaysia equity strategy.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Stock:      {company} ({ticker}) — {sector}
Title:      {row['title']}
Hypothesis: {row['hypothesis']}
Signal:     {row['factor_formula']}
Timeframe:  {row['timeframe']}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{technique_detail}

Research tasks:
1. Academic/practitioner evidence — does this factor work in ASEAN/EM equities?
2. Precise signal construction — exact calculation, look-back, rebalance frequency
3. KLSE-specific risks — GLC dynamics, EPF ownership, sector concentration, MYR
   macro impact on earnings (NOT as a trade — as an earnings risk factor)
4. Performance expectations — realistic Sharpe, CAGR, max drawdown for KLSE
5. Implementation rules — entry trigger, position size, stop-loss %, exit rule

Note: "MYR" appears here only as the earnings/valuation currency, NOT as a traded
instrument. All positions are in Bursa Malaysia stocks.
{graph_section}{papers_section}
Return JSON only (use exactly this schema):
{{
  "research_score":          7.5,
  "_score_note":             "Float 0.0–10.0 with ONE decimal place. 10.0=exceptional alpha with strong KLSE evidence; 7.0=solid; 5.0=mixed; 3.0=weak; 0.0=reject. DO NOT default to 7.25 or 72.5.",
  "factor_formula_refined":  "Precise, step-by-step signal definition",
  "entry_rule":              "Exact entry condition",
  "exit_rule":               "Exact exit condition",
  "stop_loss_pct":           0.08,
  "rebalance_frequency":     "daily|weekly|monthly",
  "data_sources":            ["Yahoo Finance daily OHLCV 1155.KL", "..."],
  "expected_sharpe":         0.0,
  "expected_annual_return_pct": 0.0,
  "expected_max_dd_pct":     0.0,
  "favorable_regimes":       ["bull market", "low OPR environment"],
  "unfavorable_regimes":     ["high EPF outflow periods", "index rebalancing"],
  "klse_specific_risks":     ["GLC ownership concentration", "..."],
  "comparable_factors":      ["AQR paper: 'Value and Momentum Everywhere'", "..."],
  "research_summary":        "3-5 sentence verdict on this strategy's merit for KLSE",
  "pass":                    true
}}"""

        # Retry parity with Gate 0 (gate audit, 2026-07-10): a malformed LLM
        # response must not hard-reject the idea as score=0 — leave it at
        # stage1/active so the next daemon cycle retries. Same guard
        # score_gate0 has carried since the "scored 0.00" bug.
        try:
            result = self.call_claude_json(
                SYSTEM,
                [{"role": "user", "content": prompt}],
                max_tokens=8192,
                task_label="deep_research",
                raise_on_error=True,
            )
        except Exception as exc:
            self.logger.error(
                f"[research_idea] LLM call/parse failed for idea {idea_id} — "
                f"leaving at stage1 for retry: {exc}")
            return {"error": str(exc), "pass": False, "retry": True}

        # Debug: log raw response so we can verify Claude is evaluating each idea
        import logging as _log
        _log.getLogger(__name__).debug(
            f"Stage 1 raw response [{idea_id}]: "
            + json.dumps({k: v for k, v in result.items() if k != "_score_note"})[:600]
        )

        # Validate research_score is not a suspiciously uniform default
        rs = float(result.get("research_score", 0) or 0)
        if abs(rs - 72.5) < 0.15 or abs(rs - 7.25) < 0.05:
            self.log_daemon(
                "WARN",
                f"Stage 1 [{idea_id}] research_score={rs:.2f} matches known default value "
                f"— Claude response may not reflect genuine per-idea evaluation",
            )

        passed = (
            result.get("pass", False)
            and rs >= GATE_CONFIG.stage1_min_research_score
        )

        with db_session() as conn:
            conn.execute("""
                UPDATE alpha_ideas
                SET research_score=?, factor_formula=?, data_sources=?,
                    stage=?, status=?, updated_at=datetime('now')
                WHERE id=?
            """, (
                rs,
                result.get("factor_formula_refined", row["factor_formula"]),
                json.dumps(result.get("data_sources", [])),
                "stage2" if passed else "stage1",
                "active" if passed else "rejected",
                idea_id,
            ))
            # Merge Stage-1 graph provenance into kb_context
            if graph_slugs:
                existing_ctx = conn.execute(
                    "SELECT kb_context FROM alpha_ideas WHERE id=?", (idea_id,)
                ).fetchone()["kb_context"]
                try:
                    merged = list(dict.fromkeys(
                        (json.loads(existing_ctx) if existing_ctx else []) + graph_slugs))
                except Exception:
                    merged = graph_slugs
                conn.execute(
                    "UPDATE alpha_ideas SET kb_context=? WHERE id=?",
                    (json.dumps(merged), idea_id))
            conn.execute("""
                INSERT INTO pipeline_events
                  (idea_id, stage, event_type, agent, notes)
                VALUES (?, 'stage1', ?, 'StrategyResearcher', ?)
            """, (idea_id,
                  "advanced" if passed else "rejected",
                  result.get("research_summary", ""),))

        self.log_daemon(
            "INFO",
            f"Stage 1 complete [{idea_id}] {ticker} — score={rs:.2f} pass={passed}",
        )
        return result

    # ── Screener-driven idea generation ───────────────────────────────────────

    def screen_and_generate(self, count: int = 5) -> list:
        """Pull live fundamental data from KLSE screener then generate ideas."""
        try:
            from data.klse.screener import get_klci_constituents
            stocks = get_klci_constituents(enrich=False)
        except Exception as e:
            logger.warning(f"KLSE screener unavailable, using static universe: {e}")
            stocks = KLCI_STOCKS

        universe_str = "\n".join(
            f"  {s['symbol']:12s} {s['name']:30s} {s['sector']:18s}"
            + (f"  P/E={s['pe']:.1f}" if s.get("pe") else "")
            + (f"  DY={s['dy_pct']:.1f}%" if s.get("dy_pct") else "")
            + (f"  ROE={s['roe_pct']:.1f}%" if s.get("roe_pct") else "")
            for s in stocks[:25]
        )

        prompt = f"""You are given live Bursa Malaysia FBM KLCI stock data. Generate {count}
data-driven alpha ideas that exploit specific, observable patterns in this data.

Live universe snapshot:
{universe_str}

Each idea MUST directly reference one of the .KL tickers shown above and explain
a specific, quantifiable signal. The "ticker" MUST end in .KL.

Return JSON array with fields:
title, hypothesis, ticker (must be X.KL), company, sector, timeframe (="1d"),
factor_formula, data_sources, strategy_type, holding_period,
novelty_score, logic_score, concerns."""

        result = self.call_claude_json(
            SYSTEM,
            [{"role": "user", "content": prompt}],
            task_label="screen_generate",
        )
        raw = result if isinstance(result, list) else result.get("ideas", [])
        clean = [idea for idea in raw if not self._reject_if_fx(idea)
                 and self._is_equity_ticker(idea.get("ticker", ""))]
        return self._filter_infeasible(clean)

    # ── KB seeding ────────────────────────────────────────────────────────────

    def seed_knowledge_base(self) -> dict:
        """
        Ingest 10 foundational Bursa Malaysia market structure facts into the
        knowledge base. Safe to call multiple times (uses ON CONFLICT DO UPDATE).
        """
        try:
            from knowledge.ingestion.kb_ingester import KBIngester
            kb = KBIngester()
        except Exception as e:
            return {"error": f"KBIngester unavailable: {e}"}

        facts = [
            {
                "title": "FBM KLCI Index — Composition and Methodology",
                "domain": "fx",   # using "fx" key → remapping to "research" below
                "content": (
                    "The FTSE Bursa Malaysia KLCI (FBM KLCI) is the headline index of Bursa Malaysia. "
                    "It comprises the 30 largest companies by full market capitalisation listed on the "
                    "Bursa Malaysia Main Market. The index is calculated using a free-float adjusted, "
                    "market-cap weighted methodology in partnership with FTSE Russell. Constituents are "
                    "reviewed semi-annually (March and September). Index divisor is adjusted for "
                    "corporate actions. As of 2024-2025 the index weight is dominated by banking stocks "
                    "(Maybank 1155.KL, Public Bank 1295.KL, CIMB 1023.KL) which together represent "
                    "roughly 30-35% of total index weight. Other heavy-weight sectors: Utilities "
                    "(Tenaga 5347.KL), Telecoms (Maxis 6012.KL, CelcomDigi 6947.KL), Plantations "
                    "(Sime Darby Plantation 5285.KL, IOI 1961.KL)."
                ),
            },
            {
                "title": "Bursa Malaysia Trading Hours and Market Structure",
                "domain": "research",
                "content": (
                    "Bursa Malaysia operates two daily sessions on business days (Monday–Friday, "
                    "excluding Malaysian public holidays). Morning session: 09:00–12:30 MYT. "
                    "Afternoon session: 14:30–17:00 MYT. Pre-opening: 08:30–09:00 MYT (order matching). "
                    "Total trading time: approximately 6 hours per day. Settlement is on a T+2 basis "
                    "(effective 2019-04-29) via Bursa Depository. All prices are quoted in Malaysian Ringgit "
                    "(MYR). Board lot is 100 shares. Odd lots (< 100 shares) can be traded on the "
                    "Odd Lot Market at a wider spread."
                ),
            },
            {
                "title": "Bursa Malaysia Short-Selling Rules and Approved Securities",
                "domain": "research",
                "content": (
                    "Short-selling on Bursa Malaysia is strictly regulated. Only 'Approved Securities' "
                    "on the Securities Commission's published list may be short-sold. As of 2024, "
                    "approximately 100-150 stocks are on the approved list; this includes most FBM KLCI "
                    "constituents. The uptick rule applies — short orders can only be executed at a "
                    "price above the last done price. Intraday short-selling (IDSS) is available but "
                    "positions must be closed by end of trading day. Regulated Short Selling (RSS) "
                    "allows overnight short positions via Bursa-approved stockbrokers with a securities "
                    "borrowing and lending (SBL) facility. Naked short-selling is prohibited. "
                    "Practically, most retail and even many institutional strategies are long-only."
                ),
            },
            {
                "title": "Transaction Costs on Bursa Malaysia — Stamp Duty, Brokerage, Clearing",
                "domain": "research",
                "content": (
                    "Round-trip transaction costs on Bursa Malaysia: "
                    "(1) Brokerage: 0.10%–0.42% per side; online platforms charge ~0.10%–0.20%. "
                    "(2) Stamp duty on purchase: 0.10% remitted rate (to 2028-07-12) of contract value, capped at RM1,000 per contract. "
                    "Stamp duty does NOT apply on sales. "
                    "(3) Clearing fee: 0.03% per side (paid to Bursa Clearing), capped at RM1,000. "
                    "(4) No capital gains tax in Malaysia on equities. "
                    "Dividend withholding tax: 0% for Malaysian residents (single-tier tax). "
                    "Total estimated round-trip cost for a typical trade: ~0.30%–0.50%. "
                    "For a quantitative strategy rebalancing weekly, annual drag from costs alone "
                    "can be 15–25% if turnover is high. Strategies should target high gross alpha "
                    "and moderate turnover."
                ),
            },
            {
                "title": "Typical Valuation Ranges for FBM KLCI Stocks",
                "domain": "research",
                "content": (
                    "As of 2023-2025, FBM KLCI historical valuation benchmarks: "
                    "Price-to-Earnings (P/E): KLCI index average 13–17×. Below 10× is considered "
                    "deep value; above 22× expensive relative to historical norms. "
                    "Price-to-Book (P/B): KLCI average 1.4–1.8×. Banks often trade at 0.8–1.3× book. "
                    "Dividend Yield: KLCI average 3.5–5.5%. High-yield stocks (>5%) tend to be GLCs, "
                    "utilities, and REITs. "
                    "ROE: KLCI average 8–12%. Quality threshold: ROE > 15% consistently. "
                    "Earnings growth: nominal GDP + 3–6% historically. Plantation earnings are highly "
                    "cyclical and tied to CPO price (RM2,000–RM6,000/tonne range). "
                    "Index EPS growth averages ~5–8% in stable years; can be negative in commodity "
                    "downturns or global risk-off events."
                ),
            },
            {
                "title": "EPF, KWAP and GLC Ownership — Institutional Flows on Bursa Malaysia",
                "domain": "research",
                "content": (
                    "The Malaysian equity market is dominated by domestic institutional investors: "
                    "Employees Provident Fund (EPF): largest single institutional investor, manages "
                    "RM1.1 trillion+ in AUM. Owns 10-25% of most KLCI blue chips. EPF rebalancing "
                    "events can cause significant short-term price impact. "
                    "KWAP (Kumpulan Wang Persaraan): government pension, RM160bn+ AUM. "
                    "Permodalan Nasional Berhad (PNB): manages Amanah Saham unit trusts, major holder "
                    "of Maybank, Sime Darby, CIMB. "
                    "Government-Linked Companies (GLCs) include Petronas subsidiaries (Petronas Gas "
                    "6033.KL, Petronas Chemicals 5183.KL), Tenaga 5347.KL, Telekom 4863.KL. GLCs "
                    "often have implicit government support, depressing downside risk but also capping "
                    "upside from M&A. Foreign ownership: typically 20-25% of KLCI market cap; MSCI EM "
                    "review events and risk-off EM episodes cause sharp foreign outflows."
                ),
            },
            {
                "title": "CPO (Crude Palm Oil) Price as a Key Driver for Malaysian Plantation Stocks",
                "domain": "research",
                "content": (
                    "Malaysia is the world's second-largest palm oil producer. CPO prices (quoted "
                    "on Bursa Malaysia Derivatives in RM/tonne) are the primary driver of plantation "
                    "stock earnings. Key stocks affected: Sime Darby Plantation (5285.KL), IOI Corp "
                    "(1961.KL), Kuala Lumpur Kepong (2445.KL), PPB Group (4065.KL via Wilmar stake). "
                    "Historical CPO price ranges: RM2,000–2,500/t (trough), RM3,500–4,500/t (normal), "
                    "RM6,000–8,000/t (peak, as in 2022). Plantation stocks typically show 60-80% "
                    "correlation with CPO price over 3-month horizons. "
                    "Strategy implication: CPO price momentum / mean-reversion can proxy plantation "
                    "sector direction. CPO above 200-day MA is positive regime for 5285.KL, 1961.KL."
                ),
            },
            {
                "title": "Earnings Calendar and Post-Earnings Drift (PEAD) on Bursa Malaysia",
                "domain": "research",
                "content": (
                    "Bursa Malaysia listed companies report quarterly: Q1 (Jan-Mar) results due by "
                    "end of May; Q2 (Apr-Jun) by end of August; Q3 (Jul-Sep) by end of November; "
                    "Q4 (Oct-Dec) by end of February. Earnings announcements are made via Bursa's "
                    "BURSA LINK system. Post-Earnings Announcement Drift (PEAD): academic evidence "
                    "for KLSE suggests stocks beating consensus estimates by >10% continue to "
                    "outperform for 20-40 trading days post-announcement. The effect is larger for "
                    "small/mid caps. For KLCI large caps, the drift is 5-15 days on average, dampened "
                    "by high institutional ownership and faster price discovery. Implementation: screen "
                    "for revenue and EPS beat vs prior quarter, enter on announcement day close, "
                    "hold 15 trading days, exit with 8% stop-loss."
                ),
            },
            {
                "title": "FBM KLCI Sector Rotation — OPR Cycle and Rate Sensitivity",
                "domain": "research",
                "content": (
                    "Bank Negara Malaysia (BNM) sets the Overnight Policy Rate (OPR), the Malaysian "
                    "policy interest rate. OPR directly impacts bank NIM (net interest margin) and "
                    "therefore Banking sector earnings. "
                    "Rate hike cycle → positive for Banking (1155.KL, 1295.KL, 1023.KL, 1066.KL); "
                    "negative for REITs and high-yield bonds proxies. "
                    "Rate cut cycle → positive for REITs, Utilities, Consumer Discretionary; "
                    "negative for Banking NIM. "
                    "Utilities (Tenaga 5347.KL) are quasi-bond proxies; regulated tariff reviews "
                    "(every 3 years) are key events. Telcos (Maxis 6012.KL, CelcomDigi 6947.KL) "
                    "also behave as rate-sensitive defensive stocks. "
                    "OPR history 2022-2024: raised from 1.75% to 3.00%, held through 2023-2024. "
                    "Sector rotation signal: BNM press release dates are known in advance; "
                    "positioning ahead of OPR decisions is a documented alpha source."
                ),
            },
            {
                "title": "MSCI Emerging Markets Index — Foreign Flow Events on Bursa Malaysia",
                "domain": "research",
                "content": (
                    "Malaysia is included in the MSCI Emerging Markets (EM) index. MSCI conducts "
                    "quarterly index reviews (February, May, August, November). Additions to MSCI EM "
                    "or MSCI Malaysia indices trigger forced buying from passive EM-tracking funds. "
                    "Deletions trigger forced selling. These events are announced ~4 weeks before "
                    "effective date. Historical pattern: announced additions appreciate 3-8% between "
                    "announcement and effective date (front-running by active managers), then give "
                    "back 1-3% in the week after inclusion. "
                    "Largest MSCI-linked foreign ownership stocks: Maybank, CIMB, Public Bank, Tenaga. "
                    "When EM funds face net outflows (USD strength, risk-off), Malaysian large-caps "
                    "with high foreign ownership (>20%) are disproportionately sold. "
                    "Tracking foreign equity flow data is available via Bursa Malaysia monthly "
                    "statistics and Bank Negara."
                ),
            },
        ]

        results = []
        for fact in facts:
            try:
                r = kb.ingest_text(
                    content=fact["content"],
                    title=fact["title"],
                    domain="research",
                    source_url="",
                )
                results.append({"title": fact["title"], "doc_id": r.get("doc_id"), "ok": True})
            except Exception as e:
                results.append({"title": fact["title"], "ok": False, "error": str(e)})

        ok_count = sum(1 for r in results if r["ok"])
        self.log_daemon("INFO", f"KB seeding complete: {ok_count}/{len(facts)} facts ingested")
        return {"seeded": ok_count, "total": len(facts), "results": results}

    # ── Screener-driven idea generation ──────────────────────────────────────

    def generate_screener_ideas(self) -> int:
        """
        Run all 8 KLSEProactiveScreener screens and generate one idea per stock
        (up to top 3 from each screen). Ideas are saved directly to gate0/pending.

        Returns: number of ideas generated.
        """
        from data.klse_screener.screener import KLSEProactiveScreener
        from data.database import db_session

        screener = KLSEProactiveScreener()
        all_results = screener.run_all_screens()

        # Store raw screener results
        from datetime import date as _date
        run_date = _date.today().isoformat()
        try:
            with db_session() as conn:
                for screen_name, result in all_results.items():
                    for stock in result["stocks"]:
                        conn.execute(
                            """
                            INSERT INTO screener_results
                              (screen_name, ticker, name, dy, pe, pb, roe, price,
                               matched_criteria, run_date)
                            VALUES (?,?,?,?,?,?,?,?,?,?)
                            """,
                            (
                                screen_name,
                                stock.get("ticker"),
                                stock.get("name"),
                                stock.get("dy"),
                                stock.get("pe"),
                                stock.get("pb"),
                                stock.get("roe"),
                                stock.get("price"),
                                result["description"],
                                run_date,
                            ),
                        )
        except Exception as e:
            self.log_daemon("WARN", f"screener_results insert failed (non-blocking): {e}")

        generated = 0
        for screen_name, result in all_results.items():
            angle = result["idea_angle"]
            for stock in result["stocks"][:3]:
                if not stock.get("ticker") or not stock["ticker"].endswith(".KL"):
                    continue
                try:
                    sticker = stock["ticker"]
                    sname = stock.get("name", sticker)

                    # KB graph grounding — the screener path was the dominant
                    # idea producer yet injected zero accumulated knowledge.
                    kb_block, kb_slugs = "", []
                    try:
                        from knowledge.search.retriever import retrieve, assemble_context
                        kb_results = retrieve(
                            f"{screen_name} {sname} Bursa Malaysia strategy",
                            k=4, hops=2,
                        )
                        if kb_results:
                            kb_slugs = [r["slug"] for r in kb_results]
                            kb_block = "\n" + assemble_context(kb_results, max_chars=1500) + "\n"
                            self.log_daemon(
                                "INFO",
                                f"KB graph context: {len(kb_results)} nodes for "
                                f"screener idea {sticker} ({screen_name})",
                            )
                    except Exception as _kb_exc:
                        self.log_daemon("WARN", f"KB context fetch failed (non-blocking): {_kb_exc}")

                    json_template = (
                        '{"title": str, "hypothesis": str, '
                        f'"ticker": "{sticker}", '
                        '"timeframe": "1d", '
                        '"factor_formula": str, '
                        '"novelty_score": float, '
                        '"logic_score": float}'
                    )
                    prompt = (
                        f"Generate one specific trading idea for "
                        f"{sname} ({sticker}) on Bursa Malaysia.\n\n"
                        f"Flagged by '{screen_name}' screen: {result['description']}\n\n"
                        f"Current signals:\n"
                        f"  Price: MYR {stock.get('price')}\n"
                        f"  DY: {stock.get('dy')}%\n"
                        f"  PE: {stock.get('pe')}\n"
                        f"  P/B: {stock.get('pb')}\n"
                        f"  ROE: {stock.get('roe')}%\n"
                        f"{kb_block}\n"
                        f"RULES:\n"
                        f"- Fundamentals above = WHY this stock is interesting "
                        f"(stock selection context only)\n"
                        f"- Ground the mechanism in the knowledge-graph context above "
                        f"where relevant; address any CONTRADICTS flags\n"
                        f"- factor_formula MUST use ONLY price/volume signals "
                        f"computable from Yahoo Finance OHLCV\n"
                        f"- Entry/exit via RSI, MA, momentum, volume, price patterns ONLY\n"
                        f"- Target ticker: {sticker}\n"
                        f"- Timeframe: 1d or 1wk\n\n"
                        f"Return JSON only:\n{json_template}"
                    )
                    idea = self.call_claude_json(
                        SYSTEM,
                        [{"role": "user", "content": prompt}],
                        model=MODEL_FAST,
                        task_label="screener_idea",
                    )
                    if not idea or "title" not in idea or idea.get("error"):
                        continue
                    # Enforce the ticker from the screener (don't let Claude hallucinate)
                    idea["ticker"] = stock["ticker"]
                    idea["screen_source"] = screen_name
                    if kb_slugs:
                        idea["kb_context"] = kb_slugs

                    # Apply infeasibility filter
                    filtered = self._filter_infeasible([idea])
                    if not filtered:
                        continue
                    idea = filtered[0]

                    idea_id = self.save_idea(idea)
                    # Store screen_source
                    with db_session() as conn:
                        conn.execute(
                            "UPDATE alpha_ideas SET screen_source=? WHERE id=?",
                            (screen_name, idea_id),
                        )
                    generated += 1

                except Exception as e:
                    self.log_daemon(
                        "WARN",
                        f"generate_screener_ideas: error on {stock.get('ticker')}: {e}",
                    )

        self.log_daemon("INFO", f"Screener ideas generated: {generated}")
        return generated

    # ── run() ─────────────────────────────────────────────────────────────────

    def run(self, task: dict) -> dict:
        action = task.get("action", "generate")

        if action == "generate":
            ideas = self.generate_ideas(
                topic=task.get("topic"),
                count=task.get("count", 5),
            )
            ids = []
            for idea in ideas:
                try:
                    iid  = self.save_idea(idea)
                    gate = self.score_gate0(iid)
                    ids.append({"id": iid, "title": idea.get("title"), "ticker": idea.get("ticker"), "gate0": gate})
                except ValueError as e:
                    self.log_daemon("ERROR", str(e))
            return {"action": "generate", "ideas_created": len(ids), "results": ids}

        elif action == "screen_generate":
            ideas = self.screen_and_generate(count=task.get("count", 5))
            ids = []
            for idea in ideas:
                try:
                    iid  = self.save_idea(idea)
                    gate = self.score_gate0(iid)
                    ids.append({"id": iid, "title": idea.get("title"), "ticker": idea.get("ticker"), "gate0": gate})
                except ValueError as e:
                    self.log_daemon("ERROR", str(e))
            return {"action": "screen_generate", "ideas_created": len(ids), "results": ids}

        elif action == "research":
            return self.research_idea(task.get("idea_id"))

        elif action == "score_gate0":
            return self.score_gate0(task.get("idea_id"))

        elif action == "seed_kb":
            return self.seed_knowledge_base()

        return {"error": f"Unknown action: {action}"}
