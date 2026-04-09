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
from agents.base_agent import BaseAgent
from config.settings import (
    MODEL_MAIN, MODEL_FAST, GATE_CONFIG,
    KLCI_STOCKS, KLCI_BY_SYMBOL, KLCI_SECTORS,
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
• Settlement:      T+3 (Bursa CDS), no short-selling for most stocks
• Transaction costs: ~0.30% round trip (brokerage 0.08–0.42% + stamp duty 0.15%)
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
• Settlement:      T+3 (Bursa Depository)
• Lot size:        100 shares minimum (1 board lot)
• Stamp duty:      0.15% per contract (max RM200) on purchase
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
exploit; GLC concentration means events move whole sectors; T+3 settlement makes very short-term
strategies expensive; approved-list restrictions mean long-only is the only viable approach for
most stocks; Yahoo Finance .KL data has quality issues (dividend adjustments, split gaps, stale
fundamentals). Score with deep scepticism and give low scores unless the edge is compelling."""


class StrategyResearcher(BaseAgent):
    name = "StrategyResearcher"
    description = "Bursa Malaysia equity alpha generation, Gate 0 screening, Stage 1 deep research"
    default_model = MODEL_MAIN

    # ── Validation ────────────────────────────────────────────────────────────

    @staticmethod
    def _is_equity_ticker(ticker: str) -> bool:
        """Return True only if ticker looks like a Bursa .KL symbol."""
        if not ticker:
            return False
        # Accept: 1155.KL, 5347.KL, 5235SS.KL, 6947.KL etc.
        return bool(re.match(r'^\d{4}[A-Z0-9]*\.KL$', ticker.strip()))

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

            # Check 4: scores
            novelty = float(idea.get("novelty_score", 0) or 0)
            logic   = float(idea.get("logic_score", 0) or 0)
            if novelty < 0.5 or logic < 0.6:
                self.log_daemon(
                    "INFO",
                    f"generate_ideas: filtered '{title[:50]}' "
                    f"(low scores: novelty={novelty:.2f} logic={logic:.2f})",
                )
                continue

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
        # ── 1. KB context — search relevant documents first ──────────────────
        kb_context = ""
        try:
            from knowledge.ingestion.kb_ingester import KBIngester
            ingester   = KBIngester()
            query      = topic if topic else "Bursa Malaysia equity strategy alpha factor"
            kb_results = ingester.search(query, limit=5)
            if kb_results:
                kb_context = "\nKNOWLEDGE BASE CONTEXT — use these research findings to ground your ideas:\n"
                for doc in kb_results:
                    snippet = (doc.get("summary") or "")[:300]
                    domain  = doc.get("domain", "")
                    kb_context += f"- [{domain}] {doc['title']}: {snippet}\n"
                kb_context += (
                    "\nGenerate ideas that reference specific techniques and factors from the above "
                    "KB documents where applicable.\n"
                )
                self.log_daemon("INFO", f"KB context: {len(kb_results)} documents found for idea generation")
            else:
                self.log_daemon("INFO", "KB context: 0 documents found — generating without KB context")
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
4. "timeframe" MUST be "1d" (daily bars — KLSE primary timeframe).
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

Return a valid JSON array of exactly {count} objects. Each object:
{{
  "title":          "Concise strategy name (e.g. 'Maybank Dividend Capture Pre-Ex')",
  "hypothesis":     "Why this stock/signal generates alpha on Bursa Malaysia",
  "ticker":         "NNNN.KL  — a valid Bursa .KL symbol, NOT a currency pair",
  "company":        "Full company name",
  "sector":         "Bursa sector (Banking / Plantations / Utilities / etc.)",
  "timeframe":      "1d",
  "factor_formula": "Precise signal construction: e.g. 'Enter long when 20-day SMA crosses above 50-day SMA and RSI(14) < 65. Exit when RSI > 75 or -8% stop-loss.'",
  "data_sources":   ["Yahoo Finance daily OHLCV", "Bursa quarterly earnings releases"],
  "strategy_type":  "momentum | value | quality | mean_reversion | event_driven | sector_rotation | technical",
  "holding_period": "e.g. 2-6 weeks",
  "novelty_score":  0.75,
  "logic_score":    0.80,
  "concerns":       "Key implementation risks on Bursa (liquidity, lot size, corporate actions)"
}}"""

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
        self.log_daemon("INFO", f"Generated {len(clean)}/{len(raw_ideas)} clean KLSE equity ideas (post-filter)")
        return clean

    # ── Persistence ────────────────────────────────────────────────────────────

    def save_idea(self, idea: dict) -> int:
        title  = idea.get("title", "idea")
        slug   = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
        slug   = f"{datetime.utcnow().strftime('%Y-%m-%d')}-{slug}"
        ticker = idea.get("ticker") or "1155.KL"

        # For sector ideas ticker may be comma-separated (e.g. "5225.KL,5878.KL,7081.KL").
        # Validate only the primary (first) ticker — the full list is stored for reference.
        primary_ticker = ticker.split(",")[0].strip()
        if not self._is_equity_ticker(primary_ticker):
            raise ValueError(
                f"save_idea() refused: primary ticker '{primary_ticker}' "
                f"(from '{ticker[:80]}') is not a valid .KL symbol. Title: '{title}'"
            )
        ticker = ticker  # keep the full comma-separated string in the DB

        with db_session() as conn:
            conn.execute("""
                INSERT OR IGNORE INTO alpha_ideas
                  (slug, title, hypothesis, ticker, timeframe, factor_formula,
                   data_sources, stage, status, novelty_score, logic_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'gate0', 'pending', ?, ?)
            """, (
                slug,
                title,
                idea.get("hypothesis", ""),
                ticker,                               # stored in `ticker` column
                idea.get("timeframe", "1d"),
                idea.get("factor_formula", ""),
                json.dumps(idea.get("data_sources", [])),
                float(idea.get("novelty_score", 0.0)),
                float(idea.get("logic_score", 0.0)),
            ))
            row = conn.execute("SELECT id FROM alpha_ideas WHERE slug=?", (slug,)).fetchone()

        self.log_daemon("INFO", f"Saved idea [{row['id']}] {ticker} — {slug}")
        return row["id"]

    # ── Gate 0 — initial screening ─────────────────────────────────────────────

    @staticmethod
    def _compute_feasibility(idea_row, ticker: str, factor_formula: str) -> float:
        """Score 0.0-1.0 based on whether the idea is actually executable on Bursa."""
        score = 0.0
        formula_lower = (factor_formula or "").lower()
        hypothesis_lower = (idea_row["hypothesis"] or "").lower()
        blob = formula_lower + " " + hypothesis_lower

        # +0.3: valid .KL ticker
        if re.match(r'^\d{4}[A-Z0-9]*\.KL$', ticker.strip()):
            score += 0.3

        # +0.2: long-only (no short/pairs language)
        short_keywords = ["short", "pairs", "spread arbitrage", "sell short", "hedge"]
        if not any(kw in blob for kw in short_keywords):
            score += 0.2

        # +0.2: data available via Yahoo Finance .KL
        unavailable_keywords = ["options", "futures contract", "otc", "dark pool",
                                 "tick data", "level 2", "order book", "forex", "fx rate"]
        if not any(kw in blob for kw in unavailable_keywords):
            score += 0.2

        # +0.15: holding period realistic for Bursa T+3
        intraday_keywords = ["intraday", "scalp", "1 minute", "5 minute", "hourly", "hft"]
        short_hold_keywords = ["1 day", "2 day", "3 day", "t+1", "t+2"]
        if any(kw in blob for kw in intraday_keywords):
            score -= 0.3
        elif any(kw in blob for kw in short_hold_keywords):
            score += 0.0   # T+3 makes 1-3 day holds awkward but not impossible
        else:
            score += 0.15   # >5 day holding period — safe for T+3

        # +0.15: factor uses available indicators (price/volume/fundamental)
        exotic_keywords = ["futures price", "options greeks", "cds spread", "credit default",
                           "bond yield curve", "repo rate"]
        if not any(kw in blob for kw in exotic_keywords):
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

        prompt = f"""Evaluate this Bursa Malaysia equity strategy at Gate 0. Your job is to REJECT it
if there are any serious flaws. Be demanding — most ideas should fail.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Stock:      {company} ({ticker}) — {sector}
Title:      {row['title']}
Hypothesis: {row['hypothesis']}
Signal:     {row['factor_formula']}
Timeframe:  {row['timeframe']}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Score each dimension 0.0–1.0 (use low scores liberally):

1. NOVELTY (0–1): Is this genuinely differentiated beyond textbook strategies?
   Score LOW if it is standard momentum/value with no KLSE-specific insight.

2. LOGIC (0–1): Is the economic mechanism sound and SPECIFIC to Bursa Malaysia?
   Must account for GLC dominance, EPF flows, OPR cycle, T+3 settlement friction.
   Score LOW if the rationale is generic (could apply to any market).

3. FEASIBILITY (0–1): Can a real fund execute this on Bursa today?
   Consider: 100-share lot minimum, stamp duty 0.15%, T+3 settlement impact on
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

Deterministic pre-score: feasibility={feasibility:.2f}/1.00 (must be >= 0.60 to pass)

Pass requires ALL of:
  novelty >= 0.60, logic >= 0.65, feasibility >= 0.70,
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
            )
        except Exception as exc:
            self.logger.error(
                f"[score_gate0] Claude API call raised exception for idea {idea_id}: {exc}",
                exc_info=True,
            )
            return {"error": str(exc), "novelty_score": 0.0, "logic_score": 0.0,
                    "feasibility_score": feasibility, "passed": False}

        # Detect silent parse failure — log raw response so we can see what Claude returned
        if "error" in result:
            self.logger.error(
                f"[score_gate0] JSON parse failed for idea {idea_id} ({ticker}). "
                f"error={result['error']!r}\n"
                f"RAW RESPONSE:\n{result.get('raw', '(no raw captured)')[:2000]}"
            )

        novelty      = float(result.get("novelty_score",     result.get("novelty", 0)))
        logic        = float(result.get("logic_score",       result.get("logic",   0)))
        claude_feas  = float(result.get("feasibility_score", 0))
        data_qual    = float(result.get("data_quality_score", 0))
        overfit      = float(result.get("overfitting_risk",  1.0))

        # Log a warning if all scores are suspiciously zero (likely a parse problem)
        if novelty == 0.0 and logic == 0.0 and "error" not in result:
            self.logger.warning(
                f"[score_gate0] Scores all zero for idea {idea_id} — "
                f"possible key mismatch. Keys: {list(result.keys())}\nFull: {result}"
            )

        # Gate 0: ALL FIVE dimensions + deterministic feasibility must pass
        passed = (
            novelty     >= 0.60
            and logic       >= 0.65
            and claude_feas >= 0.70
            and data_qual   >= 0.70
            and overfit     <= 0.40
            and feasibility >= 0.60   # deterministic pre-score
        )

        # Build a clear failure message so rejection_memory gets useful context
        if not passed:
            failed_dims = []
            if novelty     < 0.60: failed_dims.append(f"novelty={novelty:.2f}<0.60")
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
        if not passed:
            proxy = (result.get("price_based_proxy") or "").strip()
            if proxy and data_qual < 0.70:
                # Store proxy in alpha_ideas
                with db_session() as conn:
                    conn.execute(
                        "UPDATE alpha_ideas SET price_based_proxy=? WHERE id=?",
                        (proxy, idea_id),
                    )
                self.log_daemon(
                    "INFO",
                    f"Gate 0 REDIRECT: Try '{proxy}' instead",
                )
                # Auto-resubmit as a new pending Gate 0 idea
                try:
                    proxy_idea = {
                        "title":          f"Price proxy: {row['title'][:40]}",
                        "hypothesis":     proxy,
                        "ticker":         ticker,
                        "timeframe":      row.get("timeframe") or "1d",
                        "factor_formula": proxy,
                        "data_sources":   ["Yahoo Finance daily OHLCV"],
                        "novelty_score":  0.6,
                        "logic_score":    0.65,
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
{papers_section}
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

        result = self.call_claude_json(
            SYSTEM,
            [{"role": "user", "content": prompt}],
            max_tokens=8192,
            task_label="deep_research",
        )

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
                    "Total trading time: approximately 6 hours per day. Settlement is on a T+3 basis "
                    "via Bursa Depository (formerly MECTS). All prices are quoted in Malaysian Ringgit "
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
                    "(2) Stamp duty on purchase: 0.15% of contract value, capped at RM200 per contract. "
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
                        f"  ROE: {stock.get('roe')}%\n\n"
                        f"RULES:\n"
                        f"- Fundamentals above = WHY this stock is interesting "
                        f"(stock selection context only)\n"
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
