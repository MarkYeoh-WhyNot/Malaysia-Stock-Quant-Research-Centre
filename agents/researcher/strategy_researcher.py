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
SYSTEM = """You are a quantitative equity researcher whose sole focus is the Bursa Malaysia
stock exchange (KLSE). You analyse, score, and research strategies that trade individual
Malaysian LISTED STOCKS. You have NO knowledge of foreign-exchange trading and you NEVER
produce strategies involving currency pairs, forex instruments, or spot FX.

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
""".format(universe=_UNIVERSE_FULL)


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

    # ── Idea generation ────────────────────────────────────────────────────────

    def generate_ideas(self, topic: str = None, count: int = 5) -> list:
        # ── KB context injection ──────────────────────────────────────────────
        # Search the knowledge base for relevant documents before calling Claude.
        # Failure here is non-blocking — generation continues without KB context.
        kb_context = ""
        try:
            from knowledge.ingestion.kb_ingester import KBIngester
            ingester   = KBIngester()
            query      = topic if topic else "Bursa Malaysia equity strategy alpha factor"
            kb_results = ingester.search(query, limit=5)
            if kb_results:
                kb_context = "\nRelevant knowledge base context:\n"
                for doc in kb_results:
                    snippet = (doc.get("summary") or "")[:150]
                    kb_context += f"- {doc['title']}: {snippet}\n"
                kb_context += (
                    "\nUse the above KB context to generate more specific, "
                    "locally-grounded ideas referencing real Malaysian market "
                    "techniques, stocks, and factors described above.\n"
                )
                self.log_daemon("INFO", f"KB context: {len(kb_results)} documents found for idea generation")
            else:
                self.log_daemon("INFO", "KB context: 0 documents found — generating without KB context")
        except Exception as e:
            self.log_daemon("WARN", f"KB context fetch failed (non-blocking): {e}")

        topic_line = f"Focus exclusively on: {topic}" if topic else (
            "Cover a diverse mix: at least one technical, one fundamental/value, "
            "one event-driven, and one sector-rotation idea."
        )
        prompt = f"""Generate exactly {count} quantitative equity alpha ideas for Bursa Malaysia stocks.
{kb_context}

{topic_line}

HARD RULES — VIOLATIONS WILL CAUSE THE ENTIRE RESPONSE TO BE DISCARDED:
1. The "ticker" field MUST be a Yahoo Finance .KL symbol (e.g. 1155.KL, 5347.KL).
   DO NOT use currency pair notation (EUR_USD, USD_JPY, etc.) anywhere.
2. The "company" field MUST be the actual company name (e.g. "Maybank", "Tenaga Nasional").
3. The "sector" field MUST be a Bursa Malaysia sector name (Banking, Plantations, etc.).
4. "timeframe" MUST be "1d" (daily bars — KLSE primary timeframe).
5. "holding_period" must express weeks or months, NOT pips or ticks.
6. "factor_formula" must describe a STOCK price/fundamental signal, NOT an FX rate signal.

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
                # Attempt recovery: look for a .KL pattern anywhere in the idea
                all_text = json.dumps(idea)
                found = re.findall(r'\b\d{4}[A-Z0-9]*\.KL\b', all_text)
                if found:
                    idea["ticker"] = found[0]
                    self.log_daemon("WARN", f"Corrected ticker for '{idea.get('title','?')}' → {found[0]}")
                else:
                    self.log_daemon("WARN", f"Discarded idea with invalid ticker '{ticker}': {idea.get('title','?')}")
                    continue
            clean.append(idea)

        self.log_daemon("INFO", f"Generated {len(clean)}/{len(raw_ideas)} clean KLSE equity ideas")
        return clean

    # ── Persistence ────────────────────────────────────────────────────────────

    def save_idea(self, idea: dict) -> int:
        title  = idea.get("title", "idea")
        slug   = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
        slug   = f"{datetime.utcnow().strftime('%Y-%m-%d')}-{slug}"
        ticker = idea.get("ticker") or "1155.KL"

        # Final safety gate — refuse to persist FX ideas
        if not self._is_equity_ticker(ticker):
            raise ValueError(
                f"save_idea() refused: ticker '{ticker}' is not a valid .KL symbol. "
                f"Title: '{title}'"
            )

        with db_session() as conn:
            conn.execute("""
                INSERT OR IGNORE INTO alpha_ideas
                  (slug, title, hypothesis, pair, timeframe, factor_formula,
                   data_sources, stage, status, novelty_score, logic_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'gate0', 'pending', ?, ?)
            """, (
                slug,
                title,
                idea.get("hypothesis", ""),
                ticker,                               # stored in `pair` column
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

    def score_gate0(self, idea_id: int) -> dict:
        with db_session() as conn:
            row = conn.execute("SELECT * FROM alpha_ideas WHERE id=?", (idea_id,)).fetchone()
        if not row:
            return {"error": f"Idea {idea_id} not found"}

        ticker     = row["pair"] or "1155.KL"
        stock_info = KLCI_BY_SYMBOL.get(ticker, {})
        company    = stock_info.get("name", ticker)
        sector     = stock_info.get("sector", "Unknown")

        prompt = f"""Score this Bursa Malaysia equity strategy idea at Gate 0 (initial quality screen).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Stock:      {company} ({ticker}) — {sector}
Title:      {row['title']}
Hypothesis: {row['hypothesis']}
Signal:     {row['factor_formula']}
Timeframe:  {row['timeframe']}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Evaluate against these KLSE-specific criteria:

NOVELTY (0–1): Is this genuinely non-obvious for Malaysian equities? Does it go beyond
  simple "buy cheap PE" or "buy strong momentum"? Any KLSE-specific angle?

LOGIC (0–1): Does the economic mechanism hold in Malaysia's market structure?
  Consider: GLC dominance, EPF flows, illiquidity premium, OPR sensitivity.

IMPLEMENTABILITY (0–1): Can a retail/institutional investor execute this on Bursa?
  Consider: lot size (100 shares), stamp duty (0.15%), T+3 settlement,
  short-selling restrictions (approved list only), data availability.

KLSE FIT (0–1): How well does this match Bursa Malaysia's typical alpha sources?

Pass threshold: novelty ≥ 0.55, logic ≥ 0.60, implementability ≥ 0.50

Return JSON only:
{{
  "novelty_score":          0.0,
  "logic_score":            0.0,
  "implementability_score": 0.0,
  "klse_fit_score":         0.0,
  "pass":                   true,
  "rationale":              "2-3 sentences explaining the decision",
  "key_risks":              ["risk1", "risk2"],
  "data_availability":      "high|medium|low",
  "liquidity_concern":      false,
  "short_selling_required": false
}}"""

        result = self.call_claude_json(
            SYSTEM,
            [{"role": "user", "content": prompt}],
            model=MODEL_FAST,
            task_label="gate0_score",
        )
        passed = result.get("pass", False)

        with db_session() as conn:
            conn.execute("""
                UPDATE alpha_ideas
                SET novelty_score=?, logic_score=?, stage=?, status=?,
                    updated_at=datetime('now')
                WHERE id=?
            """, (
                result.get("novelty_score", 0),
                result.get("logic_score", 0),
                "stage1"   if passed else "gate0",
                "active"   if passed else "rejected",
                idea_id,
            ))
            conn.execute("""
                INSERT INTO gate_decisions
                  (idea_id, gate, decision, decided_by, rationale)
                VALUES (?, 'gate0', ?, 'StrategyResearcher', ?)
            """, (idea_id, "approve" if passed else "reject", result.get("rationale", "")))
            conn.execute("""
                INSERT INTO pipeline_events
                  (idea_id, stage, event_type, agent, notes)
                VALUES (?, 'gate0', ?, 'StrategyResearcher', ?)
            """, (idea_id, "advanced" if passed else "rejected", result.get("rationale", "")))

        self.log_daemon(
            "INFO" if passed else "WARN",
            f"Gate 0 {'PASSED' if passed else 'FAILED'}: [{idea_id}] {ticker} — {row['title']}"
            f" (novelty={result.get('novelty_score',0):.2f} logic={result.get('logic_score',0):.2f})",
        )
        return result

    # ── Stage 1 — deep research ────────────────────────────────────────────────

    def research_idea(self, idea_id: int) -> dict:
        with db_session() as conn:
            row = conn.execute("SELECT * FROM alpha_ideas WHERE id=?", (idea_id,)).fetchone()
        if not row:
            return {"error": f"Idea {idea_id} not found"}

        ticker     = row["pair"] or "1155.KL"
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

        prompt = f"""Conduct deep research on this Bursa Malaysia equity strategy.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Stock:      {company} ({ticker}) — {sector}
Title:      {row['title']}
Hypothesis: {row['hypothesis']}
Signal:     {row['factor_formula']}
Timeframe:  {row['timeframe']}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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
Return JSON only:
{{
  "research_score":          0.0,
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
        passed = (
            result.get("pass", False)
            and result.get("research_score", 0) >= GATE_CONFIG.stage1_min_research_score
        )

        with db_session() as conn:
            conn.execute("""
                UPDATE alpha_ideas
                SET research_score=?, factor_formula=?, data_sources=?,
                    stage=?, status=?, updated_at=datetime('now')
                WHERE id=?
            """, (
                result.get("research_score", 0),
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
            f"Stage 1 complete [{idea_id}] {ticker} — score={result.get('research_score',0):.2f} "
            f"pass={passed}",
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
        return [idea for idea in raw if not self._reject_if_fx(idea)
                and self._is_equity_ticker(idea.get("ticker", ""))]

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
