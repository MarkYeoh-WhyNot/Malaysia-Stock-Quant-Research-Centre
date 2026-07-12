import json
import logging
from agents.base_agent import BaseAgent
from config.settings import (
    MODEL_MAIN, MODEL_HEAVY, GATE_CONFIG,
    MARKET_NAME, MARKET_BRIEF, RED_TEAM_ATTACKS, BLUE_DEFENSE_NOTES,
    JUDGE_REJECT_RULE, MARKET_MODE,
)
from data.database import db_session

logger = logging.getLogger(__name__)

# Noun used in the red/blue/judge user prompts. Bursa keeps its historical
# wording byte-identical; other markets get the profile name.
_STRATEGY_NOUN = ("Bursa Malaysia equity strategy" if MARKET_MODE == "bursa"
                  else f"{MARKET_NAME} strategy")

# Market structure brief + attack/defense lines come from the active market
# profile (config/markets/) — Bursa text is verbatim what previously lived here;
# crypto mode swaps in the crypto brief without touching this file.
BURSA_MARKET_BRIEF = MARKET_BRIEF   # legacy name kept for external importers

RED_SYSTEM = f"""You are the Red Team — a skeptical, adversarial quant analyst whose job is to find
every possible flaw, failure mode, and hidden risk in a proposed {MARKET_NAME} strategy.
Be rigorous, specific, and ruthless. Think about: overfitting, data snooping, regime dependency,
liquidity, transaction costs, crowding, correlation with existing factors, tail risks, and
implementation gaps.

{MARKET_BRIEF}

{RED_TEAM_ATTACKS}"""

BLUE_SYSTEM = f"""You are the Blue Team — a constructive quant analyst defending a proposed
{MARKET_NAME} strategy against adversarial critique. For each red-team finding, provide
a concrete mitigation, counter-argument, or robustness check. Be intellectually honest:
acknowledge valid concerns, but fight for viable strategies with specific evidence and fixes.

{MARKET_BRIEF}

{BLUE_DEFENSE_NOTES}"""

JUDGE_SYSTEM = f"""You are the Chief Risk Officer judging a red-team vs blue-team debate about
a {MARKET_NAME} strategy. Weigh the arguments and give a final verdict on whether this
strategy should advance to paper trading. Be balanced but err on the side of caution.
Return structured JSON.

{MARKET_BRIEF}

{JUDGE_REJECT_RULE}"""


FUNDAMENTAL_SCREEN_RED_TEMPLATES = """
SIGNAL-TYPE SPECIFIC ATTACKS — fundamental_screen:
You MUST raise ALL of the following if they apply:

1. Universe size: If the ticker list contains fewer than 20 stocks, attack it directly.
   "Universe size: {n} stocks is insufficient for factor ranking. Minimum 20 stocks needed
   for statistical significance. Top tertile of {n} stocks = 1-3 stocks = dangerous
   concentration risk. A fundamental factor needs breadth to be investable."

2. Look-ahead bias: Quarterly fundamental data (ROE, DER, PE) from KLSE Screener or
   Yahoo Finance may lag actual financial report announcement dates by 30-90 days.
   "The strategy may be using data not available at trade time — this is look-ahead bias.
   If ROE from FY2024 Q3 was reported on 28 Nov 2024 but the backtest uses it from
   1 Oct 2024, every trade in that window is contaminated."

3. Single-period / bull-market bias: If train < val < test Sharpe values improve
   monotonically, attack this directly.
   "Monotonic improvement (train < val < test) indicates the strategy is capturing a
   time-specific bull market, not a persistent factor. Real alpha degrades out-of-sample.
   Improvement is a red flag, not a green flag."

4. Transaction cost drag on thin edge: Quarterly rebalancing of a fundamental screen
   incurs ~0.4% round-trip per trade (commission + stamp duty + slippage).
   "For a portfolio of {k} stocks rebalanced 4x per year = {k*4*0.4:.1f}% annual cost
   drag. If the strategy's gross alpha is less than this, the net edge is negative."
"""

FUNDAMENTAL_SCREEN_BLUE_TEMPLATES = """
SIGNAL-TYPE SPECIFIC DEFENSES — fundamental_screen:
Address these points proactively:

1. Universe expansion: "Strategy can be applied to FBM70 (70 stocks) or KLCI 30 (30 stocks)
   for better statistical power while retaining Bursa-listed stocks. The factor signal
   strengthens with more ranking candidates."

2. Look-ahead bias mitigation: "Use announcement dates from Bursa Malaysia official filings
   (bursamalaysia.com/market_information) rather than KLSE Screener snapshot dates.
   Data can be timestamped to the actual filing date for clean point-in-time backtesting."
"""


def _is_fundamental_screen(idea: dict) -> bool:
    """Return True if the idea uses a fundamental screening signal type."""
    screen_source = (idea.get("screen_source") or "").lower()
    factor = (idea.get("factor_formula") or "").lower()
    hypothesis = (idea.get("hypothesis") or "").lower()
    fundamental_keywords = ["roe", "der", "pe ratio", "p/e", "dividend yield",
                            "earnings yield", "book value", "fundamental_screen",
                            "fundamental screen", "quarterly rebalance"]
    if "fundamental" in screen_source:
        return True
    return any(kw in factor or kw in hypothesis for kw in fundamental_keywords)


class RedBlueTeam(BaseAgent):
    name = "RedBlueTeam"
    description = "Adversarial strategy stress-testing via structured red/blue debate"
    default_model = MODEL_MAIN

    # ------------------------------------------------------------------
    # Red team attack
    # ------------------------------------------------------------------

    def _failure_knowledge(self, idea: dict) -> str:
        """Accumulated failure knowledge for the red team: rejection-pattern
        and note nodes matching the hypothesis, plus contradicts-edge
        neighbors of the idea's own graph node. Real ammunition instead of
        generic skepticism."""
        lines = []
        try:
            from knowledge.search.retriever import retrieve
            hits = retrieve((idea.get("hypothesis") or idea.get("title") or "")[:300],
                            k=4, hops=2, node_types=["rejection_pattern", "note", "finding"])
            for r in hits:
                flag = "⚠ CONTRADICTS: " if r["contradicts"] else ""
                lines.append(f"• {flag}[{r['node_type']}] {r['title']}: "
                             f"{(r['summary'] or '')[:200]}")
        except Exception:
            pass
        try:
            from knowledge.graph import store
            idea_node = None
            if idea.get("id"):
                from data.database import db_session
                with db_session() as conn:
                    n = conn.execute(
                        "SELECT id FROM kb_nodes WHERE ref_table='alpha_ideas' AND ref_id=?",
                        (idea["id"],)).fetchone()
                idea_node = n["id"] if n else None
            if idea_node:
                for nb in store.neighbors(idea_node):
                    if nb["relation"] == "contradicts":
                        lines.append(f"• ⚠ CONTRADICTS this idea directly: {nb['title']}")
        except Exception:
            pass
        if not lines:
            return ""
        block = "\nACCUMULATED FAILURE KNOWLEDGE (use as attack ammunition):\n" + "\n".join(lines[:8])
        return block[:1200] + "\n"

    def _technique_caveats(self, idea: dict) -> str:
        """when_to_avoid caveats for Technique Arsenal entries the idea relies
        on — concrete attack ammunition, mirroring the technique matching in
        StrategyResearcher.research_idea(). Non-blocking."""
        try:
            from knowledge.ingestion.technique_library import TECHNIQUE_LIBRARY
            text = f"{idea.get('factor_formula') or ''} {idea.get('hypothesis') or ''}".lower()
            lines = []
            for key, tech in TECHNIQUE_LIBRARY.items():
                if key.replace("_", " ") in text or key in text:
                    avoid = "; ".join(tech.get("when_to_avoid", [])[:3])
                    if avoid:
                        lines.append(f"• [{key}] avoid when: {avoid}")
                if len(lines) >= 3:
                    break
            if not lines:
                return ""
            block = ("\nTECHNIQUE CAVEATS (attack if the strategy violates "
                     "these):\n" + "\n".join(lines))
            return block[:800] + "\n"
        except Exception:
            return ""

    def red_team_attack(self, idea: dict, backtest_results: dict) -> dict:
        signal_type_context = ""
        if _is_fundamental_screen(idea):
            n = len([t.strip() for t in (idea.get("ticker") or "").split(",") if t.strip()])
            k = max(1, n // 3)
            signal_type_context = FUNDAMENTAL_SCREEN_RED_TEMPLATES.replace(
                "{n}", str(n)
            ).replace("{k}", str(k)).replace("{k*4*0.4:.1f}", f"{k * 4 * 0.4:.1f}")

        failure_knowledge = self._failure_knowledge(idea)
        technique_caveats = self._technique_caveats(idea)

        prompt = f"""Stress-test this {_STRATEGY_NOUN} as a hostile adversary.

Strategy: {idea.get('title')}
Hypothesis: {idea.get('hypothesis')}
Ticker: {idea.get('ticker')} | Timeframe: {idea.get('timeframe')}
Factor: {idea.get('factor_formula')}
Research score: {idea.get('research_score')}

Backtest results:
{json.dumps(backtest_results, indent=2)}
{signal_type_context}{failure_knowledge}{technique_caveats}
Return JSON:
{{
  "critical_flaws": [
    {{"finding": "...", "severity": "high|medium|low", "category": "overfitting|liquidity|regime|cost|data|implementation|other"}}
  ],
  "regime_vulnerabilities": ["..."],
  "overfitting_risk": 0.0,
  "hidden_costs": ["..."],
  "tail_risk_scenarios": ["..."],
  "overall_attack_score": 0.0,
  "kill_recommendation": true,
  "kill_rationale": "..."
}}"""
        result = self.call_claude_json(
            RED_SYSTEM, [{"role": "user", "content": prompt}],
            model=MODEL_MAIN, max_tokens=4096, task_label="red_team_attack"
        )
        if "error" in result or result.get("overall_attack_score") is None:
            self.log_daemon("WARN",
                f"Red team parse failure for idea (result={list(result.keys())}) — "
                f"substituting minimal attack so debate can continue")
            result.setdefault("critical_flaws", [])
            result.setdefault("overall_attack_score", 0.5)
            result.setdefault("kill_recommendation", False)
            result.setdefault("kill_rationale", "[Red team JSON parse failed — scores unavailable]")
            result["_parse_failed"] = True
        self.log_daemon("INFO", f"Red team: {len(result.get('critical_flaws', []))} findings, attack_score={result.get('overall_attack_score')}")
        return result

    # ------------------------------------------------------------------
    # Blue team defense
    # ------------------------------------------------------------------

    def blue_team_defend(self, idea: dict, red_findings: dict) -> dict:
        signal_type_context = ""
        if _is_fundamental_screen(idea):
            signal_type_context = FUNDAMENTAL_SCREEN_BLUE_TEMPLATES

        # Truncate serialised red findings to avoid oversized prompts when red team
        # returns a large response — keep the most diagnostic fields only.
        red_summary = {
            "overall_attack_score": red_findings.get("overall_attack_score"),
            "kill_recommendation":  red_findings.get("kill_recommendation"),
            "kill_rationale":       red_findings.get("kill_rationale"),
            "critical_flaws":       red_findings.get("critical_flaws", [])[:5],
            "regime_vulnerabilities": red_findings.get("regime_vulnerabilities", [])[:3],
            "hidden_costs":         red_findings.get("hidden_costs", [])[:3],
        }
        red_json = json.dumps(red_summary, indent=2)

        prompt = f"""Defend this {_STRATEGY_NOUN} against the following red-team critique.

Strategy: {idea.get('title')}
Hypothesis: {idea.get('hypothesis')}
Ticker: {idea.get('ticker')} | Factor: {idea.get('factor_formula')}

Red team findings:
{red_json}
{signal_type_context}
Return JSON:
{{
  "rebuttals": [
    {{"finding": "...", "rebuttal": "...", "mitigation": "...", "confidence": "high|medium|low"}}
  ],
  "conceded_points": ["..."],
  "proposed_safeguards": ["..."],
  "regime_filters": ["..."],
  "parameter_robustness": "...",
  "overall_defense_score": 0.0,
  "advance_recommendation": true,
  "advance_rationale": "..."
}}"""
        result = self.call_claude_json(
            BLUE_SYSTEM, [{"role": "user", "content": prompt}],
            model=MODEL_MAIN, max_tokens=4096, task_label="blue_team_defend"
        )
        if "error" in result or result.get("overall_defense_score") is None:
            self.log_daemon("WARN",
                f"Blue team parse failure (result={list(result.keys())}) — "
                f"substituting neutral defense so debate can continue")
            result.setdefault("rebuttals", [])
            result.setdefault("conceded_points", [])
            result.setdefault("proposed_safeguards", ["Manual review required — blue team scores unavailable"])
            result.setdefault("overall_defense_score", 0.5)
            result.setdefault("advance_recommendation", True)
            result.setdefault("advance_rationale", "[Blue team JSON parse failed — scores unavailable]")
            result["_parse_failed"] = True
        self.log_daemon("INFO", f"Blue team: defense_score={result.get('overall_defense_score')}, advance={result.get('advance_recommendation')}")
        return result

    # ------------------------------------------------------------------
    # Judicial verdict
    # ------------------------------------------------------------------

    def _judge(self, idea: dict, red: dict, blue: dict, backtest_results: dict) -> dict:
        # Summarise backtest results to keep the judge prompt concise
        bt_summary = {k: backtest_results.get(k) for k in (
            "sharpe_net", "sharpe_is", "sharpe_oos", "max_dd",
            "trades", "trade_count", "regimes_positive",
            "verdict", "verdict_reason", "oos_degradation",
        )}

        prompt = f"""Judge this {_STRATEGY_NOUN} debate and give a final verdict.

Strategy: {idea.get('title')} | Ticker: {idea.get('ticker')}
Backtest: {json.dumps(bt_summary, indent=2)}

Red team (attack_score={red.get('overall_attack_score')}, kill={red.get('kill_recommendation')}):
- Critical flaws: {[f['finding'] for f in red.get('critical_flaws', [])]}
- Kill rationale: {red.get('kill_rationale')}

Blue team (defense_score={blue.get('overall_defense_score')}, advance={blue.get('advance_recommendation')}):
- Conceded: {blue.get('conceded_points')}
- Proposed safeguards: {blue.get('proposed_safeguards')}
- Advance rationale: {blue.get('advance_rationale')}

Return JSON:
{{
  "verdict": "advance|reject|conditional",
  "confidence": 0.0,
  "key_conditions": ["if conditional, list conditions that must be met"],
  "position_size_limit_pct": 1.0,
  "required_safeguards": ["..."],
  "red_score": 0.0,
  "blue_score": 0.0,
  "final_risk_rating": "low|medium|high|very_high",
  "summary": "..."
}}"""
        result = self.call_claude_json(
            JUDGE_SYSTEM, [{"role": "user", "content": prompt}],
            model=MODEL_HEAVY, max_tokens=2048, task_label="red_blue_judge"
        )
        if "error" in result or result.get("verdict") is None:
            self.log_daemon("WARN",
                "Judge parse failure — defaulting to conditional with manual review requirement")
            return {
                "verdict":              "conditional",
                "confidence":           0.5,
                "key_conditions":       ["Manual review required — judge JSON parse failed"],
                "position_size_limit_pct": 0.5,
                "required_safeguards":  ["Human review before paper trading", "Limit position to 0.5% NAV"],
                "red_score":            red.get("overall_attack_score"),
                "blue_score":           blue.get("overall_defense_score"),
                "final_risk_rating":    "medium",
                "summary":              "JSON parse failure — scores unavailable — recommend manual review",
                "_parse_failed":        True,
            }
        return result

    # ------------------------------------------------------------------
    # Full stress test
    # ------------------------------------------------------------------

    def stress_test(self, idea_id: int) -> dict:
        with db_session() as conn:
            row = conn.execute("SELECT * FROM alpha_ideas WHERE id=?", (idea_id,)).fetchone()
            bt_row = conn.execute(
                "SELECT * FROM backtest_runs WHERE idea_id=? ORDER BY created_at DESC LIMIT 1",
                (idea_id,)
            ).fetchone()

        if not row:
            return {"error": f"Idea {idea_id} not found"}

        idea = dict(row)
        backtest_results = dict(bt_row) if bt_row else {}
        if bt_row and bt_row["result_data"]:
            try:
                backtest_results.update(json.loads(bt_row["result_data"]))
            except Exception:
                pass

        self.log_daemon("INFO", f"Red-blue stress test starting: [{idea_id}] {idea['title']}")

        red = self.red_team_attack(idea, backtest_results)
        blue = self.blue_team_defend(idea, red)
        verdict = self._judge(idea, red, blue, backtest_results)

        # Detect parse failures across any step.
        any_parse_failed = (
            red.get("_parse_failed") or
            blue.get("_parse_failed") or
            verdict.get("_parse_failed")
        )

        # A JSON glitch must neither kill a good idea NOR advance an
        # un-reviewed one (the old behaviour overrode reject→conditional and
        # advanced it — a noise-passage hole). Defer instead: write nothing,
        # leave the idea at its current stage, and the next daemon cycle
        # retries the debate from scratch.
        raw_verdict = verdict.get("verdict")
        if any_parse_failed or raw_verdict is None:
            self.log_daemon("WARN",
                f"Red-Blue [{idea_id}]: parse failure — DEFERRING to next cycle "
                f"(red_failed={red.get('_parse_failed')}, blue_failed={blue.get('_parse_failed')}, "
                f"judge_failed={verdict.get('_parse_failed')})")
            return {"idea_id": idea_id, "verdict": "deferred",
                    "deferred": True, "reason": "json_parse_failure"}
        verdict_str = raw_verdict

        should_advance = verdict_str in ("advance", "conditional")

        with db_session() as conn:
            notes = json.dumps({
                "red_score":   red.get("overall_attack_score"),
                "blue_score":  blue.get("overall_defense_score"),
                "verdict":     verdict_str,
                "conditions":  verdict.get("key_conditions", []),
                "safeguards":  verdict.get("required_safeguards", []),
                "parse_failed": bool(any_parse_failed),
            })
            conn.execute("""
                INSERT INTO pipeline_events (idea_id, stage, event_type, agent, notes)
                VALUES (?, 'stage3', ?, 'RedBlueTeam', ?)
            """, (idea_id, "advanced" if should_advance else "rejected", notes))

            rationale = verdict.get("summary") or (
                "JSON parse failure — scores unavailable — recommend manual review"
                if any_parse_failed else ""
            )
            # A CONDITIONAL advance must carry its conditions visibly (audit
            # 2026-07-10: they were logged to pipeline_events but the
            # gate_decisions row looked identical to a clean advance, so the
            # conditions were never seen again).
            if verdict_str == "conditional":
                _conds = verdict.get("key_conditions") or []
                _safes = verdict.get("required_safeguards") or []
                rationale = ("CONDITIONAL ADVANCE — conditions: "
                             + ("; ".join(map(str, _conds)) or "unspecified")
                             + (" | safeguards: " + "; ".join(map(str, _safes))
                                if _safes else "")
                             + " || " + rationale)[:900]
                self.log_daemon(
                    "WARN", f"Red-Blue [{idea_id}] CONDITIONAL advance — "
                            f"conditions: {_conds} safeguards: {_safes}")
            conn.execute("""
                INSERT INTO gate_decisions (idea_id, gate, decision, decided_by, rationale)
                VALUES (?, 'gate3_rb', ?, 'RedBlueTeam', ?)
            """, (idea_id, "approve" if should_advance else "reject", rationale))

            if should_advance:
                conn.execute("""
                    UPDATE alpha_ideas SET stage='stage4a', status='active', updated_at=datetime('now')
                    WHERE id=? AND stage IN ('stage2','stage3')
                """, (idea_id,))
            elif not should_advance and idea.get("stage") not in ("stage4a", "stage4b", "stage5"):
                conn.execute("""
                    UPDATE alpha_ideas SET status='rejected', updated_at=datetime('now')
                    WHERE id=?
                """, (idea_id,))

        self.log_daemon(
            "INFO" if should_advance else "WARN",
            f"Red-Blue verdict [{idea_id}]: {verdict_str} | risk={verdict.get('final_risk_rating')}"
        )
        return {
            "idea_id": idea_id,
            "verdict": verdict_str,
            "red": red,
            "blue": blue,
            "judge": verdict,
            "advanced": should_advance,
        }

    # ------------------------------------------------------------------
    # run()
    # ------------------------------------------------------------------

    def run(self, task: dict) -> dict:
        action = task.get("action", "stress_test")
        if action == "stress_test":
            idea_id = task.get("idea_id")
            if not idea_id:
                return {"error": "idea_id required"}
            return self.stress_test(idea_id)
        elif action == "red_team":
            idea_id = task.get("idea_id")
            with db_session() as conn:
                row = conn.execute("SELECT * FROM alpha_ideas WHERE id=?", (idea_id,)).fetchone()
            return self.red_team_attack(dict(row), task.get("backtest_results", {}))
        elif action == "blue_team":
            idea_id = task.get("idea_id")
            with db_session() as conn:
                row = conn.execute("SELECT * FROM alpha_ideas WHERE id=?", (idea_id,)).fetchone()
            return self.blue_team_defend(dict(row), task.get("red_findings", {}))
        return {"error": f"Unknown action: {action}"}
