import json
import logging
from agents.base_agent import BaseAgent
from config.settings import MODEL_MAIN, MODEL_HEAVY, GATE_CONFIG
from data.database import db_session

logger = logging.getLogger(__name__)

BURSA_MARKET_BRIEF = """
BURSA MALAYSIA MARKET STRUCTURE — MUST KNOW:
- Settlement: T+3 (3 business days). Affects short-term strategies.
- Short-selling: Restricted to ~150 approved securities only.
  Retail traders cannot short most stocks. LONG-ONLY strategies only.
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
- Stamp duty: 0.15% on buy side, capped at RM200. Real cost.
- GLC dynamics: Government-linked companies (Maybank, Tenaga,
  Petronas subsidiaries) have different dynamics — policy-driven.
"""

RED_SYSTEM = f"""You are the Red Team — a skeptical, adversarial quant analyst whose job is to find
every possible flaw, failure mode, and hidden risk in a proposed Bursa Malaysia equity strategy.
Be rigorous, specific, and ruthless. Think about: overfitting, data snooping, regime dependency,
liquidity, transaction costs, crowding, correlation with existing factors, tail risks, and
implementation gaps.

{BURSA_MARKET_BRIEF}

You MUST specifically attack:
- T+3 settlement risk: does the strategy's holding period interact badly with T+3?
- Liquidity risk: can this be executed in 100-share lots without moving the price?
- EPF flow reversal risk: if EPF rebalances away, does the thesis collapse?
- OPR change risk: for banking strategies, how does a 25bp BNM rate change affect the thesis?
- Penny stock risk: is the ticker a low-liquidity or low-price stock with wide spreads?
- Feasibility: can a real retail or institutional investor in Malaysia actually execute this?"""

BLUE_SYSTEM = f"""You are the Blue Team — a constructive quant analyst defending a proposed
Bursa Malaysia equity strategy against adversarial critique. For each red-team finding, provide
a concrete mitigation, counter-argument, or robustness check. Be intellectually honest:
acknowledge valid concerns, but fight for viable strategies with specific evidence and fixes.

{BURSA_MARKET_BRIEF}

When defending, always address Bursa-specific mechanics directly:
- If T+3 is raised: explain how the holding period accommodates settlement.
- If liquidity is raised: cite the stock's average daily volume or lot-size adequacy.
- If EPF flows are raised: explain whether the thesis is EPF-dependent or independent.
- If OPR is raised: quantify the sensitivity and whether the strategy hedges rate risk."""

JUDGE_SYSTEM = f"""You are the Chief Risk Officer judging a red-team vs blue-team debate about
a Bursa Malaysia equity strategy. Weigh the arguments and give a final verdict on whether this
strategy should advance to paper trading. Be balanced but err on the side of caution.
Return structured JSON.

{BURSA_MARKET_BRIEF}

Apply Bursa-specific judgment: reject any strategy that requires short-selling unrestricted
securities, relies on intraday execution, or ignores T+3 settlement constraints."""


class RedBlueTeam(BaseAgent):
    name = "RedBlueTeam"
    description = "Adversarial strategy stress-testing via structured red/blue debate"
    default_model = MODEL_MAIN

    # ------------------------------------------------------------------
    # Red team attack
    # ------------------------------------------------------------------

    def red_team_attack(self, idea: dict, backtest_results: dict) -> dict:
        prompt = f"""Stress-test this Bursa Malaysia equity strategy as a hostile adversary.

Strategy: {idea.get('title')}
Hypothesis: {idea.get('hypothesis')}
Ticker: {idea.get('pair')} | Timeframe: {idea.get('timeframe')}
Factor: {idea.get('factor_formula')}
Research score: {idea.get('research_score')}

Backtest results:
{json.dumps(backtest_results, indent=2)}

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
        self.log_daemon("INFO", f"Red team: {len(result.get('critical_flaws', []))} findings, attack_score={result.get('overall_attack_score')}")
        return result

    # ------------------------------------------------------------------
    # Blue team defense
    # ------------------------------------------------------------------

    def blue_team_defend(self, idea: dict, red_findings: dict) -> dict:
        prompt = f"""Defend this Bursa Malaysia equity strategy against the following red-team critique.

Strategy: {idea.get('title')}
Hypothesis: {idea.get('hypothesis')}
Ticker: {idea.get('pair')} | Factor: {idea.get('factor_formula')}

Red team findings:
{json.dumps(red_findings, indent=2)}

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
        self.log_daemon("INFO", f"Blue team: defense_score={result.get('overall_defense_score')}, advance={result.get('advance_recommendation')}")
        return result

    # ------------------------------------------------------------------
    # Judicial verdict
    # ------------------------------------------------------------------

    def _judge(self, idea: dict, red: dict, blue: dict, backtest_results: dict) -> dict:
        prompt = f"""Judge this Bursa Malaysia equity strategy debate and give a final verdict.

Strategy: {idea.get('title')} | Ticker: {idea.get('pair')}
Backtest: {json.dumps(backtest_results, indent=2)}

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
        return self.call_claude_json(
            JUDGE_SYSTEM, [{"role": "user", "content": prompt}],
            model=MODEL_HEAVY, max_tokens=2048, task_label="red_blue_judge"
        )

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

        # Persist result
        verdict_str = verdict.get("verdict", "reject")
        should_advance = verdict_str in ("advance", "conditional")

        with db_session() as conn:
            notes = json.dumps({
                "red_score": red.get("overall_attack_score"),
                "blue_score": blue.get("overall_defense_score"),
                "verdict": verdict_str,
                "conditions": verdict.get("key_conditions", []),
                "safeguards": verdict.get("required_safeguards", []),
            })
            conn.execute("""
                INSERT INTO pipeline_events (idea_id, stage, event_type, agent, notes)
                VALUES (?, 'stage3', ?, 'RedBlueTeam', ?)
            """, (idea_id, "advanced" if should_advance else "rejected", notes))

            conn.execute("""
                INSERT INTO gate_decisions (idea_id, gate, decision, decided_by, rationale)
                VALUES (?, 'gate3_rb', ?, 'RedBlueTeam', ?)
            """, (idea_id, "approve" if should_advance else "reject", verdict.get("summary", "")))

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
