"""
EventClassifier — classifies market events for Bursa Malaysia trading signals.
Uses Claude Haiku (fast/cheap) to analyse news and determine actionability.
"""
import json
import logging
from datetime import datetime

from agents.base_agent import BaseAgent
from config.settings import MODEL_FAST

logger = logging.getLogger(__name__)

HISTORICAL_EDGES = {
    "earnings_beat": {
        "description": "PEAD drift on Bursa",
        "edge": "+4.2% avg over 30 days",
        "confidence_boost": 0.15,
        "min_magnitude": "medium",
    },
    "earnings_miss": {
        "description": "Negative PEAD on Bursa",
        "edge": "-3.8% avg over 30 days",
        "confidence_boost": 0.15,
        "min_magnitude": "medium",
    },
    "dividend_declared": {
        "description": "Pre-ex-dividend drift",
        "edge": "+1.9% avg over 15 days pre-ex",
        "confidence_boost": 0.10,
        "min_magnitude": "low",
    },
    "contract_win": {
        "description": "Contract announcement drift",
        "edge": "+3.2% avg over 10 days",
        "confidence_boost": 0.10,
        "min_magnitude": "medium",
        "min_value_myr": 50_000_000,
    },
    "bonus_issue": {
        "description": "Bonus issue retail excitement",
        "edge": "+2.8% avg over 10 days pre-ex",
        "confidence_boost": 0.12,
        "min_magnitude": "low",
    },
    "opr_hike": {
        "description": "NIM expansion for banks",
        "edge": "+2.5% avg over 10 days",
        "confidence_boost": 0.20,
        "affected_tickers": ["1155.KL", "1023.KL", "1295.KL", "1066.KL", "5819.KL"],
    },
    "opr_cut": {
        "description": "NIM compression for banks",
        "edge": "-2.0% avg over 10 days",
        "confidence_boost": 0.20,
        "affected_tickers": ["1155.KL", "1023.KL", "1295.KL", "1066.KL", "5819.KL"],
    },
    "cpo_move": {
        "description": "CPO lag on plantation stocks",
        "edge": "lag correlation r=0.62 at 3 days",
        "confidence_boost": 0.08,
        "affected_tickers": ["5285.KL", "2291.KL", "5182.KL", "5069.KL", "2445.KL"],
    },
    "crude_oil_move": {
        "description": "Oil price lag on O&G stocks",
        "edge": "lag correlation r=0.55 at 3 days",
        "confidence_boost": 0.08,
        "affected_tickers": ["5398.KL", "5183.KL", "6033.KL", "7277.KL"],
    },
    "china_pmi_weak": {
        "description": "Export demand reduction",
        "edge": "-1.5% avg on industrial stocks",
        "confidence_boost": 0.05,
        "affected_tickers": ["4197.KL"],
    },
    "analyst_upgrade": {
        "description": "Analyst upgrade momentum",
        "edge": "+2.1% avg over 10 days",
        "confidence_boost": 0.08,
        "min_magnitude": "low",
    },
}

CLASSIFIER_SYSTEM = """You are an expert event classifier for Bursa Malaysia equity markets.
Your job is to analyse market news and determine:
1. What type of event this is
2. Which stocks are affected and how
3. Whether there is a systematic trading edge
4. How confident you are in the signal

You know these facts about Bursa Malaysia:
- PEAD (Post-Earnings Announcement Drift) averages +4.2% over 30 days for earnings beats
- Pre-ex-dividend drift averages +1.9% over 15 days
- Contract wins >MYR 50M drive +3.2% over 10 days
- BNM OPR hikes drive banking NIM expansion (+2.5%)
- CPO price moves lag into plantation stocks 2-5 days
- Brent crude price moves lag into O&G stocks 2-5 days
- China PMI weakness hits Malaysian industrial exports

Be conservative — only flag as actionable when there is clear historical precedent for edge.
Macro/geopolitical news should almost never be actionable — flag as context only.
Always return valid JSON. Do not include any explanation outside the JSON object."""


class EventClassifier(BaseAgent):
    name = "EventClassifier"
    description = "Classifies market events for Bursa Malaysia trading signals"
    default_model = MODEL_FAST

    def run(self, task: dict) -> dict:
        return self.classify(task)

    def classify(self, event_raw: dict) -> dict:
        """
        Takes raw event dict, returns classified dict with trading signal assessment.
        Falls back to rule-based classification on Claude failure.
        """
        ticker_hint = event_raw.get("ticker") or "None"
        body_snippet = (event_raw.get("body") or "")[:500]

        prompt = f"""Classify this market event for Bursa Malaysia trading:

Source: {event_raw.get('source', 'unknown')}
Headline: {event_raw.get('headline', '')}
Body: {body_snippet}
Ticker mentioned: {ticker_hint}
Published: {event_raw.get('published_at', '')}

Return JSON:
{{
  "event_type": "<earnings_beat|earnings_miss|dividend_declared|contract_win|bonus_issue|opr_hike|opr_cut|opr_hold|china_pmi_strong|china_pmi_weak|fed_hike|fed_cut|fed_hold|cpo_move|crude_oil_move|analyst_upgrade|analyst_downgrade|macro_context|irrelevant>",
  "ticker": "<primary affected stock e.g. 1155.KL, or null>",
  "affected_tickers": ["<list of affected .KL tickers>"],
  "company": "<company name or null>",
  "sentiment": "<positive|negative|neutral>",
  "magnitude": "<high|medium|low>",
  "is_actionable": <true|false>,
  "confidence": <0.0 to 1.0>,
  "historical_edge": "<e.g. +4.2% / 30d PEAD, or null>",
  "reasoning": "<1-2 sentences why>",
  "suggested_idea_title": "<if is_actionable: suggested idea title, else null>",
  "suggested_hypothesis": "<if is_actionable: 1-sentence hypothesis, else null>"
}}"""

        try:
            result = self.call_claude_json(
                system=CLASSIFIER_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
                model=MODEL_FAST,
                max_tokens=800,
                task_label="event_classify",
            )
            if "error" in result:
                return self._rule_based_fallback(event_raw)
        except Exception as exc:
            logger.warning(f"EventClassifier: Claude call failed: {exc}")
            return self._rule_based_fallback(event_raw)

        # Apply confidence boost from HISTORICAL_EDGES if event_type matches
        event_type = result.get("event_type", "macro_context")
        if event_type in HISTORICAL_EDGES:
            boost = HISTORICAL_EDGES[event_type].get("confidence_boost", 0.0)
            current = float(result.get("confidence", 0.5))
            result["confidence"] = min(1.0, current + boost)
            if not result.get("historical_edge"):
                result["historical_edge"] = HISTORICAL_EDGES[event_type].get("edge")

        # Propagate pre-set fields from commodity monitor (already classified)
        if event_raw.get("confidence") and not result.get("confidence"):
            result["confidence"] = event_raw["confidence"]
        if event_raw.get("is_actionable") and result.get("is_actionable") is None:
            result["is_actionable"] = event_raw["is_actionable"]

        # Merge affected_tickers from historical_edges for commodity/OPR events
        edge_tickers = HISTORICAL_EDGES.get(event_type, {}).get("affected_tickers", [])
        result_tickers = result.get("affected_tickers") or []
        if event_raw.get("affected_tickers"):
            result_tickers = list(set(result_tickers + event_raw["affected_tickers"]))
        elif edge_tickers:
            result_tickers = list(set(result_tickers + edge_tickers))
        result["affected_tickers"] = result_tickers

        # Ensure is_actionable is boolean
        result["is_actionable"] = bool(result.get("is_actionable", False))
        result["confidence"] = float(result.get("confidence", 0.3))
        result["classified_at"] = datetime.utcnow().isoformat()

        return result

    def determine_action(self, classified: dict) -> str:
        """
        Decide what to do with a classified event.

        gate0_idea  → confidence ≥ 0.75 AND is_actionable = True
        alert       → confidence 0.50–0.74 AND is_actionable = True
        kb_only     → confidence < 0.50 OR is_actionable = False
        ignore      → event_type == irrelevant
        """
        event_type = classified.get("event_type", "macro_context")
        confidence = float(classified.get("confidence", 0.0))
        is_actionable = bool(classified.get("is_actionable", False))

        if event_type == "irrelevant":
            return "ignore"

        # Geopolitical or political → always KB only
        if any(w in event_type for w in ("geo", "political")):
            return "kb_only"

        # OPR decisions → always alert (require human review)
        if event_type.startswith("opr_"):
            return "alert"

        # BNM / Fed / ECB / BOJ decisions → alert
        if event_type in ("bnm_opr", "fed_decision", "ecb_decision", "boe_decision", "boj_decision"):
            return "alert"

        if is_actionable and confidence >= 0.75:
            return "gate0_idea"

        if is_actionable and confidence >= 0.50:
            return "alert"

        return "kb_only"

    def _rule_based_fallback(self, event_raw: dict) -> dict:
        """Simple rule-based classification when Claude is unavailable."""
        headline_lower = (event_raw.get("headline") or "").lower()
        event_type = "macro_context"
        sentiment = "neutral"
        is_actionable = False
        confidence = 0.30

        # Earnings
        if any(w in headline_lower for w in ["earnings beat", "profit up", "net profit", "revenue up", "eps beat"]):
            event_type = "earnings_beat"
            sentiment = "positive"
            is_actionable = True
            confidence = 0.55
        elif any(w in headline_lower for w in ["earnings miss", "profit down", "net loss", "revenue miss"]):
            event_type = "earnings_miss"
            sentiment = "negative"
            is_actionable = True
            confidence = 0.55

        # Dividend
        elif any(w in headline_lower for w in ["dividend", "distribution"]):
            event_type = "dividend_declared"
            sentiment = "positive"
            is_actionable = True
            confidence = 0.60

        # Contract
        elif any(w in headline_lower for w in ["contract", "letter of award", "awarded"]):
            event_type = "contract_win"
            sentiment = "positive"
            is_actionable = True
            confidence = 0.50

        # OPR
        elif any(w in headline_lower for w in ["opr", "overnight policy rate", "interest rate decision", "bnm"]):
            event_type = "opr_hold"
            sentiment = "neutral"
            is_actionable = False
            confidence = 0.70

        # CPO
        elif any(w in headline_lower for w in ["palm oil", "cpo"]):
            event_type = "cpo_move"
            sentiment = "positive" if any(w in headline_lower for w in ["rise", "up", "gain", "surge"]) else "negative"
            is_actionable = True
            confidence = 0.55

        # Crude oil
        elif any(w in headline_lower for w in ["crude oil", "brent", "oil price"]):
            event_type = "crude_oil_move"
            sentiment = "positive" if any(w in headline_lower for w in ["rise", "up", "gain", "surge"]) else "negative"
            is_actionable = True
            confidence = 0.50

        # Pre-set event_type from commodity monitor
        if event_raw.get("event_type") and event_raw["event_type"] != "general":
            event_type = event_raw["event_type"]
            if event_raw.get("sentiment"):
                sentiment = event_raw["sentiment"]
            if event_raw.get("is_actionable"):
                is_actionable = True
            if event_raw.get("confidence"):
                confidence = float(event_raw["confidence"])

        # Apply historical edge boost
        if event_type in HISTORICAL_EDGES:
            boost = HISTORICAL_EDGES[event_type].get("confidence_boost", 0.0)
            confidence = min(1.0, confidence + boost)

        return {
            "event_type": event_type,
            "ticker": event_raw.get("ticker"),
            "affected_tickers": event_raw.get("affected_tickers", []),
            "company": event_raw.get("company"),
            "sentiment": sentiment,
            "magnitude": event_raw.get("magnitude", "medium"),
            "is_actionable": is_actionable,
            "confidence": confidence,
            "historical_edge": HISTORICAL_EDGES.get(event_type, {}).get("edge"),
            "reasoning": "Rule-based fallback classification (Claude unavailable)",
            "suggested_idea_title": None,
            "suggested_hypothesis": None,
            "classified_at": datetime.utcnow().isoformat(),
        }
