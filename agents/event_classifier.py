"""
EventClassifier — classifies market events for trading signals.
Uses Claude Haiku (fast/cheap) to analyse news and determine actionability.

Market-aware: Bursa and crypto have entirely different event taxonomies (PEAD/
OPR/CPO vs funding/OI/unlock), so the system prompt, JSON event-type enum,
historical-edge table, and rule-based fallback keywords are all selected by
MARKET_MODE below rather than hardcoded to one market.
"""
import json
import logging
from datetime import datetime

from agents.base_agent import BaseAgent
from config.settings import MODEL_FAST, MARKET_MODE

logger = logging.getLogger(__name__)

BURSA_HISTORICAL_EDGES = {
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

CRYPTO_HISTORICAL_EDGES = {
    "funding_spike": {
        "description": "Crowded funding mean-reverts",
        "edge": "extreme funding historically fades within 1-3 days",
        "confidence_boost": 0.10,
        "min_magnitude": "medium",
    },
    "oi_surge": {
        "description": "OI buildup into a move raises liquidation-cascade risk",
        "edge": "post-cascade snapbacks common within 1-2 days",
        "confidence_boost": 0.08,
        "min_magnitude": "medium",
    },
    "basis_dislocation": {
        "description": "Perp/index basis converges",
        "edge": "cash-and-carry convergence, typically hours-to-days",
        "confidence_boost": 0.08,
        "min_magnitude": "medium",
    },
    "btc_move": {
        "description": "BTC beta drags the alt complex",
        "edge": "alts follow BTC with a 0-2 day lag, amplified",
        "confidence_boost": 0.10,
    },
    "eth_move": {
        "description": "ETH beta drags smart-contract/L2 tokens",
        "edge": "L2/DeFi tokens follow ETH with a 0-2 day lag",
        "confidence_boost": 0.08,
    },
    "dxy_move": {
        "description": "Dollar strength is a headwind for risk assets incl. crypto",
        "edge": "inverse correlation, 1-day lag",
        "confidence_boost": 0.05,
    },
    "yield_move": {
        "description": "Rising US yields pressure risk-asset valuations",
        "edge": "weak inverse correlation, 1-day lag",
        "confidence_boost": 0.05,
    },
    "listing": {
        "description": "Exchange listing announcement",
        "edge": "+5-15% avg pop on major-exchange listing news",
        "confidence_boost": 0.10,
        "min_magnitude": "medium",
    },
    "unlock": {
        "description": "Token unlock/vesting cliff sell pressure",
        "edge": "-2-8% avg into a large unlock date",
        "confidence_boost": 0.08,
    },
    "depeg": {
        "description": "Stablecoin depeg — tail risk event",
        "edge": "high-volatility, flight-to-quality across the complex",
        "confidence_boost": 0.15,
        "min_magnitude": "medium",
    },
}

HISTORICAL_EDGES = CRYPTO_HISTORICAL_EDGES if MARKET_MODE == "crypto" else BURSA_HISTORICAL_EDGES

BURSA_CLASSIFIER_SYSTEM = """You are an expert event classifier for Bursa Malaysia equity markets.
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

CRYPTO_CLASSIFIER_SYSTEM = """You are an expert event classifier for liquid crypto (perp/spot) markets.
Your job is to analyse market news and on-chain/exchange signals and determine:
1. What type of event this is
2. Which pairs are affected and how
3. Whether there is a systematic trading edge
4. How confident you are in the signal

You know these facts about crypto markets:
- Extreme perp funding (>0.05% per interval) historically fades within 1-3 days
- BTC moves drag the alt complex with amplified beta and a 0-2 day lag
- ETH moves drag smart-contract/DeFi/L2 tokens similarly
- Large open-interest buildups into a price move raise liquidation-cascade risk
- Exchange listing announcements produce a short-lived pop (+5-15%)
- Token unlock/vesting cliffs create sell pressure into the unlock date
- Stablecoin depegs are tail-risk events — flight to quality across the complex
- DXY strength and rising US yields are mild headwinds for crypto as a risk asset
- This is a 24/7 market — weekend moves on thin liquidity often retrace Monday

Be conservative — only flag as actionable when there is clear historical precedent for edge.
Pure macro/regulatory news should almost never be actionable — flag as context only.
Always return valid JSON. Do not include any explanation outside the JSON object."""

CLASSIFIER_SYSTEM = CRYPTO_CLASSIFIER_SYSTEM if MARKET_MODE == "crypto" else BURSA_CLASSIFIER_SYSTEM

_BURSA_EVENT_TYPES = (
    "earnings_beat|earnings_miss|dividend_declared|contract_win|bonus_issue|"
    "opr_hike|opr_cut|opr_hold|china_pmi_strong|china_pmi_weak|fed_hike|fed_cut|"
    "fed_hold|cpo_move|crude_oil_move|analyst_upgrade|analyst_downgrade|"
    "macro_context|irrelevant"
)
_CRYPTO_EVENT_TYPES = (
    "funding_spike|oi_surge|basis_dislocation|btc_move|eth_move|listing|unlock|"
    "depeg|btc_dominance_shift|dxy_move|yield_move|regulatory|"
    "analyst_upgrade|analyst_downgrade|macro_context|irrelevant"
)
_EVENT_TYPES = _CRYPTO_EVENT_TYPES if MARKET_MODE == "crypto" else _BURSA_EVENT_TYPES


class EventClassifier(BaseAgent):
    name = "EventClassifier"
    description = "Classifies market events for trading signals (market-aware: Bursa or crypto)"
    default_model = MODEL_FAST

    def run(self, task: dict) -> dict:
        return self.classify(task)

    def classify(self, event_raw: dict) -> dict:
        """
        Takes raw event dict, returns classified dict with trading signal assessment.
        Falls back to rule-based classification on Claude failure.
        """
        from config.settings import MARKET_NAME, TICKER_EXAMPLE

        ticker_hint = event_raw.get("ticker") or "None"
        body_snippet = (event_raw.get("body") or "")[:500]

        prompt = f"""Classify this market event for {MARKET_NAME} trading:

Source: {event_raw.get('source', 'unknown')}
Headline: {event_raw.get('headline', '')}
Body: {body_snippet}
Ticker mentioned: {ticker_hint}
Published: {event_raw.get('published_at', '')}

Return JSON:
{{
  "event_type": "<{_EVENT_TYPES}>",
  "ticker": "<primary affected instrument e.g. {TICKER_EXAMPLE}, or null>",
  "affected_tickers": ["<list of affected instruments>"],
  "company": "<company/project name or null>",
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

        # Stablecoin depeg is a tail-risk event → always alert regardless of
        # confidence, same reasoning as OPR decisions above.
        if event_type == "depeg":
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

        if MARKET_MODE == "crypto":
            event_type, sentiment, is_actionable, confidence = self._crypto_rule_fallback(headline_lower)
        else:
            event_type, sentiment, is_actionable, confidence = self._bursa_rule_fallback(headline_lower)

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

    @staticmethod
    def _bursa_rule_fallback(headline_lower: str) -> tuple:
        """Returns (event_type, sentiment, is_actionable, confidence)."""
        up = any(w in headline_lower for w in ["rise", "up", "gain", "surge"])

        if any(w in headline_lower for w in ["earnings beat", "profit up", "net profit", "revenue up", "eps beat"]):
            return "earnings_beat", "positive", True, 0.55
        if any(w in headline_lower for w in ["earnings miss", "profit down", "net loss", "revenue miss"]):
            return "earnings_miss", "negative", True, 0.55
        if any(w in headline_lower for w in ["dividend", "distribution"]):
            return "dividend_declared", "positive", True, 0.60
        if any(w in headline_lower for w in ["contract", "letter of award", "awarded"]):
            return "contract_win", "positive", True, 0.50
        if any(w in headline_lower for w in ["opr", "overnight policy rate", "interest rate decision", "bnm"]):
            return "opr_hold", "neutral", False, 0.70
        if any(w in headline_lower for w in ["palm oil", "cpo"]):
            return "cpo_move", ("positive" if up else "negative"), True, 0.55
        if any(w in headline_lower for w in ["crude oil", "brent", "oil price"]):
            return "crude_oil_move", ("positive" if up else "negative"), True, 0.50
        return "macro_context", "neutral", False, 0.30

    @staticmethod
    def _crypto_rule_fallback(headline_lower: str) -> tuple:
        """Returns (event_type, sentiment, is_actionable, confidence)."""
        up = any(w in headline_lower for w in ["rise", "up", "gain", "surge", "rally"])

        if any(w in headline_lower for w in ["depeg", "de-peg", "loses peg"]):
            return "depeg", "negative", True, 0.70
        if any(w in headline_lower for w in ["funding rate", "funding spike", "crowded long", "crowded short"]):
            return "funding_spike", "neutral", True, 0.55
        if any(w in headline_lower for w in ["open interest", "oi surge", "liquidation", "liquidated"]):
            return "oi_surge", "neutral", True, 0.55
        if any(w in headline_lower for w in ["lists on", "listing", "gets listed", "will list"]):
            return "listing", "positive", True, 0.55
        if any(w in headline_lower for w in ["unlock", "vesting", "token release"]):
            return "unlock", "negative", True, 0.55
        if any(w in headline_lower for w in ["bitcoin", "btc"]) and any(w in headline_lower for w in ["surge", "crash", "plunge", "rally", "dump"]):
            return "btc_move", ("positive" if up else "negative"), True, 0.55
        if any(w in headline_lower for w in ["ethereum", "eth"]) and any(w in headline_lower for w in ["surge", "crash", "plunge", "rally", "dump"]):
            return "eth_move", ("positive" if up else "negative"), True, 0.50
        if any(w in headline_lower for w in ["dominance"]):
            return "btc_dominance_shift", "neutral", True, 0.45
        if any(w in headline_lower for w in ["sec", "regulation", "regulatory", "ban", "lawsuit"]):
            return "regulatory", "negative", False, 0.40
        if any(w in headline_lower for w in ["dollar index", "dxy"]):
            return "dxy_move", ("negative" if up else "positive"), True, 0.40
        return "macro_context", "neutral", False, 0.30
