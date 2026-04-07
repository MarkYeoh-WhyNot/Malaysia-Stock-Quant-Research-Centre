"""
DiversityEngine — ensures balanced KB coverage across 8 Bursa Malaysia research angles.

Each angle has 3 Bursa-specific seed queries. check_balance() reports KB doc counts
per angle; daily_hunt() auto-fills the most under-researched angle.
"""
import logging

from data.database import db_session
from knowledge.ingestion.research_hunter import ResearchHunter

logger = logging.getLogger(__name__)

# 8 research angles — Bursa Malaysia specific seed queries
ANGLES = {
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
}


class DiversityEngine:
    """
    Tracks and fills KB coverage across 8 Bursa Malaysia research angles.
    Coverage is measured by counting kb_documents whose source_url was set
    by a DiversityEngine hunt (prefix 'diversity_hunt:<angle>').
    """

    # ── Coverage check ────────────────────────────────────────────────────────

    def check_balance(self) -> dict:
        """
        Count KB docs per angle and return a coverage report.

        Returns:
            {
              "coverage":     {"price_action": 3, "fundamental": 1, ...},
              "least_covered": "behavioural",
              "total_docs":    18,
              "all_angles":    list of angle names,
            }
        """
        coverage = {}
        with db_session() as conn:
            for angle in ANGLES:
                row = conn.execute(
                    "SELECT COUNT(*) AS n FROM kb_documents WHERE source_url LIKE ?",
                    (f"diversity_hunt:{angle}%",),
                ).fetchone()
                coverage[angle] = row["n"] if row else 0

        least_covered = min(coverage, key=coverage.get)
        return {
            "coverage":     coverage,
            "least_covered": least_covered,
            "total_docs":   sum(coverage.values()),
            "all_angles":   list(ANGLES.keys()),
        }

    # ── Fill a specific angle ─────────────────────────────────────────────────

    def fill_angle(self, angle_name: str) -> dict:
        """
        Manually trigger a research hunt for a specific angle.

        Args:
            angle_name: One of the 8 angle keys (e.g. 'price_action').

        Returns:
            Aggregated hunt result dict from ResearchHunter.
        """
        data = ANGLES.get(angle_name)
        if not data:
            return {"error": f"Unknown angle '{angle_name}'. Valid: {list(ANGLES.keys())}"}

        logger.info(f"[DiversityEngine] Filling angle '{angle_name}': {data['description']}")
        hunter = ResearchHunter()

        combined: dict = {"papers_found": 0, "papers_ingested": 0, "titles": [], "queries": []}
        for query in data["queries"][:2]:  # First 2 seed queries per run to control cost
            result = hunter.hunt(
                topic=query,
                context=data["description"],
                angle_tag=angle_name,
            )
            combined["papers_found"]    += result["papers_found"]
            combined["papers_ingested"] += result["papers_ingested"]
            combined["titles"].extend(result["titles"])
            combined["queries"].extend(result["queries"])

        combined["angle"] = angle_name
        logger.info(
            f"[DiversityEngine] angle='{angle_name}' found={combined['papers_found']} "
            f"ingested={combined['papers_ingested']}"
        )
        return combined

    # ── Daily hunt ────────────────────────────────────────────────────────────

    def daily_hunt(self) -> dict:
        """
        Check KB balance and fill the most under-researched angle.

        Called once per day by the daemon at ~22:00 UTC (6am KL time).

        Returns:
            Hunt result dict plus balance metadata.
        """
        balance = self.check_balance()
        target  = balance["least_covered"]
        logger.info(
            f"[DiversityEngine] Daily hunt targeting angle '{target}' "
            f"(coverage={balance['coverage'][target]}). "
            f"Full coverage: {balance['coverage']}"
        )
        result         = self.fill_angle(target)
        result["balance_before"] = balance["coverage"]
        result["target_angle"]   = target
        return result
