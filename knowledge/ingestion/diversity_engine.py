"""
DiversityEngine — ensures balanced KB coverage across 9 Bursa Malaysia research angles.

Each angle has 3 Bursa-specific seed queries. check_balance() reports KB doc counts
per angle; daily_hunt() auto-fills the most under-researched angle.
"""
import logging

from data.database import db_session
from knowledge.ingestion.research_hunter import ResearchHunter

logger = logging.getLogger(__name__)

# 9 research angles — Bursa Malaysia specific seed queries
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
    "statistical_modelling": {
        "description": "Quantitative models: GARCH, HMM, factor models, ML, cointegration for KLSE",
        "queries": [
            "GARCH volatility model Bursa Malaysia equity",
            "hidden markov regime detection ASEAN stock market",
            "random matrix theory portfolio optimization emerging markets",
            "machine learning return prediction Malaysian stocks",
            "factor model Fama French KLSE",
        ],
    },
}

# Keywords used to infer angle membership from document content/title/tags
ANGLE_KEYWORDS = {
    "price_action": [
        "momentum", "mean reversion", "mean-reversion", "technical", "moving average",
        "rsi", "breakout", "trend", "macd", "bollinger", "price action", "chart pattern",
        "support", "resistance", "crossover", "oscillator",
    ],
    "fundamental": [
        "value", "earnings", "book value", "p/e", "p/b", "roe", "fundamental",
        "dividend yield", "revenue", "balance sheet", "cash flow", "quality", "valuation",
        "earnings quality", "price-to-book", "return on equity",
    ],
    "event_driven": [
        "pead", "earnings drift", "post-earnings", "dividend capture", "corporate event",
        "announcement", "earnings surprise", "ex-dividend", "rights issue", "bonus issue",
        "earnings announcement", "event study",
    ],
    "institutional": [
        "epf", "kwap", "institutional", "glc", "government-linked", "pension fund",
        "sovereign wealth", "foreign ownership", "msci", "passive fund", "index rebalancing",
        "institutional flows", "ownership structure",
    ],
    "macro": [
        "opr", "bank negara", "bnm", "interest rate", "monetary policy", "macroeconomic",
        "gdp", "inflation", "central bank", "rate cycle", "rate sensitivity", "nim",
        "macroeconomics", "economic cycle",
    ],
    "commodity": [
        "cpo", "palm oil", "crude oil", "commodity", "plantation", "energy sector",
        "tin", "rubber", "commodity equity", "commodity correlation", "resource",
        "crude palm oil", "plantation stock",
    ],
    "sector_rotation": [
        "sector rotation", "sector momentum", "industry momentum", "cyclical", "defensive",
        "banking sector", "telco", "utilities", "reit", "sector switching",
        "sector performance", "industry rotation",
    ],
    "behavioural": [
        "sentiment", "behavioural", "behavioral", "anomaly", "anomalies",
        "investor behaviour", "investor behavior", "bias", "microstructure",
        "calendar effect", "january effect", "overreaction", "herding",
        "market anomaly", "investor sentiment",
    ],
    "statistical_modelling": [
        "garch", "egarch", "arima", "volatility model", "time series",
        "hidden markov", "regime detection", "regime switching",
        "random matrix", "eigenvalue", "minimum spanning tree", "correlation clustering",
        "factor model", "fama french", "pca", "principal component", "ica",
        "machine learning", "regression", "bayesian", "kalman filter",
        "monte carlo", "cointegration", "stationarity", "unit root",
        "statistical arbitrage", "clustering algorithm",
    ],
}


class DiversityEngine:
    """
    Tracks and fills KB coverage across 8 Bursa Malaysia research angles.
    Coverage is measured by:
      1. kb_documents whose source_url was set by DiversityEngine (prefix 'diversity_hunt:<angle>')
      2. kb_documents whose title/summary/tags match angle keywords (for legacy untagged docs)
    """

    # ── Coverage check ────────────────────────────────────────────────────────

    def check_balance(self) -> dict:
        """
        Count KB docs per angle using the unified domain field.

        Since Fix 2 (2026-04-07), kb_documents.domain is one of the 8 angle names,
        making this a simple GROUP BY query — no keyword heuristics needed.

        Returns:
            {
              "coverage":      {"price_action": 3, "fundamental": 1, ...},
              "least_covered": "behavioural",
              "total_docs":    18,
              "all_angles":    list of angle names,
            }
        """
        # Seed all angles at 0 so even uncovered ones appear in the result
        coverage = {angle: 0 for angle in ANGLES}
        with db_session() as conn:
            rows = conn.execute(
                "SELECT domain, COUNT(*) AS n FROM kb_documents GROUP BY domain"
            ).fetchall()
        for row in rows:
            angle = row["domain"]
            if angle in coverage:
                coverage[angle] += row["n"]

        least_covered = min(coverage, key=coverage.get)
        return {
            "coverage":      coverage,
            "least_covered": least_covered,
            "total_docs":    sum(coverage.values()),
            "all_angles":    list(ANGLES.keys()),
        }

    # ── Retag existing docs ───────────────────────────────────────────────────

    def retag_existing_docs(self) -> dict:
        """
        Scan all KB documents that lack a diversity_hunt source_url tag and
        assign the best-matching angle based on title/summary/tags keyword matching.

        Updates source_url to 'diversity_hunt:<angle>' for matched documents.
        Documents with no keyword match are left unchanged.

        Returns:
            {"tagged": int, "skipped": int, "by_angle": {angle: count}}
        """
        tagged = 0
        skipped = 0
        by_angle: dict = {angle: 0 for angle in ANGLES}

        with db_session() as conn:
            rows = conn.execute(
                "SELECT id, title, summary, tags FROM kb_documents "
                "WHERE source_url IS NULL OR source_url = '' OR source_url NOT LIKE 'diversity_hunt:%'"
            ).fetchall()

            for row in rows:
                text = " ".join([
                    (row["title"] or ""),
                    (row["summary"] or ""),
                    (row["tags"] or ""),
                ]).lower()

                best_angle = None
                best_score = 0

                for angle, keywords in ANGLE_KEYWORDS.items():
                    score = sum(1 for kw in keywords if kw.lower() in text)
                    if score > best_score:
                        best_score = score
                        best_angle = angle

                if best_angle and best_score > 0:
                    conn.execute(
                        "UPDATE kb_documents SET source_url=? WHERE id=?",
                        (f"diversity_hunt:{best_angle}", row["id"]),
                    )
                    by_angle[best_angle] += 1
                    tagged += 1
                    logger.info(
                        f"[DiversityEngine] Retagged doc #{row['id']} → {best_angle} (score={best_score})"
                    )
                else:
                    skipped += 1
                    logger.debug(f"[DiversityEngine] No angle match for doc #{row['id']}, skipping")

        logger.info(
            f"[DiversityEngine] retag_existing_docs complete: tagged={tagged}, skipped={skipped}, by_angle={by_angle}"
        )
        return {"tagged": tagged, "skipped": skipped, "by_angle": by_angle}

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
                domain=angle_name,   # store unified domain directly in kb_documents
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
