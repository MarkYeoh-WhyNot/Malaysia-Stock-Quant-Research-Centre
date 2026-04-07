"""
RejectionMemory — accumulates idea rejection patterns and injects avoidance
rules into future idea-generation prompts.

Pattern accumulation prevents the pipeline from wasting budget on classes of
strategies that have repeatedly failed (e.g. pairs trading, intraday scalping,
plantation momentum overfitting).
"""
import logging
from datetime import datetime
from data.database import db_session

logger = logging.getLogger(__name__)

# Map common keywords in titles/hypotheses to factor_type labels
_FACTOR_TYPE_KEYWORDS = {
    "momentum": ["momentum", "sma", "ema", "macd", "crossover", "breakout", "trend"],
    "value":    ["pe ratio", "p/e", "price-to-book", "p/b", "dividend yield", "value", "cheap"],
    "mean_reversion": ["rsi", "oversold", "overbought", "mean reversion", "revert", "bollinger"],
    "event":    ["dividend", "ex-date", "earnings", "beat", "announcement", "ipo", "privatisation"],
    "macro":    ["opr", "interest rate", "bnm", "epu", "gdp", "cpo", "crude palm oil", "aluminium"],
    "quality":  ["roe", "return on equity", "profit margin", "cash flow", "quality"],
    "pairs":    ["pairs", "spread", "arbitrage", "vs ", "relative value"],
    "intraday": ["intraday", "scalp", "tick", "minute", "hour", "hft"],
}

_SECTOR_KEYWORDS = {
    "banking":      ["bank", "maybank", "cimb", "hlbank", "rhb", "ammb", "affin", "bimb"],
    "plantation":   ["plantation", "palm oil", "sime darby plantation", "ioi", "klk", "fgv"],
    "utilities":    ["tenaga", "tnb", "gas malaysia", "ytl power", "utilities"],
    "telecoms":     ["axiata", "maxis", "digi", "celcomdigi", "telekom", "tm"],
    "technology":   ["tech", "software", "it ", "semiconductor", "vitrox", "inari"],
    "healthcare":   ["hospital", "health", "ihh", "kpj"],
    "construction": ["construction", "gamuda", "ijm", "wct"],
    "glc":          ["glc", "government-linked", "petronas", "khazanah", "pnb", "epf"],
}

_REASON_CATEGORY_KEYWORDS = {
    "overfitting":  ["overfit", "curve fit", "data snoop", "in-sample", "look-ahead"],
    "no_edge":      ["no edge", "random", "weak factor", "low ic", "low sharpe", "below threshold"],
    "infeasible":   ["infeasible", "short sell", "pairs", "intraday", "not available", "no data",
                     "cannot trade", "restricted", "lot size"],
    "low_sharpe":   ["sharpe", "poor performance", "negative return"],
    "irrelevant":   ["not klse", "foreign", "fx", "forex", "currency pair", "crypto",
                     "mobile banking", "venture", "indian", "steganograph"],
    "liquidity":    ["illiquid", "low volume", "wide spread", "penny stock"],
}


def _classify(text: str, keyword_map: dict, default: str = "other") -> str:
    text_lower = text.lower()
    for label, keywords in keyword_map.items():
        if any(kw in text_lower for kw in keywords):
            return label
    return default


class RejectionMemory:
    """Record why ideas fail; inject avoidance rules into generation prompts."""

    def record_rejection(self, idea_id: int, reason: str, stage: str) -> None:
        """Extract failure pattern from a rejected idea and accumulate its count."""
        try:
            with db_session() as conn:
                row = conn.execute(
                    "SELECT title, hypothesis, pair, factor_formula FROM alpha_ideas WHERE id=?",
                    (idea_id,),
                ).fetchone()
            if not row:
                return

            blob = f"{row['title']} {row['hypothesis'] or ''} {row['factor_formula'] or ''} {reason}"
            factor_type     = _classify(blob, _FACTOR_TYPE_KEYWORDS, "other")
            sector          = _classify(blob, _SECTOR_KEYWORDS, "general")
            reason_category = _classify(reason or blob, _REASON_CATEGORY_KEYWORDS, "other")

            with db_session() as conn:
                # Update rejection_reason on the idea
                conn.execute(
                    "UPDATE alpha_ideas SET rejection_reason=?, updated_at=datetime('now') WHERE id=?",
                    (reason[:500] if reason else None, idea_id),
                )
                # Upsert into rejection_patterns
                conn.execute("""
                    INSERT INTO rejection_patterns
                        (factor_type, sector, reason_category, count, last_seen, example_title)
                    VALUES (?, ?, ?, 1, ?, ?)
                    ON CONFLICT(factor_type, sector, reason_category) DO UPDATE SET
                        count    = count + 1,
                        last_seen = excluded.last_seen,
                        example_title = CASE WHEN excluded.count >= count
                                        THEN excluded.example_title
                                        ELSE example_title END
                """, (
                    factor_type, sector, reason_category,
                    datetime.utcnow().strftime("%Y-%m-%d"),
                    (row["title"] or "")[:80],
                ))

            logger.info(
                f"RejectionMemory: recorded [{idea_id}] "
                f"factor={factor_type} sector={sector} reason={reason_category} stage={stage}"
            )
        except Exception as e:
            logger.warning(f"RejectionMemory.record_rejection failed (non-blocking): {e}")

    def get_avoid_patterns(self) -> list[dict]:
        """Return patterns with count >= 2, sorted by frequency."""
        try:
            with db_session() as conn:
                rows = conn.execute(
                    "SELECT factor_type, sector, reason_category, count, example_title "
                    "FROM rejection_patterns WHERE count >= 2 ORDER BY count DESC LIMIT 20"
                ).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.warning(f"RejectionMemory.get_avoid_patterns failed: {e}")
            return []

    def inject_into_prompt(self) -> str:
        """Return avoidance rules formatted for Claude prompt injection."""
        patterns = self.get_avoid_patterns()
        if not patterns:
            return ""
        lines = []
        for p in patterns:
            sector_str = f" in {p['sector']}" if p["sector"] != "general" else ""
            lines.append(
                f"- Avoid: {p['factor_type']}{sector_str} "
                f"(failed {p['count']}x — {p['reason_category']})"
            )
        return "AVOID THESE KNOWN FAILURES:\n" + "\n".join(lines)
