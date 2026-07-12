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
    # Checked FIRST (dict order = match priority in _classify): "not
    # available" alone would otherwise fall into "infeasible" below, and
    # these are the parser's own honest-rejection wording (backtest_
    # engineer.py's dsl_unrepresentable gate), not a keyword guess.
    "unrepresentable": ["not representable", "cannot be expressed", "not available in the "
                        "condition set", "not a standard technical indicator", "custom "
                        "derived metric", "requires computing a custom", "not encoded"],
    "overfitting":  ["overfit", "curve fit", "data snoop", "in-sample", "look-ahead"],
    "no_edge":      ["no edge", "random", "weak factor", "low ic", "low sharpe", "below threshold"],
    "infeasible":   ["infeasible", "short sell", "pairs", "intraday", "not available", "no data",
                     "cannot trade", "restricted", "lot size"],
    "low_sharpe":   ["sharpe", "poor performance", "negative return"],
    # "crypto" deliberately NOT here (removed 2026-07-13): this list is
    # market-agnostic code shared by both daemons, and this keyword was a
    # Bursa-only-era assumption ("mentions crypto" = off-topic). In the
    # crypto daemon almost every idea legitimately mentions crypto/BTC, so
    # it was mislabeling on-topic rejections as "irrelevant" — idea #218's
    # unrepresentable cross-asset ratio landed here purely on that keyword,
    # then got chain-revived under a bucket unrelated to its real problem.
    "irrelevant":   ["not klse", "foreign", "fx", "forex", "currency pair",
                     "mobile banking", "venture", "indian", "steganograph"],
    "liquidity":    ["illiquid", "low volume", "wide spread", "penny stock"],
}


def _classify(text: str, keyword_map: dict, default: str = "other") -> str:
    text_lower = text.lower()
    for label, keywords in keyword_map.items():
        if any(kw in text_lower for kw in keywords):
            return label
    return default


# Phase 5.5 — revival conditions template per rejection reason (audit §5.4).
# Tells a future generation pass what NEW evidence would justify revisiting a
# rejected strategy family/sector combination, rather than a bare "avoid".
_REVIVAL_CONDITIONS = {
    "unrepresentable": "Only revive once a new DSL leaf exists that can express this formula "
                       "(see leaf_synthesis_attempts) — new market data, regime, or KG findings "
                       "do not change whether the parser CAN express it, so they are never "
                       "grounds for revival on their own.",
    "overfitting":  "Only revive with a lower-parameter formulation tested across a broader universe (15+ stocks).",
    "no_edge":      "Only revive with a materially different data source or a longer/shorter holding period showing IC > 0.05.",
    "infeasible":   "Only revive if the infeasible mechanic (short/pairs/intraday) is removed entirely.",
    "low_sharpe":   "Only revive with a walk-forward Sharpe improvement (net) of at least 0.3 over this attempt.",
    "irrelevant":   "Only revive if re-scoped to a genuine Bursa Malaysia .KL instrument.",
    "liquidity":    "Only revive on a higher-liquidity name (ADV value above the RM500k floor with margin).",
    "data_quality": "Only revive once the ticker's Data Confidence Score clears the Gate DQ threshold.",
    "other":        "Only revive with new evidence not present in the original hypothesis.",
}


def _tokens(text: str) -> set:
    import re
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


class RejectionMemory:
    """Record why ideas fail; inject avoidance rules into generation prompts."""

    def record_rejection(self, idea_id: int, reason: str, stage: str,
                         reason_category: str | None = None) -> None:
        """Extract failure pattern from a rejected idea and accumulate its count.

        `stage` is the pipeline stage/gate that rejected it (gate0, stage2,
        stage2_cs, human_review, or — at call sites inside _reject_idea — a
        more specific value like "unrepresentable"/"duplicate"/"data_quality"
        that ALSO doubles as strategy_cemetery.rejected_at_stage; unchanged
        by this fix for backward compatibility with existing checks on that
        column, e.g. pipeline/revisit.py's chain-revival guard).

        `reason_category` is the SEPARATE, correct bucket for
        rejection_patterns / the KG rejection_pattern node. Pass it
        explicitly whenever the caller already knows it precisely (e.g.
        _reject_idea's own reason_category param) — free-text keyword
        classification is a fallback for callers that only have a message,
        not a substitute for a category the caller already computed
        (2026-07-13 fix: this used to re-guess from text even when a
        precise category was available, and the guess had no
        "unrepresentable" bucket at all, so idea #218's cross-asset-ratio
        rejection fell into "irrelevant" purely on the word "crypto")."""
        try:
            with db_session() as conn:
                row = conn.execute(
                    "SELECT title, hypothesis, ticker, factor_formula FROM alpha_ideas WHERE id=?",
                    (idea_id,),
                ).fetchone()
            if not row:
                return

            blob = f"{row['title']} {row['hypothesis'] or ''} {row['factor_formula'] or ''} {reason}"
            factor_type     = _classify(blob, _FACTOR_TYPE_KEYWORDS, "other")
            sector          = _classify(blob, _SECTOR_KEYWORDS, "general")
            reason_category = reason_category or _classify(
                reason or blob, _REASON_CATEGORY_KEYWORDS, "other")

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
                # Phase 5.5: one row per rejected idea with revival conditions
                # (strategy_cemetery) — complements the aggregated pattern above.
                conn.execute("""
                    INSERT INTO strategy_cemetery
                      (idea_id, strategy_name, factor_type, sector,
                       rejected_at_stage, rejection_reason, revival_conditions)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    idea_id, row["title"] or f"idea {idea_id}",
                    factor_type, sector, stage, (reason or "")[:500],
                    _REVIVAL_CONDITIONS.get(reason_category, _REVIVAL_CONDITIONS["other"]),
                ))

            logger.info(
                f"RejectionMemory: recorded [{idea_id}] "
                f"factor={factor_type} sector={sector} reason={reason_category} stage={stage}"
            )

            # Knowledge graph: idea --rejected_because--> rejection_pattern.
            # Makes failure knowledge traversable (red team ammunition, KB
            # Explorer shows which patterns kill ideas).
            try:
                from knowledge.graph import store
                with db_session() as conn:
                    pat = conn.execute(
                        "SELECT id, count, last_seen FROM rejection_patterns "
                        "WHERE factor_type=? AND sector=? AND reason_category=?",
                        (factor_type, sector, reason_category),
                    ).fetchone()
                if pat:
                    pattern_node = store.upsert_node(
                        "rejection_pattern",
                        slug=f"reject-{factor_type}-{sector}-{reason_category}".lower(),
                        title=f"{factor_type} / {reason_category}",
                        summary=(f"Rejected {pat['count']}x (last {pat['last_seen']}). "
                                 f"Sector: {sector}. Example: {(row['title'] or '')[:80]}"),
                        ref=("rejection_patterns", pat["id"]),
                    )
                    idea_node = store.upsert_node(
                        "idea", slug=f"idea-{idea_id}-rejected"[:120],
                        title=row["title"] or f"idea {idea_id}",
                        summary=(reason or "")[:500],
                        ref=("alpha_ideas", idea_id),
                    )
                    store.add_edge(idea_node, pattern_node, "rejected_because",
                                   weight=0.8, origin="heuristic")
            except Exception as ge:
                logger.warning(f"RejectionMemory graph edge failed (non-blocking): {ge}")
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

    def find_similar_rejected(self, title: str, hypothesis: str = "",
                              min_overlap: float = 0.5, limit: int = 3) -> list[dict]:
        """Phase 5.5: cemetery similarity check. Jaccard word-overlap against
        past cemetery entries — cheap, no Claude call, run before saving a new
        idea. Non-blocking: callers log/inform, they don't reject on this alone
        (word overlap alone is too noisy to gate on)."""
        query_tokens = _tokens(f"{title} {hypothesis}")
        if not query_tokens:
            return []
        try:
            with db_session() as conn:
                rows = conn.execute(
                    "SELECT strategy_name, factor_type, sector, rejection_reason, "
                    "revival_conditions FROM strategy_cemetery "
                    "ORDER BY created_at DESC LIMIT 200"
                ).fetchall()
        except Exception as e:
            logger.warning(f"RejectionMemory.find_similar_rejected failed: {e}")
            return []
        hits = []
        for r in rows:
            cand_tokens = _tokens(r["strategy_name"])
            if not cand_tokens:
                continue
            overlap = len(query_tokens & cand_tokens) / len(query_tokens | cand_tokens)
            if overlap >= min_overlap:
                hits.append({**dict(r), "similarity": round(overlap, 2)})
        hits.sort(key=lambda h: h["similarity"], reverse=True)
        return hits[:limit]

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
