"""Shared factor-sandbox submission (used by the dashboard HTTP endpoint and the
Concierge agent).

An idea is inserted at stage2 so it enters the pipeline at backtest, skipping
Gate 0 — the caller is a human (directly, or via the Concierge on a human's
prompt) who has vouched for the idea. The daemon then carries it stage2 → stage3
(red/blue) → stage4a (paper) automatically; nothing here reaches live trading.

A deterministic feasibility pre-check (reused from StrategyResearcher) runs first
so an infeasible brief (a hard-blocked mode for the active market profile, bad
ticker, unavailable data) is refused before it consumes a backtest — the previous
inline sandbox skipped this.
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime

from data.database import db_session
# Ticker format + hard-blocked trading modes come from the active market profile
# (Bursa: .KL codes, long-only; crypto: /USDT pairs, long/short perps). Daily
# bars in every market.
from config.settings import (
    TICKER_REGEX, TICKER_EXAMPLE, BLOCKED_MODES, MARKET_NAME, ALLOW_SHORT,
)

# Matches the pipeline's Gate 0 feasibility bar (North Star: feasibility >= 0.60).
MIN_FEASIBILITY = 0.60

# Refusal wording tracks the active profile's capabilities (ALLOW_SHORT is the
# one semantic switch); the Bursa strings are pinned byte-identical in tests.
_CAPABILITY_NOTE = (
    "this system trades long/short on daily bars (no multi-leg spread/pairs, "
    "options, or intraday)" if ALLOW_SHORT else
    "this system is long-only and trades daily bars (no short-selling, pairs, "
    "or intraday)")
_INFEASIBLE_HINT = (
    "intraday, multi-leg structure, or unavailable-data reliance (e.g. "
    "historical funding/OI series)" if ALLOW_SHORT else
    "short-selling, intraday, or unavailable-data reliance")


def _blocked_mode(text: str) -> str | None:
    low = (text or "").lower()
    return next((p for p in BLOCKED_MODES if p in low), None)


def _signal_signature(factor_formula: str, ticker: str) -> str:
    """Same normalized text signature save_idea() uses, for cross-path dedup."""
    tokens = sorted(set(re.findall(r"[a-z0-9.]+", (factor_formula or "").lower())))
    return "txt:" + hashlib.sha256(
        (" ".join(tokens) + "|" + ticker).encode()).hexdigest()


def submit_sandbox_idea(brief: dict, run_backtest: bool = False,
                        source: str = "sandbox") -> dict:
    """Insert a human/Concierge-vouched idea at stage2 and (optionally) backtest.

    brief: {title, hypothesis, ticker, timeframe, factor_formula}.
    run_backtest=False → insert at stage2/pending; the daemon's _process_stage2
      picks it up (non-blocking — used by the Concierge chat path).
    run_backtest=True  → insert at stage2/active and run the backtest inline,
      returning its result (preserves the synchronous dashboard sandbox UX).

    Returns {"ok": True, "idea_id", "slug", "feasibility", "status", "result"?}
    or {"ok": False, "error", "feasibility"?} on refusal/dedup.
    """
    from agents.researcher.strategy_researcher import StrategyResearcher

    title = (brief.get("title") or "Untitled sandbox idea").strip()
    hypothesis = brief.get("hypothesis") or ""
    factor_formula = brief.get("factor_formula") or ""
    timeframe = brief.get("timeframe") or "1d"
    raw_ticker = brief.get("ticker") or ""

    found_tickers = TICKER_REGEX.findall(raw_ticker)
    if not found_tickers:
        return {"ok": False,
                "error": f"No valid ticker in '{raw_ticker[:60]}' — {MARKET_NAME} "
                         f"instruments look like {TICKER_EXAMPLE}."}
    seen: set = set()
    ticker = ",".join(t for t in found_tickers if not (t in seen or seen.add(t)))
    primary = found_tickers[0]

    # Hard block on structurally infeasible modes for the active market profile.
    blocked = _blocked_mode(f"{title} {hypothesis} {factor_formula}")
    if blocked:
        return {"ok": False,
                "error": f"'{blocked}' is not supported — {_CAPABILITY_NOTE}."}

    # Deterministic feasibility pre-check — fail cheap before a backtest.
    feasibility = StrategyResearcher._compute_feasibility(
        {"hypothesis": hypothesis}, primary, factor_formula)
    if feasibility < MIN_FEASIBILITY:
        return {"ok": False, "feasibility": feasibility,
                "error": f"Idea is not feasible on {MARKET_NAME} (feasibility "
                         f"{feasibility:.2f} < {MIN_FEASIBILITY:.2f}) — likely "
                         f"{_INFEASIBLE_HINT}."}

    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40]
    slug = f"{source}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{slug}"
    signature = _signal_signature(factor_formula, ticker)
    status = "active" if run_backtest else "pending"

    try:
        from knowledge.ingestion.family_quotas import classify_family
        family = classify_family(f"{title} {hypothesis} {factor_formula}")
    except Exception:
        family = "other"

    with db_session() as conn:
        dup = conn.execute(
            "SELECT id, title FROM alpha_ideas "
            "WHERE signal_signature=? AND status != 'rejected' LIMIT 1",
            (signature,),
        ).fetchone()
        if dup:
            return {"ok": False, "duplicate_of": dup["id"],
                    "error": f"This duplicates live idea #{dup['id']} "
                             f"('{dup['title'][:50]}') — not resubmitted."}
        conn.execute("""
            INSERT INTO alpha_ideas
              (slug, title, hypothesis, ticker, timeframe, factor_formula,
               data_sources, stage, status, novelty_score, logic_score,
               feasibility_score, signal_signature, screen_source, family)
            VALUES (?,?,?,?,?,?,'[]','stage2',?,0.7,0.7,?,?,?,?)
        """, (slug, title, hypothesis, ticker, timeframe, factor_formula,
              status, feasibility, signature, source, family))
        idea_id = conn.execute(
            "SELECT id FROM alpha_ideas WHERE slug=?", (slug,)).fetchone()["id"]

    out = {"ok": True, "idea_id": idea_id, "slug": slug,
           "feasibility": feasibility, "status": status, "ticker": ticker}

    if run_backtest:
        from agents.backtest_engineer.backtest_engineer import BacktestEngineer
        result = BacktestEngineer().run({"action": "backtest", "idea_id": idea_id})
        out["result"] = result
    return out
