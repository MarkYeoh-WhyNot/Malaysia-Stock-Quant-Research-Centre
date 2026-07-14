"""Shared idea-text helpers.

Kept dependency-light (stdlib only) so every idea write path can import it
without risking an import cycle — `pipeline/sandbox.py` imports StrategyResearcher
lazily, and the researcher / finding_candidates modules import this leaf module
freely.
"""
from __future__ import annotations


def ensure_description(title: str, hypothesis: str | None,
                       factor_formula: str | None) -> str:
    """Return a non-empty, human-readable description for an alpha idea.

    Every backtested strategy must always show a description in the dashboard
    (Backtest Lab / Ideas Queue) regardless of which source produced it. The
    Factor Sandbox form, the Concierge tool, and the organic researcher can all
    reach an idea write with an empty/whitespace `hypothesis`; this synthesizes a
    stand-in from the title and factor_formula rather than storing an empty
    string. A real hypothesis is always preferred and passed through untouched.
    """
    hyp = (hypothesis or "").strip()
    if hyp:
        return hyp
    name = (title or "").strip() or "Untitled idea"
    formula = (factor_formula or "").strip() or "(no formula specified)"
    return f"{name} — signal: {formula}"
