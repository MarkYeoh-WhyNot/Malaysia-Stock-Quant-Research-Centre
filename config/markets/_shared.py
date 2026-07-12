"""Shared system constitution — identical for every market profile.

Three-layer direction structure (2026-07-12, from the OpenAI dual-philosophy
consultation): ONE shared constitution (this module — how the system behaves,
validates, rejects, governs, and protects truth) + per-market charters (each
profile's DIRECTION_DOC — what that market is trying to understand and where
its edge is expected to live). Shared discipline, market-specific doctrine.

RULE (Mark, 2026-07-12): no technical results in philosophy/charter text — no
trial counts, IC/t-stats, harness outcomes, or gate mechanics. Philosophy
states timeless principle; empirical evidence lives in the research record
(KB, memory, gate docs). Enforced by a no-digits pin in
tests/test_direction_doc.py.

This module is imported directly by config/settings.py (NOT via the active
profile) so both markets always carry the byte-identical text.
"""

SYSTEM_CONSTITUTION = (
    "This is a research-validation operating system, not a signal factory. Its "
    "purpose is to find genuine, statistically defensible alpha through honest "
    "representation, rigorous validation, adversarial review, and human-gated "
    "deployment.\n\n"
    "Quality over quantity: a handful of robust, well-validated strategies beats "
    "hundreds of noise ideas — and honesty over throughput, always. No signal is "
    "trusted until it survives every check: data-quality and liquidity screens, "
    "cost and execution reality, deterministic fidelity inspection, and "
    "adversarial review — so that \"passed the gates\" means trustworthy, not "
    "merely lucky.\n\n"
    "Non-representable strategies are rejected, never silently approximated. "
    "Deterministic checks come before model judgment, and no model claim becomes "
    "trusted without human review. Negative evidence is retained as research "
    "memory — an honestly tested rejection is progress. Backtests, charts, gates, "
    "and paper results must tell the same truth; paper results are never "
    "presented as deployable capital. Every strategy paper-trades before any "
    "capital, and no live capital decision is ever automated."
)
