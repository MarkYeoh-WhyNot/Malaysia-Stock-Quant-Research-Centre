"""Strategy-family classification, quota reporting, and genome (Phase 5.4,
audit §9). Reuses RejectionMemory's keyword taxonomy so "what fails" and "what
gets generated" share one vocabulary instead of inventing a second one.

Deliberately does NOT rewrite the idea-generation LLM prompt — classification and
quota reporting happen post-hoc on saved ideas so the highest-risk, highest-value
part of the pipeline (StrategyResearcher.generate_ideas) is untouched. Quotas are
report-only: they inform, they don't block a save.
"""
from __future__ import annotations

from knowledge.ingestion.rejection_memory import _classify, _FACTOR_TYPE_KEYWORDS
from config.settings import GATE_CONFIG
from data.database import db_session


def classify_family(text: str) -> str:
    """Same factor_type taxonomy as RejectionMemory (momentum/value/quality/...)."""
    label, _ = _classify(text, _FACTOR_TYPE_KEYWORDS, "other")
    return label


def build_genome(title: str, hypothesis: str, factor_formula: str,
                 timeframe: str, family: str) -> dict:
    """Deterministic strategy-genome fields (audit §9.1), computed from what's
    already on the idea — no extra Claude call. `simplest_baseline_to_beat` is
    always equal-weight KLCI (the benchmark gate's actual hurdle, Phase 3.2), so
    this genome documents a requirement the pipeline already enforces.
    """
    return {
        "signal_family": family,
        "timeframe": timeframe,
        "expected_turnover": (
            "high" if timeframe in ("1min", "5min", "15min", "1h") else
            "low" if timeframe in ("1mo", "3mo") else "medium"
        ),
        "simplest_baseline_to_beat": "equal-weight KLCI (Phase 3.2 benchmark gate)",
        "why_malaysia_specific": bool(
            any(k in (hypothesis or "").lower() for k in
                ("epf", "klci", "bursa", "opr", "bnm", "cpo", "glc", "ringgit", "myr"))
        ),
    }


def get_family_distribution() -> dict:
    """Current share of non-rejected ideas per family, vs configured targets."""
    with db_session() as conn:
        rows = conn.execute(
            "SELECT COALESCE(family, 'unclassified') AS family, COUNT(*) AS n "
            "FROM alpha_ideas WHERE status != 'rejected' GROUP BY family"
        ).fetchall()
    counts = {r["family"]: r["n"] for r in rows}
    total = sum(counts.values())
    targets = GATE_CONFIG.family_quota_targets
    out = {}
    for fam, target in targets.items():
        n = counts.get(fam, 0)
        actual = (n / total) if total else 0.0
        out[fam] = {"count": n, "actual_pct": round(actual, 3),
                    "target_pct": target, "under_quota": actual < target}
    out["_unclassified"] = counts.get("unclassified", 0)
    out["_total"] = total
    return out


def next_underquota_family() -> str | None:
    """Family furthest below its target share — a candidate topic to steer
    generation toward, mirroring how KB angle diversity already picks the
    least-covered angle for topicless generation."""
    dist = get_family_distribution()
    gaps = {f: v["target_pct"] - v["actual_pct"]
            for f, v in dist.items() if not f.startswith("_")}
    if not gaps:
        return None
    fam, gap = max(gaps.items(), key=lambda kv: kv[1])
    return fam if gap > 0 else None
