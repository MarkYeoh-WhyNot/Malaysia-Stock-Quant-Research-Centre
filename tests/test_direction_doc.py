"""WS1a: System Direction is profile-driven (crypto content in crypto mode,
Bursa content in Bursa mode). Profiles imported directly — no env juggling.

2026-07-12 (constitution/charter split): ONE shared SYSTEM_CONSTITUTION
(config/markets/_shared.py) + per-market charters (each profile's
DIRECTION_DOC). Two drift guards are pinned here:
  - anti-duplication: shared principles ("quality over quantity") may not
    reappear inside a market charter;
  - no-technical-results (Mark's rule): philosophy/charter text carries NO
    digits — trial counts, IC/t-stats, thresholds, and window sizes live in
    the research record, never in philosophy. Numbers stay legal in the
    factual fields (constraints / transaction_costs / success_metrics).
"""
import re

from config.markets import bursa, crypto
from config.markets._shared import SYSTEM_CONSTITUTION


def _valid_doc(doc):
    for key in ("core_purpose", "design_philosophy", "success_metrics",
                "constraints", "transaction_costs", "last_updated"):
        assert key in doc, f"missing {key}"
    assert isinstance(doc["success_metrics"], list) and doc["success_metrics"]
    assert isinstance(doc["constraints"], list) and doc["constraints"]


def test_bursa_direction_is_bursa_flavoured():
    d = bursa.DIRECTION_DOC
    _valid_doc(d)
    assert "Bursa" in d["core_purpose"]
    blob = " ".join(d["constraints"]).lower()
    assert "epf" in blob and "opr" in blob
    assert d["transaction_costs"]["settlement"] == "T+2"


def test_crypto_direction_is_crypto_flavoured():
    d = crypto.DIRECTION_DOC
    _valid_doc(d)
    assert "crypto" in d["core_purpose"].lower()
    blob = " ".join(d["constraints"]).lower()
    assert "24/7" in blob or "btc" in blob
    assert d["transaction_costs"]["settlement"] == "T+0"
    # No Bursa shadows leaked in.
    assert "epf" not in blob and "opr" not in blob and "cpo" not in blob


def test_crypto_research_angles_have_no_bursa_terms():
    blob = " ".join(m["description"] for m in crypto.RESEARCH_ANGLES.values()).lower()
    assert "epf" not in blob and "bursa" not in blob and "klse" not in blob
    assert "bitcoin" in blob or "crypto" in blob


# ── Constitution / charter split (2026-07-12) ────────────────────────────────

def test_constitution_exists_and_carries_sacred_anchors():
    assert SYSTEM_CONSTITUTION and len(SYSTEM_CONSTITUTION) > 200
    low = SYSTEM_CONSTITUTION.lower()
    for anchor in ("approximated", "human", "paper", "quality over quantity"):
        assert anchor in low, f"constitution missing anchor {anchor!r}"


def test_constitution_is_shared_not_profile_owned():
    """The constitution is market-agnostic: settings re-exports the _shared
    constant directly, and neither profile carries its own copy."""
    from config import settings
    assert settings.SYSTEM_CONSTITUTION is SYSTEM_CONSTITUTION
    assert not hasattr(bursa, "SYSTEM_CONSTITUTION")
    assert not hasattr(crypto, "SYSTEM_CONSTITUTION")


def test_charters_do_not_duplicate_shared_principles():
    """Anti-drift guard: shared constitution lines must not reappear inside a
    market charter (the pre-split failure mode was 'quality over quantity'
    duplicated in both markets with slightly different wording)."""
    for profile in (bursa, crypto):
        blob = (profile.DIRECTION_DOC["core_purpose"] + " "
                + profile.DIRECTION_DOC["design_philosophy"]).lower()
        assert "quality over quantity" not in blob, profile.__name__


def test_no_technical_results_in_philosophy_text():
    """Mark's rule (2026-07-12): philosophy states timeless principle — no
    trial counts, IC/t-stats, harness outcomes, or gate mechanics. Enforced
    as: NO digits anywhere in philosophy/charter prose."""
    texts = {"constitution": SYSTEM_CONSTITUTION}
    for profile in (bursa, crypto):
        texts[f"{profile.__name__}.core_purpose"] = profile.DIRECTION_DOC["core_purpose"]
        texts[f"{profile.__name__}.design_philosophy"] = profile.DIRECTION_DOC["design_philosophy"]
    for name, text in texts.items():
        digits = re.findall(r"\d", text)
        assert not digits, f"{name} contains digits {digits[:5]} — technical results are banned in philosophy text"
