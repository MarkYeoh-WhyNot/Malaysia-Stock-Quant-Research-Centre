"""WS1a: System Direction is profile-driven (crypto content in crypto mode,
Bursa content in Bursa mode). Profiles imported directly — no env juggling."""
from config.markets import bursa, crypto


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
