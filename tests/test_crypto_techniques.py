"""WS1b: crypto technique library is crypto-native and schema-compatible with
the Bursa one. Imported directly (no MARKET_MODE juggling)."""
from knowledge.ingestion.crypto_techniques import CRYPTO_TECHNIQUE_LIBRARY as CRYPTO
from knowledge.ingestion.technique_library import BURSA_TECHNIQUE_LIBRARY as BURSA

_REQUIRED = ("name", "angle", "when_to_use", "when_to_avoid",
             "market_applicability", "implemented", "complexity", "overfitting_risk")


def test_crypto_techniques_schema_matches():
    assert len(CRYPTO) >= 10
    for key, t in CRYPTO.items():
        for field in _REQUIRED:
            assert field in t, f"{key} missing {field}"


def test_no_bursa_shadows_in_crypto_techniques():
    blob = " ".join(
        t["name"] + " " + t["market_applicability"] + " " + " ".join(t["when_to_use"])
        for t in CRYPTO.values()
    ).lower()
    for term in ("bursa", "epf", "klci", "cpo", "opr", "klse", "ringgit", "plantation"):
        assert term not in blob, f"Bursa shadow '{term}' leaked into crypto techniques"


def test_crypto_has_native_techniques():
    keys = set(CRYPTO)
    assert "funding_rate_carry" in keys
    assert "perp_basis_arb" in keys
    assert "btc_beta_neutralization" in keys


def test_methodology_gates_marked_implemented():
    assert CRYPTO["cross_sectional_ic"]["implemented"] is True
    assert CRYPTO["deflated_sharpe"]["implemented"] is True
    assert CRYPTO["funding_rate_carry"]["implemented"] is False


def test_bursa_library_still_intact():
    assert len(BURSA) >= 20
    assert any("epf" in k for k in BURSA)
