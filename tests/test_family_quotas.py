"""Phase 5.4: strategy-family classification, genome, and quota reporting."""
import pytest

from data.database import db_session, init_db
from knowledge.ingestion.family_quotas import (
    classify_family, build_genome, get_family_distribution)

SENTINELS = (900_101, 900_102, 900_103)


@pytest.fixture()
def clean():
    init_db()
    _cleanup()
    yield
    _cleanup()


def _cleanup():
    with db_session() as conn:
        for i in SENTINELS:
            conn.execute("DELETE FROM alpha_ideas WHERE id=?", (i,))


def test_classify_family_momentum():
    assert classify_family("20/50 SMA crossover breakout momentum") == "momentum"


def test_classify_family_value():
    assert classify_family("Low P/E dividend yield value screen") == "value"


def test_classify_family_unclassified_default():
    assert classify_family("completely unrelated text about weather") == "other"


def test_build_genome_marks_malaysia_specific():
    g = build_genome("EPF accumulation", "EPF has raised ownership 3 quarters",
                     "epf_ownership_delta > 0", "3mo", "event")
    assert g["why_malaysia_specific"] is True
    assert g["expected_turnover"] == "low"
    assert "equal-weight KLCI" in g["simplest_baseline_to_beat"]


def test_build_genome_generic_not_malaysia_specific():
    g = build_genome("Generic momentum", "20-day price momentum",
                     "close > sma(20)", "1d", "momentum")
    assert g["why_malaysia_specific"] is False
    assert g["expected_turnover"] == "medium"


def test_family_distribution_counts_and_targets(clean):
    with db_session() as conn:
        for i, fam in zip(SENTINELS, ("momentum", "momentum", "value")):
            conn.execute(
                "INSERT INTO alpha_ideas (id, slug, title, ticker, stage, status, family) "
                "VALUES (?, ?, 'q', '1155.KL', 'gate0', 'pending', ?)",
                (i, f"quota-test-{i}", fam))
    dist = get_family_distribution()
    assert dist["momentum"]["count"] >= 2
    assert dist["value"]["count"] >= 1
    assert dist["_total"] >= 3
