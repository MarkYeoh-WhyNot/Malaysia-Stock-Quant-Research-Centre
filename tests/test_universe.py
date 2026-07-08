"""Phase 2: point-in-time universe membership + production eligibility."""
from data.database import init_db
from data.universe import get_universe_asof, is_production_eligible
from config.settings import UNIVERSE_ASOF, DEFAULT_SYMBOLS


def setup_module(_):
    init_db()  # seeds universe_membership with current KLCI


def test_current_members_present():
    members = get_universe_asof(None)
    assert len(members) >= 25
    assert "1155.KL" in members  # Maybank
    assert set(members) == set(DEFAULT_SYMBOLS)


def test_asof_today_matches_current():
    assert set(get_universe_asof(UNIVERSE_ASOF)) == set(get_universe_asof(None))


def test_pre_asof_window_not_production_eligible():
    # membership coverage only starts at UNIVERSE_ASOF
    assert is_production_eligible("2015-01-01") is False
    assert is_production_eligible(UNIVERSE_ASOF) is True
    assert is_production_eligible(None) is False


def test_falls_back_to_defaults_on_unknown_universe():
    # unknown universe → empty query → fallback to DEFAULT_SYMBOLS
    assert set(get_universe_asof(None, "NONEXISTENT")) == set(DEFAULT_SYMBOLS)
