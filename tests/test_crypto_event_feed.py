"""WS2: crypto event feed — funding/OI monitor, market-aware classifier, and
the create_gate0_idea ticker-gate bug (crypto events could never become ideas)."""
from unittest.mock import patch

import pytest

from data.events.crypto_monitor import CryptoMonitor, FUNDING_THRESHOLD_PCT
from agents.event_classifier import EventClassifier


# ── CryptoMonitor ────────────────────────────────────────────────────────────

def test_funding_spike_detected():
    mon = CryptoMonitor()
    with patch("data.events.crypto_monitor.get_funding_rate") as gfr, \
         patch("data.events.crypto_monitor.get_open_interest") as goi:
        gfr.return_value = {"symbol": "BTC/USDT", "funding_rate": 0.001,
                            "funding_rate_pct": 0.10, "mark_price": 65000, "index_price": 64900}
        goi.return_value = None
        events = mon.check_moves()
    types = [e["event_type"] for e in events]
    assert "funding_spike" in types


def test_no_event_when_funding_normal():
    mon = CryptoMonitor()
    with patch("data.events.crypto_monitor.get_funding_rate") as gfr, \
         patch("data.events.crypto_monitor.get_open_interest") as goi:
        gfr.return_value = {"symbol": "BTC/USDT", "funding_rate": 0.0001,
                            "funding_rate_pct": 0.01, "mark_price": 65000, "index_price": 64990}
        goi.return_value = None
        events = mon.check_moves()
    assert events == []


def test_basis_dislocation_detected_alongside_funding():
    mon = CryptoMonitor()
    with patch("data.events.crypto_monitor.get_funding_rate") as gfr, \
         patch("data.events.crypto_monitor.get_open_interest") as goi:
        gfr.return_value = {"symbol": "BTC/USDT", "funding_rate": 0.001,
                            "funding_rate_pct": 0.10, "mark_price": 66000, "index_price": 65000}
        goi.return_value = None
        events = mon.check_moves()
    types = [e["event_type"] for e in events]
    assert "basis_dislocation" in types


def test_oi_surge_needs_a_baseline_first():
    mon = CryptoMonitor()
    with patch("data.events.crypto_monitor.get_funding_rate") as gfr, \
         patch("data.events.crypto_monitor.get_open_interest") as goi:
        gfr.return_value = None
        goi.return_value = {"symbol": "BTC/USDT", "open_interest": 100_000}
        first = mon.check_moves()
        goi.return_value = {"symbol": "BTC/USDT", "open_interest": 120_000}
        second = mon.check_moves()
    assert first == []  # no prior baseline yet
    assert any(e["event_type"] == "oi_surge" for e in second)


def test_monitor_resilient_to_exchange_errors():
    mon = CryptoMonitor()
    with patch("data.events.crypto_monitor.get_funding_rate", side_effect=Exception("boom")), \
         patch("data.events.crypto_monitor.get_open_interest", side_effect=Exception("boom")):
        events = mon.check_moves()  # must not raise
    assert events == []


# ── EventClassifier market-awareness ────────────────────────────────────────

def test_classifier_crypto_rule_fallback_via_market_mode(monkeypatch):
    import agents.event_classifier as ec
    monkeypatch.setattr(ec, "MARKET_MODE", "crypto")
    result = ec.EventClassifier()._rule_based_fallback(
        {"headline": "Bitcoin funding rate spikes to extreme levels"})
    assert result["event_type"] == "funding_spike"
    assert result["is_actionable"] is True


def test_classifier_bursa_rule_fallback_via_market_mode(monkeypatch):
    import agents.event_classifier as ec
    monkeypatch.setattr(ec, "MARKET_MODE", "bursa")
    result = ec.EventClassifier()._rule_based_fallback(
        {"headline": "Maybank profit up on earnings beat"})
    assert result["event_type"] == "earnings_beat"


def test_depeg_always_alerts():
    clf = EventClassifier()
    action = clf.determine_action({"event_type": "depeg", "confidence": 0.1, "is_actionable": False})
    assert action == "alert"


# ── event_watcher ticker-gate fix (regression) ──────────────────────────────

def test_gate0_ticker_gate_accepts_crypto_pairs(monkeypatch):
    import config.settings as settings
    import scripts.event_watcher as ew
    monkeypatch.setattr(settings, "TICKER_REGEX", __import__("re").compile(r"[A-Z0-9]{2,10}/USDT"))

    watcher = ew.EventWatcher.__new__(ew.EventWatcher)  # skip __init__ (no DB/clients needed)

    class _FakeResearcher:
        def save_idea(self, idea):
            return 42
    watcher.researcher = _FakeResearcher()

    idea_id = watcher.create_gate0_idea(
        {"source": "binance_perp"},
        {"event_type": "funding_spike", "ticker": "BTC/USDT", "affected_tickers": ["BTC/USDT"],
         "confidence": 0.8, "historical_edge": "test"},
    )
    assert idea_id == 42  # previously always returned None for non-.KL tickers
