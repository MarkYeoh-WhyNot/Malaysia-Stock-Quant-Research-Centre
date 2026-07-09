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


# ── Formula templates: Bursa pinned long-only, crypto long/short ─────────────

def test_bursa_templates_stay_long_only():
    """Default (Bursa) process: crypto template overrides must NOT apply —
    ALLOW_SHORT=False, so the WS2 long-only phrasing is pinned."""
    import scripts.event_watcher as ew
    assert "reduce/avoid new long entries" in ew.FACTOR_FORMULA_TEMPLATES["funding_spike"]
    assert "enter short" not in ew.FACTOR_FORMULA_TEMPLATES["funding_spike"]
    assert ew.FACTOR_FORMULA_TEMPLATES["unlock"].startswith("Reduce exposure")
    assert "Enter long when RSI(14) > 45" in ew.FACTOR_FORMULA_TEMPLATES["earnings_beat"]


def test_crypto_event_types_have_typed_emojis():
    """Every crypto event type in EVENT_DOMAIN_MAP renders a typed alert tag,
    not the [INFO] fallback."""
    import scripts.event_watcher as ew
    crypto_types = ["funding_spike", "oi_surge", "basis_dislocation", "btc_move",
                    "eth_move", "btc_dominance_shift", "listing", "unlock",
                    "depeg", "dxy_move", "yield_move", "regulatory"]
    for t in crypto_types:
        assert t in ew.EVENT_DOMAIN_MAP
        assert t in ew.EVENT_TYPE_EMOJIS, f"{t} would fall back to [INFO]"


# ── run_cycle news gating: Brave runs in BOTH markets, Bursa-only sources don't ─

class _Recorder:
    def __init__(self, result=None):
        self.calls = 0
        self.result = result if result is not None else []

    def __call__(self, *a, **kw):
        self.calls += 1
        return self.result


def _bare_watcher(ew):
    w = ew.EventWatcher.__new__(ew.EventWatcher)
    w._rss_cycle = 0
    w._cycle_count = 0
    w.rss = type("R", (), {})()
    w.bursa = type("B", (), {})()
    w.commodities = type("C", (), {})()
    w.crypto = type("X", (), {})()
    w.rss.fetch_all_feeds = _Recorder()
    w.bursa.fetch_announcements = _Recorder()
    w.commodities.check_moves = _Recorder()
    w.crypto.check_moves = _Recorder()
    w.check_economic_calendar = _Recorder()
    return w


def test_run_cycle_crypto_fetches_news_skips_bursa_sources(monkeypatch):
    import config.settings as settings
    import scripts.event_watcher as ew
    monkeypatch.setattr(settings, "MARKET_MODE", "crypto")
    monkeypatch.setattr(ew, "_log_daemon", lambda *a, **kw: None)
    w = _bare_watcher(ew)
    for _ in range(3):
        w.run_cycle()
    assert w.rss.fetch_all_feeds.calls == 1          # 3rd cycle → news search ran
    assert w.bursa.fetch_announcements.calls == 0    # Bursa-only source skipped
    assert w.check_economic_calendar.calls == 0      # Bursa-only source skipped
    assert w.crypto.check_moves.calls == 3           # perp monitor every cycle


def test_run_cycle_bursa_news_cadence_unchanged(monkeypatch):
    import config.settings as settings
    import scripts.event_watcher as ew
    monkeypatch.setattr(settings, "MARKET_MODE", "bursa")
    monkeypatch.setattr(ew, "_log_daemon", lambda *a, **kw: None)
    w = _bare_watcher(ew)
    for _ in range(6):
        w.run_cycle()
    assert w.rss.fetch_all_feeds.calls == 2          # every 3rd cycle, as before
    assert w.bursa.fetch_announcements.calls == 6
    assert w.check_economic_calendar.calls == 6
    assert w.crypto.check_moves.calls == 0
