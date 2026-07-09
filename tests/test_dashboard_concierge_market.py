"""Dashboard per-market display branches (text-level checks — no JS harness):
concierge panel, factor sandbox, event feed, paper trades."""
import os

HTML = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "dashboard", "ui", "index.html")


def _html() -> str:
    with open(HTML, encoding="utf-8") as f:
        return f.read()


def test_concierge_panel_has_market_aware_hooks():
    html = _html()
    assert 'id="cc-subtitle"' in html
    assert 'id="cc-examples"' in html
    # crypto branch mutates the concierge copy
    assert "Long/short perps, paper-only" in html
    assert "Short ETH/USDT" in html
    assert "Short SOL/USDT on RSI overbought" in html


def test_concierge_greeting_has_both_market_variants():
    html = _html()
    # Bursa default copy stays verbatim
    assert "I'm long-only and paper-only: I can get an idea ready for a live" in html
    assert "e.g. Test a 20/50 MA crossover on Malaysian banks" in html
    # crypto greeting variant exists behind IS_CRYPTO
    assert "I can trade long or short (perps, paper-only)" in html


def test_sandbox_and_paper_trades_have_market_aware_hooks():
    html = _html()
    # hooks the crypto branch mutates
    for el_id in ('id="sb-pair-label"', 'id="pt-pair-label"', 'id="sb-data-note"',
                  'id="sb-sigref-body"'):
        assert el_id in html
    # both dropdowns rebuilt from the market-aware universe endpoint
    assert "fetch(API + '/universe')" in html
    assert "'sb-pair', 'pt-pair'" in html
    # crypto copy present behind IS_CRYPTO
    assert "Binance daily data via ccxt" in html
    assert "Short ETH/USDT when RSI(14) > 75" in html
    # Bursa inline defaults untouched
    assert "1155.KL — Maybank" in html
    assert "Yahoo Finance .KL data" in html


def test_event_feed_has_market_aware_pills_and_hidden_calendar():
    html = _html()
    assert 'id="ev-filter-pills"' in html
    assert 'id="calendar-strip-panel"' in html
    # crypto pills behind IS_CRYPTO (built from the [type, label] array)
    for entry in ("'funding_spike', 'Funding'", "'oi_surge', 'OI Surge'",
                  "'basis_dislocation', 'Basis'", "'unlock', 'Unlock'"):
        assert entry in html
    # Bursa inline pills untouched
    assert ">Earnings Beat</button>" in html
    assert "filterEvents('opr_hike'" in html
    # crypto event types now have card colours
    for ev in ("funding_spike", "depeg", "listing"):
        assert f"{ev}: '#" in html


def test_universe_endpoint_returns_bursa_universe():
    from dashboard.api.server import universe
    out = universe()
    symbols = [u["symbol"] for u in out["universe"]]
    assert "1155.KL" in symbols
    assert len(symbols) >= 20
    assert all(set(u) == {"symbol", "name", "sector"} for u in out["universe"])
