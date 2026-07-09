"""Dashboard concierge panel: per-market copy (text-level checks — no JS harness)."""
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
