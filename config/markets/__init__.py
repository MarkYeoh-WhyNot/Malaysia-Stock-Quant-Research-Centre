"""Market profiles — one module per market, selected by the MARKET_MODE env var.

Each profile module exposes the same surface (universe, calendar, cost model,
ticker rules, prompt briefs, enabled jobs). `config.settings` loads exactly one
profile per process and re-exports its values under the legacy names every
consumer already imports — so one process is always exactly one market, and the
two pipelines never share state (each container also gets its own
OPENCLAW_RUNTIME_DIR, hence its own DB).

Adding a market = adding one module here + a container set in docker-compose.
"""
from __future__ import annotations

import importlib

VALID_MODES = ("bursa", "crypto")


def load_market_profile(mode: str):
    """Return the profile module for `mode` ('bursa' | 'crypto').

    Raises on unknown modes rather than silently defaulting — a typo'd
    MARKET_MODE in a container env must fail loudly at startup, not run the
    wrong market.
    """
    mode = (mode or "bursa").strip().lower()
    if mode not in VALID_MODES:
        raise ValueError(
            f"Unknown MARKET_MODE '{mode}' — expected one of {VALID_MODES}")
    return importlib.import_module(f"config.markets.{mode}")
