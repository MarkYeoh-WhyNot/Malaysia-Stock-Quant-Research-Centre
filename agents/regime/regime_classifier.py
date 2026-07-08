"""Malaysia macro/sector regime classifier (Phase 5.3, audit §7.5/§13.3).

Deliberately network-free: classification takes already-fetched price series
(from MPOBScraper.get_historical_cpo(), etc.) rather than fetching itself, so it
stays unit-testable. OPR and MYR/USD have no live feed wired into this repo yet
(no data/forex client, no BNM scraper) — those fields are accepted as optional
inputs and default to "unknown" until a real source is connected; this is a
scaffold for classification logic, not a claim that OPR/FX are live-tracked.
"""
from __future__ import annotations

import pandas as pd

from data.database import db_session


def _trend(series: pd.Series, lookback: int = 20) -> str:
    """Simple trend label from a price series: up / down / flat over `lookback`
    periods, using percent change vs a noise threshold."""
    if series is None or len(series) < lookback + 1:
        return "unknown"
    change = float(series.iloc[-1] / series.iloc[-lookback - 1] - 1)
    if change > 0.03:
        return "up"
    if change < -0.03:
        return "down"
    return "flat"


def classify_regime(cpo_prices: pd.Series | None = None,
                    opr_trend: str | None = None,
                    myr_trend: str | None = None) -> dict:
    """Combine available macro signals into a regime label.

    Only CPO has a live data source in this repo today (MPOBScraper); OPR/MYR
    trends are optional and default to "unknown" — the label degrades
    gracefully rather than guessing.
    """
    cpo_trend = _trend(cpo_prices) if cpo_prices is not None else "unknown"
    opr_trend = opr_trend or "unknown"
    myr_trend = myr_trend or "unknown"

    parts = []
    if cpo_trend != "unknown":
        parts.append(f"commodity_{cpo_trend}cycle" if cpo_trend != "flat" else "commodity_stable")
    if opr_trend != "unknown":
        parts.append(f"opr_{opr_trend}")
    if myr_trend != "unknown":
        parts.append(f"myr_{myr_trend}")

    label = "_".join(parts) if parts else "unclassified"
    return {
        "cpo_trend": cpo_trend, "opr_trend": opr_trend, "myr_trend": myr_trend,
        "regime_label": label,
    }


def persist_macro_snapshot(as_of_date: str, cpo_price: float | None,
                           regime: dict, opr: float | None = None,
                           myr_usd: float | None = None,
                           brent_crude: float | None = None,
                           source: str = "regime_classifier") -> None:
    """Write one day's macro_features row (idempotent per date)."""
    with db_session() as conn:
        conn.execute("""
            INSERT INTO macro_features
              (as_of_date, opr, myr_usd, brent_crude, cpo_price, cpo_trend,
               regime_label, source)
            VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT(as_of_date) DO UPDATE SET
              opr=excluded.opr, myr_usd=excluded.myr_usd,
              brent_crude=excluded.brent_crude, cpo_price=excluded.cpo_price,
              cpo_trend=excluded.cpo_trend, regime_label=excluded.regime_label,
              source=excluded.source
        """, (as_of_date, opr, myr_usd, brent_crude, cpo_price,
              regime.get("cpo_trend"), regime.get("regime_label"), source))
