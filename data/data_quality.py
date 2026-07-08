"""Data Confidence Score + corporate-action anomaly detection (Phase 1.2/1.4).

The pipeline runs on yfinance / scraped data, so every backtest should carry a
data-quality score (audit §6). A strategy can look profitable purely because of
bad adjusted prices, missing days, or stale quotes — this module quantifies that
risk so Gate DQ can block low-confidence data before expensive backtesting.

Pure and network-free: functions take an already-fetched OHLCV DataFrame.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# Weights for the 0–100 Data Confidence Score. Kept explicit so the score is
# auditable and tunable. Sum = 1.0.
_DQ_WEIGHTS = {
    "price_completeness":  0.35,
    "volume_completeness": 0.20,
    "stale_price":         0.25,   # scored as (1 - stale_fraction)
    "missing_day":         0.20,   # scored as (1 - missing_fraction)
}


def compute_data_confidence(df: pd.DataFrame, interval: str = "1d") -> dict:
    """Return a data-quality assessment for an OHLCV frame.

    Components (each 0..1, higher = better):
      - price_completeness:  non-null, positive closes / expected bars
      - volume_completeness: bars with volume > 0 / total bars
      - stale_price:         1 − fraction of consecutive-identical closes
      - missing_day:         1 − fraction of business days absent from the index

    Combined into confidence_score (0..100). Clean daily blue-chip data scores
    high (~95–100); thin or gappy series score low.
    """
    n = len(df)
    if n == 0 or "close" not in df:
        return {"confidence_score": 0.0, "bars": 0,
                "price_completeness": 0.0, "volume_completeness": 0.0,
                "stale_price_frac": 1.0, "missing_day_frac": 1.0,
                "notes": "empty frame"}

    close = df["close"]
    valid_close = close.notna() & (close > 0)
    price_completeness = float(valid_close.mean())

    if "volume" in df:
        volume_completeness = float((df["volume"].fillna(0) > 0).mean())
    else:
        volume_completeness = 0.0

    # Stale prices: consecutive identical closes (a frozen/failed feed).
    if n > 1:
        stale_frac = float((close.diff().fillna(1) == 0).mean())
    else:
        stale_frac = 0.0

    # Missing days: gaps vs the market's expected trading calendar over the
    # span. Bursa trades business days (bdate_range); crypto trades 24/7
    # (date_range) — using the business calendar for crypto would flag every
    # weekend as "missing" and Gate DQ would false-reject everything.
    from config.settings import MARKET_CALENDAR
    missing_frac = 0.0
    if interval == "1d" and n > 1 and isinstance(df.index, pd.DatetimeIndex):
        if MARKET_CALENDAR == "daily":
            expected = pd.date_range(df.index[0], df.index[-1], freq="D")
        else:
            expected = pd.bdate_range(df.index[0], df.index[-1])
        if len(expected):
            present = df.index.normalize().intersection(expected.normalize())
            missing_frac = float(max(0.0, 1.0 - len(present) / len(expected)))

    components = {
        "price_completeness":  price_completeness,
        "volume_completeness": volume_completeness,
        "stale_price":         1.0 - stale_frac,
        "missing_day":         1.0 - missing_frac,
    }
    score = 100.0 * sum(_DQ_WEIGHTS[k] * components[k] for k in _DQ_WEIGHTS)

    notes = []
    if price_completeness < 0.98:
        notes.append(f"price completeness {price_completeness:.1%}")
    if stale_frac > 0.05:
        notes.append(f"stale prices {stale_frac:.1%}")
    if missing_frac > 0.05:
        notes.append(f"missing days {missing_frac:.1%}")
    if volume_completeness < 0.90:
        notes.append(f"volume gaps {1 - volume_completeness:.1%}")

    return {
        "confidence_score":    round(score, 1),
        "bars":                n,
        "price_completeness":  round(price_completeness, 4),
        "volume_completeness": round(volume_completeness, 4),
        "stale_price_frac":    round(stale_frac, 4),
        "missing_day_frac":    round(missing_frac, 4),
        "notes":               "; ".join(notes) or "clean",
    }


def detect_corporate_action_anomalies(df: pd.DataFrame,
                                      gap_threshold: float = 0.25) -> list[dict]:
    """Flag large unexplained overnight price gaps as suspected unhandled
    corporate actions (bonus/rights issues — yfinance adjusts splits/dividends
    but handles Bursa bonus/rights poorly). A ~50% drop on a 2-for-1 bonus, or a
    step from a rights issue, shows up as an outsized close-to-close move.

    Returns a list of {date, prev_close, close, pct_change} for gaps beyond
    ±gap_threshold. Feeds the corporate_actions table (validation_status
    'suspected') and lowers the Data Confidence Score.
    """
    if "close" not in df or len(df) < 2:
        return []
    ret = df["close"].pct_change()
    hits = ret[ret.abs() >= gap_threshold]
    out = []
    for dt, r in hits.items():
        i = df.index.get_loc(dt)
        prev = float(df["close"].iloc[i - 1]) if i > 0 else float("nan")
        out.append({
            "date": dt.strftime("%Y-%m-%d") if hasattr(dt, "strftime") else str(dt),
            "prev_close": round(prev, 4),
            "close": round(float(df["close"].loc[dt]), 4),
            "pct_change": round(float(r), 4),
        })
    return out
