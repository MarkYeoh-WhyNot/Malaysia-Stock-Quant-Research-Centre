"""Continuous per-name factor registry — the cross-sectional counterpart of
signal_dsl's boolean leaves.

A FACTOR maps one name's OHLCV frame to a CONTINUOUS score series (higher =
more attractive on the long side); the cross-sectional machinery then ranks
those scores ACROSS the universe at each rebalance (long top-N / short
bottom-N) and computes proper Spearman ICs against forward returns — something
the boolean 0/1 DSL signals fundamentally cannot support (a binary flag has
almost no rank information).

Registry entries mirror the LEAVES spec shape (params with typed ranges,
required df columns) so validation stays uniform across both systems.

Contract: fn(df, **params) -> pd.Series aligned to df.index, NaN during
warmup, NEVER forward-looking (each bar's value uses data up to and including
that bar's close; the consumers add their own next-bar execution delay).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _f_momentum(df: pd.DataFrame, period: int = 30) -> pd.Series:
    """Trailing return over `period` bars — cross-sectional momentum classic."""
    return df["close"].pct_change(int(period))


def _f_reversal(df: pd.DataFrame, period: int = 5) -> pd.Series:
    """Negative short-horizon return — short-term reversal factor."""
    return -df["close"].pct_change(int(period))


def _f_ts_zscore(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Time-series z-score of price vs its own rolling mean (negated so that
    'washed-out below the mean' ranks HIGH = attractive long — mean-reversion
    semantics consistent with 'higher score = long side')."""
    close = df["close"]
    period = int(period)
    mean = close.rolling(period).mean()
    std = close.rolling(period).std().replace(0, np.nan)
    return -(close - mean) / std


def _f_vol_ratio(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Volume surge vs rolling average — participation/breakout factor."""
    vol = df["volume"].astype(float)
    base = vol.rolling(int(period)).mean().replace(0, np.nan)
    return vol / base


def _f_funding_avg(df: pd.DataFrame, period: int = 21) -> pd.Series:
    """NEGATED trailing mean funding rate (per-8h decimal, ffill'd per bar).

    Funding carry classic: a pair with persistently POSITIVE funding has
    crowded longs paying shorts — the attractive position is SHORT (collect
    the carry), so high positive funding must rank LOW. Negation keeps the
    registry-wide 'higher score = long side' convention: deeply negative
    funding (shorts paying) ranks high = go long, collect carry."""
    return -df["funding_rate"].rolling(int(period)).mean()


def _f_funding_zscore(df: pd.DataFrame, period: int = 60) -> pd.Series:
    """NEGATED funding z-score — how extreme funding is vs its own history,
    signed so crowded-long extremes rank LOW (short side)."""
    fr = df["funding_rate"]
    period = int(period)
    mean = fr.rolling(period).mean()
    std = fr.rolling(period).std().replace(0, np.nan)
    return -(fr - mean) / std


FACTORS: dict = {
    "momentum": {
        "fn": _f_momentum,
        "params": {"period": ("int", 5, 252)},
        "columns": ["close"],
        "describe": "trailing {period}-bar return (cross-sectional momentum)",
    },
    "reversal": {
        "fn": _f_reversal,
        "params": {"period": ("int", 2, 30)},
        "columns": ["close"],
        "describe": "negative {period}-bar return (short-term reversal)",
    },
    "ts_zscore": {
        "fn": _f_ts_zscore,
        "params": {"period": ("int", 10, 200)},
        "columns": ["close"],
        "describe": "negated z-score of price vs {period}-bar mean (mean reversion)",
    },
    "vol_ratio": {
        "fn": _f_vol_ratio,
        "params": {"period": ("int", 5, 60)},
        "columns": ["close", "volume"],
        "describe": "volume vs {period}-bar average (participation surge)",
    },
    "funding_avg": {
        "fn": _f_funding_avg,
        "params": {"period": ("int", 3, 90)},
        "columns": ["funding_rate"],
        "describe": "negated {period}-bar mean funding (carry: short crowded longs)",
    },
    "funding_zscore": {
        "fn": _f_funding_zscore,
        "params": {"period": ("int", 10, 200)},
        "columns": ["funding_rate"],
        "describe": "negated funding z-score over {period} bars (contrarian carry)",
    },
}


def validate_factor(name: str, params: dict) -> dict:
    """Validate + clamp a factor spec against the registry. Raises ValueError
    with a caller-displayable message on an unknown factor or bad param."""
    if name not in FACTORS:
        raise ValueError(f"unknown factor '{name}' — available: {sorted(FACTORS)}")
    spec = FACTORS[name]
    clean: dict = {}
    for pname, (ptype, lo, hi) in spec["params"].items():
        if pname not in (params or {}):
            continue  # factor fn's own default applies
        val = params[pname]
        try:
            val = int(val) if ptype == "int" else float(val)
        except (TypeError, ValueError):
            raise ValueError(f"factor '{name}' param '{pname}' must be {ptype}")
        if not (lo <= val <= hi):
            raise ValueError(
                f"factor '{name}' param '{pname}'={val} outside [{lo}, {hi}]")
        clean[pname] = val
    return clean


def compute_factor(name: str, df: pd.DataFrame, params: dict | None = None) -> pd.Series:
    """Compute a registered factor on one name's frame (validated params)."""
    clean = validate_factor(name, params or {})
    spec = FACTORS[name]
    missing = [c for c in spec["columns"] if c not in df.columns]
    if missing:
        # Missing column (e.g. funding_rate on a frame that never got the
        # merge) → NaN series, consistent with the DSL leaves' behavior:
        # downstream ranks simply exclude the name rather than crash.
        return pd.Series(np.nan, index=df.index)
    return spec["fn"](df, **clean)


def required_columns(name: str) -> list:
    """Extra df columns a factor needs beyond plain OHLCV."""
    if name not in FACTORS:
        return []
    return [c for c in FACTORS[name]["columns"]
            if c not in ("open", "high", "low", "close", "volume")]


def factor_catalog_text() -> str:
    """Human/LLM-readable factor list for prompts (ranges only, no defaults)."""
    lines = []
    for name, spec in FACTORS.items():
        parts = [f"{p}: {t} in [{lo}, {hi}]"
                 for p, (t, lo, hi) in spec["params"].items()]
        lines.append(f"- {name}({', '.join(parts)}) — {spec['describe']}")
    return "\n".join(lines)
