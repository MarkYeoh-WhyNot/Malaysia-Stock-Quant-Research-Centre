"""Composable signal DSL — the honest replacement for the flat template enum.

An idea's factor_formula is parsed (by Haiku, in _parse_factor) into a JSON
condition tree evaluated here with fixed numpy/pandas — the LLM chooses
structure and parameters, never writes code. Trees compose leaf conditions
with AND/OR/NOT, so multi-condition theses ("volume spike AND gap up") are
representable instead of being flattened onto the single nearest template.

Contract: if an idea cannot be expressed with these leaves, the parser must
return {"representable": false, "reason": ...} and the idea is REJECTED with
that reason — silent genericization to momentum is forbidden.

Example tree:
    {"entry": {"op": "AND", "children": [
         {"leaf": "volume_ratio", "period": 20, "min_ratio": 1.5},
         {"leaf": "gap", "direction": "up", "min_pct": 0.02}]},
     "exit":  {"leaf": "rsi", "period": 14, "above": 70}}

With an exit tree the position is a state machine (enter on entry-true, hold
until exit-true). Without one, the entry condition itself is the position
regime (like the legacy sma_crossover behaviour).
"""
import hashlib
import json

import numpy as np
import pandas as pd

# ── Leaf registry ─────────────────────────────────────────────────────────────
# Each leaf: params with type/range (used by validate() and perturbation),
# required df columns, and a compute(df, node) -> boolean Series.

def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def _leaf_rsi(df, node):
    rsi = _rsi(df["close"], int(node["period"]))
    if "below" in node:
        return rsi < float(node["below"])
    return rsi > float(node["above"])


def _leaf_ma_cross(df, node, ema: bool):
    close = df["close"]
    fp, sp = int(node["fast"]), int(node["slow"])
    if ema:
        fast = close.ewm(span=fp, adjust=False).mean()
        slow = close.ewm(span=sp, adjust=False).mean()
    else:
        fast = close.rolling(fp).mean()
        slow = close.rolling(sp).mean()
    return fast > slow if node.get("direction", "above") == "above" else fast < slow


def _leaf_momentum(df, node):
    return df["close"].pct_change(int(node["period"])) > float(node["min_return"])


def _leaf_reversal(df, node):
    return df["close"].pct_change(int(node["period"])) < float(node["max_return"])


def _leaf_bollinger(df, node):
    close = df["close"]
    period, std_mult = int(node["period"]), float(node["std"])
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    if node.get("band", "below_lower") == "below_lower":
        return close < mid - std_mult * std
    return close > mid + std_mult * std


def _leaf_macd(df, node):
    close = df["close"]
    macd_line = (close.ewm(span=int(node["fast"]), adjust=False).mean()
                 - close.ewm(span=int(node["slow"]), adjust=False).mean())
    signal_line = macd_line.ewm(span=int(node["signal"]), adjust=False).mean()
    if node.get("condition", "bullish") == "bullish":
        return macd_line > signal_line
    return macd_line < signal_line


def _leaf_volume_ratio(df, node):
    vol_ma = df["volume"].rolling(int(node["period"])).mean()
    return df["volume"] > float(node["min_ratio"]) * vol_ma


def _leaf_gap(df, node):
    open_ = df["open"] if "open" in df.columns else df["close"]
    gap_pct = (open_ - df["close"].shift(1)) / df["close"].shift(1)
    if node.get("direction", "up") == "up":
        return gap_pct > float(node["min_pct"])
    return gap_pct < -float(node["min_pct"])


def _leaf_rolling_rank(df, node):
    """Time-series percentile rank of formation-period momentum (the same
    construction as the legacy cross_sectional_momentum template, but with
    the idea's own parameters instead of hardcoded 21/126/252/0.80)."""
    formation = df["close"].shift(int(node.get("skip", 0))).pct_change(int(node["formation"]))
    pct = formation.rolling(int(node.get("window", 252))).rank(pct=True)
    if "min_pct" in node:
        return pct >= float(node["min_pct"])
    return pct <= float(node["max_pct"])


def _leaf_div_days_to_ex(df, node):
    """True when the next dividend ex-date is within max_days trading days.

    The dividends column marks cash amounts ON their ex-date. Bursa ex-dates
    are announced weeks in advance, so conditioning on an upcoming (already
    announced) ex-date is realistic, not lookahead — but the backtest can only
    approximate the announcement with the realized ex-date calendar. Ideas
    passing this leaf should be treated as conditional on announcement data
    at execution time (paper trading uses the real calendar).
    """
    div = df["dividends"].fillna(0.0)
    idx_positions = np.arange(len(div))
    ex_positions = np.where(div.values > 0, idx_positions, np.nan)
    # next ex-date position at or after each bar
    next_ex = pd.Series(ex_positions, index=div.index).bfill()
    days_until = next_ex - idx_positions
    return pd.Series(days_until <= int(node["max_days"]), index=div.index).fillna(False)


def _leaf_cpo_change(df, node):
    return df["cpo_close"].pct_change(int(node["period"])) > float(node["min_pct"])


def _leaf_zscore(df, node):
    """Rolling z-score of close: z = (close - mean(period)) / std(period).
    'below' fires when z < value (e.g. -2 → classic mean-reversion long);
    'above' fires when z > value (e.g. +2 → overextension, short leg)."""
    close = df["close"]
    period = int(node["period"])
    mean = close.rolling(period).mean()
    std = close.rolling(period).std().replace(0, np.nan)
    z = (close - mean) / std
    if "below" in node:
        return z < float(node["below"])
    return z > float(node["above"])


def _leaf_funding_level(df, node):
    """Perp funding rate vs an absolute per-8h threshold (crypto only).

    df["funding_rate"] is the LAST SETTLED rate ffill'd to each bar
    (backward-looking; the engine's shift(1) adds the trade delay on top).
    'above' 0.0005 → crowded longs (classic short-entry context);
    'below' -0.0003 → crowded shorts / washed-out (long-entry context)."""
    fr = df["funding_rate"]
    if "below" in node:
        return fr < float(node["below"])
    return fr > float(node["above"])


def _leaf_funding_zscore(df, node):
    """Rolling z-score of the funding rate — 'how extreme is funding now vs
    its own recent history' (period counted in BARS of the idea's timeframe)."""
    fr = df["funding_rate"]
    period = int(node["period"])
    mean = fr.rolling(period).mean()
    std = fr.rolling(period).std().replace(0, np.nan)
    z = (fr - mean) / std
    if "below" in node:
        return z < float(node["below"])
    return z > float(node["above"])


LEAVES = {
    "rsi": {
        "compute": _leaf_rsi,
        "columns": ["close"],
        "params": {"period": ("int", 2, 50)},
        "one_of": [("below", ("float", 1, 99)), ("above", ("float", 1, 99))],
    },
    "sma_cross": {
        "compute": lambda df, n: _leaf_ma_cross(df, n, ema=False),
        "columns": ["close"],
        "params": {"fast": ("int", 2, 100), "slow": ("int", 5, 300)},
        "choices": {"direction": ["above", "below"]},
    },
    "ema_cross": {
        "compute": lambda df, n: _leaf_ma_cross(df, n, ema=True),
        "columns": ["close"],
        "params": {"fast": ("int", 2, 100), "slow": ("int", 5, 300)},
        "choices": {"direction": ["above", "below"]},
    },
    "momentum": {
        "compute": _leaf_momentum,
        "columns": ["close"],
        "params": {"period": ("int", 2, 252), "min_return": ("float", -0.5, 0.5)},
    },
    "reversal": {
        "compute": _leaf_reversal,
        "columns": ["close"],
        "params": {"period": ("int", 2, 30), "max_return": ("float", -0.5, 0.0)},
    },
    "bollinger": {
        "compute": _leaf_bollinger,
        "columns": ["close"],
        "params": {"period": ("int", 5, 60), "std": ("float", 0.5, 4.0)},
        "choices": {"band": ["below_lower", "above_upper"]},
    },
    "macd": {
        "compute": _leaf_macd,
        "columns": ["close"],
        "params": {"fast": ("int", 2, 50), "slow": ("int", 5, 100), "signal": ("int", 2, 30)},
        "choices": {"condition": ["bullish", "bearish"]},
    },
    "volume_ratio": {
        "compute": _leaf_volume_ratio,
        "columns": ["close", "volume"],
        "params": {"period": ("int", 5, 60), "min_ratio": ("float", 1.0, 10.0)},
    },
    "gap": {
        "compute": _leaf_gap,
        "columns": ["close"],
        "params": {"min_pct": ("float", 0.001, 0.2)},
        "choices": {"direction": ["up", "down"]},
    },
    "rolling_rank": {
        "compute": _leaf_rolling_rank,
        "columns": ["close"],
        "params": {"formation": ("int", 20, 252), "skip": ("int", 0, 30),
                   "window": ("int", 60, 504)},
        "one_of": [("min_pct", ("float", 0.5, 1.0)), ("max_pct", ("float", 0.0, 0.5))],
    },
    "div_days_to_ex": {
        "compute": _leaf_div_days_to_ex,
        "columns": ["dividends"],
        "params": {"max_days": ("int", 1, 30)},
    },
    "cpo_change": {
        "compute": _leaf_cpo_change,
        "columns": ["cpo_close"],
        "params": {"period": ("int", 1, 30), "min_pct": ("float", -0.2, 0.2)},
    },
    "zscore": {
        # Rolling z-score of price ("standard deviations from the N-bar mean").
        # Mean-reversion classic: entry z < -T, short_entry z > +T.
        "compute": _leaf_zscore,
        "columns": ["close"],
        "params": {"period": ("int", 10, 200)},
        "one_of": [("below", ("float", -4.0, 0.0)), ("above", ("float", 0.0, 4.0))],
    },
    "funding_level": {
        # Perp funding vs absolute per-8h threshold (crypto only — the
        # funding_rate column is merged from REAL historical settlements).
        # Typical extremes: ±0.0005 (0.05%/8h). Crowded longs pay positive.
        "compute": _leaf_funding_level,
        "columns": ["funding_rate"],
        "params": {},
        "one_of": [("below", ("float", -0.005, 0.0)), ("above", ("float", 0.0, 0.005))],
    },
    "funding_zscore": {
        # How extreme is funding now vs its own rolling history (in bars).
        # Contrarian classic: short_entry z > +2 (crowded longs), entry z < -2.
        "compute": _leaf_funding_zscore,
        "columns": ["funding_rate"],
        "params": {"period": ("int", 10, 200)},
        "one_of": [("below", ("float", -4.0, 0.0)), ("above", ("float", 0.0, 4.0))],
    },
}

MAX_DEPTH = 4
MAX_LEAVES = 6


# ── Validation ────────────────────────────────────────────────────────────────

def validate(tree: dict) -> list[str]:
    """Structural + range validation. Returns [] when the tree is clean."""
    errors: list[str] = []
    leaf_count = [0]

    def _walk(node, depth):
        if depth > MAX_DEPTH:
            errors.append(f"tree deeper than {MAX_DEPTH}")
            return
        if not isinstance(node, dict):
            errors.append(f"node is not an object: {node!r}")
            return
        if "op" in node:
            op = node["op"]
            if op in ("AND", "OR"):
                children = node.get("children", [])
                if not isinstance(children, list) or len(children) < 2:
                    errors.append(f"{op} needs >=2 children")
                    return
                for c in children:
                    _walk(c, depth + 1)
            elif op == "NOT":
                if "child" not in node:
                    errors.append("NOT needs a child")
                    return
                _walk(node["child"], depth + 1)
            else:
                errors.append(f"unknown op {op!r}")
            return
        leaf = node.get("leaf")
        if leaf not in LEAVES:
            errors.append(f"unknown leaf {leaf!r}")
            return
        leaf_count[0] += 1
        spec = LEAVES[leaf]
        for pname, (ptype, lo, hi) in spec.get("params", {}).items():
            if pname not in node:
                errors.append(f"{leaf}: missing param {pname}")
                continue
            try:
                val = float(node[pname])
            except (TypeError, ValueError):
                errors.append(f"{leaf}.{pname}: not numeric")
                continue
            if not (lo <= val <= hi):
                errors.append(f"{leaf}.{pname}={val} outside [{lo}, {hi}]")
        one_of = spec.get("one_of")
        if one_of:
            present = [name for name, _ in one_of if name in node]
            if len(present) != 1:
                errors.append(f"{leaf}: exactly one of "
                              f"{[n for n, _ in one_of]} required, got {present}")
            else:
                name = present[0]
                _, (ptype, lo, hi) = next(x for x in one_of if x[0] == name)
                try:
                    val = float(node[name])
                    if not (lo <= val <= hi):
                        errors.append(f"{leaf}.{name}={val} outside [{lo}, {hi}]")
                except (TypeError, ValueError):
                    errors.append(f"{leaf}.{name}: not numeric")
        for cname, choices in spec.get("choices", {}).items():
            if cname in node and node[cname] not in choices:
                errors.append(f"{leaf}.{cname}={node[cname]!r} not in {choices}")

    # A tree needs at least one side: entry (long) or short_entry (short —
    # crypto perps only, WS3). A short-only tree is valid where ALLOW_SHORT.
    entry = tree.get("entry")
    short_entry = tree.get("short_entry")
    if entry is None and short_entry is None:
        errors.append("missing entry tree (need 'entry' and/or 'short_entry')")
    if entry is not None:
        _walk(entry, 1)
    if tree.get("exit") is not None:
        _walk(tree["exit"], 1)
    if short_entry is not None:
        _walk(short_entry, 1)
    if tree.get("short_exit") is not None:
        _walk(tree["short_exit"], 1)
    if leaf_count[0] > MAX_LEAVES:
        errors.append(f"more than {MAX_LEAVES} leaves")
    # sanity for MA crosses: fast < slow
    def _check_cross(node):
        if isinstance(node, dict):
            if node.get("leaf") in ("sma_cross", "ema_cross", "macd"):
                if float(node.get("fast", 0)) >= float(node.get("slow", 1e9)):
                    errors.append(f"{node['leaf']}: fast >= slow")
            for c in node.get("children", []):
                _check_cross(c)
            if "child" in node:
                _check_cross(node["child"])
    for part in ("entry", "exit", "short_entry", "short_exit"):
        if isinstance(tree.get(part), dict):
            _check_cross(tree[part])
    return errors


def required_columns(tree: dict) -> set[str]:
    """Columns the caller must supply in df (e.g. attach cpo_close)."""
    cols: set[str] = set()

    def _walk(node):
        if not isinstance(node, dict):
            return
        if "leaf" in node and node["leaf"] in LEAVES:
            cols.update(LEAVES[node["leaf"]]["columns"])
        for c in node.get("children", []):
            _walk(c)
        if "child" in node:
            _walk(node["child"])

    for part in ("entry", "exit", "short_entry", "short_exit"):
        if tree.get(part):
            _walk(tree[part])
    return cols


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(df: pd.DataFrame, node: dict) -> pd.Series:
    """Evaluate one condition tree to a boolean Series (NaN-safe: NaN=False)."""
    if "op" in node:
        op = node["op"]
        if op == "AND":
            result = evaluate(df, node["children"][0])
            for child in node["children"][1:]:
                result = result & evaluate(df, child)
            return result
        if op == "OR":
            result = evaluate(df, node["children"][0])
            for child in node["children"][1:]:
                result = result | evaluate(df, child)
            return result
        if op == "NOT":
            return ~evaluate(df, node["child"])
        raise ValueError(f"unknown op {op!r}")
    leaf = node["leaf"]
    series = LEAVES[leaf]["compute"](df, node)
    return series.fillna(False).astype(bool)


def _side_series(df: pd.DataFrame, entry_node: dict | None, exit_node: dict | None) -> pd.Series | None:
    """State machine for ONE side (long or short): in-position from
    entry-true until exit-true; without an exit, entry itself is the regime.
    Returns None if entry_node is absent (that side isn't used by this tree)."""
    if entry_node is None:
        return None
    entry = evaluate(df, entry_node)
    if exit_node:
        exit_ = evaluate(df, exit_node)
        raw = np.where(entry, 1.0, np.where(exit_, 0.0, np.nan))
        return pd.Series(raw, index=df.index).ffill().fillna(0.0)
    return entry.astype(float)


def signal_from_dsl(df: pd.DataFrame, dsl: dict) -> pd.Series:
    """Position series from a DSL tree: 0/1 long-only (Bursa, and any crypto
    tree that only sets entry/exit), or -1/0/1 when a short leg is present.

    Long leg: dsl["entry"] / dsl["exit"] (unchanged contract).
    Short leg (crypto perps, WS3): dsl["short_entry"] / dsl["short_exit"] —
    only meaningful where settings.ALLOW_SHORT; ignored otherwise so a Bursa
    idea can never accidentally short. If both legs would be in-position on
    the same bar (a malformed tree), long takes priority — documented, not
    silently arbitrary.

    Lookahead is handled downstream by _compute_performance's shift(1) guard.
    """
    long_sig = _side_series(df, dsl.get("entry"), dsl.get("exit"))

    from config.settings import ALLOW_SHORT
    short_sig = None
    if ALLOW_SHORT and dsl.get("short_entry"):
        short_sig = _side_series(df, dsl.get("short_entry"), dsl.get("short_exit"))

    if short_sig is None:
        return long_sig if long_sig is not None else pd.Series(0.0, index=df.index)
    if long_sig is None:
        return -short_sig
    return pd.Series(
        np.where(long_sig > 0, 1.0, np.where(short_sig > 0, -1.0, 0.0)),
        index=df.index,
    )


# ── Signature (semantic dedup) ────────────────────────────────────────────────

def _normalize(node):
    if isinstance(node, dict):
        out = {}
        for k in sorted(node):
            v = node[k]
            if isinstance(v, float):
                out[k] = round(v, 4)
            elif isinstance(v, (dict, list)):
                out[k] = _normalize(v)
            else:
                out[k] = v
        return out
    if isinstance(node, list):
        # AND/OR are commutative — sort children canonically
        return sorted((_normalize(x) for x in node),
                      key=lambda x: json.dumps(x, sort_keys=True))
    return node


def canonical_signature(dsl: dict, ticker: str) -> str:
    """Stable hash of the strategy's semantic content: same tree + ticker →
    same signature regardless of title wording or key ordering."""
    payload = {
        "ticker": (ticker or "").strip().upper(),
        "entry": _normalize(dsl.get("entry")),
        "exit": _normalize(dsl.get("exit")),
    }
    # Only add short-leg keys when present, so a long-only tree (Bursa, or any
    # crypto idea with no short leg) hashes IDENTICALLY to before this change —
    # changing the payload shape unconditionally would silently break dedup
    # continuity against signal_signature values already stored in the DB.
    if dsl.get("short_entry") is not None:
        payload["short_entry"] = _normalize(dsl.get("short_entry"))
    if dsl.get("short_exit") is not None:
        payload["short_exit"] = _normalize(dsl.get("short_exit"))
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()


# ── Parameter perturbation (robustness gate) ─────────────────────────────────

def perturb_tree(tree: dict, rng: np.random.RandomState, scale: float = 0.2) -> dict:
    """Copy of the tree with every numeric param multiplied by
    U(1-scale, 1+scale), clamped to its declared range (ints re-rounded).
    Used by the robustness gate: a real edge should survive ±20% parameter
    jitter; a knife-edge fit will not."""
    def _perturb(node):
        if not isinstance(node, dict):
            return node
        out = dict(node)
        leaf = node.get("leaf")
        if leaf in LEAVES:
            spec = LEAVES[leaf]
            all_params = dict(spec.get("params", {}))
            for name, prange in spec.get("one_of", []):
                all_params[name] = prange
            for pname, (ptype, lo, hi) in all_params.items():
                if pname in out:
                    val = float(out[pname]) * rng.uniform(1 - scale, 1 + scale)
                    val = min(max(val, lo), hi)
                    out[pname] = int(round(val)) if ptype == "int" else round(val, 6)
            # keep MA fast<slow invariant after jitter
            if leaf in ("sma_cross", "ema_cross", "macd"):
                if float(out.get("fast", 0)) >= float(out.get("slow", 1e9)):
                    out["fast"] = max(2, int(out["slow"]) - 1)
        if "children" in out:
            out["children"] = [_perturb(c) for c in out["children"]]
        if "child" in out:
            out["child"] = _perturb(out["child"])
        return out

    out = {
        "entry": _perturb(tree["entry"]) if tree.get("entry") else None,
        "exit": _perturb(tree["exit"]) if tree.get("exit") else None,
    }
    if tree.get("short_entry"):
        out["short_entry"] = _perturb(tree["short_entry"])
    if tree.get("short_exit"):
        out["short_exit"] = _perturb(tree["short_exit"])
    return out


# ── Parser-facing catalog (embedded in the Haiku prompt) ─────────────────────

def leaf_catalog_text() -> str:
    """Human/LLM-readable leaf catalog with param names and RANGES only —
    deliberately no default values, so the parser extracts parameters from
    the idea text instead of anchoring on suggestions."""
    lines = []
    for name, spec in LEAVES.items():
        parts = []
        for pname, (ptype, lo, hi) in spec.get("params", {}).items():
            parts.append(f"{pname}: {ptype} in [{lo}, {hi}]")
        for oname, (ptype, lo, hi) in spec.get("one_of", []):
            parts.append(f"{oname} (pick one): {ptype} in [{lo}, {hi}]")
        for cname, choices in spec.get("choices", {}).items():
            parts.append(f"{cname}: one of {choices}")
        lines.append(f"- {name}({', '.join(parts)})")
    return "\n".join(lines)
