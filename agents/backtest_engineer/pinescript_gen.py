"""Deterministic DSL condition tree -> TradingView Pine Script v5 translator.

Design principle (mirrors signal_dsl.py's own honesty contract): this module
translates the EXACT tree an idea was backtested with — it never asks an LLM
to write Pine Script from free text. If the tree uses a leaf backed by data
TradingView charts don't carry (dividends, CPO futures, perp funding), the
translator DECLINES with a specific reason instead of fabricating an
approximation. The code shown to a user is therefore always faithful to what
was actually gated/backtested, or absent — never silently wrong.

Public entry point: generate_pinescript(dsl, title, timeframe, allow_short).
"""
from __future__ import annotations

# Leaves backed only by OHLCV — safely translatable to Pine's built-in ta.*.
_SUPPORTED_LEAVES = {
    "rsi", "sma_cross", "ema_cross", "momentum", "reversal", "bollinger",
    "macd", "volume_ratio", "gap", "rolling_rank", "zscore",
}

# Leaves needing data no TradingView chart carries — decline, don't guess.
_UNSUPPORTED_REASONS = {
    "div_days_to_ex": "uses the Bursa ex-dividend calendar, which TradingView "
                      "charts do not carry",
    "cpo_change": "uses the CPO futures data column, which isn't available as "
                 "a plain OHLCV series on a TradingView chart",
    "funding_level": "uses the perp funding rate, which TradingView has no "
                     "built-in access to",
    "funding_zscore": "uses the perp funding rate, which TradingView has no "
                      "built-in access to",
}


class _Unrepresentable(Exception):
    def __init__(self, leaf: str):
        self.leaf = leaf
        super().__init__(_UNSUPPORTED_REASONS.get(leaf, f"leaf '{leaf}' unsupported"))


def _fnum(x) -> str:
    """Pine float literal — always with a decimal point."""
    v = float(x)
    return repr(v) if not v.is_integer() else f"{v:.1f}"


class _Ctx:
    """Collects deduplicated indicator setup lines, keyed by a value that
    uniquely identifies the indicator (leaf name + its numeric params)."""

    def __init__(self):
        self._cache: dict[tuple, object] = {}
        self.lines: list[str] = []
        self._counter = 0

    def fresh(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}{self._counter}"

    def cached(self, key: tuple):
        return self._cache.get(key)

    def store(self, key: tuple, value):
        self._cache[key] = value
        return value


def _leaf_to_pine(node: dict, ctx: _Ctx) -> str:
    leaf = node["leaf"]
    if leaf not in _SUPPORTED_LEAVES:
        raise _Unrepresentable(leaf)

    if leaf == "rsi":
        period = int(node["period"])
        key = ("rsi", period)
        var = ctx.cached(key)
        if var is None:
            var = ctx.fresh("rsi")
            ctx.lines.append(f"{var} = ta.rsi(close, {period})")
            ctx.store(key, var)
        if "below" in node:
            return f"{var} < {_fnum(node['below'])}"
        return f"{var} > {_fnum(node['above'])}"

    if leaf in ("sma_cross", "ema_cross"):
        fast, slow = int(node["fast"]), int(node["slow"])
        fn = "ta.sma" if leaf == "sma_cross" else "ta.ema"
        key = (leaf, fast, slow)
        pair = ctx.cached(key)
        if pair is None:
            fvar, svar = ctx.fresh("fastMA"), ctx.fresh("slowMA")
            ctx.lines.append(f"{fvar} = {fn}(close, {fast})")
            ctx.lines.append(f"{svar} = {fn}(close, {slow})")
            pair = ctx.store(key, (fvar, svar))
        fvar, svar = pair
        direction = node.get("direction", "above")
        return f"{fvar} > {svar}" if direction == "above" else f"{fvar} < {svar}"

    if leaf == "momentum":
        period = int(node["period"])
        key = ("mom", period)
        var = ctx.cached(key)
        if var is None:
            var = ctx.fresh("mom")
            ctx.lines.append(f"{var} = close / close[{period}] - 1")
            ctx.store(key, var)
        return f"{var} > {_fnum(node['min_return'])}"

    if leaf == "reversal":
        period = int(node["period"])
        key = ("rev", period)
        var = ctx.cached(key)
        if var is None:
            var = ctx.fresh("rev")
            ctx.lines.append(f"{var} = close / close[{period}] - 1")
            ctx.store(key, var)
        return f"{var} < {_fnum(node['max_return'])}"

    if leaf == "bollinger":
        period, std = int(node["period"]), float(node["std"])
        key = ("boll", period, std)
        pair = ctx.cached(key)
        if pair is None:
            basis, dev = ctx.fresh("bbBasis"), ctx.fresh("bbDev")
            upper, lower = ctx.fresh("bbUpper"), ctx.fresh("bbLower")
            ctx.lines.append(f"{basis} = ta.sma(close, {period})")
            ctx.lines.append(f"{dev} = {_fnum(std)} * ta.stdev(close, {period})")
            ctx.lines.append(f"{upper} = {basis} + {dev}")
            ctx.lines.append(f"{lower} = {basis} - {dev}")
            pair = ctx.store(key, (upper, lower))
        upper, lower = pair
        band = node.get("band", "below_lower")
        return f"close < {lower}" if band == "below_lower" else f"close > {upper}"

    if leaf == "macd":
        fast, slow, signal = int(node["fast"]), int(node["slow"]), int(node["signal"])
        key = ("macd", fast, slow, signal)
        pair = ctx.cached(key)
        if pair is None:
            mvar, svar = ctx.fresh("macdLine"), ctx.fresh("signalLine")
            ctx.lines.append(f"[{mvar}, {svar}, _] = ta.macd(close, {fast}, {slow}, {signal})")
            pair = ctx.store(key, (mvar, svar))
        mvar, svar = pair
        condition = node.get("condition", "bullish")
        return f"{mvar} > {svar}" if condition == "bullish" else f"{mvar} < {svar}"

    if leaf == "volume_ratio":
        period = int(node["period"])
        key = ("volratio", period)
        var = ctx.cached(key)
        if var is None:
            var = ctx.fresh("volMA")
            ctx.lines.append(f"{var} = ta.sma(volume, {period})")
            ctx.store(key, var)
        return f"volume > {_fnum(node['min_ratio'])} * {var}"

    if leaf == "gap":
        key = ("gap",)
        var = ctx.cached(key)
        if var is None:
            var = ctx.fresh("gapPct")
            ctx.lines.append(f"{var} = (open - close[1]) / close[1]")
            ctx.store(key, var)
        direction = node.get("direction", "up")
        if direction == "up":
            return f"{var} > {_fnum(node['min_pct'])}"
        return f"{var} < -{_fnum(node['min_pct'])}"

    if leaf == "rolling_rank":
        formation = int(node["formation"])
        skip = int(node.get("skip", 0))
        window = int(node.get("window", 252))
        key = ("rank", formation, skip, window)
        var = ctx.cached(key)
        if var is None:
            ret_var = ctx.fresh("formRet")
            var = ctx.fresh("rankPct")
            ctx.lines.append(f"{ret_var} = close[{skip}] / close[{skip + formation}] - 1")
            ctx.lines.append(f"{var} = ta.percentrank({ret_var}, {window})")
            ctx.store(key, var)
        if "min_pct" in node:
            return f"{var} >= {_fnum(float(node['min_pct']) * 100)}"
        return f"{var} <= {_fnum(float(node['max_pct']) * 100)}"

    if leaf == "zscore":
        period = int(node["period"])
        key = ("zscore", period)
        var = ctx.cached(key)
        if var is None:
            mean_var, std_var = ctx.fresh("zMean"), ctx.fresh("zStd")
            var = ctx.fresh("z")
            ctx.lines.append(f"{mean_var} = ta.sma(close, {period})")
            ctx.lines.append(f"{std_var} = ta.stdev(close, {period})")
            ctx.lines.append(f"{var} = (close - {mean_var}) / {std_var}")
            ctx.store(key, var)
        if "below" in node:
            return f"{var} < {_fnum(node['below'])}"
        return f"{var} > {_fnum(node['above'])}"

    raise _Unrepresentable(leaf)   # unreachable given the membership check above


def _node_to_pine(node: dict | None, ctx: _Ctx) -> str | None:
    if node is None:
        return None
    if "op" in node:
        op = node["op"]
        if op == "AND":
            parts = [_node_to_pine(c, ctx) for c in node["children"]]
            return "(" + " and ".join(parts) + ")"
        if op == "OR":
            parts = [_node_to_pine(c, ctx) for c in node["children"]]
            return "(" + " or ".join(parts) + ")"
        if op == "NOT":
            # Explicit inner parens: Pine's `not` binds looser than comparison
            # operators, so `not a < b` is technically unambiguous — but a
            # generated script a human may read/edit shouldn't rely on that.
            return f"(not ({_node_to_pine(node['child'], ctx)}))"
        raise ValueError(f"unknown op {op!r}")
    return _leaf_to_pine(node, ctx)


def generate_pinescript(dsl: dict, title: str, timeframe: str,
                        allow_short: bool) -> dict:
    """Translate a validated DSL tree into a pasteable Pine Script v5 strategy.

    Returns {"ok": True, "code": str} or {"ok": False, "reason": str} — the
    reason names the specific unsupported leaf, mirroring _parse_factor's
    representable:false contract rather than approximating.
    """
    ctx = _Ctx()
    try:
        long_entry = _node_to_pine(dsl.get("entry"), ctx)
        long_exit = _node_to_pine(dsl.get("exit"), ctx)
        short_entry = None
        short_exit = None
        if allow_short and dsl.get("short_entry"):
            short_entry = _node_to_pine(dsl.get("short_entry"), ctx)
            short_exit = _node_to_pine(dsl.get("short_exit"), ctx)
    except _Unrepresentable as exc:
        return {"ok": False,
                "reason": f"'{exc.leaf}' {_UNSUPPORTED_REASONS.get(exc.leaf, 'is unsupported')}"}

    safe_title = (title or "Strategy").replace('"', "'")[:60]
    lines: list[str] = []
    lines.append("//@version=5")
    lines.append(f'strategy("{safe_title}", overlay=true, pyramiding=0, '
                 "default_qty_type=strategy.percent_of_equity, default_qty_value=100)")
    lines.append("")
    lines.append("// Auto-generated from the EXACT condition tree this idea was")
    lines.append("// backtested with in Mark's Research Centre — not independently")
    lines.append(f"// written. Intended chart timeframe: {timeframe}.")
    lines.append("// NOTE: Bursa stamp duty / crypto funding accrual are NOT modeled")
    lines.append("// by TradingView's default strategy tester — expect this script's")
    lines.append("// backtest numbers to differ from this system's reported Sharpe.")
    lines.append("")
    lines.extend(ctx.lines)
    lines.append("")

    if long_entry is not None:
        lines.append(f"longEntryCond = {long_entry}")
    if long_exit is not None:
        lines.append(f"longExitCond = {long_exit}")
    if short_entry is not None:
        # long takes priority on a same-bar overlap — mirrors signal_from_dsl's
        # documented tie-break exactly.
        lines.append(f"shortEntryCond = ({short_entry}) and not longEntryCond")
    if short_exit is not None:
        lines.append(f"shortExitCond = {short_exit}")

    lines.append("")
    if long_entry is not None:
        lines.append('strategy.entry("Long", strategy.long, when=longEntryCond)')
    if long_exit is not None:
        lines.append('strategy.close("Long", when=longExitCond)')
    if short_entry is not None:
        lines.append('strategy.entry("Short", strategy.short, when=shortEntryCond)')
    if short_exit is not None:
        lines.append('strategy.close("Short", when=shortExitCond)')

    return {"ok": True, "code": "\n".join(lines) + "\n"}
