"""LLM-backed signal parsing: translating free-text factor formulas into
the signal DSL condition tree, and post-hoc verifying that generated
signals match the formula's stated intent.

These are the ONLY two Claude API calls in the backtest_engineer package.

The parser honesty contract is SACRED (see CLAUDE.md): a strategy that
cannot be represented EXACTLY with the available DSL leaves must return
{"representable": false, "reason": ...} — silent approximation onto the
nearest template is forbidden. This module only relocates the existing
parsing/verification logic; the contract itself is unchanged.

Both functions take `engine` (a BacktestEngineer instance) as their first
argument so they can call `engine.call_claude_json(...)` and
`engine.log_daemon(...)` (BaseAgent methods) and, for verify_formula,
`engine._compute_signals` via the engine module. BacktestEngineer keeps
thin delegating instance methods (`_parse_factor`, `verify_formula`) so
existing instance-level test/harness monkeypatches
(e.g. `eng._parse_factor = lambda f, t, h: {...}`) keep working — they
override the instance attribute, which Python's method resolution checks
before the class (and therefore before this module is ever reached).
"""
from __future__ import annotations

import pandas as pd

from config.settings import MODEL_FAST, MARKET_NAME as _MARKET_NAME_FOR_SYSTEM

SYSTEM = f"""You are a quantitative backtesting engineer specialising in {_MARKET_NAME_FOR_SYSTEM}.
Parse strategy descriptions into structured signal parameters for vectorised backtesting.
Output only valid JSON."""


def parse_factor(engine, factor_formula: str, title: str, hypothesis: str) -> dict:
    """Parse the idea's free text into a signal DSL condition tree.

    Honesty contract: if the idea cannot be expressed with the available
    leaves, the parser must say so ({"representable": false, "reason"})
    and the idea is rejected with that reason — it is NEVER silently
    genericized onto the nearest template (the historical failure mode
    that flattened every thesis into 20-day momentum).

    The catalog shows parameter RANGES only, no example values — the old
    prompt pre-filled defaults (20/50/14/35/65...) and Haiku anchored on
    them instead of extracting the idea's own parameters.
    """
    from agents.backtest_engineer import signal_dsl
    from config.settings import ALLOW_SHORT, MARKET_NAME

    short_shape = ""
    short_rule = "Long-only (Bursa short-selling restricted): the entry tree describes when to be LONG."
    if ALLOW_SHORT:
        short_shape = (
            '  "short_entry": <condition tree or null — when to go SHORT, if the strategy '
            'has a short thesis>,\n'
            '  "short_exit": <condition tree or null — when to cover the short; null means '
            'hold short while short_entry is true>,\n'
        )
        short_rule = ("Long AND short are both supported (perpetuals). A tree may set entry/exit "
                     "(long leg), short_entry/short_exit (short leg), or both if the strategy "
                     "genuinely trades both directions — most single-direction ideas need only one "
                     "leg. If the strategy is a pure short thesis, entry/exit may be null.")

    prompt = f"""Translate this {MARKET_NAME} strategy into a signal condition tree.

Factor formula: {factor_formula}
Strategy title: {title}
Hypothesis: {hypothesis}

AVAILABLE CONDITIONS (parameters MUST come from the strategy text; ranges are hard limits):
{signal_dsl.leaf_catalog_text()}

CONDITION SHAPE GUIDE (structure only — every parameter VALUE must come from the strategy text):
{signal_dsl.shape_cards_text()}

{signal_dsl.PARSER_NEGATIVE_EXAMPLE}

Combinators: {{"op": "AND"|"OR", "children": [<node>, <node>, ...]}} and {{"op": "NOT", "child": <node>}}.
A leaf node looks like {{"leaf": "<name>", <params>}}.

Return JSON, one of these three shapes:

1. Representable as price/volume/dividend/CPO conditions:
{{
  "representable": true,
  "entry": <condition tree or null — when to be LONG>,
  "exit": <condition tree or null — when to flatten the long; null means hold while entry condition is true>,
{short_shape}  "notes": "one line: how the tree captures the strategy's actual thesis"
}}

2. A fundamental screen across 5+ stocks (ROE/PB/PE/DY ranking or filtering):
{{"representable": true, "route": "fundamental_screen"}}

3. NOT expressible with the available conditions (requires data or logic none of the leaves provide):
{{"representable": false, "reason": "one specific sentence — what the strategy needs that is unavailable"}}

Rules:
- Extract every numeric parameter from the strategy text. If the text gives no value for a
  required parameter, the strategy is underspecified — use shape 3 with reason "parameter X unspecified".
- NEVER approximate an unrelated mechanism with a price proxy. If the thesis is about earnings
  surprises, analyst coverage, sentiment, or anything with no matching leaf, use shape 3.
- {short_rule}"""
    result = engine.call_claude_json(
        SYSTEM,
        [{"role": "user", "content": prompt}],
        model=MODEL_FAST,
        task_label="parse_factor",
    )
    if not isinstance(result, dict):
        return {"representable": False, "reason": "parser returned non-JSON"}
    if "error" in result:
        return {"representable": False, "reason": "parser JSON parse failure"}

    if result.get("route") == "fundamental_screen":
        return {"signal_type": "fundamental_screen", "route": "fundamental_screen",
                "representable": True}

    if not result.get("representable"):
        return {"representable": False,
                "reason": result.get("reason", "not representable (no reason given)")}

    tree = {"entry": result.get("entry"), "exit": result.get("exit")}
    if ALLOW_SHORT:
        if result.get("short_entry"):
            tree["short_entry"] = result.get("short_entry")
        if result.get("short_exit"):
            tree["short_exit"] = result.get("short_exit")
    errors = signal_dsl.validate(tree)
    if errors:
        return {"representable": False,
                "reason": f"invalid condition tree: {'; '.join(errors[:4])}"}
    return {
        "signal_type": "dsl",
        "representable": True,
        "dsl": tree,
        "long_only": not ALLOW_SHORT,
        "notes": result.get("notes", ""),
    }


def verify_formula(engine, params: dict, factor_formula: str, df: pd.DataFrame) -> dict:
    """Verify that the parsed signal code matches the formula description.

    Runs the signal on the last 20 bars and asks Claude to confirm the
    signals are directionally correct and match the formula intent.

    Returns dict with keys: verified (bool), confidence (float), issue (str).
    """
    if df.empty or len(df) < 30:
        return {"verified": False, "confidence": 0.0, "issue": "insufficient data for verification"}

    from agents.backtest_engineer import engine as engine_mod

    sample_df = df.iloc[-20:].copy()
    try:
        signals = engine_mod._compute_signals(engine, sample_df, params)
    except Exception as e:
        return {"verified": False, "confidence": 0.0, "issue": f"signal computation error: {e}"}

    # Build human-readable samples
    close_sample = sample_df["close"].round(4).tolist()
    signal_sample = signals.fillna(0).astype(int).tolist()
    dates_sample  = [str(d)[:10] for d in sample_df.index.tolist()]

    bars_table = "\n".join(
        f"  {dates_sample[i]}: close={close_sample[i]:.3f}  signal={signal_sample[i]}"
        for i in range(len(dates_sample))
    )

    verify_prompt = f"""The factor formula says: {factor_formula}

The code produced these signals on the last 20 bars (1=long, 0=flat):
{bars_table}

Does the signal output match what the formula describes?
Are the entry/exit points logical given the price data?
Is the direction correct (long when signal=1, flat/no position when signal=0)?

Return JSON only:
{{
  "verified": true,
  "confidence": 0.0,
  "issue": "description of any problem found, or empty string if verified"
}}"""

    result = engine.call_claude_json(
        SYSTEM,
        [{"role": "user", "content": verify_prompt}],
        model=MODEL_FAST,
        task_label="verify_formula",
    )
    verified   = bool(result.get("verified", False))
    confidence = float(result.get("confidence", 0.0))
    issue      = result.get("issue", "")

    if not verified or confidence < 0.7:
        engine.log_daemon("ERROR", f"Formula verification failed (confidence={confidence:.2f}): {issue}")
    else:
        engine.log_daemon("INFO", f"Formula verified with confidence {confidence:.2f}")

    return {
        "verified":   verified,
        "confidence": confidence,
        "issue":      issue,
    }
