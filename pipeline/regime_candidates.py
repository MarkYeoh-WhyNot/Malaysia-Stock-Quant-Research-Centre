#!/usr/bin/env python3
"""Submit regime-scoped strategy candidates — (base strategy, vol-tercile
filter) pairs backtested as ONE unit.

A regime-scoped candidate is an ordinary DSL tree plus a top-level
`regime_filter` key (signal_dsl validates it; signal_from_dsl masks the
position flat outside the declared terciles with an ex-ante, 1-bar-lagged
expanding-quantile mask). It enters the pipeline at stage2/pending like any
other idea and faces the full gate stack on its composite equity curve —
flat segments included.

Honesty invariant: choosing WHICH terciles to trade is a degree of freedom.
Every submission therefore carries an optimizer_runs row charging at least
the 6-member implicit choice set (3 singleton + 3 pair active-sets) to the
deflated-Sharpe hurdle. gates.evaluate_gates WARNs if the row is missing.
The winner_json carries the full DSL so the backtest never re-parses (and
so the regime_filter can't be lost to an LLM round-trip).

Slug prefix: `rg-`.
"""
from __future__ import annotations

import json
import logging

from data.database import db_session

logger = logging.getLogger(__name__)

# 3 singleton + 3 pair active-sets over {low, mid, high}: the implicit menu
# every regime-scoped submission chose from, charged whether or not the
# submitter "looked" at the alternatives.
REGIME_CHOICE_SET_SIZE = 6


def submit_regime_scoped_idea(base_tree: dict, active: list[str], *,
                              title: str, hypothesis: str, ticker: str,
                              timeframe: str, factor_formula: str = "",
                              extra_trials: int = 0) -> dict:
    """Insert a regime-scoped idea at stage2/pending with its DOF charge.

    Returns {"ok": True, "idea_id": ...} or {"ok": False, "error": ...}.
    `extra_trials` lets a caller that screened base_tree variants add its own
    honest trial count on top of the regime-choice charge.
    """
    from agents.backtest_engineer.signal_dsl import canonical_signature, validate

    tree = dict(base_tree)
    tree["regime_filter"] = {"type": "vol_tercile", "active": list(active)}
    errors = validate(tree)
    if errors:
        return {"ok": False, "error": f"invalid tree: {errors}"}

    signature = canonical_signature(tree, ticker)
    slug = (f"rg-{'-'.join(sorted(active))}-"
            f"{str(ticker).replace('/', '')}-{timeframe}-"
            f"{signature.split(':')[-1][:10]}")

    with db_session() as conn:
        dup = conn.execute(
            "SELECT id FROM alpha_ideas WHERE signal_signature=? "
            "AND status != 'rejected' LIMIT 1", (signature,)).fetchone()
        if dup:
            return {"ok": False, "error": f"duplicate of idea {dup['id']}"}

        cur = conn.execute(
            """INSERT INTO alpha_ideas
                 (slug, title, hypothesis, ticker, timeframe, factor_formula,
                  data_sources, stage, status, novelty_score, logic_score,
                  feasibility_score, signal_signature, family)
               VALUES (?,?,?,?,?,?,'[]','stage2','pending',0.7,0.7,0.7,?,?)""",
            (slug, title,
             f"{hypothesis} [regime-scoped: active only in {sorted(active)} "
             f"vol terciles, flat otherwise]",
             ticker, timeframe, factor_formula or json.dumps(tree),
             signature, "regime_scoped"))
        idea_id = cur.lastrowid

        n_configs = REGIME_CHOICE_SET_SIZE + max(0, int(extra_trials))
        conn.execute(
            """INSERT INTO optimizer_runs
                 (idea_id, status, seed, n_configs, started_at, finished_at,
                  summary_json, winner_json)
               VALUES (?, 'done', 0, ?, datetime('now'), datetime('now'), ?, ?)""",
            (idea_id, n_configs,
             json.dumps({"note": "regime-scoped DOF charge",
                         "regime_choice_set": REGIME_CHOICE_SET_SIZE,
                         "extra_trials": int(extra_trials)}),
             json.dumps({"dsl": tree, "instrument": ticker,
                         "timeframe": timeframe})))

    logger.info(f"[RegimeCandidates] submitted idea {idea_id} slug={slug} "
                f"active={sorted(active)} n_configs={n_configs}")
    return {"ok": True, "idea_id": idea_id, "slug": slug,
            "signature": signature, "n_configs": n_configs}
