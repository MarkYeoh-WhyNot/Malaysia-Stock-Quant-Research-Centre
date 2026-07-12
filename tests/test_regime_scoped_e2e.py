"""End-to-end regime-scoped candidate test through the REAL gate stack, using
the calibration harness's injection seams (synthetic prices, patched parser —
no LLM, no network).

Scenario: an edge that exists ONLY in the high-vol regime — calm low-vol
random walk for the first 60% of history, violent OU mean-reversion for the
last 40%. The same z-score reversion rule is submitted twice:

  * unscoped — trades the whole history; its edge is diluted across terciles
    and the QC5 regime gate demands >= 2/3 positive;
  * scoped to ["high_vol"] — flat outside its regime by construction; QC5
    demands its ONE declared tercile be positive.

The pin is the QC5 verdicts, not overall_pass (PSR/n_trials depend on shared
DB state and are pinned elsewhere).
"""
import numpy as np
import pandas as pd
import pytest

from data.database import db_session, init_db
from scripts.calibration_harness import _ohlcv, ou_series, random_walk

_TICKER = "TESTRG"
_N_CALM, _N_WILD = 960, 640
_DAILY_VALUE = 60_000_000.0

_TREE = {"entry": {"leaf": "zscore", "period": 20, "below": -0.5},
         "exit":  {"leaf": "zscore", "period": 20, "above": 0.0}}


_CHILD_TABLES = ("backtest_runs", "optimizer_runs", "gate_decisions",
                 "pipeline_events", "paper_trades")


def _purge():
    with db_session() as conn:
        for r in conn.execute(
                "SELECT id FROM alpha_ideas WHERE slug LIKE 'test-rge2e-%' "
                "OR (slug LIKE 'rg-%' AND ticker=?)", (_TICKER,)):
            for tbl in _CHILD_TABLES:
                conn.execute(f"DELETE FROM {tbl} WHERE idea_id=?", (r["id"],))
            conn.execute("DELETE FROM alpha_ideas WHERE id=?", (r["id"],))


@pytest.fixture(autouse=True)
def _setup():
    init_db()
    _purge()
    yield
    _purge()


def _series(seed=6):
    calm = random_walk(_N_CALM, seed, sigma=0.004)
    wild = ou_series(_N_WILD, seed + 1, kappa=0.13, sigma=0.03)
    # splice the OU segment onto the calm walk's terminal level
    wild = wild * (calm[-1] / wild[0])
    return np.concatenate([calm, wild])


def _patched_engine(df, tree):
    from agents.backtest_engineer.backtest_engineer import BacktestEngineer
    from config.settings import DEFAULT_SYMBOLS
    n = len(df)
    index = df.index
    bench = {sym: _ohlcv(random_walk(n, 500 + i), index, _DAILY_VALUE)
             for i, sym in enumerate(DEFAULT_SYMBOLS)}
    eng = BacktestEngineer()
    eng._fetch_prices = (  # type: ignore[assignment]
        lambda symbol, interval="1d", days=1825, _df=df, _b=bench: (
            _df if symbol.split(",")[0].strip() == _TICKER
            else _b.get(symbol.split(",")[0].strip(),
                        _ohlcv(random_walk(n, 999), index, _DAILY_VALUE))
        ).copy())
    eng._parse_factor = (  # type: ignore[assignment]
        lambda formula, title, hyp, _t=tree: {
            "signal_type": "dsl", "dsl": _t, "representable": True})
    return eng


def _insert_unscoped(tree):
    import json
    with db_session() as conn:
        cur = conn.execute(
            """INSERT INTO alpha_ideas
                 (slug, title, hypothesis, ticker, timeframe, factor_formula,
                  stage, status, novelty_score, logic_score, feasibility_score)
               VALUES (?,?,?,?,?,?, 'stage2', 'processing', 0.8, 0.8, 0.8)""",
            (f"test-rge2e-unscoped-{np.random.randint(1_000_000)}",
             "regime e2e unscoped probe", "SYNTHETIC regime-scoped e2e fixture",
             _TICKER, "1d", "zscore reversion (synthetic)"))
        idea_id = cur.lastrowid
        conn.execute(
            """INSERT INTO optimizer_runs
                 (idea_id, status, seed, n_configs, started_at, finished_at,
                  summary_json, winner_json)
               VALUES (?, 'done', 0, 1, datetime('now'), datetime('now'), ?, ?)""",
            (idea_id, json.dumps({"note": "e2e fixture"}),
             json.dumps({"dsl": tree, "instrument": _TICKER, "timeframe": "1d"})))
    return idea_id


def test_scoped_passes_qc5_where_unscoped_fails():
    from pipeline.regime_candidates import submit_regime_scoped_idea

    n = _N_CALM + _N_WILD
    index = pd.date_range("2020-01-01", periods=n, freq="D")
    df = _ohlcv(_series(), index, _DAILY_VALUE)

    # unscoped: edge lives only in the violent tail → not robust across terciles
    unscoped_id = _insert_unscoped(_TREE)
    r_un = _patched_engine(df, _TREE).backtest_idea(unscoped_id)
    assert r_un.get("regime_pass") is False, (
        f"unscoped should fail QC5: regimes={r_un.get('regimes_positive')}")

    # scoped to high_vol: every DECLARED tercile positive → QC5 passes
    sub = submit_regime_scoped_idea(
        _TREE, ["high_vol"], title="regime e2e scoped probe",
        hypothesis="SYNTHETIC regime-scoped e2e fixture", ticker=_TICKER,
        timeframe="1d")
    assert sub["ok"], sub
    scoped_tree = dict(_TREE)
    scoped_tree["regime_filter"] = {"type": "vol_tercile", "active": ["high_vol"]}
    r_sc = _patched_engine(df, scoped_tree).backtest_idea(sub["idea_id"])
    assert r_sc.get("regime_pass") is True, (
        f"scoped should pass QC5: high-vol sharpe="
        f"{(r_sc.get('regimes') or {}).get('sharpe_high_vol')}, "
        f"verdict={r_sc.get('verdict_reason')}")
