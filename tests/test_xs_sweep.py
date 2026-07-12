"""Cross-sectional funding-carry sweep (2026-07-12,
docs/funding_carry_sweep_design.md): config sampling stays inside the factor
registry's declared ranges, the basket scorer mirrors the gated rebalance
semantics, the sweep never touches the test slice (pinned by data-corruption
invariance, not by trust), and the sweep's n_configs honestly inflates the
winner's deflated hurdle through the existing optimizer_runs wiring.

Offline/deterministic throughout — crypto-mode paths run via the same
subprocess pattern as tests/test_funding_and_xs.py.
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

from agents.backtest_engineer.optimizer import (
    randomize_xs_config, _xs_basket_score,
    XS_TOP_N_CHOICES, XS_BOTTOM_N_CHOICES, XS_REBALANCE_CHOICES,
    XS_DEFAULT_N_CONFIGS)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run_mode(market_mode: str, code: str, timeout: int = 240) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        env = {**os.environ, "MARKET_MODE": market_mode,
               "OPENCLAW_RUNTIME_DIR": tmp, "PYTHONPATH": REPO}
        r = subprocess.run([sys.executable, "-c", code], env=env, cwd=REPO,
                           capture_output=True, text=True, timeout=timeout)
        assert r.returncode == 0, f"stderr:\n{r.stderr[-2500:]}"
        return r.stdout


# ── config sampling ───────────────────────────────────────────────────────────

def test_randomize_xs_config_in_registry_range_and_choice_grids():
    from agents.backtest_engineer.factors import FACTORS
    for fname in ("funding_avg", "funding_zscore"):
        _, lo, hi = FACTORS[fname]["params"]["period"]
        rng = np.random.RandomState(5)
        for _ in range(50):
            cfg = randomize_xs_config(fname, rng)
            assert cfg["factor"]["name"] == fname
            assert lo <= cfg["factor"]["params"]["period"] <= hi
            assert cfg["top_n"] in XS_TOP_N_CHOICES
            assert cfg["rebalance_bars"] in XS_REBALANCE_CHOICES
            # default test mode is Bursa: no shorting → bottom leg forced off
            assert cfg["bottom_n"] == 0


def test_randomize_xs_config_seeded_deterministic():
    a = [randomize_xs_config("funding_avg", np.random.RandomState(9)) for _ in range(1)]
    b = [randomize_xs_config("funding_avg", np.random.RandomState(9)) for _ in range(1)]
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_randomize_xs_config_crypto_samples_bottom_n():
    out = _run_mode("crypto", """
import json
import numpy as np
from agents.backtest_engineer.optimizer import (
    randomize_xs_config, XS_BOTTOM_N_CHOICES)
rng = np.random.RandomState(3)
draws = [randomize_xs_config("funding_avg", rng)["bottom_n"] for _ in range(60)]
assert all(b in XS_BOTTOM_N_CHOICES for b in draws)
print("RESULT " + json.dumps({"has_short": any(b > 0 for b in draws)}))
""")
    line = [l for l in out.splitlines() if l.startswith("RESULT ")][-1]
    assert json.loads(line[len("RESULT "):])["has_short"] is True


# ── _ic_series extraction regression (byte-identical to the old inline loop) ─

def test_ic_series_matches_original_inline_loop():
    from agents.backtest_engineer.cross_sectional import _ic_series
    from agents.backtest_engineer import stats

    rng = np.random.RandomState(21)
    idx = pd.date_range("2024-01-01", periods=60, freq="D")
    cols = [f"S{i}" for i in range(8)]
    sig_panel = pd.DataFrame(rng.randn(60, 8), index=idx, columns=cols)
    ret_panel = pd.DataFrame(rng.randn(60, 8) * 0.01, index=idx, columns=cols)
    # sprinkle NaNs (missing names on some dates) + one date below the 5-name floor
    sig_panel.iloc[5, :3] = np.nan
    ret_panel.iloc[9, 2:6] = np.nan
    sig_panel.iloc[30, :4] = np.nan
    ret_panel.iloc[30, 4:] = np.nan

    for spread_ok in (False, True):
        # verbatim copy of the loop as it stood inline in cross_sectional_test
        ic_series, portfolio_rets, spread_rets = [], [], []
        for date in sig_panel.index:
            sig_row = sig_panel.loc[date].dropna()
            ret_row = ret_panel.loc[date].dropna()
            common_stocks = sig_row.index.intersection(ret_row.index)
            if len(common_stocks) < 5:
                continue
            sv = sig_row[common_stocks].values
            rv = ret_row[common_stocks].values
            ic = stats.spearman(sv, rv)
            if not np.isnan(ic):
                ic_series.append(ic)
            n_q = max(1, len(common_stocks) // 5)
            top_idx = np.argsort(sv)[-n_q:]
            if len(top_idx) > 0:
                portfolio_rets.append(float(np.mean(rv[top_idx])))
            if spread_ok:
                bot_idx = np.argsort(sv)[:n_q]
                spread_rets.append(float(np.mean(rv[top_idx]) - np.mean(rv[bot_idx])))

        got_ic, got_port, got_spread = _ic_series(sig_panel, ret_panel, spread_ok)
        assert got_ic == ic_series
        assert got_port == portfolio_rets
        assert got_spread == spread_rets


# ── basket scorer semantics ───────────────────────────────────────────────────

def _tiny_panel(n=60, cols=("A", "B", "C", "D", "E")):
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    close = pd.DataFrame(100.0, index=idx, columns=list(cols))
    zeros = pd.DataFrame(0.0, index=idx, columns=list(cols))
    rates = pd.Series(0.0, index=list(cols))
    return close, zeros, rates


def test_basket_score_no_lookahead_one_bar_delay():
    """A move on the DECISION bar itself must not be captured — weights take
    effect the next bar (shift(1)), same as the gated engine."""
    close, zeros, rates = _tiny_panel()
    score = pd.DataFrame(
        {c: float(i) for i, c in enumerate(close.columns)},
        index=close.index)          # constant ranks: E always top
    # jump in E's price ON bar 0 (the first decision bar): unreachable
    close.loc[close.index[0], "E"] = 110.0
    _, _, _, port = _xs_basket_score(close, score, zeros, rates,
                                     top_n=1, bottom_n=0, rebalance_bars=5,
                                     t=40, ann=252)
    # pct_change bar1 = (100-110)/110 for E — held from bar 0's decision, so
    # the REVERSION is captured (weights active bar 1+), but bar 0 itself
    # contributes nothing (w_held row 0 is all zero).
    assert port.iloc[0] == 0.0


def test_basket_score_costs_and_funding_reduce_returns():
    close, zeros, _ = _tiny_panel()
    # E ranks top until bar 5, then D takes over → one full switch at the
    # second rebalance decision. This pins the 2026-07-12 exit-ffill fix: the
    # old replace(0→NaN).ffill() kept E's stale weight forever (no exit, no
    # exit cost, leverage creep).
    score = pd.DataFrame(
        {c: float(i) for i, c in enumerate(close.columns)},
        index=close.index)
    score.loc[close.index[5]:, "D"] = 99.0
    rates = pd.Series(0.001, index=close.columns)
    _, _, _, port = _xs_basket_score(close, score, zeros, rates,
                                     top_n=1, bottom_n=0, rebalance_bars=5,
                                     t=40, ann=252)
    # flat prices, no funding: the very first entry is uncharged (turnover
    # row 0 is NaN→0 — gated-engine convention, immaterial over a full
    # window); the E→D switch at bar 5 charges BOTH sides (sell E + buy D =
    # 2 × 0.1%/side) — the half the old exit-ffill bug silently dropped.
    assert port.iloc[0] == 0.0
    assert port.iloc[5] == pytest.approx(-0.002)
    assert (port.drop(port.index[5]) == 0.0).all()

    # positive funding drags a long book every bar it is held
    fund = pd.DataFrame(0.0005, index=close.index, columns=close.columns)
    _, _, _, port_f = _xs_basket_score(close, score, fund, rates * 0.0,
                                       top_n=1, bottom_n=0, rebalance_bars=5,
                                       t=40, ann=252)
    assert port_f.iloc[0] == 0.0                    # nothing held on bar 0
    assert port_f.iloc[1:].values == pytest.approx(-0.0005)


def test_basket_score_gross_leverage_never_exceeds_book():
    """Regression pin for the exit-ffill bug: with top_n=1/bottom_n=0 the
    held gross exposure can never exceed 1.0 no matter how membership
    churns."""
    close, zeros, rates = _tiny_panel(n=80)
    rng = np.random.RandomState(13)
    score = pd.DataFrame(rng.randn(80, 5), index=close.index,
                         columns=close.columns)
    _, _, _, port = _xs_basket_score(close, score, zeros,
                                     pd.Series(0.0, index=close.columns),
                                     top_n=1, bottom_n=0, rebalance_bars=5,
                                     t=60, ann=252)
    # flat prices + zero costs + zero funding → port must be exactly zero;
    # under the old bug, stale weights still produced zero HERE (flat prices)
    # so ALSO pin leverage directly via the funding channel: constant funding
    # on a churning top-1 book must drag exactly 1 unit per bar, never more.
    fund = pd.DataFrame(0.001, index=close.index, columns=close.columns)
    _, _, _, port_f = _xs_basket_score(close, score, fund,
                                       pd.Series(0.0, index=close.columns),
                                       top_n=1, bottom_n=0, rebalance_bars=5,
                                       t=60, ann=252)
    assert (port_f.iloc[1:] >= -0.001 - 1e-12).all()   # ≤ 1.0 gross held
    assert port_f.iloc[1:].values == pytest.approx(-0.001)


def test_basket_score_counts_val_rebalances():
    close, zeros, rates = _tiny_panel(n=100)
    rng = np.random.RandomState(4)
    score = pd.DataFrame(rng.randn(100, 5), index=close.index,
                         columns=close.columns)   # ranks churn every rebalance
    train_sh, val_sh, n_val, port = _xs_basket_score(
        close, score, zeros, rates, top_n=2, bottom_n=0, rebalance_bars=5,
        t=60, ann=252)
    # 8 decision rows fall in val (60,65,...,95); with seed 4's random ranks
    # one draw repeats the previous top-2 set, so 7 CHANGES are counted —
    # only weight-changing decisions count as rebalances, same as the engine.
    assert n_val == 7
    assert len(port) == 100


# ── run_xs_sweep end-to-end (crypto subprocess, planted carry edge) ──────────

_XS_SWEEP_SNIPPET = """
import json
import numpy as np
import pandas as pd
from data.database import init_db, db_session
init_db()
from agents.backtest_engineer.backtest_engineer import BacktestEngineer
from config.settings import DEFAULT_SYMBOLS

# Planted funding-carry edge: each name's NEXT-bar return moves AGAINST its
# current funding (crowded longs underperform) — funding_avg (negated trailing
# funding) genuinely predicts.
n = 800
idx = pd.date_range("2022-01-01", periods=n, freq="D")
rng = np.random.default_rng(7)
t_arr = np.arange(n)
frames = {}
for i, sym in enumerate(DEFAULT_SYMBOLS):
    phase = 2 * np.pi * i / max(len(DEFAULT_SYMBOLS), 1)
    funding = 0.0005 * np.sin(2 * np.pi * t_arr / 180.0 + phase)
    rets = -8.0 * np.roll(funding, 1) + 0.01 * rng.standard_normal(n)
    rets[0] = 0.0
    close = 100 * np.exp(np.cumsum(rets))
    frames[sym] = pd.DataFrame({"open": close, "high": close * 1.001,
                                "low": close * 0.999, "close": close,
                                "volume": np.full(n, 1e9),
                                "funding_rate": funding}, index=idx)

CORRUPT = __CORRUPT__
if CORRUPT:
    # scramble ONLY the test slice (rows >= t+v) — the sweep must not notice
    cut = int(n * 0.6) + int(n * 0.2)
    g = np.random.default_rng(999)
    for sym, df in frames.items():
        junk = 100 * np.exp(np.cumsum(0.05 * g.standard_normal(n - cut)))
        for col in ("open", "high", "low", "close"):
            df.iloc[cut:, df.columns.get_loc(col)] = junk
        df.iloc[cut:, df.columns.get_loc("funding_rate")] = \\
            0.01 * g.standard_normal(n - cut)

BacktestEngineer._fetch_prices = lambda self, s, interval="1d", days=1825: \\
    frames.get(s.split(",")[0].strip()).copy()
BacktestEngineer._fetch_funding_history = lambda self, s, days=1825: pd.DataFrame()

with db_session() as conn:
    cur = conn.execute(
        \"\"\"INSERT INTO alpha_ideas
             (slug, title, hypothesis, ticker, timeframe, factor_formula,
              stage, status)
           VALUES ('xs-sweep-test','xs sweep test','carry','UNIVERSE','1d',
                   ?, 'stage2','optimizing')\"\"\",
        ("xs:" + json.dumps({"signal_type": "cross_sectional",
                              "factor": {"name": "funding_avg",
                                         "params": {"period": 21}},
                              "top_n": 4, "bottom_n": 4, "rebalance_bars": 7,
                              "interval": "1d"}),))
    iid = cur.lastrowid

from agents.backtest_engineer.optimizer import run_xs_sweep
r = run_xs_sweep(iid, seed=42, n_configs=40)
assert not r.get("error"), r
w = r["winner"]
print("RESULT " + json.dumps({
    "n_configs": r["n_configs"], "n_evaluated": r["n_evaluated"],
    "n_eligible": r["n_eligible"],
    "winner": w, "top3": r["top"][:3],
    "val_sharpe_dist": r["val_sharpe_dist"], "val_ic_dist": r["val_ic_dist"]},
    sort_keys=True))
"""


def _sweep_result(corrupt: bool) -> dict:
    out = _run_mode("crypto", _XS_SWEEP_SNIPPET.replace("__CORRUPT__", str(corrupt)))
    line = [l for l in out.splitlines() if l.startswith("RESULT ")][-1]
    return json.loads(line[len("RESULT "):])


def test_run_xs_sweep_planted_edge_selects_and_never_touches_test():
    clean = _sweep_result(corrupt=False)
    assert clean["n_configs"] == 40
    assert clean["n_eligible"] > 0
    w = clean["winner"]
    assert w["factor"]["name"] == "funding_avg"
    assert {"top_n", "bottom_n", "rebalance_bars", "interval",
            "val_sharpe", "val_mean_ic"} <= set(w)
    assert "test_sharpe" not in w          # the sweep never peeks at test
    assert w["val_sharpe"] > 0             # planted edge found
    # ranked by val Sharpe, descending
    tops = [t["val_sharpe"] for t in clean["top3"]]
    assert tops == sorted(tops, reverse=True)

    # THE no-test-contact pin: scrambling the test slice changes NOTHING.
    corrupted = _sweep_result(corrupt=True)
    assert corrupted == clean


def test_run_xs_sweep_rejects_non_xs_idea():
    out = _run_mode("crypto", """
import json
from data.database import init_db, db_session
init_db()
with db_session() as conn:
    cur = conn.execute(
        "INSERT INTO alpha_ideas (slug, title, ticker, timeframe, factor_formula, "
        "stage, status) VALUES ('xs-guard','g','BTC/USDT','1d','rsi < 30','stage2','optimizing')")
    iid = cur.lastrowid
from agents.backtest_engineer.optimizer import run_xs_sweep
r = run_xs_sweep(iid, seed=1, n_configs=5)
print("RESULT " + json.dumps({"error": r.get("error", "")}))
""")
    line = [l for l in out.splitlines() if l.startswith("RESULT ")][-1]
    assert "xs:" in json.loads(line[len("RESULT "):])["error"]


# ── deflated-hurdle integration: sweep n_configs reaches the gated run ───────

_HURDLE_SNIPPET = """
import json
import numpy as np
import pandas as pd
from data.database import init_db, db_session
init_db()
from agents.backtest_engineer.backtest_engineer import BacktestEngineer
from config.settings import DEFAULT_SYMBOLS

n = 1200
idx = pd.date_range("2021-01-01", periods=n, freq="D")
rng = np.random.default_rng(11)
t = np.arange(n)
frames = {}
for i, sym in enumerate(DEFAULT_SYMBOLS):
    phase = 2 * np.pi * i / max(len(DEFAULT_SYMBOLS), 1)
    drift = 0.004 * np.sin(2 * np.pi * t / 300.0 + phase)
    close = 100 * np.exp(np.cumsum(drift + 0.010 * rng.standard_normal(n)))
    frames[sym] = pd.DataFrame({"open": close, "high": close * 1.001,
                                "low": close * 0.999, "close": close,
                                "volume": np.full(n, 1e9)}, index=idx)

eng = BacktestEngineer()
eng._fetch_prices = lambda s, interval="1d", days=1825: frames.get(
    s.split(",")[0].strip(), next(iter(frames.values()))).copy()
eng._fetch_funding_history = lambda s, days=1825: pd.DataFrame()

spec = "xs:" + json.dumps({"signal_type": "cross_sectional",
                            "factor": {"name": "momentum", "params": {"period": 30}},
                            "top_n": 4, "bottom_n": 4, "rebalance_bars": 7,
                            "interval": "1d"})
with db_session() as conn:
    cur = conn.execute(
        \"\"\"INSERT INTO alpha_ideas
             (slug, title, hypothesis, ticker, timeframe, factor_formula, stage, status)
           VALUES ('xs-hurdle','xs hurdle','m','UNIVERSE','1d',?,'stage2','processing')\"\"\",
        (spec,))
    iid = cur.lastrowid

r1 = eng.backtest_idea(iid)
base_trials = r1["n_trials"]

# same idea, second run, after a 'done' xs sweep of 200 configs is recorded
with db_session() as conn:
    conn.execute("INSERT INTO optimizer_runs (idea_id, status, seed, n_configs) "
                 "VALUES (?, 'done', 42, 200)", (iid,))
    conn.execute("UPDATE alpha_ideas SET stage='stage2', status='processing' WHERE id=?",
                 (iid,))
with db_session() as conn:
    row = dict(conn.execute("SELECT * FROM alpha_ideas WHERE id=?", (iid,)).fetchone())
params = json.loads(spec[3:])
r2 = eng._run_cross_sectional_backtest(iid, row, params)

print("RESULT " + json.dumps({
    "base_trials": base_trials, "swept_trials": r2["n_trials"],
    "base_hurdle": r1["deflated_hurdle"], "swept_hurdle": r2["deflated_hurdle"]}))
"""


def test_xs_sweep_trials_inflate_gated_deflation_hurdle():
    """THE load-bearing honesty pin: a done optimizer_runs row from the xs
    sweep must raise the same idea's gated-basket deflated hurdle."""
    out = _run_mode("crypto", _HURDLE_SNIPPET)
    line = [l for l in out.splitlines() if l.startswith("RESULT ")][-1]
    res = json.loads(line[len("RESULT "):])
    # +200 sweep trials (r2 also sees r1's own backtest_runs row: >= not ==)
    assert res["swept_trials"] >= res["base_trials"] + 200
    assert res["swept_hurdle"] > res["base_hurdle"]


def test_xs_default_n_configs_is_preregistered_value():
    assert XS_DEFAULT_N_CONFIGS == 200   # pinned to docs/funding_carry_sweep_design.md
