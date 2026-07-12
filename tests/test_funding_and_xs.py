"""Pins for the funding-history integration (WS1) and cross-sectional
validation/strategies (WS2), 2026-07-10.

Everything here is offline/deterministic — network paths are exercised by the
live acceptance runs, not unit tests.
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run_mode(market_mode: str, code: str, timeout: int = 240) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        env = {**os.environ, "MARKET_MODE": market_mode,
               "OPENCLAW_RUNTIME_DIR": tmp, "PYTHONPATH": REPO}
        r = subprocess.run([sys.executable, "-c", code], env=env, cwd=REPO,
                           capture_output=True, text=True, timeout=timeout)
        assert r.returncode == 0, f"stderr:\n{r.stderr[-2500:]}"
        return r.stdout


# ── WS1: funding resample + drag ─────────────────────────────────────────────

def test_funding_bar_sum_assignment_and_no_smearing():
    from agents.backtest_engineer.backtest_engineer import _funding_bar_sum

    # 3 daily bars; settlements at 00:00/08:00/16:00 (+ms jitter like Binance)
    bars = pd.date_range("2024-01-01", periods=3, freq="D")
    settles = pd.DataFrame(
        {"funding_rate": [0.0001, 0.0002, 0.0003, 0.0004]},
        index=pd.to_datetime(["2024-01-01 00:00:00.001",
                               "2024-01-01 08:00:00.001",
                               "2024-01-01 16:00:00.001",
                               "2024-01-02 00:00:00.001"]))
    out = _funding_bar_sum(settles, bars)
    assert abs(out.iloc[0] - 0.0006) < 1e-12   # 3 settlements summed into day 1
    assert abs(out.iloc[1] - 0.0004) < 1e-12   # exactly one into day 2
    assert out.iloc[2] == 0.0                  # none into day 3

    # hourly bars: each settlement lands in EXACTLY one bar, never smeared
    hbars = pd.date_range("2024-01-01", periods=24, freq="h")
    hout = _funding_bar_sum(settles.iloc[:3], hbars)
    assert (hout != 0).sum() == 3
    assert abs(hout.sum() - 0.0006) < 1e-12

    # settlements before the first bar are dropped, not misassigned
    early = pd.DataFrame({"funding_rate": [0.5]},
                         index=pd.to_datetime(["2023-12-31 16:00"]))
    assert _funding_bar_sum(early, bars).sum() == 0.0


def test_drag_uses_real_series_and_short_sign():
    """With a funding_bar_sum column the engine must charge the REAL rates:
    a permanently-long position pays positive funding; a permanently-short
    position RECEIVES it. Runs in a crypto subprocess — funding does not
    exist on Bursa (FUNDING_INTERVAL_HOURS is None → zero drag)."""
    out = _run_mode("crypto", """
import json
import numpy as np
import pandas as pd
from agents.backtest_engineer.backtest_engineer import BacktestEngineer
from agents.backtest_engineer import engine

n = 400
idx = pd.date_range("2023-01-01", periods=n, freq="D")
rng = np.random.default_rng(3)
close = 100 * np.exp(np.cumsum(0.01 * rng.standard_normal(n)))
base = pd.DataFrame({"open": close, "high": close * 1.001,
                     "low": close * 0.999, "close": close,
                     "volume": np.full(n, 1e9)}, index=idx)
eng = BacktestEngineer()
sig_long = pd.Series(1.0, index=idx)
with_col = base.copy()
with_col["funding_bar_sum"] = 0.0009   # brutal +0.03%/8h x 3
r_real = engine._compute_performance(eng, with_col, sig_long, "1d")
r_none = engine._compute_performance(eng, base, sig_long, "1d")   # modeled 0.0003/bar
r_short = engine._compute_performance(eng, with_col, pd.Series(-1.0, index=idx), "1d")
print("RESULT " + json.dumps({
    "real_net": r_real["sharpe_net"], "none_net": r_none["sharpe_net"],
    "real_drag": r_real["funding_drag_pct"], "none_drag": r_none["funding_drag_pct"],
    "short_drag": r_short["funding_drag_pct"]}))
""")
    line = [l for l in out.splitlines() if l.startswith("RESULT ")][-1]
    res = json.loads(line[len("RESULT "):])
    # long pays: heavier real funding → lower net sharpe than modeled fallback
    assert res["real_net"] < res["none_net"], res
    assert res["real_drag"] < res["none_drag"] < 0, res
    assert res["short_drag"] > 0, res   # short RECEIVES positive funding


def test_zero_column_disables_modeled_fallback():
    """funding_bar_sum == 0.0 must mean ZERO drag (the xs engine embeds
    funding in the NAV and attaches a zero column to prevent double-count)."""
    from agents.backtest_engineer.backtest_engineer import BacktestEngineer
    from agents.backtest_engineer import engine
    n = 300
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    close = 100 + np.cumsum(np.random.default_rng(1).standard_normal(n) * 0.5)
    df = pd.DataFrame({"open": close, "high": close, "low": close,
                       "close": close, "volume": np.full(n, 1e9),
                       "funding_bar_sum": 0.0}, index=idx)
    r = engine._compute_performance(BacktestEngineer(), df, pd.Series(1.0, index=idx), "1d")
    assert r["funding_drag_pct"] == 0.0


# ── WS1: DSL leaves + catalog ────────────────────────────────────────────────

def test_funding_leaves_compute_and_catalog():
    from agents.backtest_engineer import signal_dsl

    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    fr = pd.Series(0.0001, index=idx)
    fr.iloc[100:110] = 0.002    # crowded-long extreme
    fr.iloc[150:160] = -0.002   # crowded-short washout
    df = pd.DataFrame({"close": np.linspace(100, 110, n),
                       "volume": 1e9, "funding_rate": fr}, index=idx)

    hot = signal_dsl.LEAVES["funding_level"]["compute"](df, {"above": 0.001})
    cold = signal_dsl.LEAVES["funding_level"]["compute"](df, {"below": -0.001})
    assert hot.iloc[100:110].all() and not hot.iloc[:100].any()
    assert cold.iloc[150:160].all() and not cold.iloc[:150].any()

    z = signal_dsl.LEAVES["funding_zscore"]["compute"](
        df, {"period": 60, "above": 2.0})
    assert z.iloc[100:105].any()          # the spike is a z-extreme
    assert not z.iloc[:99].any()

    cat = signal_dsl.leaf_catalog_text()
    assert "funding_level" in cat and "funding_zscore" in cat
    assert set(signal_dsl.required_columns(
        {"entry": {"leaf": "funding_zscore", "period": 30, "below": -2.0}}
    )) >= {"funding_rate"}


# ── WS2: factor registry ─────────────────────────────────────────────────────

def test_factor_registry_contract():
    from agents.backtest_engineer import factors

    n = 120
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    df = pd.DataFrame({"close": np.linspace(100, 150, n),
                       "volume": np.full(n, 1e9)}, index=idx)

    mom = factors.compute_factor("momentum", df, {"period": 30})
    assert mom.index.equals(idx) and mom.iloc[:30].isna().all()
    assert (mom.dropna() > 0).all()   # monotonic uptrend → positive momentum

    rev = factors.compute_factor("reversal", df, {"period": 5})
    assert (rev.dropna() < 0).all()   # reversal = negated return

    with pytest.raises(ValueError):
        factors.validate_factor("nonexistent", {})
    with pytest.raises(ValueError):
        factors.validate_factor("momentum", {"period": 9999})

    # funding factor without the column → NaN series (name excluded from ranks)
    fa = factors.compute_factor("funding_avg", df, {"period": 21})
    assert fa.isna().all()
    assert factors.required_columns("funding_avg") == ["funding_rate"]

    # funding_avg is NEGATED: crowded-long positive funding must rank LOW
    df2 = df.copy(); df2["funding_rate"] = 0.002
    assert (factors.compute_factor("funding_avg", df2, {"period": 21}).dropna() < 0).all()


# ── WS2: cross-sectional engine (calibration-style, synthetic, offline) ─────

_XS_ENGINE_SNIPPET = """
import json
import numpy as np
import pandas as pd
from data.database import init_db, db_session
init_db()
from agents.backtest_engineer.backtest_engineer import BacktestEngineer
from config.settings import DEFAULT_SYMBOLS

# Synthetic universe: momentum factor GENUINELY predicts (planted edge).
# Each name's drift varies SLOWLY over time with a different phase, so
# trailing 30d return predicts the next bar both CROSS-SECTIONALLY (names in
# their high-drift phase outrank the rest) and WITHIN each name (the breadth
# gate counts per-name time-series ICs — a constant drift would leave those
# at noise level even with a huge cross-sectional IC).
n = 1200
idx = pd.date_range("2021-01-01", periods=n, freq="D")
rng = np.random.default_rng(11)
t = np.arange(n)
frames = {}
for i, sym in enumerate(DEFAULT_SYMBOLS):
    phase = 2 * np.pi * i / max(len(DEFAULT_SYMBOLS), 1)
    drift = 0.004 * np.sin(2 * np.pi * t / 300.0 + phase)   # slow regime waves
    close = 100 * np.exp(np.cumsum(drift + 0.010 * rng.standard_normal(n)))
    frames[sym] = pd.DataFrame({"open": close, "high": close * 1.001,
                                "low": close * 0.999, "close": close,
                                "volume": np.full(n, 1e9)}, index=idx)

eng = BacktestEngineer()
eng._fetch_prices = lambda s, interval="1d", days=1825: frames.get(
    s.split(",")[0].strip(), next(iter(frames.values()))).copy()
eng._fetch_funding_history = lambda s, days=1825: pd.DataFrame()

with db_session() as conn:
    cur = conn.execute(
        \"\"\"INSERT INTO alpha_ideas
             (slug, title, hypothesis, ticker, timeframe, factor_formula,
              stage, status, novelty_score, logic_score, feasibility_score)
           VALUES ('xs-test','xs test','planted momentum spread','UNIVERSE','1d',
                   ?, 'stage2','processing',0.8,0.8,0.8)\"\"\",
        ("xs:" + json.dumps({"signal_type": "cross_sectional",
                              "factor": {"name": "momentum", "params": {"period": 30}},
                              "top_n": 4, "bottom_n": 4, "rebalance_bars": 7,
                              "interval": "1d"}),))
    iid = cur.lastrowid
r = eng.backtest_idea(iid)
with db_session() as conn:
    run = conn.execute("SELECT run_type, passed FROM backtest_runs WHERE idea_id=?",
                       (iid,)).fetchone()
    idea = conn.execute("SELECT stage, status FROM alpha_ideas WHERE id=?",
                        (iid,)).fetchone()
print("RESULT " + json.dumps({
    "run_type": run["run_type"], "passed": bool(run["passed"]),
    "overall_pass": bool(r.get("overall_pass")),
    "ic": r.get("ic"), "parked": r.get("parked"),
    "stage": idea["stage"], "status": idea["status"],
    "n_rebalances": r.get("n_rebalances")}))
"""


@pytest.mark.parametrize("market_mode", ["crypto"])
def test_xs_engine_planted_edge_passes_and_parks(market_mode):
    """A PLANTED persistent cross-sectional spread must clear the gates via
    the xs: route, produce run_type='cross_sectional', and PARK at stage3
    (never stage4a — basket paper-trading doesn't exist)."""
    out = _run_mode(market_mode, _XS_ENGINE_SNIPPET)
    line = [l for l in out.splitlines() if l.startswith("RESULT ")][-1]
    res = json.loads(line[len("RESULT "):])
    assert res["run_type"] == "cross_sectional"
    assert res["overall_pass"] is True, res
    assert res["parked"] is True
    assert res["stage"] == "stage3" and res["status"] == "active", res
    assert res["ic"]["mean_ic"] > 0.05, res


def test_xs_backtest_persists_ic_columns_on_own_row():
    """Regression: run_cross_sectional_backtest calls cross_sectional_test
    BEFORE inserting its own backtest_runs row, so the old "UPDATE latest
    row" persistence silently matched nothing and left mean_ic/ic_tstat/
    stocks_positive_ic/best_stocks NULL on the basket's row (the numbers
    still made it into result_data JSON, so nothing was lost — just the
    queryable columns). Pin that the basket's OWN row now carries them."""
    out = _run_mode("crypto", _XS_ENGINE_SNIPPET.replace(
        'SELECT run_type, passed FROM backtest_runs WHERE idea_id=?',
        'SELECT run_type, passed, mean_ic, ic_tstat, stocks_positive_ic, '
        'best_stocks FROM backtest_runs WHERE idea_id=?',
    ).replace(
        '"n_rebalances": r.get("n_rebalances")}))',
        '"n_rebalances": r.get("n_rebalances"),\n'
        '    "row_mean_ic": run["mean_ic"], "row_ic_tstat": run["ic_tstat"],\n'
        '    "row_stocks_positive_ic": run["stocks_positive_ic"],\n'
        '    "row_best_stocks": run["best_stocks"]}))',
    ))
    line = [l for l in out.splitlines() if l.startswith("RESULT ")][-1]
    res = json.loads(line[len("RESULT "):])
    assert res["row_mean_ic"] is not None, res
    assert res["row_ic_tstat"] is not None, res
    assert res["row_stocks_positive_ic"] is not None, res
    assert res["row_best_stocks"] is not None, res
    assert abs(res["row_mean_ic"] - res["ic"]["mean_ic"]) < 1e-9, res


def test_xs_sandbox_submission_and_bursa_long_only():
    """sandbox xs briefs build the xs: spec; Bursa forces bottom_n=0 at the
    ENGINE level (long-only structurally) — pin the sandbox contract here."""
    out = _run_mode("crypto", """
import json
from data.database import init_db
init_db()
from pipeline.sandbox import submit_sandbox_idea, _signal_signature
r = submit_sandbox_idea({'title': 'xs t', 'hypothesis': 'h', 'ticker': 'UNIVERSE',
                         'timeframe': '1d', 'factor_formula': '',
                         'xs': {'factor': {'name': 'momentum', 'params': {'period': 30}},
                                'top_n': 4, 'bottom_n': 4, 'rebalance_bars': 7}})
bad = submit_sandbox_idea({'title': 'xs bad', 'hypothesis': 'h', 'ticker': 'UNIVERSE',
                           'timeframe': '1d', 'factor_formula': '',
                           'xs': {'factor': {'name': 'no_such_factor'}}})
from data.database import db_session
with db_session() as conn:
    row = conn.execute("SELECT ticker, factor_formula FROM alpha_ideas WHERE id=?",
                       (r["idea_id"],)).fetchone()
print("RESULT " + json.dumps({"ok": r["ok"], "bad_ok": bad["ok"],
                               "bad_err": bad.get("error", "")[:40],
                               "ticker": row["ticker"],
                               "ff_prefix": row["factor_formula"][:3]}))
""")
    line = [l for l in out.splitlines() if l.startswith("RESULT ")][-1]
    res = json.loads(line[len("RESULT "):])
    assert res["ok"] is True
    assert res["ticker"] == "UNIVERSE" and res["ff_prefix"] == "xs:"
    assert res["bad_ok"] is False and "factor invalid" in res["bad_err"]


def test_gateconfig_xs_fields_and_crypto_override():
    from config.settings import GateConfig
    cfg = GateConfig()
    # Defaults = the previously hardcoded North Star values (Bursa parity)
    assert cfg.xs_min_mean_ic == 0.05
    assert cfg.xs_min_ic_tstat == 1.5
    assert cfg.xs_min_positive_names == 15
    from config.markets import crypto
    assert crypto.GATE_OVERRIDES["xs_min_positive_names"] == 12
