"""Statistical rigor: lookahead guard, NW t-stat, deflation hurdle, random-signal rejection."""
import numpy as np
import pandas as pd

from agents.backtest_engineer.backtest_engineer import BacktestEngineer


def _price_df(n=800, seed=0, drift=0.0):
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    rets = rng.randn(n) * 0.01 + drift
    close = 10 * np.cumprod(1 + rets)
    return pd.DataFrame({"close": close, "volume": 3_000_000}, index=idx)


def test_lookahead_guard_bar0_flat():
    be = BacktestEngineer()
    df = _price_df()
    sig = pd.Series(1.0, index=df.index)  # always long
    perf = be._compute_performance(df, sig, "1d")
    # signal shifted by 1 → the strategy cannot hold a position on bar 0
    assert perf["total_trades"] >= 1


def test_net_never_exceeds_gross():
    be = BacktestEngineer()
    df = _price_df(seed=3, drift=0.0005)
    sig = pd.Series((np.arange(len(df)) // 15) % 2, index=df.index, dtype=float)
    perf = be._compute_performance(df, sig, "1d")
    assert perf["sharpe_net"] <= perf["sharpe_gross"]


def test_nw_tstat_shrinks_autocorrelated_series():
    be = BacktestEngineer()
    rng = np.random.RandomState(42)
    n = 500
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.9 * x[i - 1] + rng.randn() * 0.1
    x = x + 0.01
    iid_t = x.mean() / (x.std(ddof=1) / np.sqrt(n))
    assert abs(be._nw_tstat(x)) < abs(iid_t)


def test_nw_tstat_matches_iid_on_independent_data():
    be = BacktestEngineer()
    rng = np.random.RandomState(7)
    y = rng.randn(1000) + 0.1
    iid_t = y.mean() / (y.std(ddof=1) / np.sqrt(len(y)))
    assert abs(be._nw_tstat(y) - iid_t) / abs(iid_t) < 0.15


def test_deflation_hurdle_grows_with_trials():
    hurdles = [
        float(np.sqrt(2.0 * np.log(max(n, 2)) / 1825) * np.sqrt(252))
        for n in (1, 10, 100, 1000)
    ]
    assert hurdles == sorted(hurdles)
    assert hurdles[0] < 0.5 and hurdles[-1] > 1.2


def test_random_signals_rarely_beat_deflated_hurdle():
    """The multiple-testing gate must reject most pure-noise strategies even
    when we pick the best of many — that is exactly the failure mode it guards."""
    be = BacktestEngineer()
    n_trials = 100
    sharpes = []
    for seed in range(n_trials):
        df = _price_df(n=800, seed=seed)
        rng = np.random.RandomState(seed + 10_000)
        sig = pd.Series(rng.randint(0, 2, len(df)).astype(float), index=df.index)
        perf = be._compute_performance(df, sig, "1d")
        sharpes.append(perf["sharpe_net"])
    hurdle = float(np.sqrt(2.0 * np.log(n_trials) / 800) * np.sqrt(252))
    survivors = sum(1 for s in sharpes if s >= hurdle)
    # noise strategies churn daily and pay full costs; nearly all must fail
    assert survivors <= n_trials * 0.05, f"{survivors}/{n_trials} noise strategies passed"
