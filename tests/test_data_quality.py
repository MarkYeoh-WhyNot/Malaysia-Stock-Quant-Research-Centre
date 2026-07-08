"""Phase 1.2/1.4: Data Confidence Score + corporate-action anomaly detection."""
import numpy as np
import pandas as pd

from data.data_quality import (
    compute_data_confidence, detect_corporate_action_anomalies)


def _clean(n=400, start="2021-01-01"):
    idx = pd.bdate_range(start, periods=n)
    close = np.linspace(10.0, 14.0, n) + np.sin(np.arange(n) / 5) * 0.1
    return pd.DataFrame({"close": close, "volume": 3_000_000}, index=idx)


def test_clean_bluechip_scores_high():
    dq = compute_data_confidence(_clean())
    assert dq["confidence_score"] >= 90
    assert dq["notes"] == "clean"


def test_stale_prices_lower_score():
    clean_score = compute_data_confidence(_clean())["confidence_score"]
    df = _clean()
    # freeze the last 80 closes (failed/frozen feed)
    df.iloc[-80:, df.columns.get_loc("close")] = df["close"].iloc[-81]
    dq = compute_data_confidence(df)
    assert dq["stale_price_frac"] > 0.15
    assert dq["confidence_score"] < clean_score


def test_missing_days_lower_score():
    df = _clean()
    # drop ~30% of rows to create calendar gaps
    df = df.iloc[::10].append(df.iloc[1:200]) if hasattr(df, "append") else \
        pd.concat([df.iloc[::10], df.iloc[1:200]]).sort_index()
    df = df[~df.index.duplicated()]
    dq = compute_data_confidence(df)
    assert dq["missing_day_frac"] > 0.1


def test_empty_frame_zero_confidence():
    dq = compute_data_confidence(pd.DataFrame())
    assert dq["confidence_score"] == 0.0


def test_corporate_action_gap_detected():
    df = _clean(n=200)
    # simulate an unhandled 2-for-1 bonus: price halves overnight
    loc = 100
    df.iloc[loc:, df.columns.get_loc("close")] *= 0.5
    hits = detect_corporate_action_anomalies(df, gap_threshold=0.25)
    assert len(hits) == 1
    assert hits[0]["pct_change"] < -0.4


def test_no_false_positive_on_smooth_series():
    assert detect_corporate_action_anomalies(_clean(), 0.25) == []
