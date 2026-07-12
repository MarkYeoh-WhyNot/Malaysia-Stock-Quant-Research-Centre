"""Tests for CapacityAggregationInspector — shared-liquidity capacity check
across concurrently active strategies (D3, governance work order).

Liquidity is a SHARED resource: the per-idea capacity gate (gates.py) and the
per-idea capacity_adjusted_sharpe (backtest_engineer.py) each assume a
strategy has an instrument's whole ADV to itself. This inspector recomputes
Sharpe as if several currently-active strategies compete for the same ADV
pool simultaneously, and flags (report-only, WARNING) meaningful degradation.
"""

from governance.inspectors.capacity_aggregation import (
    CapacityAggregationInspector,
    CAPACITY_SHARED_SHARPE_FLOOR,
)


def test_capacity_aggregation_no_strategies_returns_none():
    """No strategies in ctx: nothing to check."""
    inspector = CapacityAggregationInspector()
    finding = inspector.inspect(scope="capacity_aggregation:test", ctx={"strategies": []})
    assert finding is None


def test_capacity_aggregation_missing_adv_returns_none():
    """Strategies present but no ADV data for their instrument: nothing to check."""
    inspector = CapacityAggregationInspector()
    finding = inspector.inspect(
        scope="capacity_aggregation:test",
        ctx={
            "strategies": [
                {"strategy_id": 1, "instrument": "AAA", "notional_myr": 100_000,
                 "solo_sharpe": 1.2, "ann_vol": 0.25, "trades_per_year": 12},
            ],
            "adv_by_instrument": {},
        },
    )
    assert finding is None


def test_capacity_aggregation_bad_case_crowded_thin_liquidity_warns():
    """BAD case: 3 strategies compete for the same thin-liquidity instrument.

    Each individually looks fine in isolation (solo_sharpe=1.2), but the
    combined participation on a thin ADV pool means the REAL market-impact
    haircut each one would face is much larger than its own solo estimate —
    shared-capacity Sharpe should collapse well below the solo Sharpe and
    below CAPACITY_SHARED_SHARPE_FLOOR.
    """
    inspector = CapacityAggregationInspector()

    crowded_instrument = "THINCO"
    strategies = [
        {"strategy_id": f"idea_{i}", "instrument": crowded_instrument,
         "notional_myr": 300_000, "solo_sharpe": 1.2,
         "ann_vol": 0.25, "trades_per_year": 12}
        for i in range(3)
    ]

    finding = inspector.inspect(
        scope="capacity_aggregation:2026-07-12",
        ctx={
            "strategies": strategies,
            "adv_by_instrument": {crowded_instrument: 1_000_000},  # thin ADV
        },
    )

    assert finding is not None
    assert finding.status == "FAIL"
    # Report-only: must be WARNING, never BLOCKER.
    assert finding.severity == "WARNING"
    assert finding.escalate_to == "PortfolioExecutor"

    results = finding.evidence["results"]
    assert len(results) == 3
    for r in results:
        assert r["instrument"] == crowded_instrument
        assert r["participants"] == 3
        # Shared-capacity Sharpe must be materially worse than the solo estimate.
        assert r["shared_capacity_sharpe"] < r["solo_sharpe"]
        assert r["crowding_degradation"] > 0.5
        assert r["shared_capacity_sharpe"] < CAPACITY_SHARED_SHARPE_FLOOR
        assert r["below_floor"] is True


def test_capacity_aggregation_good_case_different_instruments_passes():
    """GOOD case: same notional/solo_sharpe profile as the bad case, but each
    strategy trades a DIFFERENT instrument — no shared liquidity, so no
    crowding effect and no degradation.
    """
    inspector = CapacityAggregationInspector()

    strategies = [
        {"strategy_id": f"idea_{i}", "instrument": f"SOLO_{i}",
         "notional_myr": 300_000, "solo_sharpe": 1.2,
         "ann_vol": 0.25, "trades_per_year": 12}
        for i in range(3)
    ]
    adv_by_instrument = {f"SOLO_{i}": 1_000_000 for i in range(3)}

    finding = inspector.inspect(
        scope="capacity_aggregation:2026-07-12",
        ctx={"strategies": strategies, "adv_by_instrument": adv_by_instrument},
    )

    assert finding is not None
    assert finding.status == "PASS"
    assert finding.severity == "INFO"

    results = finding.evidence["results"]
    assert len(results) == 3
    for r in results:
        assert r["participants"] == 1
        # No other participant on the same instrument -> no crowding at all.
        assert r["crowding_degradation"] == 0.0
        assert r["shared_capacity_sharpe"] == r["solo_sharpe"]
        assert r["below_floor"] is False


def test_capacity_aggregation_good_case_ample_liquidity_passes():
    """GOOD case: several strategies DO share one instrument, but ADV is huge
    relative to their notional — participation is tiny, so the crowding
    haircut is negligible and shared-capacity Sharpe stays above the floor.
    """
    inspector = CapacityAggregationInspector()

    instrument = "DEEPCO"
    strategies = [
        {"strategy_id": f"idea_{i}", "instrument": instrument,
         "notional_myr": 300_000, "solo_sharpe": 1.2,
         "ann_vol": 0.25, "trades_per_year": 12}
        for i in range(3)
    ]

    finding = inspector.inspect(
        scope="capacity_aggregation:2026-07-12",
        ctx={
            "strategies": strategies,
            "adv_by_instrument": {instrument: 500_000_000},  # very deep liquidity
        },
    )

    assert finding is not None
    assert finding.status == "PASS"
    assert finding.severity == "INFO"

    results = finding.evidence["results"]
    assert len(results) == 3
    for r in results:
        assert r["participants"] == 3
        assert r["shared_capacity_sharpe"] >= CAPACITY_SHARED_SHARPE_FLOOR
        assert r["below_floor"] is False
        # Some tiny crowding effect is fine, just not enough to matter.
        assert r["crowding_degradation"] < 0.05


def test_capacity_aggregation_skips_strategies_missing_required_fields():
    """Strategies without enough data to recompute (e.g. no ann_vol) are
    skipped rather than crashing the whole inspection.
    """
    inspector = CapacityAggregationInspector()

    instrument = "PARTIALCO"
    strategies = [
        {"strategy_id": "idea_ok", "instrument": instrument,
         "notional_myr": 100_000, "solo_sharpe": 1.0,
         "ann_vol": 0.2, "trades_per_year": 10},
        {"strategy_id": "idea_missing_vol", "instrument": instrument,
         "notional_myr": 100_000, "solo_sharpe": 1.0,
         "ann_vol": None, "trades_per_year": 10},
    ]

    finding = inspector.inspect(
        scope="capacity_aggregation:test",
        ctx={"strategies": strategies, "adv_by_instrument": {instrument: 10_000_000}},
    )

    assert finding is not None
    results = finding.evidence["results"]
    assert len(results) == 1
    assert results[0]["strategy_id"] == "idea_ok"


def test_capacity_aggregation_finding_recorded():
    """Findings from this inspector persist via the shared record() path."""
    inspector = CapacityAggregationInspector()

    instrument = "RECCO"
    strategies = [
        {"strategy_id": f"idea_{i}", "instrument": instrument,
         "notional_myr": 300_000, "solo_sharpe": 1.2,
         "ann_vol": 0.25, "trades_per_year": 12}
        for i in range(3)
    ]

    finding = inspector.inspect(
        scope="capacity_aggregation:record_test",
        ctx={"strategies": strategies, "adv_by_instrument": {instrument: 1_000_000}},
    )
    row_id = inspector.record(finding)
    assert isinstance(row_id, int)
    assert row_id > 0
