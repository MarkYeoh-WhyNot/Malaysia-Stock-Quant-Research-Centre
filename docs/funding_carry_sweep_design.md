# Funding-Carry Cross-Sectional Sweep — Pre-Registration

**Date registered: 2026-07-12 — BEFORE any sweep was executed.** This document
fixes the search space, selection protocol, and acceptance bar in advance, the
same discipline `scripts/alpha_hunt.py` used. If a future reader finds the grid
below differs from what was actually run, that is a protocol violation, not a
tweak.

Reviewed pre-execution by an independent model review (APPROVE-WITH-CHANGES);
the material change adopted: selection metric is **val basket net Sharpe**, not
val mean IC — IC only identifies the factor period, while Sharpe is affected by
every swept dimension, and it is the quantity the deflated-PSR hurdle then
corrects for. Crypto market only (`MARKET_MODE=crypto`).

## Motivation (evidence on record)

- Exhausted: daily/sub-daily OHLCV technicals on the 20-pair liquid majors —
  2,100-trial alpha hunt (0 survivors), 0/64 classic scan, 0/300 z-score sweep.
- XS momentum-30d: NEGATIVE mean IC (−0.015, NW t −2.26) — momentum reversed
  among majors in this window; momentum is a dead end, not a sweep candidate.
- XS funding carry, tested ONCE at registry defaults: mean IC 0.018,
  NW t ≈ 2.97 — the only statistically real signal in this system's history.
  Honestly rejected at that setting (IC < 0.05 bar, 9/20 positive names, L/S
  basket lost after costs). Never parameter-explored. This sweep is that
  exploration.

## Search space (fixed)

One factor family per sweep run (`funding_avg` or `funding_zscore` — whichever
the submitted idea's `xs:` spec names; a sweep never crosses families):

| Dimension        | Values                                        | Source |
|------------------|-----------------------------------------------|--------|
| factor `period`  | uniform int over the registry range (funding_avg 3–90, funding_zscore 10–200) | `factors.FACTORS[fname]["params"]` at draw time |
| `top_n`          | {2, 3, 4, 5, 6}                               | `optimizer.XS_TOP_N_CHOICES` |
| `bottom_n`       | {0, 2, 3, 4, 5, 6} (crypto; forced 0 without shorting) | `optimizer.XS_BOTTOM_N_CHOICES` |
| `rebalance_bars` | {3, 5, 7, 10, 14, 21, 28}                     | `optimizer.XS_REBALANCE_CHOICES` |
| `interval`       | FIXED at the idea's interval (not swept)      | deliberate scope cut |

**n_configs = 200 seeded random draws (seed 42)** over that space — not the
full grid; 200 is the honest trial count charged to deflation.

## Selection protocol (fixed)

1. The universe panel (closes + factor funding column + settlement-summed
   funding drag + per-name side cost rates) is fetched once and **truncated to
   the train+val window (first 80% of common bars, `GATE_CONFIG`
   stage3_data_split ratios) before any config is scored** — the test slice is
   untouched by construction.
2. Each config runs the actual L/S basket rebalance loop (identical semantics
   to `run_cross_sectional_backtest`: rank at bar close, weights effective
   next bar, equal-weight legs, turnover costs, real funding drag) on that
   truncated panel.
3. Eligibility: train net Sharpe > 0 AND ≥ 10 rebalance events inside the val
   window (`XS_MIN_VAL_REBALANCES` — mirrors the DSL sweep's MIN_VAL_TRADES).
4. Eligible configs ranked by **val net Sharpe, descending**; rank 1 is the
   winner. Val mean IC is recorded per config for the report — never used for
   selection.
5. The winner is promoted by rewriting the idea's `factor_formula` to the
   winning `xs:` spec and releasing it to stage2, where the standard
   `run_cross_sectional_backtest` gate stack evaluates it — **the sweep itself
   never touches the test slice, not even for the winner** (stricter than the
   DSL `run_sweep`, which one-shot-peeks; that asymmetry is intentional).

## Acceptance bar (fixed = existing gates, zero changes)

The winner passes iff the standard gated basket run passes, with every
threshold exactly as configured before this work:

- Deflated PSR ≥ `GATE_CONFIG.psr_confidence_test` (0.70) vs SR* from the
  90-day noise window, **with this sweep's full 200 configs added to n_trials**
  via `optimizer_runs.n_configs` (the same wiring the DSL sweep uses —
  verified present in `cross_sectional.py` and `gates.py`).
- IC gate: mean IC > 0.05, NW t > 1.5, > 12/20 positive names.
- All orthogonal guards: DD caps, noise-aware train/val gap, OOS walk-forward,
  regime terciles, EW-benchmark Sharpe, rebalance-count floor.

No gate threshold is modified by this campaign. A pass PARKS at stage3
(basket paper-trading doesn't exist yet — disclosed limitation).

## Engine fix shipped WITH this campaign (disclosed)

Pinning the sweep scorer's semantics against the gated engine exposed a real
fidelity bug in `run_cross_sectional_backtest`'s weight pipeline (present
since the basket engine landed 2026-07-10): `replace(0.0→NaN).ffill()` was
meant to carry weights BETWEEN rebalance rows but also forward-filled a
dropped name's stale weight over its legitimate 0 — **names could enter the
basket but never exit**, gross leverage crept up on every membership change,
and exit-side turnover was never costed. Fixed (Mark-approved 2026-07-12) in
both the gated engine and the sweep scorer: weights start NaN, rebalance rows
are literal, then ffill. Regression-pinned in
`tests/test_xs_sweep.py::test_basket_score_gross_leverage_never_exceeds_book`
and the switch-cost test.

Consequence for the historical record: the original funding-carry acceptance
run's "L/S basket lost money after costs" verdict was produced by the buggy
engine and is **not reliable evidence about the basket's economics** (the IC
numbers — 0.018, t≈2.97 — are unaffected: the IC path never used the weight
pipeline). This sweep is therefore the first honest measurement of the
factor's basket economics, not a re-measurement. No gate threshold changed;
this is the same fidelity-bug class as the 2026-07-11 PnL unification.

## Known caveats (disclosed up front)

- **Partial double-dip on IC:** the gated IC test runs on the full window,
  ~80% of which is the selection data. Deflated-PSR corrects the Sharpe
  selection channel (now exactly matched to the selection metric); the IC
  channel retains a modest selection bias. Mitigants: adjacent periods are
  highly correlated (effective independent trials ≪ 200), and the 0.05 IC bar
  sits several standard errors above zero on a ~1,400-date panel. To make any
  residual bias visible, the final report MUST include, report-only:
  (a) the winner's **test-slice-only** mean IC and NW t (data never touched by
  selection); (b) the sweep's val-Sharpe and val-IC distributions
  (max/median/min), so the winner's advantage over the field is on record.
- **Panel-floor mismatch:** the sweep excludes names with < 100 bars; the
  gated `cross_sectional_test` floor is 60 bars — panel membership can differ
  slightly between sweep-time and gate-time IC numbers.
- **Fresh idea required:** the daemon's stage2 queue skips ideas that already
  have any `backtest_runs` row, so the sweep must be submitted as a NEW
  sandbox idea (`optimize=True`), never by re-activating the original rejected
  funding-carry idea_id.
- **Survivorship:** the universe is the current 20 majors — same disclosed
  bias as every other basket run in this system.

## Explicitly out of scope (each would need its own pre-registration)

- Composite funding+reversal (or any multi-factor) spec — a NEW hypothesis
  with its own registry entry and its own trial charge, never folded into this
  sweep's 200.
- Sweeping `interval` / sub-daily rebalancing.
- Any change to any gate threshold.

## Dry-run record (2026-07-12, local, wiring verification — NOT the gated run)

One local execution of the registered sweep (seed 42, n=200, `funding_avg`,
real Binance data, isolated scratch DB; test slice untouched by construction):
200/200 evaluated, 124 eligible.

- **val mean IC distribution: min 0.0146 / median 0.0218 / max 0.0356** — every
  eligible config's IC is positive (consistent with the original t≈3 finding)
  but ALL are below the 0.05 gate bar. Honest expectation: the winner likely
  fails the gated IC check again; the sweep cannot manufacture breadth the
  factor doesn't have.
- **val net Sharpe distribution: min −1.38 / median 0.25 / max 1.89.** Winner:
  period 3, top 6 / bottom 6, rebalance 10 bars — val Sharpe 1.89, val IC
  0.032, but train Sharpe only 0.30 (temporal instability on display).
- The winner's single gated run (test-slice contact #1 and only) is to happen
  via the daemon path on the production DB — deliberately NOT executed
  locally, to keep exactly one registered look.

## Outcome

_To be filled in AFTER the winner's single gated run. Recording a rejection
here is a valid and expected outcome — the point is the answer, not a pass._
