# Board Architecture — Work Orders

Dispatch-ready task breakdown for the bottom-up board architecture
(plan: `~/.claude/plans/this-is-the-full-wondrous-bird.md`). Each **task card below
is a standalone prompt**: prepend the [Shared Preamble](#shared-preamble) and hand the
card to a Haiku or Sonnet agent (or a separate Claude Code session). Independent tasks
in the same parallel group can run concurrently in different agents; cards inside one
sequential track must run in order, ideally in one session.

Branch for all of this: `restructure/8-department-architecture` (already has the
uncommitted `DIRECTION_DOC` rewrite + alert→`daemon_logs` change).

---

## Shared Preamble

> Copy this block to the top of every task prompt.

```
Repo: /Users/markyeoh/Documents/GitHub/Malaysia-Stock-Quant-Research-Centre
Branch: restructure/8-department-architecture (do not create new branches; do not commit unless told)
Language: Python 3.12. Activate the venv for any command: `source venv/bin/activate`.
Run tests with: `python -m pytest -q`. Run a single file: `python -m pytest tests/<file> -q`.

Dual-market rule: this codebase runs two market profiles (bursa | crypto) selected by
MARKET_MODE. config/settings.py loads ONE profile and re-exports both legacy Bursa names
and generic aliases. NEVER hardcode a `.KL` ticker check or a market assumption — read the
active profile constants. Bursa behavior must stay byte-identical (regression-pinned).

SACRED — human-only, never change without an explicit human-approval task:
- gate thresholds in config/settings.py GATE_CONFIG / GATE_OVERRIDES
- the parser honesty contract (a strategy that can't be represented EXACTLY must return
  {"representable": false, "reason": ...} — silent approximation is forbidden)
- adding new executable DSL leaves to signal_dsl.py
- anything touching live capital or loosening a risk limit

Every task ends by: (1) running the relevant pytest and pasting the result, (2) stating
plainly what you changed and did NOT change. If a check would require editing sacred code,
STOP and report instead of proceeding.
```

---

## Dependency & Parallelization Map

```text
T0 (foundation) ─────────────┐
                             │ (governance_findings table + Inspector base + schemas)
                             ▼
   ┌──────────────┬──────────┼───────────────┬───────────────┐
   │              │          │               │               │
TRACK A        TRACK D     TRACK C         TRACK E        (T0 also
monolith split risk insp.  parser honesty  data phase 1    unblocks B)
A1→A2→…→A7     D1 D2 D3 D4  C1 C2 C3        E1 E2 E4
(sequential,   (parallel,   (parallel,      (parallel,
 one session)   need T0)     need T0)        E1/E2 need nothing;
   │                                          E4 needs T0)
   ▼
TRACK B  fidelity inspectors  B1 B2 B3 B4 B5 B6
(parallel among themselves; need T0 AND Track A done)
   │
   ▼
TRACK F  managers + Command Centre  F1 → F2   (needs findings from B/C/D)
   │
   ▼
TRACK G  L2/L3 board scaffolding  G1 G2   (last; dormant until a promotion candidate exists)

E3 (new DSL leaves) = HUMAN-GATED proposal, NOT an agent task.
```

**What can run at the same time, right now:** `T0`, all of `Track A`, `E1`, `E2` — four
independent workstreams. `Track C` and `Track D` start the moment `T0` lands. `Track B`
waits for `Track A`. `Track F` waits for B/C/D. `Track G` is last.

**Model mix at a glance:** Sonnet for the refactor (Track A), new-logic and cross-domain
cards; Haiku for the narrow deterministic inspectors and pattern-following assembly.

---

> **Philosophy Reconciliation** has been moved to its own file:
> `docs/philosophy_reconciliation_work_order.md`. It is independent of everything here and
> can run in parallel.

## T0 — Governance Foundation (BLOCKING)

- **Model:** Sonnet · **Depends on:** none · **Parallel with:** Track A, E1, E2
- **Read first:** `data/database.py` (init_db table pattern, `db_session`),
  `agents/base_agent.py`, `agents/risk_monitor/risk_monitor.py` (an existing check to mirror style).
- **Do:**
  1. Add a `governance_findings` table in `data/database.py` `init_db()` (follow the exact
     `CREATE TABLE IF NOT EXISTS` + index style already there). Columns: `id`, `agent` TEXT,
     `level` TEXT ('L0'/'L1'/'L2'/'L3'), `scope` TEXT (e.g. `backtest_run:1234`), `status`
     TEXT ('PASS'/'WARN'/'FAIL'), `severity` TEXT ('INFO'/'WARNING'/'BLOCKER'), `evidence`
     TEXT (JSON array), `local_recommendation` TEXT, `escalate_to` TEXT, `created_at` TEXT
     DEFAULT (datetime('now')). Add `idx_gov_scope` and `idx_gov_status`.
  2. Create `governance/__init__.py` and `governance/base.py` with a `class Inspector(ABC)`:
     `name: str`, `level = "L0"`, abstract `inspect(self, scope, ctx) -> Finding`, and a
     `record(finding)` helper that writes one `governance_findings` row via `db_session`.
  3. Define the four packet dataclasses/Pydantic models in `governance/schemas.py`:
     `Finding`, `DepartmentSummary`, `ExecutiveDecisionPacket`, `HumanApprovalRequest`
     (fields exactly as in plan §5). Use `pydantic` (already a dependency).
- **Acceptance:** `python -m pytest -q` green; add `tests/test_governance_foundation.py`
  proving the table is created by `init_db()` on a temp DB and that `Inspector.record()`
  writes and reads back one finding. Paste output.
- **Do NOT:** wire anything into the daemon yet; that is F1. Do not add LLM calls — this
  layer is pure code.

---

## TRACK A — Split `backtest_engineer.py` (SEQUENTIAL, one Sonnet session)

> The whole track is one 4010-line file. Run A1→A7 in order in a SINGLE session; after
> EACH card, run the full acceptance gate. Do not parallelize across agents — they would
> collide on the same file. This does not depend on T0.

**Track-wide acceptance gate (run after every card A1–A7):**
```
source venv/bin/activate
python -m pytest -q                                   # must stay fully green
MARKET_MODE=bursa  PYTHONPATH=. ./venv/bin/python scripts/calibration_harness.py
MARKET_MODE=crypto PYTHONPATH=. ./venv/bin/python scripts/calibration_harness.py
```
Pass-rate tiers must match the pre-split baseline (noise 0%, strong ≥90%, moderate ≥60%).
Capture the baseline BEFORE A1. Behaviour-preserving means: no numeric constant moves, no
logic changes — only code is relocated and imported back. If a tier shifts, revert the last
card and report.

- **A1 — Extend `stats.py`.** Move the pure stat helpers out of `backtest_engineer.py`
  into the existing `agents/backtest_engineer/stats.py`: `_spearman`, `_nw_tstat`,
  `_sharpe_stderr`, `_train_val_gap_tolerance`, `_robustness_check`. Re-import them in
  `backtest_engineer.py`. (Model: Sonnet)
- **A2 — Create `engine.py`.** Move the return/backtest math: `_net_return_series`,
  `_compute_signals`, `_rsi`, `_reconstruct_trades`, `_apply_exit_logic`,
  `_get_exit_profile_by_key`, `_compute_performance`, `_compute_walk_forward`,
  `_compute_regimes`, `_detect_sanity_flags`, plus module helper `_funding_bar_sum`.
  This is the highest-value extraction — `engine._net_return_series` becomes the single
  PnL source the Track B inspectors target. (Model: Sonnet)
- **A3 — Create `signal_parsing.py`.** Move the two LLM methods `_parse_factor` and
  `verify_formula` (the ONLY Claude calls in the file). Keep the parser honesty contract
  exactly. (Model: Sonnet)
- **A4 — Create `cross_sectional.py`.** Move `cross_sectional_test` and
  `_run_cross_sectional_backtest` (~700 lines, the largest cluster). (Model: Sonnet)
- **A5 — Create `gates.py`.** Move the gate-enforcement half of `_run_backtest` (the PSR
  principal rule + orthogonal guards: DD caps, gap tolerance, OOS, regime, robustness,
  cost drag, trade floors, IC, benchmark, liquidity, capacity). Keep it calling into
  `stats.py`. Do NOT change any threshold. (Model: Sonnet)
- **A6 — Create `fundamental_screen.py`.** Move `_run_fundamental_screen_backtest`
  (~530 lines, Bursa path). (Model: Sonnet)
- **A7 — Slim `backtest_engineer.py`.** What remains is the orchestrator: `backtest_idea`,
  `run`, `run_backtest`, `check_data_requirements`, `classify_holding_period`, and the
  wiring that composes engine/stats/cross_sectional/gates/parsing. Confirm the public API
  (`backtest_idea`, `run`) is unchanged so the daemon needs no edits. (Model: Sonnet)

---

## TRACK B — Backtest Fidelity Inspectors (PARALLEL; need T0 + Track A)

> Each card is independent — dispatch to separate agents. Each inspector lives in
> `governance/inspectors/` and subclasses `Inspector` from T0. Each ships a pytest that
> **plants a known-bad case and asserts a BLOCKER finding** (self-calibrating, like the
> gate harness). Read `agents/backtest_engineer/engine.py` (post-split) first.

- **B1 — PnL Consistency Inspector.** (Sonnet) Assert every consumer of per-bar net
  returns routes through `engine._net_return_series` — the persisted equity curve, regime
  attribution, and gated metrics must produce the SAME array for a given run. Planted bad
  case: a run where the equity curve is recomputed with different costs → BLOCKER. Fixes
  issues #1, #3. (Plan Case A.)
- **B2 — Funding Cost Auditor.** (Sonnet) Crypto only: assert every net-return path
  includes the funding term — compute path-with vs path-without funding on a crypto run
  with funding≠0; they MUST differ. Planted bad case: a path omitting funding → BLOCKER.
  Fixes issue #2. (Plan Case B.)
- **B3 — Fill Convention Auditor.** (Haiku) Assert `_reconstruct_trades` splits transition
  costs consistently (summed trade net reconciles to backtest return, already pinned in
  `tests/test_pnl_unification.py` — extend that invariant into an inspector). Fixes #4.
- **B4 — Cost Model Auditor.** (Haiku) Assert the cost rate applied matches the run's
  interval and market profile (no daily-cost default on a sub-daily run). Fixes #5.
- **B5 — Metric Consistency Auditor.** (Haiku) Assert arithmetic annual return and the
  compounded equity CAGR are both present and internally consistent (compounded curve
  endpoint matches reported CAGR). Fixes #3.
- **B6 — Regime Attribution Auditor.** (Sonnet) Assert `_compute_regimes` uses the same
  net-return series as the gate (incl. funding on crypto). Planted bad case: regime split
  on gross returns → BLOCKER. Fixes #2.

---

## TRACK C — Parser Honesty Inspectors (PARALLEL; need T0)

> Low coupling to the split — these target `agents/backtest_engineer/signal_dsl.py`
> (`LEAVES`, `shape_cards`, `PARSER_NEGATIVE_EXAMPLE`) which already exist. Read that file
> and `tests/test_arsenal_v2.py` first. All deterministic, all Haiku.

- **C1 — DSL Representability Checker.** (Haiku) Assert that when a strategy references a
  leaf/shape not in the registry, the parse result is `{"representable": false, ...}` and
  never a substituted tree. Drive it with fixture strings, no live LLM.
- **C2 — Leaf Semantics Auditor.** (Haiku) Assert every entry in `LEAVES` has a
  `shape_card` and (for `required_choices` leaves like `ma_level`) that a missing required
  choice is rejected, not defaulted.
- **C3 — Negative-Mapping Guard.** (Haiku) The canonical EMA-50 case: assert
  "long BTC/USDT when close > 50-day EMA, short below" does NOT compile to
  `ema_cross(fast=2, slow=50)`. This is the flagship parser-honesty regression test. If
  the correct representation needs a leaf that doesn't exist, the guard asserts a
  `representable: false` result — it must NOT propose adding the leaf (that's human-gated).

---

## TRACK D — Risk Inspectors (PARALLEL; need T0 only — independent of Track A)

> Touch `agents/risk_monitor/risk_monitor.py`, `paper_trades`, and the shadow-portfolio
> logic — NOT the backtester. Can run fully concurrently with Track A. Read
> `agents/risk_monitor/risk_monitor.py` first.

- **D1 — Shadow-NAV Inspector.** (Haiku) Assert any "total paper NAV" figure is the
  shared-book figure, never `SUM(sandbox NAV)`. Planted bad case: summed sandboxes →
  BLOCKER. Surfaces `paper_capital_multiplier`. Fixes #6. (Plan Case F.)
- **D2 — Concentration/Correlation Inspector.** (Sonnet — genuinely new logic) Flag
  same-symbol overlap across open sandbox strategies AND compute pairwise return
  correlation of active paper strategies (correlation logic does not exist yet — issue #9).
  Emit WARNING on overlap, escalate on a configurable threshold (threshold is a NEW config
  constant, not a GATE_CONFIG change). Fixes #7, #9. (Plan Case D.)
- **D3 — Capacity Aggregation Inspector.** (Sonnet) Recompute a strategy's PnL under
  SHARED capacity participation across all active strategies (current capacity is per-idea
  — issue #8). If capacity-adjusted Sharpe < a report threshold, FAIL. Fixes #8.
  (Plan Case E.)
- **D4 — Kill-Switch Inspector.** (Haiku) Wrap the existing
  `risk_monitor.check_kill_switches` outcome as a per-cycle Finding so kill events land in
  `governance_findings` (mostly surfacing existing logic).

---

## TRACK E — Data Roadmap Phase 1 (PARALLEL; free feeds only)

> E1 and E2 depend on nothing — dispatch immediately alongside Track A. E4 needs T0.
> All clients get OFFLINE-FIXTURE tests (no live network in CI). Read `data/market_data.py`
> and `data/binance/client.py` for the client style; `data/database.py` for tables.

- **E1 — DefiLlama client.** (Sonnet for schema, Haiku can do the fetch half) New
  `data/defillama/client.py` (free API, no key) for TVL / fees / revenue per protocol.
  New `protocol_metrics` table. Fixture-based tests. This is the highest-value free feed —
  it enables the Tokenomics Trap Screener and Protocol Revenue Quality.
- **E2 — CoinGecko client.** (Haiku) New `data/coingecko/client.py` (free tier,
  rate-limited) for circulating/total supply, FDV, market cap. New `token_supply` table.
  Fixture-based tests.
- **E3 — [HUMAN-GATED — NOT AN AGENT TASK].** Adding the new DSL leaves that consume this
  data (`fdv_ratio`, `revenue_momentum`, `tvl_change`) is on the sacred list. Prepare a
  written proposal for Mark (leaf name, exact math, shape_card, Pine mapping) and STOP.
- **E4 — Source Health Inspector.** (Haiku · needs T0) For each data source, assert a dead
  or rate-limited feed DEGRADES the data-confidence score rather than crashing (mirror the
  dead-Brave-key handling). Emits a daily Finding per source.

---

## TRACK F — Managers + Command Centre (SEQUENTIAL; needs B/C/D findings)

- **F1 — L1 Department Managers.** (Haiku) In `governance/managers.py`, one deterministic
  rollup per department (Backtest Fidelity, Parser Honesty, Data Integrity, Portfolio Risk,
  Paper Trading): read that department's `governance_findings`, emit a `DepartmentSummary`
  (GREEN/AMBER/RED = worst child severity). Register a `_process_fidelity_audit` step in
  the `scripts/research_daemon.py` `steps` tuple (follow the existing `_process_*` pattern)
  that runs the inspectors and writes findings each cycle.
- **F2 — Command Centre dashboard.** (Sonnet) This ABSORBS the paused 8-department
  restructure. In `dashboard/api/server.py` `/api/departments/overview` (line ~1741),
  replace/extend the 7 hardcoded department cards so each shows its manager's GREEN/AMBER/RED
  rollup + top blocker from `governance_findings`. Update the frontend grid in
  `dashboard/ui/index.html` (`refreshDeptCards`, ~line 4541) for the new department set.
  Keep the `_REAL_IDEA_FILTER` hygiene already in that endpoint. Verify per the preview/curl
  workflow (note: `preview_start` reuses the `dashboard` server name — verify crypto mode
  via curl on the VPS as the memory notes describe).

---

## TRACK G — Board Scaffolding (LAST; dormant until a promotion candidate)

- **G1 — Research Integrity Director.** (Sonnet · fires on promotion only) In
  `governance/directors.py`, consume the 5 `DepartmentSummary` packets for a candidate idea
  and emit one `trust_verdict` (TRUSTWORTHY / NOT_TRUSTWORTHY) with tradeoffs. One bounded
  Sonnet call, gated on an idea actually reaching a promotion decision.
- **G2 — Executive Board packet assembly.** (Sonnet · on promotion/breach only) Assemble
  the `ExecutiveDecisionPacket` and, when `requires_human`, the `HumanApprovalRequest`
  (`default_if_no_response = DENY`). No auto-approval of anything on the sacred list.

---

## Execution Guide

1. **Kick off four things at once now:** `T0`, `Track A` (one session), `E1`, `E2`.
   (Philosophy Reconciliation runs independently from its own file.)
2. **When `T0` lands:** start `Track C` (3 Haiku agents) and `Track D` (2 Haiku + 2 Sonnet),
   plus `E4`. These run alongside Track A.
3. **When `Track A` lands:** start `Track B` (up to 6 agents; 3 Haiku + 3 Sonnet).
4. **When B/C/D have landed:** `F1` then `F2`.
5. **`Track G` last**, and only becomes live when the first strategy is a promotion
   candidate — until then it's dormant tested code.
6. **Never batch a sacred-list change into any card.** `E3` and any new-leaf/threshold need
   a separate human-approval step.

### Model assignment summary
| Track | Cards | Model |
|---|---|---|
| T0 foundation | T0 | Sonnet |
| A monolith split | A1–A7 | Sonnet (one session) |
| B fidelity | B1,B2,B6 | Sonnet · B3,B4,B5 | Haiku |
| C parser honesty | C1,C2,C3 | Haiku |
| D risk | D2,D3 | Sonnet · D1,D4 | Haiku |
| E data | E1 | Sonnet · E2,E4 | Haiku · E3 | HUMAN |
| F managers/UI | F1 | Haiku · F2 | Sonnet |
| G board | G1,G2 | Sonnet |

### Definition of done (whole effort)
Full `pytest` green; calibration harness tiers unchanged in both markets after the split;
every inspector has a planted-bad-case test that produces a BLOCKER; the daemon writes
`governance_findings`; the Command Centre shows live department rollups; two free data
feeds land with fixture tests; nothing on the sacred list changed without a human-approval
task.
```
