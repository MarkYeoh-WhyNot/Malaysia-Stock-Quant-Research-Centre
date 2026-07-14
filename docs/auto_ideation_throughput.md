# Auto-Ideation Throughput — Arithmetic Record (2026-07-13)

P1-4 of the self-audit remediation plan: revisit_scan, finding_driven_candidates,
alpha_seeds, screener_ideas, and LeafSynthesizer all run concurrently, each
independently designed with its own cooldown/cap, but nobody had added up what
they produce or cost *together*, or what that volume does to the gate bar
downstream. This is that arithmetic, grounded in live data pulled from both
production DBs on 2026-07-13.

## What each mechanism actually produces

| Mechanism | Cadence | Cap per run | Worst case/day | LLM cost | Reaches backtest? |
|---|---|---|---|---|---|
| `revisit_scan` | every 6h (4x/day) | `MAX_REVISITS_PER_CYCLE=3` (pipeline/revisit.py) | 12 | none — deterministic, query-only | near-100%: inserted at `stage2/pending`, **bypasses gate0** |
| `finding_driven_candidates` | every 6h (4x/day) | `MAX_CANDIDATES_PER_CYCLE=4` (pipeline/finding_candidates.py) | 16 (mix of `auto-finding-*` + `rg-*`, same cap) | none — deterministic, query-only | near-100%: same as above, hardcoded `novelty=logic=feasibility=0.7` |
| `alpha_seeds` | hourly, `limit=5`/run | 5 | 120 (theoretical; gated by `kb_documents.seeded=0` availability) | yes (idea generation call) | goes through gate0 normally |
| `screener_ideas` | once daily | — | ~unbounded per scrape, historically small | yes | goes through gate0 normally |
| `LeafSynthesizer` | per unrepresentable rejection | — | budget-capped only | `LEAF_SYNTH_DAILY_BUDGET_USD=$10/day` (config/settings.py) | N/A — produces a DSL leaf, not an idea |

**revisit_scan + finding_driven_candidates combined worst case: 28 ideas/day**,
zero direct LLM cost, but see below for why they're not actually free.

## Why revisit/finding-driven are different from organic ideas

Both `run_revisit_scan()` and `run_finding_driven_candidates()` `INSERT INTO
alpha_ideas` directly with `stage='stage2', status='pending'` and hardcoded
`novelty_score=logic_score=feasibility_score=0.7` — **they skip Gate 0
entirely**. An organic idea has to clear Gate 0 (logic ≥ 0.65, feasibility ≥
0.70, data_quality ≥ 0.70, overfitting_risk ≤ 0.40) before it ever reaches a
backtest; most don't (see the funnel report's `generated → gate0_pass` ratio).
These two mechanisms have no such filter — nearly every one they submit reaches
a real backtest run.

## The downstream cost: recent_trial_count() → n_trials → SR\*

`agents/backtest_engineer/gates.py::recent_trial_count()` counts distinct
`backtest_runs.idea_id` in the trailing `GATE_CONFIG.deflation_window_days`
(90 days), excluding `calib-%` probes. That count feeds
`agents/backtest_engineer/stats.py::deflated_sr_star()`:

```
SR* = sqrt(2 · ln(n_trials) / n_obs) · sqrt(annualization)
```

`n_trials` is **global** — every idea in the system (organic, seeded, revisit,
finding-driven) is judged against the same SR\* once it's backtested. So a
higher rate of revisit/finding-driven submissions → more backtested ideas →
higher `n_trials` → a higher bar for *everyone*, not just the auto-submitted
ones. This is real, but the relationship is **logarithmic, not linear** —
doubling `n_trials` does not double SR\*. Concretely: going from 500 to 1000
trials multiplies SR\* by `sqrt(ln(1000)/ln(500)) ≈ 1.054` — about a 5%
increase in the hurdle. The mechanism is honest (more noise trials really does
justify a higher bar against noise), but the magnitude is dampened, not
explosive.

## Live numbers (2026-07-13, both production DBs)

Current `n_trials` (90-day window, excluding calib probes):
- **Bursa: 11**
- **Crypto: 19**

7-day `alpha_ideas` volume by source (slug-prefix classified):
- **Bursa** (2026-07-08 to 07-13): organic dominates most days (32–85/day);
  `revisit`=6 and `seed`=2 appeared on 07-12; `calib`=36/day on harness-run
  days (excluded from `n_trials` by design).
- **Crypto** (same window): 2026-07-12 had `revisit`=8, `auto-finding`=2,
  `rg`=2, `seed`=7, `organic`=3 — i.e. **12 of that day's 22 real ideas were
  revisit/finding-driven** (before this fix, uncapped).

`leaf_synthesis_attempts` (crypto, this session's P1-2/P2-3 dry-runs):
8 attempts on 07-13 totaling $0.44, 4 attempts on 07-12 totaling $0.14 — well
inside the $10/day `LEAF_SYNTH_DAILY_BUDGET_USD` cap on its own.

Daily `ai_usage` spend, both markets: roughly $0.4–$2.6/day over the same
week — nowhere near `AI_DAILY_BUDGET_USD=$50`, so cost was never the binding
constraint here. **Volume and its effect on `n_trials` was the unpriced risk**,
not money — which is exactly why this needed a count-based cap
(`AUTO_IDEAS_DAILY_CAP`), not another budget cap.

## The fix

`pipeline/throughput_guard.py::auto_ideation_cap_reached()` — counts today's
`alpha_ideas` with slug prefix `revisit-%`, `auto-finding-%`, or `rg-%` and
compares against `AUTO_IDEAS_DAILY_CAP` (default 20, `config/settings.py`).
When reached, `_process_revisit_scan` and `_process_finding_driven_candidates`
in `scripts/research_daemon.py` skip and log rather than submit — the two
mechanisms combined can produce up to 28/day, so a cap of 20 is a real
constraint (not a formality) while still leaving room for a normal day's
worth of both. `alpha_seeds`/`screener_ideas`/organic generation are
deliberately NOT covered by this cap — they already go through Gate 0 and
have their own, much lower, per-run limits.

`scripts/research_daemon.py::_process_funnel_report`'s daily report now
includes an "Auto-mechanisms 24h" line — revisit/finding-driven counts,
today's quota usage against the cap, and LeafSynthesizer attempt/approval/cost
counts — so this volume is visible on Telegram every day instead of silently
accumulating.

## contradicting_finding trigger: code-complete but never actually fired (2026-07-13 follow-up)

P2-6 tightened `pipeline/revisit.py::detect_triggers()`'s `contradicting_finding`
branch to only accept `contradicts` edges from a genuine `finding-campaign-*`
node with `origin='heuristic'` — i.e. edges written by
`knowledge/ingestion/campaign_findings.py::record_campaign_finding()`'s own
`contradicts_slugs` parameter, never the LLM graph extractor. The follow-up
audit checked whether this branch can ever actually fire in production:

- **Every call site was grepped.** There are exactly two:
  `scripts/alpha_hunt.py`'s `emit_alpha_hunt_findings()` (runs after every
  alpha-hunt campaign) and the one-off `scripts/backfill_campaign_findings.py`.
  **Neither ever passes `contradicts_slugs`** — the emitter only ever wires
  `leaf_names`, and the backfill script only ever wires `refines_slugs`.
  `contradicts_slugs` was also uncovered by any test until this audit.
- **The consumer side works.** `tests/test_revisit.py` already proved
  `detect_triggers()` correctly picks up a heuristic-origin `finding-campaign-*`
  contradicts edge and correctly ignores LLM-origin / non-campaign / wrong-type
  sources — but those tests wired the edge directly via `store.add_edge()`,
  not through the real production entry point.
- **The producer side is now proven too.** `test_contradicts_slugs_wires_a_trigger_ready_edge`
  in `tests/test_campaign_findings.py` calls `record_campaign_finding()` with
  `contradicts_slugs` set and confirms `detect_triggers()` picks up the
  resulting edge end to end. The wiring is correct.

**Net status: this trigger is fully functional but currently dormant by
construction, not by bug.** Nothing in the live pipeline ever produces the
kind of edge it's looking for. It will only fire if either (a) the alpha-hunt
emitter is extended to detect when a new campaign verdict contradicts an
existing `rejection_pattern` and pass its slug via `contradicts_slugs`, or
(b) a human running a campaign manually calls `record_campaign_finding()`
with `contradicts_slugs` set (the way `backfill_campaign_findings.py` already
does for `refines_slugs`). Until one of those happens, "zero live
`contradicting_finding` triggers" is the expected, honest state — not a sign
the P2-6 fix needs more work.

## AUTO_IDEAS_DAILY_CAP: confirmed to fire cleanly on a real skip (2026-07-13 follow-up)

Before this check, the cap had never actually fired in production — both
`auto_submissions_today()` sat at 0/20 in both markets (nowhere near the cap
naturally), and grepping all retained daemon container logs for
`AUTO_IDEAS_DAILY_CAP` in both markets returned zero hits. The mechanism was
tested but never observed live.

Verified live on `daemon-crypto` with Mark's explicit go-ahead: ran a one-off
script inside the running container that (1) snapshotted the `job_state` rows
for `revisit_scan`/`finding_driven_candidates`, (2) instantiated
`ResearchDaemon` and called `_process_revisit_scan()` and
`_process_finding_driven_candidates()` directly with `AUTO_IDEAS_DAILY_CAP=0`
env-overridden for that single process only, (3) captured the logger output,
(4) restored `job_state` to its exact pre-test values. Both jobs logged their
skip message cleanly with no exception:
```
[Revisit] Skipped — AUTO_IDEAS_DAILY_CAP reached for today
[FindingCandidates] Skipped — AUTO_IDEAS_DAILY_CAP reached for today
```
`job_state` was confirmed byte-identical to its pre-test snapshot afterward
(`revisit_scan` / `finding_driven_candidates` timestamps unchanged) — no
lasting effect on the live daemon's schedule. **The guard is proven correct
end to end; it just hasn't had a reason to fire yet at real traffic levels.**
