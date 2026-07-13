# LeafSynthesizer REVIEW Gate — Decision Record (2026-07-13)

Outcome of an external Gen-AI consultation (`leaf_synthesizer_review_gate_question.md`),
evaluated against the actual codebase before adoption. Motivating failure: the P1-2 live
dry-run (2026-07-13 self-audit remediation) found the original REVIEW gate rejected a
CORRECT Chaikin Money Flow implementation from Haiku, twice, because it disagreed with
Opus's own hand-typed `worked_example.expected_output` — and Opus's arithmetic on a 7-bar
rolling-window sum was wrong, not Haiku's code (verified by hand both times).

The consultation named this the **test oracle problem**: the old gate was deterministic in
*mechanism* (it ran real code and asserted an exact match) but not in *ground truth* (the
value it asserted against was itself an LLM's mental arithmetic on a multi-step calculation
— precisely the kind of task LLMs are unreliable at).

## Adopted (implemented in this change)

1. **Differential testing as the real oracle.** REVIEW now generates a SECOND, independent
   implementation via a new Sonnet call (`_generate_reference_impl`) — given only the spec,
   never shown Haiku's code, explicitly prompted for a different style (plain loop, not
   vectorized) to reduce the odds of a correlated mistake. Both implementations run against
   ~20 randomized synthetic dataframes (`_generate_random_dataframes`, pure Python/numpy, no
   model call) plus PLAN's original worked_example, in one subprocess
   (`_differential_and_property_test`). Agreement across all random trials is the approval
   bar; the worked_example is just one more trial, not the sole authority.
2. **PLAN's worked_example demoted to documentation.** If Haiku's code disagrees ONLY with
   the worked_example while agreeing with the reference implementation on every random
   trial, that's logged as a PLAN arithmetic error (`review_notes`) and the leaf is still
   approved — this is the exact scenario that was previously a false rejection.
3. **Property/metamorphic checks** (self-consistency, no ground truth needed) run on
   Haiku's candidate: prefix-stability (`f(df[:k]) == f(df)[:k]`, mechanically catches
   lookahead), no-input-mutation, shape/index preservation, all-NaN-input safety.
4. **System-generated test file at landing** (`_build_test_file_source`) — the landed
   `tests/test_leaves_generated_{name}.py` now asserts against a value that was actually
   COMPUTED and cross-verified, never LLM-hand-typed. Haiku no longer writes `test_code` at
   all (`_implement`'s Stage 2 prompt simplified to `compute_code` only).
5. **Validated the core assumption empirically before building**, per the consultation's own
   caveat about correlated errors: ran a cheap, isolated probe asking Sonnet for a reference
   CMF implementation on the exact dataset already hand-verified wrong in PLAN's worked
   example — Sonnet's independent, loop-based implementation matched ground truth exactly
   (`[False, False, False, True, True, True, False]`). Confirms the design targets the
   observed failure, not just a plausible-sounding theory.
6. **Isolated temp-directory execution** — the differential-test subprocess now runs
   entirely in a `tempfile.mkdtemp()` working directory, never writing into the real
   `agents/backtest_engineer/leaves_generated/`/`tests/` package dirs until `_land()`
   explicitly commits an approved result. A partial failure can no longer leave stray files
   in the real repo (the old `_run_generated_test` wrote throwaway files directly into the
   real package dirs).

## Deferred (deliberately NOT built)

- **Named-indicator library oracle** (`ta`/`pandas-ta`) — would be a stronger, non-LLM
  ground truth for well-known indicators like CMF, but requires a new dependency plus a
  fragile "is this a recognizable named indicator" classification step. Revisit if
  differential testing's correlated-error risk turns out to bite in practice.
- **Formula-template compiler** for common patterns (rolling_sum_ratio, ema, zscore, ...) —
  genuinely cheaper than a second LLM call once built, but is its own small DSL/compiler;
  meaningful engineering for a mechanism that fires rarely (only on genuinely-unrepresentable
  idea rejections).
- **AST whitelist** upgrading the current banned-token string scan — real hardening, but the
  current scan already covers the practical attack surface (import os/sys/subprocess,
  eval/exec, file/network access); not the bug that was actually found this pass.
- **Quarantine status / PLAN-arithmetic-error-rate telemetry dashboard** — worth doing once
  there's enough synthesis volume for a rate to be meaningful; for now the mismatch is
  recorded in free-text `review_notes` on the audit row, not a dedicated column/metric.
- **Param-space fuzzing** — differential trials randomize the INPUT DATA only; `node_params`
  are fixed (from `worked_example.node_params`, or the declared range's lower bound as a
  fallback) across all trials. Randomizing params too would widen coverage further.

## Known residual risk (not solved, only mitigated)

Correlated errors: if Haiku and Sonnet misread the same ambiguous phrase in
`compute_description` the same way, differential testing gives false confidence — it would
approve wrong code with high certainty instead of the old design's no-certainty-either-way.
Mitigated by forcing a different implementation style and never showing one implementation's
code to the other's author; not eliminated. The empirical validation probe (item 5 above)
is one data point in favor, not proof this can't happen.
