# Question: how to fix an LLM-code-review gate whose ground truth is itself unreliable

## System context

A quant research pipeline has a component ("LeafSynthesizer") that turns a plain-English
trading hypothesis into a new, tested indicator function when the hypothesis needs a
mathematical building block the system doesn't have yet. It's a three-stage LLM pipeline:

1. **PLAN** (a large/expensive model, e.g. Claude Opus or GPT-4-class): given the hypothesis,
   produces a structured spec — a natural-language description of the computation, required
   input columns, parameters, AND a hand-typed **worked numeric example**: sample input data,
   parameter values, and the expected boolean output the model computed by hand.
2. **CODE** (a smaller/cheaper model, e.g. Claude Haiku or GPT-4-mini-class): given the spec,
   writes the actual Python implementation (a pure function over a pandas DataFrame) plus a
   pytest test that asserts the function's output on the worked example's input matches the
   worked example's expected output exactly.
3. **REVIEW** (a mid-tier model, e.g. Claude Sonnet or GPT-4-class): does a safety/lookahead
   read of the code, then the pipeline actually **executes** the generated pytest test in a
   subprocess. If the test passes (i.e., the CODE-stage implementation's output matches the
   PLAN-stage model's hand-typed expected output), the new indicator is approved and shipped
   into the live system. If the test fails, the whole attempt is rejected — the code is thrown
   away, nothing ships.

The stated design intent: the pytest execution is meant to be "a deterministic gate, not an
LLM's opinion of its own code" — i.e., the team believed that because the test *runs real
code and asserts an exact match*, it was immune to LLM unreliability.

## What actually happened (live evidence, not a hypothetical)

Running this pipeline for real (not mocked) against a genuinely representable indicator —
Chaikin Money Flow, a rolling-window volume/price-location formula over 3-7 bars — the PLAN
model was asked twice, independently, to hand-compute a worked example. **Both times its
hand-typed `expected_output` was arithmetically wrong** on a 7-element rolling-sum
calculation (index-by-index mismatches starting around the 4th bar, once the rolling window
starts requiring multi-step summation instead of a lookup).

In the first of these two runs, we independently re-implemented the exact formula in plain
Python and confirmed: the CODE-stage model's generated implementation was **100% correct** —
its output matched ground truth exactly. The PLAN-stage model's hand-typed "expected" values
were the ones that were wrong. The pipeline nonetheless rejected the (correct) generated
code, because it disagreed with the (incorrect) hand-typed example.

So the "deterministic gate" is executing real code and doing an exact-match assertion — but
the thing it's asserting against (the expected values) was itself produced by an LLM doing
mental arithmetic on a multi-step rolling calculation, which is exactly the kind of task LLMs
are known to be unreliable at. The gate is deterministic in *mechanism* but not in *ground
truth*.

## The design question

**How should this correctness-verification step be redesigned so it doesn't depend on any
single LLM's ability to hand-compute a worked example correctly, while still catching
genuinely wrong or unsafe generated code?**

Two candidate directions under consideration, not mutually exclusive:

**A. Property-based checks instead of/in addition to exact-match:**
Replace or supplement the "matches one hand-typed example" gate with checks that don't
require knowing the "correct" numeric answer in advance — e.g., a prefix-stability probe
(the function's output on the first N rows of a dataframe must equal the first N rows of its
output on the full dataframe — catches lookahead/future-peeking mechanically), NaN/edge-case
safety, output type/shape invariants. These are self-consistency properties, not
correctness-vs-ground-truth checks, so they sidestep the arithmetic-reliability problem
entirely — but they don't verify the formula was implemented *correctly*, only that it's
*well-behaved*.

**B. Differential / N-version testing:**
Have a second, independently-prompted model (or the same REVIEW-stage model) generate a
*second*, independent implementation of the same spec. Run both implementations against
several rounds of random synthetic input data (not just the one hand-typed example) and
require they agree within floating-point tolerance. Two independently-generated
implementations agreeing across many random inputs is much stronger evidence of correctness
than one model's hand-arithmetic on one example — and it never requires trusting any single
model's mental math. Costs one extra model call and more compute per synthesis attempt.

## What I want from you

1. Is there a well-known name/pattern for this failure mode (an LLM-graded or LLM-oracled
   test whose oracle is itself LLM-generated and unreliable) — pointers to how other systems
   (LLM-as-judge literature, AI-assisted code generation pipelines, N-version programming,
   metamorphic testing) have solved this class of problem would help.
2. Which of A, B, both, or a different approach entirely would you recommend, and why?
3. Are there cheaper/lighter-weight alternatives to full differential testing (option B) that
   still avoid trusting a single model's hand arithmetic — e.g., using a deterministic
   numeric library call instead of a second LLM implementation where the formula is a named,
   well-known indicator (this one happens to be Chaikin Money Flow, a real technical
   indicator with existing reference implementations in libraries like `pandas-ta` or `ta-lib`
   — is "check against a library implementation when the indicator is a recognizable named
   one, fall back to differential/property testing when it's genuinely novel" a reasonable
   hybrid)?
4. Any failure modes or false-confidence traps in options A/B I should watch for before
   building either?
