# Consultation Request: Designing a Strategy-Signature Database / Technique Arsenal for an Autonomous Quant Research Pipeline

You are being consulted as a senior quantitative-research systems architect. Below is a
precise description of a working production system, its core purpose, its current
"Technique Arsenal" knowledge base, and a concrete failure that motivated this
consultation. Your task is at the end. Please reason from the system AS DESCRIBED — do
not assume standard architectures it doesn't have.

## 1. The system (ground truth, verbatim from the codebase)

A dual-market autonomous quant research pipeline (Bursa Malaysia KLCI equities +
Binance USDT-M crypto perpetuals; one market per process). Ideas flow:
idea generation / human chat submission → LLM quality screen → deep research →
backtest gauntlet → adversarial red/blue review → paper trading → (human-gated) live.

**Core purpose (North Star, verbatim):** "Find genuine, statistically robust alpha
factors. Prove them cross-sectionally. Deploy them safely with human oversight at every
capital decision point. Quality over quantity — always. 10 robust, well-validated
strategies beats 300 hastily generated noise ideas."

**The honesty contract (load-bearing design principle):** a strategy described in free
text is parsed by an LLM into a JSON condition tree over a FIXED registry of executable
"leaves". If the idea cannot be expressed with the available leaves, the parser MUST
return `{"representable": false, "reason": ...}` and the idea is rejected with that
reason — silent approximation onto the nearest available mechanism is forbidden (it was
a historical failure mode that flattened every thesis into 20-day momentum and made real
edges indistinguishable from noise).

**Gate design:** the backtest gauntlet's principal rule is a deflated Probabilistic
Sharpe Ratio (PSR ≥ calibrated confidence that the true net Sharpe beats the expected
max Sharpe of the trailing 90 days of noise trials), surrounded by orthogonal guards:
drawdown caps, noise-aware train/val-gap tolerance, out-of-sample walk-forward
degradation, volatility-regime terciles, ±20% parameter-perturbation robustness, cost
drag, minimum trade counts, liquidity/capacity floors, a risk-adjusted equal-weight
benchmark gate, and a cross-sectional IC gate (mean IC > 0.05, Newey-West t > 1.5,
positive on 15/30 names Bursa, 12/20 crypto). Gate honesty is verified by a calibration
harness (planted synthetic edges must pass; pure noise must not).

**The executable DSL (the vocabulary any "design example" must conform to):**
A strategy tree is `{"entry": <node|null>, "exit": <node|null>, "short_entry":
<node|null>, "short_exit": <node|null>}` (short legs only where the market allows
shorts). A node is a leaf `{"leaf": "<name>", <params>}` or a combinator
`{"op": "AND"|"OR", "children": [...]}` / `{"op": "NOT", "child": ...}`. Max depth 4,
max 6 leaves. The full current leaf registry (params with hard valid ranges):

- rsi(period: int[2,50]; one of below: float[1,99] / above: float[1,99])
- sma_cross(fast: int[2,100], slow: int[5,300]; direction: above|below)  — TWO-MA level comparison
- ema_cross(fast: int[2,100], slow: int[5,300]; direction: above|below)  — TWO-MA level comparison
- momentum(period: int[2,252], min_return: float[-0.5,0.5])
- reversal(period: int[2,30], max_return: float[-0.5,0.0])
- bollinger(period: int[5,60], std: float[0.5,4.0]; band: below_lower|above_upper)
- macd(fast: int[2,50], slow: int[5,100], signal: int[2,30]; condition: bullish|bearish)
- volume_ratio(period: int[5,60], min_ratio: float[1.0,10.0])
- gap(min_pct: float[0.001,0.2]; direction: up|down)
- rolling_rank(formation: int[20,252], skip: int[0,30], window: int[60,504]; one of min_pct/max_pct)  — time-series percentile rank of own momentum
- div_days_to_ex(max_days: int[1,30])                     — Bursa dividend calendar
- cpo_change(period: int[1,30], min_pct: float[-0.2,0.2]) — palm-oil futures (Bursa)
- zscore(period: int[10,200]; one of below: float[-4,0] / above: float[0,4])  — price z-score
- funding_level(one of below: float[-0.005,0] / above: float[0,0.005])        — crypto perp funding
- funding_zscore(period: int[10,200]; one of below/above like zscore)         — crypto perp funding

Separately, cross-sectional BASKET strategies use a continuous factor registry
(momentum/reversal/ts_zscore/vol_ratio/funding_avg/funding_zscore, each with typed param
ranges) with a submission shape `{"factor": {"name", "params"}, "top_n", "bottom_n",
"rebalance_bars"}` — rank the universe, long top-N / short bottom-N.

**The current Technique Arsenal (what we want to redesign):** two Python dicts —
21 Bursa techniques, 12 crypto techniques. Schema per entry, 100% prose:
`name, angle (one of 9 research categories: price_action, fundamental, event_driven,
institutional, macro, commodity, sector_rotation, behavioural, statistical_modelling),
when_to_use[], when_to_avoid[], market_applicability, ic_improvement_vs_sma,
stock_types[], strategy_types[], holding_periods[], signal_types[], implemented (bool),
complexity, overfitting_risk`. Entries range from directly-backtestable patterns
(sma_crossover, rsi_mean_reversion, gap_fill) through cross-sectional factors
(funding_rate_carry, xs_momentum_majors) to validation methodologies
(cross_sectional_ic, deflated_sharpe) to aspirational/unimplemented concepts
(kalman_filter, hidden_markov_model, garch). There is NO concrete example — no canonical
tree, no parameter set, no worked formula — anywhere in the arsenal.

**How the arsenal is consumed (three LLM injection points, all prose):**
1. Idea generation: top-3 relevance-scored techniques rendered as prose bullets.
2. A chat Concierge agent that turns a user's natural-language strategy into a free-text
   `factor_formula` string (it has a compact key/name index + an on-demand full-detail
   tool — both prose).
3. Red-team adversarial review: `when_to_avoid` bullets as attack ammunition.

**Critical architecture fact — the two-hop cold-parser problem:** the Concierge writes
free text only. The DSL tree is built LATER, asynchronously, by a SEPARATE small-model
LLM call ("the parser") that sees ONLY the stored free text + the bare leaf catalog
(names + parameter ranges — deliberately NO example values, because an earlier prompt
that pre-filled default values caused the parser to anchor on the defaults instead of
extracting the idea's own parameters). The parser has no memory of the chat, no
technique context, and no worked examples.

## 2. The observed failure that motivated this

A user asked for: "Long BTC/USDT when it closes above the 50-day EMA, short when it
closes below." There is NO leaf comparing price to a single moving average — only the
two-MA crossover leaves. The parser, rather than returning `representable: false`,
approximated it as `ema_cross(fast=2, slow=50)` (EMA(2) as a stand-in for price — fast
has a hard minimum of 2). Structurally legal, semantically wrong, silently accepted —
precisely the class of failure the honesty contract exists to prevent. The strategy was
then backtested, rejected by the gates for parameter fragility, and the user was told a
story about the wrong strategy.

## 3. What we've already concluded (challenge these if you disagree)

- Add the missing `ma_level` leaf (price vs one SMA/EMA) — vocabulary gap, clear-cut.
- Every arsenal entry should carry an EXPLICIT design example, machine-validated against
  the leaf registry: a canonical DSL tree for tree-shaped techniques, a factor spec for
  cross-sectional ones, and an honest `example: none — <reason>` for methodology entries
  and unimplemented concepts (never a fabricated tree).
- The parser prompt should carry a small fixed leaf-level worked-example cheat-sheet
  (one validated tree per leaf) rather than technique-level retrieval, because retrieval-
  by-keyword into a cold parser is fragile.

## 4. YOUR TASK

Design the target architecture for this "signature database" (arsenal v2). Specifically:

1. **Schema**: propose the full per-entry schema for a technique/signature entry that
   serves this system's core purpose. What fields beyond our current prose + your view
   on our planned `dsl_example`/`factor_example`? Consider: canonical parameter sets vs
   sweepable ranges, regime/market-condition tags, cost-sensitivity class, expected
   trade frequency, empirical evidence links (this system records every backtest verdict
   in SQLite — should signatures accumulate LIVING evidence from pipeline outcomes, and
   if so what exact fields/update rules?), provenance (literature vs internal
   discovery), and versioning.
2. **Taxonomy**: we have 9 research angles and 33 entries across 2 markets. Is
   per-technique the right granularity, or should the database be organized as
   signature FAMILIES (e.g. "trend filter" family containing price-vs-MA, two-MA cross,
   MACD variants) with per-family canonical examples and per-variant parameter maps?
   Show the tradeoff explicitly against LLM-prompt-injection practicality.
3. **The anchoring tension (hardest question)**: our parser deliberately shows parameter
   RANGES with no example values, because examples caused value-anchoring. But zero
   examples caused structural improvisation (the EMA failure). How do you design example
   injection that teaches STRUCTURE without anchoring VALUES? Be concrete — show the
   exact prompt-block format you'd use.
4. **Sync/validation discipline**: examples must never drift from the executable
   registry. Propose the validation/CI rules (we already plan: every example must pass
   the tree validator + fire on synthetic data — extend or amend this).
5. **Lifecycle/governance**: how do entries get added, promoted (conceptual →
   implemented), evidence-weighted, deprecated? Who/what writes to it (the pipeline
   automatically? human-curated only? LLM-proposed + human-approved?), given the
   quality-over-quantity mandate and a $20/day LLM budget.
6. **Two or three fully-worked example entries** in your proposed schema: one DSL-tree
   technique (e.g. RSI mean reversion), one cross-sectional factor (e.g. funding-rate
   carry), one honest not-yet-implementable concept (e.g. HMM regime detection) — showing
   exactly what "explicit design example" means in your design.
7. **Pitfalls**: what failure modes does YOUR design introduce (e.g. examples becoming
   de-facto templates that homogenize all ideas back into a few shapes — the exact
   disease the honesty contract cured), and what guardrails prevent them?

Constraints to respect: SQLite + Python dicts (no new infra), two market profiles from
one codebase (every Bursa example needs a crypto-valid counterpart or an explicit
market-only marker), small-model parser (concise prompts matter), and the honesty
contract above all: an entry with no truthful example must say so rather than carry a
plausible fake.

Please be opinionated and specific. Disagree with our Section-3 conclusions where you
have a better design. Prioritize what most improves: (a) parser fidelity (free text →
correct tree), (b) idea quality at generation time, (c) the user-facing Concierge's
ability to set correct expectations about what is and isn't expressible.
