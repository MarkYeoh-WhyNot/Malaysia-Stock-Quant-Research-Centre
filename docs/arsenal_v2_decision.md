# Arsenal v2 — Decision Record (2026-07-11)

Outcome of the external Gen-AI consultation (`consultation_prompt_signature_db.md`,
OpenAI response received 2026-07-11), evaluated against the actual codebase before
adoption. Motivating failure: idea #73 — the parser approximated "close > 50-day
EMA" as `ema_cross(fast=2, slow=50)`; the Concierge then narrated a wrong
rejection story instead of quoting the stored `verdict_reason`.

## Adopted (implemented in this change)

1. **`ma_level` leaf** — price vs ONE moving average (`ma_type` sma|ema is a
   REQUIRED choice via the new `required_choices` validation; `period` [2,300];
   `direction` above|below). Pine Script mapping included. Auto-covered by
   `perturb_tree` and the optimizer's `randomize_tree` (both derive from the
   registry generically).
2. **Type A / Type B example split** (the consultation's strongest idea):
   - *Type A — validated examples* live in the arsenal entries, are CI-tested
     against the live registries, surface in `format_full_detail` (concierge
     `suggest_techniques`, researcher stage-1) and `/api/system/arsenal` — and
     are **never injected into the cold parser** (value anchoring).
   - *Type B — parser shape cards*: parameter-free, structure-only cards with
     `<EXTRACTED_*>` slots and negative mappings, co-located with each leaf in
     `signal_dsl.LEAVES` (`shape_cards_text()`), injected into `_parse_factor`
     alongside one WRONG-vs-RIGHT negative example (`PARSER_NEGATIVE_EXAMPLE`,
     the idea-#73 case verbatim).
3. **Arsenal v2 slim fields on all 33 entries** (keys unchanged): `family_id`,
   `strategy_shape` (dsl_tree | cross_sectional_factor | methodology |
   unimplemented_concept), `representability` (required_leaves/required_factor/
   missing_leaves), `example` (validated DSL tree | factor spec | honest
   `{"none": reason}` — never a fabricated tree).
4. **Validation-as-pytest** (`tests/test_arsenal_v2.py`) against the LIVE
   registries — drift detection with no hash machinery: change a leaf and its
   examples fail; implement a leaf named in `missing_leaves` and the
   disjointness test forces the entry update.
5. **Concierge verbatim-verdict rule**: rejections are explained by QUOTING the
   stored `rejection_reason`/`verdict_reason` verbatim before any
   interpretation; null reasons are reported as "not recorded", never guessed.

## Deferred (deliberately NOT built)

- Layer-3 evidence ledger aggregation / nightly evidence jobs / `evidence_state`
  lifecycle — SQLite already stores all raw evidence (`backtest_runs`,
  `rejection_patterns`, `strategy_cemetery`); nothing would consume the
  aggregates yet. Revisit when a per-technique evidence view has a consumer.
- Per-market evidence blocks *inside* arsenal entries (duplicates SQLite,
  guaranteed staleness).
- LLM semantic verifier after parsing (cost per parse; deterministic firing
  verify already exists).
- Deterministic parameter-provenance check (Mark voted skip; revisit if the
  parser misbehaves again).
- Family re-keying (red team / dashboard / telegram / knowledge graph match on
  current keys) — `family_id` field delivers the taxonomy without the breakage.
- nl_tests live round-trip runner (LLM cost in tests; prompt-pin tests instead).
- Concierge capability matrix beyond the existing feasibility path.
- §9-style lifecycle governance — the arsenal is a git-controlled Python dict;
  humans are already the only writers.
