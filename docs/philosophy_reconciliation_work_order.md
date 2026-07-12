# Philosophy Reconciliation — Work Order

Standalone work order, separated from `docs/board_architecture_work_orders.md` so it can
be executed independently. It aligns the system's stated philosophy with the bottom-up
board architecture (plan: `~/.claude/plans/this-is-the-full-wondrous-bird.md`).

Branch: `restructure/8-department-architecture` (already holds the uncommitted crypto
`DIRECTION_DOC` rewrite + alert→`daemon_logs` change).

---

## Shared Preamble

> Copy this block to the top of the task prompt.

```
Repo: /Users/markyeoh/Documents/GitHub/Malaysia-Stock-Quant-Research-Centre
Branch: restructure/8-department-architecture (do not create new branches; do not commit unless told)
Language: Python 3.12. Activate the venv for any command: `source venv/bin/activate`.
Run tests with: `python -m pytest -q`.

Dual-market rule: two market profiles (bursa | crypto) selected by MARKET_MODE.
config/settings.py loads ONE profile and re-exports both legacy Bursa names and generic
aliases. Bursa behavior and Bursa philosophy text must stay byte-identical.

SACRED — human-only: gate thresholds (GATE_CONFIG/GATE_OVERRIDES), the parser honesty
contract, new executable DSL leaves, live capital, loosening risk limits. This task
touches NONE of them — it is text/docs only.

End by: (1) running `python -m pytest tests/test_direction_doc.py -q` and pasting the
result, (2) stating what you changed and did NOT change. Identity-defining wording is
Mark's call — treat the philosophy text as DRAFT until he approves it.
```

---

## P0 — Philosophy Reconciliation

- **Model:** Sonnet to draft · **Mark approves final wording** · **Depends on:** none ·
  **Parallel with:** everything (pure text/docs, touches no runtime logic).
- **Why:** the crypto `DIRECTION_DOC` was already partly rewritten (data-breadth reframe:
  liquid-major OHLCV is empirically exhausted, edge lives in new data/structure). It does
  NOT yet carry the two framing principles this whole architecture is built on:
  (1) *this is a crypto research/intelligence operating system, not a trading bot* — the
  product is explaining what is moving, why, whether the move is real, and what risk hides
  underneath; (2) *no signal is trusted until it survives its checks* (data-quality,
  liquidity, funding/derivatives, and adversarial/red-team attack) — which is exactly what
  the L0 inspector spine + red/blue team implement. The philosophy text and the board
  architecture must state the same thing.
- **Read first:** `config/markets/crypto.py` (`DIRECTION_DOC`, already partly rewritten),
  `config/markets/bursa.py` (`DIRECTION_DOC` — leave untouched), `tests/test_direction_doc.py`,
  and the `SYSTEM DIRECTION` section in `CLAUDE.md` (stale, Bursa-only, April 2026).
- **Do:**
  1. EXTEND (do not restart) the crypto `DIRECTION_DOC.design_philosophy` / `core_purpose`
     to name both principles above and tie "trust through survival of checks" to the
     inspector/board architecture. Keep the existing data-breadth content.
  2. Make the `CLAUDE.md` `SYSTEM DIRECTION` block market-aware, or add a crypto section,
     that points to the bottom-up board architecture (plan + the board work orders) so new
     sessions start aligned. Keep the Bursa direction intact.
  3. Keep it honest: still paper-only, still human-gated, still zero strategies validated.
- **Acceptance:** `python -m pytest tests/test_direction_doc.py -q` stays green (crypto
  keeps "crypto"/T+0/no-Bursa-shadow invariants; bursa unchanged). Paste the before/after
  of the two philosophy fields. No runtime/logic files touched.
- **Do NOT:** change any Bursa philosophy text; change any gate/threshold; treat this as
  merged until Mark approves the final wording (identity-defining text is his call).
