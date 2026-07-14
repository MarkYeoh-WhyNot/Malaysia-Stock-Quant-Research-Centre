# Audit Log

Record of the "weekly audit corner" practice (CLAUDE.md DEVELOPMENT RULES
#15, established 2026-07-13): once a week, a focused /code-review-level pass
over one subsystem that hasn't been deliberately looked at recently, logged
here — fixes made, or "confirmed clean."

---

## 2026-07-13 — `knowledge/graph/extractor.py`

**Trigger**: P2-6 of the self-audit remediation plan
(`~/.claude/plans/atomic-imagining-wirth.md`) — idea #218's bad revival
chain went through a `contradicts` edge this module produced, and the
extractor itself had never been read end-to-end before that incident.

**Scope**: read the module fully (prompt construction, candidate selection,
weight/origin recording); pulled every live `contradicts` edge from both
Bursa and crypto production DBs (a few hundred rows total) and hand-checked
a representative sample against the source/target node titles.

**Findings**:
- 100% of live `contradicts` edges are `origin='llm'` — zero
  `origin='heuristic'` edges exist in either market.
- Confidence weight (average 0.80-0.82) does not correlate with
  correctness — edges at 0.88-0.95 weight were wrong just as often as lower
  ones. Recurring failure patterns: near-duplicate ideas about the identical
  strategy labeled as contradicting each other; an idea explicitly derived
  FROM a finding labeled as contradicting that same finding; technique-usage
  relationships inverted into contradictions.
- Separately, `pipeline/revisit.py::detect_triggers()`'s contradicting-finding
  query never actually checked the source node's type, contrary to its own
  docstring — any node type could fire the trigger, not just genuine
  `campaign_findings.py` findings.

**Outcome**: fixed, not just confirmed. See commit
"Stop trusting LLM-extracted 'contradicts' edges for idea revival (P2-6)" —
the revival trigger now requires a genuine `finding-campaign-*` node with a
heuristic-origin edge; the extractor's prompt was tightened to require an
articulable opposing claim before using `contradicts` at all. Currently
means the trigger is dormant (no live edge qualifies) until a real campaign
produces one — the honest outcome given what the audit found.

### Follow-up verification (2026-07-13, same day, self-audit follow-up task 3)

The tightened prompt had never actually run against real pending notes —
`graph_maintain` (every 2h) hadn't cycled since the deploy. With Mark's
go-ahead, manually ran `GraphExtractor().extract_pending()` live on the
Bursa daemon (identical to the automatic job, just ~40 min early) against
its 32 pending nodes.

**Result: `{'processed': 32, 'edges_added': 103, 'concepts_created': 4}`,
zero new `contradicts` edges (226 before, 226 after).** Notably, this batch
was disproportionately exactly the failure pattern the audit flagged worst —
a dozen-plus near-duplicate "Brent Crude Lag → Malaysian O&G" ideas with
different tickers/wording/horizons (`idea-999509`, `999510`, `999526`,
`999527`, `999528`, the `2026-07-13-long-malaysian-o-g-...` idea, etc.).
Under the old prompt this exact shape produced backwards `contradicts` edges
between near-identical ideas (see the OLD edges still sitting in the graph,
e.g. `idea-999482` "3-day entry" ↔ `idea-999480` "3-day horizon",
`idea-999487` ↔ `idea-129`, `idea-999488` ↔ `idea-108` — all pre-fix,
un-repaired, no longer consumed by the revisit trigger but still visible on
the KG dashboard). Under the new prompt, the extractor still linked these
notes (103 edges via `supports`/`refines`/`derived_from`/`uses_technique`/
`about_ticker`/`mentions` — not silently dropped) but produced **no**
`contradicts` edges among them. One batch, not a large sample, but it's the
adversarial case that mattered most, and the prompt handled it correctly.
Historical bad edges from before the fix remain unrepaired — see self-audit
follow-up task 4 (Mark decision needed on cleanup).

### Historical data cleanup (2026-07-13, task 4, Mark's decisions executed)

Mark decided both open questions from the follow-up plan:

1. **`strategy_cemetery` reclassification: batch reclassify via keyword
   fallback.** `scripts/reclassify_cemetery_buckets.py` re-runs the FIXED
   `_classify()` against every cemetery row's stored `rejection_reason`
   (skipping rows already `classified_by='explicit:*'` from the new
   score-based gate0 path) and rewrites `revival_conditions`/`classified_by`.
   Applied live: **Bursa 31/254 eligible rows reclassified** (`overfitting`
   241→211, largest single correction was 18 rows moving to `data_quality`
   for the "NOT reliably available" phrasing gap); **crypto 85/129 eligible
   rows reclassified** (`overfitting` 99→23 — lands almost exactly on the
   audit's originally-reported "23/129 genuinely overfitting" figure,
   confirming the fix reproduces the intended ground truth). Most crypto
   corrections moved to `infeasible` (39) — off-market crypto-perp ideas
   that were previously mislabeled overfitting instead of the more obviously
   correct "wrong venue" bucket. Does not touch `rejection_patterns`
   aggregate counts (separate table, out of scope for this pass) or rows
   already gate0-explicit.
2. **Bad `contradicts` edges: batch delete.** `scripts/delete_bad_contradicts_edges.py`
   deletes every `relation='contradicts' AND origin='llm'` edge (heuristic-origin
   edges and all other relation types untouched). Applied live: **244 Bursa +
   145 crypto = 389 edges deleted** — matches the original audit's "a few
   hundred rows total" figure exactly, confirming the query targeted the same
   population the audit reviewed.

Both scripts are idempotent and safe to re-run (dry-run by default, `--apply`
to write) if either market accumulates new bad edges/mislabels before the
underlying causes are further hardened.

---

## 2026-07-13 — `dashboard/api/server.py`

**Trigger**: self-audit follow-up task 5 (the weekly audit-corner practice
established the same day) — this file is user-facing, has accumulated many
changes, and had never been deliberately audited. Delegated a full read
(2305 lines) plus schema/config cross-check to a research subagent, then
verified every claim against the actual code before fixing anything.

**Findings, verified and fixed**:
- **Stale gate thresholds on `/api/system/direction`**: the `gate_thresholds`
  dict was hand-copied and frozen at pre-2026-07-10 values — `logic` showed
  0.70 (actual 0.65), `stage2_sharpe`/`stage2_tv_gap` were dead fields for
  fixed-Sharpe gates that no longer exist (replaced by the deflated-PSR
  principal rule), `stage4a_sharpe`/`stage4a_max_dd` showed 1.0/15%
  (actual 0.8/20%). The dashboard's System Direction page was actively
  misinforming anyone who checked it against a real gate decision. Fixed by
  deriving the dict live from `GATE_CONFIG` instead of hand-copying, and
  updated `dashboard/ui/index.html`'s matching cards/labels (the frontend had
  its own independent stale fallback defaults). Verified live in a browser —
  all five numbers render correctly post-fix.
- **`market_events` column bug**: `e.get("tickers_mentioned")` referenced a
  column that has never existed (schema has `affected_tickers`) — multi-ticker
  event matching in the Market Intelligence department silently always fell
  back to the single `ticker` field. A second bug on the same line,
  `e.get("title")`, also referenced a non-existent column (schema has
  `headline`) — event snippets always silently showed the generic
  `event_type` instead of the real headline. Both fixed.
- **Hardcoded "PnL (MYR)" label**: unconditional on `MARKET_MODE` — a crypto
  instance's USDT-denominated PnL was mislabeled MYR. Now uses
  `config.settings.MARKET_CURRENCY`.
- **Red-Blue KPI double-count**: `advances` counted `verdict in ("advance",
  "conditional")`, so a conditional-advance debate was counted in both the
  Advances and Conditionals tiles the dashboard renders side by side as a
  breakdown of total debates. Fixed to `verdict == "advance"` only, so the
  four tiles partition cleanly.
- **`/advance` endpoint fabricated audit-trail entries**: calling it on an
  idea already at an unmapped/terminal stage (e.g. `stage5`) silently
  no-op'd the stage transition but still unconditionally wrote an `'advanced'`
  `pipeline_events` row and an `'approve'` `gate_decisions` row — a false
  audit-log entry claiming a transition that never happened, directly at odds
  with the system constitution's "one truth across ... gates" principle. Now
  returns 400 instead.
- **Unbounded PDF upload**: `/api/kb/ingest-pdf` buffered the entire file into
  memory before any size check — real risk on a t3.small VPS already running
  tight on RAM (see `vps-outage-playbook` memory). Added a 25MB cap, read in
  bounded chunks.
- **Non-constant-time API key comparison**: the sole auth gate for the entire
  `/api` surface used plain `!=`. Switched to `secrets.compare_digest`.

**Also corrected**: CLAUDE.md's "CORS is open (allow all origins)" claim was
stale — the code has restricted `allow_origins` to a single configurable
origin for a while. Doc updated to match.

**Confirmed clean**: no SQL injection (every query uses `?` placeholders;
f-strings only interpolate internal fixed constants) — checked against every
table this file queries.

**Regression tests**: added `tests/test_audit_2026_07_13_server_fixes.py`
(5 tests, all passing) covering the gate-threshold drift, the `/advance`
terminal-stage rejection, the red_blue double-count, and the ticker_overlap
column fix. Full suite: 810 passed, 56 skipped (was 805 before this session).
