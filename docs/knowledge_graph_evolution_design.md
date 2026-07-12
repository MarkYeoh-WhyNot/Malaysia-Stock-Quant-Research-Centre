# Knowledge Graph Evolution — Design Document

**Status:** IMPLEMENTED 2026-07-12 (Slices 1–5, local; not yet deployed/committed).
Proven against a DB copy + live dashboard; full test suite 697 passed / 0 failed.
**Revision:** 2 (incorporates external review — see §14 for dispositions)
**Date:** 2026-07-12
**Author:** Claude (synthesising two external proposals against the live codebase)
**Decision owner:** Mark

---

## 0. Purpose

Decide how to evolve the existing knowledge graph (`kb_nodes` / `kb_edges` /
`kb_fts` / `kb_embeddings`) from a **concept graph** (techniques, ideas,
rejections) into a **research-evidence / "truth" graph** that also carries
strategies, backtests, gate decisions, signatures, executable leaves, risks, and
agent findings — so the graph becomes the queryable *memory ledger* of the
research centre, not just a note store.

This document synthesises an external (OpenAI) green-field proposal and a
subsequent external review of Revision 1 of this document. Both are
**directionally excellent and philosophically aligned** with this system's
parser-honesty / quality-over-quantity ethos, but both were written **without
the schema in front of them**, so ~70% of the green-field proposal is already
built, and two review amendments duplicate columns that already exist. This
document keeps the good philosophy, discards "build from scratch," and defines
the *minimum* real additions, grounded in the actual tables.

---

## 1. Core decision

> **Extend the existing `kb_*` graph. Do NOT create a parallel `kg_*` schema.**

A parallel `kg_nodes` / `kg_edges` / `kg_claims` schema would create a
**split-brain graph**: two sources of truth, two exporters, two retrievers, two
FTS indexes, two review flows, eventual disagreement. Rejected. Everything below
reuses the single write path (`knowledge/graph/store.py` — the ONLY writer that
keeps `kb_fts` in sync) and the existing retrieval / export / dashboard
surfaces.

---

## 2. What already exists vs. what the proposals assume is missing

| Proposal says "build…" | Current reality | Action |
|---|---|---|
| `kg_nodes` / `kg_edges` tables | `kb_nodes` / `kb_edges` (typed, weighted, `origin`, `UNIQUE(src,tgt,relation)`, `ref_table`/`ref_id`, `content_hash`) | **Extend** |
| Markdown vault w/ frontmatter + wikilinks | `scripts/export_obsidian.py`, now **two-way** (feedback zone) | Done |
| Dashboard network graph | KB Explorer Graph tab — Cytoscape `fcose`, `/api/kb/graph`, `kbGraphShowNode` | Enrich |
| Graph-aware AI retrieval | `retriever.retrieve(query,k,hops)` — hybrid BM25+cosine seeds, 1–2 hop BFS, `0.5^hop` decay, `assemble_context()` | Extend |
| Rejection nodes + `rejected_*` edges | `rejection_pattern` nodes + `rejected_because` edges + `rejection_patterns.count` + `strategy_cemetery` revival rows | Done |
| Standard agent-finding packets | `governance_findings` table (`agent, level, scope, status, severity, evidence, local_recommendation, escalate_to`) + `governance/` module | **Promote into graph** |
| Node source-traceability (`ref_table`, `ref_id`, `source_hash`) | **`ref_table`, `ref_id`, `content_hash` already columns on `kb_nodes`** | Reuse; add only `ingestion_version` |
| Human review → "trusted" state | Two-way Obsidian feedback loop (`kb_feedback` verdicts) | Map onto it |

### 2.1 What is currently registered as a node

```
technique_library   33      (node_type technique)
alpha_ideas       1180      (node_type idea)
rejection_patterns   3      (node_type rejection_pattern)
```

Not registered: `backtest_runs`, `gate_decisions`, `paper_trades`,
`signal_signature` values, DSL leaves (`agents/backtest_engineer/signal_dsl.py`),
`governance_findings`. That gap is exactly the difference between a *concept*
graph and a *truth* graph.

---

## 3. Principles

### 3.1 Adopted (match our ethos — keep)

1. **Truth graph, not note graph.** Encode causality/evidence/decision —
   `idea → compiled_to → strategy → produced → backtest_run → failed →
   gate_decision → rejected_because → rejection_pattern` — not vague
   `related_to`.
2. **No claim without a source.** Evidence nodes carry `ref_table`/`ref_id`
   (existing) and `origin`.
3. **Deprecate, don't delete.** Obsolete nodes get a status, never a `DELETE`.
4. **Deterministic > LLM > human-reviewed** trust order. LLM extraction never
   auto-promotes to `trusted`.
5. **Provenance first-class.** `origin` (`llm`/`heuristic`/`human`) on edges;
   `review_state` on nodes driven by the feedback loop.
6. **Anti-garbage governance** as an explicit, tested daily job (§8).

### 3.2 Rejected / softened

1. **Rejected: parallel `kg_*` schema** (§1).
2. **Softened: four physical layers** (Research/Evidence/System/Document) → one
   graph + `node_type` facet + saved views. Same data, different lenses, no
   duplicated truth.
3. **Deferred: the full 24-node / 26-edge ontology, `kg_claims`,
   `kg_snapshots`.** Start with 7 node types and a tight relation set. Declaring
   types before data exists is how graphs rot.
4. **Kept tight: LLM extraction** stays budget-capped, candidates-only.

---

## 4. Target architecture

```
operational tables
  (alpha_ideas, signal_signature, signal_dsl leaves, backtest_runs,
   gate_decisions, paper_trades, governance_findings)
        │  deterministic ingesters → store.upsert_node() / store.add_edge()   [NEW]
        ▼
  kb_nodes / kb_edges     node_type registry + RELATIONS extended             [EXTEND]
  + kb_aliases            entity resolution (BTC/BTCUSDT/XBT, DPSR/…)          [NEW, small]
  + kb_node_type_registry validated types without table rebuilds              [NEW, tiny]
        │
        ├── KB Explorer graph tab   + node_type/status filters + saved views   [ENRICH existing]
        ├── Obsidian vault (2-way)  human_reviewed/trusted == kb_feedback       [DONE]
        ├── retriever               typed facets (past_failures, …)             [EXTEND existing]
        └── graph_health_check      anti-garbage rules as a daily job           [NEW, small]
```

No box forks the schema; each is an extension of an existing module.

---

## 5. Schema changes (additive, migration-safe)

All via `data/database.py::init_db` (idempotent `CREATE TABLE IF NOT EXISTS` /
guarded `ALTER … ADD COLUMN`), which auto-runs at daemon startup.

### 5.1 New node types (v1 = 7 additions)

```
existing: note, concept, technique, idea, rejection_pattern
add:      strategy, signature, backtest_run, gate_decision, risk, finding, leaf
```

- `strategy` — a promoted/evaluated idea (see §5.5); NOT every idea.
- `signature` — `signal_signature` identity → "which strategies share a hidden factor" in one hop. Identity = the signature string (slug); `ref` null.
- `backtest_run`, `gate_decision` — the evidence chain, one node per row.
- `risk` — cost drag, parser approximation, BTC-beta overlap, liquidation fragility. Identity-defined (slug); `ref` null.
- `finding` — `governance_findings` promoted.
- `leaf` — executable DSL leaf from `agents/backtest_engineer/signal_dsl.py`. Added in v1 (not later) because parser honesty and signature representability depend on leaves — the `ma_level` failure is the motivating case. Identity = leaf name (slug); `ref` null.

Everything else in the green-field 24-type list (`factor`, `metric`,
`instrument`, `regime`, `dataset`, `feature`, `module`, `bug`, `claim`,
`todo`, …) is **deferred** until a concrete query needs it.

### 5.2 Node-type validation — registry, not a bare CHECK-drop

SQLite cannot `ALTER` a CHECK, and rebuilding `kb_nodes` on every type addition
is costly. But simply dropping the CHECK and trusting the app loses discipline.
Middle path:

```sql
CREATE TABLE IF NOT EXISTS kb_node_type_registry (
    node_type   TEXT PRIMARY KEY,
    description TEXT,
    status      TEXT DEFAULT 'active',   -- active | deprecated
    created_at  TEXT DEFAULT (datetime('now'))
);
```

- Drop the DB-level CHECK on `kb_nodes.node_type` (table rebuild, one-time).
- `store.py::upsert_node` validates `node_type` against the registry (as
  `add_edge` already validates relations against `RELATIONS`).
- `graph_health_check` treats an unregistered `node_type` as a **BLOCKER**.

Flexibility (add a type = one INSERT) without losing loud enforcement.

### 5.3 New relations (v1)

Extend the `RELATIONS` tuple in `store.py`:

```
existing: supports, contradicts, refines, derived_from, about_ticker,
          uses_technique, rejected_because, shared_concept, shared_tag, mentions
add:      produced, failed, passed, shares_signature, reported_by, blocks,
          measured_by, exposed_to, affects, compiled_to, uses_leaf
```

**Do NOT add `rejected_for`** — the existing `rejected_because` already means
this; two relations for one concept is a schema smell. Everything rejection-
related uses `rejected_because`, targeting `rejection_pattern` nodes.

`affects` and `compiled_to` are explicitly in the list (Rev 1 referenced them in
prose but omitted them from the tuple — fixed). `compiled_to` / `uses_leaf`
carry the parser-honesty mini-graph (§5.6, Slice 1.5).

### 5.4 Node provenance / trust — reuse what exists, add two columns

`kb_nodes` **already has** `ref_table`, `ref_id` (INTEGER), `content_hash`,
`updated_at`. `content_hash` already serves the "did the source row change?"
role — `upsert_node` resets `extracted_at` when it changes. So the review's
proposed `source_hash` is **redundant**. Add only:

```sql
ALTER TABLE kb_nodes ADD COLUMN confidence        REAL;   -- nullable
ALTER TABLE kb_nodes ADD COLUMN review_state      TEXT;   -- machine | human_reviewed | trusted | deprecated
ALTER TABLE kb_nodes ADD COLUMN ingestion_version TEXT;   -- lets a schema bump mark old nodes stale
```

**Identity-defined nodes** (`signature`, `risk`, `leaf`) have no source row —
their identity IS the slug, so `ref_table`/`ref_id` stay null for them; only
`strategy`/`backtest_run`/`gate_decision`/`finding` carry a `ref`.

**Trust linkage:** `review_state` transitions are driven by the existing
two-way feedback loop — a `promote` verdict in `kb_feedback` → `trusted`;
`reject` → `deprecated`. The human-review state machine is already half-built.

### 5.5 `idea` vs `strategy` promotion (keep them separate)

Do NOT promote all 1,180 `alpha_ideas` to `strategy` (pollution + low signal).

```
idea      = raw / low-stage candidate                     (default)
strategy  = executable or evaluated candidate
signature = shared structure / factor identity
```

Promote `idea → strategy` (emit a `strategy` node + `idea —compiled_to→ strategy`
edge) only when the idea:

- parses to a valid DSL/factor tree, **or**
- enters a backtest, **or**
- receives a gate decision, **or**
- is human-promoted (a `promote` verdict via the feedback loop).

### 5.6 Edge evidence — node-per-run first, `evidence_count` as hedge

`UNIQUE(src,tgt,relation)` collapses repeats, which could hide "cost drag killed
this signature 17 times." Two mitigations, in priority order:

1. **Primary (already the design):** one node per `backtest_run` / per
   `gate_decision`. The "17 runs" query is answered by counting distinct paths
   `signature ← shares_signature ← strategy → produced → backtest_run → failed →
   gate_decision`, and `rejection_patterns.count` + `strategy_cemetery` already
   aggregate rejection frequency. Auditability comes from the nodes, not a
   collapsed edge.
2. **Hedge (cheap):** for any *rollup* edge we later materialise for fast
   retrieval, add to `kb_edges`:
   ```sql
   ALTER TABLE kb_edges ADD COLUMN evidence_count INTEGER DEFAULT 1;
   ALTER TABLE kb_edges ADD COLUMN last_seen_at   TEXT;
   ```

A separate `kb_edge_evidence` table (the review's Option B) is **deferred** —
the node-per-row structure already preserves the individual evidence; we only
build the table if/when materialised rollup edges prove necessary.

### 5.7 New table: `kb_aliases` (entity resolution)

```sql
CREATE TABLE IF NOT EXISTS kb_aliases (
    alias       TEXT PRIMARY KEY,        -- "Bitcoin", "XBT", "Deflated PSR"
    node_id     INTEGER REFERENCES kb_nodes(id),
    alias_type  TEXT,                    -- ticker | metric | technique | leaf | …
    confidence  REAL DEFAULT 1.0,
    origin      TEXT DEFAULT 'human',    -- human | heuristic | llm
    created_at  TEXT DEFAULT (datetime('now'))
);
```

Seeded deterministically (§12 Q4): Bursa from `stock_universe` (+ `.KL`
variants), crypto from the exchange symbol map (base/quote, BTC/BTCUSDT/XBT),
metrics from a hardcoded dictionary (DPSR→Deflated PSR, IC→Information
Coefficient, ADV→Average Daily Value). LLM-suggested aliases land as
`origin='llm'`, candidates only, promoted by human review.

---

## 6. Ingestion design

### 6.1 Channel 1 — deterministic system ingestion (core of v1)

New `knowledge/ingestion/evidence_graph.py`, run as a daemon step. For each
new/updated row → `store.upsert_node()` + `store.add_edge()`:

```
alpha_ideas (promoted) → strategy node ; idea —compiled_to→ strategy
signal_signature       → signature node ; strategy —shares_signature→ signature
signal_dsl leaves      → leaf node ; strategy —uses_leaf→ leaf
backtest_runs          → backtest_run node ; strategy —produced→ backtest_run
gate_decisions         → gate_decision node ; backtest_run —failed/passed→ gate_decision
                          gate_decision —rejected_because→ rejection_pattern (existing)
governance_findings    → finding node ; finding —reported_by→ agent
                          finding —affects→ module ; finding —blocks→ gate/stage
                          parser findings: finding —exposed_to→ risk(parser_approximation)
```

Fully deterministic, idempotent (`content_hash` handles change detection), zero
LLM budget.

### 6.2 Channel 2 — document ingestion (exists, unchanged)

`kb_documents` → chunks → `kb_concepts`, Haiku extractor. LLM claims stay
`origin='llm'` / `review_state='machine'`.

### 6.3 Channel 3 — agent findings

`governance_findings` already IS the standard packet; v1 only *promotes* rows to
`finding` nodes (Channel 1). No new packet format. Makes the board architecture
queryable: "all unresolved blocker findings under the Backtest Fidelity dept."

---

## 7. Retrieval extension (typed facets)

Extend `assemble_context()` (or add `retrieve_facets(query)`) to return a
structured packet grouped by relation, e.g. "evaluate this BTC funding strategy":

```json
{
  "direct_matches":           ["Funding Rate Carry", "funding_avg"],
  "past_failures":            ["strategy_881", "strategy_902"],
  "common_rejection_reasons": ["cost_drag", "liquidation_regime_fragility"],
  "relevant_metrics":         ["funding_missing_pct", "paper_backtest_drift"],
  "open_risks":               ["exchange-specific funding distortion"]
}
```

A formatting/aggregation layer over the existing traversal — not a new
retriever.

---

## 8. Anti-garbage governance (`scripts/graph_health_check.py`, new, daily)

Enforces the rules as assertions, logging/alerting violations (dogfoods the
finding machinery):

```
- node_type ∈ kb_node_type_registry (else BLOCKER)
- evidence nodes (strategy/backtest_run/gate_decision/finding) have ref_table+ref_id
- every edge has relation ∈ RELATIONS and an origin
- duplicate-title detection → alias candidates
- orphan detection: active signatures with < N supporting edges
- deprecated nodes retained (never deleted)
```

---

## 9. Surfaces

- **Dashboard KB Explorer graph tab** (exists): add `node_type`/`status` filters
  + saved views — "Why did this strategy fail?", "Parser honesty map",
  "Portfolio overlap map", "Open blocker findings". Cytoscape/`fcose` +
  click-to-detail already exist; this is filter + preset work.
- **Obsidian vault** (two-way, exists): new node types export as notes with
  typed `[[wikilinks]]`; feedback verdicts drive `review_state`.
- **AI retrieval** (exists): typed facets per §7.

---

## 10. Phased build plan

Sequenced to serve the current research direction (strategy exploration /
funding-carry sweep), not a big-bang.

**Slice 1 — Strategy evidence graph (the wedge).**
Registry + app-level `node_type` validation; extend `RELATIONS`; new
`evidence_graph.py` registering `strategy`, `signature`, `backtest_run`,
`gate_decision` nodes and `compiled_to / shares_signature / produced /
failed|passed / rejected_because` edges from existing rows. Prove with *"which
funding-carry strategies were tried, which died, and why"* against a DB copy.
Deterministic, zero LLM. **Directly serves the next research move.**

**Slice 1.5 — Parser-honesty mini-graph.**
Add `leaf` nodes (seeded from `signal_dsl.py`) and the `uses_leaf` / `compiled_to`
edges; emit a `finding` + `exposed_to → risk(parser_approximation)` when a parse
is non-representable. Just enough to make parser failures visible early — NOT a
full parser graph. Motivated by the SACRED parser-honesty contract and the
`ma_level` history.

**Slice 2 — Aliases + provenance.**
`kb_aliases` (+ deterministic seeding); `confidence` / `review_state` /
`ingestion_version` columns; wire `kb_feedback` verdicts → `review_state`.

**Slice 3 — Finding graph.**
Promote `governance_findings` → `finding` nodes + `reported_by / affects /
blocks` edges. Unlocks "open blocker findings by department."

**Slice 4 — Surfaces.**
Saved views + type/status filters in the graph tab; render new node types in the
Obsidian export.

**Slice 5 — Typed retrieval facets + graph health check.**
`retrieve_facets()` and `scripts/graph_health_check.py` daily job.

Each slice is independently shippable, reversible, additive.

---

## 11. SACRED / non-goals

- **Does not touch** gate thresholds (`GATE_CONFIG`/`GATE_OVERRIDES`), the
  parser-honesty contract, DSL leaves in `signal_dsl.py`, or anything touching
  live capital / risk limits. The graph *records* these; it never *decides*
  them.
- **Dual-market:** ingesters read the active market profile — no hardcoded
  `.KL` checks. Bursa behaviour stays byte-identical.
- **No parallel `kg_*` schema.** (Restated — primary risk.)
- **No auto-trust:** LLM nodes/edges never reach `trusted` without a human
  verdict.
- **v1 is deterministic-first:** Slices 1 and 1.5 spend zero LLM budget.

---

## 12. Open questions — RESOLVED (per review)

1. **CHECK vs app-level validation** → app-level in `store.py`, backed by
   `kb_node_type_registry`; health check treats invalid type as BLOCKER. (§5.2)
2. **`strategy` vs `idea`** → keep separate; promote only on parse/backtest/
   gate/human-promotion. (§5.5)
3. **Ingest cadence** → three tiers: `evidence_ingest` frequent+cheap (each
   daemon cycle / after a backtest), `graph_maintain` (LLM edges/embeds) every
   2h, `graph_health_check` daily.
4. **Alias seeding** → deterministic first (`stock_universe` + `.KL`; crypto
   exchange symbol map; hardcoded metric dictionary); LLM = candidates only.
5. **Slice 1 proof query** → funding-carry failure map (crypto/funding is the
   live direction; exercises signatures + rejections + evidence chain fastest).

---

## 13. One-line summary

Keep the proposals' *philosophy* — a provenance-first truth graph — and throw
away their *green-field schema*. Extend `kb_*` with 7 node types, a tight
relation set (reusing `rejected_because`, adding `affects`/`compiled_to`/
`uses_leaf`), a registry table, an alias table, and two node columns
(`review_state`, `ingestion_version`) atop the `ref_table`/`ref_id`/
`content_hash` that already exist. Feed it deterministically; drive trust from
the two-way feedback loop already built; surface it through the Cytoscape tab and
Obsidian vault already built. Start with the strategy evidence graph + parser
mini-graph, because they answer the question the next research sprint will ask.

---

## 14. Review dispositions (Rev 1 → Rev 2)

| # | External amendment | Disposition |
|---|---|---|
| 1 | Don't run both `rejected_because` and `rejected_for` | **Accepted** — dropped `rejected_for`; reuse `rejected_because` (§5.3) |
| 2 | Add `affects` to the relation list | **Accepted** — added; also added `compiled_to`/`uses_leaf` (§5.3) |
| 3 | Add `ref_table`, `ref_id`, `source_hash`, `ingestion_version` to nodes | **Partial** — `ref_table`/`ref_id`/`content_hash` already exist; added only `ingestion_version` (+ `review_state`, `confidence`) (§5.4) |
| 4 | Edge evidence table / `evidence_count` | **Accepted lite** — node-per-row + `rejection_patterns.count` are primary audit; `evidence_count`/`last_seen_at` as hedge; full table deferred (§5.6) |
| 5 | Keep `idea` vs `strategy` separate | **Accepted** — promotion rule (§5.5) |
| 6 | Parser-honesty mini-graph + `compiled_to` | **Accepted** — Slice 1.5 (§5.6→§10) |
| 7 | Registry + app validation, not bare CHECK-drop | **Accepted** — `kb_node_type_registry` (§5.2) |
| 8 | Add `leaf` as a v1 node type | **Accepted** — seeded from `signal_dsl.py` (§5.1) |
