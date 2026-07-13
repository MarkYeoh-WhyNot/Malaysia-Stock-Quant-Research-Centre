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
