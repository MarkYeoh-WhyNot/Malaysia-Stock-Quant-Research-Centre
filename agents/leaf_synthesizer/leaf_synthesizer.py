"""LeafSynthesizer — turns a genuinely-unrepresentable formula into a new,
tested DSL leaf instead of a permanent dead end (Mark-approved 2026-07-13,
after idea #218/#220's cross-asset-ratio z-score got rejected and then
chain-revived forever with no way to ever pass).

Three-stage pipeline, one model tier per stage — same call_claude() budget/
cost tracking every other agent already uses:
  1. PLAN   (MODEL_HEAVY / Opus)  — a structured leaf spec from the
     hypothesis + rejection reason + the existing LEAVES catalog (so it
     doesn't propose a duplicate). Scoped to NEW MATH over columns already
     available (close/open/high/low/volume/funding_rate) — a hypothesis
     needing a genuinely new external data series (e.g. idea #218's
     cross-asset index ratio) is out of scope for this mechanism and Opus
     is instructed to say so, not to fabricate a proxy. Also produces a
     hand-typed worked_example — kept as documentation only (see REVIEW),
     never trusted as ground truth.
  2. CODE   (MODEL_FAST / Haiku)  — the compute function only.
  3. REVIEW (MODEL_MAIN / Sonnet) — a safety checklist (no lookahead, no
     network/filesystem/exec, no duplicate semantics) PLUS a deterministic
     static scan for banned tokens PLUS a SECOND, independent Sonnet call
     writing a differently-styled reference implementation of the same spec
     (never shown Haiku's code) PLUS differential testing: both
     implementations run against ~20 randomized synthetic dataframes and
     PLAN's worked_example, in a subprocess, and must agree. Plus
     deterministic property checks on Haiku's candidate (prefix-stability —
     catches lookahead mechanically — no-input-mutation, shape/index
     preservation, all-NaN safety). Approval requires ALL of these — an
     LLM's say-so alone is never sufficient, since this is the one place in
     the system where model-written code runs unreviewed inside the
     backtest engine.

     2026-07-13 finding (the reason this replaced the original "Haiku's own
     test vs Opus's hand-typed answer" design): live testing found Opus got
     a 7-bar rolling-window sum wrong TWICE independently, and the old gate
     rejected Haiku's CORRECT code because it disagreed with Opus's bad
     arithmetic. The gate was deterministic in *mechanism* (it ran real
     code) but not in *ground truth* (the expected value was an LLM's mental
     math). Differential testing against an independently-written reference
     removes any single model's arithmetic from the approval decision — if
     Haiku's code disagrees only with PLAN's worked_example while agreeing
     with the independent reference across every random trial, that's
     treated as a PLAN arithmetic error (logged, not fatal), not a code
     defect. See docs/leaf_synthesizer_review_gate_decision.md for the
     adopt/defer record and docs/leaf_synthesizer_review_gate_question.md
     for the external consultation this design is based on; known residual risk
     (documented, not solved): correlated errors if Haiku and Sonnet
     misread the same ambiguous spec the same way — mitigated by forcing a
     different implementation style (vectorized vs loop-based) and never
     showing one implementation to the other's author, not eliminated.

Approved leaves are written to agents/backtest_engineer/leaves_generated/
(physically separate from the hand-authored core catalog, for
auditability) AND dual-written to $OPENCLAW_RUNTIME_DIR/leaves_generated/
(the persistent volume) — the package dir is an image layer that evaporates
on the next docker build, so the runtime-volume copy is the real source of
truth in production. signal_dsl.py auto-loads from both at import time,
runtime volume taking priority on a name clash. Every attempt — approved or
not — is logged to leaf_synthesis_attempts, including the full generated
module source on approval (belt-and-suspenders: even if both file copies
are lost, the source is never gone).

Git: production containers have no git binary at all, so this was always a
best-effort local commit that silently failed there (2026-07-13 self-audit).
Where git IS available it still commits locally (push only if
LEAF_SYNTH_AUTO_PUSH=true, off by default), but nothing in this pipeline
depends on that succeeding — see _persist_to_runtime_volume and the
approval-time WATCH alert for the mechanism that actually survives a
rebuild. Approved leaves not yet in git can be recovered with
scripts/export_synthesized_leaves.py.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

from agents.base_agent import BaseAgent, get_agent_daily_spend
from config.settings import (
    MODEL_FAST, MODEL_HEAVY, LEAF_SYNTH_AUTO_PUSH, LEAF_SYNTH_DAILY_BUDGET_USD,
    RUNTIME_DIR,
)
from data.database import db_session

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_GENERATED_DIR = os.path.join(_REPO_ROOT, "agents", "backtest_engineer", "leaves_generated")
_TESTS_DIR = os.path.join(_REPO_ROOT, "tests")
_RUNTIME_GENERATED_DIR = os.path.join(str(RUNTIME_DIR), "leaves_generated")
_RUNTIME_TESTS_DIR = os.path.join(_RUNTIME_GENERATED_DIR, "tests")

_LEAF_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{2,40}$")

# Deterministic safety scan — an LLM's own "safety_pass": true is never
# sufficient by itself. Any of these in generated code is an automatic reject,
# regardless of what Sonnet says.
_BANNED_TOKENS = (
    "import os", "import sys", "import subprocess", "import socket",
    "import requests", "import urllib", "__import__", "eval(", "exec(",
    "open(", "os.system", "os.popen", ".system(", "compile(",
)

_BASE_COLUMNS = ("close", "open", "high", "low", "volume", "funding_rate")


class LeafSynthesizer(BaseAgent):
    name = "LeafSynthesizer"

    def synthesize(self, idea_id: int, hypothesis: str, factor_formula: str,
                   rejection_reason: str) -> dict:
        """Best-effort, always non-fatal to the caller — every path below
        logs a leaf_synthesis_attempts row and returns a summary dict;
        exceptions are caught at the call site (backtest_engineer.py), not
        here, so a synthesis bug can never block the idea's own rejection
        from being recorded."""
        spend_before = get_agent_daily_spend(self.name)
        if spend_before >= LEAF_SYNTH_DAILY_BUDGET_USD:
            self.log_daemon(
                "INFO", f"[{idea_id}] Leaf synthesis skipped — daily budget "
                        f"(${LEAF_SYNTH_DAILY_BUDGET_USD:.2f}) reached")
            return self._log_attempt(idea_id, hypothesis, rejection_reason,
                                     status="budget_exceeded")

        existing_catalog = self._existing_catalog_summary()

        spec = self._plan(hypothesis, factor_formula, rejection_reason, existing_catalog)
        if spec is None:
            return self._log_attempt(idea_id, hypothesis, rejection_reason,
                                     status="plan_failed",
                                     cost_usd=get_agent_daily_spend(self.name) - spend_before)
        if not spec.get("feasible"):
            self.log_daemon(
                "INFO", f"[{idea_id}] Leaf synthesis: not feasible as new math — "
                        f"{spec.get('reason', 'no reason given')}")
            return self._log_attempt(idea_id, hypothesis, rejection_reason,
                                     status="infeasible", spec=spec,
                                     cost_usd=get_agent_daily_spend(self.name) - spend_before)

        leaf_name = (spec.get("leaf_name") or "").strip().lower()
        if not _LEAF_NAME_RE.match(leaf_name) or leaf_name in existing_catalog:
            return self._log_attempt(idea_id, hypothesis, rejection_reason,
                                     status="plan_invalid", spec=spec,
                                     cost_usd=get_agent_daily_spend(self.name) - spend_before)
        bad_columns = [c for c in spec.get("required_columns", []) if c not in _BASE_COLUMNS]
        if bad_columns:
            self.log_daemon(
                "INFO", f"[{idea_id}] Leaf synthesis: spec needs columns "
                        f"{bad_columns} not derivable from OHLCV/funding — out of "
                        f"scope for new-math synthesis, needs a new data source")
            return self._log_attempt(idea_id, hypothesis, rejection_reason,
                                     status="infeasible", spec=spec, leaf_name=leaf_name,
                                     cost_usd=get_agent_daily_spend(self.name) - spend_before)

        code = self._implement(spec)
        if code is None:
            return self._log_attempt(idea_id, hypothesis, rejection_reason,
                                     status="code_failed", spec=spec, leaf_name=leaf_name,
                                     cost_usd=get_agent_daily_spend(self.name) - spend_before)

        review = self._review_and_test(leaf_name, spec, code)
        cost_usd = get_agent_daily_spend(self.name) - spend_before
        if not review["approved"]:
            self.log_daemon(
                "INFO", f"[{idea_id}] Leaf synthesis for '{leaf_name}' rejected: "
                        f"{review['notes']}")
            return self._log_attempt(idea_id, hypothesis, rejection_reason,
                                     status=review["status"], spec=spec, leaf_name=leaf_name,
                                     review_notes=review["notes"], cost_usd=cost_usd)

        land_result = self._land(leaf_name, spec, code, review["confirmed_trial"])
        commit_sha = land_result["commit_sha"]
        self.log_daemon(
            "INFO", f"[{idea_id}] Leaf synthesis APPROVED: new leaf '{leaf_name}' "
                    f"written to leaves_generated/ + runtime volume "
                    f"(persisted={land_result['runtime_persisted']}), "
                    f"commit={commit_sha or 'not committed — see runtime volume + audit table'}")
        self._alert_new_leaf(idea_id, leaf_name, spec, land_result["runtime_persisted"])
        return self._log_attempt(
            idea_id, hypothesis, rejection_reason, status="approved", spec=spec,
            leaf_name=leaf_name, review_notes=review["notes"], cost_usd=cost_usd,
            generated_file=os.path.join("agents/backtest_engineer/leaves_generated",
                                        f"{leaf_name}.py"),
            test_file=os.path.join("tests", f"test_leaves_generated_{leaf_name}.py"),
            git_commit_sha=commit_sha, module_source=land_result["module_source"])

    # ── Stage 1: PLAN (Opus) ────────────────────────────────────────────────

    def _existing_catalog_summary(self) -> dict:
        from agents.backtest_engineer.signal_dsl import LEAVES
        return {name: (spec.get("shape_card") or "")[:160] for name, spec in LEAVES.items()}

    def _plan(self, hypothesis: str, factor_formula: str, rejection_reason: str,
              catalog: dict) -> dict | None:
        system = f"""You design new DSL "leaf" indicators for a quant backtest engine.

A leaf is a pure function `compute(df, node) -> bool Series` over an OHLCV(+funding_rate)
dataframe. Existing leaves (name: one-line description) — do NOT propose a duplicate:
{json.dumps(catalog, indent=1)}

SCOPE — this is NEW MATH over EXISTING columns only: close, open, high, low, volume,
funding_rate. If the hypothesis genuinely needs data not derivable from these (e.g. an
external index, a second instrument's price series, on-chain data), that is a NEW DATA
SOURCE problem, not a new-leaf problem — set "feasible": false and say why. Do not invent
a proxy or approximation; an honest "infeasible" beats a silently wrong indicator.

Respond with ONLY this JSON shape:
{{"feasible": true/false,
  "reason": "<if infeasible, why — e.g. 'needs external BTC dominance index, not derivable from single-instrument OHLCV'>",
  "leaf_name": "<snake_case, e.g. chaikin_money_flow>",
  "description": "<one line>",
  "required_columns": ["close", ...],
  "params": {{"period": ["int", 5, 100]}},
  "one_of": [["below", ["float", -4, 0]], ["above", ["float", 0, 4]]],
  "compute_description": "<precise plain-English computation, bar by bar>",
  "worked_example": {{
     "input_columns": {{"close": [100, 101, 99, 105, 98, 97, 103]}},
     "node_params": {{"period": 3, "below": -1.0}},
     "expected_output": [false, false, false, true, false, false, false]
  }}}}"""
        try:
            return self.call_claude_json(
                system, [{"role": "user", "content":
                         f"Hypothesis: {hypothesis}\n\nFactor formula text: {factor_formula}\n\n"
                         f"Rejection reason from the parser: {rejection_reason}"}],
                model=MODEL_HEAVY, max_tokens=2048, task_label="leaf_synth_plan")
        except Exception as exc:
            self.log_daemon("WARN", f"Leaf synthesis PLAN stage failed: {exc}")
            return None

    # ── Stage 2: CODE (Haiku) ────────────────────────────────────────────────

    def _implement(self, spec: dict) -> dict | None:
        leaf_name = spec["leaf_name"]
        system = f"""Write a Python DSL leaf compute function, matching this codebase's
exact conventions.

The function signature is EXACTLY:
    def _leaf_{leaf_name}(df, node):
        ...
        return series   # a boolean pandas Series, same index as df

`node` is a dict of the leaf's params/thresholds (e.g. node["period"], node["below"]).
Use only pandas/numpy already imported by the caller — do NOT add your own import
statements inside the function body (they are added by the loader). No file, network,
process, or eval/exec access — pure computation over the `df` argument only. Never
mutate `df` in place (no `df[...] = ...` assignment) — read-only over the input.

Respond with ONLY this JSON shape:
{{"compute_code": "<full function source, starting with 'def _leaf_{leaf_name}(df, node):'>"}}"""
        try:
            return self.call_claude_json(
                system, [{"role": "user", "content": json.dumps(spec)}],
                model=MODEL_FAST, max_tokens=2048, task_label="leaf_synth_code")
        except Exception as exc:
            self.log_daemon("WARN", f"Leaf synthesis CODE stage failed: {exc}")
            return None

    def _generate_reference_impl(self, spec: dict) -> str | None:
        """A SECOND, independent implementation of the same spec — written by
        Sonnet, given ONLY the spec (never shown the CODE-stage implementation)
        — used as a differential-testing oracle instead of trusting the
        PLAN-stage model's own hand-typed worked_example (2026-07-13 finding:
        Opus got a 7-bar rolling-window sum wrong twice live, rejecting
        Haiku's CORRECT code because it disagreed with Opus's bad arithmetic).
        Deliberately prompted for a different implementation STYLE (plain
        loop, not vectorized) to reduce the odds both implementations make
        the same mistake the same way."""
        leaf_name = f"{spec['leaf_name']}_reference"
        system = f"""You write a Python reference implementation for a DSL "leaf" indicator,
used ONLY to cross-check another, independently-written implementation via differential
testing on randomized data — you have NOT been shown that other implementation and must not
try to guess or match it, just implement the spec correctly.

Write a DELIBERATELY SIMPLE, EXPLICIT, LOOP-BASED implementation. Do NOT use pandas
rolling()/vectorized tricks — iterate bar by bar with a plain Python for-loop and explicit
variable bookkeeping, so this reference has a completely different implementation style
from a typical vectorized pandas solution. Prioritize being slow-but-obviously-correct over
being fast or idiomatic. Work through the arithmetic step by step as you reason, since this
implementation IS the ground truth other code will be checked against.

The function signature is EXACTLY:
    def _leaf_{leaf_name}(df, node):
        ...
        return series   # a boolean pandas Series, same index as df

`node` is a dict of the leaf's params/thresholds. `pd` and `np` are ALREADY imported in
the execution namespace — do not add any import statements anywhere in your answer, not
even inside the function body. No file, network, process, or eval/exec access. Never
mutate `df` in place.

Respond with ONLY this JSON shape:
{{"reference_code": "<full function source, starting with 'def _leaf_{leaf_name}(df, node):'>"}}"""
        try:
            result = self.call_claude_json(
                system, [{"role": "user", "content": json.dumps(spec)}],
                model=None, max_tokens=1536, task_label="leaf_synth_reference")
            return result.get("reference_code")
        except Exception as exc:
            self.log_daemon("WARN", f"Leaf synthesis REFERENCE stage failed: {exc}")
            return None

    # ── Stage 3: REVIEW (Sonnet) — safety prose + differential/property testing ──

    def _static_safety_scan(self, code_text: str) -> str | None:
        for token in _BANNED_TOKENS:
            if token in code_text:
                return f"banned token found: {token!r}"
        if not code_text.strip().startswith("def _leaf_"):
            return "code does not start with a def _leaf_<name> declaration"
        return None

    def _pick_node_params(self, spec: dict) -> dict:
        """Fixed params reused across every differential-test trial — only the
        INPUT DATA is randomized, not the parameter space (param fuzzing is
        future work, out of scope for this pass). Prefers the worked_example's
        own node_params (already in-range, chosen by PLAN); falls back to the
        lower bound of each declared param/one_of range if that's missing."""
        node_params = dict((spec.get("worked_example") or {}).get("node_params") or {})
        if node_params:
            return node_params
        for name, bounds in (spec.get("params") or {}).items():
            if isinstance(bounds, list) and len(bounds) >= 3:
                node_params[name] = bounds[1]
        for name, rng_ in (spec.get("one_of") or []):
            if isinstance(rng_, list) and len(rng_) >= 3:
                node_params[name] = rng_[1]
                break
        return node_params

    def _generate_random_dataframes(self, spec: dict, n: int = 20,
                                    seed: int = 20260713) -> list[dict]:
        """Deterministically-seeded synthetic OHLCV(+funding_rate) trial
        inputs for differential testing — pure Python/numpy, NO model call
        involved. This generator is itself part of the trust chain (a buggy
        generator could hide real bugs), so it prints what it built rather
        than silently handing off opaque data."""
        import numpy as np
        rng = np.random.default_rng(seed)
        columns = spec.get("required_columns") or list(_BASE_COLUMNS)
        period = 20
        params = spec.get("params") or {}
        if (isinstance(params.get("period"), list) and len(params["period"]) >= 3):
            period = int(params["period"][1])
        length = max(40, period * 4)

        trials = []
        for i in range(n):
            close = np.maximum(100 + np.cumsum(rng.normal(0, 1.5, length)), 1.0)
            spread = np.abs(rng.normal(1.0, 0.5, length)) + 0.01
            high = close + spread * rng.uniform(0.2, 1.0, length)
            low = np.minimum(close - spread * rng.uniform(0.2, 1.0, length), close - 0.001)
            open_ = low + (high - low) * rng.uniform(0, 1, length)
            volume = np.abs(rng.normal(1000, 400, length))
            funding_rate = rng.normal(0, 0.0005, length)
            # Edge cases sprinkled in periodically: zero volume, high==low bars.
            if i % 5 == 0 and length > 3:
                volume[2] = 0.0
            if i % 7 == 0 and length > 4:
                high[3] = low[3] = close[3]

            data = {}
            for col in columns:
                data[col] = {
                    "close": close, "open": open_, "high": high,
                    "low": low, "volume": volume, "funding_rate": funding_rate,
                }[col].tolist()
            trials.append({"label": f"random_{i}", "input_columns": data,
                           "node_params": self._pick_node_params(spec)})
        print(f"[LeafSynthesizer._generate_random_dataframes] built {n} trials, "
              f"length={length}, columns={columns}, seed={seed}")
        return trials

    def _review_and_test(self, leaf_name: str, spec: dict, code: dict) -> dict:
        static_issue = self._static_safety_scan(code.get("compute_code", ""))
        if static_issue:
            print(f"[LeafSynthesizer._review_and_test] static safety scan REJECTED "
                  f"'{leaf_name}': {static_issue}")
            return {"approved": False, "status": "safety_rejected", "notes": static_issue}

        system = """You are the safety reviewer for auto-generated backtest indicator code.
Given a leaf spec and its generated implementation, check for: (1) lookahead bias — does
the function use any FUTURE bar's data relative to the row it's computing (e.g. .shift(-1),
reversed rolling windows)? Lag is handled by the CALLER, so a leaf itself must not shift
backwards. (2) any network/filesystem/process access. (3) whether this duplicates an
existing leaf's semantics rather than being genuinely new. Respond with ONLY:
{"safety_pass": true/false, "safety_notes": "<why>", "duplicate_of": "<leaf name or null>"}"""
        try:
            verdict = self.call_claude_json(
                system, [{"role": "user", "content": json.dumps({"spec": spec, "code": code})}],
                model=None, max_tokens=1024, task_label="leaf_synth_review")
        except Exception as exc:
            print(f"[LeafSynthesizer._review_and_test] Sonnet safety call FAILED "
                  f"for '{leaf_name}': {exc}")
            return {"approved": False, "status": "review_failed", "notes": str(exc)}

        if not verdict.get("safety_pass") or verdict.get("duplicate_of"):
            print(f"[LeafSynthesizer._review_and_test] Sonnet safety verdict REJECTED "
                  f"'{leaf_name}': {verdict}")
            return {"approved": False, "status": "safety_rejected",
                   "notes": verdict.get("safety_notes", "")}

        reference_code = self._generate_reference_impl(spec)
        if reference_code is None:
            print(f"[LeafSynthesizer._review_and_test] reference implementation stage "
                  f"produced nothing for '{leaf_name}'")
            return {"approved": False, "status": "reference_failed",
                   "notes": "REVIEW stage's independent reference implementation call failed"}
        ref_issue = self._static_safety_scan(reference_code)
        if ref_issue:
            print(f"[LeafSynthesizer._review_and_test] reference impl for '{leaf_name}' "
                  f"failed its OWN safety scan: {ref_issue}")
            return {"approved": False, "status": "reference_failed",
                   "notes": f"reference implementation failed safety scan: {ref_issue}"}

        dt = self._differential_and_property_test(leaf_name, spec, code, reference_code)
        print(f"[LeafSynthesizer._review_and_test] differential/property result for "
              f"'{leaf_name}': property_ok={dt['property_ok']} "
              f"all_trials_agree={dt['all_trials_agree']} "
              f"plan_arithmetic_matches={dt['plan_arithmetic_matches']}")

        if not dt["property_ok"]:
            return {"approved": False, "status": "property_failed", "notes": dt["notes"]}
        if not dt["all_trials_agree"]:
            # Haiku's candidate disagrees with the INDEPENDENT reference
            # implementation — real uncertainty between two implementations,
            # not just PLAN's hand-typed number being wrong. Unsafe to guess.
            return {"approved": False, "status": "test_failed", "notes": dt["notes"]}

        notes = verdict.get("safety_notes", "")
        if dt["plan_arithmetic_matches"] is False:
            # This is the exact 2026-07-13 finding: Haiku's code agrees with
            # an INDEPENDENT reference implementation across every trial
            # (including PLAN's own worked_example input) but PLAN's
            # hand-typed expected_output for that same input was wrong.
            # PLAN's number was never part of the approval decision above —
            # this is purely informational telemetry on PLAN's arithmetic.
            notes += (" | NOTE: PLAN's worked_example hand-typed expected_output did not "
                     "match the verified implementation (Haiku's code agrees with an "
                     "independent reference implementation on this exact input and on "
                     "every randomized trial) — this is a PLAN arithmetic error, not a "
                     "code defect.")
            print(f"[LeafSynthesizer._review_and_test] '{leaf_name}' APPROVED — PLAN's "
                  f"worked_example arithmetic was wrong but the code is correct, confirmed "
                  f"by {dt['agreeing_trial_count']} agreeing trials (Haiku vs independent "
                  f"reference)")
        return {"approved": True, "status": "approved", "notes": notes,
               "confirmed_trial": dt["confirmed_trial"]}

    _DIFFTEST_RUNNER = '''
import sys, json, importlib.util
import pandas as pd


def load_fn(path, modname, funcname):
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, funcname)


def to_bool_list(series):
    return list(series.fillna(False).astype(bool))


def main():
    config = json.load(open(sys.argv[1]))
    haiku_fn = load_fn(config["haiku_module_path"], "_haiku_impl", config["haiku_func_name"])
    ref_fn = load_fn(config["reference_module_path"], "_reference_impl", config["reference_func_name"])

    trial_results = []
    for trial in config["trials"]:
        df = pd.DataFrame(trial["input_columns"])
        node = trial["node_params"]
        entry = {"label": trial["label"], "input_columns": trial["input_columns"],
                 "node_params": trial["node_params"]}
        try:
            haiku_list = to_bool_list(haiku_fn(df, node))
        except Exception as exc:
            print(f"[DIFFTEST] trial {trial['label']}: HAIKU IMPL RAISED: {exc}")
            entry.update(match=False, error=f"haiku raised: {exc}")
            trial_results.append(entry)
            continue
        try:
            ref_list = to_bool_list(ref_fn(df, node))
        except Exception as exc:
            print(f"[DIFFTEST] trial {trial['label']}: REFERENCE IMPL RAISED: {exc}")
            entry.update(match=False, error=f"reference raised: {exc}")
            trial_results.append(entry)
            continue
        match = haiku_list == ref_list
        print(f"[DIFFTEST] trial {trial['label']}: match={match} "
              f"haiku_tail={haiku_list[-5:]} reference_tail={ref_list[-5:]}")
        entry.update(match=match, haiku_output=haiku_list)
        if "plan_expected_output" in trial:
            plan_match = haiku_list == trial["plan_expected_output"]
            entry["plan_match"] = plan_match
            print(f"[DIFFTEST] trial {trial['label']}: PLAN's hand-typed expected_output "
                  f"{'MATCHES' if plan_match else 'DOES NOT MATCH'} the verified output "
                  f"(informational only, does not gate approval)")
        trial_results.append(entry)

    property_failures = []
    sample = config["trials"][1] if len(config["trials"]) > 1 else config["trials"][0]
    sample_df = pd.DataFrame(sample["input_columns"])
    sample_node = sample["node_params"]

    try:
        out = haiku_fn(sample_df.copy(), sample_node)
        if len(out) != len(sample_df):
            property_failures.append("output length != input length")
        elif not out.index.equals(sample_df.index):
            property_failures.append("output index does not match input index")
        else:
            print("[DIFFTEST] property: shape/index preservation OK")
    except Exception as exc:
        property_failures.append(f"shape/index check raised: {exc}")

    try:
        df_before = sample_df.copy(deep=True)
        haiku_fn(sample_df, sample_node)
        if not sample_df.equals(df_before):
            property_failures.append("function mutated its input dataframe")
        else:
            print("[DIFFTEST] property: no-mutation OK")
    except Exception as exc:
        property_failures.append(f"no-mutation check raised: {exc}")

    try:
        # A lookahead bug (e.g. .shift(-1)) can only ever disagree at the
        # SINGLE last row of the truncated prefix (every earlier row's
        # "future" neighbor is still in range either way) -- checking just
        # one k means a real bug has roughly a coin-flip chance of
        # coincidentally landing on the same boolean by luck. Check several
        # k values so that chance becomes negligible instead of a gamble.
        full_list = to_bool_list(haiku_fn(sample_df.copy(), sample_node))
        n = len(sample_df)
        ks = sorted(set(max(2, int(n * frac)) for frac in (0.15, 0.3, 0.5, 0.65, 0.8, 0.95)))
        violations = []
        for k in ks:
            prefix_list = to_bool_list(haiku_fn(sample_df.iloc[:k].copy(), sample_node))
            if prefix_list != full_list[:k]:
                violations.append(k)
        if violations:
            property_failures.append(
                f"prefix-stability violated at k={violations}: computing on the first k "
                f"rows alone gave a different answer than the first k rows of computing "
                f"on the full dataframe -- likely lookahead/future-peeking")
        else:
            print(f"[DIFFTEST] property: prefix-stability OK (checked k={ks})")
    except Exception as exc:
        property_failures.append(f"prefix-stability check raised: {exc}")

    try:
        nan_df = sample_df.copy()
        for col in nan_df.columns:
            nan_df[col] = float("nan")
        haiku_fn(nan_df, sample_node)
        print("[DIFFTEST] property: all-NaN input handled without raising")
    except Exception as exc:
        property_failures.append(
            f"all-NaN input raised an exception instead of returning a safe result: {exc}")

    for f in property_failures:
        print(f"[DIFFTEST] PROPERTY FAILURE: {f}")

    result = {
        "property_ok": len(property_failures) == 0,
        "property_failures": property_failures,
        "trial_results": trial_results,
    }
    print("RESULT_JSON:" + json.dumps(result))


if __name__ == "__main__":
    main()
'''

    def _differential_and_property_test(self, leaf_name: str, spec: dict,
                                        code: dict, reference_code: str) -> dict:
        """Runs entirely in an isolated temp directory (never touches the real
        agents/backtest_engineer/leaves_generated/ or tests/ package dirs) —
        a partial failure here can never leave stray files in the real repo.
        Executes Haiku's compute_code AND Sonnet's independent reference_code
        against N randomized dataframes plus the PLAN-stage worked_example, in
        a single subprocess, and compares outputs. See _DIFFTEST_RUNNER."""
        import tempfile as _tempfile

        def _early_reject(notes: str) -> dict:
            return {"property_ok": False, "all_trials_agree": False,
                   "plan_arithmetic_matches": None, "notes": notes,
                   "confirmed_trial": None, "agreeing_trial_count": 0}

        haiku_func = f"_leaf_{leaf_name}"
        ref_func = f"_leaf_{leaf_name}_reference"
        if f"def {haiku_func}(" not in code.get("compute_code", ""):
            return _early_reject(
                f"compute_code does not define the expected {haiku_func}(df, node)")
        if f"def {ref_func}(" not in reference_code:
            return _early_reject(
                f"reference implementation does not define the expected {ref_func}(df, node)")

        worked_example = spec.get("worked_example") or {}
        trials = []
        if worked_example.get("input_columns") and worked_example.get("node_params"):
            we_trial = {"label": "worked_example",
                       "input_columns": worked_example["input_columns"],
                       "node_params": worked_example["node_params"]}
            if "expected_output" in worked_example:
                we_trial["plan_expected_output"] = worked_example["expected_output"]
            trials.append(we_trial)
        trials.extend(self._generate_random_dataframes(spec, n=20))

        work_dir = _tempfile.mkdtemp(prefix="leaf_synth_difftest_")
        try:
            haiku_path = os.path.join(work_dir, "haiku_impl.py")
            ref_path = os.path.join(work_dir, "reference_impl.py")
            config_path = os.path.join(work_dir, "config.json")
            runner_path = os.path.join(work_dir, "runner.py")

            with open(haiku_path, "w") as fh:
                fh.write(self._module_source(leaf_name, code["compute_code"], {}))
            with open(ref_path, "w") as fh:
                fh.write(self._module_source(f"{leaf_name}_reference", reference_code, {}))
            with open(config_path, "w") as fh:
                json.dump({
                    "haiku_module_path": haiku_path, "reference_module_path": ref_path,
                    "haiku_func_name": haiku_func, "reference_func_name": ref_func,
                    "trials": trials,
                }, fh)
            with open(runner_path, "w") as fh:
                fh.write(self._DIFFTEST_RUNNER)

            proc = subprocess.run(
                [sys.executable, runner_path, config_path],
                cwd=_REPO_ROOT, capture_output=True, text=True, timeout=90,
                env={**os.environ, "PYTHONPATH": _REPO_ROOT,
                    "PYTHONDONTWRITEBYTECODE": "1"})
            print(f"[LeafSynthesizer._differential_and_property_test] subprocess for "
                  f"'{leaf_name}' exit={proc.returncode}\n--- stdout ---\n{proc.stdout}"
                  f"\n--- stderr ---\n{proc.stderr}")

            result_line = next((l for l in proc.stdout.splitlines()
                               if l.startswith("RESULT_JSON:")), None)
            if proc.returncode != 0 or result_line is None:
                return _early_reject(
                    f"differential test subprocess failed (exit={proc.returncode}): "
                    f"{(proc.stdout + proc.stderr)[-800:]}")
            parsed = json.loads(result_line[len("RESULT_JSON:"):])
        except Exception as exc:
            print(f"[LeafSynthesizer._differential_and_property_test] EXCEPTION for "
                  f"'{leaf_name}': {exc}")
            return _early_reject(f"differential test raised: {exc}")
        finally:
            import shutil as _shutil2
            _shutil2.rmtree(work_dir, ignore_errors=True)

        # ALL trials (worked_example if present + every random trial) must
        # show Haiku agreeing with the independent reference — that is the
        # sole differential-testing pass/fail bar. PLAN's own hand-typed
        # expected_output is a SEPARATE, purely informational comparison
        # (plan_arithmetic_matches below) that never gates approval — this is
        # the direct fix for the 2026-07-13 finding.
        trial_results = parsed["trial_results"]
        agreeing = [t for t in trial_results if t["match"]]
        all_trials_agree = len(agreeing) == len(trial_results) and len(trial_results) > 0

        worked_results = [t for t in trial_results if t["label"] == "worked_example"]
        plan_arithmetic_matches = (worked_results[0]["plan_match"]
                                   if worked_results and "plan_match" in worked_results[0]
                                   else None)

        confirmed_trial = agreeing[0] if agreeing else None
        # Prefer the worked_example as the landed regression test's basis
        # when it's among the agreeing trials — more readable for a human
        # reviewing the leaf later than an arbitrary random dataframe.
        for t in agreeing:
            if t["label"] == "worked_example":
                confirmed_trial = t
                break

        notes_parts = []
        if parsed["property_failures"]:
            notes_parts.append("property check failures: " + "; ".join(parsed["property_failures"]))
        if not all_trials_agree:
            mismatched = [t["label"] for t in trial_results if not t["match"]]
            notes_parts.append(f"differential mismatch on trial(s) {mismatched} — Haiku's "
                              f"implementation disagrees with the independent reference "
                              f"implementation")

        return {
            "property_ok": parsed["property_ok"],
            "all_trials_agree": all_trials_agree,
            "plan_arithmetic_matches": plan_arithmetic_matches,
            "notes": " | ".join(notes_parts) or "all checks passed",
            "confirmed_trial": confirmed_trial,
            "agreeing_trial_count": len(agreeing),
        }

    # ── Landing an approved leaf ──────────────────────────────────────────────

    def _module_source(self, leaf_name: str, compute_code: str, spec_extra: dict) -> str:
        return (
            f'"""AI-synthesized DSL leaf — see leaf_synthesis_attempts for the '
            f'approval record."""\n'
            f"import numpy as np\nimport pandas as pd\n\n\n"
            f"{compute_code}\n\n\n"
            f'LEAF_NAME = "{leaf_name}"\n'
            f"LEAF_SPEC = {json.dumps(spec_extra, indent=4)}\n"
        )

    def _build_test_file_source(self, leaf_name: str, confirmed_trial: dict) -> str:
        """System-generated (NOT LLM-authored) — asserts against a value that
        was actually COMPUTED and cross-verified via differential testing,
        never hand-typed by a model. This is the direct fix for the 2026-07-13
        finding: the previous design trusted an LLM's hand-typed
        expected_output as the sole approval oracle, and that was wrong twice
        live. This test can't suffer that failure mode, since its expected
        value came from running (confirmed-correct) code, not from asking a
        model to do mental arithmetic."""
        input_columns = confirmed_trial["input_columns"]
        node_params = confirmed_trial["node_params"]
        expected_output = confirmed_trial["haiku_output"]
        source_label = ("PLAN's worked_example" if confirmed_trial["label"] == "worked_example"
                        else "a randomized differential-test trial")
        return (
            f'"""System-generated regression test for the AI-synthesized leaf '
            f"'{leaf_name}' -- NOT LLM-authored. Expected output was computed by "
            f'running the implementation and cross-verified via differential '
            f'testing against an independent reference implementation (see '
            f'leaf_synthesis_attempts for the approval record); it was never '
            f'hand-typed by a model. Input data source: {source_label}."""\n'
            f"import pandas as pd\n\n"
            f"from agents.backtest_engineer.leaves_generated.{leaf_name} "
            f"import _leaf_{leaf_name}\n\n\n"
            f"def test_confirmed_trial():\n"
            f"    df = pd.DataFrame({input_columns!r})\n"
            f"    node = {node_params!r}\n"
            f"    result = _leaf_{leaf_name}(df, node)\n"
            f"    assert list(result.fillna(False)) == {expected_output!r}\n"
        )

    def _land(self, leaf_name: str, spec: dict, code: dict, confirmed_trial: dict) -> dict:
        # LEAF_SPEC's "compute" must reference the real function object, not a
        # JSON-serializable value — build the module source with repr(), not
        # json.dumps, for that one field.
        columns = spec.get("required_columns", [])
        params = {k: tuple(v) for k, v in (spec.get("params") or {}).items()}
        one_of_repr = (f',\n    "one_of": {[(n, tuple(r)) for n, r in spec["one_of"]]!r}'
                      if spec.get("one_of") else "")
        choices_repr = (f',\n    "choices": {spec["choices"]!r}'
                       if spec.get("choices") else "")
        shape_card = (f"AI-SYNTHESIZED LEAF. {spec.get('description', '')} "
                     f"{spec.get('compute_description', '')}")[:500]
        source = (
            f'"""AI-synthesized DSL leaf — see leaf_synthesis_attempts for the '
            f'approval record (leaf_name={leaf_name!r}).\n'
            f'Spec: {spec.get("description", "")}\n"""\n'
            f"import numpy as np\nimport pandas as pd\n\n\n"
            f"{code['compute_code']}\n\n\n"
            f'LEAF_NAME = "{leaf_name}"\n'
            f"LEAF_SPEC = {{\n"
            f'    "compute": _leaf_{leaf_name},\n'
            f'    "columns": {columns!r},\n'
            f'    "params": {params!r}'
            f"{one_of_repr}{choices_repr},\n"
            f'    "shape_card": {shape_card!r},\n'
            f"}}\n"
        )

        test_source = self._build_test_file_source(leaf_name, confirmed_trial)

        os.makedirs(_GENERATED_DIR, exist_ok=True)
        leaf_path = os.path.join(_GENERATED_DIR, f"{leaf_name}.py")
        test_path = os.path.join(_TESTS_DIR, f"test_leaves_generated_{leaf_name}.py")
        with open(leaf_path, "w") as fh:
            fh.write(source)
        with open(test_path, "w") as fh:
            fh.write(test_source)
        print(f"[LeafSynthesizer._land] landing '{leaf_name}' — test built from "
              f"confirmed_trial label={confirmed_trial['label']!r}")

        # Sanity: the real files, in place, must import and pass too — not
        # just the throwaway temp copy used during review. This test file is
        # now system-generated from an already-verified value, so unlike the
        # old design this should never spuriously fail.
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", test_path, "-q"],
            cwd=_REPO_ROOT, capture_output=True, text=True, timeout=60,
            env={**os.environ, "PYTHONPATH": _REPO_ROOT,
                "PYTHONDONTWRITEBYTECODE": "1"})
        if proc.returncode != 0:
            print(f"[LeafSynthesizer._land] UNEXPECTED post-write test failure for "
                  f"'{leaf_name}' (system-generated test should not fail): "
                  f"{proc.stdout}\n{proc.stderr}")
            os.remove(leaf_path)
            os.remove(test_path)
            raise RuntimeError(f"post-write test failed: {proc.stdout[-500:]}")

        runtime_persisted = self._persist_to_runtime_volume(leaf_name, source, test_source)
        commit_sha = self._git_commit_and_maybe_push(leaf_name, leaf_path, test_path)
        return {"commit_sha": commit_sha, "module_source": source,
               "runtime_persisted": runtime_persisted}

    def _persist_to_runtime_volume(self, leaf_name: str, module_source: str,
                                   test_source: str) -> bool:
        """Dual-write the approved leaf + test to $OPENCLAW_RUNTIME_DIR/leaves_generated/
        — the persistent volume, unlike agents/backtest_engineer/leaves_generated/
        which lives on the image layer and is wiped on the next `docker compose
        build`. Best-effort: a failure here is logged but never blocks approval,
        since the audit table's module_source column is the final fallback."""
        try:
            os.makedirs(_RUNTIME_GENERATED_DIR, exist_ok=True)
            os.makedirs(_RUNTIME_TESTS_DIR, exist_ok=True)
            with open(os.path.join(_RUNTIME_GENERATED_DIR, f"{leaf_name}.py"), "w") as fh:
                fh.write(module_source)
            with open(os.path.join(_RUNTIME_TESTS_DIR,
                                   f"test_leaves_generated_{leaf_name}.py"), "w") as fh:
                fh.write(test_source)
            return True
        except Exception as exc:
            self.log_daemon("WARN", f"Leaf synthesis: failed to persist '{leaf_name}' to "
                                    f"runtime volume ({_RUNTIME_GENERATED_DIR}): {exc}")
            return False

    def _git_commit_and_maybe_push(self, leaf_name: str, leaf_path: str, test_path: str) -> str | None:
        """Opportunistic only — production containers have no git binary, so
        this deliberately no-ops there instead of pretending (previously: a
        caught FileNotFoundError logged as an ambiguous WARN, and the leaf
        would then be silently lost on the next image rebuild since nothing
        else persisted it). Where git IS present this still commits locally
        for convenience, but _persist_to_runtime_volume + the audit table's
        module_source column are what actually guarantee survival now."""
        if shutil.which("git") is None:
            self.log_daemon(
                "INFO", f"Leaf synthesis: no git binary in this environment — "
                        f"'{leaf_name}' not committed; recoverable from the runtime "
                        f"volume or leaf_synthesis_attempts.module_source via "
                        f"scripts/export_synthesized_leaves.py")
            return None
        rel_paths = [os.path.relpath(p, _REPO_ROOT) for p in (leaf_path, test_path)]
        try:
            subprocess.run(["git", "add", *rel_paths], cwd=_REPO_ROOT, check=True,
                           capture_output=True, timeout=30)
            subprocess.run(
                ["git", "-c", "user.name=LeafSynthesizer",
                 "-c", "user.email=leaf-synthesizer@openclaw.local",
                 "commit", "-m", f"Auto-synthesize DSL leaf '{leaf_name}'",
                 "-m", "Co-Authored-By: LeafSynthesizer <noreply@anthropic.com>"],
                cwd=_REPO_ROOT, check=True, capture_output=True, timeout=30)
            sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT,
                                 capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception as exc:
            self.log_daemon("WARN", f"Leaf synthesis git commit failed for "
                                    f"'{leaf_name}' (files kept on disk + runtime volume): {exc}")
            return None
        if LEAF_SYNTH_AUTO_PUSH:
            try:
                subprocess.run(["git", "push", "origin", "HEAD"], cwd=_REPO_ROOT,
                               check=True, capture_output=True, timeout=60)
            except Exception as exc:
                self.log_daemon("WARN", f"Leaf synthesis git push failed for "
                                        f"'{leaf_name}' (committed locally only): {exc}")
        return sha

    def _alert_new_leaf(self, idea_id: int, leaf_name: str, spec: dict,
                        runtime_persisted: bool) -> None:
        from scripts.alerts import send_alert
        persisted_note = ("runtime volume + audit table" if runtime_persisted
                          else "audit table only — runtime volume write FAILED, check logs")
        send_alert(
            f"LeafSynthesizer approved new DSL leaf '{leaf_name}' (idea #{idea_id}): "
            f"{spec.get('description', '')}. Persisted to {persisted_note}. Pull it into "
            f"git manually with scripts/export_synthesized_leaves.py if it should stay.",
            level="WATCH")

    # ── Audit log ────────────────────────────────────────────────────────────

    def _log_attempt(self, idea_id, hypothesis, rejection_reason, *, status,
                     spec=None, leaf_name=None, review_notes=None, cost_usd=0.0,
                     generated_file=None, test_file=None, git_commit_sha=None,
                     module_source=None) -> dict:
        with db_session() as conn:
            conn.execute(
                """INSERT INTO leaf_synthesis_attempts
                     (idea_id, hypothesis, rejection_reason, status, leaf_name,
                      spec_json, generated_file, test_file, review_notes,
                      cost_usd, git_commit_sha, module_source)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (idea_id, (hypothesis or "")[:2000], (rejection_reason or "")[:1000],
                 status, leaf_name, json.dumps(spec) if spec else None,
                 generated_file, test_file, review_notes, round(cost_usd or 0.0, 4),
                 git_commit_sha, module_source))
        return {"status": status, "leaf_name": leaf_name, "cost_usd": round(cost_usd or 0.0, 4)}

    def run(self, task: dict) -> dict:
        return self.synthesize(
            task.get("idea_id"), task.get("hypothesis", ""),
            task.get("factor_formula", ""), task.get("rejection_reason", ""))
