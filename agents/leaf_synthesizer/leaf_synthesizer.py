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
     is instructed to say so, not to fabricate a proxy.
  2. CODE   (MODEL_FAST / Haiku)  — the compute function + a pytest test
     that hand-verifies the spec's worked numeric example.
  3. REVIEW (MODEL_MAIN / Sonnet) — a safety checklist (no lookahead, no
     network/filesystem/exec, no duplicate semantics) PLUS a deterministic
     static scan for banned tokens PLUS actually running the generated
     test in a subprocess. Approval requires ALL THREE — an LLM's say-so
     alone is never sufic ient, since this is the one place in the system
     where model-written code runs unreviewed inside the backtest engine.

Approved leaves are written to agents/backtest_engineer/leaves_generated/
(physically separate from the hand-authored core catalog, for
auditability) and auto-loaded by signal_dsl.py at import time. Every
attempt — approved or not — is logged to leaf_synthesis_attempts.

Git: commits locally on approval always; pushes to origin only if
LEAF_SYNTH_AUTO_PUSH=true (off by default — see config/settings.py for why).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile

from agents.base_agent import BaseAgent, get_agent_daily_spend
from config.settings import (
    MODEL_FAST, MODEL_HEAVY, LEAF_SYNTH_AUTO_PUSH, LEAF_SYNTH_DAILY_BUDGET_USD,
)
from data.database import db_session

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_GENERATED_DIR = os.path.join(_REPO_ROOT, "agents", "backtest_engineer", "leaves_generated")
_TESTS_DIR = os.path.join(_REPO_ROOT, "tests")

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

        commit_sha = self._land(leaf_name, spec, code)
        self.log_daemon(
            "INFO", f"[{idea_id}] Leaf synthesis APPROVED: new leaf '{leaf_name}' "
                    f"written to leaves_generated/, commit={commit_sha or 'local-only'}")
        return self._log_attempt(
            idea_id, hypothesis, rejection_reason, status="approved", spec=spec,
            leaf_name=leaf_name, review_notes=review["notes"], cost_usd=cost_usd,
            generated_file=os.path.join("agents/backtest_engineer/leaves_generated",
                                        f"{leaf_name}.py"),
            test_file=os.path.join("tests", f"test_leaves_generated_{leaf_name}.py"),
            git_commit_sha=commit_sha)

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
        system = f"""Write a Python DSL leaf compute function and its test, matching this
codebase's exact conventions.

The function signature is EXACTLY:
    def _leaf_{leaf_name}(df, node):
        ...
        return series   # a boolean pandas Series, same index as df

`node` is a dict of the leaf's params/thresholds (e.g. node["period"], node["below"]).
Use only pandas/numpy already imported by the caller — do NOT add your own import
statements inside the function body (they are added by the loader). No file, network,
process, or eval/exec access — pure computation over the `df` argument only.

Also write a pytest test function `test_worked_example()` that builds a small DataFrame
from the spec's worked_example.input_columns, calls _leaf_{leaf_name}(df, node) with
worked_example.node_params, and asserts the result matches worked_example.expected_output
exactly (list(result) == expected_output, treating NaN as False).

Respond with ONLY this JSON shape:
{{"compute_code": "<full function source, starting with 'def _leaf_{leaf_name}(df, node):'>",
  "test_code": "<full test file source: imports + def test_worked_example(): ...>"}}"""
        try:
            return self.call_claude_json(
                system, [{"role": "user", "content": json.dumps(spec)}],
                model=MODEL_FAST, max_tokens=2048, task_label="leaf_synth_code")
        except Exception as exc:
            self.log_daemon("WARN", f"Leaf synthesis CODE stage failed: {exc}")
            return None

    # ── Stage 3: REVIEW (Sonnet) + real test execution ───────────────────────

    def _static_safety_scan(self, code: dict) -> str | None:
        blob = code.get("compute_code", "") + "\n" + code.get("test_code", "")
        for token in _BANNED_TOKENS:
            if token in blob:
                return f"banned token found: {token!r}"
        if not code.get("compute_code", "").strip().startswith("def _leaf_"):
            return "compute_code does not start with a def _leaf_<name> declaration"
        return None

    def _review_and_test(self, leaf_name: str, spec: dict, code: dict) -> dict:
        static_issue = self._static_safety_scan(code)
        if static_issue:
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
            return {"approved": False, "status": "review_failed", "notes": str(exc)}

        if not verdict.get("safety_pass") or verdict.get("duplicate_of"):
            return {"approved": False, "status": "safety_rejected",
                   "notes": verdict.get("safety_notes", "")}

        test_result = self._run_generated_test(leaf_name, code)
        if not test_result["passed"]:
            return {"approved": False, "status": "test_failed",
                   "notes": f"generated test failed: {test_result['output'][-500:]}"}

        return {"approved": True, "status": "approved",
               "notes": verdict.get("safety_notes", "")}

    def _run_generated_test(self, leaf_name: str, code: dict) -> dict:
        """Write compute+test to a throwaway temp package under the real repo
        (so `agents.backtest_engineer...` imports resolve) and run pytest on
        just that file, in a subprocess — the test's own PASS/FAIL is a
        deterministic gate, not an LLM's opinion of its own code."""
        tmp_mod = f"_leaf_synth_tmp_{leaf_name}"
        tmp_path = os.path.join(_GENERATED_DIR, f"{tmp_mod}.py")
        tmp_test_path = os.path.join(_TESTS_DIR, f"test_{tmp_mod}.py")
        try:
            with open(tmp_path, "w") as fh:
                fh.write(self._module_source(leaf_name, code["compute_code"], {}))
            test_src = code["test_code"].replace(
                f"leaves_generated.{leaf_name} import", f"leaves_generated.{tmp_mod} import")
            with open(tmp_test_path, "w") as fh:
                fh.write(test_src)
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", tmp_test_path, "-q"],
                cwd=_REPO_ROOT, capture_output=True, text=True, timeout=60,
                env={**os.environ, "PYTHONPATH": _REPO_ROOT,
                    "PYTHONDONTWRITEBYTECODE": "1"})
            return {"passed": proc.returncode == 0, "output": proc.stdout + proc.stderr}
        except Exception as exc:
            return {"passed": False, "output": str(exc)}
        finally:
            for p in (tmp_path, tmp_test_path):
                try:
                    os.remove(p)
                except OSError:
                    pass

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

    def _land(self, leaf_name: str, spec: dict, code: dict) -> str | None:
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

        os.makedirs(_GENERATED_DIR, exist_ok=True)
        leaf_path = os.path.join(_GENERATED_DIR, f"{leaf_name}.py")
        test_path = os.path.join(_TESTS_DIR, f"test_leaves_generated_{leaf_name}.py")
        with open(leaf_path, "w") as fh:
            fh.write(source)
        with open(test_path, "w") as fh:
            fh.write(code["test_code"])

        # Sanity: the real files, in place, must import and pass too — not
        # just the throwaway temp copy used during review.
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", test_path, "-q"],
            cwd=_REPO_ROOT, capture_output=True, text=True, timeout=60,
            env={**os.environ, "PYTHONPATH": _REPO_ROOT,
                "PYTHONDONTWRITEBYTECODE": "1"})
        if proc.returncode != 0:
            os.remove(leaf_path)
            os.remove(test_path)
            raise RuntimeError(f"post-write test failed: {proc.stdout[-500:]}")

        return self._git_commit_and_maybe_push(leaf_name, leaf_path, test_path)

    def _git_commit_and_maybe_push(self, leaf_name: str, leaf_path: str, test_path: str) -> str | None:
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
                                    f"'{leaf_name}' (files kept on disk): {exc}")
            return None
        if LEAF_SYNTH_AUTO_PUSH:
            try:
                subprocess.run(["git", "push", "origin", "HEAD"], cwd=_REPO_ROOT,
                               check=True, capture_output=True, timeout=60)
            except Exception as exc:
                self.log_daemon("WARN", f"Leaf synthesis git push failed for "
                                        f"'{leaf_name}' (committed locally only): {exc}")
        return sha

    # ── Audit log ────────────────────────────────────────────────────────────

    def _log_attempt(self, idea_id, hypothesis, rejection_reason, *, status,
                     spec=None, leaf_name=None, review_notes=None, cost_usd=0.0,
                     generated_file=None, test_file=None, git_commit_sha=None) -> dict:
        with db_session() as conn:
            conn.execute(
                """INSERT INTO leaf_synthesis_attempts
                     (idea_id, hypothesis, rejection_reason, status, leaf_name,
                      spec_json, generated_file, test_file, review_notes,
                      cost_usd, git_commit_sha)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (idea_id, (hypothesis or "")[:2000], (rejection_reason or "")[:1000],
                 status, leaf_name, json.dumps(spec) if spec else None,
                 generated_file, test_file, review_notes, round(cost_usd or 0.0, 4),
                 git_commit_sha))
        return {"status": status, "leaf_name": leaf_name, "cost_usd": round(cost_usd or 0.0, 4)}

    def run(self, task: dict) -> dict:
        return self.synthesize(
            task.get("idea_id"), task.get("hypothesis", ""),
            task.get("factor_formula", ""), task.get("rejection_reason", ""))
