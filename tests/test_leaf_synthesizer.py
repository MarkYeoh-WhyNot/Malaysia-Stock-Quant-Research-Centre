"""LeafSynthesizer: turns a genuinely-unrepresentable formula into a new,
tested DSL leaf instead of a permanent dead end. All three model-call stages
(plan/code/review) are monkeypatched — this suite tests the ORCHESTRATION
(routing, validation, the deterministic safety scan, real test execution,
file landing, loadability) not real LLM behavior. Git commit is monkeypatched
out entirely — this suite must never touch the real repo's git history.

House pattern: share the local DB, isolate by idea_id / leaf_name prefix,
clean up before + after (including any files actually written to
leaves_generated/ and tests/ by the "approved" path).
"""
import os
import shutil

import pytest

import agents.leaf_synthesizer.leaf_synthesizer as leaf_synthesizer_mod
from agents.leaf_synthesizer.leaf_synthesizer import LeafSynthesizer, _GENERATED_DIR, _TESTS_DIR
from data.database import db_session, init_db

_IDEA_ID = 999100001
_LEAF_NAME = "test_probe_leaf"

_PLAN_FEASIBLE = {
    "feasible": True,
    "leaf_name": _LEAF_NAME,
    "description": "close below its immediately prior close",
    "required_columns": ["close"],
    "params": {"period": [3, 2, 50]},
    "compute_description": "true when close[t] < close[t-1]",
    "worked_example": {
        "input_columns": {"close": [10, 9, 11, 8, 12]},
        "node_params": {"period": 3},
        "expected_output": [False, True, False, True, False],
    },
}

_CODE_OK = {
    "compute_code": (
        f"def _leaf_{_LEAF_NAME}(df, node):\n"
        f"    return df[\"close\"] < df[\"close\"].shift(1)\n"
    ),
    "test_code": (
        "import pandas as pd\n"
        f"from agents.backtest_engineer.leaves_generated.{_LEAF_NAME} import _leaf_{_LEAF_NAME}\n\n"
        "def test_worked_example():\n"
        "    df = pd.DataFrame({\"close\": [10, 9, 11, 8, 12]})\n"
        f"    result = _leaf_{_LEAF_NAME}(df, {{\"period\": 3}})\n"
        "    assert list(result.fillna(False)) == [False, True, False, True, False]\n"
    ),
}

_CODE_BUGGY = {
    "compute_code": (
        f"def _leaf_{_LEAF_NAME}(df, node):\n"
        f"    return df[\"close\"] > df[\"close\"].shift(1)\n"  # flipped, wrong
    ),
    "test_code": _CODE_OK["test_code"],
}

_REVIEW_OK = {"safety_pass": True, "safety_notes": "no lookahead, pure function",
              "duplicate_of": None}
_REVIEW_UNSAFE = {"safety_pass": False, "safety_notes": "uses future data", "duplicate_of": None}


def _purge():
    with db_session() as conn:
        conn.execute("DELETE FROM leaf_synthesis_attempts WHERE idea_id=?", (_IDEA_ID,))
        conn.execute("DELETE FROM alpha_ideas WHERE id=?", (_IDEA_ID,))
    # Bytecode caches for these dynamically-written modules must be cleaned
    # too — a stale .pyc from a PREVIOUS test's (working) code can silently
    # outlive the deleted .py source and get reused for a LATER test's
    # (deliberately buggy) rewrite of the same module name, since
    # PYTHONDONTWRITEBYTECODE only prevents the leaf_synthesizer's OWN
    # subprocess runs from writing new ones, not this process's imports
    # (e.g. the happy-path test's importlib.reload).
    # Also sweep the default (unpatched) runtime-volume dirs — belt-and-
    # suspenders for any test that lands a real leaf without monkeypatching
    # leaf_synthesizer_mod._RUNTIME_GENERATED_DIR/_RUNTIME_TESTS_DIR, so a
    # stray file there can never leak into the next `pytest` collection
    # (data/leaves_generated/ matches the test_*.py discovery pattern).
    dirs = [_GENERATED_DIR, _TESTS_DIR,
           leaf_synthesizer_mod._RUNTIME_GENERATED_DIR,
           leaf_synthesizer_mod._RUNTIME_TESTS_DIR]
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for fname in os.listdir(d):
            if _LEAF_NAME in fname:
                try:
                    os.remove(os.path.join(d, fname))
                except OSError:
                    pass
        pycache = os.path.join(d, "__pycache__")
        if os.path.isdir(pycache):
            for fname in os.listdir(pycache):
                if _LEAF_NAME in fname:
                    try:
                        os.remove(os.path.join(pycache, fname))
                    except OSError:
                        pass


@pytest.fixture(autouse=True)
def _setup():
    init_db()
    _purge()
    with db_session() as conn:
        conn.execute(
            "INSERT INTO alpha_ideas (id, slug, title, stage, status) "
            "VALUES (?, 'test-leaf-synth', 'test', 'stage2', 'rejected')", (_IDEA_ID,))
    yield
    _purge()


def _patch_stage(monkeypatch, plan=None, code=None, review=None, plan_calls=None):
    """Route call_claude_json by task_label to canned per-stage responses."""
    def fake(self, system, messages, model=None, max_tokens=4096, task_label="",
            raise_on_error=False):
        if plan_calls is not None:
            plan_calls.append(task_label)
        if task_label == "leaf_synth_plan":
            if plan is None:
                raise AssertionError("plan stage should not have been called")
            return plan
        if task_label == "leaf_synth_code":
            if code is None:
                raise AssertionError("code stage should not have been called")
            return code
        if task_label == "leaf_synth_review":
            if review is None:
                raise AssertionError("review stage should not have been called")
            return review
        raise AssertionError(f"unexpected task_label {task_label!r}")
    monkeypatch.setattr(LeafSynthesizer, "call_claude_json", fake)
    monkeypatch.setattr(LeafSynthesizer, "_git_commit_and_maybe_push",
                        lambda self, *a, **kw: "fake-sha-0000000")


def _last_attempt():
    with db_session() as conn:
        return conn.execute(
            "SELECT * FROM leaf_synthesis_attempts WHERE idea_id=? "
            "ORDER BY id DESC LIMIT 1", (_IDEA_ID,)).fetchone()


# ── happy path ───────────────────────────────────────────────────────────────

def test_full_pipeline_approves_lands_and_loads_the_new_leaf(monkeypatch, tmp_path):
    # Redirect the runtime-volume dual-write so this test never touches the
    # real repo's default runtime dir (data/leaves_generated/).
    monkeypatch.setattr(leaf_synthesizer_mod, "_RUNTIME_GENERATED_DIR", str(tmp_path / "leaves_generated"))
    monkeypatch.setattr(leaf_synthesizer_mod, "_RUNTIME_TESTS_DIR", str(tmp_path / "leaves_generated" / "tests"))
    _patch_stage(monkeypatch, plan=_PLAN_FEASIBLE, code=_CODE_OK, review=_REVIEW_OK)
    result = LeafSynthesizer().synthesize(
        _IDEA_ID, "test hypothesis", "test formula", "not representable")

    assert result["status"] == "approved"
    assert result["leaf_name"] == _LEAF_NAME

    leaf_path = os.path.join(_GENERATED_DIR, f"{_LEAF_NAME}.py")
    test_path = os.path.join(_TESTS_DIR, f"test_leaves_generated_{_LEAF_NAME}.py")
    assert os.path.exists(leaf_path)
    assert os.path.exists(test_path)

    row = _last_attempt()
    assert row["status"] == "approved"
    assert row["leaf_name"] == _LEAF_NAME
    assert row["git_commit_sha"] == "fake-sha-0000000"

    # signal_dsl's loader must pick it up on a fresh import
    import importlib
    import agents.backtest_engineer.signal_dsl as signal_dsl
    importlib.reload(signal_dsl)
    try:
        assert _LEAF_NAME in signal_dsl.LEAVES
        assert signal_dsl.LEAVES[_LEAF_NAME]["columns"] == ["close"]
    finally:
        _purge()
        importlib.reload(signal_dsl)  # restore the module for subsequent tests


# ── budget ───────────────────────────────────────────────────────────────────

def test_budget_exceeded_skips_all_llm_calls(monkeypatch):
    monkeypatch.setattr(
        "agents.leaf_synthesizer.leaf_synthesizer.get_agent_daily_spend",
        lambda agent: 999.0)

    def fail_if_called(self, *a, **kw):
        raise AssertionError("no LLM call should happen when budget is exceeded")
    monkeypatch.setattr(LeafSynthesizer, "call_claude_json", fail_if_called)

    result = LeafSynthesizer().synthesize(_IDEA_ID, "h", "f", "r")
    assert result["status"] == "budget_exceeded"
    assert _last_attempt()["status"] == "budget_exceeded"


# ── plan stage ───────────────────────────────────────────────────────────────

def test_infeasible_plan_stops_before_code_or_review(monkeypatch):
    calls = []
    _patch_stage(monkeypatch, plan={"feasible": False,
                                    "reason": "needs an external index series"},
                plan_calls=calls)
    result = LeafSynthesizer().synthesize(_IDEA_ID, "h", "f", "r")
    assert result["status"] == "infeasible"
    assert calls == ["leaf_synth_plan"]


def test_plan_with_duplicate_leaf_name_is_rejected(monkeypatch):
    dup_plan = dict(_PLAN_FEASIBLE, leaf_name="rsi")  # already in the hand-authored catalog
    _patch_stage(monkeypatch, plan=dup_plan)
    result = LeafSynthesizer().synthesize(_IDEA_ID, "h", "f", "r")
    assert result["status"] == "plan_invalid"


def test_plan_requiring_non_base_columns_is_out_of_scope(monkeypatch):
    plan = dict(_PLAN_FEASIBLE, required_columns=["btc_dominance_index"])
    _patch_stage(monkeypatch, plan=plan)
    result = LeafSynthesizer().synthesize(_IDEA_ID, "h", "f", "r")
    assert result["status"] == "infeasible"


# ── code / review stage ──────────────────────────────────────────────────────

def test_banned_token_in_generated_code_is_rejected_without_calling_review(monkeypatch):
    unsafe_code = {
        "compute_code": (
            f"def _leaf_{_LEAF_NAME}(df, node):\n"
            f"    import os\n    os.system('echo hi')\n"
            f"    return df[\"close\"] < df[\"close\"].shift(1)\n"
        ),
        "test_code": _CODE_OK["test_code"],
    }
    calls = []
    _patch_stage(monkeypatch, plan=_PLAN_FEASIBLE, code=unsafe_code, plan_calls=calls)
    result = LeafSynthesizer().synthesize(_IDEA_ID, "h", "f", "r")
    assert result["status"] == "safety_rejected"
    assert "leaf_synth_review" not in calls
    assert not os.path.exists(os.path.join(_GENERATED_DIR, f"{_LEAF_NAME}.py"))


def test_sonnet_safety_fail_rejects_even_if_test_would_pass(monkeypatch):
    _patch_stage(monkeypatch, plan=_PLAN_FEASIBLE, code=_CODE_OK, review=_REVIEW_UNSAFE)
    result = LeafSynthesizer().synthesize(_IDEA_ID, "h", "f", "r")
    assert result["status"] == "safety_rejected"
    assert not os.path.exists(os.path.join(_GENERATED_DIR, f"{_LEAF_NAME}.py"))


def test_failing_generated_test_blocks_approval_even_if_sonnet_approves(monkeypatch):
    """The deterministic test-execution gate, not the LLM's safety verdict,
    has final say — a wrong compute function must never land."""
    _patch_stage(monkeypatch, plan=_PLAN_FEASIBLE, code=_CODE_BUGGY, review=_REVIEW_OK)
    result = LeafSynthesizer().synthesize(_IDEA_ID, "h", "f", "r")
    assert result["status"] == "test_failed"
    assert not os.path.exists(os.path.join(_GENERATED_DIR, f"{_LEAF_NAME}.py"))


# ── P0-1 persistence (2026-07-13 self-audit): production containers have no
# git binary and /app is an ephemeral image layer, so approved leaves must
# survive via the runtime volume + audit table, not git. ──────────────────────

def test_approved_leaf_dual_written_to_runtime_volume_and_audit_table(monkeypatch, tmp_path):
    runtime_generated = tmp_path / "leaves_generated"
    runtime_tests = runtime_generated / "tests"
    monkeypatch.setattr(leaf_synthesizer_mod, "_RUNTIME_GENERATED_DIR", str(runtime_generated))
    monkeypatch.setattr(leaf_synthesizer_mod, "_RUNTIME_TESTS_DIR", str(runtime_tests))
    _patch_stage(monkeypatch, plan=_PLAN_FEASIBLE, code=_CODE_OK, review=_REVIEW_OK)

    result = LeafSynthesizer().synthesize(
        _IDEA_ID, "test hypothesis", "test formula", "not representable")
    assert result["status"] == "approved"

    assert (runtime_generated / f"{_LEAF_NAME}.py").exists()
    assert (runtime_tests / f"test_leaves_generated_{_LEAF_NAME}.py").exists()

    row = _last_attempt()
    assert row["module_source"]
    assert f"_leaf_{_LEAF_NAME}" in row["module_source"]


def test_runtime_volume_write_failure_does_not_block_approval(monkeypatch, tmp_path):
    """The runtime volume is best-effort — if it's unwritable for some reason,
    the audit table's module_source is the fallback of last resort, so
    approval must still succeed rather than crash the caller."""
    unwritable = tmp_path / "not_a_dir"
    unwritable.write_text("blocking a directory from being created here")
    monkeypatch.setattr(leaf_synthesizer_mod, "_RUNTIME_GENERATED_DIR", str(unwritable / "leaves_generated"))
    monkeypatch.setattr(leaf_synthesizer_mod, "_RUNTIME_TESTS_DIR", str(unwritable / "leaves_generated" / "tests"))
    _patch_stage(monkeypatch, plan=_PLAN_FEASIBLE, code=_CODE_OK, review=_REVIEW_OK)

    result = LeafSynthesizer().synthesize(_IDEA_ID, "h", "f", "r")
    assert result["status"] == "approved"
    row = _last_attempt()
    assert row["module_source"]  # audit table still has the source


def test_git_commit_skips_cleanly_when_no_git_binary(monkeypatch):
    """Production containers have no git binary — this must be a deliberate,
    logged no-op, never a caught exception masquerading as one."""
    monkeypatch.setattr(shutil, "which", lambda name: None)
    sha = LeafSynthesizer()._git_commit_and_maybe_push(
        _LEAF_NAME, "/tmp/fake_leaf.py", "/tmp/fake_test.py")
    assert sha is None


def test_signal_dsl_loads_leaf_from_runtime_volume(tmp_path, monkeypatch):
    """After a docker rebuild wipes agents/backtest_engineer/leaves_generated/
    (image layer), the runtime volume copy must still be loadable."""
    runtime_generated = tmp_path / "leaves_generated"
    runtime_generated.mkdir()
    (runtime_generated / "probe_runtime_leaf.py").write_text(
        'def _leaf_probe_runtime_leaf(df, node):\n'
        '    return df["close"] > 0\n\n'
        'LEAF_NAME = "probe_runtime_leaf"\n'
        'LEAF_SPEC = {"compute": _leaf_probe_runtime_leaf, "columns": ["close"], '
        '"params": {}, "shape_card": "probe"}\n'
    )
    monkeypatch.setenv("OPENCLAW_RUNTIME_DIR", str(tmp_path))

    from agents.backtest_engineer.signal_dsl import _load_generated_leaves
    generated = _load_generated_leaves()
    assert "probe_runtime_leaf" in generated
    assert generated["probe_runtime_leaf"]["columns"] == ["close"]


def test_signal_dsl_runtime_volume_overrides_package_dir_on_name_clash(tmp_path, monkeypatch):
    """Per the plan: when the same leaf name exists in both places, the
    runtime volume — the real source of truth in prod — wins."""
    runtime_generated = tmp_path / "leaves_generated"
    runtime_generated.mkdir()
    (runtime_generated / f"{_LEAF_NAME}.py").write_text(
        f'def _leaf_{_LEAF_NAME}(df, node):\n'
        f'    return df["close"] > 0\n\n'
        f'LEAF_NAME = "{_LEAF_NAME}"\n'
        f'LEAF_SPEC = {{"compute": _leaf_{_LEAF_NAME}, "columns": ["close"], '
        f'"params": {{}}, "shape_card": "from runtime volume"}}\n'
    )
    monkeypatch.setenv("OPENCLAW_RUNTIME_DIR", str(tmp_path))
    package_leaf_path = os.path.join(_GENERATED_DIR, f"{_LEAF_NAME}.py")
    with open(package_leaf_path, "w") as fh:
        fh.write(
            f'def _leaf_{_LEAF_NAME}(df, node):\n'
            f'    return df["close"] < 0\n\n'
            f'LEAF_NAME = "{_LEAF_NAME}"\n'
            f'LEAF_SPEC = {{"compute": _leaf_{_LEAF_NAME}, "columns": ["close"], '
            f'"params": {{}}, "shape_card": "from package dir"}}\n'
        )
    try:
        from agents.backtest_engineer.signal_dsl import _load_generated_leaves
        generated = _load_generated_leaves()
        assert generated[_LEAF_NAME]["shape_card"] == "from runtime volume"
    finally:
        os.remove(package_leaf_path)


# ── P1-2 real dry-run finding (2026-07-13): the production image has no
# pytest, so REVIEW's `python -m pytest` subprocess silently dead-ended every
# real synthesis attempt at status=test_failed with "No module named pytest"
# — invisible to this suite since it always ran inside a dev venv that
# already had pytest. Pin the fix source so it can't silently regress. ──────

def test_pytest_is_pinned_as_a_production_dependency():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lock_text = open(os.path.join(repo_root, "requirements.lock")).read()
    assert any(line.strip().lower().startswith("pytest==")
              for line in lock_text.splitlines()), (
        "requirements.lock must pin pytest — LeafSynthesizer's REVIEW stage "
        "runs generated tests via `python -m pytest` in the deployed image, "
        "not just in a dev venv")
