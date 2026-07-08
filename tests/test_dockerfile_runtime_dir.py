"""Regression guard for the volume-ownership bug (2026-07-09): a fresh named
volume mounted at a path absent from the image is created root:root by Docker,
crashing the non-root 'openclaw' container on sqlite3.OperationalError. Actual
volume-ownership behavior can only be verified against a running Docker daemon
(done live against the VPS — see DEPLOYMENT notes); this pins the Dockerfile
source so the fix can't silently regress.
"""
import re
from pathlib import Path

DOCKERFILE = Path(__file__).resolve().parent.parent / "Dockerfile"


def test_runtime_dir_is_precreated_and_chowned():
    text = DOCKERFILE.read_text()
    mkdir_lines = [l for l in text.splitlines() if "mkdir" in l and "chown" in l]
    assert mkdir_lines, "expected one RUN line combining mkdir + chown"
    line = mkdir_lines[0]
    assert re.search(r"\bruntime\b", line), (
        "Dockerfile must pre-create+chown 'runtime' — OPENCLAW_RUNTIME_DIR mounts "
        "there, and a path absent from the image gets created root:root on a "
        "fresh named volume mount"
    )
    for d in ("data", "logs", "backups"):
        assert re.search(rf"\b{d}\b", line), f"Dockerfile should still pre-create '{d}'"
    assert "chown -R openclaw:openclaw" in line
