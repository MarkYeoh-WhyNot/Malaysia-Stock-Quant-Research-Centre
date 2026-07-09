"""Gate calibration harness tests — runs in both market modes.

The harness feeds synthetic series with known properties through the real gate
stack. Two properties must hold in every market:

  * SAFETY (hard): pure random-walk noise is almost never passed. A false
    positive means the pipeline can promote luck as alpha — the dangerous
    failure — so this is asserted tightly.
  * PLUMBING (soft): a genuine mean-reverting edge passes at least sometimes.
    If it never passes, the synthetic-data injection is broken (not a real
    calibration signal). The exact winner rate is a *reported* number, not a
    hard gate — it is expected to be well below 100% because the train/val-gap
    gate rejects genuine edges on Sharpe-estimation noise (see the harness
    diagnosis); that is a known finding, not a test failure.

Kept to 6 seeds so each subprocess stays under ~1 min.
"""
import json
import os
import subprocess
import sys
import tempfile

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_SNIPPET = """
import json
from scripts.calibration_harness import run_calibration
rep = run_calibration(seeds=list(range(1, 7)), n_bars=2000, verbose=False)
print("RESULT " + json.dumps({
    "winner": rep["winner_pass_rate"],
    "loser": rep["loser_pass_rate"],
    "gates": rep["winner_reject_by_gate"],
}))
"""


def _run(market_mode: str) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        env = {**os.environ, "MARKET_MODE": market_mode,
               "OPENCLAW_RUNTIME_DIR": tmp, "PYTHONPATH": REPO}
        r = subprocess.run([sys.executable, "-c", _SNIPPET], env=env, cwd=REPO,
                           capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, f"stderr:\n{r.stderr[-2000:]}"
        line = [l for l in r.stdout.splitlines() if l.startswith("RESULT ")][-1]
        return json.loads(line[len("RESULT "):])


@pytest.mark.parametrize("market_mode", ["bursa", "crypto"])
def test_noise_is_rejected(market_mode):
    """SAFETY: pure random-walk noise must clear the gates ≤ ~1/6 of the time.
    A higher rate means the gate stack produces false positives."""
    res = _run(market_mode)
    assert res["loser"] <= 0.17, (
        f"{market_mode}: pure noise passed {res['loser']:.0%} of trials — "
        f"the gates produce false positives (promoting luck as alpha).")


@pytest.mark.parametrize("market_mode", ["bursa", "crypto"])
def test_genuine_edge_can_pass(market_mode):
    """PLUMBING: a genuine mean-reverting edge must pass at least sometimes.
    0% would mean synthetic-data injection is broken, not a calibration signal."""
    res = _run(market_mode)
    assert res["winner"] > 0.0, (
        f"{market_mode}: genuine edge never passed — harness injection is "
        f"likely broken (rejections: {res['gates']}).")
