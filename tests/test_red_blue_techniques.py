"""Red team ← Technique Arsenal wiring, and market-aware debate-prompt noun."""
from unittest.mock import patch

from agents.red_blue_team.red_blue_team import RedBlueTeam, _STRATEGY_NOUN


def _capture_red_prompt(idea):
    rb = RedBlueTeam()
    captured = {}

    def fake_json(system, messages, **kw):
        captured["prompt"] = messages[0]["content"]
        return {"critical_flaws": [], "overall_attack_score": 0.5,
                "kill_recommendation": False, "kill_rationale": "x"}

    with patch.object(rb, "call_claude_json", side_effect=fake_json), \
         patch.object(rb, "log_daemon"), \
         patch.object(rb, "_failure_knowledge", return_value=""):
        rb.red_team_attack(idea, {})
    return captured["prompt"]


def test_red_prompt_carries_technique_caveats_when_idea_uses_one():
    prompt = _capture_red_prompt({
        "title": "Kalman smoothing momentum",
        "hypothesis": "use a kalman filter to smooth the trend signal",
        "ticker": "1155.KL", "timeframe": "1d",
        "factor_formula": "kalman filter smoothed close above sma(50)",
    })
    assert "TECHNIQUE CAVEATS" in prompt
    assert "kalman_filter" in prompt
    assert "avoid when:" in prompt


def test_red_prompt_has_no_caveat_block_without_a_match():
    prompt = _capture_red_prompt({
        "title": "Plain MA cross", "hypothesis": "simple trend following",
        "ticker": "1155.KL", "timeframe": "1d",
        "factor_formula": "close crosses above sma(50)",
    })
    assert "TECHNIQUE CAVEATS" not in prompt


def test_bursa_debate_noun_pinned():
    # Bursa (default mode) must keep its historical prompt wording verbatim.
    assert _STRATEGY_NOUN == "Bursa Malaysia equity strategy"
    prompt = _capture_red_prompt({
        "title": "Plain MA cross", "hypothesis": "simple trend following",
        "ticker": "1155.KL", "timeframe": "1d",
        "factor_formula": "close crosses above sma(50)",
    })
    assert "Stress-test this Bursa Malaysia equity strategy" in prompt
