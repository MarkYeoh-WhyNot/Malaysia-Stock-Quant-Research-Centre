import json
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional
import anthropic
from config.settings import ANTHROPIC_API_KEY, MODEL_MAIN, AI_DAILY_BUDGET_USD
from data.database import db_session

logger = logging.getLogger(__name__)


class ClaudeJSONError(RuntimeError):
    """call_claude_json could not parse the model's response as JSON, and the
    caller opted into raise_on_error=True instead of the silent
    {"error": "json_parse_failed"} fallback — used at gate-scoring call sites
    where a silently-swallowed failure previously caused ideas to be scored
    novelty=0.00/logic=0.00 and auto-rejected instead of retried."""

    def __init__(self, message: str, raw: str = ""):
        super().__init__(message)
        self.raw = raw


COST_TABLE = {
    "claude-haiku-4-5-20251001": {"input": 0.80,  "output": 4.00},
    "claude-sonnet-4-6":         {"input": 3.00,  "output": 15.00},
    "claude-opus-4-6":           {"input": 15.00, "output": 75.00},
}

def estimate_cost(model, input_tokens, output_tokens):
    costs = COST_TABLE.get(model, {"input": 3.0, "output": 15.0})
    return (input_tokens * costs["input"] + output_tokens * costs["output"]) / 1_000_000

def get_daily_spend():
    today = datetime.utcnow().strftime("%Y-%m-%d")
    with db_session() as conn:
        row = conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) as total FROM ai_usage WHERE created_at LIKE ?",
            (f"{today}%",)
        ).fetchone()
    return float(row["total"])

def get_agent_daily_spend(agent: str):
    """Today's AI spend for a single agent (e.g. the Concierge sub-cap)."""
    today = datetime.utcnow().strftime("%Y-%m-%d")
    with db_session() as conn:
        row = conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) as total FROM ai_usage "
            "WHERE agent=? AND created_at LIKE ?",
            (agent, f"{today}%"),
        ).fetchone()
    return float(row["total"])

class BaseAgent(ABC):
    name: str = "BaseAgent"
    description: str = ""
    default_model: str = MODEL_MAIN

    def __init__(self):
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.logger = logging.getLogger(f"openclaw.{self.name}")

    def call_claude(self, system, messages, model=None, max_tokens=4096, task_label="", budget_check=True):
        model = model or self.default_model
        if budget_check:
            spend = get_daily_spend()
            if spend >= AI_DAILY_BUDGET_USD:
                if not self._budget_alert_sent_today():
                    from scripts.alerts import send_alert
                    send_alert(f"Daily AI budget cap reached (${spend:.2f} >= ${AI_DAILY_BUDGET_USD:.2f}) "
                              f"— pipeline calls are now blocked until UTC midnight")
                raise RuntimeError(f"Daily AI budget cap reached (${spend:.2f})")
        start = time.time()
        response = self.client.messages.create(
            model=model, max_tokens=max_tokens, system=system, messages=messages
        )
        elapsed = time.time() - start
        input_tokens  = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cost = estimate_cost(model, input_tokens, output_tokens)
        self._log_usage(model, input_tokens, output_tokens, cost, task_label)
        self.logger.debug(f"[{self.name}] {model} | ${cost:.4f} | {elapsed:.1f}s")
        return response.content[0].text

    def call_claude_tools(self, system, messages, tools, tool_handler,
                          model=None, max_tokens=2048, task_label="tools",
                          max_iters=6, budget_check=True):
        """Tool-use conversation loop.

        Runs Claude with `tools`; whenever it emits tool_use blocks, dispatches
        each through `tool_handler(name, input_dict) -> result` (result is
        JSON-serialisable), feeds the results back, and repeats — up to
        `max_iters` rounds to bound cost. Budget + cost tracking mirror
        call_claude(). Mutates and returns via a dict:

          {"text": <final assistant text>,
           "tool_calls": [{"name","input","result"}...],
           "stopped": "end_turn" | "max_iters"}

        `messages` is copied, not mutated in place.
        """
        model = model or self.default_model
        convo = list(messages)
        tool_calls = []

        for _ in range(max_iters):
            if budget_check:
                spend = get_daily_spend()
                if spend >= AI_DAILY_BUDGET_USD:
                    raise RuntimeError(f"Daily AI budget cap reached (${spend:.2f})")
            response = self.client.messages.create(
                model=model, max_tokens=max_tokens, system=system,
                messages=convo, tools=tools,
            )
            cost = estimate_cost(model, response.usage.input_tokens,
                                 response.usage.output_tokens)
            self._log_usage(model, response.usage.input_tokens,
                            response.usage.output_tokens, cost, task_label)

            if response.stop_reason != "tool_use":
                text = "".join(b.text for b in response.content if b.type == "text")
                return {"text": text, "tool_calls": tool_calls, "stopped": "end_turn"}

            # Echo the assistant's tool_use turn, then answer each tool call.
            convo.append({"role": "assistant", "content": response.content})
            results_block = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                try:
                    result = tool_handler(block.name, block.input)
                except Exception as e:
                    result = {"error": f"tool '{block.name}' failed: {e}"}
                tool_calls.append({"name": block.name, "input": block.input,
                                   "result": result})
                results_block.append({
                    "type": "tool_result", "tool_use_id": block.id,
                    "content": json.dumps(result, default=str),
                })
            convo.append({"role": "user", "content": results_block})

        # Hit the iteration cap — ask for a final plain-text summary, no tools.
        response = self.client.messages.create(
            model=model, max_tokens=max_tokens, system=system, messages=convo,
        )
        cost = estimate_cost(model, response.usage.input_tokens,
                             response.usage.output_tokens)
        self._log_usage(model, response.usage.input_tokens,
                        response.usage.output_tokens, cost, task_label)
        text = "".join(b.text for b in response.content if b.type == "text")
        return {"text": text, "tool_calls": tool_calls, "stopped": "max_iters"}

    def call_claude_json(self, system, messages, model=None, max_tokens=4096, task_label="",
                         raise_on_error: bool = False):
        system_with_json = system + "\n\nRespond ONLY with valid JSON. No markdown, no preamble."
        text = self.call_claude(system_with_json, messages, model, max_tokens, task_label)
        raw = text  # preserve original for error reporting
        text = text.strip()
        # Strip any markdown code fences (```json ... ``` or ``` ... ```)
        if text.startswith("```"):
            lines = text.split("\n")
            # Drop first line (``` or ```json) and last line (```)
            inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
            text = "\n".join(inner).strip()
        # Some models wrap JSON in a single-line JSON block without newlines
        # Try to extract the first {...} or [...] substring if direct parse fails
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Fallback: find first { or [ and last } or ] and try parsing that slice
        for start_char, end_char in [('{', '}'), ('[', ']')]:
            start = text.find(start_char)
            end   = text.rfind(end_char)
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    pass
        self.logger.error(
            f"[{self.name}] call_claude_json FAILED for task={task_label!r} — "
            f"could not parse JSON.\nRAW RESPONSE ({len(raw)} chars):\n{raw[:2000]}"
        )
        if raise_on_error:
            raise ClaudeJSONError(
                f"[{self.name}] call_claude_json could not parse a JSON response "
                f"for task={task_label!r}", raw=raw
            )
        return {"error": "json_parse_failed", "raw": raw}

    @staticmethod
    def _budget_alert_sent_today() -> bool:
        """Dedupe the budget-exhausted alert to one per UTC day (job_state
        doubles as a simple once-per-day marker, keyed by date)."""
        today = datetime.utcnow().strftime("%Y-%m-%d")
        job_name = f"budget_alert_{today}"
        with db_session() as conn:
            row = conn.execute(
                "SELECT 1 FROM job_state WHERE job_name=?", (job_name,)
            ).fetchone()
            if row:
                return True
            conn.execute(
                "INSERT INTO job_state (job_name, last_run_utc) VALUES (?, ?)",
                (job_name, datetime.utcnow().isoformat()),
            )
        return False

    def _log_usage(self, model, input_tokens, output_tokens, cost, task):
        with db_session() as conn:
            conn.execute(
                "INSERT INTO ai_usage (model, agent, input_tokens, output_tokens, cost_usd, task) VALUES (?, ?, ?, ?, ?, ?)",
                (model, self.name, input_tokens, output_tokens, cost, task)
            )

    def log_daemon(self, level, message):
        with db_session() as conn:
            conn.execute(
                "INSERT INTO daemon_logs (level, source, message) VALUES (?, ?, ?)",
                (level.upper(), self.name, message)
            )
        log_fn = getattr(self.logger, level.lower(), self.logger.info)
        log_fn(f"[{self.name}] {message}")

    @abstractmethod
    def run(self, task: dict) -> dict:
        ...

    def health_check(self):
        return {"agent": self.name, "status": "ok", "model": self.default_model}
