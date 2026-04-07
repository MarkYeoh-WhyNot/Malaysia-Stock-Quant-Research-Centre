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

    def call_claude_json(self, system, messages, model=None, max_tokens=4096, task_label=""):
        system_with_json = system + "\n\nRespond ONLY with valid JSON. No markdown, no preamble."
        text = self.call_claude(system_with_json, messages, model, max_tokens, task_label)
        text = text.strip()
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:-1])
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parse failed: {e}")
            return {"error": str(e), "raw": text}

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
