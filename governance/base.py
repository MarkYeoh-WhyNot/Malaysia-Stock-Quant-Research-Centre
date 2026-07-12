"""Base class for governance inspectors at all levels (L0–L3)."""

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Dict, Any
from governance.schemas import Finding
from data.database import db_session

logger = logging.getLogger(__name__)


class Inspector(ABC):
    """Abstract base class for governance inspection at a single level.

    Each Inspector subclass implements `inspect(scope, ctx)` to run one specific
    check and return a Finding (or None if the check doesn't apply). The base
    class handles database persistence via `record()`.
    """

    name: str = "BaseInspector"
    level: str = "L0"  # L0/L1/L2/L3

    def inspect(self, scope: str, ctx: Dict[str, Any]) -> Optional[Finding]:
        """Run the inspection and return a Finding, or None if not applicable.

        Args:
            scope: Context identifier (e.g. "backtest_run:1234", "idea:567")
            ctx: Dictionary of context data needed for the check

        Returns:
            A Finding object, or None if the check doesn't apply
        """
        raise NotImplementedError

    def record(self, finding: Finding) -> int:
        """Write a Finding to governance_findings and return the row ID.

        State-change-only: if the most recent row for this (agent, scope) has
        the same status/severity/evidence/recommendation/escalation, no new
        row is inserted and that row's id is returned instead. Inspectors run
        every daemon cycle (~60s) regardless of whether the verdict changed,
        so without this the table (and the finding-node KB graph it feeds via
        knowledge/ingestion/evidence_graph.py) grows unbounded with identical
        repeats.

        Args:
            finding: The Finding to persist

        Returns:
            The row ID from governance_findings (newly inserted, or the
            existing unchanged row if this is a repeat of the last state)
        """
        evidence_json = None
        if finding.evidence is not None:
            if isinstance(finding.evidence, (list, dict)):
                evidence_json = json.dumps(finding.evidence)
            else:
                evidence_json = str(finding.evidence)

        with db_session() as conn:
            prev = conn.execute(
                """
                SELECT id, status, severity, evidence, local_recommendation, escalate_to
                FROM governance_findings
                WHERE agent = ? AND scope IS ?
                ORDER BY id DESC LIMIT 1
                """,
                (finding.agent, finding.scope),
            ).fetchone()
            if prev is not None and (
                prev["status"] == finding.status
                and prev["severity"] == finding.severity
                and prev["evidence"] == evidence_json
                and prev["local_recommendation"] == finding.local_recommendation
                and prev["escalate_to"] == finding.escalate_to
            ):
                self.logger.debug(
                    f"Unchanged finding for {finding.agent}/{finding.level} "
                    f"{finding.scope} {finding.status}/{finding.severity} — "
                    f"skipping duplicate row (reusing id={prev['id']})"
                )
                return prev["id"]

            cursor = conn.execute(
                """
                INSERT INTO governance_findings
                  (agent, level, scope, status, severity, evidence,
                   local_recommendation, escalate_to)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    finding.agent,
                    finding.level,
                    finding.scope,
                    finding.status,
                    finding.severity,
                    evidence_json,
                    finding.local_recommendation,
                    finding.escalate_to,
                ),
            )
        row_id = cursor.lastrowid
        self.logger.debug(
            f"Recorded finding id={row_id}: {finding.agent}/{finding.level} "
            f"{finding.scope} {finding.status}/{finding.severity}"
        )
        return row_id

    def __init__(self):
        self.logger = logging.getLogger(f"governance.{self.name}")
