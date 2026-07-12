"""Governance packet dataclasses and schemas."""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


@dataclass
class Finding:
    """A single governance finding from an Inspector.

    Attributes:
        agent: Name of the inspector (e.g. "DataQualityInspector")
        level: L0/L1/L2/L3 inspection level
        scope: Context identifier (e.g. "backtest_run:1234", "idea:567")
        status: PASS/WARN/FAIL
        severity: INFO/WARNING/BLOCKER
        evidence: List of supporting data points (JSON array as string or list)
        local_recommendation: Inspector's suggested action
        escalate_to: Which level (or role) should handle escalation
    """
    agent: str
    level: str
    scope: Optional[str]
    status: str
    severity: str
    evidence: Optional[Any] = None
    local_recommendation: Optional[str] = None
    escalate_to: Optional[str] = None


class DepartmentSummary(BaseModel):
    """High-level status of one research department."""
    department: str = Field(..., description="e.g. 'Strategy Research', 'Backtesting'")
    status: str = Field(..., description="HEALTHY / WARNING / ALERT / BLOCKED")
    findings_count: int = Field(default=0, description="Total findings in last review period")
    blocking_issues: int = Field(default=0, description="Number of BLOCKER-severity findings")
    recommendations: List[str] = Field(default_factory=list, description="Summary actions needed")


class ExecutiveDecisionPacket(BaseModel):
    """Packet for human decision-maker at each gate level.

    Summarizes all L0–L3 findings into a structured brief for a human
    to approve or reject an idea's gate transition.
    """
    idea_id: int = Field(..., description="Alpha idea being evaluated")
    idea_title: str = Field(..., description="Human-readable strategy name")
    stage_from: str = Field(..., description="Current stage (e.g. 'gate0', 'stage2')")
    stage_to: str = Field(..., description="Proposed next stage")

    department_summaries: List[DepartmentSummary] = Field(
        default_factory=list,
        description="Status roll-up per research function"
    )

    critical_findings: List[Finding] = Field(
        default_factory=list,
        description="All BLOCKER-severity findings (L0–L3)"
    )

    warning_findings: List[Finding] = Field(
        default_factory=list,
        description="All WARNING-severity findings (L0–L3)"
    )

    recommendation: str = Field(
        ...,
        description="Consensus recommendation: APPROVE / APPROVE_WITH_CONDITION / REJECT / ESCALATE"
    )

    decision_rationale: str = Field(
        ...,
        description="Why this recommendation (2–3 sentences)"
    )

    conditions_if_conditional: Optional[List[str]] = Field(
        default=None,
        description="If recommendation is APPROVE_WITH_CONDITION, the specific conditions"
    )

    escalation_to: Optional[str] = Field(
        default=None,
        description="If escalating, who should handle it (human role or higher-level system)"
    )


class HumanApprovalRequest(BaseModel):
    """Request for human approval of a gate transition.

    Wraps an ExecutiveDecisionPacket with metadata for routing and tracking.
    """
    model_config = ConfigDict(frozen=False)  # Allows mutable approval tracking

    request_id: str = Field(..., description="Unique ID for tracking (UUID)")
    packet: ExecutiveDecisionPacket = Field(..., description="The decision packet")
    required_approver_role: str = Field(
        ...,
        description="e.g. 'quant_researcher', 'portfolio_manager', 'cto'"
    )
    created_at: str = Field(..., description="ISO 8601 timestamp")
    expires_at: str = Field(..., description="ISO 8601 timestamp when approval is no longer valid")
    context_url: Optional[str] = Field(
        default=None,
        description="Link to idea detail page or dashboard for reference"
    )
