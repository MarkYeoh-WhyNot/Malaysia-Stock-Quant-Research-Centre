"""
Governance framework for multi-level inspection and finding escalation.

Levels:
- L0: Deterministic data quality checks (no LLM)
- L1: Single-idea validation (basic sanity)
- L2: Cross-sectional consistency (portfolio coherence)
- L3: System-wide audit (budget, capacity, policy)
"""

from governance.base import Inspector, Finding
from governance.schemas import (
    Finding,
    DepartmentSummary,
    ExecutiveDecisionPacket,
    HumanApprovalRequest,
)

__all__ = [
    "Inspector",
    "Finding",
    "DepartmentSummary",
    "ExecutiveDecisionPacket",
    "HumanApprovalRequest",
]
