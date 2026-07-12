"""L1 Department Managers — deterministic rollup of L0 inspector findings.

Each department manager aggregates findings from its assigned inspectors over
a lookback window, computes health status (GREEN/AMBER/RED), and surfaces
key recommendations for escalation.
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging

from governance.schemas import DepartmentSummary
from data.database import db_session

logger = logging.getLogger(__name__)


# Mapping of department name → list of inspector agent names (L0 class names)
# that belong to that department. Update this dict when new inspectors land.
DEPARTMENT_AGENT_MAP: Dict[str, List[str]] = {
    "Backtest Fidelity": [
        "PnLConsistencyInspector",
        "FundingCostAuditor",
        "FillConventionAuditor",
        "CostModelAuditor",
        "MetricConsistencyAuditor",
        "RegimeAttributionAuditor",
    ],
    "Parser Honesty": [
        "DSLRepresentabilityChecker",
        "LeafSemanticsAuditor",
        "NegativeMappingGuard",
    ],
    "Portfolio Risk": [
        "ShadowNAVInspector",
        "ConcentrationCorrelationInspector",
        "CapacityAggregationInspector",
        "KillSwitchInspector",
    ],
    "Data Integrity": [
        "SourceHealthInspector",
    ],
    "Paper Trading": [
        # No inspectors landed yet; placeholder for future Fill/Slippage Auditor
    ],
}


def summarize_department(
    department: str, lookback_hours: int = 24
) -> DepartmentSummary:
    """Compute the health status of one department.

    Queries governance_findings for all findings from inspectors assigned to
    this department over the lookback window. Computes:
    - status: GREEN (no findings or all PASS/INFO), AMBER (any WARNING),
      RED (any BLOCKER).
    - findings_count: total findings in the window
    - blocking_issues: count of BLOCKER-severity findings
    - recommendations: dedup list of non-null local_recommendation values
      from WARNING/BLOCKER findings, capped at ~5

    Args:
        department: Department name (must be a key in DEPARTMENT_AGENT_MAP)
        lookback_hours: Window to aggregate over (default 24 hours)

    Returns:
        A DepartmentSummary with status, findings_count, blocking_issues, and
        recommendations populated deterministically from the DB.
    """
    if department not in DEPARTMENT_AGENT_MAP:
        logger.warning(f"[Managers] Unknown department: {department}")
        return DepartmentSummary(
            department=department,
            status="HEALTHY",
            findings_count=0,
            blocking_issues=0,
            recommendations=[],
        )

    agent_names = DEPARTMENT_AGENT_MAP[department]

    with db_session() as conn:
        # Query findings for all inspectors in this department, within lookback
        cutoff = datetime.utcnow() - timedelta(hours=lookback_hours)
        placeholders = ", ".join("?" * len(agent_names))
        query = f"""
        SELECT agent, status, severity, local_recommendation
        FROM governance_findings
        WHERE agent IN ({placeholders})
        AND created_at >= ?
        ORDER BY created_at DESC
        """
        findings = conn.execute(query, agent_names + [cutoff.isoformat()]).fetchall()

    findings_count = len(findings)
    blocking_issues = sum(1 for f in findings if f["severity"] == "BLOCKER")

    # Determine status: worst-child-severity rule
    # - RED if any BLOCKER
    # - AMBER if any WARNING (but no BLOCKER)
    # - GREEN if all PASS/INFO or empty
    has_blocker = any(f["severity"] == "BLOCKER" for f in findings)
    has_warning = any(f["severity"] == "WARNING" for f in findings)

    if has_blocker:
        status = "ALERT"  # RED → ALERT per the mapping
    elif has_warning:
        status = "WARNING"  # AMBER → WARNING per the mapping
    else:
        status = "HEALTHY"  # GREEN → HEALTHY per the mapping

    # Collect recommendations: deduplicate, cap at ~5
    recommendations = []
    seen_recommendations = set()
    for f in findings:
        # sqlite3.Row is dict-like but doesn't have .get() in all versions
        try:
            rec = f["local_recommendation"]
        except (KeyError, TypeError):
            rec = None
        if rec and rec not in seen_recommendations:
            recommendations.append(rec)
            seen_recommendations.add(rec)
            if len(recommendations) >= 5:
                break

    return DepartmentSummary(
        department=department,
        status=status,
        findings_count=findings_count,
        blocking_issues=blocking_issues,
        recommendations=recommendations,
    )


def summarize_all_departments(lookback_hours: int = 24) -> List[DepartmentSummary]:
    """Compute health status for all departments.

    Runs summarize_department() for each department in DEPARTMENT_AGENT_MAP
    and returns the list. Gracefully handles departments with zero findings
    (returns GREEN/HEALTHY status).

    Args:
        lookback_hours: Window to aggregate over (default 24 hours)

    Returns:
        List of DepartmentSummary objects, one per department, sorted by
        department name.
    """
    summaries = []
    for department in sorted(DEPARTMENT_AGENT_MAP.keys()):
        summary = summarize_department(department, lookback_hours=lookback_hours)
        summaries.append(summary)
    return summaries


def get_all_inspector_names() -> List[str]:
    """Return a flat list of all inspector class names across all departments.

    Useful for validating that new inspectors are registered in
    DEPARTMENT_AGENT_MAP.
    """
    all_names = []
    for inspectors in DEPARTMENT_AGENT_MAP.values():
        all_names.extend(inspectors)
    return all_names
