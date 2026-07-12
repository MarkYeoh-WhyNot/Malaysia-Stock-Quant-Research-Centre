"""Leaf Semantics Auditor — verifies signal_dsl.LEAVES registry completeness and validation.

Checks:
1. Every leaf has a shape_card (required for parser prompt)
2. Leaves with required_choices enforce validation at parse time
   (e.g., ma_level's ma_type is REQUIRED; a tree missing it fails validate())
3. No silent defaults — if a required choice is missing, validation must reject it
"""

import logging
from typing import Optional, Dict, Any
from governance.base import Inspector
from governance.schemas import Finding
from agents.backtest_engineer import signal_dsl

logger = logging.getLogger(__name__)


class LeafSemanticsAuditor(Inspector):
    """Audits the signal_dsl.LEAVES registry for completeness and enforcement."""

    name = "LeafSemanticsAuditor"
    level = "L0"

    def inspect(self, scope: str, ctx: Dict[str, Any]) -> Optional[Finding]:
        """
        Run the inspection across all LEAVES in signal_dsl.

        Args:
            scope: Should be "signal_dsl:leaves" or similar
            ctx: Not required for this check

        Returns:
            A Finding with status PASS or BLOCKER
        """
        issues = []

        # Check 1: Every leaf has a shape_card
        try:
            shape_cards = signal_dsl.shape_cards_text()
        except KeyError as e:
            issues.append(f"shape_cards_text() KeyError: {e} — a leaf is missing 'shape_card'")

        # Check 2: Iterate LEAVES and verify shape_card exists
        for leaf_name, leaf_spec in signal_dsl.LEAVES.items():
            if "shape_card" not in leaf_spec:
                issues.append(
                    f"Leaf '{leaf_name}' missing 'shape_card' in LEAVES registry"
                )

        # Check 3: For leaves with required_choices, verify validation enforces them
        for leaf_name, leaf_spec in signal_dsl.LEAVES.items():
            required_choices = leaf_spec.get("required_choices", [])
            if required_choices:
                # Build a malformed tree that OMITS the required choice
                test_tree = self._build_malformed_tree(leaf_name, required_choices)
                if test_tree is not None:
                    errors = signal_dsl.validate(test_tree)
                    # Validation MUST catch the missing required choice
                    has_required_choice_error = any(
                        f"missing required choice" in err for err in errors
                    )
                    if not has_required_choice_error:
                        issues.append(
                            f"Leaf '{leaf_name}' has required_choices={required_choices}, "
                            f"but validate() does NOT reject a tree missing {required_choices[0]}. "
                            f"Validation errors: {errors}"
                        )

        # Compile findings
        if issues:
            severity = "BLOCKER"
            status = "FAIL"
            evidence = issues
            recommendation = "Fix the LEAVES registry: add missing shape_cards and verify required_choices validation"
            finding = Finding(
                agent=self.name,
                level=self.level,
                scope=scope,
                status=status,
                severity=severity,
                evidence=evidence,
                local_recommendation=recommendation,
            )
        else:
            severity = "INFO"
            status = "PASS"
            evidence = [
                f"{len(signal_dsl.LEAVES)} leaves registered",
                "All leaves have shape_card",
                f"{sum(1 for s in signal_dsl.LEAVES.values() if 'required_choices' in s)} leaves with required_choices",
                "All required_choices are enforced by validate()",
            ]
            finding = Finding(
                agent=self.name,
                level=self.level,
                scope=scope,
                status=status,
                severity=severity,
                evidence=evidence,
            )

        return finding

    def _build_malformed_tree(
        self, leaf_name: str, required_choices: list
    ) -> Optional[Dict[str, Any]]:
        """Build a test tree that uses a leaf but OMITS a required choice."""
        leaf_spec = signal_dsl.LEAVES.get(leaf_name)
        if leaf_spec is None:
            return None

        # Build a minimal valid node for this leaf with all params, but OMIT required choice
        node = {"leaf": leaf_name}

        # Add all params
        for pname, (ptype, lo, hi) in leaf_spec.get("params", {}).items():
            val = lo if ptype == "int" else float(lo)
            node[pname] = int(val) if ptype == "int" else val

        # Add one_of choices (pick first)
        for oname, (ptype, lo, hi) in leaf_spec.get("one_of", []):
            val = lo if ptype == "int" else float(lo)
            node[oname] = int(val) if ptype == "int" else val
            break  # Only add one

        # Add optional choices EXCEPT the required ones
        for cname, choices in leaf_spec.get("choices", {}).items():
            if cname not in required_choices:
                node[cname] = choices[0]

        # Deliberately OMIT required_choices — this is the malformed part

        # Wrap in a minimal tree
        tree = {"entry": node}
        return tree
