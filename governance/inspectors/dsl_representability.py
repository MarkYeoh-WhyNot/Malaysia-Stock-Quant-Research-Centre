"""DSL Representability Checker — validates the parser honesty contract.

The parser contract (SACRED in CLAUDE.md): if a strategy cannot be expressed
with the available DSL leaves, the parser must say so
({"representable": false, "reason": ...}) — silent approximation or
substitution is forbidden.

This inspector audits the parse result to ensure:
1. If all leaves in the parse result are valid (exist in signal_dsl.LEAVES),
   and the tree structure is correct, the parse is honest.
2. If the parser returned true but used an unknown/substituted leaf, the
   parse is BLOCKER (silent genericization).
3. If the parser returned false when it should have returned true (or vice
   versa), investigate the mismatch.
"""

from typing import Optional, Dict, Any
from governance.base import Inspector
from governance.schemas import Finding


class DSLRepresentabilityChecker(Inspector):
    """L0 deterministic auditor for DSL parser honesty contract.

    Validates that:
    - Unknown leaves are not silently substituted with nearby leaves
    - Representable strategies parse to valid trees with known leaves only
    - Unrepresentable strategies correctly return representable=false
    """

    name = "DSLRepresentabilityChecker"
    level = "L0"

    def inspect(
        self, scope: str, ctx: Dict[str, Any]
    ) -> Optional[Finding]:
        """Validate DSL parser honesty — no silent approximations allowed.

        Args:
            scope: Context identifier (e.g. "idea:567", "parse_test:1")
            ctx: Dictionary containing:
                - "parse_result": dict from parse_factor() (or mocked equivalent)
                - "expected_representable": bool (whether we expect it to be representable)
                - "leaf_registry": dict of valid leaves (signal_dsl.LEAVES)
                - "factor_formula": str (for diagnostics)
                - "title": str (for diagnostics)
                - "used_unknown_leaf": bool (optional; if True, the parser used a leaf
                  not in leaf_registry — this is BLOCKER even if representable=true)

        Returns:
            Finding with status PASS (honest parse) or FAIL (silent substitution).
        """
        from agents.backtest_engineer import signal_dsl

        parse_result = ctx.get("parse_result", {})
        expected_representable = ctx.get("expected_representable", None)
        leaf_registry = ctx.get("leaf_registry", signal_dsl.LEAVES)
        factor_formula = ctx.get("factor_formula", "")
        title = ctx.get("title", "")
        used_unknown_leaf = ctx.get("used_unknown_leaf", False)

        # Check 1: If the parser claims representable=true, verify the tree
        # only uses known leaves (no silent substitutions).
        is_representable = parse_result.get("representable", False)

        if is_representable:
            # The parser returned true. Check if the tree uses only known leaves.
            dsl = parse_result.get("dsl")
            if dsl:
                unknown_leaves = self._find_unknown_leaves(dsl, leaf_registry)
                if unknown_leaves or used_unknown_leaf:
                    # BLOCKER: silent substitution detected
                    return Finding(
                        agent=self.name,
                        level=self.level,
                        scope=scope,
                        status="FAIL",
                        severity="BLOCKER",
                        evidence={
                            "issue": "silent substitution or unknown leaf used",
                            "unknown_leaves": unknown_leaves,
                            "factor_formula": factor_formula,
                            "title": title,
                            "parse_result": parse_result,
                        },
                        local_recommendation=(
                            f"Parser used unknown leaves {unknown_leaves} instead of "
                            f"returning representable=false. This violates the honesty "
                            f"contract. Check signal_dsl.LEAVES and verify the parse "
                            f"logic does not genericize unrepresentable strategies."
                        ),
                        escalate_to="BacktestEngineer",
                    )

            # Tree looks honest. Check if representable=true is expected.
            if expected_representable is False:
                # We expected unrepresentable but got representable — mismatch
                return Finding(
                    agent=self.name,
                    level=self.level,
                    scope=scope,
                    status="WARN",
                    severity="WARNING",
                    evidence={
                        "issue": "parser returned representable=true when test expected false",
                        "factor_formula": factor_formula,
                        "title": title,
                        "parse_result": parse_result,
                    },
                    local_recommendation=(
                        "Parser unexpectedly found a representation. Verify the strategy "
                        "can actually be expressed with available leaves, or update the test."
                    ),
                )

            # PASS: representable=true and tree uses only known leaves
            return Finding(
                agent=self.name,
                level=self.level,
                scope=scope,
                status="PASS",
                severity="INFO",
                evidence={
                    "issue": "honest representable parse",
                    "factor_formula": factor_formula,
                    "title": title,
                    "leaves_used": self._extract_leaves_used(dsl, leaf_registry),
                },
                local_recommendation="Parse contract honored: strategy represented honestly.",
            )

        else:
            # Parser returned representable=false. Check if this is expected
            # or a legitimate unrepresentable.
            reason = parse_result.get("reason", "")

            if expected_representable is True:
                # We expected representable but got false — mismatch
                return Finding(
                    agent=self.name,
                    level=self.level,
                    scope=scope,
                    status="WARN",
                    severity="WARNING",
                    evidence={
                        "issue": "parser returned representable=false when test expected true",
                        "reason": reason,
                        "factor_formula": factor_formula,
                        "title": title,
                    },
                    local_recommendation=(
                        "Parser rejected the strategy as unrepresentable. "
                        "Verify the strategy truly cannot be expressed, or update the test."
                    ),
                )

            # PASS: representable=false and reason is provided (honest rejection)
            return Finding(
                agent=self.name,
                level=self.level,
                scope=scope,
                status="PASS",
                severity="INFO",
                evidence={
                    "issue": "honest unrepresentable rejection",
                    "reason": reason,
                    "factor_formula": factor_formula,
                    "title": title,
                },
                local_recommendation=f"Parse contract honored: strategy rejected as unrepresentable ({reason}).",
            )

    def _find_unknown_leaves(self, tree: dict, leaf_registry: dict) -> list:
        """Recursively search tree for leaves not in leaf_registry."""
        unknown = []

        def _walk(node):
            if not isinstance(node, dict):
                return
            if "leaf" in node:
                leaf_name = node["leaf"]
                if leaf_name not in leaf_registry:
                    unknown.append(leaf_name)
            for c in node.get("children", []):
                _walk(c)
            if "child" in node:
                _walk(node["child"])

        for part in ("entry", "exit", "short_entry", "short_exit"):
            if tree.get(part):
                _walk(tree[part])

        return list(set(unknown))  # deduplicate

    def _extract_leaves_used(self, tree: dict, leaf_registry: dict) -> list:
        """Recursively extract all leaf names used in tree."""
        leaves = []

        def _walk(node):
            if not isinstance(node, dict):
                return
            if "leaf" in node and node["leaf"] in leaf_registry:
                leaves.append(node["leaf"])
            for c in node.get("children", []):
                _walk(c)
            if "child" in node:
                _walk(node["child"])

        for part in ("entry", "exit", "short_entry", "short_exit"):
            if tree.get(part):
                _walk(tree[part])

        return list(set(leaves))  # deduplicate
