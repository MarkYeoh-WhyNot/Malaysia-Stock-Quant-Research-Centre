"""Negative Mapping Guard — prevents silent wrong-mechanism substitutions in parsing.

The keystone parser honesty contract: "close > 50-day EMA" must NOT compile
to ema_cross(fast=2, slow=50) (a real historical bug — idea #73). The correct
outcome is EITHER a ma_level tree (price vs ONE moving average) OR an honest
{"representable": false} rejection.

This inspector audits the parser's result for the canonical negative example
and any other wrong-mechanism substitutions.
"""

from typing import Optional, Dict, Any
from governance.base import Inspector
from governance.schemas import Finding


class NegativeMappingGuard(Inspector):
    """L0 parser honesty auditor — guards against silent DSL substitution errors.

    Validates that parse results for price-vs-moving-average strategies do NOT
    return cross-type leaves (ema_cross, sma_cross) with tiny fast periods
    (standing in for price). Such results indicate the parser silently approximated
    an idea onto a plausible but structurally wrong condition tree.
    """

    name = "NegativeMappingGuard"
    level = "L0"

    def inspect(
        self, scope: str, ctx: Dict[str, Any]
    ) -> Optional[Finding]:
        """Validate parse result against the honesty contract.

        Args:
            scope: Context identifier (e.g. "idea:123", "parse:456")
            ctx: Dictionary containing:
                - "parse_result": the parsed signal tree (dict)
                - "factor_formula": the original text (str, optional, for logging)
                - "title": the strategy title (str, optional, for logging)

        Returns:
            Finding with status PASS if the result is honest,
            or BLOCKER if a wrong-mechanism substitution is detected.
        """
        parse_result = ctx.get("parse_result", {})
        factor_formula = ctx.get("factor_formula", "(formula unavailable)")
        title = ctx.get("title", "(untitled)")

        # If the parser already rejected it as unrepresentable, it's honest.
        if not parse_result.get("representable"):
            return Finding(
                agent=self.name,
                level=self.level,
                scope=scope,
                status="PASS",
                severity="INFO",
                evidence={
                    "reason": parse_result.get("reason", "not representable"),
                    "formula": factor_formula,
                },
                local_recommendation="Parser correctly rejected an unrepresentable idea.",
            )

        # If it's a fundamental screen route, it's honest (not a DSL tree).
        if parse_result.get("route") == "fundamental_screen":
            return Finding(
                agent=self.name,
                level=self.level,
                scope=scope,
                status="PASS",
                severity="INFO",
                evidence={"route": "fundamental_screen"},
                local_recommendation="Fundamental screen — no DSL honesty checks apply.",
            )

        # Check the DSL tree for wrong-mechanism substitutions.
        dsl = parse_result.get("dsl", {})
        if not isinstance(dsl, dict):
            return Finding(
                agent=self.name,
                level=self.level,
                scope=scope,
                status="PASS",
                severity="INFO",
                evidence={"dsl_type": type(dsl).__name__},
                local_recommendation="No DSL tree to audit.",
            )

        # Walk the tree and look for wrong-mechanism leaves.
        violations = self._check_tree(dsl)
        if violations:
            return Finding(
                agent=self.name,
                level=self.level,
                scope=scope,
                status="FAIL",
                severity="BLOCKER",
                evidence={
                    "violations": violations,
                    "formula": factor_formula,
                    "title": title,
                },
                local_recommendation=(
                    "Parser committed a silent substitution error: the formula's "
                    "price-vs-moving-average structure was replaced with a cross-type "
                    "leaf. The formula must be represented EXACTLY (ma_level) or rejected "
                    "as unrepresentable (if no leaf matches). Review and re-parse."
                ),
                escalate_to="StrategyResearcher",
            )

        # No violations — tree is honest.
        return Finding(
            agent=self.name,
            level=self.level,
            scope=scope,
            status="PASS",
            severity="INFO",
            evidence={"formula": factor_formula, "violations_found": False},
            local_recommendation="Parse result respects the honesty contract.",
        )

    def _check_tree(self, dsl: dict) -> list[str]:
        """Walk the DSL tree and return a list of detected violations.

        Each violation is a human-readable description of the wrong-mechanism
        substitution (e.g. "ema_cross(fast=2, slow=50) used as price proxy").
        """
        violations = []

        def _walk(node, path_label=""):
            if not isinstance(node, dict):
                return
            if "op" in node:
                for i, child in enumerate(node.get("children", [])):
                    _walk(child, f"{path_label}/children[{i}]")
                if "child" in node:
                    _walk(node["child"], f"{path_label}/child")
                return
            leaf = node.get("leaf")
            if leaf in ("ema_cross", "sma_cross"):
                fast = node.get("fast")
                slow = node.get("slow")
                # The canonical negative example: fast=2, slow=50 (or any small
                # fast period < 5, which is suspiciously close to "price") with
                # slow >= 40 (semantic sign of a price proxy).
                if (
                    fast is not None
                    and slow is not None
                    and float(fast) < 5
                    and float(slow) >= 40
                ):
                    violations.append(
                        f"{leaf}(fast={fast}, slow={slow}) at {path_label}: "
                        f"tiny fast period is a price proxy (confusing price vs TWO-MA cross). "
                        f"Use ma_level leaf instead."
                    )

        for part in ("entry", "exit", "short_entry", "short_exit"):
            if dsl.get(part):
                _walk(dsl[part], part)

        return violations
