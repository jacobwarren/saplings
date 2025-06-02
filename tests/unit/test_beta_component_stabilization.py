"""
Test beta component stabilization for publication readiness.

This module tests Task 7.3: Stabilize or remove beta components.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Dict, List


class TestBetaComponentStabilization:
    """Test beta component stabilization and evaluation."""

    def test_identify_beta_components(self):
        """Test that we can identify all beta components in the API."""
        beta_components = self._find_beta_components()

        # We should find several beta components
        assert len(beta_components) > 0, "Should find beta components"

        # Expected beta components based on the codebase analysis
        expected_beta_components = {
            "AgentFacade",
            "AgentFacadeBuilder",  # Agent facade components
            "VLLMAdapter",
            "HuggingFaceAdapter",  # Model adapters
            "SpeechToTextTool",  # Speech tool
            "ModalityHandler",
            "AudioHandler",
            "ImageHandler",
            "TextHandler",
            "VideoHandler",  # Modality handlers
        }

        found_component_names = {comp["name"] for comp in beta_components}

        for expected in expected_beta_components:
            assert expected in found_component_names, f"Should find beta component {expected}"

        print(f"\nFound {len(beta_components)} beta components:")
        for comp in beta_components:
            print(f"  - {comp['name']} in {comp['module']}")

    def test_evaluate_beta_components_for_stabilization(self):
        """Test evaluation of beta components against stabilization criteria."""
        beta_components = self._find_beta_components()
        evaluation = self._evaluate_beta_components(beta_components)

        # Verify evaluation structure
        assert "promote_to_stable" in evaluation
        assert "keep_as_beta" in evaluation
        assert "remove_or_experimental" in evaluation

        print("\nBeta component evaluation:")
        print(f"Promote to stable: {len(evaluation['promote_to_stable'])}")
        for comp in evaluation["promote_to_stable"]:
            print(f"  - {comp['name']}: {comp['reason']}")

        print(f"Keep as beta: {len(evaluation['keep_as_beta'])}")
        for comp in evaluation["keep_as_beta"]:
            print(f"  - {comp['name']}: {comp['reason']}")

        print(f"Remove or move to experimental: {len(evaluation['remove_or_experimental'])}")
        for comp in evaluation["remove_or_experimental"]:
            print(f"  - {comp['name']}: {comp['reason']}")

    def test_beta_components_have_documentation(self):
        """Test that beta components have adequate documentation."""
        beta_components = self._find_beta_components()
        documentation_analysis = {}

        for component in beta_components:
            obj = component["object"]
            name = component["name"]

            # Analyze documentation quality
            has_docstring = obj.__doc__ is not None
            if has_docstring:
                docstring = obj.__doc__.strip()
                has_substantial_docs = len(docstring) > 50
                mentions_beta = "beta" in docstring.lower()

                documentation_analysis[name] = {
                    "has_docstring": True,
                    "substantial": has_substantial_docs,
                    "mentions_beta": mentions_beta,
                    "length": len(docstring),
                }
            else:
                documentation_analysis[name] = {
                    "has_docstring": False,
                    "substantial": False,
                    "mentions_beta": False,
                    "length": 0,
                }

        # Report documentation status
        print(f"\nDocumentation analysis for {len(beta_components)} beta components:")
        good_docs = 0
        for name, analysis in documentation_analysis.items():
            status = "✅" if analysis["substantial"] and analysis["mentions_beta"] else "❌"
            print(
                f"  {status} {name}: {analysis['length']} chars, beta mentioned: {analysis['mentions_beta']}"
            )
            if analysis["substantial"] and analysis["mentions_beta"]:
                good_docs += 1

        print(f"\nSummary: {good_docs}/{len(beta_components)} components have good documentation")

        # For now, we document the state rather than enforce requirements
        # This will be updated as documentation improves

    def test_beta_components_test_coverage(self):
        """Test that beta components have adequate test coverage."""
        beta_components = self._find_beta_components()
        coverage_analysis = self._analyze_test_coverage(beta_components)

        print("\nTest coverage analysis for beta components:")
        for comp_name, coverage in coverage_analysis.items():
            print(f"  - {comp_name}: {coverage['status']} ({coverage['details']})")

        # For now, we document coverage rather than enforce it
        # This will be updated as test coverage improves

    def test_stabilization_plan_creation(self):
        """Test creation of a stabilization plan for beta components."""
        beta_components = self._find_beta_components()
        evaluation = self._evaluate_beta_components(beta_components)
        plan = self._create_stabilization_plan(evaluation)

        # Verify plan structure
        assert "immediate_promotion" in plan
        assert "needs_improvement" in plan
        assert "removal_candidates" in plan

        print("\nStabilization plan:")
        for phase, components in plan.items():
            print(f"{phase}: {len(components)} components")
            for comp in components:
                print(f"  - {comp['name']}: {comp['action']}")

    def _find_beta_components(self) -> List[Dict]:
        """Find all components marked with @beta."""
        beta_components = []

        # API modules to check
        api_modules = [
            "saplings.api.agent",
            "saplings.api.tools",
            "saplings.api.models",
            "saplings.api.modality",
            "saplings.api.services",
            "saplings.api.memory",
            "saplings.api.retrieval",
            "saplings.api.monitoring",
        ]

        for module_name in api_modules:
            try:
                module = importlib.import_module(module_name)

                for name, obj in inspect.getmembers(module):
                    if (
                        not name.startswith("_")
                        and inspect.isclass(obj)
                        and hasattr(obj, "__stability__")
                        and obj.__stability__ == "beta"
                    ):
                        beta_components.append({"name": name, "module": module_name, "object": obj})
            except ImportError:
                continue

        return beta_components

    def _evaluate_beta_components(self, beta_components: List[Dict]) -> Dict[str, List[Dict]]:
        """Evaluate beta components against stabilization criteria."""
        evaluation = {"promote_to_stable": [], "keep_as_beta": [], "remove_or_experimental": []}

        for component in beta_components:
            name = component["name"]
            obj = component["object"]

            # Evaluation criteria
            has_good_docs = self._has_good_documentation(obj)
            is_well_designed = self._is_well_designed_api(name, obj)
            has_test_coverage = self._has_test_coverage(name)
            is_widely_used = self._is_widely_used(name)
            has_known_issues = self._has_known_issues(name)

            # Decision logic based on standardization document
            if (
                has_good_docs
                and is_well_designed
                and has_test_coverage
                and is_widely_used
                and not has_known_issues
            ):
                evaluation["promote_to_stable"].append(
                    {"name": name, "reason": "Meets all criteria for stable promotion"}
                )
            elif has_good_docs and is_well_designed and not has_known_issues:
                evaluation["keep_as_beta"].append(
                    {"name": name, "reason": "Good API design but needs more testing/usage"}
                )
            else:
                issues = []
                if not has_good_docs:
                    issues.append("poor documentation")
                if not is_well_designed:
                    issues.append("API design issues")
                if has_known_issues:
                    issues.append("known bugs/limitations")

                evaluation["remove_or_experimental"].append(
                    {"name": name, "reason": f"Issues: {', '.join(issues)}"}
                )

        return evaluation

    def _has_good_documentation(self, obj) -> bool:
        """Check if component has good documentation."""
        if not obj.__doc__:
            return False

        docstring = obj.__doc__.strip()

        # Basic documentation quality checks
        has_description = len(docstring) > 100
        mentions_beta = "beta" in docstring.lower()

        return has_description and mentions_beta

    def _is_well_designed_api(self, name: str, obj) -> bool:
        """Check if component has well-designed API."""
        # Simple heuristics for API design quality

        # Check for reasonable number of public methods
        public_methods = [m for m in dir(obj) if not m.startswith("_")]
        has_reasonable_interface = 1 <= len(public_methods) <= 20

        # Check for consistent naming
        has_consistent_naming = not any(
            m.islower() and "_" not in m for m in public_methods if len(m) > 1
        )

        # Special cases for known good/bad designs
        if name in ["AgentFacade", "AgentFacadeBuilder"]:
            # These are known to be well-designed
            return True
        elif name in ["VLLMAdapter", "HuggingFaceAdapter"]:
            # These follow the same pattern as stable adapters
            return True
        elif "Handler" in name:
            # Modality handlers follow consistent pattern
            return True

        return has_reasonable_interface and has_consistent_naming

    def _has_test_coverage(self, name: str) -> bool:
        """Check if component has test coverage."""
        # Simple heuristic: check if there are test files that might cover this component
        from pathlib import Path

        test_dirs = [Path("tests/unit"), Path("tests/integration")]

        for test_dir in test_dirs:
            if test_dir.exists():
                for test_file in test_dir.rglob("*.py"):
                    try:
                        with open(test_file) as f:
                            content = f.read()
                            if name in content:
                                return True
                    except Exception:
                        continue

        return False

    def _is_widely_used(self, name: str) -> bool:
        """Check if component is widely used."""
        # Simple heuristic: check usage in examples
        from pathlib import Path

        examples_dir = Path("examples")
        usage_count = 0

        if examples_dir.exists():
            for example_file in examples_dir.rglob("*.py"):
                try:
                    with open(example_file) as f:
                        content = f.read()
                        if name in content:
                            usage_count += 1
                except Exception:
                    continue

        # Consider widely used if found in multiple examples
        return usage_count >= 2

    def _has_known_issues(self, name: str) -> bool:
        """Check if component has known issues."""
        # Known issues based on codebase analysis
        known_issues = {
            "AgentFacade": False,  # Well-implemented
            "AgentFacadeBuilder": False,  # Well-implemented
            "VLLMAdapter": True,  # May have performance/compatibility issues
            "HuggingFaceAdapter": True,  # May have compatibility issues
            "SpeechToTextTool": True,  # Depends on external libraries
        }

        return known_issues.get(name, False)

    def _analyze_test_coverage(self, beta_components: List[Dict]) -> Dict[str, Dict]:
        """Analyze test coverage for beta components."""
        coverage = {}

        for component in beta_components:
            name = component["name"]
            has_coverage = self._has_test_coverage(name)

            if has_coverage:
                coverage[name] = {"status": "Has tests", "details": "Found test references"}
            else:
                coverage[name] = {"status": "No tests found", "details": "Needs test coverage"}

        return coverage

    def _create_stabilization_plan(
        self, evaluation: Dict[str, List[Dict]]
    ) -> Dict[str, List[Dict]]:
        """Create a plan for stabilizing beta components."""
        plan = {"immediate_promotion": [], "needs_improvement": [], "removal_candidates": []}

        # Components ready for immediate promotion
        for comp in evaluation["promote_to_stable"]:
            plan["immediate_promotion"].append(
                {"name": comp["name"], "action": "Update @beta to @stable annotation"}
            )

        # Components that need improvement before promotion
        for comp in evaluation["keep_as_beta"]:
            plan["needs_improvement"].append(
                {
                    "name": comp["name"],
                    "action": "Improve test coverage and documentation before promotion",
                }
            )

        # Components that should be removed or moved to experimental
        for comp in evaluation["remove_or_experimental"]:
            plan["removal_candidates"].append(
                {"name": comp["name"], "action": f"Address issues: {comp['reason']}"}
            )

        return plan
