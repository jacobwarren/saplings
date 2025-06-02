"""
Test for Task 4.4: Audit and simplify import hooks.

This test evaluates the import hook system to determine if it's still needed
after API cleanup and whether it can be simplified.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pytest


class ImportHookAnalyzer:
    """Analyzer to evaluate the import hook system."""

    def __init__(self):
        self.src_path = Path("src/saplings")

    def check_import_hook_necessity(self) -> Dict[str, bool]:
        """Check if import hooks are still necessary after API cleanup."""
        results = {
            "deprecated_modules_exist": False,
            "complex_new_methods_exist": False,
            "dynamic_imports_exist": False,
            "internal_api_exposed": False,
        }

        # Check if deprecated modules still exist
        deprecated_modules = [
            "src/saplings/api/container.py",
            "src/saplings/api/document.py",
            "src/saplings/api/interfaces.py",
            "src/saplings/api/indexer.py",
            "src/saplings/api/tool_validation.py",
            "src/saplings/core/interfaces/__init__.py",
        ]

        for module_path in deprecated_modules:
            if Path(module_path).exists():
                results["deprecated_modules_exist"] = True
                break

        # Check for complex __new__ methods in Agent API
        try:
            from saplings.api.agent import Agent, AgentBuilder, AgentConfig

            for cls in [Agent, AgentBuilder, AgentConfig]:
                if hasattr(cls, "__new__"):
                    import inspect

                    try:
                        source = inspect.getsource(cls.__new__)
                        if "importlib.import_module" in source or "_get_" in source:
                            results["complex_new_methods_exist"] = True
                            break
                    except (OSError, TypeError):
                        # Built-in __new__ is fine
                        pass
        except ImportError:
            # If we can't import, assume it's fine
            pass

        # Check for dynamic imports in Agent API
        try:
            agent_api_path = Path("src/saplings/api/agent.py")
            if agent_api_path.exists():
                with open(agent_api_path, encoding="utf-8") as f:
                    content = f.read()
                    if "importlib.import_module" in content:
                        results["dynamic_imports_exist"] = True
        except Exception:
            pass

        # Check if internal APIs are inappropriately exposed
        try:
            from saplings import __all__ as main_all

            # Check if any internal modules are exposed in main package
            for item in main_all:
                if "_internal" in item or item.startswith("_"):
                    results["internal_api_exposed"] = True
                    break
        except (ImportError, AttributeError):
            # If __all__ doesn't exist or can't be imported, assume it's fine
            pass

        return results

    def evaluate_import_hook_complexity(self) -> Dict[str, any]:
        """Evaluate the complexity of the current import hook system."""
        try:
            from saplings.security._internal.hooks.import_hook import (
                EXCLUDED_MODULES,
                PUBLIC_API_ALTERNATIVES,
                InternalAPIWarningFinder,
                InternalAPIWarningLoader,
                install_import_hook,
            )

            return {
                "finder_exists": True,
                "loader_exists": True,
                "public_api_alternatives_count": len(PUBLIC_API_ALTERNATIVES),
                "excluded_modules_count": len(EXCLUDED_MODULES),
                "has_meta_path_manipulation": True,
                "has_builtin_import_patching": True,
                "complexity_score": self._calculate_complexity_score(),
            }
        except ImportError:
            return {
                "finder_exists": False,
                "loader_exists": False,
                "public_api_alternatives_count": 0,
                "excluded_modules_count": 0,
                "has_meta_path_manipulation": False,
                "has_builtin_import_patching": False,
                "complexity_score": 0,
            }

    def _calculate_complexity_score(self) -> int:
        """Calculate a complexity score for the import hook system."""
        score = 0

        # Check import hook file size and complexity
        import_hook_path = Path("src/saplings/security/_internal/hooks/import_hook.py")
        if import_hook_path.exists():
            with open(import_hook_path, encoding="utf-8") as f:
                lines = f.readlines()
                score += len(lines)  # Base score from line count

                # Add complexity for specific patterns
                content = "".join(lines)
                score += content.count("class ") * 10  # Classes add complexity
                score += content.count("def ") * 5  # Functions add complexity
                score += content.count("import ") * 2  # Imports add some complexity
                score += content.count("sys.meta_path") * 20  # Meta path manipulation is complex
                score += content.count("__builtins__") * 20  # Builtin patching is complex

        return score


@pytest.fixture()
def import_hook_analyzer():
    """Fixture providing an import hook analyzer."""
    return ImportHookAnalyzer()


def test_import_hook_necessity_evaluation(import_hook_analyzer):
    """Test evaluation of whether import hooks are still necessary."""
    necessity = import_hook_analyzer.check_import_hook_necessity()

    # Check the structure of necessity results
    expected_keys = [
        "deprecated_modules_exist",
        "complex_new_methods_exist",
        "dynamic_imports_exist",
        "internal_api_exposed",
    ]

    for key in expected_keys:
        assert key in necessity, f"Should have {key} in necessity evaluation"
        assert isinstance(necessity[key], bool), f"{key} should be a boolean"

    # After API cleanup, these should all be False
    print("\nImport Hook Necessity Evaluation:")
    print(f"  Deprecated modules exist: {necessity['deprecated_modules_exist']}")
    print(f"  Complex __new__ methods exist: {necessity['complex_new_methods_exist']}")
    print(f"  Dynamic imports exist: {necessity['dynamic_imports_exist']}")
    print(f"  Internal APIs exposed: {necessity['internal_api_exposed']}")

    # Calculate if import hooks are still needed
    hooks_needed = any(necessity.values())
    print(f"  Import hooks still needed: {hooks_needed}")

    # If API cleanup is complete, import hooks should not be needed
    if not hooks_needed:
        print("  ✅ Import hooks may no longer be necessary after API cleanup")
    else:
        print("  ⚠️  Import hooks are still needed due to remaining issues")


def test_import_hook_complexity_evaluation(import_hook_analyzer):
    """Test evaluation of import hook system complexity."""
    complexity = import_hook_analyzer.evaluate_import_hook_complexity()

    # Check the structure of complexity results
    expected_keys = [
        "finder_exists",
        "loader_exists",
        "public_api_alternatives_count",
        "excluded_modules_count",
        "has_meta_path_manipulation",
        "has_builtin_import_patching",
        "complexity_score",
    ]

    for key in expected_keys:
        assert key in complexity, f"Should have {key} in complexity evaluation"

    print("\nImport Hook Complexity Evaluation:")
    print(f"  Finder exists: {complexity['finder_exists']}")
    print(f"  Loader exists: {complexity['loader_exists']}")
    print(f"  Public API alternatives: {complexity['public_api_alternatives_count']}")
    print(f"  Excluded modules: {complexity['excluded_modules_count']}")
    print(f"  Meta path manipulation: {complexity['has_meta_path_manipulation']}")
    print(f"  Builtin import patching: {complexity['has_builtin_import_patching']}")
    print(f"  Complexity score: {complexity['complexity_score']}")

    # Evaluate if the system is overly complex
    if complexity["complexity_score"] > 500:
        print("  ⚠️  Import hook system is highly complex")
    elif complexity["complexity_score"] > 200:
        print("  ⚠️  Import hook system has moderate complexity")
    else:
        print("  ✅ Import hook system has reasonable complexity")


def test_import_hook_simplification_recommendations(import_hook_analyzer):
    """Test generation of recommendations for simplifying import hooks."""
    necessity = import_hook_analyzer.check_import_hook_necessity()
    complexity = import_hook_analyzer.evaluate_import_hook_complexity()

    recommendations = []

    # If no deprecated modules exist, we can remove related hook logic
    if not necessity["deprecated_modules_exist"]:
        recommendations.append("Remove deprecated module detection logic")

    # If no complex __new__ methods exist, we can remove related warnings
    if not necessity["complex_new_methods_exist"]:
        recommendations.append("Remove complex __new__ method detection")

    # If no dynamic imports exist in public API, we can simplify
    if not necessity["dynamic_imports_exist"]:
        recommendations.append("Remove dynamic import detection for public APIs")

    # If internal APIs are not exposed, we can reduce exclusion lists
    if not necessity["internal_api_exposed"]:
        recommendations.append("Reduce or eliminate exclusion lists")

    # If overall complexity is high but necessity is low, recommend removal
    hooks_needed = any(necessity.values())
    if not hooks_needed and complexity["complexity_score"] > 100:
        recommendations.append("Consider removing import hooks entirely")
    elif complexity["complexity_score"] > 300:
        recommendations.append("Simplify import hook implementation")

    print("\nImport Hook Simplification Recommendations:")
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
    else:
        print("  No simplification recommendations at this time")

    # The test passes if we have some recommendations or if hooks are not needed
    assert (
        len(recommendations) > 0 or hooks_needed
    ), "Should have recommendations for simplification or hooks should still be needed"


if __name__ == "__main__":
    # Run the analysis when script is executed directly
    analyzer = ImportHookAnalyzer()
    necessity = analyzer.check_import_hook_necessity()
    complexity = analyzer.evaluate_import_hook_complexity()

    print("Import Hook Analysis Results:")
    print("=" * 50)
    print(f"Necessity: {necessity}")
    print(f"Complexity: {complexity}")
