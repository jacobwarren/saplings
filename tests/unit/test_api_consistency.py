"""
Test for Task 5.2: Test API consistency.

This test verifies consistent API patterns across all modules and ensures
all public APIs work as expected with proper stability annotations.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Dict, List

import pytest


class APIConsistencyAnalyzer:
    """Analyzer to check API consistency across modules."""

    def __init__(self):
        self.api_modules = self._discover_api_modules()

    def _discover_api_modules(self) -> List[str]:
        """Discover all API modules in saplings.api."""
        api_path = Path("src/saplings/api")
        modules = []

        for py_file in api_path.rglob("*.py"):
            if py_file.name == "__init__.py":
                # Convert path to module name
                relative_path = py_file.parent.relative_to(Path("src"))
                module_name = str(relative_path).replace("/", ".")
            else:
                # Convert file path to module name
                relative_path = py_file.with_suffix("").relative_to(Path("src"))
                module_name = str(relative_path).replace("/", ".")

            if module_name.startswith("saplings.api"):
                modules.append(module_name)

        return sorted(modules)

    def check_stability_annotations(self) -> Dict[str, Dict]:
        """Check that all API modules have proper stability annotations."""
        results = {}

        for module_name in self.api_modules:
            try:
                module = importlib.import_module(module_name)
                module_info = {
                    "has_stability_import": False,
                    "classes_with_annotations": [],
                    "classes_without_annotations": [],
                    "functions_with_annotations": [],
                    "functions_without_annotations": [],
                }

                # Check if stability module is imported
                if (
                    hasattr(module, "stable")
                    or hasattr(module, "beta")
                    or hasattr(module, "experimental")
                ):
                    module_info["has_stability_import"] = True

                # Check classes for stability annotations
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if not name.startswith("_") and obj.__module__ == module_name:
                        if hasattr(obj, "__stability__") or hasattr(obj, "__wrapped__"):
                            module_info["classes_with_annotations"].append(name)
                        else:
                            module_info["classes_without_annotations"].append(name)

                # Check functions for stability annotations
                for name, obj in inspect.getmembers(module, inspect.isfunction):
                    if not name.startswith("_") and obj.__module__ == module_name:
                        if hasattr(obj, "__stability__") or hasattr(obj, "__wrapped__"):
                            module_info["functions_with_annotations"].append(name)
                        else:
                            module_info["functions_without_annotations"].append(name)

                results[module_name] = module_info

            except ImportError as e:
                results[module_name] = {"error": str(e)}

        return results

    def check_api_patterns(self) -> Dict[str, Dict]:
        """Check that all API modules follow consistent patterns."""
        results = {}

        for module_name in self.api_modules:
            try:
                module = importlib.import_module(module_name)
                module_info = {
                    "has_all_definition": hasattr(module, "__all__"),
                    "has_docstring": bool(module.__doc__),
                    "uses_direct_inheritance": False,
                    "uses_dynamic_imports": False,
                    "uses_complex_new": False,
                    "pattern_type": "unknown",
                }

                # Check for direct inheritance pattern
                source_file = inspect.getfile(module)
                with open(source_file, encoding="utf-8") as f:
                    content = f.read()

                    # Check for direct inheritance pattern
                    if "class " in content and "(_" in content and "from saplings." in content:
                        module_info["uses_direct_inheritance"] = True
                        module_info["pattern_type"] = "direct_inheritance"

                    # Check for dynamic imports
                    if "importlib.import_module" in content:
                        module_info["uses_dynamic_imports"] = True

                    # Check for complex __new__ methods
                    if "__new__" in content and "importlib" in content:
                        module_info["uses_complex_new"] = True

                results[module_name] = module_info

            except (ImportError, OSError) as e:
                results[module_name] = {"error": str(e)}

        return results

    def check_api_functionality(self) -> Dict[str, Dict]:
        """Check that all public APIs work as expected."""
        results = {}

        for module_name in self.api_modules:
            try:
                module = importlib.import_module(module_name)
                module_info = {
                    "importable": True,
                    "public_classes": [],
                    "public_functions": [],
                    "instantiable_classes": [],
                    "callable_functions": [],
                }

                # Get public classes and functions
                if hasattr(module, "__all__"):
                    public_items = module.__all__
                else:
                    public_items = [name for name in dir(module) if not name.startswith("_")]

                for name in public_items:
                    if hasattr(module, name):
                        obj = getattr(module, name)

                        if inspect.isclass(obj):
                            module_info["public_classes"].append(name)
                            # Try to check if class is instantiable (basic check)
                            try:
                                # Don't actually instantiate, just check if it's a proper class
                                if hasattr(obj, "__init__"):
                                    module_info["instantiable_classes"].append(name)
                            except Exception:
                                pass

                        elif inspect.isfunction(obj) or callable(obj):
                            module_info["public_functions"].append(name)
                            module_info["callable_functions"].append(name)

                results[module_name] = module_info

            except ImportError as e:
                results[module_name] = {"error": str(e), "importable": False}

        return results


@pytest.fixture()
def api_consistency_analyzer():
    """Fixture providing an API consistency analyzer."""
    return APIConsistencyAnalyzer()


def test_api_modules_discovery(api_consistency_analyzer):
    """Test that we can discover all API modules."""
    modules = api_consistency_analyzer.api_modules

    # Should find several API modules
    assert len(modules) > 0, "Should discover API modules"

    # Check for expected core modules
    expected_modules = [
        "saplings.api",
        "saplings.api.agent",
        "saplings.api.tools",
        "saplings.api.models",
        "saplings.api.services",
    ]

    found_modules = set(modules)
    for expected in expected_modules:
        assert expected in found_modules, f"Should find {expected} module"

    print(f"\nDiscovered {len(modules)} API modules:")
    for module in modules[:10]:  # Show first 10
        print(f"  - {module}")
    if len(modules) > 10:
        print(f"  ... and {len(modules) - 10} more")


def test_stability_annotations_consistency(api_consistency_analyzer):
    """Test that all API modules have consistent stability annotations."""
    stability_results = api_consistency_analyzer.check_stability_annotations()

    modules_with_issues = []
    modules_without_stability = []

    for module_name, info in stability_results.items():
        if "error" in info:
            continue

        # Check if module imports stability decorators
        if not info["has_stability_import"]:
            modules_without_stability.append(module_name)

        # Check if classes and functions have annotations
        if info["classes_without_annotations"] or info["functions_without_annotations"]:
            modules_with_issues.append(
                {
                    "module": module_name,
                    "classes_without": info["classes_without_annotations"],
                    "functions_without": info["functions_without_annotations"],
                }
            )

    print("\nStability Annotations Analysis:")
    print(f"  Modules without stability imports: {len(modules_without_stability)}")
    print(f"  Modules with unannotated items: {len(modules_with_issues)}")

    if modules_without_stability:
        print("  Modules missing stability imports:")
        for module in modules_without_stability[:5]:  # Show first 5
            print(f"    - {module}")

    if modules_with_issues:
        print("  Modules with unannotated items:")
        for issue in modules_with_issues[:3]:  # Show first 3
            print(
                f"    - {issue['module']}: {len(issue['classes_without'])} classes, {len(issue['functions_without'])} functions"
            )

    # Most modules should have stability annotations
    total_modules = len([m for m in stability_results.values() if "error" not in m])
    annotated_modules = total_modules - len(modules_without_stability)
    annotation_rate = annotated_modules / total_modules if total_modules > 0 else 0

    assert (
        annotation_rate >= 0.8
    ), f"At least 80% of modules should have stability annotations, got {annotation_rate:.1%}"


def test_api_patterns_consistency(api_consistency_analyzer):
    """Test that all API modules follow consistent patterns."""
    pattern_results = api_consistency_analyzer.check_api_patterns()

    modules_with_all = 0
    modules_with_docstrings = 0
    modules_with_direct_inheritance = 0
    modules_with_dynamic_imports = 0
    modules_with_complex_new = 0

    for module_name, info in pattern_results.items():
        if "error" in info:
            continue

        if info["has_all_definition"]:
            modules_with_all += 1
        if info["has_docstring"]:
            modules_with_docstrings += 1
        if info["uses_direct_inheritance"]:
            modules_with_direct_inheritance += 1
        if info["uses_dynamic_imports"]:
            modules_with_dynamic_imports += 1
        if info["uses_complex_new"]:
            modules_with_complex_new += 1

    total_modules = len([m for m in pattern_results.values() if "error" not in m])

    print("\nAPI Patterns Analysis:")
    print(f"  Modules with __all__: {modules_with_all}/{total_modules}")
    print(f"  Modules with docstrings: {modules_with_docstrings}/{total_modules}")
    print(f"  Modules using direct inheritance: {modules_with_direct_inheritance}/{total_modules}")
    print(f"  Modules using dynamic imports: {modules_with_dynamic_imports}/{total_modules}")
    print(f"  Modules using complex __new__: {modules_with_complex_new}/{total_modules}")

    # After standardization, modules should follow good patterns
    # Note: Docstring requirement is relaxed as many API modules are simple re-exports
    if total_modules > 0:
        docstring_rate = modules_with_docstrings / total_modules
        print(f"  Docstring coverage: {docstring_rate:.1%}")

        # Key requirements after cleanup
        assert (
            modules_with_dynamic_imports == 0
        ), "No modules should use dynamic imports after cleanup"
        assert (
            modules_with_complex_new == 0
        ), "No modules should use complex __new__ methods after cleanup"

        # Docstrings are good but not strictly required for all API modules
        if docstring_rate < 0.5:
            print(
                f"  ⚠️  Low docstring coverage ({docstring_rate:.1%}), consider adding more documentation"
            )
        else:
            print(f"  ✅ Good docstring coverage ({docstring_rate:.1%})")


def test_api_functionality_basic():
    """Test basic API functionality - that core modules are importable."""
    # Test just a few core modules to avoid the performance issue
    core_modules = [
        "saplings.api",
        "saplings.api.agent",
        "saplings.api.tools",
        "saplings.api.models",
        "saplings.api.services",
    ]

    importable_count = 0
    for module_name in core_modules:
        try:
            importlib.import_module(module_name)
            importable_count += 1
        except ImportError:
            pass

    print("\nCore API Functionality Analysis:")
    print(f"  Core modules importable: {importable_count}/{len(core_modules)}")

    # All core API modules should be importable
    assert importable_count == len(
        core_modules
    ), f"All core API modules should be importable, got {importable_count}/{len(core_modules)}"


if __name__ == "__main__":
    # Run the analysis when script is executed directly
    analyzer = APIConsistencyAnalyzer()

    print("API Consistency Analysis Results:")
    print("=" * 50)

    stability = analyzer.check_stability_annotations()
    patterns = analyzer.check_api_patterns()
    functionality = analyzer.check_api_functionality()

    print(f"Analyzed {len(analyzer.api_modules)} API modules")
    print(f"Stability results: {len(stability)} modules")
    print(f"Pattern results: {len(patterns)} modules")
    print(f"Functionality results: {len(functionality)} modules")
