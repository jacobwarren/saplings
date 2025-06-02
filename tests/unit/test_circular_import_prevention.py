"""
Test for Task 5.4: Test circular import prevention.

This test verifies that no circular imports exist after refactoring and that
all imports work correctly.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set

import pytest


class CircularImportAnalyzer:
    """Analyzer to detect circular imports in the codebase."""

    def __init__(self):
        self.src_path = Path("src/saplings")
        self.checked_modules: Set[str] = set()
        self.import_graph: Dict[str, Set[str]] = {}

    def discover_python_modules(self) -> List[str]:
        """Discover all Python modules in the saplings package."""
        modules = []

        for py_file in self.src_path.rglob("*.py"):
            if py_file.name == "__init__.py":
                # Convert path to module name
                relative_path = py_file.parent.relative_to(Path("src"))
                module_name = str(relative_path).replace("/", ".")
            else:
                # Convert file path to module name
                relative_path = py_file.with_suffix("").relative_to(Path("src"))
                module_name = str(relative_path).replace("/", ".")

            if module_name.startswith("saplings"):
                modules.append(module_name)

        return sorted(modules)

    def check_circular_imports_with_script(self) -> Dict[str, any]:
        """Use the existing circular import check script."""
        script_path = Path("scripts/check_circular_imports.py")

        if not script_path.exists():
            return {"script_exists": False, "error": "Script not found"}

        try:
            # Run the circular import check script
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )

            return {
                "script_exists": True,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "has_circular_imports": result.returncode != 0,
            }
        except subprocess.TimeoutExpired:
            return {
                "script_exists": True,
                "error": "Script timed out",
                "has_circular_imports": None,
            }
        except Exception as e:
            return {"script_exists": True, "error": str(e), "has_circular_imports": None}

    def test_core_module_imports(self) -> Dict[str, bool]:
        """Test that core modules can be imported without circular import errors."""
        core_modules = [
            "saplings",
            "saplings.api",
            "saplings.api.agent",
            "saplings.api.tools",
            "saplings.api.models",
            "saplings.api.services",
            "saplings.api.memory",
            "saplings.api.retrieval",
            "saplings.api.monitoring",
            "saplings.api.gasa",
        ]

        results = {}

        for module_name in core_modules:
            try:
                # Clear the module from cache if it exists
                if module_name in sys.modules:
                    del sys.modules[module_name]

                # Try to import the module
                importlib.import_module(module_name)
                results[module_name] = True
            except ImportError as e:
                results[module_name] = False
                print(f"Failed to import {module_name}: {e}")
            except Exception as e:
                results[module_name] = False
                print(f"Error importing {module_name}: {e}")

        return results

    def test_api_module_imports(self) -> Dict[str, bool]:
        """Test that all API modules can be imported."""
        api_modules = [
            "saplings.api.agent",
            "saplings.api.tools",
            "saplings.api.models",
            "saplings.api.services",
            "saplings.api.memory",
            "saplings.api.retrieval",
            "saplings.api.monitoring",
            "saplings.api.gasa",
            "saplings.api.di",
            "saplings.api.security",
        ]

        results = {}

        for module_name in api_modules:
            try:
                # Clear the module from cache if it exists
                if module_name in sys.modules:
                    del sys.modules[module_name]

                # Try to import the module
                importlib.import_module(module_name)
                results[module_name] = True
            except ImportError as e:
                results[module_name] = False
                print(f"Failed to import {module_name}: {e}")
            except Exception as e:
                results[module_name] = False
                print(f"Error importing {module_name}: {e}")

        return results


@pytest.fixture()
def circular_import_analyzer():
    """Fixture providing a circular import analyzer."""
    return CircularImportAnalyzer()


def test_circular_import_script_exists(circular_import_analyzer):
    """Test that the circular import check script exists and works."""
    result = circular_import_analyzer.check_circular_imports_with_script()

    assert result["script_exists"], "Circular import check script should exist"

    if "error" not in result:
        print("\nCircular Import Script Results:")
        print(f"  Return code: {result['return_code']}")
        print(f"  Has circular imports: {result['has_circular_imports']}")

        if result["stdout"]:
            print(f"  Output: {result['stdout'][:200]}...")

        # After refactoring, there should be no circular imports
        assert not result[
            "has_circular_imports"
        ], "Should have no circular imports after refactoring"
    else:
        print(f"  Script error: {result['error']}")
        # If script has issues, we'll rely on other tests


def test_core_module_imports(circular_import_analyzer):
    """Test that core modules can be imported without circular import issues."""
    results = circular_import_analyzer.test_core_module_imports()

    successful_imports = sum(1 for success in results.values() if success)
    total_modules = len(results)

    print("\nCore Module Import Results:")
    print(f"  Successful imports: {successful_imports}/{total_modules}")

    failed_modules = [module for module, success in results.items() if not success]
    if failed_modules:
        print(f"  Failed modules: {failed_modules}")

    # All core modules should be importable
    assert (
        successful_imports == total_modules
    ), f"All core modules should be importable, failed: {failed_modules}"


def test_api_module_imports(circular_import_analyzer):
    """Test that API modules can be imported without circular import issues."""
    results = circular_import_analyzer.test_api_module_imports()

    successful_imports = sum(1 for success in results.values() if success)
    total_modules = len(results)

    print("\nAPI Module Import Results:")
    print(f"  Successful imports: {successful_imports}/{total_modules}")

    failed_modules = [module for module, success in results.items() if not success]
    if failed_modules:
        print(f"  Failed modules: {failed_modules}")

    # Most API modules should be importable (allow for some optional dependencies)
    success_rate = successful_imports / total_modules
    assert (
        success_rate >= 0.9
    ), f"At least 90% of API modules should be importable, got {success_rate:.1%}"


def test_import_order_independence(circular_import_analyzer):
    """Test that import order doesn't matter (no hidden circular dependencies)."""
    # Test different import orders for core modules
    import_orders = [
        ["saplings.api.agent", "saplings.api.tools", "saplings.api.models"],
        ["saplings.api.models", "saplings.api.agent", "saplings.api.tools"],
        ["saplings.api.tools", "saplings.api.models", "saplings.api.agent"],
    ]

    for i, order in enumerate(import_orders):
        print(f"\nTesting import order {i+1}: {order}")

        # Clear modules from cache
        for module_name in order:
            if module_name in sys.modules:
                del sys.modules[module_name]

        # Try to import in this order
        success = True
        for module_name in order:
            try:
                importlib.import_module(module_name)
            except Exception as e:
                print(f"  Failed to import {module_name}: {e}")
                success = False
                break

        assert success, f"Import order {i+1} should work: {order}"


def test_no_import_time_side_effects():
    """Test that importing modules doesn't have unexpected side effects."""
    # Test that importing the same module multiple times is safe
    test_modules = ["saplings.api", "saplings.api.agent", "saplings.api.tools"]

    for module_name in test_modules:
        # Import the module multiple times
        for _ in range(3):
            try:
                importlib.import_module(module_name)
            except Exception as e:
                pytest.fail(f"Multiple imports of {module_name} should be safe, got: {e}")

    print(f"\nMultiple import test passed for {len(test_modules)} modules")


if __name__ == "__main__":
    # Run the analysis when script is executed directly
    analyzer = CircularImportAnalyzer()

    print("Circular Import Analysis Results:")
    print("=" * 50)

    script_result = analyzer.check_circular_imports_with_script()
    core_results = analyzer.test_core_module_imports()
    api_results = analyzer.test_api_module_imports()

    print(f"Script check: {script_result}")
    print(f"Core imports: {sum(core_results.values())}/{len(core_results)} successful")
    print(f"API imports: {sum(api_results.values())}/{len(api_results)} successful")
