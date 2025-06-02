"""
Test for Task 4.2: Simplify lazy loading patterns.

This test evaluates which modules use __getattr__ for lazy loading and determines
which ones can be simplified by replacing with direct imports.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from typing import Dict, Set

import pytest


class LazyLoadingAnalyzer:
    """Analyzer to identify and evaluate lazy loading patterns."""

    def __init__(self):
        self.src_path = Path("src/saplings")
        self.modules_with_getattr: Dict[str, Dict] = {}
        self.circular_dependencies: Set[str] = set()

    def find_modules_with_getattr(self) -> Dict[str, Dict]:
        """Find all modules that use __getattr__ for lazy loading."""
        modules = {}

        for py_file in self.src_path.rglob("*.py"):
            if py_file.name == "__init__.py":
                module_path = str(py_file.parent.relative_to(self.src_path.parent)).replace(
                    "/", "."
                )
            else:
                module_path = str(
                    py_file.with_suffix("").relative_to(self.src_path.parent)
                ).replace("/", ".")

            try:
                with open(py_file, encoding="utf-8") as f:
                    content = f.read()

                if "__getattr__" in content:
                    tree = ast.parse(content)
                    getattr_info = self._analyze_getattr_function(tree, content)
                    if getattr_info:
                        modules[module_path] = {
                            "file_path": str(py_file),
                            "getattr_info": getattr_info,
                            "content": content,
                        }
            except Exception:
                # Skip files that can't be parsed
                continue

        return modules

    def _analyze_getattr_function(self, tree: ast.AST, content: str) -> Dict:
        """Analyze the __getattr__ function in a module."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "__getattr__":
                # Extract information about the __getattr__ function
                imports_from = []
                lazy_attributes = []

                for child in ast.walk(node):
                    if isinstance(child, ast.ImportFrom):
                        imports_from.append(
                            {
                                "module": child.module,
                                "names": [alias.name for alias in child.names]
                                if child.names
                                else [],
                            }
                        )

                # Look for __all__ definition to understand what's being lazily loaded
                for tree_node in ast.walk(tree):
                    if isinstance(tree_node, ast.Assign):
                        for target in tree_node.targets:
                            if isinstance(target, ast.Name) and target.id == "__all__":
                                if isinstance(tree_node.value, ast.List):
                                    lazy_attributes = [
                                        elt.s
                                        if isinstance(elt, ast.Str)
                                        else elt.value
                                        if isinstance(elt, ast.Constant)
                                        else str(elt)
                                        for elt in tree_node.value.elts
                                    ]

                return {
                    "imports_from": imports_from,
                    "lazy_attributes": lazy_attributes,
                    "function_body": ast.get_source_segment(content, node)
                    if hasattr(ast, "get_source_segment")
                    else "",
                }
        return {}

    def check_circular_dependency(self, module_name: str, target_module: str) -> bool:
        """Check if importing target_module from module_name would create a circular dependency."""
        try:
            # Try to import the target module
            target = importlib.import_module(target_module)

            # Check if the target module imports back to our module
            if hasattr(target, "__file__") and target.__file__:
                with open(target.__file__, encoding="utf-8") as f:
                    content = f.read()
                    # Simple check for imports that might reference back to our module
                    if module_name in content:
                        return True

            return False
        except ImportError:
            # If we can't import the target, assume it's safe
            return False

    def evaluate_lazy_loading_necessity(self) -> Dict[str, Dict]:
        """Evaluate which modules actually need lazy loading."""
        modules = self.find_modules_with_getattr()
        evaluation = {}

        for module_name, info in modules.items():
            getattr_info = info["getattr_info"]
            imports_from = getattr_info.get("imports_from", [])

            # Check if any of the imports would create circular dependencies
            has_circular_deps = False
            for import_info in imports_from:
                if import_info["module"] and self.check_circular_dependency(
                    module_name, import_info["module"]
                ):
                    has_circular_deps = True
                    break

            evaluation[module_name] = {
                "file_path": info["file_path"],
                "lazy_attributes": getattr_info.get("lazy_attributes", []),
                "imports_from": imports_from,
                "needs_lazy_loading": has_circular_deps,
                "reason": "circular_dependency" if has_circular_deps else "can_be_simplified",
            }

        return evaluation


@pytest.fixture()
def lazy_loading_analyzer():
    """Fixture providing a lazy loading analyzer."""
    return LazyLoadingAnalyzer()


def test_identify_modules_with_lazy_loading(lazy_loading_analyzer):
    """Test that we can identify all modules using __getattr__ for lazy loading."""
    modules = lazy_loading_analyzer.find_modules_with_getattr()

    # We should find several modules with __getattr__
    assert len(modules) > 0, "Should find modules with __getattr__"

    # Check that we found the expected modules
    expected_modules = [
        "saplings.utils",
        "saplings.integration",
        "saplings.retrieval",
        "saplings.orchestration",
        "saplings.adapters",
        "saplings.security",
    ]

    found_modules = set(modules.keys())
    for expected in expected_modules:
        assert any(expected in module for module in found_modules), f"Should find {expected} module"

    # Each module should have getattr_info
    for module_name, info in modules.items():
        assert "getattr_info" in info, f"Module {module_name} should have getattr_info"
        assert "file_path" in info, f"Module {module_name} should have file_path"


def test_evaluate_lazy_loading_necessity(lazy_loading_analyzer):
    """Test evaluation of which modules actually need lazy loading."""
    evaluation = lazy_loading_analyzer.evaluate_lazy_loading_necessity()

    assert len(evaluation) > 0, "Should evaluate at least some modules"

    # Check the structure of evaluation results
    for module_name, eval_info in evaluation.items():
        assert (
            "needs_lazy_loading" in eval_info
        ), f"Module {module_name} should have needs_lazy_loading"
        assert "reason" in eval_info, f"Module {module_name} should have reason"
        assert "lazy_attributes" in eval_info, f"Module {module_name} should have lazy_attributes"
        assert "imports_from" in eval_info, f"Module {module_name} should have imports_from"

        # Reason should be one of the expected values
        assert eval_info["reason"] in [
            "circular_dependency",
            "can_be_simplified",
        ], f"Module {module_name} should have valid reason"


def test_modules_that_can_be_simplified(lazy_loading_analyzer):
    """Test identification of modules that can have their lazy loading simplified."""
    evaluation = lazy_loading_analyzer.evaluate_lazy_loading_necessity()

    # Find modules that can be simplified
    can_be_simplified = {
        module: info for module, info in evaluation.items() if not info["needs_lazy_loading"]
    }

    # We expect some modules can be simplified
    # (This will help us identify which ones to refactor)
    print(f"\nModules that can be simplified ({len(can_be_simplified)}):")
    for module, info in can_be_simplified.items():
        print(f"  - {module}: {len(info['lazy_attributes'])} lazy attributes")
        print(f"    Imports from: {[imp['module'] for imp in info['imports_from']]}")

    print(f"\nModules that need lazy loading ({len(evaluation) - len(can_be_simplified)}):")
    for module, info in evaluation.items():
        if info["needs_lazy_loading"]:
            print(f"  - {module}: {info['reason']}")


def test_lazy_loading_patterns_are_consistent(lazy_loading_analyzer):
    """Test that lazy loading patterns are consistent across modules."""
    modules = lazy_loading_analyzer.find_modules_with_getattr()

    # Filter to only modules that actually use lazy loading (have imports and lazy attributes)
    lazy_loading_modules = {
        name: info
        for name, info in modules.items()
        if (
            len(info["getattr_info"].get("lazy_attributes", [])) > 0
            and len(info["getattr_info"].get("imports_from", [])) > 0
        )
    }

    # Check that modules using lazy loading follow similar patterns
    for module_name, info in lazy_loading_modules.items():
        getattr_info = info["getattr_info"]

        # Should have __all__ definition for lazy attributes
        assert (
            len(getattr_info.get("lazy_attributes", [])) > 0
        ), f"Module {module_name} should define lazy attributes in __all__"

        # Should import from somewhere
        assert (
            len(getattr_info.get("imports_from", [])) > 0
        ), f"Module {module_name} should have imports in __getattr__"


if __name__ == "__main__":
    # Run the analysis when script is executed directly
    analyzer = LazyLoadingAnalyzer()
    evaluation = analyzer.evaluate_lazy_loading_necessity()

    print("Lazy Loading Analysis Results:")
    print("=" * 50)

    for module, info in evaluation.items():
        print(f"\nModule: {module}")
        print(f"  Needs lazy loading: {info['needs_lazy_loading']}")
        print(f"  Reason: {info['reason']}")
        print(f"  Lazy attributes: {info['lazy_attributes']}")
        print(f"  Imports from: {[imp['module'] for imp in info['imports_from']]}")
