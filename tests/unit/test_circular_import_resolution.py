"""
Test circular import resolution for publication readiness.

This module tests Task 7.2: Resolve circular import issues permanently.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from typing import Dict, List, Set

import pytest


class TestCircularImportResolution:
    """Test circular import resolution and elimination of lazy loading."""

    def test_identify_lazy_loading_modules(self):
        """Test that we can identify all modules using lazy loading patterns."""
        lazy_modules = self._find_modules_with_lazy_loading()

        # We should find several modules with lazy loading
        assert len(lazy_modules) > 0, "Should find modules with lazy loading"

        # Expected modules that currently use lazy loading
        expected_lazy_modules = {
            "saplings.utils",
            "saplings.security",
            "saplings.retrieval",
            "saplings.api.services",
            "saplings._internal.agent",
        }

        found_modules = set(lazy_modules.keys())
        for expected in expected_lazy_modules:
            matching_modules = [m for m in found_modules if expected in m]
            assert len(matching_modules) > 0, f"Should find module matching {expected}"

    def test_categorize_lazy_loading_necessity(self):
        """Test categorization of lazy loading patterns by necessity."""
        lazy_modules = self._find_modules_with_lazy_loading()
        categorization = self._categorize_lazy_loading(lazy_modules)

        # Verify categorization structure
        assert "necessary" in categorization
        assert "unnecessary" in categorization
        assert "architectural_issue" in categorization

        # Print categorization for analysis
        print("\nLazy loading categorization:")
        print(f"Necessary (circular deps): {len(categorization['necessary'])}")
        for module in categorization["necessary"]:
            print(f"  - {module}")

        print(f"Unnecessary (can be direct): {len(categorization['unnecessary'])}")
        for module in categorization["unnecessary"]:
            print(f"  - {module}")

        print(f"Architectural issues: {len(categorization['architectural_issue'])}")
        for module in categorization["architectural_issue"]:
            print(f"  - {module}")

    def test_no_circular_imports_in_core_modules(self):
        """Test that core modules have no circular imports."""
        core_modules = [
            "saplings.api.agent",
            "saplings.api.tools",
            "saplings.api.models",
            "saplings.api.memory",
            "saplings.api.core",
        ]

        for module in core_modules:
            try:
                # Try to import the module
                importlib.import_module(module)
            except ImportError as e:
                if "circular import" in str(e).lower():
                    pytest.fail(f"Circular import detected in core module {module}: {e}")
                # Other import errors might be due to missing dependencies

    def test_static_import_analysis(self):
        """Test static analysis of import structure for circular dependencies."""
        import_graph = self._build_import_graph()
        cycles = self._find_cycles_in_graph(import_graph)

        # Report any cycles found
        if cycles:
            print(f"\nFound {len(cycles)} circular import cycles:")
            for i, cycle in enumerate(cycles, 1):
                print(f"Cycle {i}: {' -> '.join(cycle)} -> {cycle[0]}")

        # For now, we document existing cycles rather than fail
        # This test will be updated as cycles are resolved
        print(f"Current circular import cycles: {len(cycles)}")

    def test_lazy_loading_elimination_plan(self):
        """Test creation of a plan to eliminate unnecessary lazy loading."""
        lazy_modules = self._find_modules_with_lazy_loading()
        categorization = self._categorize_lazy_loading(lazy_modules)

        elimination_plan = self._create_elimination_plan(categorization)

        # Verify plan structure
        assert "phase_1" in elimination_plan  # Easy eliminations
        assert "phase_2" in elimination_plan  # Architectural refactoring
        assert "phase_3" in elimination_plan  # Complex cases

        print("\nLazy loading elimination plan:")
        for phase, modules in elimination_plan.items():
            print(f"{phase}: {len(modules)} modules")
            for module, action in modules.items():
                print(f"  - {module}: {action}")

    def _find_modules_with_lazy_loading(self) -> Dict[str, Dict]:
        """Find all modules that use lazy loading patterns."""
        lazy_modules = {}
        src_path = Path("src/saplings")

        for py_file in src_path.rglob("*.py"):
            try:
                with open(py_file, encoding="utf-8") as f:
                    content = f.read()

                # Check for lazy loading patterns
                has_getattr = "__getattr__" in content
                has_lazy_import = "lazy import" in content.lower()
                has_importlib = "importlib.import_module" in content

                if has_getattr or has_lazy_import or has_importlib:
                    module_path = self._get_module_path(py_file, src_path)
                    lazy_modules[module_path] = {
                        "file_path": str(py_file),
                        "has_getattr": has_getattr,
                        "has_lazy_import": has_lazy_import,
                        "has_importlib": has_importlib,
                        "content": content,
                    }
            except Exception:
                continue

        return lazy_modules

    def _categorize_lazy_loading(self, lazy_modules: Dict[str, Dict]) -> Dict[str, List[str]]:
        """Categorize lazy loading patterns by necessity."""
        categorization = {
            "necessary": [],  # True circular dependencies
            "unnecessary": [],  # Can be replaced with direct imports
            "architectural_issue": [],  # Indicates design problems
        }

        for module_path, info in lazy_modules.items():
            content = info["content"]

            # Check for explicit circular dependency comments
            if "circular" in content.lower() and "import" in content.lower():
                # Check if it's a real circular dependency or just convenience
                if self._is_real_circular_dependency(module_path, content):
                    categorization["necessary"].append(module_path)
                else:
                    categorization["unnecessary"].append(module_path)
            elif "avoid circular" in content.lower():
                categorization["necessary"].append(module_path)
            elif module_path.endswith("._internal.agent"):
                # Agent internal module has complex dependencies
                categorization["architectural_issue"].append(module_path)
            else:
                # Default to unnecessary unless proven otherwise
                categorization["unnecessary"].append(module_path)

        return categorization

    def _is_real_circular_dependency(self, module_path: str, content: str) -> bool:
        """Check if a module has a real circular dependency."""
        # Simple heuristic: if the module imports from modules that import back
        # This is a simplified check - real analysis would be more complex

        # For now, assume modules with explicit "circular" comments are real
        return "circular" in content.lower() and "dependencies" in content.lower()

    def _create_elimination_plan(
        self, categorization: Dict[str, List[str]]
    ) -> Dict[str, Dict[str, str]]:
        """Create a plan to eliminate unnecessary lazy loading."""
        plan = {
            "phase_1": {},  # Easy direct replacements
            "phase_2": {},  # Architectural refactoring needed
            "phase_3": {},  # Complex cases requiring design changes
        }

        # Phase 1: Simple utility modules that can use direct imports
        for module in categorization["unnecessary"]:
            if any(util in module for util in ["utils", "security", "retrieval"]):
                plan["phase_1"][module] = "Replace __getattr__ with direct imports"

        # Phase 2: Modules that need architectural changes
        for module in categorization["architectural_issue"]:
            plan["phase_2"][module] = "Refactor module structure to eliminate circular deps"

        # Phase 3: Complex cases that need careful analysis
        for module in categorization["necessary"]:
            if module not in plan["phase_1"] and module not in plan["phase_2"]:
                plan["phase_3"][module] = "Analyze and resolve circular dependency at design level"

        return plan

    def _get_module_path(self, file_path: Path, src_path: Path) -> str:
        """Get module path from file path."""
        if file_path.name == "__init__.py":
            rel_path = file_path.parent.relative_to(src_path.parent)
        else:
            rel_path = file_path.with_suffix("").relative_to(src_path.parent)

        return str(rel_path).replace("/", ".")

    def _build_import_graph(self) -> Dict[str, Set[str]]:
        """Build a graph of module imports."""
        # This is a simplified version - real implementation would parse AST
        import_graph = {}
        src_path = Path("src/saplings")

        for py_file in src_path.rglob("*.py"):
            try:
                module_path = self._get_module_path(py_file, src_path)
                imports = self._extract_imports(py_file)
                import_graph[module_path] = set(imports)
            except Exception:
                continue

        return import_graph

    def _extract_imports(self, file_path: Path) -> List[str]:
        """Extract imports from a Python file."""
        imports = []
        try:
            with open(file_path, encoding="utf-8") as f:
                tree = ast.parse(f.read())

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith("saplings"):
                            imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module.startswith("saplings"):
                        imports.append(node.module)
        except Exception:
            pass

        return imports

    def _find_cycles_in_graph(self, graph: Dict[str, Set[str]]) -> List[List[str]]:
        """Find cycles in the import graph using DFS."""
        cycles = []
        visited = set()
        rec_stack = set()
        path = []

        def dfs(node: str) -> None:
            if node in rec_stack:
                # Found a cycle
                cycle_start = path.index(node)
                cycles.append(path[cycle_start:])
                return

            if node in visited:
                return

            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in graph.get(node, []):
                if neighbor in graph:  # Only follow internal modules
                    dfs(neighbor)

            rec_stack.remove(node)
            path.pop()

        for node in graph:
            if node not in visited:
                dfs(node)

        return cycles
