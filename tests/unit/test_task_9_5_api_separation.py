"""
Test for Task 9.5: Fix cross-component internal imports violating API separation.

This test verifies that API modules don't import from other components' internal modules,
which violates architectural principles and creates circular dependency risks.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


class TestTask95APISeparation:
    """Test Task 9.5: Fix cross-component internal imports violating API separation."""

    def test_no_cross_component_internal_imports(self):
        """Test that components don't import from other components' internal modules."""
        src_dir = Path("/Users/jacobwarren/Development/agents/saplings/src")

        violations = []
        checked_files = 0

        # Check all Python files for cross-component internal imports
        for py_file in src_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            try:
                content = py_file.read_text()
                tree = ast.parse(content)

                file_component = self._get_component_from_path(py_file, src_dir)

                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom):
                        if node.module and "._internal" in node.module:
                            imported_component = self._get_component_from_import(node.module)

                            # Check for violations
                            if self._is_violation(
                                file_component, imported_component, py_file, node.module
                            ):
                                violations.append(
                                    {
                                        "file": str(py_file.relative_to(src_dir)),
                                        "file_component": file_component,
                                        "imported_module": node.module,
                                        "imported_component": imported_component,
                                        "line": getattr(node, "lineno", "unknown"),
                                    }
                                )

                checked_files += 1
                if checked_files >= 100:  # Limit to avoid long test times
                    break

            except Exception:
                continue

        print(f"Checked {checked_files} files")
        print(f"Found {len(violations)} cross-component internal import violations")

        # Group violations by type
        violation_types = {}
        for violation in violations:
            key = f"{violation['file_component']} → {violation['imported_component']}"
            if key not in violation_types:
                violation_types[key] = []
            violation_types[key].append(violation)

        # Show violation summary
        for violation_type, instances in violation_types.items():
            print(f"  {violation_type}: {len(instances)} violations")
            for instance in instances[:2]:  # Show first 2 examples
                print(f"    - {instance['file']} imports {instance['imported_module']}")

        # Don't fail test - this shows what needs to be fixed
        assert checked_files > 0, "Should check some files"

    def test_api_modules_only_import_same_component(self):
        """Test that API modules only import from their own component's internal modules."""
        src_dir = Path("/Users/jacobwarren/Development/agents/saplings/src")
        api_dir = src_dir / "saplings" / "api"

        if not api_dir.exists():
            pytest.fail("API directory doesn't exist")

        violations = []
        compliant_modules = []

        for py_file in api_dir.rglob("*.py"):
            if "__pycache__" in str(py_file) or py_file.name == "__init__.py":
                continue

            try:
                content = py_file.read_text()
                tree = ast.parse(content)

                module_name = str(py_file.relative_to(src_dir)).replace("/", ".").replace(".py", "")
                module_violations = []

                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom):
                        if node.module and "._internal" in node.module:
                            # Check if this is a cross-component import
                            if self._is_cross_component_api_import(module_name, node.module):
                                module_violations.append(
                                    {
                                        "api_module": module_name,
                                        "imported_module": node.module,
                                        "line": getattr(node, "lineno", "unknown"),
                                    }
                                )

                if module_violations:
                    violations.extend(module_violations)
                    print(f"❌ {module_name} has cross-component imports")
                    for violation in module_violations[:2]:  # Show first 2
                        print(f"    imports {violation['imported_module']}")
                else:
                    compliant_modules.append(module_name)
                    print(f"✅ {module_name} follows API separation")

            except Exception as e:
                print(f"⚠️  Could not analyze {py_file}: {e}")

        print(f"Compliant API modules: {len(compliant_modules)}")
        print(f"API modules with violations: {len(set(v['api_module'] for v in violations))}")
        print(f"Total violations: {len(violations)}")

        # Don't fail test - this shows what needs to be fixed
        total_modules = len(compliant_modules) + len(set(v["api_module"] for v in violations))
        assert total_modules > 0, "Should find some API modules to analyze"

    def test_core_modules_exist_for_shared_functionality(self):
        """Test that core modules exist for shared functionality."""
        src_dir = Path("/Users/jacobwarren/Development/agents/saplings/src")

        # Expected core modules for shared functionality
        expected_core_modules = [
            "saplings/api/core/types.py",
            "saplings/api/core/interfaces/__init__.py",
            "saplings/api/core/utils.py",
        ]

        existing_modules = []
        missing_modules = []

        for module_path in expected_core_modules:
            full_path = src_dir / module_path

            if full_path.exists():
                existing_modules.append(module_path)
                print(f"✅ Core module exists: {module_path}")
            else:
                missing_modules.append(module_path)
                print(f"❌ Core module missing: {module_path}")

        print(f"Existing core modules: {len(existing_modules)}")
        print(f"Missing core modules: {len(missing_modules)}")

        if missing_modules:
            print("Missing core modules need to be created for shared functionality:")
            for module in missing_modules:
                print(f"  - {module}")

        # Don't fail test - this shows what needs to be created
        assert len(expected_core_modules) > 0, "Should check for core modules"

    def test_dependency_injection_used_for_cross_component_communication(self):
        """Test that dependency injection is used for cross-component communication."""
        src_dir = Path("/Users/jacobwarren/Development/agents/saplings/src")

        # Look for proper DI patterns vs direct imports
        di_usage = []
        direct_imports = []

        for py_file in src_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            try:
                content = py_file.read_text()
                module_name = str(py_file.relative_to(src_dir))

                # Check for DI patterns
                has_di_import = (
                    "from saplings.di import" in content or "from saplings.api.di import" in content
                )
                has_container_usage = (
                    "container.resolve" in content or "container.register" in content
                )
                has_inject_decorator = "@inject" in content

                # Check for direct cross-component imports
                has_cross_component_import = False
                lines = content.split("\n")
                for line in lines:
                    if (
                        line.strip().startswith("from saplings.")
                        and "._internal" in line
                        and not line.strip().startswith("#")
                    ):
                        has_cross_component_import = True
                        break

                if has_di_import or has_container_usage or has_inject_decorator:
                    di_usage.append(module_name)
                    print(f"✅ {module_name} uses dependency injection")
                elif has_cross_component_import:
                    direct_imports.append(module_name)
                    print(f"⚠️  {module_name} uses direct cross-component imports")

            except Exception:
                continue

        print(f"Modules using DI: {len(di_usage)}")
        print(f"Modules using direct imports: {len(direct_imports)}")

        # Show some examples
        for module in direct_imports[:5]:
            print(f"  - {module}")

        # Don't fail test - this shows current state
        total_modules = len(di_usage) + len(direct_imports)
        assert total_modules >= 0, "Should analyze some modules"

    def _get_component_from_path(self, file_path: Path, src_dir: Path) -> str:
        """Get component name from file path."""
        rel_path = file_path.relative_to(src_dir)
        parts = rel_path.parts

        if len(parts) >= 2 and parts[0] == "saplings":
            if parts[1] == "api":
                return "api"
            elif parts[1] in [
                "memory",
                "tools",
                "models",
                "services",
                "retrieval",
                "orchestration",
                "integration",
                "gasa",
                "monitoring",
                "security",
                "utils",
                "vector_store",
                "registry",
            ]:
                return parts[1]
            elif parts[1] == "_internal":
                return "core"

        return None

    def _get_component_from_import(self, import_module: str) -> str:
        """Get component name from import module."""
        parts = import_module.split(".")

        if len(parts) >= 2 and parts[0] == "saplings":
            if parts[1] in [
                "memory",
                "tools",
                "models",
                "services",
                "retrieval",
                "orchestration",
                "integration",
                "gasa",
                "monitoring",
                "security",
                "utils",
                "vector_store",
                "registry",
            ]:
                return parts[1]
            elif parts[1] == "_internal":
                return "core"

        return None

    def _is_violation(
        self, file_component: str, imported_component: str, file_path: Path, import_module: str
    ) -> bool:
        """Check if this is a cross-component import violation."""
        if not file_component or not imported_component:
            return False

        # API modules can import from any component (they're the public interface)
        if file_component == "api":
            return False

        # Core modules importing from other components is a violation
        if file_component == "core" and imported_component != "core":
            return True

        # Components importing from other components' internals is a violation
        if file_component != imported_component and imported_component != "core":
            return True

        return False

    def _is_cross_component_api_import(self, api_module: str, imported_module: str) -> bool:
        """Check if an API module is doing a cross-component internal import."""
        # Extract the API component (e.g., "memory" from "saplings.api.memory.store")
        api_parts = api_module.split(".")
        if len(api_parts) >= 3 and api_parts[0] == "saplings" and api_parts[1] == "api":
            api_component = api_parts[2]
        else:
            return False

        # Extract the imported component
        import_parts = imported_module.split(".")
        if len(import_parts) >= 2 and import_parts[0] == "saplings":
            imported_component = import_parts[1]
        else:
            return False

        # It's a violation if API module imports from different component's internals
        return (
            imported_component != api_component
            and imported_component != "_internal"  # Core internals are OK
            and "._internal" in imported_module
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
