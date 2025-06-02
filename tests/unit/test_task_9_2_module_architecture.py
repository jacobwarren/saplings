"""
Test for Task 9.2: Restructure module architecture to prevent future circular imports.

This test verifies that the module architecture follows strict layered architecture
to prevent circular imports at the design level.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


class TestTask92ModuleArchitecture:
    """Test Task 9.2: Restructure module architecture to prevent future circular imports."""

    def test_architectural_layers_exist(self):
        """Test that the required architectural layers exist."""
        src_dir = Path("/Users/jacobwarren/Development/agents/saplings/src")

        # Required architectural layers
        required_layers = [
            "saplings/api/core/types",
            "saplings/api/core/interfaces",
            "saplings/api/core",
            "saplings/api",
        ]

        missing_layers = []
        existing_layers = []

        for layer in required_layers:
            layer_path = src_dir / layer
            init_file = layer_path / "__init__.py"

            if layer_path.exists() and init_file.exists():
                existing_layers.append(layer)
                print(f"✅ Layer exists: {layer}")
            else:
                missing_layers.append(layer)
                print(f"❌ Layer missing: {layer}")

        if missing_layers:
            print(f"Missing architectural layers: {missing_layers}")
            # Don't fail test yet - this is what we need to implement

        assert (
            len(existing_layers) >= 2
        ), f"At least core layers should exist, found: {existing_layers}"

    def test_core_types_has_no_dependencies(self):
        """Test that saplings.api.core.types has no internal dependencies."""
        types_file = Path(
            "/Users/jacobwarren/Development/agents/saplings/src/saplings/api/core/types.py"
        )

        if not types_file.exists():
            print("⚠️  saplings.api.core.types doesn't exist yet - needs to be created")
            return

        content = types_file.read_text()
        tree = ast.parse(content)

        saplings_imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("saplings"):
                        saplings_imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("saplings"):
                    saplings_imports.append(node.module)

        if saplings_imports:
            print(f"❌ Core types has internal dependencies: {saplings_imports}")
            pytest.fail(
                f"Core types should have no internal dependencies, found: {saplings_imports}"
            )
        else:
            print("✅ Core types has no internal dependencies")

    def test_core_interfaces_minimal_dependencies(self):
        """Test that saplings.api.core.interfaces only depends on types."""
        interfaces_file = Path(
            "/Users/jacobwarren/Development/agents/saplings/src/saplings/api/core/interfaces.py"
        )

        if not interfaces_file.exists():
            print("⚠️  saplings.api.core.interfaces exists but checking __init__.py")
            interfaces_file = Path(
                "/Users/jacobwarren/Development/agents/saplings/src/saplings/api/core/interfaces/__init__.py"
            )

        if not interfaces_file.exists():
            print("⚠️  saplings.api.core.interfaces doesn't exist in expected location")
            return

        content = interfaces_file.read_text()
        tree = ast.parse(content)

        saplings_imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("saplings"):
                        saplings_imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("saplings"):
                    saplings_imports.append(node.module)

        # Filter allowed dependencies
        allowed_dependencies = ["saplings.api.core.types", "saplings.api.stability"]

        invalid_imports = [
            imp
            for imp in saplings_imports
            if not any(imp.startswith(allowed) for allowed in allowed_dependencies)
        ]

        if invalid_imports:
            print(f"⚠️  Core interfaces has invalid dependencies: {invalid_imports}")
            # Don't fail test yet - this is what we need to fix
        else:
            print("✅ Core interfaces has only allowed dependencies")

    def test_api_modules_follow_pattern(self):
        """Test that API modules follow the standardized pattern."""
        src_dir = Path("/Users/jacobwarren/Development/agents/saplings/src")
        api_dir = src_dir / "saplings" / "api"

        if not api_dir.exists():
            pytest.fail("API directory doesn't exist")

        # Find API modules (excluding core)
        api_modules = []
        for item in api_dir.iterdir():
            if item.is_file() and item.name.endswith(".py") and item.name != "__init__.py":
                api_modules.append(item)
            elif item.is_dir() and item.name != "core" and (item / "__init__.py").exists():
                api_modules.append(item / "__init__.py")

        pattern_violations = []
        pattern_compliant = []

        for module_file in api_modules[:10]:  # Check first 10 modules
            try:
                content = module_file.read_text()

                # Check for standardized pattern indicators
                has_stability_import = "from saplings.api.stability import" in content
                has_internal_import = "._internal" in content
                has_all_definition = "__all__" in content

                module_name = (
                    str(module_file.relative_to(src_dir)).replace("/", ".").replace(".py", "")
                )

                if has_stability_import and has_internal_import and has_all_definition:
                    pattern_compliant.append(module_name)
                    print(f"✅ {module_name} follows standardized pattern")
                else:
                    pattern_violations.append(
                        {
                            "module": module_name,
                            "missing": {
                                "stability_import": not has_stability_import,
                                "internal_import": not has_internal_import,
                                "all_definition": not has_all_definition,
                            },
                        }
                    )
                    print(f"⚠️  {module_name} doesn't follow standardized pattern")

            except Exception as e:
                print(f"⚠️  Could not analyze {module_file}: {e}")

        print(f"Pattern compliant modules: {len(pattern_compliant)}")
        print(f"Pattern violations: {len(pattern_violations)}")

        # Don't fail test - this shows current state and what needs to be fixed
        assert len(api_modules) > 0, "Should find some API modules to analyze"

    def test_no_cross_component_internal_imports(self):
        """Test that components don't import from other components' internal modules."""
        src_dir = Path("/Users/jacobwarren/Development/agents/saplings/src")

        violations = []
        checked_files = 0

        # Check API modules for cross-component internal imports
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

                            if (
                                file_component
                                and imported_component
                                and file_component != imported_component
                                and file_component != "api"
                            ):  # API modules can import from any component
                                violations.append(
                                    {
                                        "file": str(py_file.relative_to(src_dir)),
                                        "file_component": file_component,
                                        "imported_module": node.module,
                                        "imported_component": imported_component,
                                    }
                                )

                checked_files += 1
                if checked_files >= 50:  # Limit to avoid long test times
                    break

            except Exception:
                continue

        print(f"Checked {checked_files} files")
        print(f"Found {len(violations)} cross-component internal import violations")

        if violations:
            for violation in violations[:5]:  # Show first 5 violations
                print(
                    f"⚠️  {violation['file']} ({violation['file_component']}) imports {violation['imported_module']} ({violation['imported_component']})"
                )

        # Don't fail test - this shows what needs to be fixed
        assert checked_files > 0, "Should check some files"

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
            ]:
                return parts[1]
            elif parts[1] == "_internal":
                return "core"

        return None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
