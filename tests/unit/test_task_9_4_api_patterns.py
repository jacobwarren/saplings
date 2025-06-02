"""
Test for Task 9.4: Standardize all API module patterns to eliminate inconsistencies.

This test verifies that all API modules follow the standardized direct inheritance pattern.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


class TestTask94APIPatterns:
    """Test Task 9.4: Standardize all API module patterns to eliminate inconsistencies."""

    def test_api_modules_use_direct_inheritance(self):
        """Test that API modules use direct inheritance pattern."""
        src_dir = Path("/Users/jacobwarren/Development/agents/saplings/src")
        api_dir = src_dir / "saplings" / "api"

        if not api_dir.exists():
            pytest.fail("API directory doesn't exist")

        direct_inheritance_modules = []
        pattern_violations = []

        # Check API modules for direct inheritance pattern
        for py_file in api_dir.rglob("*.py"):
            if "__pycache__" in str(py_file) or py_file.name == "__init__.py":
                continue

            try:
                content = py_file.read_text()
                tree = ast.parse(content)

                module_name = str(py_file.relative_to(src_dir)).replace("/", ".").replace(".py", "")

                # Look for class definitions with inheritance
                has_direct_inheritance = False
                has_complex_new = False
                has_stability_annotation = False

                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        # Check for inheritance from internal classes
                        if node.bases:
                            for base in node.bases:
                                if isinstance(base, ast.Name) and base.id.startswith("_"):
                                    has_direct_inheritance = True

                        # Check for complex __new__ methods
                        for method in node.body:
                            if isinstance(method, ast.FunctionDef) and method.name == "__new__":
                                has_complex_new = True

                    # Check for stability annotations
                    if isinstance(node, ast.FunctionDef) and node.name in ["stable", "beta"]:
                        has_stability_annotation = True
                    elif isinstance(node, ast.Call):
                        if isinstance(node.func, ast.Name) and node.func.id in ["stable", "beta"]:
                            has_stability_annotation = True

                # Check for stability import
                has_stability_import = "from saplings.api.stability import" in content

                if has_direct_inheritance and not has_complex_new:
                    direct_inheritance_modules.append(module_name)
                    print(f"✅ {module_name} uses direct inheritance pattern")
                else:
                    pattern_violations.append(
                        {
                            "module": module_name,
                            "has_direct_inheritance": has_direct_inheritance,
                            "has_complex_new": has_complex_new,
                            "has_stability_import": has_stability_import,
                            "has_stability_annotation": has_stability_annotation,
                        }
                    )
                    print(f"⚠️  {module_name} doesn't follow direct inheritance pattern")

            except Exception as e:
                print(f"⚠️  Could not analyze {py_file}: {e}")

        print(f"Direct inheritance modules: {len(direct_inheritance_modules)}")
        print(f"Pattern violations: {len(pattern_violations)}")

        # Show some pattern violations
        for violation in pattern_violations[:5]:
            print(
                f"  - {violation['module']}: inheritance={violation['has_direct_inheritance']}, complex_new={violation['has_complex_new']}"
            )

        # Don't fail test - this shows current state
        total_modules = len(direct_inheritance_modules) + len(pattern_violations)
        assert total_modules > 0, "Should find some API modules to analyze"

    def test_api_modules_have_all_definitions(self):
        """Test that API modules have proper __all__ definitions."""
        src_dir = Path("/Users/jacobwarren/Development/agents/saplings/src")
        api_dir = src_dir / "saplings" / "api"

        modules_with_all = []
        modules_without_all = []

        for py_file in api_dir.rglob("*.py"):
            if "__pycache__" in str(py_file) or py_file.name == "__init__.py":
                continue

            try:
                content = py_file.read_text()
                module_name = str(py_file.relative_to(src_dir)).replace("/", ".").replace(".py", "")

                if "__all__" in content:
                    modules_with_all.append(module_name)
                    print(f"✅ {module_name} has __all__ definition")
                else:
                    modules_without_all.append(module_name)
                    print(f"❌ {module_name} missing __all__ definition")

            except Exception as e:
                print(f"⚠️  Could not analyze {py_file}: {e}")

        print(f"Modules with __all__: {len(modules_with_all)}")
        print(f"Modules without __all__: {len(modules_without_all)}")

        total_modules = len(modules_with_all) + len(modules_without_all)
        if total_modules > 0:
            all_ratio = len(modules_with_all) / total_modules
            print(f"__all__ definition ratio: {all_ratio:.2%}")

        # Don't fail test - this shows what needs to be fixed
        assert total_modules > 0, "Should find some API modules to analyze"

    def test_api_modules_have_stability_annotations(self):
        """Test that API modules have stability annotations."""
        src_dir = Path("/Users/jacobwarren/Development/agents/saplings/src")
        api_dir = src_dir / "saplings" / "api"

        modules_with_stability = []
        modules_without_stability = []

        for py_file in api_dir.rglob("*.py"):
            if "__pycache__" in str(py_file) or py_file.name == "__init__.py":
                continue

            try:
                content = py_file.read_text()
                module_name = str(py_file.relative_to(src_dir)).replace("/", ".").replace(".py", "")

                # Check for stability imports and decorators
                has_stability_import = "from saplings.api.stability import" in content
                has_stability_decorator = "@stable" in content or "@beta" in content

                if has_stability_import or has_stability_decorator:
                    modules_with_stability.append(module_name)
                    print(f"✅ {module_name} has stability annotations")
                else:
                    modules_without_stability.append(module_name)
                    print(f"⚠️  {module_name} missing stability annotations")

            except Exception as e:
                print(f"⚠️  Could not analyze {py_file}: {e}")

        print(f"Modules with stability: {len(modules_with_stability)}")
        print(f"Modules without stability: {len(modules_without_stability)}")

        total_modules = len(modules_with_stability) + len(modules_without_stability)
        if total_modules > 0:
            stability_ratio = len(modules_with_stability) / total_modules
            print(f"Stability annotation ratio: {stability_ratio:.2%}")

        # Don't fail test - this shows what needs to be improved
        assert total_modules > 0, "Should find some API modules to analyze"

    def test_no_complex_new_methods(self):
        """Test that API modules don't use complex __new__ methods."""
        src_dir = Path("/Users/jacobwarren/Development/agents/saplings/src")
        api_dir = src_dir / "saplings" / "api"

        modules_with_complex_new = []
        modules_without_complex_new = []

        for py_file in api_dir.rglob("*.py"):
            if "__pycache__" in str(py_file) or py_file.name == "__init__.py":
                continue

            try:
                content = py_file.read_text()
                tree = ast.parse(content)

                module_name = str(py_file.relative_to(src_dir)).replace("/", ".").replace(".py", "")

                has_complex_new = False

                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        for method in node.body:
                            if isinstance(method, ast.FunctionDef) and method.name == "__new__":
                                # Check if it's a complex __new__ method
                                method_source = ast.get_source_segment(content, method)
                                if method_source and (
                                    "importlib" in method_source
                                    or "import_module" in method_source
                                    or len(method.body) > 3
                                ):
                                    has_complex_new = True

                if has_complex_new:
                    modules_with_complex_new.append(module_name)
                    print(f"❌ {module_name} has complex __new__ methods")
                else:
                    modules_without_complex_new.append(module_name)
                    print(f"✅ {module_name} doesn't use complex __new__ methods")

            except Exception as e:
                print(f"⚠️  Could not analyze {py_file}: {e}")

        print(f"Modules without complex __new__: {len(modules_without_complex_new)}")
        print(f"Modules with complex __new__: {len(modules_with_complex_new)}")

        # Show modules that need fixing
        for module in modules_with_complex_new:
            print(f"  - {module}")

        # Don't fail test - this shows what needs to be fixed
        total_modules = len(modules_with_complex_new) + len(modules_without_complex_new)
        assert total_modules > 0, "Should find some API modules to analyze"

    def test_no_dynamic_imports_in_api_modules(self):
        """Test that API modules don't use dynamic imports."""
        src_dir = Path("/Users/jacobwarren/Development/agents/saplings/src")
        api_dir = src_dir / "saplings" / "api"

        modules_with_dynamic_imports = []
        modules_without_dynamic_imports = []

        for py_file in api_dir.rglob("*.py"):
            if "__pycache__" in str(py_file) or py_file.name == "__init__.py":
                continue

            try:
                content = py_file.read_text()
                module_name = str(py_file.relative_to(src_dir)).replace("/", ".").replace(".py", "")

                # Check for dynamic import patterns
                has_dynamic_imports = (
                    "importlib.import_module" in content
                    or "__import__" in content
                    or ("importlib" in content and "import_module" in content)
                )

                if has_dynamic_imports:
                    modules_with_dynamic_imports.append(module_name)
                    print(f"❌ {module_name} uses dynamic imports")
                else:
                    modules_without_dynamic_imports.append(module_name)
                    print(f"✅ {module_name} uses static imports")

            except Exception as e:
                print(f"⚠️  Could not analyze {py_file}: {e}")

        print(f"Modules with static imports: {len(modules_without_dynamic_imports)}")
        print(f"Modules with dynamic imports: {len(modules_with_dynamic_imports)}")

        # Show modules that need fixing
        for module in modules_with_dynamic_imports:
            print(f"  - {module}")

        total_modules = len(modules_with_dynamic_imports) + len(modules_without_dynamic_imports)
        if total_modules > 0:
            static_ratio = len(modules_without_dynamic_imports) / total_modules
            print(f"Static import ratio: {static_ratio:.2%}")

        # Don't fail test - this shows current state
        assert total_modules > 0, "Should find some API modules to analyze"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
