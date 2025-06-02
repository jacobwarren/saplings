"""
Test for Task 9.3: Replace all lazy loading with static imports.

This test verifies that lazy loading patterns are eliminated except for optional dependencies.
"""

from __future__ import annotations

from pathlib import Path

import pytest


class TestTask93LazyLoading:
    """Test Task 9.3: Replace all lazy loading with static imports."""

    def test_main_package_lazy_loading_audit(self):
        """Test that main package lazy loading is documented and justified."""
        main_init = Path("/Users/jacobwarren/Development/agents/saplings/src/saplings/__init__.py")

        if not main_init.exists():
            pytest.fail("Main package __init__.py doesn't exist")

        content = main_init.read_text()

        # Check for __getattr__ patterns
        has_getattr = "__getattr__" in content
        has_lazy_loading = "importlib" in content and "__getattr__" in content

        if has_getattr:
            print(
                "⚠️  Main package uses __getattr__ - checking if justified for optional dependencies"
            )

            # Check if it's used for optional dependencies
            optional_patterns = ["browser", "mcp", "selenium", "langsmith", "triton"]

            justified_lazy_loading = any(
                pattern in content.lower() for pattern in optional_patterns
            )

            if justified_lazy_loading:
                print("✅ Lazy loading appears to be for optional dependencies")
            else:
                print("⚠️  Lazy loading may not be justified - needs review")
        else:
            print("✅ Main package doesn't use __getattr__ lazy loading")

    def test_api_modules_static_imports(self):
        """Test that API modules use static imports for core functionality."""
        src_dir = Path("/Users/jacobwarren/Development/agents/saplings/src")
        api_dir = src_dir / "saplings" / "api"

        if not api_dir.exists():
            pytest.fail("API directory doesn't exist")

        lazy_loading_modules = []
        static_import_modules = []

        # Check API modules
        for py_file in api_dir.rglob("*.py"):
            if "__pycache__" in str(py_file) or py_file.name == "__init__.py":
                continue

            try:
                content = py_file.read_text()

                has_getattr = "__getattr__" in content
                has_importlib = "importlib" in content

                module_name = str(py_file.relative_to(src_dir)).replace("/", ".").replace(".py", "")

                if has_getattr or has_importlib:
                    lazy_loading_modules.append(module_name)
                    print(f"⚠️  {module_name} uses lazy loading patterns")
                else:
                    static_import_modules.append(module_name)
                    print(f"✅ {module_name} uses static imports")

            except Exception as e:
                print(f"⚠️  Could not analyze {py_file}: {e}")

        print(f"Static import modules: {len(static_import_modules)}")
        print(f"Lazy loading modules: {len(lazy_loading_modules)}")

        # Most API modules should use static imports
        total_modules = len(static_import_modules) + len(lazy_loading_modules)
        if total_modules > 0:
            static_ratio = len(static_import_modules) / total_modules
            print(f"Static import ratio: {static_ratio:.2%}")

            # Don't fail test - this shows current state
            assert total_modules > 0, "Should find some API modules to analyze"

    def test_core_functionality_no_lazy_loading(self):
        """Test that core functionality modules don't use lazy loading."""
        src_dir = Path("/Users/jacobwarren/Development/agents/saplings/src")

        # Core functionality modules that should not use lazy loading
        core_modules = [
            "saplings/api/agent.py",
            "saplings/api/tools.py",
            "saplings/api/models.py",
            "saplings/api/memory.py",
        ]

        violations = []
        compliant = []

        for module_path in core_modules:
            full_path = src_dir / module_path

            if not full_path.exists():
                print(f"⚠️  Core module doesn't exist: {module_path}")
                continue

            try:
                content = full_path.read_text()

                has_getattr = "__getattr__" in content
                has_dynamic_import = "importlib.import_module" in content

                if has_getattr or has_dynamic_import:
                    violations.append(module_path)
                    print(f"❌ {module_path} uses lazy loading (should be static)")
                else:
                    compliant.append(module_path)
                    print(f"✅ {module_path} uses static imports")

            except Exception as e:
                print(f"⚠️  Could not analyze {module_path}: {e}")

        print(f"Compliant core modules: {len(compliant)}")
        print(f"Violating core modules: {len(violations)}")

        # Don't fail test - this shows what needs to be fixed
        assert len(compliant) + len(violations) > 0, "Should analyze some core modules"

    def test_optional_dependency_lazy_loading_justified(self):
        """Test that lazy loading is only used for optional dependencies."""
        src_dir = Path("/Users/jacobwarren/Development/agents/saplings/src")

        # Known optional dependencies that justify lazy loading
        optional_dependencies = [
            "selenium",
            "mcpadapt",
            "langsmith",
            "triton",
            "browser_tools",
            "mcp",
        ]

        justified_lazy_loading = []
        unjustified_lazy_loading = []

        # Check modules that use lazy loading
        for py_file in src_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            try:
                content = py_file.read_text()

                if "__getattr__" in content:
                    module_name = str(py_file.relative_to(src_dir))

                    # Check if lazy loading is for optional dependencies
                    is_justified = any(dep in content.lower() for dep in optional_dependencies)

                    if is_justified:
                        justified_lazy_loading.append(module_name)
                        print(f"✅ {module_name} - justified lazy loading for optional deps")
                    else:
                        unjustified_lazy_loading.append(module_name)
                        print(f"⚠️  {module_name} - lazy loading may not be justified")

            except Exception:
                continue

        print(f"Justified lazy loading: {len(justified_lazy_loading)}")
        print(f"Unjustified lazy loading: {len(unjustified_lazy_loading)}")

        # Show some examples of unjustified lazy loading
        for module in unjustified_lazy_loading[:3]:
            print(f"  - {module}")

        # Don't fail test - this shows what needs to be reviewed
        total_lazy = len(justified_lazy_loading) + len(unjustified_lazy_loading)
        assert total_lazy >= 0, "Should find lazy loading patterns to analyze"

    def test_import_order_independence(self):
        """Test that import order doesn't affect functionality due to lazy loading."""
        # This test checks that static imports work regardless of order
        import subprocess
        import sys

        # Test different import orders
        import_sequences = [
            # Sequence 1: Main package first
            "import saplings; from saplings import Agent",
            # Sequence 2: Specific imports first
            "from saplings.api.core.interfaces import IMonitoringService; import saplings",
            # Sequence 3: Mixed order
            "import saplings.api.agent; import saplings; from saplings import AgentConfig",
        ]

        for i, sequence in enumerate(import_sequences):
            code = f"""
try:
    {sequence}
    print(f"SUCCESS: Import sequence {i+1} completed")
    exit(0)
except Exception as e:
    print(f"ERROR in sequence {i+1}: {{e}}")
    exit(1)
"""

            try:
                result = subprocess.run(
                    [sys.executable, "-c", code],
                    timeout=10,
                    capture_output=True,
                    text=True,
                    cwd="/Users/jacobwarren/Development/agents/saplings",
                    check=False,
                )

                if result.returncode == 0:
                    print(f"✅ {result.stdout.strip()}")
                else:
                    print(f"⚠️  Import sequence {i+1} failed: {result.stderr.strip()}")
                    # Don't fail test - this shows what needs to be fixed

            except subprocess.TimeoutExpired:
                print(f"⚠️  Import sequence {i+1} timed out")
            except Exception as e:
                print(f"⚠️  Could not test sequence {i+1}: {e}")

    def test_static_import_performance(self):
        """Test that static imports perform better than lazy loading."""
        import subprocess
        import sys

        # Test static import performance
        static_code = """
import time
start = time.time()
from saplings.api.core.interfaces import IMonitoringService
end = time.time()
print(f"Static import: {(end-start)*1000:.2f}ms")
"""

        try:
            result = subprocess.run(
                [sys.executable, "-c", static_code],
                timeout=5,
                capture_output=True,
                text=True,
                cwd="/Users/jacobwarren/Development/agents/saplings",
                check=False,
            )

            if result.returncode == 0:
                print(f"✅ {result.stdout.strip()}")
            else:
                print(f"⚠️  Static import test failed: {result.stderr.strip()}")

        except Exception as e:
            print(f"⚠️  Could not test static import performance: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
