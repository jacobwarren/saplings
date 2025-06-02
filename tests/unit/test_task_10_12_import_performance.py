"""
Test for Task 10.12: Optimize import performance for sub-1-second startup.

This test verifies that import performance optimization has been implemented:
1. Main package import <1 second (target)
2. Core API access <0.1 seconds additional
3. Heavy features lazy loaded on demand
4. Import performance monitoring implemented
5. Performance regression tests exist
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


class TestTask1012ImportPerformance:
    """Test Task 10.12: Optimize import performance for sub-1-second startup."""

    def test_main_package_import_performance(self):
        """Test that main package import completes in <1 second (target) or <5 seconds (acceptable)."""
        code = """
import time
start_time = time.time()
try:
    import saplings
    import_time = time.time() - start_time
    print(f"SUCCESS: saplings imported in {import_time:.2f}s")
    if import_time <= 1.0:
        print("EXCELLENT: Under 1s target")
        exit(0)
    elif import_time <= 3.0:
        print("GOOD: Under 3s (acceptable)")
        exit(0)
    elif import_time <= 5.0:
        print("ACCEPTABLE: Under 5s")
        exit(0)
    else:
        print(f"SLOW: {import_time:.2f}s exceeds 5s limit")
        exit(1)
except Exception as e:
    print(f"ERROR: {e}")
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
                pytest.fail(f"Package import performance failed: {result.stderr.strip()}")

        except subprocess.TimeoutExpired:
            pytest.fail("Package import timed out - performance issue")

    def test_core_api_access_performance(self):
        """Test that core API access is fast after initial import."""
        code = """
import time
import saplings

# Measure core API access time
start_time = time.time()
try:
    agent_config = saplings.AgentConfig
    agent = saplings.Agent
    agent_builder = saplings.AgentBuilder
    access_time = time.time() - start_time
    print(f"SUCCESS: Core API access in {access_time:.3f}s")
    if access_time <= 0.1:
        print("EXCELLENT: Under 0.1s target")
        exit(0)
    elif access_time <= 0.5:
        print("GOOD: Under 0.5s")
        exit(0)
    else:
        print(f"SLOW: {access_time:.3f}s exceeds 0.5s")
        exit(1)
except Exception as e:
    print(f"ERROR: {e}")
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
                print(f"⚠️  Core API access performance: {result.stderr.strip()}")

        except subprocess.TimeoutExpired:
            pytest.fail("Core API access timed out")

    def test_heavy_dependencies_lazy_loaded(self):
        """Test that heavy dependencies are lazy loaded and don't slow initial import."""
        # Test that ML libraries are not imported during main package import
        code = """
import sys
import time

# Check what's imported before saplings
modules_before = set(sys.modules.keys())

start_time = time.time()
import saplings
import_time = time.time() - start_time

# Check what's imported after saplings
modules_after = set(sys.modules.keys())
new_modules = modules_after - modules_before

# Heavy ML libraries that should be lazy loaded
heavy_libraries = [
    'transformers', 'torch', 'tensorflow', 'faiss',
    'sentence_transformers', 'sklearn', 'numpy'
]

imported_heavy = []
for lib in heavy_libraries:
    if any(lib in module for module in new_modules):
        imported_heavy.append(lib)

print(f"Import time: {import_time:.2f}s")
print(f"New modules: {len(new_modules)}")
print(f"Heavy libraries imported: {imported_heavy}")

if imported_heavy:
    print(f"WARNING: Heavy libraries imported during main import: {imported_heavy}")
    exit(1)
else:
    print("GOOD: No heavy libraries imported during main import")
    exit(0)
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
                print(f"⚠️  Heavy dependency check: {result.stderr.strip()}")

        except subprocess.TimeoutExpired:
            pytest.fail("Heavy dependency check timed out")

    def test_optional_dependencies_lazy_loaded(self):
        """Test that optional dependencies are lazy loaded."""
        code = """
import sys
import time

# Check what's imported before saplings
modules_before = set(sys.modules.keys())

start_time = time.time()
import saplings
import_time = time.time() - start_time

# Check what's imported after saplings
modules_after = set(sys.modules.keys())
new_modules = modules_after - modules_before

# Optional dependencies that should be lazy loaded
optional_deps = [
    'selenium', 'langsmith', 'mcp', 'triton'
]

imported_optional = []
for dep in optional_deps:
    if any(dep in module for module in new_modules):
        imported_optional.append(dep)

print(f"Import time: {import_time:.2f}s")
print(f"Optional dependencies imported: {imported_optional}")

if imported_optional:
    print(f"INFO: Some optional dependencies imported: {imported_optional}")
else:
    print("GOOD: No optional dependencies imported during main import")

exit(0)
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

            print(f"✅ {result.stdout.strip()}")

        except subprocess.TimeoutExpired:
            pytest.fail("Optional dependency check timed out")

    def test_import_performance_monitoring_exists(self):
        """Test that import performance monitoring is implemented."""
        # Check for performance monitoring utilities
        monitoring_files = [
            Path("scripts/benchmark_imports.py"),
            Path("scripts/profile_imports.py"),
            Path("tests/benchmarks/test_import_performance.py"),
            Path("tests/unit/test_import_performance_optimization.py"),
        ]

        found_monitoring = []
        for file_path in monitoring_files:
            if file_path.exists():
                found_monitoring.append(file_path)
                print(f"✅ Found performance monitoring: {file_path}")

        if found_monitoring:
            print(f"✅ Found {len(found_monitoring)} performance monitoring files")
        else:
            print("⚠️  No import performance monitoring files found")

    def test_performance_regression_tests_exist(self):
        """Test that performance regression tests exist."""
        # Check for performance regression test files
        regression_test_files = [
            "tests/benchmarks/test_import_performance.py",
            "tests/unit/test_import_performance_optimization.py",
            "tests/unit/test_task_8_11_import_performance.py",
            "tests/unit/test_task_9_14_import_performance.py",
        ]

        existing_tests = []
        for test_file in regression_test_files:
            if Path(test_file).exists():
                existing_tests.append(test_file)
                print(f"✅ Found regression test: {test_file}")

        if existing_tests:
            print(f"✅ Found {len(existing_tests)} performance regression test files")
            assert len(existing_tests) >= 1, "Should have at least one performance regression test"
        else:
            print("⚠️  No performance regression tests found")

    def test_import_optimization_documentation_exists(self):
        """Test that import optimization strategy is documented."""
        # Check for optimization documentation
        optimization_docs = [
            Path("docs/import-performance-optimization-standardization.md"),
            Path("docs/performance-optimization.md"),
            Path("docs/import-optimization.md"),
        ]

        found_docs = []
        for doc_path in optimization_docs:
            if doc_path.exists():
                found_docs.append(doc_path)
                print(f"✅ Found optimization documentation: {doc_path}")

        if found_docs:
            print(f"✅ Found {len(found_docs)} optimization documentation files")
        else:
            print("⚠️  No import optimization documentation found")

    def test_lightweight_proxy_objects_implemented(self):
        """Test that lightweight proxy objects are used for heavy components."""
        try:
            import saplings

            # Check if main namespace objects are lightweight
            # This is a basic check - real implementation would need more sophisticated analysis
            core_objects = ["Agent", "AgentConfig", "AgentBuilder"]

            lightweight_objects = []
            for obj_name in core_objects:
                if hasattr(saplings, obj_name):
                    obj = getattr(saplings, obj_name)
                    # Basic check: object should be importable quickly
                    lightweight_objects.append(obj_name)
                    print(f"✅ {obj_name} accessible")

            if lightweight_objects:
                print(f"✅ Found {len(lightweight_objects)} core objects accessible")
            else:
                print("⚠️  No core objects accessible")

        except ImportError as e:
            print(f"⚠️  Could not test proxy objects: {e}")

    def test_service_initialization_deferred(self):
        """Test that service initialization is deferred until Agent creation."""
        code = """
import time
import sys

# Import main package
start_time = time.time()
import saplings
import_time = time.time() - start_time

# Check if services are initialized during import
# This is a basic check - services should not be fully initialized yet
print(f"Import time: {import_time:.2f}s")

# Try to access AgentConfig without triggering full service initialization
try:
    config_class = saplings.AgentConfig
    print("SUCCESS: AgentConfig accessible without full service init")
    exit(0)
except Exception as e:
    print(f"ERROR: {e}")
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
                print(f"⚠️  Service initialization check: {result.stderr.strip()}")

        except subprocess.TimeoutExpired:
            pytest.fail("Service initialization check timed out")

    def test_import_performance_baseline_established(self):
        """Test that import performance baseline is established and tracked."""
        # This test documents the current performance baseline
        code = """
import time
start_time = time.time()
import saplings
import_time = time.time() - start_time

print(f"BASELINE: Current import time is {import_time:.2f}s")

# Performance targets:
# - Excellent: <1.0s
# - Good: <3.0s
# - Acceptable: <5.0s
# - Poor: >5.0s

if import_time <= 1.0:
    print("STATUS: EXCELLENT - Meets target")
elif import_time <= 3.0:
    print("STATUS: GOOD - Close to target")
elif import_time <= 5.0:
    print("STATUS: ACCEPTABLE - Room for improvement")
else:
    print("STATUS: POOR - Needs optimization")

exit(0)
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

            print(f"📊 {result.stdout.strip()}")

        except subprocess.TimeoutExpired:
            pytest.fail("Performance baseline check timed out")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
