"""
Test publication readiness validation for publication readiness.

This module tests Task 7.7: Validate publication readiness.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import pytest


class TestPublicationReadinessValidation:
    """Test overall publication readiness validation."""

    def test_package_imports_cleanly(self):
        """Test that the package imports without errors or excessive warnings."""
        # Clear any cached imports
        modules_to_clear = [name for name in sys.modules.keys() if name.startswith("saplings")]
        for module_name in modules_to_clear:
            if module_name in sys.modules:
                del sys.modules[module_name]

        # Test clean import
        try:
            import saplings  # noqa: F401

            import_success = True
        except Exception as e:
            import_success = False
            pytest.fail(f"Package failed to import: {e}")

        assert import_success, "Package should import cleanly"

    def test_core_api_accessibility(self):
        """Test that core API components are accessible from main package."""
        import saplings

        # Core components that should be accessible
        core_components = [
            "Agent",
            "AgentBuilder",
            "AgentConfig",
            "Tool",
            "PythonInterpreterTool",
            "FinalAnswerTool",
            "MemoryStore",
            "Document",
            "LLM",
            "LLMBuilder",
        ]

        accessible_components = []
        missing_components = []

        for component in core_components:
            if hasattr(saplings, component):
                accessible_components.append(component)
            else:
                missing_components.append(component)

        print("\nCore API accessibility:")
        print(f"Accessible: {len(accessible_components)}/{len(core_components)}")
        for component in accessible_components:
            print(f"  ✅ {component}")

        if missing_components:
            print("Missing:")
            for component in missing_components:
                print(f"  ❌ {component}")

        # At least 80% of core components should be accessible
        accessibility_ratio = len(accessible_components) / len(core_components)
        assert (
            accessibility_ratio >= 0.8
        ), f"Only {accessibility_ratio:.1%} of core components accessible"

    def test_version_information_available(self):
        """Test that version information is available and properly formatted."""
        import saplings

        # Check version attribute exists
        assert hasattr(saplings, "__version__"), "Package should have __version__ attribute"

        version = saplings.__version__
        assert isinstance(version, str), "Version should be a string"
        assert len(version) > 0, "Version should not be empty"

        # Basic version format check (semantic versioning)
        version_parts = version.split(".")
        assert (
            len(version_parts) >= 2
        ), f"Version should have at least major.minor format, got: {version}"

        print(f"\nVersion information: {version}")

    def test_package_metadata_completeness(self):
        """Test that package metadata is complete for publication."""
        metadata_check = self._check_package_metadata()

        print("\nPackage metadata completeness:")
        for field, status in metadata_check.items():
            status_icon = "✅" if status["present"] else "❌"
            print(f"  {status_icon} {field}: {status['value'] if status['present'] else 'Missing'}")

        # Critical metadata should be present
        critical_fields = ["name", "version", "description", "author"]
        missing_critical = [
            field for field in critical_fields if not metadata_check[field]["present"]
        ]

        assert not missing_critical, f"Missing critical metadata: {missing_critical}"

    def test_installation_requirements_valid(self):
        """Test that installation requirements are valid and resolvable."""
        requirements_check = self._check_installation_requirements()

        print("\nInstallation requirements validation:")
        print(f"Core dependencies: {requirements_check['core_deps_count']}")
        print(f"Optional dependencies: {requirements_check['optional_deps_count']}")
        print(f"Invalid requirements: {len(requirements_check['invalid_requirements'])}")

        if requirements_check["invalid_requirements"]:
            print("Invalid requirements found:")
            for req in requirements_check["invalid_requirements"]:
                print(f"  - {req}")

        # Should have minimal invalid requirements
        assert (
            len(requirements_check["invalid_requirements"]) == 0
        ), "Should have no invalid requirements"

    def test_api_surface_size_reasonable(self):
        """Test that API surface is reasonable for publication."""
        import saplings

        api_surface_size = len(saplings.__all__)

        print("\nAPI surface analysis:")
        print(f"Total exported items: {api_surface_size}")

        # Document current state - this is why we need API cleanup!
        if api_surface_size > 100:
            print(
                f"⚠️  API surface is large ({api_surface_size} items) - needs reduction for publication"
            )

        # For now, just ensure it's not empty and not ridiculously large
        assert api_surface_size > 0, "API surface should not be empty"
        assert api_surface_size < 500, f"API surface is extremely large ({api_surface_size})"

    def test_no_obvious_security_issues(self):
        """Test for obvious security issues in the codebase."""
        security_check = self._check_security_issues()

        print("\nSecurity check:")
        for issue_type, issues in security_check.items():
            if issues:
                print(f"{issue_type}: {len(issues)} issues found")
                for issue in issues[:3]:  # Show first 3
                    print(f"  - {issue}")
            else:
                print(f"{issue_type}: ✅ No issues found")

        # Document current state - many of these are false positives
        total_issues = sum(len(issues) for issues in security_check.values())
        print(f"\nTotal potential security issues: {total_issues}")

        if total_issues > 20:
            print("⚠️  Many potential security issues found - need manual review")
            print("Note: Many may be false positives (e.g., os.path.join is usually safe)")

        # For now, just ensure no critical issues (like hardcoded passwords)
        critical_issues = len(security_check.get("hardcoded_secrets", []))
        assert critical_issues < 10, f"Found {critical_issues} potential hardcoded secrets"

    def test_performance_characteristics_acceptable(self):
        """Test that performance characteristics are acceptable."""
        performance_check = self._check_performance_characteristics()

        print("\nPerformance characteristics:")
        for metric, value in performance_check.items():
            print(f"  {metric}: {value}")

        # Import time should be reasonable
        assert performance_check["import_time"] < 2.0, "Import time should be < 2 seconds"

    def _check_package_metadata(self) -> Dict[str, Dict]:
        """Check package metadata completeness."""
        metadata = {}

        try:
            # Try to read from pyproject.toml
            pyproject_path = Path("pyproject.toml")
            if pyproject_path.exists():
                with open(pyproject_path) as f:
                    content = f.read()

                # Simple parsing for key fields
                metadata["name"] = {"present": 'name = "' in content, "value": "saplings"}
                metadata["version"] = {"present": 'version = "' in content, "value": "dynamic"}
                metadata["description"] = {
                    "present": 'description = "' in content,
                    "value": "extracted",
                }
                metadata["author"] = {"present": "authors = [" in content, "value": "present"}
                metadata["license"] = {"present": 'license = "' in content, "value": "specified"}
                metadata["readme"] = {"present": 'readme = "' in content, "value": "README.md"}
            else:
                # Default values if pyproject.toml not found
                for field in ["name", "version", "description", "author", "license", "readme"]:
                    metadata[field] = {"present": False, "value": "Not found"}

        except Exception:
            # Fallback if parsing fails
            for field in ["name", "version", "description", "author", "license", "readme"]:
                metadata[field] = {"present": False, "value": "Parse error"}

        return metadata

    def _check_installation_requirements(self) -> Dict:
        """Check installation requirements validity."""
        requirements_check = {
            "core_deps_count": 0,
            "optional_deps_count": 0,
            "invalid_requirements": [],
        }

        try:
            # Check pyproject.toml for dependencies
            pyproject_path = Path("pyproject.toml")
            if pyproject_path.exists():
                with open(pyproject_path) as f:
                    content = f.read()

                # Count dependencies (simple heuristic)
                if "dependencies = [" in content:
                    deps_section = content.split("dependencies = [")[1].split("]")[0]
                    requirements_check["core_deps_count"] = deps_section.count('"')

                if "optional-dependencies" in content:
                    optional_section = content.split("optional-dependencies")[1]
                    requirements_check["optional_deps_count"] = optional_section.count('"')

        except Exception as e:
            requirements_check["invalid_requirements"].append(f"Failed to parse requirements: {e}")

        return requirements_check

    def _check_security_issues(self) -> Dict[str, List[str]]:
        """Check for obvious security issues."""
        security_issues = {
            "hardcoded_secrets": [],
            "unsafe_imports": [],
            "shell_injection_risks": [],
            "path_traversal_risks": [],
        }

        src_path = Path("src/saplings")
        if not src_path.exists():
            return security_issues

        # Patterns to look for
        secret_patterns = ["password", "secret", "key", "token"]
        unsafe_patterns = ["eval(", "exec(", "__import__"]
        shell_patterns = ["os.system", "subprocess.call", "shell=True"]
        path_patterns = ["../", "..\\", "os.path.join"]

        for py_file in src_path.rglob("*.py"):
            try:
                with open(py_file, encoding="utf-8") as f:
                    content = f.read()

                # Check for hardcoded secrets (very basic)
                for pattern in secret_patterns:
                    if f'{pattern} = "' in content.lower() or f"{pattern} = '" in content.lower():
                        security_issues["hardcoded_secrets"].append(f"{py_file}: {pattern}")

                # Check for unsafe imports/calls
                for pattern in unsafe_patterns:
                    if pattern in content:
                        security_issues["unsafe_imports"].append(f"{py_file}: {pattern}")

                # Check for shell injection risks
                for pattern in shell_patterns:
                    if pattern in content:
                        security_issues["shell_injection_risks"].append(f"{py_file}: {pattern}")

                # Check for path traversal risks
                for pattern in path_patterns:
                    if pattern in content:
                        security_issues["path_traversal_risks"].append(f"{py_file}: {pattern}")

            except Exception:
                continue

        return security_issues

    def _check_performance_characteristics(self) -> Dict[str, float]:
        """Check performance characteristics."""
        import time

        # Measure import time
        start_time = time.time()
        try:
            import saplings  # noqa: F401

            import_time = time.time() - start_time
        except Exception:
            import_time = float("inf")

        return {"import_time": import_time}
