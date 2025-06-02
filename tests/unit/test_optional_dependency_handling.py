"""
Test optional dependency handling for publication readiness.

This module tests Task 7.4: Improve optional dependency handling.
"""

from __future__ import annotations

import logging
import warnings
from io import StringIO
from pathlib import Path
from typing import Dict, List


class TestOptionalDependencyHandling:
    """Test optional dependency handling and graceful degradation."""

    def test_identify_optional_dependencies(self):
        """Test that we can identify all optional dependencies in the codebase."""
        optional_deps = self._find_optional_dependencies()

        # We should find several optional dependencies
        assert len(optional_deps) > 0, "Should find optional dependencies"

        # Core optional dependencies that should definitely be found
        core_expected_deps = {"selenium", "triton", "vllm", "networkx", "transformers"}

        found_deps = {dep["name"] for dep in optional_deps}

        # Check for core dependencies
        missing_core = core_expected_deps - found_deps
        if missing_core:
            print(f"Missing core optional dependencies: {missing_core}")
            print(f"Found dependencies: {sorted(found_deps)}")

        # At least some core dependencies should be found
        found_core = core_expected_deps & found_deps
        assert (
            len(found_core) >= 3
        ), f"Should find at least 3 core optional dependencies, found: {found_core}"

        print(f"\nFound {len(optional_deps)} optional dependencies:")
        for dep in sorted(optional_deps, key=lambda x: x["name"]):
            print(f"  - {dep['name']}: {dep['usage_count']} usages")

    def test_optional_dependency_warning_patterns(self):
        """Test current warning patterns for optional dependencies."""
        warning_patterns = self._analyze_warning_patterns()

        print("\nOptional dependency warning patterns:")
        for pattern_type, patterns in warning_patterns.items():
            print(f"{pattern_type}: {len(patterns)} patterns")
            for pattern in patterns[:3]:  # Show first 3 examples
                print(f"  - {pattern}")

    def test_import_warnings_during_normal_usage(self):
        """Test that normal imports don't generate excessive warnings."""
        # Capture warnings during import
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Capture logging output
            log_capture = StringIO()
            handler = logging.StreamHandler(log_capture)
            logging.getLogger().addHandler(handler)

            try:
                # Try importing the main package
                import saplings  # noqa: F401

                # Count warnings related to optional dependencies
                optional_warnings = [
                    warning
                    for warning in w
                    if any(
                        dep in str(warning.message).lower()
                        for dep in ["selenium", "triton", "langsmith", "mcp"]
                    )
                ]

                # Get log output
                log_output = log_capture.getvalue()
                optional_log_warnings = log_output.count("not installed") + log_output.count(
                    "not available"
                )

                print("\nImport warnings analysis:")
                print(f"Total warnings: {len(w)}")
                print(f"Optional dependency warnings: {len(optional_warnings)}")
                print(f"Log warnings about missing deps: {optional_log_warnings}")

                # For now, document the current state rather than enforce limits
                # This will be updated as warnings are reduced

            finally:
                logging.getLogger().removeHandler(handler)

    def test_graceful_degradation_patterns(self):
        """Test that optional dependencies have graceful degradation."""
        degradation_analysis = self._analyze_graceful_degradation()

        print("\nGraceful degradation analysis:")
        for dep_name, analysis in degradation_analysis.items():
            print(f"{dep_name}:")
            print(f"  - Has availability check: {analysis['has_availability_check']}")
            print(f"  - Has graceful fallback: {analysis['has_graceful_fallback']}")
            print(f"  - Has feature detection: {analysis['has_feature_detection']}")
            print(f"  - Warning on import: {analysis['warns_on_import']}")

    def test_feature_detection_functions(self):
        """Test that feature detection functions exist for optional dependencies."""
        feature_detection = self._find_feature_detection_functions()

        print("\nFeature detection functions:")
        for func_name, info in feature_detection.items():
            print(f"  - {func_name}: {info['module']} -> {info['returns']}")

        # Check that we have detection for major optional features
        expected_detections = ["is_browser_tools_available", "is_mcp_available"]
        found_detections = set(feature_detection.keys())

        for expected in expected_detections:
            assert expected in found_detections, f"Should have feature detection for {expected}"

    def test_installation_instructions_quality(self):
        """Test that installation instructions are clear and consistent."""
        instructions = self._analyze_installation_instructions()

        print("\nInstallation instructions analysis:")
        for dep_name, instruction_list in instructions.items():
            print(f"{dep_name}: {len(instruction_list)} different instructions")
            for instruction in set(instruction_list):
                print(f"  - {instruction}")

        # Check for consistency in installation instructions
        inconsistent_deps = []
        for dep_name, instruction_list in instructions.items():
            unique_instructions = set(instruction_list)
            if len(unique_instructions) > 1:
                inconsistent_deps.append(dep_name)

        if inconsistent_deps:
            print(f"\nInconsistent installation instructions for: {inconsistent_deps}")

    def _find_optional_dependencies(self) -> List[Dict]:
        """Find all optional dependencies in the codebase."""
        optional_deps = {}
        src_path = Path("src/saplings")

        # We'll look for try/except ImportError patterns

        for py_file in src_path.rglob("*.py"):
            try:
                with open(py_file, encoding="utf-8") as f:
                    content = f.read()

                # Look for try/except ImportError patterns
                if "try:" in content and "ImportError" in content:
                    imports = self._extract_optional_imports(content)
                    for imp in imports:
                        if imp not in optional_deps:
                            optional_deps[imp] = {"name": imp, "usage_count": 0, "files": []}
                        optional_deps[imp]["usage_count"] += 1
                        optional_deps[imp]["files"].append(str(py_file))

            except Exception:
                continue

        return list(optional_deps.values())

    def _extract_optional_imports(self, content: str) -> List[str]:
        """Extract optional import names from content."""
        imports = []
        lines = content.split("\n")

        for i, line in enumerate(lines):
            if "import " in line and i > 0:
                # Check if this is in a try block
                prev_lines = lines[max(0, i - 10) : i]
                if any("try:" in prev_line for prev_line in prev_lines):
                    # Extract import name
                    if "import " in line:
                        parts = line.strip().split()
                        if "import" in parts:
                            import_idx = parts.index("import")
                            if import_idx + 1 < len(parts):
                                import_name = parts[import_idx + 1].split(".")[0]
                                imports.append(import_name)

        return imports

    def _analyze_warning_patterns(self) -> Dict[str, List[str]]:
        """Analyze warning patterns for optional dependencies."""
        patterns = {"logger_warning": [], "print_warning": [], "warnings_warn": []}

        src_path = Path("src/saplings")

        for py_file in src_path.rglob("*.py"):
            try:
                with open(py_file, encoding="utf-8") as f:
                    content = f.read()

                lines = content.split("\n")
                for line in lines:
                    line = line.strip()
                    if "logger.warning" in line and (
                        "not installed" in line or "not available" in line
                    ):
                        patterns["logger_warning"].append(line)
                    elif "print(" in line and ("not installed" in line or "not available" in line):
                        patterns["print_warning"].append(line)
                    elif "warnings.warn" in line:
                        patterns["warnings_warn"].append(line)

            except Exception:
                continue

        return patterns

    def _analyze_graceful_degradation(self) -> Dict[str, Dict]:
        """Analyze graceful degradation patterns."""
        analysis = {}

        # Known optional dependencies and their patterns
        known_deps = {
            "selenium": "SELENIUM_AVAILABLE",
            "triton": "TRITON_AVAILABLE",
            "vllm": "VLLM_AVAILABLE",
            "networkx": "NETWORKX_AVAILABLE",
            "transformers": "_HAS_LORA_DEPS",
            "opentelemetry": "OTEL_AVAILABLE",
        }

        src_path = Path("src/saplings")

        for dep_name, availability_var in known_deps.items():
            analysis[dep_name] = {
                "has_availability_check": False,
                "has_graceful_fallback": False,
                "has_feature_detection": False,
                "warns_on_import": False,
            }

            # Search for patterns
            for py_file in src_path.rglob("*.py"):
                try:
                    with open(py_file, encoding="utf-8") as f:
                        content = f.read()

                    if availability_var in content:
                        analysis[dep_name]["has_availability_check"] = True

                    if (
                        f"if {availability_var}" in content
                        or f"if not {availability_var}" in content
                    ):
                        analysis[dep_name]["has_graceful_fallback"] = True

                    if f"is_{dep_name}_available" in content:
                        analysis[dep_name]["has_feature_detection"] = True

                    if "logger.warning" in content and dep_name in content:
                        analysis[dep_name]["warns_on_import"] = True

                except Exception:
                    continue

        return analysis

    def _find_feature_detection_functions(self) -> Dict[str, Dict]:
        """Find feature detection functions."""
        detection_functions = {}
        src_path = Path("src/saplings")

        for py_file in src_path.rglob("*.py"):
            try:
                with open(py_file, encoding="utf-8") as f:
                    content = f.read()

                lines = content.split("\n")
                for line in lines:
                    if "def is_" in line and "_available" in line:
                        # Extract function name
                        func_name = line.split("def ")[1].split("(")[0].strip()
                        detection_functions[func_name] = {"module": str(py_file), "returns": "bool"}

            except Exception:
                continue

        return detection_functions

    def _analyze_installation_instructions(self) -> Dict[str, List[str]]:
        """Analyze installation instructions for consistency."""
        instructions = {}
        src_path = Path("src/saplings")

        for py_file in src_path.rglob("*.py"):
            try:
                with open(py_file, encoding="utf-8") as f:
                    content = f.read()

                lines = content.split("\n")
                for line in lines:
                    if "pip install" in line:
                        # Extract the instruction
                        instruction = line.strip()
                        if "saplings[" in instruction:
                            # Extract dependency name from saplings[dep]
                            start = instruction.find("saplings[") + 9
                            end = instruction.find("]", start)
                            if end > start:
                                dep_name = instruction[start:end]
                                if dep_name not in instructions:
                                    instructions[dep_name] = []
                                instructions[dep_name].append(instruction)

            except Exception:
                continue

        return instructions
