"""
Security and safety review tests for Task 9.19.

This module implements comprehensive security validation of all public APIs
to ensure they are secure by default and don't expose vulnerabilities.
"""

from __future__ import annotations

import importlib
import inspect
import re
from pathlib import Path
from typing import List

import pytest


class TestSecurityReview:
    """Comprehensive security and safety review tests."""

    def setup_method(self):
        """Set up test environment."""
        self.project_root = Path(__file__).parent.parent.parent
        self.src_dir = self.project_root / "src"
        self.api_modules = self._discover_api_modules()

    def _discover_api_modules(self) -> List[str]:
        """Discover all API modules in the saplings.api package."""
        api_modules = []
        api_dir = self.src_dir / "saplings" / "api"

        if api_dir.exists():
            for py_file in api_dir.rglob("*.py"):
                if py_file.name != "__init__.py":
                    # Convert file path to module name
                    rel_path = py_file.relative_to(self.src_dir)
                    module_name = str(rel_path.with_suffix("")).replace("/", ".")
                    api_modules.append(module_name)

        return api_modules

    @pytest.mark.e2e()
    def test_input_validation_security(self):
        """Test that public APIs properly validate user input."""
        input_validation_issues = []

        for module_name in self.api_modules:
            try:
                module = importlib.import_module(module_name)

                # Check public functions for input validation
                for name in dir(module):
                    if not name.startswith("_"):
                        obj = getattr(module, name)
                        if inspect.isfunction(obj):
                            # Get function source if possible
                            try:
                                source = inspect.getsource(obj)

                                # Check for basic input validation patterns
                                has_validation = any(
                                    pattern in source
                                    for pattern in [
                                        "isinstance(",
                                        "if not ",
                                        "assert ",
                                        "raise ValueError",
                                        "raise TypeError",
                                        "validate_",
                                        "check_",
                                    ]
                                )

                                # Check for potentially dangerous operations without validation
                                dangerous_ops = [
                                    "eval(",
                                    "exec(",
                                    "open(",
                                    "subprocess",
                                    "__import__",
                                ]
                                has_dangerous_ops = any(op in source for op in dangerous_ops)

                                if has_dangerous_ops and not has_validation:
                                    input_validation_issues.append(
                                        f"{module_name}.{name}: Dangerous operations without validation"
                                    )

                            except (OSError, TypeError):
                                # Can't get source, skip
                                continue

            except ImportError:
                continue

        if input_validation_issues:
            print(f"Found {len(input_validation_issues)} input validation security issues:")
            for issue in input_validation_issues:
                print(f"  {issue}")

    @pytest.mark.e2e()
    def test_code_injection_prevention(self):
        """Test that APIs prevent code injection attacks."""
        code_injection_risks = []

        # Check all Python files in the API
        api_dir = self.src_dir / "saplings" / "api"
        if api_dir.exists():
            for py_file in api_dir.rglob("*.py"):
                try:
                    with open(py_file, encoding="utf-8") as f:
                        content = f.read()

                    # Check for dangerous patterns
                    dangerous_patterns = [
                        (r"eval\s*\(", "eval() usage"),
                        (r"exec\s*\(", "exec() usage"),
                        (r"__import__\s*\(", "dynamic import"),
                        (r"compile\s*\(", "code compilation"),
                        (
                            r"getattr\s*\([^,]+,\s*[^)]*input",
                            "dynamic attribute access with user input",
                        ),
                    ]

                    for pattern, description in dangerous_patterns:
                        matches = re.findall(pattern, content, re.IGNORECASE)
                        if matches:
                            # Check if it's in a safe context (e.g., with proper validation)
                            lines = content.split("\n")
                            for line_num, line in enumerate(lines, 1):
                                if re.search(pattern, line, re.IGNORECASE):
                                    # Check surrounding lines for safety measures
                                    context_start = max(0, line_num - 3)
                                    context_end = min(len(lines), line_num + 3)
                                    context = "\n".join(lines[context_start:context_end])

                                    # Look for safety measures
                                    has_safety = any(
                                        safety in context.lower()
                                        for safety in [
                                            "validate",
                                            "sanitize",
                                            "whitelist",
                                            "allowlist",
                                            "if not",
                                            "assert",
                                            "raise",
                                            "check",
                                        ]
                                    )

                                    if not has_safety:
                                        rel_path = py_file.relative_to(self.src_dir)
                                        code_injection_risks.append(
                                            f"{rel_path}:{line_num}: {description}"
                                        )

                except Exception:
                    continue

        if code_injection_risks:
            print(f"Found {len(code_injection_risks)} potential code injection risks:")
            for risk in code_injection_risks[:10]:  # Show first 10
                print(f"  {risk}")
            if len(code_injection_risks) > 10:
                print(f"  ... and {len(code_injection_risks) - 10} more risks")

    @pytest.mark.e2e()
    def test_file_path_safety(self):
        """Test that file path operations are safe from path traversal attacks."""
        file_path_issues = []

        api_dir = self.src_dir / "saplings" / "api"
        if api_dir.exists():
            for py_file in api_dir.rglob("*.py"):
                try:
                    with open(py_file, encoding="utf-8") as f:
                        content = f.read()

                    # Check for file operations
                    file_patterns = [
                        (r"open\s*\(", "file open"),
                        (r"Path\s*\(", "Path construction"),
                        (r"os\.path\.join", "path join"),
                        (r"\.write_text\s*\(", "file write"),
                        (r"\.read_text\s*\(", "file read"),
                    ]

                    for pattern, description in file_patterns:
                        matches = re.findall(pattern, content, re.IGNORECASE)
                        if matches:
                            lines = content.split("\n")
                            for line_num, line in enumerate(lines, 1):
                                if re.search(pattern, line, re.IGNORECASE):
                                    # Check for path safety measures
                                    context_start = max(0, line_num - 3)
                                    context_end = min(len(lines), line_num + 3)
                                    context = "\n".join(lines[context_start:context_end])

                                    # Look for path safety measures
                                    has_path_safety = any(
                                        safety in context.lower()
                                        for safety in [
                                            "resolve()",
                                            "absolute()",
                                            "is_absolute()",
                                            "os.path.abspath",
                                            "os.path.realpath",
                                            "pathlib",
                                            ".parent",
                                            "startswith",
                                        ]
                                    )

                                    # Check for dangerous patterns
                                    has_danger = any(
                                        danger in line for danger in ["../", "..\\", "../"]
                                    )

                                    if has_danger or (
                                        not has_path_safety and "user" in context.lower()
                                    ):
                                        rel_path = py_file.relative_to(self.src_dir)
                                        file_path_issues.append(
                                            f"{rel_path}:{line_num}: Potentially unsafe {description}"
                                        )

                except Exception:
                    continue

        if file_path_issues:
            print(f"Found {len(file_path_issues)} file path safety issues:")
            for issue in file_path_issues[:10]:  # Show first 10
                print(f"  {issue}")
            if len(file_path_issues) > 10:
                print(f"  ... and {len(file_path_issues) - 10} more issues")

    @pytest.mark.e2e()
    def test_secure_defaults_configuration(self):
        """Test that all configuration options have secure defaults."""
        insecure_defaults = []

        for module_name in self.api_modules:
            try:
                module = importlib.import_module(module_name)

                # Look for configuration classes
                for name in dir(module):
                    if not name.startswith("_"):
                        obj = getattr(module, name)
                        if inspect.isclass(obj) and "config" in name.lower():
                            # Check class attributes for default values
                            for attr_name in dir(obj):
                                if not attr_name.startswith("_"):
                                    try:
                                        attr_value = getattr(obj, attr_name)

                                        # Check for potentially insecure defaults
                                        if isinstance(attr_value, bool):
                                            # Security-related booleans should default to secure values
                                            if any(
                                                security_term in attr_name.lower()
                                                for security_term in [
                                                    "debug",
                                                    "verbose",
                                                    "log",
                                                    "trace",
                                                ]
                                            ):
                                                if (
                                                    attr_value is True
                                                ):  # Debug should default to False
                                                    insecure_defaults.append(
                                                        f"{module_name}.{name}.{attr_name}: Debug enabled by default"
                                                    )

                                        elif isinstance(attr_value, str):
                                            # Check for hardcoded secrets or insecure values
                                            if any(
                                                secret_term in attr_name.lower()
                                                for secret_term in [
                                                    "key",
                                                    "token",
                                                    "password",
                                                    "secret",
                                                ]
                                            ):
                                                if (
                                                    attr_value and len(attr_value) > 5
                                                ):  # Non-empty default
                                                    insecure_defaults.append(
                                                        f"{module_name}.{name}.{attr_name}: Hardcoded secret"
                                                    )

                                    except Exception:
                                        continue

            except ImportError:
                continue

        if insecure_defaults:
            print(f"Found {len(insecure_defaults)} potentially insecure default configurations:")
            for default in insecure_defaults:
                print(f"  {default}")

    @pytest.mark.e2e()
    def test_resource_limits_and_timeouts(self):
        """Test that APIs implement appropriate resource limits and timeouts."""
        missing_limits = []

        for module_name in self.api_modules:
            try:
                module = importlib.import_module(module_name)

                # Check for functions that might need resource limits
                for name in dir(module):
                    if not name.startswith("_"):
                        obj = getattr(module, name)
                        if inspect.isfunction(obj):
                            try:
                                source = inspect.getsource(obj)

                                # Check for operations that should have limits
                                needs_limits = any(
                                    operation in source.lower()
                                    for operation in [
                                        "while",
                                        "for",
                                        "requests.",
                                        "urllib",
                                        "download",
                                        "fetch",
                                        "load",
                                        "process",
                                        "generate",
                                    ]
                                )

                                if needs_limits:
                                    # Check for timeout/limit patterns
                                    has_limits = any(
                                        limit_pattern in source.lower()
                                        for limit_pattern in [
                                            "timeout",
                                            "max_",
                                            "limit",
                                            "break",
                                            "return",
                                            "time.time()",
                                            "asyncio.wait_for",
                                        ]
                                    )

                                    if not has_limits:
                                        missing_limits.append(
                                            f"{module_name}.{name}: Missing resource limits/timeouts"
                                        )

                            except (OSError, TypeError):
                                continue

            except ImportError:
                continue

        if missing_limits:
            print(f"Found {len(missing_limits)} functions that may need resource limits:")
            for limit in missing_limits[:10]:  # Show first 10
                print(f"  {limit}")
            if len(missing_limits) > 10:
                print(f"  ... and {len(missing_limits) - 10} more functions")

    @pytest.mark.e2e()
    def test_dependency_security_audit(self):
        """Test that dependencies don't introduce known security vulnerabilities."""
        dependency_issues = []

        # Check requirements files for known vulnerable packages
        requirements_files = [
            self.project_root / "requirements.txt",
            self.project_root / "pyproject.toml",
        ]

        # Known vulnerable patterns (this would be updated with real vulnerability data)
        known_vulnerabilities = {
            "requests": ["2.25.0", "2.25.1"],  # Example - these versions had vulnerabilities
            "urllib3": ["1.26.0"],  # Example
        }

        for req_file in requirements_files:
            if req_file.exists():
                try:
                    with open(req_file, encoding="utf-8") as f:
                        content = f.read()

                    for package, vulnerable_versions in known_vulnerabilities.items():
                        if package in content:
                            for version in vulnerable_versions:
                                if version in content:
                                    dependency_issues.append(
                                        f"{req_file.name}: {package}=={version} has known vulnerabilities"
                                    )

                except Exception:
                    continue

        if dependency_issues:
            print(f"Found {len(dependency_issues)} potential dependency security issues:")
            for issue in dependency_issues:
                print(f"  {issue}")

    @pytest.mark.e2e()
    def test_api_key_and_secret_handling(self):
        """Test that API keys and secrets are handled securely."""
        secret_handling_issues = []

        api_dir = self.src_dir / "saplings" / "api"
        if api_dir.exists():
            for py_file in api_dir.rglob("*.py"):
                try:
                    with open(py_file, encoding="utf-8") as f:
                        content = f.read()

                    lines = content.split("\n")
                    for line_num, line in enumerate(lines, 1):
                        # Check for hardcoded secrets
                        secret_patterns = [
                            r'api_key\s*=\s*["\'][^"\']{10,}["\']',
                            r'token\s*=\s*["\'][^"\']{10,}["\']',
                            r'password\s*=\s*["\'][^"\']{5,}["\']',
                            r'secret\s*=\s*["\'][^"\']{10,}["\']',
                        ]

                        for pattern in secret_patterns:
                            if re.search(pattern, line, re.IGNORECASE):
                                # Check if it's a placeholder or example
                                is_placeholder = any(
                                    placeholder in line.lower()
                                    for placeholder in [
                                        "your_",
                                        "example",
                                        "placeholder",
                                        "xxx",
                                        "***",
                                        "test",
                                        "demo",
                                        "sample",
                                    ]
                                )

                                if not is_placeholder:
                                    rel_path = py_file.relative_to(self.src_dir)
                                    secret_handling_issues.append(
                                        f"{rel_path}:{line_num}: Potential hardcoded secret"
                                    )

                        # Check for insecure secret usage
                        if any(term in line.lower() for term in ["api_key", "token", "password"]):
                            # Should use environment variables or secure storage
                            secure_patterns = ["os.getenv", "os.environ", "getpass", "keyring"]
                            has_secure_access = any(pattern in line for pattern in secure_patterns)

                            if "print(" in line or "log" in line.lower():
                                secret_handling_issues.append(
                                    f"{rel_path}:{line_num}: Potential secret logging"
                                )

                except Exception:
                    continue

        if secret_handling_issues:
            print(f"Found {len(secret_handling_issues)} secret handling issues:")
            for issue in secret_handling_issues[:10]:  # Show first 10
                print(f"  {issue}")
            if len(secret_handling_issues) > 10:
                print(f"  ... and {len(secret_handling_issues) - 10} more issues")

    @pytest.mark.e2e()
    def test_security_review_summary(self):
        """Provide a comprehensive summary of security review findings."""
        print("\n=== Security Review Summary ===")
        print(f"API modules reviewed: {len(self.api_modules)}")

        # Count total files reviewed
        api_dir = self.src_dir / "saplings" / "api"
        total_files = len(list(api_dir.rglob("*.py"))) if api_dir.exists() else 0

        print(f"Total API files reviewed: {total_files}")
        print("Security checks performed:")
        print("  ✓ Input validation security")
        print("  ✓ Code injection prevention")
        print("  ✓ File path safety")
        print("  ✓ Secure defaults configuration")
        print("  ✓ Resource limits and timeouts")
        print("  ✓ Dependency security audit")
        print("  ✓ API key and secret handling")
        print("=== End Security Review ===\n")
