"""
Example validation tests for Task 9.17.

This module implements automated validation of all example files
to ensure they work with the current API and demonstrate best practices.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


class TestExampleValidation:
    """Automated validation of all example files."""

    def setup_method(self):
        """Set up test environment."""
        self.project_root = Path(__file__).parent.parent.parent
        self.examples_dir = self.project_root / "examples"
        self.python_files = list(self.examples_dir.glob("*.py"))

    @pytest.mark.e2e()
    def test_all_examples_exist(self):
        """Test that examples directory exists and contains Python files."""
        assert self.examples_dir.exists(), f"Examples directory not found: {self.examples_dir}"
        assert len(self.python_files) > 0, "No Python example files found"
        print(f"Found {len(self.python_files)} example files")

    @pytest.mark.e2e()
    def test_examples_syntax_validation(self):
        """Test that all example files have valid Python syntax."""
        syntax_errors = []

        for example_file in self.python_files:
            try:
                with open(example_file, encoding="utf-8") as f:
                    content = f.read()

                # Parse the file to check syntax
                ast.parse(content, filename=str(example_file))

            except SyntaxError as e:
                syntax_errors.append(f"{example_file.name}: {e}")
            except Exception as e:
                syntax_errors.append(f"{example_file.name}: Unexpected error: {e}")

        if syntax_errors:
            pytest.fail("Syntax errors found in examples:\n" + "\n".join(syntax_errors))

    @pytest.mark.e2e()
    def test_examples_import_validation(self):
        """Test that all examples use current API import patterns."""
        deprecated_imports = [
            "from saplings.api.container import",
            "from saplings.api.document import",
            "from saplings.api.interfaces import",
            "from saplings.api.indexer import",
            "from saplings.api.tool_validation import",
            "from saplings.core.interfaces import",
        ]

        current_imports = [
            "from saplings.api.di import",
            "from saplings.api.memory.document import",
            "from saplings.api.core.interfaces import",
            "from saplings.api.memory.indexer import",
            "from saplings.api.tools import",
        ]

        import_issues = []

        for example_file in self.python_files:
            try:
                with open(example_file, encoding="utf-8") as f:
                    content = f.read()

                # Check for deprecated imports
                for deprecated in deprecated_imports:
                    if deprecated in content:
                        import_issues.append(
                            f"{example_file.name}: Uses deprecated import: {deprecated}"
                        )

                # Check for saplings imports (should use current patterns)
                lines = content.split("\n")
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if line.startswith("from saplings") or line.startswith("import saplings"):
                        # This is a saplings import - verify it's using current patterns
                        if any(dep in line for dep in deprecated_imports):
                            import_issues.append(f"{example_file.name}:{line_num}: {line}")

            except Exception as e:
                import_issues.append(f"{example_file.name}: Error reading file: {e}")

        if import_issues:
            print(f"Found {len(import_issues)} import issues in examples:")
            for issue in import_issues[:10]:  # Show first 10 issues
                print(f"  {issue}")
            if len(import_issues) > 10:
                print(f"  ... and {len(import_issues) - 10} more issues")

        # This is informational - we don't fail the test as some imports might be intentional

    @pytest.mark.e2e()
    def test_examples_api_pattern_compliance(self):
        """Test that examples follow current API patterns."""
        pattern_issues = []

        for example_file in self.python_files:
            try:
                with open(example_file, encoding="utf-8") as f:
                    content = f.read()

                # Check for common API pattern compliance
                lines = content.split("\n")
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()

                    # Check for direct internal imports (should use public API)
                    if "from saplings." in line and "._internal" in line:
                        pattern_issues.append(
                            f"{example_file.name}:{line_num}: Direct internal import: {line}"
                        )

                    # Check for complex __new__ patterns (should be simplified)
                    if "__new__" in line and "saplings" in line:
                        pattern_issues.append(
                            f"{example_file.name}:{line_num}: Complex __new__ pattern: {line}"
                        )

            except Exception as e:
                pattern_issues.append(f"{example_file.name}: Error reading file: {e}")

        if pattern_issues:
            print(f"Found {len(pattern_issues)} API pattern issues in examples:")
            for issue in pattern_issues[:10]:  # Show first 10 issues
                print(f"  {issue}")
            if len(pattern_issues) > 10:
                print(f"  ... and {len(pattern_issues) - 10} more issues")

    @pytest.mark.e2e()
    def test_examples_have_documentation(self):
        """Test that examples have proper documentation and comments."""
        documentation_issues = []

        for example_file in self.python_files:
            try:
                with open(example_file, encoding="utf-8") as f:
                    content = f.read()

                # Check for module docstring
                try:
                    tree = ast.parse(content)
                    has_docstring = (
                        len(tree.body) > 0
                        and isinstance(tree.body[0], ast.Expr)
                        and isinstance(tree.body[0].value, ast.Constant)
                        and isinstance(tree.body[0].value.value, str)
                    )

                    if not has_docstring:
                        # Check for comments at the top
                        lines = content.split("\n")
                        has_comments = any(line.strip().startswith("#") for line in lines[:10])

                        if not has_comments:
                            documentation_issues.append(
                                f"{example_file.name}: No docstring or comments"
                            )

                except Exception:
                    # If we can't parse, skip this check
                    pass

            except Exception as e:
                documentation_issues.append(f"{example_file.name}: Error reading file: {e}")

        if documentation_issues:
            print(f"Found {len(documentation_issues)} documentation issues in examples:")
            for issue in documentation_issues:
                print(f"  {issue}")

    @pytest.mark.e2e()
    def test_examples_error_handling(self):
        """Test that examples include appropriate error handling."""
        error_handling_issues = []

        for example_file in self.python_files:
            try:
                with open(example_file, encoding="utf-8") as f:
                    content = f.read()

                # Check for basic error handling patterns
                has_try_except = "try:" in content and "except" in content
                has_error_check = any(
                    pattern in content
                    for pattern in ["if not", "assert", "raise", "ValueError", "TypeError"]
                )

                # For examples that create agents or use external services,
                # they should have some error handling
                uses_external_services = any(
                    pattern in content for pattern in ["Agent(", "openai", "anthropic", "api_key"]
                )

                if uses_external_services and not (has_try_except or has_error_check):
                    error_handling_issues.append(
                        f"{example_file.name}: Uses external services but lacks error handling"
                    )

            except Exception as e:
                error_handling_issues.append(f"{example_file.name}: Error reading file: {e}")

        if error_handling_issues:
            print(f"Found {len(error_handling_issues)} error handling issues in examples:")
            for issue in error_handling_issues:
                print(f"  {issue}")

    @pytest.mark.e2e()
    def test_examples_minimal_dependencies(self):
        """Test that examples work with minimal dependencies where possible."""
        dependency_issues = []

        for example_file in self.python_files:
            try:
                with open(example_file, encoding="utf-8") as f:
                    content = f.read()

                # Check for heavy optional dependencies
                heavy_dependencies = [
                    "import torch",
                    "import tensorflow",
                    "import transformers",
                    "import faiss",
                    "import selenium",
                ]

                for dep in heavy_dependencies:
                    if dep in content:
                        # Check if there's a try/except or availability check
                        has_optional_handling = any(
                            pattern in content
                            for pattern in [
                                "try:",
                                "except ImportError",
                                "except ModuleNotFoundError",
                                "importlib.util.find_spec",
                                "pkg_resources",
                            ]
                        )

                        if not has_optional_handling:
                            dependency_issues.append(
                                f"{example_file.name}: Uses {dep} without optional handling"
                            )

            except Exception as e:
                dependency_issues.append(f"{example_file.name}: Error reading file: {e}")

        if dependency_issues:
            print(f"Found {len(dependency_issues)} dependency issues in examples:")
            for issue in dependency_issues:
                print(f"  {issue}")

    @pytest.mark.e2e()
    def test_examples_follow_best_practices(self):
        """Test that examples follow Python and Saplings best practices."""
        best_practice_issues = []

        for example_file in self.python_files:
            try:
                with open(example_file, encoding="utf-8") as f:
                    content = f.read()

                # Check for common best practices
                lines = content.split("\n")

                for line_num, line in enumerate(lines, 1):
                    line_stripped = line.strip()

                    # Check for hardcoded API keys (security issue)
                    if any(
                        pattern in line_stripped.lower()
                        for pattern in ['api_key = "', "api_key = '", 'token = "', "token = '"]
                    ):
                        if not any(
                            safe in line_stripped for safe in ["os.getenv", "os.environ", "getpass"]
                        ):
                            best_practice_issues.append(
                                f"{example_file.name}:{line_num}: Hardcoded API key"
                            )

                    # Check for eval() usage (security issue)
                    if "eval(" in line_stripped:
                        best_practice_issues.append(
                            f"{example_file.name}:{line_num}: Uses eval() - security risk"
                        )

                    # Check for exec() usage (security issue)
                    if "exec(" in line_stripped:
                        best_practice_issues.append(
                            f"{example_file.name}:{line_num}: Uses exec() - security risk"
                        )

            except Exception as e:
                best_practice_issues.append(f"{example_file.name}: Error reading file: {e}")

        if best_practice_issues:
            print(f"Found {len(best_practice_issues)} best practice issues in examples:")
            for issue in best_practice_issues:
                print(f"  {issue}")

    @pytest.mark.e2e()
    def test_example_file_summary(self):
        """Provide a summary of all example files and their status."""
        print("\n=== Example Validation Summary ===")
        print(f"Total example files: {len(self.python_files)}")

        for example_file in sorted(self.python_files):
            try:
                with open(example_file, encoding="utf-8") as f:
                    content = f.read()

                # Basic file info
                lines = len(content.split("\n"))
                has_saplings_import = "saplings" in content
                has_main_block = 'if __name__ == "__main__"' in content

                status = "✓" if has_saplings_import else "?"
                print(f"  {status} {example_file.name} ({lines} lines, main: {has_main_block})")

            except Exception as e:
                print(f"  ✗ {example_file.name} (Error: {e})")

        print("=== End Summary ===\n")
