"""
CodeValidator plugin for Saplings.

This module provides a validator for code outputs, checking for syntax errors,
security issues, and code quality.
"""

import ast
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from saplings.core.plugin import PluginType
from saplings.validator.validator import RuntimeValidator, ValidationResult, ValidationStatus

logger = logging.getLogger(__name__)


class CodeValidator(RuntimeValidator):
    """
    Validator for code outputs.

    This validator checks code for syntax errors, security issues, and code quality.
    """

    @property
    def id(self) -> str:
        """ID of the validator."""
        return "code_validator"

    @property
    def name(self) -> str:
        """Name of the plugin."""
        return "code_validator"

    @property
    def version(self) -> str:
        """Version of the plugin."""
        return "0.1.0"

    @property
    def description(self) -> str:
        """Description of the plugin."""
        return "Validates code outputs for syntax errors, security issues, and code quality"

    @property
    def plugin_type(self) -> PluginType:
        """Type of the plugin."""
        return PluginType.VALIDATOR

    async def validate_output(self, output: str, prompt: str, **kwargs) -> ValidationResult:
        """
        Validate a code output.

        Args:
            output: Output to validate
            prompt: Prompt that generated the output
            **kwargs: Additional validation parameters

        Returns:
            ValidationResult: Validation result
        """
        # Extract code blocks from the output
        code_blocks = self._extract_code_blocks(output)

        if not code_blocks:
            return ValidationResult(
                validator_id=self.id,
                status=ValidationStatus.WARNING,
                message="No code blocks found in the output",
                metadata={"code_blocks_found": 0},
            )

        # Check each code block
        issues = []
        for i, (language, code) in enumerate(code_blocks):
            block_issues = self._check_code_block(code, language)
            if block_issues:
                issues.append(
                    {
                        "block_index": i,
                        "language": language,
                        "issues": block_issues,
                    }
                )

        if issues:
            return ValidationResult(
                validator_id=self.id,
                status=ValidationStatus.FAILED,
                message=f"Found {len(issues)} code blocks with issues",
                metadata={
                    "code_blocks_found": len(code_blocks),
                    "code_blocks_with_issues": len(issues),
                    "issues": issues,
                },
            )

        return ValidationResult(
            validator_id=self.id,
            status=ValidationStatus.PASSED,
            message=f"All {len(code_blocks)} code blocks passed validation",
            metadata={"code_blocks_found": len(code_blocks)},
        )

    def _extract_code_blocks(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract code blocks from text.

        Args:
            text: Text to extract code blocks from

        Returns:
            List[Tuple[str, str]]: List of (language, code) tuples
        """
        # Match Markdown code blocks
        pattern = r"```(\w*)\n(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)

        # Clean up matches
        result = []
        for language, code in matches:
            # Default to python if no language specified
            if not language:
                language = "python"

            # Strip trailing whitespace
            code = code.rstrip()

            result.append((language, code))

        return result

    def _check_code_block(self, code: str, language: str) -> List[Dict[str, Any]]:
        """
        Check a code block for issues.

        Args:
            code: Code to check
            language: Programming language

        Returns:
            List[Dict[str, Any]]: List of issues found
        """
        issues = []

        # Only check Python code for now
        if language.lower() != "python":
            return []

        # Check for syntax errors
        try:
            ast.parse(code)
        except SyntaxError as e:
            issues.append(
                {
                    "type": "syntax_error",
                    "message": str(e),
                    "line": e.lineno,
                    "offset": e.offset,
                }
            )
            # Don't continue checking if there are syntax errors
            return issues

        # Check for security issues
        security_issues = self._check_security_issues(code)
        issues.extend(security_issues)

        # Check for code quality issues
        quality_issues = self._check_code_quality(code)
        issues.extend(quality_issues)

        return issues

    def _check_security_issues(self, code: str) -> List[Dict[str, Any]]:
        """
        Check code for security issues.

        Args:
            code: Code to check

        Returns:
            List[Dict[str, Any]]: List of security issues found
        """
        issues = []

        # Check for dangerous imports
        dangerous_imports = [
            "os.system",
            "subprocess",
            "eval",
            "exec",
            "pickle.loads",
            "__import__",
            "importlib.import_module",
        ]

        for dangerous_import in dangerous_imports:
            if dangerous_import in code:
                issues.append(
                    {
                        "type": "security_issue",
                        "message": f"Potentially dangerous import or function: {dangerous_import}",
                        "severity": "high",
                    }
                )

        # Check for hardcoded credentials
        credential_patterns = [
            r"password\s*=\s*['\"].*?['\"]",
            r"api_key\s*=\s*['\"].*?['\"]",
            r"secret\s*=\s*['\"].*?['\"]",
            r"token\s*=\s*['\"].*?['\"]",
        ]

        for pattern in credential_patterns:
            matches = re.findall(pattern, code, re.IGNORECASE)
            for match in matches:
                issues.append(
                    {
                        "type": "security_issue",
                        "message": f"Hardcoded credential found: {match}",
                        "severity": "medium",
                    }
                )

        return issues

    def _check_code_quality(self, code: str) -> List[Dict[str, Any]]:
        """
        Check code for quality issues.

        Args:
            code: Code to check

        Returns:
            List[Dict[str, Any]]: List of quality issues found
        """
        issues = []

        # Check for long lines
        lines = code.split("\n")
        for i, line in enumerate(lines):
            if len(line) > 100:
                issues.append(
                    {
                        "type": "quality_issue",
                        "message": f"Line {i+1} is too long ({len(line)} > 100 characters)",
                        "line": i + 1,
                        "severity": "low",
                    }
                )

        # Check for TODO comments
        todo_pattern = r"#\s*TODO"
        for i, line in enumerate(lines):
            if re.search(todo_pattern, line):
                issues.append(
                    {
                        "type": "quality_issue",
                        "message": f"TODO comment found on line {i+1}",
                        "line": i + 1,
                        "severity": "low",
                    }
                )

        return issues
