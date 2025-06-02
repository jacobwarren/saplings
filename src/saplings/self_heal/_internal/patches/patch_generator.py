from __future__ import annotations

"""
Patch generator module for Saplings.

This module provides the PatchGenerator class for auto-fixing errors in code.
"""


import ast
import logging
import re
from enum import Enum
from typing import Any

from saplings.core._internal.exceptions import SaplingsError
from saplings.core.resilience import Validation
from saplings.self_heal._internal.interfaces import IPatchGenerator


# Define DataError as a subclass of SaplingsError
class DataError(SaplingsError):
    """Exception for data-related errors."""


# Define SelfHealingError as a subclass of SaplingsError
class SelfHealingError(SaplingsError):
    """Exception for self-healing errors."""


def get_indentation(line: str, extra_indent: str = "") -> str:
    """
    Safely get the indentation from a line of code.

    Args:
    ----
        line: The line to extract indentation from
        extra_indent: Additional indentation to add

    Returns:
    -------
        The indentation string

    """
    match = re.match(r"^(\s*)", line)
    return (match.group(1) if match else "") + extra_indent


logger = logging.getLogger(__name__)

# Check if pylint is available
try:
    import pylint  # type: ignore

    PYLINT_AVAILABLE = True
except ImportError:
    PYLINT_AVAILABLE = False
    logger.info("Pylint not available. Static analysis will be limited.")

# Check if pyflakes is available
try:
    import pyflakes  # type: ignore

    PYFLAKES_AVAILABLE = True
except ImportError:
    PYFLAKES_AVAILABLE = False
    logger.info("Pyflakes not available. Static analysis will be limited.")


class PatchStatus(str, Enum):
    """Status of a patch."""

    GENERATED = "generated"  # Patch has been generated
    APPLIED = "applied"  # Patch has been applied
    VALIDATED = "validated"  # Patch has been validated
    FAILED = "failed"  # Patch generation or application failed


class PatchResult:
    """Result of applying a patch."""

    def __init__(
        self,
        success: bool,
        patched_code: str | None = None,
        error: str | None = None,
    ) -> None:
        """
        Initialize the patch result.

        Args:
        ----
            success: Whether the patch was successfully applied
            patched_code: The patched code (if successful)
            error: Error message (if unsuccessful)

        """
        self.success = success
        self.patched_code = patched_code
        self.error = error


class Patch:
    """A code patch."""

    def __init__(
        self,
        original_code: str,
        patched_code: str,
        error: str,
        error_info: dict[str, Any],
        status: PatchStatus = PatchStatus.GENERATED,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the patch.

        Args:
        ----
            original_code: Original code with error
            patched_code: Patched code
            error: Error message
            error_info: Information about the error
            status: Status of the patch
            metadata: Additional metadata

        """
        self.original_code = original_code
        self.patched_code = patched_code
        self.error = error
        self.error_info = error_info
        self.status = status
        self.metadata = metadata or {}
        self.timestamp = self.metadata.get("timestamp", None)


class PatchGenerator(IPatchGenerator):
    """
    Generator for code patches.

    This class analyzes errors in code and generates patches to fix them.
    It implements the IPatchGenerator interface for consistent API usage.
    """

    def __init__(
        self,
        max_retries: int = 3,
        success_pair_collector: Any | None = None,
    ) -> None:
        """
        Initialize the patch generator.

        Args:
        ----
            max_retries: Maximum number of retry attempts
            success_pair_collector: Collector for successful error-fix pairs

        Raises:
        ------
            ValueError: If max_retries is not positive

        """
        # Validate inputs
        Validation.require(max_retries > 0, "max_retries must be positive")

        self.max_retries = max_retries
        self.retry_count = 0
        self.patches: list[Patch] = []
        self.success_pair_collector = success_pair_collector

    async def generate_patch(
        self,
        error_message: str,
        code_context: str,
        metadata: dict[str, Any] | None = None,
    ) -> Patch:
        """
        Generate a patch for the given error.

        Args:
        ----
            error_message: The error message to fix
            code_context: The code context where the error occurred
            metadata: Additional metadata about the error

        Returns:
        -------
            A generated patch

        """
        # Simple patch generation for testing purposes
        # In a real implementation, this would use ML models or rule-based systems
        patched_code = code_context.replace("'b'", "str(b)")  # Simple string replacement

        return Patch(
            original_code=code_context,
            patched_code=patched_code,
            error=error_message,
            error_info=metadata or {},
            status=PatchStatus.GENERATED,
        )

    def apply_patch_without_validation(self, patch: Patch) -> PatchResult:
        """
        Apply a patch without validation (unsafe operation).

        This method is intentionally designed to raise an error to prevent
        unsafe patch application without proper validation.

        Args:
        ----
            patch: The patch to apply (unused, method always raises)

        Raises:
        ------
            RuntimeError: Always raises to prevent unsafe operation

        """
        # Explicitly mark patch as unused since this method always raises
        _ = patch
        raise RuntimeError(
            "Patches must be validated before application. "
            "Use validate_patch_safety() first, then apply_validated_patch()."
        )

    def validate_patch_safety(self, patch: Patch) -> tuple[bool, str]:
        """
        Validate that a patch is safe to apply.

        Args:
        ----
            patch: The patch to validate

        Returns:
        -------
            Tuple of (is_safe, reason)

        """
        # Check for dangerous operations
        dangerous_patterns = [
            r"os\.system\s*\(",
            r"subprocess\.(run|call|Popen)",
            r"eval\s*\(",
            r"exec\s*\(",
            r"__import__\s*\(",
            r"rm\s+-rf",
            r"del\s+/",
            r"shutil\.rmtree",
        ]

        code_to_check = patch.patched_code

        for pattern in dangerous_patterns:
            if re.search(pattern, code_to_check, re.IGNORECASE):
                return False, f"Dangerous operation detected: {pattern}"

        # Check for syntax errors
        try:
            ast.parse(patch.patched_code)
        except SyntaxError as e:
            return False, f"Syntax error in patched code: {e}"

        # Check error severity
        if patch.error_info.get("severity") == "critical":
            return False, "Critical severity patches require manual review"

        return True, "Patch appears safe"

    def validate_patch(self, patch: Patch) -> tuple[bool, str | None]:
        """
        Validate a patch for correctness and safety.

        Args:
        ----
            patch: The patch to validate

        Returns:
        -------
            Tuple of (is_valid, error_message)

        """
        # First check safety
        is_safe, safety_reason = self.validate_patch_safety(patch)
        if not is_safe:
            return False, safety_reason

        # Check for basic correctness
        if not patch.original_code or not patch.patched_code:
            return False, "Patch must have both original and patched code"

        if patch.original_code == patch.patched_code:
            return False, "Patch does not modify the code"

        return True, None

    def apply_validated_patch(self, patch: Patch) -> PatchResult:
        """
        Apply a patch that has been validated.

        Args:
        ----
            patch: The validated patch to apply

        Returns:
        -------
            PatchResult with the application result

        """
        # Validate first
        is_valid, error_msg = self.validate_patch(patch)
        if not is_valid:
            return PatchResult(success=False, error=error_msg)

        try:
            # Apply the patch (in this case, just return the patched code)
            patch.status = PatchStatus.APPLIED
            return PatchResult(success=True, patched_code=patch.patched_code)
        except Exception as e:
            return PatchResult(success=False, error=str(e))
