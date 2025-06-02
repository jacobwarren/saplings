from __future__ import annotations

"""
Execution validator for Saplings.

This module provides a validator for execution outputs.
"""

import logging

from saplings.core._internal.model_adapter import LLM
from saplings.validator._internal.result import ValidationResult, ValidationStatus
from saplings.validator._internal.validator import RuntimeValidator

logger = logging.getLogger(__name__)


class ExecutionValidator(RuntimeValidator):
    """
    Validator for execution outputs.

    This validator checks if the execution output is valid.
    """

    def __init__(self) -> None:
        """Initialize the execution validator."""
        super().__init__()

    @property
    def id(self) -> str:
        """ID of the validator."""
        return "execution"

    @property
    def name(self) -> str:
        """Name of the validator."""
        return "Execution Validator"

    @property
    def description(self) -> str:
        """Description of the validator."""
        return "Validates execution outputs for correctness and completeness."

    @property
    def version(self) -> str:
        """Version of the validator."""
        return "1.0.0"

    async def validate_output(
        self, output: str, _prompt: str, _model: LLM | None = None, **_kwargs
    ) -> ValidationResult:
        """
        Validate an execution output.

        Args:
        ----
            output: Output to validate
            _prompt: Prompt that generated the output (unused)
            _model: LLM model to use for validation (unused)
            **_kwargs: Additional validation parameters (unused)

        Returns:
        -------
            ValidationResult: Validation result

        """
        # Basic validation - check if output is not empty
        if not output or not output.strip():
            return ValidationResult(
                validator_id=self.id,
                status=ValidationStatus.FAILED,
                message="Output is empty",
                metadata={"output_length": 0},
            )

        # Check if output is too short (less than 10 characters)
        if len(output.strip()) < 10:
            return ValidationResult(
                validator_id=self.id,
                status=ValidationStatus.WARNING,
                message="Output is very short",
                metadata={"output_length": len(output.strip())},
            )

        # If we have a model, we could use it for more sophisticated validation
        # But for now, we'll just do basic checks

        # Check if output seems to be a valid response to the prompt
        # This is a very basic check - in a real implementation, you would use
        # more sophisticated techniques

        # For now, just return a successful validation
        return ValidationResult(
            validator_id=self.id,
            status=ValidationStatus.PASSED,
            message="Output passed basic validation",
            metadata={"output_length": len(output.strip())},
        )
