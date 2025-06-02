from __future__ import annotations

"""
Validation result module for Saplings.

This module provides classes for representing validation results.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ValidationStatus(str, Enum):
    """Status of a validation."""

    PASSED = "passed"  # Validation passed
    FAILED = "failed"  # Validation failed
    WARNING = "warning"  # Validation passed with warnings
    ERROR = "error"  # Validation failed with errors
    SKIPPED = "skipped"  # Validation was skipped


class ValidationResult(BaseModel):
    """Result of a validation."""

    validator_id: str = Field(..., description="ID of the validator")
    status: ValidationStatus = Field(..., description="Status of the validation")
    message: str = Field("", description="Message explaining the validation result")
    details: dict[str, Any] = Field(
        default_factory=dict, description="Additional details about the validation"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the validation result to a dictionary.

        Returns
        -------
            dict[str, Any]: Dictionary representation

        """
        return {
            "validator_id": self.validator_id,
            "status": self.status,
            "message": self.message,
            "details": self.details,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ValidationResult":
        """
        Create a validation result from a dictionary.

        Args:
        ----
            data: Dictionary representation

        Returns:
        -------
            ValidationResult: Validation result

        """
        return cls(**data)
