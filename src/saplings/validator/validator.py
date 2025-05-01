"""
Validator module for Saplings.

This module provides the base Validator class and its implementations.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from saplings.core.plugin import Plugin, PluginType
from saplings.validator.config import ValidatorType

logger = logging.getLogger(__name__)


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
    details: Dict[str, Any] = Field(
        default_factory=dict, description="Additional details about the validation"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the validation result to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            "validator_id": self.validator_id,
            "status": self.status,
            "message": self.message,
            "details": self.details,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValidationResult":
        """
        Create a validation result from a dictionary.

        Args:
            data: Dictionary representation

        Returns:
            ValidationResult: Validation result
        """
        return cls(**data)


class Validator(Plugin, ABC):
    """
    Base class for validators.

    Validators are used to validate outputs against specific criteria.
    """

    @property
    @abstractmethod
    def id(self) -> str:
        """ID of the validator."""
        pass

    @property
    def plugin_type(self) -> PluginType:
        """Type of the plugin."""
        return PluginType.VALIDATOR

    @property
    @abstractmethod
    def validator_type(self) -> ValidatorType:
        """Type of the validator."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of the validator."""
        pass

    @abstractmethod
    async def validate(self, output: str, prompt: str, **kwargs) -> ValidationResult:
        """
        Validate an output.

        Args:
            output: Output to validate
            prompt: Prompt that generated the output
            **kwargs: Additional validation parameters

        Returns:
            ValidationResult: Validation result
        """
        pass


class StaticValidator(Validator):
    """
    Base class for static validators.

    Static validators run before execution and validate the prompt.
    """

    @property
    def validator_type(self) -> ValidatorType:
        """Type of the validator."""
        return ValidatorType.STATIC

    @abstractmethod
    async def validate_prompt(self, prompt: str, **kwargs) -> ValidationResult:
        """
        Validate a prompt.

        Args:
            prompt: Prompt to validate
            **kwargs: Additional validation parameters

        Returns:
            ValidationResult: Validation result
        """
        pass

    async def validate(self, output: str, prompt: str, **kwargs) -> ValidationResult:
        """
        Validate an output.

        For static validators, this delegates to validate_prompt.

        Args:
            output: Output to validate (ignored for static validators)
            prompt: Prompt to validate
            **kwargs: Additional validation parameters

        Returns:
            ValidationResult: Validation result
        """
        return await self.validate_prompt(prompt, **kwargs)


class RuntimeValidator(Validator):
    """
    Base class for runtime validators.

    Runtime validators run during or after execution and validate the output.
    """

    @property
    def validator_type(self) -> ValidatorType:
        """Type of the validator."""
        return ValidatorType.RUNTIME

    @abstractmethod
    async def validate_output(self, output: str, prompt: str, **kwargs) -> ValidationResult:
        """
        Validate an output.

        Args:
            output: Output to validate
            prompt: Prompt that generated the output
            **kwargs: Additional validation parameters

        Returns:
            ValidationResult: Validation result
        """
        pass

    async def validate(self, output: str, prompt: str, **kwargs) -> ValidationResult:
        """
        Validate an output.

        For runtime validators, this delegates to validate_output.

        Args:
            output: Output to validate
            prompt: Prompt that generated the output
            **kwargs: Additional validation parameters

        Returns:
            ValidationResult: Validation result
        """
        return await self.validate_output(output, prompt, **kwargs)
