from __future__ import annotations

"""
Validator module for Saplings.

This module provides the base Validator class and its implementations.
"""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from saplings.api.registry import Plugin, PluginType
from saplings.validator._internal.config import ValidatorType

# Forward reference for ValidationResult to avoid circular imports
if TYPE_CHECKING:
    from saplings.validator._internal.result import ValidationResult

logger = logging.getLogger(__name__)


class Validator(Plugin, ABC):
    """
    Base class for validators.

    Validators are used to validate outputs against specific criteria.
    """

    @property
    @abstractmethod
    def id(self) -> str:
        """ID of the validator."""

    @property
    def plugin_type(self) -> PluginType:
        """Type of the plugin."""
        return PluginType.VALIDATOR

    @property
    @abstractmethod
    def validator_type(self) -> ValidatorType:
        """Type of the validator."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of the validator."""

    @abstractmethod
    async def validate(self, output: str, prompt: str, **kwargs) -> ValidationResult:
        """
        Validate an output.

        Args:
        ----
            output: Output to validate
            prompt: Prompt that generated the output
            **kwargs: Additional validation parameters

        Returns:
        -------
            ValidationResult: Validation result

        """


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
        ----
            prompt: Prompt to validate
            **kwargs: Additional validation parameters

        Returns:
        -------
            ValidationResult: Validation result

        """

    async def validate(self, _output: str, prompt: str, **kwargs) -> ValidationResult:
        """
        Validate an output.

        For static validators, this delegates to validate_prompt.

        Args:
        ----
            _output: Output to validate (ignored for static validators)
            prompt: Prompt to validate
            **kwargs: Additional validation parameters

        Returns:
        -------
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
        ----
            output: Output to validate
            prompt: Prompt that generated the output
            **kwargs: Additional validation parameters

        Returns:
        -------
            ValidationResult: Validation result

        """

    async def validate(self, output: str, prompt: str, **kwargs) -> ValidationResult:
        """
        Validate an output.

        For runtime validators, this delegates to validate_output.

        Args:
        ----
            output: Output to validate
            prompt: Prompt that generated the output
            **kwargs: Additional validation parameters

        Returns:
        -------
            ValidationResult: Validation result

        """
        return await self.validate_output(output, prompt, **kwargs)
