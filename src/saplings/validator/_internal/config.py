from __future__ import annotations

"""
Configuration module for the Validator.

This module defines the configuration classes for the Validator module.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ValidatorType(str, Enum):
    """Types of validators."""

    STATIC = "static"  # Static validators run before execution
    RUNTIME = "runtime"  # Runtime validators run during or after execution
    HYBRID = "hybrid"  # Hybrid validators can run both before and after execution


class ValidatorConfig(BaseModel):
    """Configuration for the Validator module."""

    # General settings
    enabled: bool = Field(True, description="Whether validation is enabled")
    fail_fast: bool = Field(False, description="Whether to stop validation on first failure")

    # Plugin settings
    plugin_dirs: list[str] = Field(
        default_factory=list, description="Directories to search for validator plugins"
    )
    use_entry_points: bool = Field(
        True, description="Whether to use entry points for validator discovery"
    )

    # Execution settings
    parallel_validation: bool = Field(True, description="Whether to run validators in parallel")
    max_parallel_validators: int = Field(
        10, description="Maximum number of validators to run in parallel"
    )

    # Timeout settings
    timeout_seconds: float | None = Field(None, description="Timeout for validation in seconds")

    # Budget settings
    enforce_budget: bool = Field(False, description="Whether to enforce budget constraints")
    max_validations_per_session: int = Field(
        100, description="Maximum number of validations per session"
    )
    max_validations_per_day: int | None = Field(
        None, description="Maximum number of validations per day"
    )

    @classmethod
    def default(cls):
        """
        Create a default configuration.

        Returns
        -------
            ValidatorConfig: Default configuration

        """
        return cls(
            enabled=True,
            fail_fast=False,
            plugin_dirs=[],
            use_entry_points=True,
            parallel_validation=True,
            max_parallel_validators=10,
            timeout_seconds=None,
            enforce_budget=False,
            max_validations_per_session=100,
            max_validations_per_day=None,
        )

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "ValidatorConfig":
        """
        Create a configuration from a dictionary.

        Args:
        ----
            config_dict: Configuration dictionary

        Returns:
        -------
            ValidatorConfig: Configuration

        """
        return cls(**config_dict)
