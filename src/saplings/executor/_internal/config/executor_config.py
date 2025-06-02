from __future__ import annotations

"""
Configuration module for the Executor.

This module defines the configuration classes for the Executor module.
"""


from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ValidationStrategy(str, Enum):
    """Strategy for validating generated outputs."""

    NONE = "none"  # No validation
    BASIC = "basic"  # Basic validation with simple checks
    JUDGE = "judge"  # Validation using JudgeAgent
    VALIDATOR = "validator"  # Validation using ValidatorRegistry
    FULL = "full"  # Full validation using both JudgeAgent and ValidatorRegistry
    SELF_CONSISTENCY = "self_consistency"  # Validation using self-consistency checking


# ValidationStrategy is the only strategy enum for validation


class RefinementStrategy(str, Enum):
    """Strategy for refining rejected outputs."""

    NONE = "none"  # No refinement
    RETRY = "retry"  # Simple retry with the same prompt
    FEEDBACK = "feedback"  # Retry with feedback from verification
    ITERATIVE = "iterative"  # Iterative refinement with multiple feedback cycles


class ExecutorConfig(BaseModel):
    """Configuration for the Executor module."""

    # Model settings
    execution_model_provider: str | None = Field(
        None, description="Provider for the execution model"
    )
    execution_model_name: str | None = Field(None, description="Name of the execution model")

    # Model settings
    max_tokens: int | None = Field(None, description="Maximum number of tokens to generate")
    temperature: float = Field(0.7, description="Temperature for generation")

    # Speculative execution settings
    enable_speculative_execution: bool = Field(
        True, description="Whether to enable speculative execution"
    )
    draft_temperature: float = Field(0.2, description="Temperature for draft generation")
    final_temperature: float = Field(0.7, description="Temperature for final generation")
    max_draft_tokens: int | None = Field(
        None, description="Maximum number of tokens for draft generation"
    )
    max_final_tokens: int | None = Field(
        None, description="Maximum number of tokens for final generation"
    )

    # Streaming settings
    enable_streaming: bool = Field(True, description="Whether to enable streaming output")
    stream_chunk_size: int = Field(
        10, description="Number of tokens to generate per streaming chunk"
    )

    # GASA settings
    enable_gasa: bool = Field(True, description="Whether to enable GASA")
    gasa_config: dict[str, Any] | None = Field(None, description="GASA configuration")

    # Validation settings
    validation_type: ValidationStrategy = Field(
        ValidationStrategy.JUDGE, description="Strategy for validating generated outputs"
    )
    validation_threshold: float = Field(0.7, description="Threshold for validation (0.0 to 1.0)")

    # Refinement settings
    refinement_strategy: RefinementStrategy = Field(
        RefinementStrategy.FEEDBACK, description="Strategy for refining rejected outputs"
    )
    max_refinement_attempts: int = Field(3, description="Maximum number of refinement attempts")

    # Performance settings
    cache_results: bool = Field(True, description="Whether to cache results")
    cache_dir: str | None = Field(None, description="Directory to cache results")

    # Logging settings
    log_level: str = Field("INFO", description="Logging level")

    @classmethod
    def default(cls):
        """
        Create a default configuration.

        Returns
        -------
            ExecutorConfig: Default configuration

        """
        return cls.model_construct()

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "ExecutorConfig":
        """
        Create a configuration from a dictionary.

        Args:
        ----
            config_dict: Configuration dictionary

        Returns:
        -------
            ExecutorConfig: Configuration

        """
        return cls(**config_dict)
