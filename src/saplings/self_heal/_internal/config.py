from __future__ import annotations

"""
Internal implementation of the configuration module for self-healing capabilities.

This module defines the configuration classes for self-healing capabilities.
"""


from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RetryStrategy(str, Enum):
    """Strategy for retrying failed operations."""

    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    CONSTANT_BACKOFF = "constant_backoff"
    NO_BACKOFF = "no_backoff"


class AdapterPriority(str, Enum):
    """Priority levels for adapters."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SelfHealingConfig(BaseModel):
    """Configuration for self-healing capabilities."""

    # General settings
    enabled: bool = Field(
        default=True,
        description="Whether self-healing is enabled",
    )
    max_retries: int = Field(
        default=3,
        description="Maximum number of retry attempts",
        ge=0,
    )
    retry_strategy: RetryStrategy = Field(
        default=RetryStrategy.EXPONENTIAL_BACKOFF,
        description="Strategy for retrying failed operations",
    )
    retry_base_delay_ms: int = Field(
        default=1000,
        description="Base delay for retries in milliseconds",
        ge=0,
    )
    retry_max_delay_ms: int = Field(
        default=30000,
        description="Maximum delay for retries in milliseconds",
        ge=0,
    )

    # Success pair collection
    collect_success_pairs: bool = Field(
        default=True,
        description="Whether to collect success pairs for training",
    )
    success_pair_output_dir: str = Field(
        default="./success_pairs",
        description="Directory for storing success pairs",
    )
    max_success_pairs: int = Field(
        default=1000,
        description="Maximum number of success pairs to store",
        ge=0,
    )

    # LoRA fine-tuning
    enable_lora_training: bool = Field(
        default=False,
        description="Whether to enable LoRA fine-tuning",
    )
    lora_r: int = Field(
        default=8,
        description="LoRA r parameter",
        ge=1,
    )
    lora_alpha: int = Field(
        default=16,
        description="LoRA alpha parameter",
        ge=1,
    )
    lora_dropout: float = Field(
        default=0.05,
        description="LoRA dropout rate",
        ge=0.0,
        le=1.0,
    )
    lora_target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "v_proj"],
        description="LoRA target modules",
    )
    lora_learning_rate: float = Field(
        default=3e-4,
        description="LoRA learning rate",
        gt=0.0,
    )
    lora_batch_size: int = Field(
        default=4,
        description="LoRA batch size",
        gt=0,
    )
    lora_num_epochs: int = Field(
        default=3,
        description="LoRA number of epochs",
        gt=0,
    )
    lora_output_dir: str = Field(
        default="./lora_adapters",
        description="Directory for storing LoRA adapters",
    )

    # Adapter management
    adapter_dir: str = Field(
        default="./lora_adapters",
        description="Directory for storing adapters",
    )
    enable_adapter_management: bool = Field(
        default=True,
        description="Whether to enable adapter management",
    )
    default_adapter_priority: AdapterPriority = Field(
        default=AdapterPriority.MEDIUM,
        description="Default priority for adapters",
    )

    # Patch generation
    enable_patch_generation: bool = Field(
        default=True,
        description="Whether to enable patch generation",
    )
    patch_output_dir: str = Field(
        default="./patches",
        description="Directory for storing generated patches",
    )
    max_patch_size_kb: int = Field(
        default=100,
        description="Maximum patch size in KB",
        gt=0,
    )

    # Scheduling
    enable_scheduled_training: bool = Field(
        default=False,
        description="Whether to enable scheduled training",
    )
    training_schedule: str = Field(
        default="0 0 * * 0",  # Midnight on Sunday
        description="Cron expression for scheduled training",
    )

    # Metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )

    class Config:
        """Pydantic configuration."""

        extra = "ignore"
