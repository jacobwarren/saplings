from __future__ import annotations

"""
Configuration module for Graph-Aligned Sparse Attention (GASA).

This module defines the configuration classes for the GASA module.
"""


import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

# Constants
PATH_PATTERN = r"^[a-zA-Z0-9_\-./]*$"


class MaskStrategy(str, Enum):
    """Strategy for applying attention masks."""

    BINARY = "binary"  # Binary mask (0/1)
    SOFT = "soft"  # Soft mask (continuous values between 0 and 1)
    LEARNED = "learned"  # Learned mask (requires fine-tuning)


class FallbackStrategy(str, Enum):
    """Fallback strategy for models that don't support sparse attention."""

    BLOCK_DIAGONAL = "block_diagonal"  # Reorder tokens into block-diagonal structure
    DENSE = "dense"  # Fall back to dense attention
    WINDOWED = "windowed"  # Use sliding window attention
    PROMPT_COMPOSER = "prompt_composer"  # Use the graph-aware prompt composer
    SHADOW_MODEL = "shadow_model"  # Use a shadow model for tokenization and rewriting


class GASAConfig(BaseModel):
    """Configuration for Graph-Aligned Sparse Attention (GASA)."""

    enabled: bool = Field(True, description="Whether to enable GASA")
    max_hops: int = Field(2, description="Maximum number of hops for attention (h parameter)")
    mask_strategy: MaskStrategy = Field(
        MaskStrategy.BINARY, description="Strategy for applying attention masks"
    )
    fallback_strategy: FallbackStrategy = Field(
        FallbackStrategy.BLOCK_DIAGONAL,
        description="Fallback strategy for models that don't support sparse attention",
    )
    global_tokens: list[str] = Field(
        ["[CLS]", "[SEP]", "<s>", "</s>", "[SUM]"],
        description="Tokens that should attend to all other tokens",
    )
    summary_token: str = Field("[SUM]", description="Token used for global summary")
    add_summary_token: bool = Field(
        True, description="Whether to add a summary token if not present"
    )
    block_size: int = Field(
        512, description="Block size for block-diagonal packing", ge=64, le=4096
    )
    overlap: int = Field(64, description="Overlap between blocks for block-diagonal packing", ge=0)
    soft_mask_temperature: float = Field(
        0.1, description="Temperature for soft masks (lower = closer to binary)"
    )
    cache_masks: bool = Field(True, description="Whether to cache generated masks")
    cache_dir: str | None = Field(
        None, description="Directory to cache masks", pattern=PATH_PATTERN
    )
    visualize: bool = Field(False, description="Whether to generate visualizations")
    visualization_dir: str | None = Field(
        None, description="Directory to save visualizations", pattern=PATH_PATTERN
    )

    # Shadow model configuration
    enable_shadow_model: bool = Field(
        False, description="Whether to enable shadow model for tokenization"
    )
    shadow_model_name: str = Field("Qwen/Qwen3-1.8B", description="Name of the shadow model to use")
    shadow_model_device: str = Field("cpu", description="Device to use for the shadow model")
    shadow_model_cache_dir: str | None = Field(
        None, description="Directory to cache the shadow model"
    )

    # Prompt composer configuration
    enable_prompt_composer: bool = Field(
        False, description="Whether to enable the graph-aware prompt composer"
    )
    focus_tags: bool = Field(True, description="Whether to add focus tags to important context")
    core_tag: str = Field("[CORE_CTX]", description="Tag for core context")
    near_tag: str = Field("[NEAR_CTX]", description="Tag for near context")
    summary_tag: str = Field("[SUMMARY_CTX]", description="Tag for summary context")

    @field_validator("max_hops")
    def validate_max_hops(cls, v: int) -> int:
        """Validate max_hops."""
        if not isinstance(v, int):
            msg = f"max_hops must be an integer, got {type(v)}"
            raise ValueError(msg)
        return v

    @field_validator("block_size")
    def validate_block_size(cls, v: int) -> int:
        """Validate block_size."""
        if not isinstance(v, int):
            msg = f"block_size must be an integer, got {type(v)}"
            raise ValueError(msg)
        if not 64 <= v <= 4096:
            msg = f"block_size must be between 64 and 4096, got {v}"
            raise ValueError(msg)
        return v

    @field_validator("overlap")
    def validate_overlap(cls, v: int) -> int:
        """Validate overlap."""
        if not isinstance(v, int):
            msg = f"overlap must be an integer, got {type(v)}"
            raise ValueError(msg)
        if v < 0:
            msg = f"overlap must be non-negative, got {v}"
            raise ValueError(msg)
        return v

    @field_validator("cache_dir", "visualization_dir", "shadow_model_cache_dir")
    def validate_path(cls, v: str | None) -> str | None:
        """Validate path arguments."""
        if v is not None:
            if not isinstance(v, str):
                msg = f"Path must be a string, got {type(v)}"
                raise ValueError(msg)
            if not re.match(PATH_PATTERN, v):
                msg = f"Path contains invalid characters: {v}"
                raise ValueError(msg)
        return v

    @classmethod
    def default(cls) -> "GASAConfig":
        """
        Create a default configuration.

        Returns
        -------
            GASAConfig: Default configuration

        """
        return cls(
            enabled=True,
            max_hops=2,
            mask_strategy=MaskStrategy.BINARY,
            fallback_strategy=FallbackStrategy.BLOCK_DIAGONAL,
            global_tokens=["[CLS]", "[SEP]", "<s>", "</s>", "[SUM]"],
            summary_token="[SUM]",
            add_summary_token=True,
            block_size=512,
            overlap=64,
            soft_mask_temperature=0.1,
            cache_masks=True,
            cache_dir=None,
            visualize=False,
            visualization_dir=None,
            enable_shadow_model=False,
            shadow_model_name="Qwen/Qwen3-1.8B",
            shadow_model_device="cpu",
            shadow_model_cache_dir=None,
            enable_prompt_composer=False,
            focus_tags=False,
            core_tag="[CORE_CTX]",
            near_tag="[NEAR_CTX]",
            summary_tag="[SUMMARY_CTX]",
        )

    @classmethod
    def for_openai(cls) -> "GASAConfig":
        """
        Create a configuration optimized for OpenAI models.

        Returns
        -------
            GASAConfig: Configuration for OpenAI

        """
        config = cls.default()
        config.enable_shadow_model = True
        config.shadow_model_name = "Qwen/Qwen3-1.8B"
        config.fallback_strategy = FallbackStrategy.PROMPT_COMPOSER
        config.enable_prompt_composer = True
        config.focus_tags = True
        return config

    @classmethod
    def for_vllm(cls) -> "GASAConfig":
        """
        Create a configuration optimized for vLLM models.

        Returns
        -------
            GASAConfig: Configuration for vLLM

        """
        config = cls.default()
        config.enable_shadow_model = False
        config.fallback_strategy = FallbackStrategy.BLOCK_DIAGONAL
        config.enable_prompt_composer = True
        config.focus_tags = True
        return config

    @classmethod
    def for_local_models(cls) -> "GASAConfig":
        """
        Create a configuration optimized for local models.

        Returns
        -------
            GASAConfig: Configuration for local models

        """
        config = cls.default()
        config.enable_shadow_model = False
        config.fallback_strategy = FallbackStrategy.BLOCK_DIAGONAL
        config.enable_prompt_composer = False
        config.focus_tags = False
        return config

    @classmethod
    def from_cli_args(cls, args: dict[str, Any]) -> "GASAConfig":
        """
        Create a configuration from command-line arguments.

        Args:
        ----
            args: Command-line arguments

        Returns:
        -------
            GASAConfig: Configuration

        Raises:
        ------
            ValueError: If any argument values are invalid

        """
        # Start with default configuration
        config = cls.default()

        # Map CLI args to config attributes
        arg_to_attr = {
            "gasa": "enabled",
            "gasa_hop": "max_hops",
            "gasa_strategy": "mask_strategy",
            "gasa_fallback": "fallback_strategy",
            "gasa_block_size": "block_size",
            "gasa_overlap": "overlap",
            "gasa_cache": "cache_masks",
            "gasa_cache_dir": "cache_dir",
            "gasa_visualize": "visualize",
            "gasa_visualization_dir": "visualization_dir",
            "gasa_shadow_model": "enable_shadow_model",
            "gasa_shadow_model_name": "shadow_model_name",
            "gasa_shadow_model_device": "shadow_model_device",
            "gasa_shadow_model_cache_dir": "shadow_model_cache_dir",
            "gasa_prompt_composer": "enable_prompt_composer",
            "gasa_focus_tags": "focus_tags",
        }

        # Update config with CLI args
        for arg_name, attr_name in arg_to_attr.items():
            if arg_name in args:
                # Special handling for enum types
                if attr_name == "mask_strategy":
                    config.mask_strategy = MaskStrategy(args[arg_name])
                elif attr_name == "fallback_strategy":
                    config.fallback_strategy = FallbackStrategy(args[arg_name])
                else:
                    setattr(config, attr_name, args[arg_name])

        return config
