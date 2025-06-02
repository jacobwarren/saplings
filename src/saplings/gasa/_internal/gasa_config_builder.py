from __future__ import annotations

"""
GASA Config Builder module for Saplings.

This module provides a builder class for creating GASAConfig instances with
proper configuration. It separates configuration from initialization and
provides a fluent interface for setting configuration parameters.
"""

import logging
from typing import Any, Dict, List, Optional

from saplings.gasa._internal.config import FallbackStrategy, GASAConfig, MaskStrategy

logger = logging.getLogger(__name__)


class GASAConfigBuilder:
    """
    Builder for GASAConfig.

    This class provides a fluent interface for building GASAConfig instances with
    proper configuration. It separates configuration from initialization and
    provides a fluent interface for setting configuration parameters.

    Example:
    -------
    ```python
    # Create a builder for GASAConfig
    builder = GASAConfigBuilder()

    # Configure the builder with options
    config = builder.with_enabled(True) \
                   .with_max_hops(2) \
                   .with_mask_strategy(MaskStrategy.BINARY) \
                   .with_fallback_strategy(FallbackStrategy.BLOCK_DIAGONAL) \
                   .build()
    ```

    """

    def __init__(self) -> None:
        """Initialize the GASA config builder with default values."""
        self._config_params = GASAConfig.default().model_dump()

    def with_enabled(self, enabled: bool) -> GASAConfigBuilder:
        """
        Set whether GASA is enabled.

        Args:
        ----
            enabled: Whether GASA is enabled

        Returns:
        -------
            The builder instance for method chaining

        """
        self._config_params["enabled"] = enabled
        return self

    def with_max_hops(self, max_hops: int) -> GASAConfigBuilder:
        """
        Set the maximum number of hops for attention.

        Args:
        ----
            max_hops: Maximum number of hops for attention (h parameter)

        Returns:
        -------
            The builder instance for method chaining

        """
        self._config_params["max_hops"] = max_hops
        return self

    def with_mask_strategy(self, strategy: MaskStrategy) -> GASAConfigBuilder:
        """
        Set the mask strategy.

        Args:
        ----
            strategy: Strategy for applying attention masks

        Returns:
        -------
            The builder instance for method chaining

        """
        self._config_params["mask_strategy"] = strategy
        return self

    def with_fallback_strategy(self, strategy: FallbackStrategy) -> GASAConfigBuilder:
        """
        Set the fallback strategy.

        Args:
        ----
            strategy: Fallback strategy for models that don't support sparse attention

        Returns:
        -------
            The builder instance for method chaining

        """
        self._config_params["fallback_strategy"] = strategy
        return self

    def with_global_tokens(self, tokens: List[str]) -> GASAConfigBuilder:
        """
        Set the global tokens.

        Args:
        ----
            tokens: Tokens that should attend to all other tokens

        Returns:
        -------
            The builder instance for method chaining

        """
        self._config_params["global_tokens"] = tokens
        return self

    def with_summary_token(self, token: str) -> GASAConfigBuilder:
        """
        Set the summary token.

        Args:
        ----
            token: Token used for global summary

        Returns:
        -------
            The builder instance for method chaining

        """
        self._config_params["summary_token"] = token
        return self

    def with_add_summary_token(self, add: bool) -> GASAConfigBuilder:
        """
        Set whether to add a summary token if not present.

        Args:
        ----
            add: Whether to add a summary token if not present

        Returns:
        -------
            The builder instance for method chaining

        """
        self._config_params["add_summary_token"] = add
        return self

    def with_block_size(self, size: int) -> GASAConfigBuilder:
        """
        Set the block size for block-diagonal packing.

        Args:
        ----
            size: Block size for block-diagonal packing

        Returns:
        -------
            The builder instance for method chaining

        """
        self._config_params["block_size"] = size
        return self

    def with_overlap(self, overlap: int) -> GASAConfigBuilder:
        """
        Set the overlap between blocks for block-diagonal packing.

        Args:
        ----
            overlap: Overlap between blocks for block-diagonal packing

        Returns:
        -------
            The builder instance for method chaining

        """
        self._config_params["overlap"] = overlap
        return self

    def with_soft_mask_temperature(self, temperature: float) -> GASAConfigBuilder:
        """
        Set the temperature for soft masks.

        Args:
        ----
            temperature: Temperature for soft masks (lower = closer to binary)

        Returns:
        -------
            The builder instance for method chaining

        """
        self._config_params["soft_mask_temperature"] = temperature
        return self

    def with_cache_masks(self, cache: bool) -> GASAConfigBuilder:
        """
        Set whether to cache generated masks.

        Args:
        ----
            cache: Whether to cache generated masks

        Returns:
        -------
            The builder instance for method chaining

        """
        self._config_params["cache_masks"] = cache
        return self

    def with_cache_dir(self, directory: Optional[str]) -> GASAConfigBuilder:
        """
        Set the directory to cache masks.

        Args:
        ----
            directory: Directory to cache masks

        Returns:
        -------
            The builder instance for method chaining

        """
        self._config_params["cache_dir"] = directory
        return self

    def with_visualize(self, visualize: bool) -> GASAConfigBuilder:
        """
        Set whether to visualize masks.

        Args:
        ----
            visualize: Whether to visualize masks

        Returns:
        -------
            The builder instance for method chaining

        """
        self._config_params["visualize"] = visualize
        return self

    def with_visualization_dir(self, directory: Optional[str]) -> GASAConfigBuilder:
        """
        Set the directory to save visualizations.

        Args:
        ----
            directory: Directory to save visualizations

        Returns:
        -------
            The builder instance for method chaining

        """
        self._config_params["visualization_dir"] = directory
        return self

    def with_shadow_model(self, enabled: bool) -> GASAConfigBuilder:
        """
        Set whether to use a shadow model for tokenization.

        Args:
        ----
            enabled: Whether to use a shadow model for tokenization

        Returns:
        -------
            The builder instance for method chaining

        """
        self._config_params["enable_shadow_model"] = enabled
        return self

    def with_shadow_model_name(self, name: str) -> GASAConfigBuilder:
        """
        Set the shadow model name.

        Args:
        ----
            name: Shadow model name

        Returns:
        -------
            The builder instance for method chaining

        """
        self._config_params["shadow_model_name"] = name
        return self

    def with_shadow_model_device(self, device: str) -> GASAConfigBuilder:
        """
        Set the shadow model device.

        Args:
        ----
            device: Shadow model device

        Returns:
        -------
            The builder instance for method chaining

        """
        self._config_params["shadow_model_device"] = device
        return self

    def with_prompt_composer(self, enabled: bool) -> GASAConfigBuilder:
        """
        Set whether to use prompt composer.

        Args:
        ----
            enabled: Whether to use prompt composer

        Returns:
        -------
            The builder instance for method chaining

        """
        self._config_params["enable_prompt_composer"] = enabled
        return self

    def with_focus_tags(self, enabled: bool) -> GASAConfigBuilder:
        """
        Set whether to use focus tags.

        Args:
        ----
            enabled: Whether to use focus tags

        Returns:
        -------
            The builder instance for method chaining

        """
        self._config_params["focus_tags"] = enabled
        return self

    def with_config(self, config: Dict[str, Any]) -> GASAConfigBuilder:
        """
        Update configuration with a dictionary.

        Args:
        ----
            config: Configuration dictionary

        Returns:
        -------
            The builder instance for method chaining

        """
        self._config_params.update(config)
        return self

    def build(self) -> GASAConfig:
        """
        Build the GASA config instance with the configured parameters.

        Returns
        -------
            The initialized GASA config instance

        """
        return GASAConfig(**self._config_params)
