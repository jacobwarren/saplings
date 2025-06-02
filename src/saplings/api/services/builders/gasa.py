from __future__ import annotations

"""
GASA Service Builder API module for Saplings.

This module provides the GASA service builder implementations.
"""

from typing import Any

from saplings.api.stability import stable


@stable
class GASAConfigBuilder:
    """
    Builder for creating GASAConfig instances with a fluent interface.

    This builder provides a convenient way to configure and create GASAConfig
    instances with various options.
    """

    def __init__(self):
        """Initialize the GASA config builder."""
        self._config = {}

    def with_mask_type(self, mask_type: str) -> "GASAConfigBuilder":
        """
        Set the mask type for the GASA config.

        Args:
        ----
            mask_type: Type of mask to use

        Returns:
        -------
            Self for method chaining

        """
        self._config["mask_type"] = mask_type
        return self

    def with_mask_strategy(self, mask_strategy: str) -> "GASAConfigBuilder":
        """
        Set the mask strategy for the GASA config.

        Args:
        ----
            mask_strategy: Strategy for applying masks

        Returns:
        -------
            Self for method chaining

        """
        self._config["mask_strategy"] = mask_strategy
        return self

    def with_config(self, **kwargs) -> "GASAConfigBuilder":
        """
        Set additional configuration options for the GASA config.

        Args:
        ----
            **kwargs: Additional configuration options

        Returns:
        -------
            Self for method chaining

        """
        self._config.update(kwargs)
        return self

    def build(self) -> Any:
        """
        Build the GASA config instance.

        Returns
        -------
            GASAConfig instance

        """

        # Use a factory function to avoid circular imports
        def create_config():
            from saplings.api.gasa import GASAConfig

            return GASAConfig(**self._config)

        return create_config()


@stable
class GASAServiceBuilder:
    """
    Builder for creating GASAService instances with a fluent interface.

    This builder provides a convenient way to configure and create GASAService
    instances with various options and dependencies.
    """

    def __init__(self):
        """Initialize the GASA service builder."""
        self._model = None
        self._config = None
        self._memory_manager = None
        self._trace_manager = None
        self._extra_config = {}

    def with_model(self, model: Any) -> "GASAServiceBuilder":
        """
        Set the model for the GASA service.

        Args:
        ----
            model: Model to use for GASA

        Returns:
        -------
            Self for method chaining

        """
        self._model = model
        return self

    def with_config(self, config: Any) -> "GASAServiceBuilder":
        """
        Set the config for the GASA service.

        Args:
        ----
            config: GASA configuration

        Returns:
        -------
            Self for method chaining

        """
        self._config = config
        return self

    def with_memory_manager(self, memory_manager: Any) -> "GASAServiceBuilder":
        """
        Set the memory manager for the GASA service.

        Args:
        ----
            memory_manager: Memory manager for retrieving documents

        Returns:
        -------
            Self for method chaining

        """
        self._memory_manager = memory_manager
        return self

    def with_trace_manager(self, trace_manager: Any) -> "GASAServiceBuilder":
        """
        Set the trace manager for the GASA service.

        Args:
        ----
            trace_manager: Trace manager for monitoring

        Returns:
        -------
            Self for method chaining

        """
        self._trace_manager = trace_manager
        return self

    def with_extra_config(self, **kwargs) -> "GASAServiceBuilder":
        """
        Set additional configuration options for the GASA service.

        Args:
        ----
            **kwargs: Additional configuration options

        Returns:
        -------
            Self for method chaining

        """
        self._extra_config.update(kwargs)
        return self

    def build(self) -> Any:
        """
        Build the GASA service instance.

        Returns
        -------
            GASAService instance

        """

        # Use a factory function to avoid circular imports
        def create_service():
            from saplings.api.gasa import GASAService

            return GASAService(
                model=self._model,
                config=self._config,
                memory_manager=self._memory_manager,
                trace_manager=self._trace_manager,
                **self._extra_config,
            )

        return create_service()


__all__ = [
    "GASAConfigBuilder",
    "GASAServiceBuilder",
]
