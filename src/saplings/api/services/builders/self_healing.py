from __future__ import annotations

"""
Self-Healing Service Builder API module for Saplings.

This module provides the self-healing service builder implementation.
"""

from typing import Any

from saplings.api.stability import stable


@stable
class SelfHealingServiceBuilder:
    """
    Builder for creating SelfHealingService instances with a fluent interface.

    This builder provides a convenient way to configure and create SelfHealingService
    instances with various options and dependencies.
    """

    def __init__(self):
        """Initialize the self-healing service builder."""
        self._model = None
        self._memory_manager = None
        self._trace_manager = None
        self._config = {}

    def with_model(self, model: Any) -> "SelfHealingServiceBuilder":
        """
        Set the model for the self-healing service.

        Args:
        ----
            model: Model to use for self-healing

        Returns:
        -------
            Self for method chaining

        """
        self._model = model
        return self

    def with_memory_manager(self, memory_manager: Any) -> "SelfHealingServiceBuilder":
        """
        Set the memory manager for the self-healing service.

        Args:
        ----
            memory_manager: Memory manager for storing and retrieving documents

        Returns:
        -------
            Self for method chaining

        """
        self._memory_manager = memory_manager
        return self

    def with_trace_manager(self, trace_manager: Any) -> "SelfHealingServiceBuilder":
        """
        Set the trace manager for the self-healing service.

        Args:
        ----
            trace_manager: Trace manager for monitoring

        Returns:
        -------
            Self for method chaining

        """
        self._trace_manager = trace_manager
        return self

    def with_config(self, **kwargs) -> "SelfHealingServiceBuilder":
        """
        Set additional configuration options for the self-healing service.

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
        Build the self-healing service instance.

        Returns
        -------
            SelfHealingService instance

        """

        # Use a factory function to avoid circular imports
        def create_service():
            # Import directly from the service module to avoid circular imports
            # Import the SelfHealingConfig
            from saplings.api.core import SelfHealingConfig
            from saplings.api.services.self_healing import SelfHealingService

            # Create a config object
            config = SelfHealingConfig(**self._config)

            # Create the service
            return SelfHealingService(
                model=self._model,
                memory_manager=self._memory_manager,
                config=config,
                trace_manager=self._trace_manager,
            )

        return create_service()


__all__ = [
    "SelfHealingServiceBuilder",
]
