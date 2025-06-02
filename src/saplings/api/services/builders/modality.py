from __future__ import annotations

"""
Modality Service Builder API module for Saplings.

This module provides the modality service builder implementation.
"""

from typing import Any

from saplings.api.stability import stable


@stable
class ModalityServiceBuilder:
    """
    Builder for creating ModalityService instances with a fluent interface.

    This builder provides a convenient way to configure and create ModalityService
    instances with various options and dependencies.
    """

    def __init__(self):
        """Initialize the modality service builder."""
        self._model = None
        self._trace_manager = None
        self._config = {}

    def with_model(self, model: Any) -> "ModalityServiceBuilder":
        """
        Set the model for the modality service.

        Args:
        ----
            model: Model to use for modality handling

        Returns:
        -------
            Self for method chaining

        """
        self._model = model
        return self

    def with_trace_manager(self, trace_manager: Any) -> "ModalityServiceBuilder":
        """
        Set the trace manager for the modality service.

        Args:
        ----
            trace_manager: Trace manager for monitoring

        Returns:
        -------
            Self for method chaining

        """
        self._trace_manager = trace_manager
        return self

    def with_config(self, **kwargs) -> "ModalityServiceBuilder":
        """
        Set additional configuration options for the modality service.

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
        Build the modality service instance.

        Returns
        -------
            ModalityService instance

        """

        # Use a factory function to avoid circular imports
        def create_service():
            # Import the ModalityConfig
            from saplings.api.core import ModalityConfig
            from saplings.api.services.modality import ModalityService

            # Create a config object
            config = ModalityConfig(**self._config)

            # Create the service
            return ModalityService(
                model=self._model, config=config, trace_manager=self._trace_manager
            )

        return create_service()


__all__ = [
    "ModalityServiceBuilder",
]
