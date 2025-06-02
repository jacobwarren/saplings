from __future__ import annotations

"""
Tool Service Builder API module for Saplings.

This module provides the tool service builder implementation.
"""

from typing import Any

from saplings.api.stability import stable


@stable
class ToolServiceBuilder:
    """
    Builder for creating ToolService instances with a fluent interface.

    This builder provides a convenient way to configure and create ToolService
    instances with various options and dependencies.
    """

    def __init__(self):
        """Initialize the tool service builder."""
        self._model = None
        self._trace_manager = None
        self._config = {}

    def with_model(self, model: Any) -> "ToolServiceBuilder":
        """
        Set the model for the tool service.

        Args:
        ----
            model: Model to use for tool execution

        Returns:
        -------
            Self for method chaining

        """
        self._model = model
        return self

    def with_trace_manager(self, trace_manager: Any) -> "ToolServiceBuilder":
        """
        Set the trace manager for the tool service.

        Args:
        ----
            trace_manager: Trace manager for monitoring

        Returns:
        -------
            Self for method chaining

        """
        self._trace_manager = trace_manager
        return self

    def with_config(self, **kwargs) -> "ToolServiceBuilder":
        """
        Set additional configuration options for the tool service.

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
        Build the tool service instance.

        Returns
        -------
            ToolService instance

        """

        # Use a factory function to avoid circular imports
        def create_service():
            # Import the ToolConfig
            from saplings.api.core import ToolConfig
            from saplings.api.services.tool import ToolService

            # Create a config object
            config = ToolConfig(**self._config)

            # Create the service
            return ToolService(model=self._model, config=config, trace_manager=self._trace_manager)

        return create_service()


__all__ = [
    "ToolServiceBuilder",
]
