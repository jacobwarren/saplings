from __future__ import annotations

"""
Judge Service Builder API module for Saplings.

This module provides the judge service builder implementation.
"""

from typing import Any

from saplings.api.stability import stable


@stable
class JudgeServiceBuilder:
    """
    Builder for creating JudgeService instances with a fluent interface.

    This builder provides a convenient way to configure and create JudgeService
    instances with various options and dependencies.
    """

    def __init__(self):
        """Initialize the judge service builder."""
        self._model = None
        self._trace_manager = None
        self._config = {}

    def with_model(self, model: Any) -> "JudgeServiceBuilder":
        """
        Set the model for the judge service.

        Args:
        ----
            model: Model to use for judging

        Returns:
        -------
            Self for method chaining

        """
        self._model = model
        return self

    def with_trace_manager(self, trace_manager: Any) -> "JudgeServiceBuilder":
        """
        Set the trace manager for the judge service.

        Args:
        ----
            trace_manager: Trace manager for monitoring

        Returns:
        -------
            Self for method chaining

        """
        self._trace_manager = trace_manager
        return self

    def with_config(self, **kwargs) -> "JudgeServiceBuilder":
        """
        Set additional configuration options for the judge service.

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
        Build the judge service instance.

        Returns
        -------
            JudgeService instance

        """

        # Use a factory function to avoid circular imports
        def create_service():
            # Import the JudgeConfig
            from saplings.api.core import JudgeConfig
            from saplings.api.services.judge import JudgeService

            # Create a config object
            config = JudgeConfig(**self._config)

            # Create the service
            return JudgeService(model=self._model, config=config, trace_manager=self._trace_manager)

        return create_service()


__all__ = [
    "JudgeServiceBuilder",
]
