from __future__ import annotations

"""
Validator Service Builder API module for Saplings.

This module provides the validator service builder implementation.
"""

from typing import Any

from saplings.api.stability import stable


@stable
class ValidatorServiceBuilder:
    """
    Builder for creating ValidatorService instances with a fluent interface.

    This builder provides a convenient way to configure and create ValidatorService
    instances with various options and dependencies.
    """

    def __init__(self):
        """Initialize the validator service builder."""
        self._model = None
        self._trace_manager = None
        self._config = {}

    def with_model(self, model: Any) -> "ValidatorServiceBuilder":
        """
        Set the model for the validator service.

        Args:
        ----
            model: Model to use for validation

        Returns:
        -------
            Self for method chaining

        """
        self._model = model
        return self

    def with_trace_manager(self, trace_manager: Any) -> "ValidatorServiceBuilder":
        """
        Set the trace manager for the validator service.

        Args:
        ----
            trace_manager: Trace manager for monitoring

        Returns:
        -------
            Self for method chaining

        """
        self._trace_manager = trace_manager
        return self

    def with_config(self, **kwargs) -> "ValidatorServiceBuilder":
        """
        Set additional configuration options for the validator service.

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
        Build the validator service instance.

        Returns
        -------
            ValidatorService instance

        """

        # Use a factory function to avoid circular imports
        def create_service():
            # Import the ValidationConfig
            from saplings.api.core import ValidationConfig
            from saplings.api.services.validator import ValidatorService

            # Create a config object
            config = ValidationConfig(**self._config)

            # Create the service
            return ValidatorService(
                model=self._model, config=config, trace_manager=self._trace_manager
            )

        return create_service()


__all__ = [
    "ValidatorServiceBuilder",
]
