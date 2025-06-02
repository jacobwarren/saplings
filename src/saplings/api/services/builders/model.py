from __future__ import annotations

"""
Model Service Builder API module for Saplings.

This module provides the model service builder implementation.
"""

from typing import Any

from saplings.api.stability import stable


@stable
class ModelServiceBuilder:
    """
    Builder for creating model service instances with a fluent interface.

    This builder provides a convenient way to configure and create model service
    instances with various options and dependencies.
    """

    def __init__(self):
        """Initialize the model service builder."""
        self._provider = None
        self._model_name = None
        self._api_key = None
        self._config = {}

    def with_provider(self, provider: str) -> "ModelServiceBuilder":
        """
        Set the provider for the model service.

        Args:
        ----
            provider: Provider name (e.g., 'openai', 'anthropic', 'huggingface')

        Returns:
        -------
            Self for method chaining

        """
        self._provider = provider
        return self

    def with_model_name(self, model_name: str) -> "ModelServiceBuilder":
        """
        Set the model name for the model service.

        Args:
        ----
            model_name: Name of the model to use

        Returns:
        -------
            Self for method chaining

        """
        self._model_name = model_name
        return self

    def with_api_key(self, api_key: str) -> "ModelServiceBuilder":
        """
        Set the API key for the model service.

        Args:
        ----
            api_key: API key for the provider

        Returns:
        -------
            Self for method chaining

        """
        self._api_key = api_key
        return self

    def with_config(self, **kwargs) -> "ModelServiceBuilder":
        """
        Set additional configuration options for the model service.

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
        Build the model service instance.

        Returns
        -------
            Model service instance

        """

        # Use a factory function to avoid circular imports
        def create_service():
            # Import the LLMBuilder
            from saplings.api.model import LLMBuilder

            # Create the builder
            builder = LLMBuilder()

            # Configure the builder
            if self._provider:
                builder.with_provider(self._provider)
            if self._model_name:
                builder.with_model_name(self._model_name)
            if self._api_key:
                builder.with_api_key(self._api_key)

            # Add additional configuration
            for key, value in self._config.items():
                builder.with_config(**{key: value})

            # Build the model
            return builder.build()

        return create_service()


__all__ = [
    "ModelServiceBuilder",
]
