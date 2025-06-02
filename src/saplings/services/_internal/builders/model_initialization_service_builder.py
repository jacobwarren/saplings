from __future__ import annotations

"""
Builder for ModelInitializationService.

This module provides a builder for the ModelInitializationService to simplify initialization
and configuration. The builder pattern helps separate configuration from initialization
and provides a fluent interface for service creation.
"""

import logging
from typing import Any, Dict

from saplings.core._internal.builder import ServiceBuilder
from saplings.services._internal.managers.model_initialization_service import (
    ModelInitializationService,
)

logger = logging.getLogger(__name__)


class ModelInitializationServiceBuilder(ServiceBuilder[ModelInitializationService]):
    """
    Builder for ModelInitializationService.

    This class provides a fluent interface for building ModelInitializationService instances with
    proper configuration and dependency injection. It separates configuration from initialization
    and ensures all required dependencies are provided.

    Example:
    -------
    ```python
    # Create a builder for ModelInitializationService
    builder = ModelInitializationServiceBuilder()

    # Configure the builder with dependencies and options
    service = builder.with_provider("openai") \
                    .with_model_name("gpt-4o") \
                    .with_model_parameters({"temperature": 0.7}) \
                    .build()
    ```

    """

    def __init__(self) -> None:
        """Initialize the ModelInitializationService builder."""
        super().__init__(ModelInitializationService)
        self.require_dependency("provider")
        self.require_dependency("model_name")
        self._model_parameters: Dict[str, Any] = {}

    def with_provider(self, provider: str) -> ModelInitializationServiceBuilder:
        """
        Set the provider for the ModelInitializationService.

        Args:
        ----
            provider: The provider name (e.g., "openai", "anthropic")

        Returns:
        -------
            The builder instance for method chaining

        """
        return self.with_dependency("provider", provider)

    def with_model_name(self, model_name: str) -> ModelInitializationServiceBuilder:
        """
        Set the model name for the ModelInitializationService.

        Args:
        ----
            model_name: The model name

        Returns:
        -------
            The builder instance for method chaining

        """
        return self.with_dependency("model_name", model_name)

    def with_retry_config(self, retry_config: Dict[str, Any]) -> ModelInitializationServiceBuilder:
        """
        Set the retry configuration for the ModelInitializationService.

        Args:
        ----
            retry_config: Retry configuration dictionary

        Returns:
        -------
            The builder instance for method chaining

        """
        return self.with_dependency("retry_config", retry_config)

    def with_circuit_breaker_config(
        self, circuit_breaker_config: Dict[str, Any]
    ) -> ModelInitializationServiceBuilder:
        """
        Set the circuit breaker configuration for the ModelInitializationService.

        Args:
        ----
            circuit_breaker_config: Circuit breaker configuration dictionary

        Returns:
        -------
            The builder instance for method chaining

        """
        return self.with_dependency("circuit_breaker_config", circuit_breaker_config)

    def with_model_parameters(
        self, model_parameters: Dict[str, Any]
    ) -> ModelInitializationServiceBuilder:
        """
        Set additional model parameters for the ModelInitializationService.

        Args:
        ----
            model_parameters: Model parameters dictionary

        Returns:
        -------
            The builder instance for method chaining

        """
        self._model_parameters.update(model_parameters)
        return self

    def build(self) -> ModelInitializationService:
        """
        Build the ModelInitializationService instance with the configured dependencies.

        Returns
        -------
            The initialized ModelInitializationService instance

        Raises
        ------
            InitializationError: If service initialization fails

        """
        # Add model parameters to dependencies
        for key, value in self._model_parameters.items():
            self._dependencies[key] = value

        return super().build()
