from __future__ import annotations

"""
Builder for ModalityService.

This module provides a builder for creating ModalityService instances with
proper configuration and dependency injection.
"""

from typing import Any, Callable, TypeVar, cast

from saplings.core._internal.builder import ServiceBuilder
from saplings.modality._internal.service.modality_service import ModalityService

T = TypeVar("T")


class ModalityServiceBuilder(ServiceBuilder[ModalityService]):
    """
    Builder for ModalityService.

    This class provides a fluent interface for building ModalityService instances with
    proper configuration and dependency injection. It separates configuration from
    initialization and ensures all required dependencies are provided.

    Example:
    -------
    ```python
    # Create a builder for ModalityService
    builder = ModalityServiceBuilder()

    # Configure the builder with dependencies and options
    service = builder.with_model(model) \
                    .with_supported_modalities(["text", "image"]) \
                    .with_trace_manager(trace_manager) \
                    .build()
    ```

    """

    def __init__(self) -> None:
        """Initialize the ModalityService builder."""
        super().__init__(ModalityService)
        self.require_dependency("model")

    def with_model(self, model: Any) -> ModalityServiceBuilder:
        """
        Set the model to use for processing.

        Args:
        ----
            model: LLM model to use for processing

        Returns:
        -------
            ModalityServiceBuilder: Builder instance for method chaining

        """
        self.with_dependency("model", model)
        return self

    def with_model_provider(self, provider: Callable[[], Any]) -> ModalityServiceBuilder:
        """
        Set the model provider function.

        Args:
        ----
            provider: Function that returns an LLM model

        Returns:
        -------
            ModalityServiceBuilder: Builder instance for method chaining

        """
        self.with_dependency("model_provider", provider)
        # Remove model from required dependencies if we have a provider
        if "model" in self._required_dependencies:
            self._required_dependencies.remove("model")
        return self

    def with_supported_modalities(self, modalities: list[str]) -> ModalityServiceBuilder:
        """
        Set the supported modalities.

        Args:
        ----
            modalities: List of supported modality names

        Returns:
        -------
            ModalityServiceBuilder: Builder instance for method chaining

        """
        self.with_dependency("supported_modalities", modalities)
        return self

    def with_trace_manager(self, trace_manager: Any) -> ModalityServiceBuilder:
        """
        Set the trace manager for monitoring.

        Args:
        ----
            trace_manager: Trace manager instance

        Returns:
        -------
            ModalityServiceBuilder: Builder instance for method chaining

        """
        self.with_dependency("trace_manager", trace_manager)
        return self

    def with_config(self, config: Any) -> ModalityServiceBuilder:
        """
        Set the configuration object.

        Args:
        ----
            config: Configuration object

        Returns:
        -------
            ModalityServiceBuilder: Builder instance for method chaining

        """
        self.with_dependency("config", config)
        return self

    def build(self) -> ModalityService:
        """
        Build the ModalityService instance.

        Returns
        -------
            ModalityService: Configured ModalityService instance

        Raises
        ------
            ValueError: If required dependencies are missing

        """
        # Validate dependencies
        self._validate_dependencies()

        # Create the service with dependencies
        return cast(
            ModalityService,
            self._service_class(
                model=self._dependencies.get("model"),
                supported_modalities=self._dependencies.get("supported_modalities"),
                trace_manager=self._dependencies.get("trace_manager"),
                config=self._dependencies.get("config"),
                model_provider=self._dependencies.get("model_provider"),
            ),
        )
