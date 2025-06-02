from __future__ import annotations

"""
Builder for ModalityService.

This module provides a builder for the ModalityService to simplify initialization
and configuration. The builder pattern helps separate configuration from initialization
and provides a fluent interface for service creation.
"""

import logging
from typing import Any, Callable, List

from saplings.core._internal.builder import ServiceBuilder
from saplings.core._internal.model_adapter import LLM
from saplings.modality._internal.registry import get_modality_handler_registry
from saplings.services._internal.providers.modality_service import ModalityService

logger = logging.getLogger(__name__)


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
        # Model is no longer required, as we can use model_provider instead
        self._registry = get_modality_handler_registry()
        self._custom_handlers = {}

    def with_model(self, model: LLM) -> "ModalityServiceBuilder":
        """
        Set the model for the ModalityService.

        Args:
        ----
            model: The LLM model to use

        Returns:
        -------
            The builder instance for method chaining

        """
        self.with_dependency("model", model)
        return self

    def with_model_provider(self, provider: Callable[[], LLM]) -> "ModalityServiceBuilder":
        """
        Set a model provider function for the ModalityService.

        This allows lazy initialization of the model when needed.

        Args:
        ----
            provider: Function that returns an LLM model

        Returns:
        -------
            The builder instance for method chaining

        """
        self.with_dependency("model_provider", provider)
        return self

    def with_supported_modalities(
        self, supported_modalities: List[str]
    ) -> "ModalityServiceBuilder":
        """
        Set the supported modalities for the ModalityService.

        Args:
        ----
            supported_modalities: List of supported modalities

        Returns:
        -------
            The builder instance for method chaining

        """
        self.with_dependency("supported_modalities", supported_modalities)
        return self

    def with_trace_manager(self, trace_manager: Any) -> "ModalityServiceBuilder":
        """
        Set the trace manager for the ModalityService.

        Args:
        ----
            trace_manager: The trace manager to use

        Returns:
        -------
            The builder instance for method chaining

        """
        self.with_dependency("trace_manager", trace_manager)
        return self

    def with_config(self, config: Any) -> "ModalityServiceBuilder":
        """
        Set the configuration for the ModalityService.

        Args:
        ----
            config: Configuration object

        Returns:
        -------
            The builder instance for method chaining

        """
        self.with_dependency("config", config)
        return self

    def register_handler_class(self, modality: str, handler_class: Any) -> "ModalityServiceBuilder":
        """
        Register a handler class for a specific modality.

        Args:
        ----
            modality: Modality name
            handler_class: Handler class to register

        Returns:
        -------
            The builder instance for method chaining

        """
        self._registry.register_handler_class(modality, handler_class)
        return self

    def register_handler_factory(
        self, modality: str, factory: Callable
    ) -> "ModalityServiceBuilder":
        """
        Register a handler factory for a specific modality.

        Args:
        ----
            modality: Modality name
            factory: Factory function to create handler instances

        Returns:
        -------
            The builder instance for method chaining

        """
        self._registry.register_handler_factory(modality, factory)
        return self
