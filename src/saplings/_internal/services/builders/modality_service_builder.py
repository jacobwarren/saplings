from __future__ import annotations

"""
Builder for ModalityService.

This module provides a builder for the ModalityService to simplify initialization
and configuration. The builder pattern helps separate configuration from initialization
and provides a fluent interface for service creation.
"""

import logging
from typing import Any, List

from saplings._internal.services.modality_service import ModalityService
from saplings.core._internal.builder import ServiceBuilder
from saplings.core._internal.model_adapter import LLM

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
        self.require_dependency("model")

    def with_model(self, model: LLM) -> ModalityServiceBuilder:
        """
        Set the model for the ModalityService.

        Args:
        ----
            model: The LLM model to use

        Returns:
        -------
            The builder instance for method chaining

        """
        return self.with_dependency("model", model)

    def with_supported_modalities(self, supported_modalities: List[str]) -> ModalityServiceBuilder:
        """
        Set the supported modalities for the ModalityService.

        Args:
        ----
            supported_modalities: List of supported modalities

        Returns:
        -------
            The builder instance for method chaining

        """
        return self.with_dependency("supported_modalities", supported_modalities)

    def with_trace_manager(self, trace_manager: Any) -> ModalityServiceBuilder:
        """
        Set the trace manager for the ModalityService.

        Args:
        ----
            trace_manager: The trace manager to use

        Returns:
        -------
            The builder instance for method chaining

        """
        return self.with_dependency("trace_manager", trace_manager)
