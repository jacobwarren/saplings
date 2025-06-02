from __future__ import annotations

"""
Builder for OrchestrationService.

This module provides a builder for the OrchestrationService to simplify initialization
and configuration. The builder pattern helps separate configuration from initialization
and provides a fluent interface for service creation.
"""

import logging
from typing import Any

from saplings.api.orchestration import GraphRunnerConfig
from saplings.core._internal.builder import ServiceBuilder
from saplings.core._internal.model_adapter import LLM
from saplings.services._internal.providers.orchestration_service import OrchestrationService

logger = logging.getLogger(__name__)


class OrchestrationServiceBuilder(ServiceBuilder[OrchestrationService]):
    """
    Builder for OrchestrationService.

    This class provides a fluent interface for building OrchestrationService instances with
    proper configuration and dependency injection. It separates configuration from
    initialization and ensures all required dependencies are provided.

    Example:
    -------
    ```python
    # Create a builder for OrchestrationService
    builder = OrchestrationServiceBuilder()

    # Configure the builder with dependencies and options
    service = builder.with_model(model) \
                    .with_trace_manager(trace_manager) \
                    .with_config(config) \
                    .build()
    ```

    """

    def __init__(self) -> None:
        """Initialize the OrchestrationService builder."""
        super().__init__(OrchestrationService)
        # Model is optional for lazy initialization
        # self.require_dependency("model")

    def with_model(self, model: LLM) -> "OrchestrationServiceBuilder":
        """
        Set the model for the OrchestrationService.

        Args:
        ----
            model: The LLM model to use

        Returns:
        -------
            The builder instance for method chaining

        """
        self.with_dependency("model", model)
        return self

    def with_trace_manager(self, trace_manager: Any) -> "OrchestrationServiceBuilder":
        """
        Set the trace manager for the OrchestrationService.

        Args:
        ----
            trace_manager: The trace manager to use

        Returns:
        -------
            The builder instance for method chaining

        """
        self.with_dependency("trace_manager", trace_manager)
        return self

    def with_config(self, config: GraphRunnerConfig) -> "OrchestrationServiceBuilder":
        """
        Set the configuration for the OrchestrationService.

        Args:
        ----
            config: The configuration for the graph runner

        Returns:
        -------
            The builder instance for method chaining

        """
        self.with_dependency("config", config)
        return self
