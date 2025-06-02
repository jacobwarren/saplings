from __future__ import annotations

"""
Builder for ExecutionService.

This module provides a builder for the ExecutionService to simplify initialization
and configuration. The builder pattern helps separate configuration from initialization
and provides a fluent interface for service creation.
"""

import logging
from typing import Any, Dict, Optional

from saplings.api.core.interfaces import IGasaService as IGASAService
from saplings.api.core.interfaces import IValidatorService
from saplings.core._internal.builder import ServiceBuilder
from saplings.core._internal.model_adapter import LLM
from saplings.services._internal.providers.execution_service import ExecutionService

logger = logging.getLogger(__name__)


class ExecutionServiceBuilder(ServiceBuilder[ExecutionService]):
    """
    Builder for ExecutionService.

    This class provides a fluent interface for building ExecutionService instances with
    proper configuration and dependency injection. It separates configuration from
    initialization and ensures all required dependencies are provided.

    Example:
    -------
    ```python
    # Create a builder for ExecutionService
    builder = ExecutionServiceBuilder()

    # Configure the builder with dependencies and options
    service = builder.with_model(model) \
                    .with_gasa(gasa_service) \
                    .with_validator(validator_service) \
                    .with_trace_manager(trace_manager) \
                    .with_config(config) \
                    .build()
    ```

    """

    def __init__(self) -> None:
        """Initialize the ExecutionService builder."""
        super().__init__(ExecutionService)
        # Model is optional for lazy initialization
        # self.require_dependency("model")

    def with_model(self, model: LLM) -> ExecutionServiceBuilder:
        """
        Set the model for the ExecutionService.

        Args:
        ----
            model: The LLM model to use

        Returns:
        -------
            The builder instance for method chaining

        """
        return self.with_dependency("model", model)

    def with_gasa(self, gasa_service: Optional[IGASAService]) -> ExecutionServiceBuilder:
        """
        Set the GASA service for the ExecutionService.

        Args:
        ----
            gasa_service: The GASA service to use

        Returns:
        -------
            The builder instance for method chaining

        """
        return self.with_dependency("gasa_service", gasa_service)

    def with_validator(
        self, validator_service: Optional[IValidatorService]
    ) -> ExecutionServiceBuilder:
        """
        Set the validator service for the ExecutionService.

        Args:
        ----
            validator_service: The validator service to use

        Returns:
        -------
            The builder instance for method chaining

        """
        return self.with_dependency("validator_service", validator_service)

    def with_trace_manager(self, trace_manager: Any) -> ExecutionServiceBuilder:
        """
        Set the trace manager for the ExecutionService.

        Args:
        ----
            trace_manager: The trace manager to use

        Returns:
        -------
            The builder instance for method chaining

        """
        return self.with_dependency("trace_manager", trace_manager)

    def with_config(self, config: Any) -> ExecutionServiceBuilder:
        """
        Set the configuration for the ExecutionService.

        Args:
        ----
            config: Configuration object (AgentConfig or dict)

        Returns:
        -------
            The builder instance for method chaining

        """
        self.with_dependency("config", config)
        return self
