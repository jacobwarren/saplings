from __future__ import annotations

"""
Builder for ValidatorService.

This module provides a builder for the ValidatorService to simplify initialization
and configuration. The builder pattern helps separate configuration from initialization
and provides a fluent interface for service creation.
"""

import logging
from typing import Any, Optional, Protocol

from saplings.api.core.interfaces import IJudgeService
from saplings.core._internal.builder import ServiceBuilder
from saplings.core._internal.model_adapter import LLM
from saplings.services._internal.providers.validator_service import ValidatorService

logger = logging.getLogger(__name__)


class IValidationStrategy(Protocol):
    """Interface for validation strategies."""

    async def validate(self, input_data, output_data, validation_type, trace_id):
        """Validate output data against input data."""
        ...


class ValidatorServiceBuilder(ServiceBuilder[ValidatorService]):
    """
    Builder for ValidatorService.

    This class provides a fluent interface for building ValidatorService instances with
    proper configuration and dependency injection. It separates configuration from
    initialization and ensures all required dependencies are provided.

    Example:
    -------
    ```python
    # Create a builder for ValidatorService
    builder = ValidatorServiceBuilder()

    # Configure the builder with dependencies and options
    service = builder.with_model(model) \
                    .with_judge_service(judge_service) \
                    .with_trace_manager(trace_manager) \
                    .with_validation_strategy(validation_strategy) \
                    .build()
    ```

    """

    def __init__(self) -> None:
        """Initialize the ValidatorService builder."""
        super().__init__(ValidatorService)

    def with_model(self, model: Optional[LLM]) -> "ValidatorServiceBuilder":
        """
        Set the model for the ValidatorService.

        Args:
        ----
            model: The LLM model to use

        Returns:
        -------
            The builder instance for method chaining

        """
        self.with_dependency("model", model)
        return self

    def with_judge_service(
        self, judge_service: Optional[IJudgeService]
    ) -> "ValidatorServiceBuilder":
        """
        Set the judge service for the ValidatorService.

        Args:
        ----
            judge_service: The judge service to use

        Returns:
        -------
            The builder instance for method chaining

        """
        self.with_dependency("judge_service", judge_service)
        return self

    def with_trace_manager(self, trace_manager: Any) -> "ValidatorServiceBuilder":
        """
        Set the trace manager for the ValidatorService.

        Args:
        ----
            trace_manager: The trace manager to use

        Returns:
        -------
            The builder instance for method chaining

        """
        self.with_dependency("trace_manager", trace_manager)
        return self

    def with_validation_strategy(
        self, validation_strategy: Optional[IValidationStrategy]
    ) -> "ValidatorServiceBuilder":
        """
        Set the validation strategy for the ValidatorService.

        Args:
        ----
            validation_strategy: The validation strategy to use

        Returns:
        -------
            The builder instance for method chaining

        """
        self.with_dependency("validation_strategy", validation_strategy)
        return self

    def with_validator_registry(self, validator_registry: Any) -> "ValidatorServiceBuilder":
        """
        Set the validator registry for the ValidatorService.

        Args:
        ----
            validator_registry: The validator registry to use

        Returns:
        -------
            The builder instance for method chaining

        """
        self.with_dependency("validator_registry", validator_registry)
        return self
