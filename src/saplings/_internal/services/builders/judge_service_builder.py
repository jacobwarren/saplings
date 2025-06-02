from __future__ import annotations

"""
Builder for JudgeService.

This module provides a builder for the JudgeService to simplify initialization
and configuration. The builder pattern helps separate configuration from initialization
and provides a fluent interface for service creation.
"""

import logging
from typing import Any

from saplings._internal.services.judge_service import JudgeService
from saplings.core._internal.builder import ServiceBuilder
from saplings.core._internal.model_adapter import LLM

logger = logging.getLogger(__name__)


class JudgeServiceBuilder(ServiceBuilder[JudgeService]):
    """
    Builder for JudgeService.

    This class provides a fluent interface for building JudgeService instances with
    proper configuration and dependency injection. It separates configuration from
    initialization and ensures all required dependencies are provided.

    Example:
    -------
    ```python
    # Create a builder for JudgeService
    builder = JudgeServiceBuilder()

    # Configure the builder with dependencies and options
    service = builder.with_model(model) \
                    .with_trace_manager(trace_manager) \
                    .with_initialize_judge(True) \
                    .build()
    ```

    """

    def __init__(self) -> None:
        """Initialize the JudgeService builder."""
        super().__init__(JudgeService)
        self.require_dependency("model")
        self._initialize_judge = False

    def with_model(self, model: LLM) -> JudgeServiceBuilder:
        """
        Set the model for the JudgeService.

        Args:
        ----
            model: The LLM model to use

        Returns:
        -------
            The builder instance for method chaining

        """
        return self.with_dependency("model", model)

    def with_trace_manager(self, trace_manager: Any) -> JudgeServiceBuilder:
        """
        Set the trace manager for the JudgeService.

        Args:
        ----
            trace_manager: The trace manager to use

        Returns:
        -------
            The builder instance for method chaining

        """
        return self.with_dependency("trace_manager", trace_manager)

    def with_initialize_judge(self, initialize_judge: bool) -> JudgeServiceBuilder:
        """
        Set whether to initialize the judge during build.

        Args:
        ----
            initialize_judge: Whether to initialize the judge during build

        Returns:
        -------
            The builder instance for method chaining

        """
        self._initialize_judge = initialize_judge
        return self

    async def build_async(self) -> JudgeService:
        """
        Build the JudgeService instance with the configured dependencies and initialize the judge.

        This async version of build allows for initializing the judge during construction.

        Returns
        -------
            The initialized JudgeService instance

        Raises
        ------
            InitializationError: If service initialization fails

        """
        # Build the service
        service = self.build()

        # Initialize the judge if requested
        if self._initialize_judge:
            await service.initialize_judge()

        return service
