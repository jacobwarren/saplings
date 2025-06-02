from __future__ import annotations

"""
Orchestration Service Builder API module for Saplings.

This module provides the orchestration service builder implementation.
"""

from typing import Any

from saplings.api.stability import stable


@stable
class OrchestrationServiceBuilder:
    """
    Builder for creating OrchestrationService instances with a fluent interface.

    This builder provides a convenient way to configure and create OrchestrationService
    instances with various options and dependencies.
    """

    def __init__(self):
        """Initialize the orchestration service builder."""
        self._model = None
        self._memory_manager = None
        self._planner_service = None
        self._execution_service = None
        self._validator_service = None
        self._trace_manager = None
        self._config = {}

    def with_model(self, model: Any) -> "OrchestrationServiceBuilder":
        """
        Set the model for the orchestration service.

        Args:
        ----
            model: Model to use for orchestration

        Returns:
        -------
            Self for method chaining

        """
        self._model = model
        return self

    def with_memory_manager(self, memory_manager: Any) -> "OrchestrationServiceBuilder":
        """
        Set the memory manager for the orchestration service.

        Args:
        ----
            memory_manager: Memory manager for storing and retrieving documents

        Returns:
        -------
            Self for method chaining

        """
        self._memory_manager = memory_manager
        return self

    def with_planner_service(self, planner_service: Any) -> "OrchestrationServiceBuilder":
        """
        Set the planner service for the orchestration service.

        Args:
        ----
            planner_service: Planner service for creating plans

        Returns:
        -------
            Self for method chaining

        """
        self._planner_service = planner_service
        return self

    def with_execution_service(self, execution_service: Any) -> "OrchestrationServiceBuilder":
        """
        Set the execution service for the orchestration service.

        Args:
        ----
            execution_service: Execution service for executing tasks

        Returns:
        -------
            Self for method chaining

        """
        self._execution_service = execution_service
        return self

    def with_validator_service(self, validator_service: Any) -> "OrchestrationServiceBuilder":
        """
        Set the validator service for the orchestration service.

        Args:
        ----
            validator_service: Validator service for validating outputs

        Returns:
        -------
            Self for method chaining

        """
        self._validator_service = validator_service
        return self

    def with_trace_manager(self, trace_manager: Any) -> "OrchestrationServiceBuilder":
        """
        Set the trace manager for the orchestration service.

        Args:
        ----
            trace_manager: Trace manager for monitoring

        Returns:
        -------
            Self for method chaining

        """
        self._trace_manager = trace_manager
        return self

    def with_config(self, **kwargs) -> "OrchestrationServiceBuilder":
        """
        Set additional configuration options for the orchestration service.

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
        Build the orchestration service instance.

        Returns
        -------
            OrchestrationService instance

        """

        # Use a factory function to avoid circular imports
        def create_service():
            # Import directly from the service module to avoid circular imports
            # Import the OrchestrationConfig
            from saplings.api.core import OrchestrationConfig
            from saplings.api.services.orchestration import OrchestrationService

            # Create a config object
            config = OrchestrationConfig(**self._config)

            # Create the service
            return OrchestrationService(
                model=self._model,
                memory_manager=self._memory_manager,
                planner_service=self._planner_service,
                execution_service=self._execution_service,
                validator_service=self._validator_service,
                config=config,
                trace_manager=self._trace_manager,
            )

        return create_service()


__all__ = [
    "OrchestrationServiceBuilder",
]
