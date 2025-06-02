from __future__ import annotations

"""
Execution Service Builder API module for Saplings.

This module provides the execution service builder implementation.
"""

from typing import Any

from saplings.api.stability import stable


@stable
class ExecutionServiceBuilder:
    """
    Builder for creating ExecutionService instances with a fluent interface.

    This builder provides a convenient way to configure and create ExecutionService
    instances with various options and dependencies.
    """

    def __init__(self):
        """Initialize the execution service builder."""
        self._model = None
        self._tool_service = None
        self._memory_manager = None
        self._trace_manager = None
        self._config = {}

    def with_model(self, model: Any) -> "ExecutionServiceBuilder":
        """
        Set the model for the execution service.

        Args:
        ----
            model: Model to use for execution

        Returns:
        -------
            Self for method chaining

        """
        self._model = model
        return self

    def with_tool_service(self, tool_service: Any) -> "ExecutionServiceBuilder":
        """
        Set the tool service for the execution service.

        Args:
        ----
            tool_service: Tool service for executing tools

        Returns:
        -------
            Self for method chaining

        """
        self._tool_service = tool_service
        return self

    def with_memory_manager(self, memory_manager: Any) -> "ExecutionServiceBuilder":
        """
        Set the memory manager for the execution service.

        Args:
        ----
            memory_manager: Memory manager for storing and retrieving documents

        Returns:
        -------
            Self for method chaining

        """
        self._memory_manager = memory_manager
        return self

    def with_trace_manager(self, trace_manager: Any) -> "ExecutionServiceBuilder":
        """
        Set the trace manager for the execution service.

        Args:
        ----
            trace_manager: Trace manager for monitoring

        Returns:
        -------
            Self for method chaining

        """
        self._trace_manager = trace_manager
        return self

    def with_config(self, **kwargs) -> "ExecutionServiceBuilder":
        """
        Set additional configuration options for the execution service.

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
        Build the execution service instance.

        Returns
        -------
            ExecutionService instance

        """

        # Use a factory function to avoid circular imports
        def create_service():
            from saplings.api.services.execution import ExecutionService

            # Create the service
            return ExecutionService(
                model=self._model,
                tool_service=self._tool_service,
                memory_manager=self._memory_manager,
                trace_manager=self._trace_manager,
                **self._config,
            )

        return create_service()


__all__ = [
    "ExecutionServiceBuilder",
]
