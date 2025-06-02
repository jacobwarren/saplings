from __future__ import annotations

"""
Builder for ToolService.

This module provides a builder for the ToolService to simplify initialization
and configuration. The builder pattern helps separate configuration from initialization
and provides a fluent interface for service creation.
"""

import logging
from typing import TYPE_CHECKING, Any, List, Optional

from saplings.core._internal.builder import ServiceBuilder
from saplings.services._internal.providers.tool_service import ToolService

if TYPE_CHECKING:
    from saplings.api.tools import ToolRegistry

logger = logging.getLogger(__name__)


class ToolServiceBuilder(ServiceBuilder[ToolService]):
    """
    Builder for ToolService.

    This class provides a fluent interface for building ToolService instances with
    proper configuration and dependency injection. It separates configuration from
    initialization and ensures all required dependencies are provided.

    Example:
    -------
    ```python
    # Create a builder for ToolService
    builder = ToolServiceBuilder()

    # Configure the builder with dependencies and options
    service = builder.with_executor(executor) \
                    .with_allowed_imports(["os", "json"]) \
                    .with_sandbox_enabled(True) \
                    .with_enabled(True) \
                    .with_trace_manager(trace_manager) \
                    .build()
    ```

    """

    def __init__(self) -> None:
        """Initialize the ToolService builder."""
        super().__init__(ToolService)
        # Executor is no longer required for initialization
        # It can be provided later for lazy initialization

    def with_executor(self, executor: Any) -> "ToolServiceBuilder":
        """
        Set the executor for the ToolService.

        Args:
        ----
            executor: The executor to use for tool factory initialization

        Returns:
        -------
            The builder instance for method chaining

        """
        self.with_dependency("executor", executor)
        return self

    def with_allowed_imports(self, allowed_imports: Optional[List[str]]) -> "ToolServiceBuilder":
        """
        Set the allowed imports for the ToolService.

        Args:
        ----
            allowed_imports: List of allowed imports for dynamic tools

        Returns:
        -------
            The builder instance for method chaining

        """
        self.with_dependency("allowed_imports", allowed_imports)
        return self

    def with_sandbox_enabled(self, sandbox_enabled: bool) -> "ToolServiceBuilder":
        """
        Set whether the sandbox is enabled for the ToolService.

        Args:
        ----
            sandbox_enabled: Whether the sandbox is enabled for dynamic tools

        Returns:
        -------
            The builder instance for method chaining

        """
        self.with_dependency("sandbox_enabled", sandbox_enabled)
        return self

    def with_enabled(self, enabled: bool) -> "ToolServiceBuilder":
        """
        Set whether the ToolService is enabled.

        Args:
        ----
            enabled: Whether the ToolService is enabled

        Returns:
        -------
            The builder instance for method chaining

        """
        self.with_dependency("enabled", enabled)
        return self

    def with_trace_manager(self, trace_manager: Any) -> "ToolServiceBuilder":
        """
        Set the trace manager for the ToolService.

        Args:
        ----
            trace_manager: The trace manager to use for monitoring

        Returns:
        -------
            The builder instance for method chaining

        """
        self.with_dependency("trace_manager", trace_manager)
        return self

    def with_registry(self, registry: "ToolRegistry") -> "ToolServiceBuilder":
        """
        Set the tool registry for the ToolService.

        Args:
        ----
            registry: The tool registry to use

        Returns:
        -------
            The builder instance for method chaining

        """
        self.with_dependency("registry", registry)
        return self
