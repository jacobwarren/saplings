from __future__ import annotations

"""
Builder for MonitoringService.

This module provides a builder for the MonitoringService, following the
builder pattern for service initialization.
"""

import logging
import os
from typing import Any, Protocol

from saplings._internal.monitoring.config import MonitoringConfig
from saplings._internal.monitoring.types import TracingBackend
from saplings._internal.services.builders.service_builder import ServiceBuilder
from saplings._internal.services.monitoring_service import MonitoringService


class MonitoringServiceBuilderProtocol(Protocol):
    """Protocol for MonitoringServiceBuilder to avoid circular imports."""

    def with_enabled(self, enabled: bool) -> MonitoringServiceBuilderProtocol:
        """Set whether monitoring is enabled."""
        ...

    def with_output_dir(self, output_dir: str) -> MonitoringServiceBuilderProtocol:
        """Set the output directory for visualizations."""
        ...

    def with_tracing_backend(self, backend: TracingBackend) -> MonitoringServiceBuilderProtocol:
        """Set the tracing backend."""
        ...

    def build(self) -> Any:
        """Build the monitoring service."""
        ...


# No need for TYPE_CHECKING imports

logger = logging.getLogger(__name__)


class MonitoringServiceBuilder(ServiceBuilder[MonitoringService], MonitoringServiceBuilderProtocol):
    """
    Builder for MonitoringService.

    This class provides a fluent interface for building MonitoringService instances with
    proper configuration and dependency injection. It separates configuration
    from initialization and ensures all required dependencies are provided.

    Example:
    -------
    ```python
    # Create a builder for MonitoringService
    builder = MonitoringServiceBuilder()

    # Configure the builder with dependencies and options
    service = builder.with_output_dir("./visualizations") \
                    .with_enabled(True) \
                    .with_tracing_backend(TracingBackend.CONSOLE) \
                    .build()
    ```

    """

    def __init__(self) -> None:
        """Initialize the monitoring service builder."""
        super().__init__(MonitoringService)
        self._output_dir = None
        self._enabled = True
        self._testing = False
        self._config = None
        self._tracing_backend = TracingBackend.CONSOLE
        self._visualization_output_dir = None
        self._enable_blame_graph = True
        self._enable_gasa_heatmap = True
        self._langsmith_api_key = None
        self._langsmith_project = None
        self._trace_callbacks = []

    def with_output_dir(self, output_dir: str) -> MonitoringServiceBuilder:
        """
        Set the output directory for visualizations.

        Args:
        ----
            output_dir: Directory for visualization outputs

        Returns:
        -------
            The builder instance for method chaining

        """
        self._output_dir = output_dir
        self.with_dependency("output_dir", output_dir)
        return self

    def with_enabled(self, enabled: bool) -> MonitoringServiceBuilder:
        """
        Set whether monitoring is enabled.

        Args:
        ----
            enabled: Whether monitoring is enabled

        Returns:
        -------
            The builder instance for method chaining

        """
        self._enabled = enabled
        self.with_dependency("enabled", enabled)
        return self

    def with_testing(self, testing: bool) -> MonitoringServiceBuilder:
        """
        Set whether the service is in testing mode.

        Args:
        ----
            testing: Whether the service is in testing mode

        Returns:
        -------
            The builder instance for method chaining

        """
        self._testing = testing
        self.with_dependency("testing", testing)
        return self

    def with_config(self, config: MonitoringConfig) -> MonitoringServiceBuilder:
        """
        Set the monitoring configuration.

        Args:
        ----
            config: Monitoring configuration

        Returns:
        -------
            The builder instance for method chaining

        """
        self._config = config
        self.with_dependency("config", config)
        return self

    def with_tracing_backend(self, backend: TracingBackend) -> MonitoringServiceBuilder:
        """
        Set the tracing backend.

        Args:
        ----
            backend: Tracing backend to use

        Returns:
        -------
            The builder instance for method chaining

        """
        self._tracing_backend = backend
        return self

    def with_visualization_output_dir(self, dir_path: str) -> MonitoringServiceBuilder:
        """
        Set the visualization output directory.

        Args:
        ----
            dir_path: Directory for visualization outputs

        Returns:
        -------
            The builder instance for method chaining

        """
        self._visualization_output_dir = dir_path
        return self

    def with_enable_blame_graph(self, enabled: bool) -> MonitoringServiceBuilder:
        """
        Set whether to enable the blame graph.

        Args:
        ----
            enabled: Whether to enable the blame graph

        Returns:
        -------
            The builder instance for method chaining

        """
        self._enable_blame_graph = enabled
        return self

    def with_enable_gasa_heatmap(self, enabled: bool) -> MonitoringServiceBuilder:
        """
        Set whether to enable GASA heatmap visualization.

        Args:
        ----
            enabled: Whether to enable GASA heatmap visualization

        Returns:
        -------
            The builder instance for method chaining

        """
        self._enable_gasa_heatmap = enabled
        return self

    def with_langsmith_config(self, api_key: str, project: str) -> MonitoringServiceBuilder:
        """
        Set LangSmith configuration.

        Args:
        ----
            api_key: LangSmith API key
            project: LangSmith project name

        Returns:
        -------
            The builder instance for method chaining

        """
        self._langsmith_api_key = api_key
        self._langsmith_project = project
        return self

    def with_trace_callback(self, callback: Any) -> MonitoringServiceBuilder:
        """
        Add a trace callback.

        Args:
        ----
            callback: Callback function for trace events

        Returns:
        -------
            The builder instance for method chaining

        """
        self._trace_callbacks.append(callback)
        return self

    def build(self) -> MonitoringService:
        """
        Build the monitoring service instance with the configured dependencies.

        Returns
        -------
            The initialized monitoring service instance

        """
        # Create config if not provided
        if not self._config:
            config = MonitoringConfig(
                enabled=self._enabled,
                tracing_backend=self._tracing_backend,
                enable_blame_graph=self._enable_blame_graph,
                enable_gasa_heatmap=self._enable_gasa_heatmap,
                langsmith_api_key=self._langsmith_api_key,
                langsmith_project=self._langsmith_project,
            )

            # Set visualization output directory if provided
            if self._visualization_output_dir:
                config.visualization_output_dir = self._visualization_output_dir
            elif self._output_dir:
                config.visualization_output_dir = os.path.join(self._output_dir, "visualizations")

            self.with_dependency("config", config)

        # Build the service
        service = super().build()

        # Add trace callbacks if any
        if hasattr(service, "_trace_manager") and service._trace_manager:
            for callback in self._trace_callbacks:
                service._trace_manager.trace_callbacks.append(callback)

        return service
