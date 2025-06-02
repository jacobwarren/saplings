from __future__ import annotations

"""
Monitoring service builder API module for Saplings.

This module provides the public API for building monitoring services.
"""


from saplings.services._internal.builders.monitoring_service_builder import (
    MonitoringServiceBuilder as _MonitoringServiceBuilder,
)
from saplings.api.stability import stable

# Use forward references for types to avoid circular imports


@stable
class MonitoringServiceBuilder(_MonitoringServiceBuilder):
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


__all__ = [
    "MonitoringServiceBuilder",
]
