from __future__ import annotations

"""
Monitoring configuration API module for Saplings.

This module provides the public API for monitoring configuration.
"""

from enum import Enum

from saplings.api.stability import beta
from saplings.monitoring._internal.config import MonitoringConfig as _MonitoringConfig
from saplings.monitoring._internal.types import MonitoringEvent as _MonitoringEvent


@beta
class MonitoringConfig(_MonitoringConfig):
    """
    Configuration for monitoring.

    This class provides configuration options for monitoring, including
    whether monitoring is enabled, the output directory for visualizations,
    and the tracing backend to use.
    """


@beta
class MonitoringEvent(_MonitoringEvent):
    """
    Event for monitoring.

    This class represents an event that can be monitored, such as a model
    call, tool execution, or agent action.
    """


@beta
class TracingBackend(str, Enum):
    """
    Backend for tracing.

    This enum defines the available backends for tracing, such as console,
    file, or OpenTelemetry.
    """

    NONE = "none"
    """No tracing."""

    CONSOLE = "console"
    """Console output for traces."""

    OTEL = "otel"
    """OpenTelemetry."""

    LANGSMITH = "langsmith"
    """LangSmith integration for traces."""


__all__ = [
    "MonitoringConfig",
    "MonitoringEvent",
    "TracingBackend",
]
