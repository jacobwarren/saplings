from __future__ import annotations

"""
Core types for the monitoring component.

This module defines the core types used by the monitoring component to avoid circular imports.
"""

from enum import Enum
from typing import Any, Dict


class TracingBackend(str, Enum):
    """Tracing backend options."""

    NONE = "none"
    """No tracing."""

    CONSOLE = "console"
    """Console output for traces."""

    OTEL = "otel"
    """OpenTelemetry."""

    LANGSMITH = "langsmith"
    """LangSmith integration for traces."""


class VisualizationFormat(str, Enum):
    """Visualization format options."""

    PNG = "png"
    """PNG image."""

    HTML = "html"
    """Interactive HTML."""

    JSON = "json"
    """JSON data."""

    SVG = "svg"
    """SVG image."""


class MonitoringEvent:
    """Base class for monitoring events."""

    def __init__(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Initialize a monitoring event.

        Args:
        ----
            event_type: Type of the event
            data: Event data

        """
        self.event_type = event_type
        self.data = data


__all__ = [
    "TracingBackend",
    "VisualizationFormat",
    "MonitoringEvent",
]
