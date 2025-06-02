from __future__ import annotations

"""
Interface module for Saplings monitoring.

This module provides common interfaces and types for the monitoring system
to avoid circular dependencies between components.
"""

import logging
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Union

logger = logging.getLogger(__name__)

# Enums for configuration


class TracingBackend(str, Enum):
    """Tracing backend options."""

    NONE = "none"  # No tracing
    CONSOLE = "console"  # Console output
    OTEL = "otel"  # OpenTelemetry
    LANGSMITH = "langsmith"  # LangSmith


class VisualizationFormat(str, Enum):
    """Visualization format options."""

    PNG = "png"  # PNG image
    HTML = "html"  # Interactive HTML
    JSON = "json"  # JSON data
    SVG = "svg"  # SVG image


# Base interfaces for monitoring components


class IMonitoringConfig(Protocol):
    """Interface for monitoring configuration."""

    enabled: bool
    tracing_backend: TracingBackend
    otel_endpoint: Optional[str]
    langsmith_api_key: Optional[str]
    langsmith_project: Optional[str]
    trace_sampling_rate: float
    visualization_format: VisualizationFormat
    visualization_output_dir: str
    enable_blame_graph: bool
    enable_gasa_heatmap: bool
    max_spans_per_trace: int
    metadata: Dict[str, str]


class ISpanEvent(Protocol):
    """Interface for span events."""

    name: str
    timestamp: datetime
    attributes: Dict[str, Any]


class ISpanContext(Protocol):
    """Interface for span context."""

    trace_id: str
    span_id: str
    parent_id: Optional[str]


class ISpan(Protocol):
    """Interface for spans."""

    name: str
    context: ISpanContext
    span_id: str
    parent_id: Optional[str]
    trace_id: str
    start_time: datetime
    end_time: Optional[datetime]
    status: str
    attributes: Dict[str, Any]
    events: List[ISpanEvent]

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> ISpanEvent:
        """Add an event to the span."""
        ...

    def set_status(self, status: str) -> None:
        """Set the status of the span."""
        ...

    def set_attribute(self, key: str, value: Any) -> None:
        """Set an attribute of the span."""
        ...

    def end(self) -> None:
        """End the span."""
        ...

    def duration_ms(self) -> float:
        """Get the duration of the span in milliseconds."""
        ...


class ITrace(Protocol):
    """Interface for traces."""

    trace_id: str
    start_time: datetime
    end_time: Optional[datetime]
    status: str
    attributes: Dict[str, Any]
    spans: List[ISpan]

    def add_span(self, span: ISpan) -> None:
        """Add a span to the trace."""
        ...

    def end(self) -> None:
        """End the trace."""
        ...

    def duration_ms(self) -> float:
        """Get the duration of the trace in milliseconds."""
        ...

    def get_root_spans(self) -> List[ISpan]:
        """Get the root spans of the trace."""
        ...

    def get_child_spans(self, parent_id: str) -> List[ISpan]:
        """Get the child spans of a parent span."""
        ...


class ITraceManager(Protocol):
    """Interface for trace managers."""

    traces: Dict[str, ITrace]
    active_traces: set[str]
    active_spans: Dict[str, ISpan]
    trace_callbacks: List[Callable[[str, str], None]]

    def register_trace_callback(self, callback: Callable[[str, str], None]) -> None:
        """Register a callback for trace lifecycle events."""
        ...

    def unregister_trace_callback(self, callback: Callable[[str, str], None]) -> bool:
        """Unregister a trace callback."""
        ...

    def create_trace(
        self, trace_id: Optional[str] = None, attributes: Optional[Dict[str, Any]] = None
    ) -> ITrace:
        """Create a new trace."""
        ...

    def start_span(
        self,
        name: str,
        trace_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> ISpan:
        """Start a new span."""
        ...

    def get_trace(self, trace_id: str) -> Optional[ITrace]:
        """Get a trace by ID."""
        ...


class IBlameNode(Protocol):
    """Interface for blame nodes."""

    node_id: str
    name: str
    component: str
    attributes: Dict[str, Any]
    total_time_ms: float
    call_count: int
    error_count: int
    avg_time_ms: float
    max_time_ms: float
    min_time_ms: float

    def update_metrics(self, duration_ms: float, is_error: bool = False) -> None:
        """Update performance metrics."""
        ...

    def to_dict(self) -> Dict[str, Any]:
        """Convert the node to a dictionary."""
        ...


class IBlameEdge(Protocol):
    """Interface for blame edges."""

    source_id: str
    target_id: str
    relationship: str
    attributes: Dict[str, Any]
    total_time_ms: float
    call_count: int
    error_count: int
    avg_time_ms: float

    def update_metrics(self, duration_ms: float, is_error: bool = False) -> None:
        """Update performance metrics."""
        ...

    def to_dict(self) -> Dict[str, Any]:
        """Convert the edge to a dictionary."""
        ...


class IBlameGraph(Protocol):
    """Interface for blame graphs."""

    nodes: Dict[str, IBlameNode]
    edges: Dict[tuple[str, str], IBlameEdge]

    def process_trace(self, trace: Union[str, ITrace]) -> None:
        """Process a trace to update the blame graph."""
        ...

    def identify_bottlenecks(
        self, threshold_ms: float = 100.0, min_call_count: int = 5
    ) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks in the graph."""
        ...

    def identify_error_sources(
        self, min_error_rate: float = 0.1, min_call_count: int = 5
    ) -> List[Dict[str, Any]]:
        """Identify error sources in the graph."""
        ...

    def get_critical_path(self, trace_id: str) -> List[Dict[str, Any]]:
        """Get the critical path for a trace."""
        ...

    def export_graph(self, output_path: str) -> bool:
        """Export the blame graph to a file."""
        ...
