from __future__ import annotations

"""
Monitoring API module for Saplings.

This module provides the public API for monitoring and tracing.
"""

# Import from submodules to avoid circular imports
from saplings.api.monitoring.blame_graph import BlameEdge, BlameGraph, BlameNode
from saplings.api.monitoring.config import MonitoringConfig, MonitoringEvent, TracingBackend
from saplings.api.monitoring.langsmith import LANGSMITH_AVAILABLE, LangSmithExporter
from saplings.api.monitoring.service import IMonitoringService, MonitoringService
from saplings.api.monitoring.trace import Span, SpanContext, Trace, TraceManager
from saplings.api.monitoring.visualization import GASAHeatmap, PerformanceVisualizer, TraceViewer

__all__ = [
    # Blame graph
    "BlameEdge",
    "BlameGraph",
    "BlameNode",
    # Configuration
    "MonitoringConfig",
    "MonitoringEvent",
    "TracingBackend",
    # Service
    "IMonitoringService",
    "MonitoringService",
    # Trace
    "Span",
    "SpanContext",
    "Trace",
    "TraceManager",
    # Visualization
    "GASAHeatmap",
    "PerformanceVisualizer",
    "TraceViewer",
    # LangSmith
    "LangSmithExporter",
    "LANGSMITH_AVAILABLE",
]


# Use __getattr__ for lazy loading of builders
def __getattr__(name):
    """Lazy import attributes to avoid circular dependencies."""
    if name == "MonitoringServiceBuilder":
        from saplings.api.monitoring.builders import MonitoringServiceBuilder

        return MonitoringServiceBuilder

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
