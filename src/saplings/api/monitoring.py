from __future__ import annotations

"""
Monitoring API module for Saplings.

This module provides the public API for monitoring components, including:
- OpenTelemetry (OTEL) tracing infrastructure
- Causal blame graph for identifying bottlenecks
- GASA heatmap visualization
- TraceViewer interface for trace exploration
- LangSmith export capabilities
"""

from saplings.api.stability import beta

# Import internal implementations
try:
    from saplings.monitoring._internal.blame_graph import BlameEdge as _BlameEdge
    from saplings.monitoring._internal.blame_graph import BlameGraph as _BlameGraph
    from saplings.monitoring._internal.blame_graph import BlameNode as _BlameNode
    from saplings.monitoring._internal.config import MonitoringConfig as _MonitoringConfig
    from saplings.monitoring._internal.langsmith import LangSmithExporter as _LangSmithExporter
    from saplings.monitoring._internal.trace import Span as _Span
    from saplings.monitoring._internal.trace import SpanContext as _SpanContext
    from saplings.monitoring._internal.trace import TraceManager as _TraceManager
    from saplings.monitoring._internal.trace_viewer import TraceViewer as _TraceViewer
    from saplings.monitoring._internal.visualization import GASAHeatmap as _GASAHeatmap
    from saplings.monitoring._internal.visualization import (
        PerformanceVisualizer as _PerformanceVisualizer,
    )
except ImportError as e:
    import logging

    logging.getLogger(__name__).warning(f"Error importing monitoring components: {e}")


@beta
class BlameEdge(_BlameEdge):
    """
    Edge in a blame graph connecting two nodes.

    Represents a causal relationship between two nodes in the blame graph,
    with an associated weight indicating the strength of the relationship.
    """


@beta
class BlameGraph(_BlameGraph):
    """
    Graph for identifying bottlenecks in agent execution.

    The blame graph is a directed graph where nodes represent components
    and edges represent causal relationships between components. The weights
    on edges indicate the strength of the causal relationship.
    """


@beta
class BlameNode(_BlameNode):
    """
    Node in a blame graph representing a component.

    Represents a component in the blame graph, with associated metadata
    such as the component name, type, and performance metrics.
    """


@beta
class GASAHeatmap(_GASAHeatmap):
    """
    Visualization of GASA attention patterns.

    Generates a heatmap visualization of the attention patterns in the
    GASA (Graph-Aligned Sparse Attention) mechanism, showing which tokens
    attend to which other tokens.
    """


@beta
class LangSmithExporter(_LangSmithExporter):
    """
    Exporter for sending traces to LangSmith.

    Exports traces from Saplings to LangSmith for visualization and analysis.
    Requires a LangSmith API key to be configured.
    """


@beta
class MonitoringConfig(_MonitoringConfig):
    """
    Configuration for monitoring components.

    This class defines the configuration options for monitoring components,
    including tracing, visualization, and export settings.
    """


@beta
class PerformanceVisualizer(_PerformanceVisualizer):
    """
    Visualizer for agent performance metrics.

    Generates visualizations of agent performance metrics, such as
    execution time, token usage, and component performance.
    """


@beta
class Span(_Span):
    """
    Span representing a unit of work in a trace.

    A span represents a single operation within a trace, with a start time,
    end time, and associated metadata. Spans can be nested to represent
    hierarchical relationships between operations.
    """


@beta
class SpanContext(_SpanContext):
    """
    Context for a span in a trace.

    Contains the context information for a span, including the trace ID,
    span ID, and parent span ID. Used to correlate spans across different
    components and services.
    """


@beta
class TraceManager(_TraceManager):
    """
    Manager for creating and managing traces.

    The trace manager is responsible for creating and managing traces,
    including creating spans, recording events, and exporting traces
    to external systems.
    """


@beta
class TraceViewer(_TraceViewer):
    """
    Interface for exploring traces.

    Provides an interface for exploring traces, including filtering,
    searching, and visualizing traces and spans.
    """
