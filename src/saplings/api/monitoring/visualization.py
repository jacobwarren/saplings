from __future__ import annotations

"""
Visualization API module for Saplings.

This module provides the public API for visualization components.
"""

from saplings.api.stability import beta
from saplings.monitoring._internal.trace_viewer import TraceViewer as _TraceViewer
from saplings.monitoring._internal.visualization import (
    GASAHeatmap as _GASAHeatmap,
)
from saplings.monitoring._internal.visualization import (
    PerformanceVisualizer as _PerformanceVisualizer,
)


@beta
class TraceViewer(_TraceViewer):
    """
    Viewer for traces.

    This class provides functionality for viewing and analyzing traces,
    including visualizing spans, events, and performance metrics.
    """


@beta
class GASAHeatmap(_GASAHeatmap):
    """
    Heatmap for GASA attention.

    This class provides functionality for visualizing GASA attention
    as a heatmap, showing the attention weights between tokens.
    """


@beta
class PerformanceVisualizer(_PerformanceVisualizer):
    """
    Visualizer for performance metrics.

    This class provides functionality for visualizing performance metrics,
    such as latency, throughput, and error rates.
    """


__all__ = [
    "TraceViewer",
    "GASAHeatmap",
    "PerformanceVisualizer",
]
