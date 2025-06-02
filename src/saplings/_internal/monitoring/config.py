from __future__ import annotations

"""
Monitoring configuration for Saplings.

This module provides configuration classes for monitoring services.
"""

import os
from dataclasses import dataclass, field
from typing import Any, List, Optional

from saplings._internal.monitoring.types import TracingBackend


@dataclass
class MonitoringConfig:
    """
    Configuration for monitoring services.

    This class provides configuration options for monitoring services, including
    tracing, visualization, and integration with external monitoring systems.
    """

    enabled: bool = True
    """Whether monitoring is enabled."""

    tracing_backend: TracingBackend = TracingBackend.CONSOLE
    """The tracing backend to use."""

    visualization_output_dir: Optional[str] = None
    """Directory for visualization outputs."""

    enable_blame_graph: bool = True
    """Whether to enable the blame graph."""

    enable_gasa_heatmap: bool = True
    """Whether to enable GASA heatmap visualization."""

    langsmith_api_key: Optional[str] = None
    """LangSmith API key."""

    langsmith_project: Optional[str] = None
    """LangSmith project name."""

    trace_callbacks: List[Any] = field(default_factory=list)
    """Callbacks for trace events."""

    def __post_init__(self):
        """Initialize the configuration."""
        # Set default visualization output directory if not provided
        if self.visualization_output_dir is None:
            self.visualization_output_dir = os.path.join(os.getcwd(), "visualizations")

        # Create the visualization output directory if it doesn't exist
        if self.enabled and (self.enable_blame_graph or self.enable_gasa_heatmap):
            os.makedirs(self.visualization_output_dir, exist_ok=True)
