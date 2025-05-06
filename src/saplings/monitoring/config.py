from __future__ import annotations

"""
Configuration module for Saplings monitoring.

This module provides configuration options for the monitoring system.
"""


from enum import Enum

from pydantic import BaseModel, Field


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


class MonitoringConfig(BaseModel):
    """Configuration for the monitoring system."""

    enabled: bool = Field(
        default=True,
        description="Whether monitoring is enabled",
    )

    tracing_backend: TracingBackend = Field(
        default=TracingBackend.CONSOLE,
        description="Tracing backend to use",
    )

    otel_endpoint: str | None = Field(
        default=None,
        description="OpenTelemetry endpoint URL",
    )

    langsmith_api_key: str | None = Field(
        default=None,
        description="LangSmith API key",
    )

    langsmith_project: str | None = Field(
        default=None,
        description="LangSmith project name",
    )

    trace_sampling_rate: float = Field(
        default=1.0,
        description="Sampling rate for traces (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
    )

    visualization_format: VisualizationFormat = Field(
        default=VisualizationFormat.HTML,
        description="Default format for visualizations",
    )

    visualization_output_dir: str = Field(
        default="./visualizations",
        description="Directory for visualization outputs",
    )

    enable_blame_graph: bool = Field(
        default=True,
        description="Whether to enable the causal blame graph",
    )

    enable_gasa_heatmap: bool = Field(
        default=True,
        description="Whether to enable GASA heatmap visualization",
    )

    max_spans_per_trace: int = Field(
        default=1000,
        description="Maximum number of spans per trace",
        gt=0,
    )

    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Additional metadata for traces",
    )

    class Config:
        """Pydantic configuration."""

        extra = "ignore"
