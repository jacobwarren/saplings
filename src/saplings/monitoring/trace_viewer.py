from __future__ import annotations

"""
TraceViewer module for Saplings monitoring.

This module provides the TraceViewer interface for exploring and visualizing traces.
"""


import json
import logging
import os
from typing import TYPE_CHECKING, Any

from saplings.monitoring.config import MonitoringConfig, VisualizationFormat

if TYPE_CHECKING:
    from datetime import datetime

    from saplings.monitoring.trace import TraceManager

logger = logging.getLogger(__name__)

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning(
        "Plotly not installed. Interactive visualizations will not be available. "
        "Install plotly with: pip install plotly"
    )


class TraceViewer:
    """
    TraceViewer interface for exploring and visualizing traces.

    This class provides functionality for exploring and visualizing traces
    collected by the monitoring system.
    """

    def __init__(
        self,
        trace_manager: TraceManager | None = None,
        config: MonitoringConfig | None = None,
    ) -> None:
        """
        Initialize the TraceViewer.

        Args:
        ----
            trace_manager: Trace manager to use
            config: Monitoring configuration

        """
        self.trace_manager = trace_manager
        self.config = config or MonitoringConfig()

        # Create output directory if it doesn't exist
        os.makedirs(self.config.visualization_output_dir, exist_ok=True)

        if not PLOTLY_AVAILABLE:
            logger.warning(
                "Plotly not installed. Interactive visualizations will not be available. "
                "Install plotly with: pip install plotly"
            )

    def view_trace(
        self,
        trace_id: str,
        output_path: str | None = None,
        format: VisualizationFormat | None = None,
        show: bool = False,
    ) -> Any | None:
        """
        View a trace.

        Args:
        ----
            trace_id: ID of the trace to view
            output_path: Path to save the visualization
            format: Format of the visualization
            show: Whether to show the visualization

        Returns:
        -------
            Optional[Any]: Visualization object if available

        """
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not installed. Cannot visualize trace.")
            return None

        # Get the trace
        if not self.trace_manager:
            logger.error("Trace manager not provided, cannot get trace")
            return None

        trace = self.trace_manager.get_trace(trace_id)
        if not trace:
            logger.error(f"Trace {trace_id} not found")
            return None

        # Create the visualization
        fig = self._create_trace_visualization(trace)

        # Save the visualization
        if output_path:
            self._save_visualization(fig, output_path, format)

        # Show the visualization
        if show:
            fig.show()

        return fig

    def view_traces(
        self,
        trace_ids: list[str],
        output_path: str | None = None,
        format: VisualizationFormat | None = None,
        show: bool = False,
    ) -> Any | None:
        """
        View multiple traces.

        Args:
        ----
            trace_ids: IDs of the traces to view
            output_path: Path to save the visualization
            format: Format of the visualization
            show: Whether to show the visualization

        Returns:
        -------
            Optional[Any]: Visualization object if available

        """
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not installed. Cannot visualize traces.")
            return None

        # Get the traces
        if not self.trace_manager:
            logger.error("Trace manager not provided, cannot get traces")
            return None

        traces = []
        for trace_id in trace_ids:
            trace = self.trace_manager.get_trace(trace_id)
            if trace:
                traces.append(trace)
            else:
                logger.warning(f"Trace {trace_id} not found")

        if not traces:
            logger.error("No valid traces found")
            return None

        # Create the visualization
        fig = self._create_traces_comparison(traces)

        # Save the visualization
        if output_path:
            self._save_visualization(fig, output_path, format)

        # Show the visualization
        if show:
            fig.show()

        return fig

    def view_span(
        self,
        trace_id: str,
        span_id: str,
        output_path: str | None = None,
        format: VisualizationFormat | None = None,
        show: bool = False,
    ) -> Any | None:
        """
        View a specific span within a trace.

        Args:
        ----
            trace_id: ID of the trace
            span_id: ID of the span to view
            output_path: Path to save the visualization
            format: Format of the visualization
            show: Whether to show the visualization

        Returns:
        -------
            Optional[Any]: Visualization object if available

        """
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not installed. Cannot visualize span.")
            return None

        # Get the trace
        if not self.trace_manager:
            logger.error("Trace manager not provided, cannot get trace")
            return None

        trace = self.trace_manager.get_trace(trace_id)
        if not trace:
            logger.error(f"Trace {trace_id} not found")
            return None

        # Find the span
        span = None
        for s in trace.spans:
            if s.span_id == span_id:
                span = s
                break

        if not span:
            logger.error(f"Span {span_id} not found in trace {trace_id}")
            return None

        # Create the visualization
        fig = self._create_span_visualization(span, trace)

        # Save the visualization
        if output_path:
            self._save_visualization(fig, output_path, format)

        # Show the visualization
        if show:
            fig.show()

        return fig

    def search_traces(
        self,
        query: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        max_results: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Search for traces matching a query.

        Args:
        ----
            query: Search query
            start_time: Start time for the search
            end_time: End time for the search
            max_results: Maximum number of results to return

        Returns:
        -------
            List[Dict[str, Any]]: List of matching traces

        """
        # Get all traces
        if not self.trace_manager:
            logger.error("Trace manager not provided, cannot list traces")
            return []

        all_traces = self.trace_manager.list_traces(start_time, end_time)

        # Filter traces by query
        matching_traces = []
        for trace in all_traces:
            if self._matches_query(trace, query):
                matching_traces.append(self._trace_to_dict(trace))
                if len(matching_traces) >= max_results:
                    break

        return matching_traces

    def filter_traces_by_component(
        self,
        component: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        max_results: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Filter traces by component.

        Args:
        ----
            component: Component name to filter by
            start_time: Start time for the filter
            end_time: End time for the filter
            max_results: Maximum number of results to return

        Returns:
        -------
            List[Dict[str, Any]]: List of matching traces

        """
        # Get all traces
        if not self.trace_manager:
            logger.error("Trace manager not provided, cannot list traces")
            return []

        all_traces = self.trace_manager.list_traces(start_time, end_time)

        # Filter traces by component
        matching_traces = []
        for trace in all_traces:
            if self._contains_component(trace, component):
                matching_traces.append(self._trace_to_dict(trace))
                if len(matching_traces) >= max_results:
                    break

        return matching_traces

    def export_trace(
        self,
        trace_id: str,
        output_path: str,
        format: VisualizationFormat = VisualizationFormat.JSON,
    ) -> bool:
        """
        Export a trace to a file.

        Args:
        ----
            trace_id: ID of the trace to export
            output_path: Path to save the trace
            format: Format of the export

        Returns:
        -------
            bool: Whether the export was successful

        """
        # Get the trace
        if not self.trace_manager:
            logger.error("Trace manager not provided, cannot get trace")
            return False

        trace = self.trace_manager.get_trace(trace_id)
        if not trace:
            logger.error(f"Trace {trace_id} not found")
            return False

        # Convert the trace to the desired format
        if format == VisualizationFormat.JSON:
            data = json.dumps(self._trace_to_dict(trace), indent=2)
            with open(output_path, "w") as f:
                f.write(data)
            return True
        if format in [VisualizationFormat.HTML, VisualizationFormat.PNG, VisualizationFormat.SVG]:
            if not PLOTLY_AVAILABLE:
                logger.error("Plotly not installed. Cannot export trace as visualization.")
                return False

            fig = self._create_trace_visualization(trace)
            self._save_visualization(fig, output_path, format)
            return True
        logger.error(f"Unsupported export format: {format}")
        return False

    def _create_trace_visualization(self, trace: Any) -> Any:
        """
        Create a visualization for a trace.

        Args:
        ----
            trace: Trace to visualize

        Returns:
        -------
            Any: Visualization object

        """
        # Create a timeline visualization
        fig = make_subplots(
            rows=1,
            cols=1,
            subplot_titles=["Trace Timeline"],
            specs=[[{"type": "scatter"}]],
        )

        # Add spans to the timeline
        spans = sorted(trace.spans, key=lambda s: s.start_time)

        for _i, span in enumerate(spans):
            # Calculate duration
            duration = (span.end_time - span.start_time).total_seconds() * 1000  # ms

            # Add span to the timeline
            fig.add_trace(
                go.Bar(
                    x=[duration],
                    y=[span.name],
                    orientation="h",
                    name=span.name,
                    hovertext=f"{span.name}<br>Duration: {duration:.2f} ms<br>Status: {span.status}",
                    marker={
                        "color": self._get_color_for_span(span),
                    },
                ),
                row=1,
                col=1,
            )

        # Update layout
        fig.update_layout(
            title=f"Trace: {trace.trace_id}",
            xaxis_title="Duration (ms)",
            yaxis_title="Span",
            height=max(600, len(spans) * 30),
            width=800,
            showlegend=False,
        )

        return fig

    def _create_traces_comparison(self, traces: list[Any]) -> Any:
        """
        Create a comparison visualization for multiple traces.

        Args:
        ----
            traces: Traces to compare

        Returns:
        -------
            Any: Visualization object

        """
        # Create a comparison visualization
        fig = make_subplots(
            rows=len(traces),
            cols=1,
            subplot_titles=[f"Trace: {trace.trace_id}" for trace in traces],
            specs=[[{"type": "scatter"}] for _ in traces],
            vertical_spacing=0.05,
        )

        for i, trace in enumerate(traces):
            # Add spans to the timeline
            spans = sorted(trace.spans, key=lambda s: s.start_time)

            for span in spans:
                # Calculate duration
                duration = (span.end_time - span.start_time).total_seconds() * 1000  # ms

                # Add span to the timeline
                fig.add_trace(
                    go.Bar(
                        x=[duration],
                        y=[span.name],
                        orientation="h",
                        name=span.name,
                        hovertext=f"{span.name}<br>Duration: {duration:.2f} ms<br>Status: {span.status}",
                        marker={
                            "color": self._get_color_for_span(span),
                        },
                    ),
                    row=i + 1,
                    col=1,
                )

        # Update layout
        fig.update_layout(
            title="Trace Comparison",
            height=max(600, len(traces) * 300),
            width=800,
            showlegend=False,
        )

        # Update x-axis titles
        for i in range(len(traces)):
            fig.update_xaxes(title_text="Duration (ms)", row=i + 1, col=1)

        return fig

    def _create_span_visualization(self, span: Any, trace: Any) -> Any:
        """
        Create a visualization for a specific span.

        Args:
        ----
            span: Span to visualize
            trace: Trace containing the span

        Returns:
        -------
            Any: Visualization object

        """
        # Create a detailed span visualization
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=["Span Timeline", "Span Details"],
            specs=[[{"type": "scatter"}], [{"type": "table"}]],
            row_heights=[0.7, 0.3],
        )

        # Calculate duration
        duration = (span.end_time - span.start_time).total_seconds() * 1000  # ms

        # Add span to the timeline
        fig.add_trace(
            go.Bar(
                x=[duration],
                y=[span.name],
                orientation="h",
                name=span.name,
                hovertext=f"{span.name}<br>Duration: {duration:.2f} ms<br>Status: {span.status}",
                marker={
                    "color": self._get_color_for_span(span),
                },
            ),
            row=1,
            col=1,
        )

        # Add child spans
        child_spans = [s for s in trace.spans if s.parent_id == span.span_id]
        for child in child_spans:
            # Calculate child duration
            child_duration = (child.end_time - child.start_time).total_seconds() * 1000  # ms

            # Add child span to the timeline
            fig.add_trace(
                go.Bar(
                    x=[child_duration],
                    y=[f"  {child.name}"],  # Indent child spans
                    orientation="h",
                    name=child.name,
                    hovertext=f"{child.name}<br>Duration: {child_duration:.2f} ms<br>Status: {child.status}",
                    marker={
                        "color": self._get_color_for_span(child),
                    },
                ),
                row=1,
                col=1,
            )

        # Add span details table
        details = []
        headers = ["Property", "Value"]

        # Add basic properties
        details.append(["Span ID", span.span_id])
        details.append(["Name", span.name])
        details.append(["Start Time", span.start_time.isoformat()])
        details.append(["End Time", span.end_time.isoformat()])
        details.append(["Duration", f"{duration:.2f} ms"])
        details.append(["Status", span.status])
        details.append(["Parent ID", span.parent_id or "None"])

        # Add attributes
        for key, value in span.attributes.items():
            details.append([f"Attribute: {key}", str(value)])

        # Add events
        for event in span.events:
            details.append([f"Event: {event.name}", event.timestamp.isoformat()])

        # Add table to the figure
        fig.add_trace(
            go.Table(
                header={
                    "values": headers,
                    "fill_color": "paleturquoise",
                    "align": "left",
                },
                cells={
                    "values": list(zip(*details)),
                    "fill_color": "lavender",
                    "align": "left",
                },
            ),
            row=2,
            col=1,
        )

        # Update layout
        fig.update_layout(
            title=f"Span: {span.name}",
            height=800,
            width=800,
            showlegend=False,
        )

        # Update x-axis title
        fig.update_xaxes(title_text="Duration (ms)", row=1, col=1)

        return fig

    def _save_visualization(
        self,
        fig: Any,
        output_path: str,
        format: VisualizationFormat | None = None,
    ) -> None:
        """
        Save a visualization to disk.

        Args:
        ----
            fig: Visualization to save
            output_path: Path to save the visualization
            format: Format of the visualization

        """
        # Use default format if not specified
        format = format or self.config.visualization_format

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save the visualization
        if format == VisualizationFormat.HTML:
            fig.write_html(output_path)
        elif format in (VisualizationFormat.PNG, VisualizationFormat.SVG):
            fig.write_image(output_path)
        elif format == VisualizationFormat.JSON:
            with open(output_path, "w") as f:
                json.dump(fig.to_dict(), f, indent=2)
        else:
            logger.warning(f"Unsupported visualization format: {format}")
            # Default to HTML
            fig.write_html(output_path)

        logger.info(f"Saved visualization to {output_path}")

    def _get_color_for_span(self, span: Any) -> str:
        """
        Get a color for a span based on its status.

        Args:
        ----
            span: Span to get color for

        Returns:
        -------
            str: Color for the span

        """
        if span.status == "OK":
            return "rgba(0, 128, 0, 0.7)"  # Green
        if span.status == "ERROR":
            return "rgba(255, 0, 0, 0.7)"  # Red
        if span.status == "WARNING":
            return "rgba(255, 165, 0, 0.7)"  # Orange
        return "rgba(0, 0, 255, 0.7)"  # Blue

    def _matches_query(self, trace: Any, query: str) -> bool:
        """
        Check if a trace matches a search query.

        Args:
        ----
            trace: Trace to check
            query: Search query

        Returns:
        -------
            bool: Whether the trace matches the query

        """
        # Check trace ID
        if query.lower() in trace.trace_id.lower():
            return True

        # Check span names
        for span in trace.spans:
            if query.lower() in span.name.lower():
                return True

            # Check span attributes
            for key, value in span.attributes.items():
                if query.lower() in key.lower() or query.lower() in str(value).lower():
                    return True

        return False

    def _contains_component(self, trace: Any, component: str) -> bool:
        """
        Check if a trace contains a specific component.

        Args:
        ----
            trace: Trace to check
            component: Component name to check for

        Returns:
        -------
            bool: Whether the trace contains the component

        """
        # Check if any span has the component attribute
        for span in trace.spans:
            # Check if the component is in the attributes
            if "component" in span.attributes:
                if component.lower() in span.attributes["component"].lower():
                    return True

        return False

    def _trace_to_dict(self, trace: Any) -> dict[str, Any]:
        """
        Convert a trace to a dictionary.

        Args:
        ----
            trace: Trace to convert

        Returns:
        -------
            Dict[str, Any]: Dictionary representation of the trace

        """
        # Convert spans to dictionaries
        spans = []
        for span in trace.spans:
            spans.append(
                {
                    "span_id": span.span_id,
                    "name": span.name,
                    "start_time": span.start_time.isoformat(),
                    "end_time": span.end_time.isoformat(),
                    "status": span.status,
                    "parent_id": span.parent_id,
                    "attributes": span.attributes,
                    "events": [
                        {
                            "name": event.name,
                            "timestamp": event.timestamp.isoformat(),
                            "attributes": event.attributes,
                        }
                        for event in span.events
                    ],
                }
            )

        # Create trace dictionary
        return {
            "trace_id": trace.trace_id,
            "start_time": trace.start_time.isoformat(),
            "end_time": trace.end_time.isoformat(),
            "status": trace.status,
            "attributes": trace.attributes,
            "spans": spans,
        }
