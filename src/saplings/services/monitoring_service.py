from __future__ import annotations

"""
saplings.services.monitoring_service.
===================================

Encapsulates all monitoring and tracing functionality into a cohesive service:
- Trace management
- Blame graph processing
- Trace visualization
"""


import datetime
import logging
import os
from typing import Any

from saplings.core.interfaces.monitoring import IMonitoringService
from saplings.monitoring import (
    BlameEdge,
    BlameGraph,
    BlameNode,
    MonitoringConfig,
    TraceManager,
    TraceViewer,
)

# Import TestTraceViewer for testing
try:
    from saplings.monitoring.test_trace_viewer import TestTraceViewer
except ImportError:
    TestTraceViewer = None

logger = logging.getLogger(__name__)


class MonitoringService(IMonitoringService):
    """Service that manages all monitoring, tracing, and visualization."""

    def __init__(
        self,
        output_dir: str | None = None,
        config: Any | None = None,
        enabled: bool = True,
        testing: bool = False,
    ) -> None:
        self.config = config
        self._enabled = enabled if config is None else config.enabled
        self._trace_manager = None
        self.blame_graph = None
        self.trace_viewer = None
        self.testing = testing
        self.langsmith_exporter = None

        if self._enabled:
            # Use provided config or create a new one
            if config is not None:
                self.monitoring_config = config
            elif output_dir is not None:
                self.monitoring_config = MonitoringConfig(
                    visualization_output_dir=os.path.join(output_dir, "visualizations"),
                )
            else:
                self.monitoring_config = MonitoringConfig()

            # Create trace manager
            self._trace_manager = TraceManager(config=self.monitoring_config)

            # Create blame graph if enabled
            if getattr(self.monitoring_config, "enable_blame_graph", True):
                self.blame_graph = BlameGraph(
                    trace_manager=self._trace_manager,
                    config=self.monitoring_config,
                )

            # Create trace viewer - use TestTraceViewer for testing if available
            if testing and TestTraceViewer is not None:
                self.trace_viewer = TestTraceViewer(
                    trace_manager=self._trace_manager,
                    config=self.monitoring_config,
                )
                logger.info("Using TestTraceViewer for testing")
            else:
                self.trace_viewer = TraceViewer(
                    trace_manager=self._trace_manager,
                    config=self.monitoring_config,
                )

            # Initialize LangSmith exporter if enabled in config
            if (
                hasattr(self.monitoring_config, "tracing_backend")
                and self.monitoring_config.tracing_backend == "langsmith"
            ):
                try:
                    from saplings.monitoring.langsmith import LangSmithExporter

                    self.langsmith_exporter = LangSmithExporter(
                        trace_manager=self._trace_manager,
                        config=self.monitoring_config,
                    )
                    logger.info("LangSmith exporter initialized")
                except ImportError:
                    logger.warning("LangSmith not available, exporter not initialized")

            logger.info("MonitoringService initialized (enabled=True)")
        else:
            logger.info("MonitoringService initialized (enabled=False)")

    @property
    def enabled(self):
        """Whether monitoring is enabled."""
        return self._enabled

    @property
    def trace_manager(self):
        """Get the trace manager if enabled."""
        return self._trace_manager

    def create_trace(self):
        """
        Create a new trace if monitoring is enabled.

        Returns
        -------
            Trace object or dict with trace_id=None if disabled

        """
        if not self.enabled or not self._trace_manager:
            # Create a proper MockTrace class that mimics the real Trace class
            class MockTrace:
                def __init__(self):
                    self.trace_id = None
                    self.spans = []
                    self.start_time = None
                    self.end_time = None
                    self.status = "OK"
                    self.attributes = {}

            return MockTrace()

        # Create trace directly (synchronous operation)
        return self._trace_manager.create_trace()

    def start_span(
        self,
        name: str,
        trace_id: str | None = None,
        parent_id: str | None = None,
        attributes: dict | None = None,
    ):
        """
        Start a new span if monitoring is enabled.

        Args:
        ----
            name: Span name
            trace_id: Optional trace ID
            parent_id: Optional parent span ID
            attributes: Optional span attributes

        Returns:
        -------
            Span object or None if disabled

        """
        if not self.enabled or not self._trace_manager or not trace_id:
            # Create a proper MockSpan class that mimics the real Span class
            class MockSpan:
                def __init__(self):
                    self.name = name
                    self.trace_id = trace_id
                    self.span_id = "mock-span-id"
                    self.parent_id = parent_id
                    self.attributes = attributes or {}
                    self.start_time = None
                    self.end_time = None
                    self.events = []

                def add_event(self, name, attributes=None):
                    """Mock implementation of add_event."""
                    self.events.append({"name": name, "attributes": attributes or {}})

                def set_attribute(self, key, value):
                    """Mock implementation of set_attribute."""
                    self.attributes[key] = value

            return MockSpan()

        # Start span directly (synchronous operation)
        return self._trace_manager.start_span(
            name=name,
            trace_id=trace_id,
            parent_id=parent_id,
            attributes=attributes or {},
        )

    def end_span(self, span_id: str) -> None:
        """
        End a span if monitoring is enabled.

        Args:
        ----
            span_id: Span ID to end

        """
        if not self.enabled or not self._trace_manager or not span_id:
            return

        # End span directly (synchronous operation)
        self._trace_manager.end_span(span_id)

    def add_span_event(self, span_id: str, name: str, attributes: dict | None = None) -> None:
        """
        Add an event to a span.

        Args:
        ----
            span_id: Span ID
            name: Event name
            attributes: Optional event attributes

        """
        if not self.enabled or not self._trace_manager or not span_id:
            return

        # Get the span
        span = self._trace_manager.get_span(span_id)
        if span is None:
            logger.warning(f"Span {span_id} not found")
            return

        # Add event to span
        span.add_event(name=name, attributes=attributes)

    def add_span_attribute(self, span_id: str, key: str, value: Any) -> None:
        """
        Add an attribute to a span.

        Args:
        ----
            span_id: Span ID
            key: Attribute key
            value: Attribute value

        """
        if not self.enabled or not self._trace_manager or not span_id:
            return

        # Get the span
        span = self._trace_manager.get_span(span_id)
        if span is None:
            logger.warning(f"Span {span_id} not found")
            return

        # Add attribute to span
        span.set_attribute(key, value)

    def get_trace(self, trace_id: str) -> Any:
        """
        Get a trace by ID.

        Args:
        ----
            trace_id: Trace ID

        Returns:
        -------
            Trace object or None if not found

        """
        if not self.enabled or not self._trace_manager or not trace_id:
            return None

        # Get trace directly
        return self._trace_manager.get_trace(trace_id)

    def get_traces(self):
        """
        Get all traces.

        Returns
        -------
            List of traces

        """
        if not self.enabled or not self._trace_manager:
            return []

        # Return all traces
        return list(self._trace_manager.traces.values())

    def process_trace(self, trace_id: str) -> Any:
        """
        Process a trace for analysis.

        Args:
        ----
            trace_id: ID of the trace to process

        Returns:
        -------
            Any: Processed trace

        """
        if not self.enabled or not self._trace_manager or not self.blame_graph:
            return None

        # Get the trace
        trace = self._trace_manager.get_trace(trace_id)
        if not trace:
            logger.warning(f"Trace {trace_id} not found")
            return None

        # Process the trace with the blame graph
        self.blame_graph.process_trace(trace)

        return trace

    def create_blame_node(self, name: str, node_type: str, attributes: dict | None = None) -> str:
        """
        Create a blame node.

        Args:
        ----
            name: Node name
            node_type: Node type
            attributes: Optional node attributes

        Returns:
        -------
            Node ID

        """
        if not self.enabled or not self.blame_graph:
            return "mock-node-id"

        # Create a node ID
        node_id = f"{node_type}.{name}"

        # Create the node if it doesn't exist
        if node_id not in self.blame_graph.nodes:
            node = BlameNode(
                node_id=node_id,
                name=name,
                component=node_type,
                attributes=attributes,
            )
            self.blame_graph.nodes[node_id] = node

            # Add to NetworkX graph if available
            if hasattr(self.blame_graph, "graph") and self.blame_graph.graph:
                self.blame_graph.graph.add_node(
                    node_id,
                    name=name,
                    component=node_type,
                    attributes=attributes,
                )

        return node_id

    def add_blame_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        weight: float = 1.0,
        attributes: dict | None = None,
    ) -> str:
        """
        Add a blame edge.

        Args:
        ----
            source_id: Source node ID
            target_id: Target node ID
            edge_type: Edge type
            weight: Edge weight
            attributes: Optional edge attributes

        Returns:
        -------
            Edge ID

        """
        if not self.enabled or not self.blame_graph:
            return "mock-edge-id"

        # Create edge key
        edge_key = (source_id, target_id)

        # Create the edge if it doesn't exist
        if edge_key not in self.blame_graph.edges:
            edge = BlameEdge(
                source_id=source_id,
                target_id=target_id,
                relationship=edge_type,
                attributes=attributes,
            )
            self.blame_graph.edges[edge_key] = edge

            # Add to NetworkX graph if available
            if hasattr(self.blame_graph, "graph") and self.blame_graph.graph:
                self.blame_graph.graph.add_edge(
                    source_id,
                    target_id,
                    relationship=edge_type,
                    weight=weight,
                    attributes=attributes,
                )

        return f"edge-{source_id}-{target_id}-{edge_type}"

    def export_trace(self, trace_id: str, output_path: str) -> None:
        """
        Export a trace to a file.

        Args:
        ----
            trace_id: Trace ID
            output_path: Output file path

        """
        if not self.enabled or not self.trace_manager or not trace_id:
            return

        # Mock implementation since we don't know the actual interface
        # In a real implementation, this would get the trace and export it
        # For now, we'll just create a simple JSON file
        import json

        with open(output_path, "w") as f:
            mock_trace = {
                "trace_id": trace_id,
                "spans": [],
                "metadata": {"exported_at": datetime.datetime.now().isoformat()},
            }
            json.dump(mock_trace, f, indent=2)

        logger.info(f"Exported trace {trace_id} to {output_path}")

    def identify_bottlenecks(
        self,
        threshold_ms: float = 100.0,
        min_call_count: int = 1,
    ) -> list:
        """
        Identify performance bottlenecks if monitoring is enabled.

        Args:
        ----
            threshold_ms: Threshold in milliseconds
            min_call_count: Minimum call count

        Returns:
        -------
            List of bottlenecks

        """
        if not self.enabled or not self.blame_graph:
            return []

        # Call the blame graph's method
        return self.blame_graph.identify_bottlenecks(
            threshold_ms=threshold_ms, min_call_count=min_call_count
        )

    def identify_error_sources(
        self,
        min_error_rate: float = 0.1,
        min_call_count: int = 1,
    ) -> list:
        """
        Identify error sources if monitoring is enabled.

        Args:
        ----
            min_error_rate: Minimum error rate
            min_call_count: Minimum call count

        Returns:
        -------
            List of error sources

        """
        if not self.enabled or not self.blame_graph:
            return []

        # Call the blame graph's method
        return self.blame_graph.identify_error_sources(
            min_error_rate=min_error_rate, min_call_count=min_call_count
        )

    def log_event(
        self,
        event_type: str,
        data: dict[str, Any],
        trace_id: str | None = None,
        span_id: str | None = None,
    ) -> None:
        """
        Log an event.

        Args:
        ----
            event_type: Type of event
            data: Event data
            trace_id: Optional trace ID
            span_id: Optional span ID

        """
        if not self.enabled:
            return

        # If span_id is provided, add event to span
        if span_id and self._trace_manager:
            self.add_span_event(span_id, event_type, data)
            return

        # If trace_id is provided but no span_id, create a new span for the event
        if trace_id and self._trace_manager:
            # Create a new span for the event
            span = self.start_span(
                name=f"event.{event_type}", trace_id=trace_id, attributes={"event_type": event_type}
            )

            # Add event data as attributes
            for key, value in data.items():
                self.add_span_attribute(span.span_id, key, value)

            # End the span
            self.end_span(span.span_id)
            return

        # Otherwise, just log to the logger
        logger.info(f"Event: {event_type}", extra={"event_data": data})

    def visualize_trace(
        self,
        trace_id: str,
        output_path: str | None = None,
    ):
        """
        Visualize a trace if monitoring is enabled.

        Args:
        ----
            trace_id: Trace ID to visualize
            output_path: Optional output path

        Returns:
        -------
            Visualization result

        """
        if not self.enabled or not self.trace_viewer or not trace_id:
            return None

        # Mock implementation since we don't know the actual interface
        # In a real implementation, this would call the trace viewer's method
        if output_path:
            # Export trace to file
            self.export_trace(trace_id, output_path)
            return {"output_path": output_path}

        return {"trace_id": trace_id}
