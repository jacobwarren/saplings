from __future__ import annotations

"""
Unit tests for the monitoring service.
"""


from unittest.mock import MagicMock, patch

from saplings.core.interfaces import IMonitoringService
from saplings.monitoring.config import MonitoringConfig
from saplings.services.monitoring_service import MonitoringService


class TestMonitoringService:
    THRESHOLD_1 = 0.8

    """Test the monitoring service."""

    def setup_method(self) -> None:
        """Set up test environment."""
        # Create monitoring service
        self.config = MonitoringConfig(
            enable_tracing=True, enable_langsmith=False, enable_blame_graph=True
        )
        self.service = MonitoringService(config=self.config)

    def test_initialization(self) -> None:
        """Test monitoring service initialization."""
        assert self.service.config is self.config
        assert self.service.trace_manager is not None
        assert self.service.blame_graph is not None
        assert self.service.langsmith_exporter is None  # Disabled in config

    def test_create_trace(self) -> None:
        """Test creating a trace."""
        # Create a trace
        trace = self.service.create_trace()

        # Verify trace
        assert trace is not None
        assert trace.trace_id is not None
        assert len(trace.spans) == 0

    def test_start_span(self) -> None:
        """Test starting a span."""
        # Create a trace
        trace = self.service.create_trace()

        # Start a span
        span = self.service.start_span(
            name="test_span", trace_id=trace.trace_id, attributes={"test_attr": "test_value"}
        )

        # Verify span
        assert span is not None
        assert span.name == "test_span"
        assert span.trace_id == trace.trace_id
        assert span.span_id is not None
        assert span.parent_id is None
        assert span.attributes["test_attr"] == "test_value"
        assert span.start_time is not None
        assert span.end_time is None  # Not ended yet

    def test_end_span(self) -> None:
        """Test ending a span."""
        # Create a trace
        trace = self.service.create_trace()

        # Start a span
        span = self.service.start_span(name="test_span", trace_id=trace.trace_id)

        # End the span
        self.service.end_span(span.span_id)

        # Get the updated span
        updated_span = self.service.trace_manager.get_span(span.span_id)

        # Verify span was ended
        assert updated_span.end_time is not None

    def test_add_span_event(self) -> None:
        """Test adding an event to a span."""
        # Create a trace
        trace = self.service.create_trace()

        # Start a span
        span = self.service.start_span(name="test_span", trace_id=trace.trace_id)

        # Add an event
        self.service.add_span_event(
            span_id=span.span_id, name="test_event", attributes={"event_attr": "event_value"}
        )

        # Get the updated span
        updated_span = self.service.trace_manager.get_span(span.span_id)

        # Verify event was added
        assert len(updated_span.events) == 1
        assert updated_span.events[0].name == "test_event"
        assert updated_span.events[0].attributes["event_attr"] == "event_value"

    def test_add_span_attribute(self) -> None:
        """Test adding an attribute to a span."""
        # Create a trace
        trace = self.service.create_trace()

        # Start a span
        span = self.service.start_span(name="test_span", trace_id=trace.trace_id)

        # Add an attribute
        self.service.add_span_attribute(span_id=span.span_id, key="new_attr", value="new_value")

        # Get the updated span
        updated_span = self.service.trace_manager.get_span(span.span_id)

        # Verify attribute was added
        assert updated_span.attributes["new_attr"] == "new_value"

    def test_get_trace(self) -> None:
        """Test getting a trace."""
        # Create a trace
        trace = self.service.create_trace()

        # Start a span
        span = self.service.start_span(name="test_span", trace_id=trace.trace_id)

        # Get the trace
        retrieved_trace = self.service.get_trace(trace.trace_id)

        # Verify trace
        assert retrieved_trace is not None
        assert retrieved_trace.trace_id == trace.trace_id
        assert len(retrieved_trace.spans) == 1
        assert retrieved_trace.spans[0].span_id == span.span_id

    def test_get_traces(self) -> None:
        """Test getting all traces."""
        # Create traces
        trace1 = self.service.create_trace()
        trace2 = self.service.create_trace()

        # Start spans
        self.service.start_span(name="span1", trace_id=trace1.trace_id)
        self.service.start_span(name="span2", trace_id=trace2.trace_id)

        # Get all traces
        traces = self.service.get_traces()

        # Verify traces
        assert len(traces) >= 2
        assert any(t.trace_id == trace1.trace_id for t in traces)
        assert any(t.trace_id == trace2.trace_id for t in traces)

    def test_create_blame_node(self) -> None:
        """Test creating a blame node."""
        # Create a blame node
        node_id = self.service.create_blame_node(
            name="test_node", node_type="test_type", attributes={"node_attr": "node_value"}
        )

        # Verify node
        assert node_id is not None
        assert self.service.blame_graph.has_node(node_id)
        node = self.service.blame_graph.get_node(node_id)
        assert node.name == "test_node"
        assert node.component == "test_type"
        assert node.attributes["node_attr"] == "node_value"

    def test_add_blame_edge(self) -> None:
        """Test adding a blame edge."""
        # Create blame nodes
        source_id = self.service.create_blame_node(name="source", node_type="source_type")
        target_id = self.service.create_blame_node(name="target", node_type="target_type")

        # Add a blame edge
        edge_id = self.service.add_blame_edge(
            source_id=source_id,
            target_id=target_id,
            edge_type="test_edge",
            weight=0.8,
            attributes={"edge_attr": "edge_value"},
        )

        # Verify edge
        assert edge_id is not None
        assert self.service.blame_graph.has_edge(edge_id)
        edge = self.service.blame_graph.get_edge(edge_id)
        assert edge.source_id == source_id
        assert edge.target_id == target_id
        assert edge.relationship == "test_edge"
        assert edge.attributes["edge_attr"] == "edge_value"

    def test_export_trace(self) -> None:
        """Test exporting a trace."""
        # Create a trace
        trace = self.service.create_trace()

        # Start a span
        span = self.service.start_span(name="test_span", trace_id=trace.trace_id)

        # End the span
        self.service.end_span(span.span_id)

        # Export the trace
        with patch("builtins.open", MagicMock()) as mock_open:
            self.service.export_trace(trace.trace_id, "test_trace.json")
            mock_open.assert_called_once()

    def test_interface_compliance(self) -> None:
        """Test that MonitoringService implements IMonitoringService."""
        assert isinstance(self.service, IMonitoringService)

        # Check required methods
        assert hasattr(self.service, "create_trace")
        assert hasattr(self.service, "start_span")
        assert hasattr(self.service, "end_span")
        assert hasattr(self.service, "add_span_event")
        assert hasattr(self.service, "add_span_attribute")
        assert hasattr(self.service, "get_trace")
        assert hasattr(self.service, "get_traces")
        assert hasattr(self.service, "create_blame_node")
        assert hasattr(self.service, "add_blame_edge")
        assert hasattr(self.service, "export_trace")
