from __future__ import annotations

"""
Unit tests for the monitoring service.
"""


from unittest.mock import MagicMock, patch

from saplings.core.interfaces import IMonitoringService
from saplings.monitoring.config import MonitoringConfig, TracingBackend
from saplings.services.monitoring_service import MonitoringService


class TestMonitoringService:
    THRESHOLD_1 = 0.8

    """Test the monitoring service."""

    def setup_method(self) -> None:
        """Set up test environment."""
        # Create monitoring service
        self.config = MonitoringConfig(enabled=True, tracing_backend=TracingBackend.CONSOLE)
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

        # Mock the trace manager
        self.service._trace_manager = MagicMock()

        # End the span
        self.service.end_span(span.span_id)

        # Verify end_span was called on the trace manager
        self.service._trace_manager.end_span.assert_called_once_with(span.span_id)

    def test_add_span_event(self) -> None:
        """Test adding an event to a span."""
        # Create a trace
        trace = self.service.create_trace()

        # Start a span
        span = self.service.start_span(name="test_span", trace_id=trace.trace_id)

        # Mock the add_event method on the span
        with patch.object(span, "add_event") as mock_add_event:
            # Add an event
            self.service.add_span_event(
                span_id=span.span_id, name="test_event", attributes={"event_attr": "event_value"}
            )

            # Verify add_event was called with the right parameters
            mock_add_event.assert_called_once_with(
                name="test_event", attributes={"event_attr": "event_value"}
            )

    def test_add_span_attribute(self) -> None:
        """Test adding an attribute to a span."""
        # Create a trace
        trace = self.service.create_trace()

        # Start a span
        span = self.service.start_span(name="test_span", trace_id=trace.trace_id)

        # Mock the set_attribute method on the span
        with patch.object(span, "set_attribute") as mock_set_attribute:
            # Add an attribute
            self.service.add_span_attribute(span_id=span.span_id, key="new_attr", value="new_value")

            # Verify set_attribute was called with the right parameters
            mock_set_attribute.assert_called_once_with("new_attr", "new_value")

    def test_get_trace(self) -> None:
        """Test getting a trace."""
        # Create a trace (not used in this test but demonstrates the API)
        _ = self.service.create_trace()

        # Create a fixed trace ID for testing
        test_trace_id = "test-trace-id"

        # Mock the trace manager
        mock_trace = MagicMock()
        mock_trace.trace_id = test_trace_id
        mock_trace.spans = [MagicMock(span_id="test-span-id")]

        self.service._trace_manager = MagicMock()
        self.service._trace_manager.get_trace.return_value = mock_trace

        # Get the trace using the fixed trace ID
        retrieved_trace = self.service.get_trace(test_trace_id)

        # Verify trace
        assert retrieved_trace is not None
        assert retrieved_trace.trace_id == test_trace_id
        assert len(retrieved_trace.spans) == 1

        # Verify get_trace was called on the trace manager
        self.service._trace_manager.get_trace.assert_called_once_with(test_trace_id)

    def test_get_traces(self) -> None:
        """Test getting all traces."""
        # Create traces
        trace1 = self.service.create_trace()
        trace2 = self.service.create_trace()

        # Mock the trace manager
        mock_trace1 = MagicMock()
        mock_trace1.trace_id = trace1.trace_id

        mock_trace2 = MagicMock()
        mock_trace2.trace_id = trace2.trace_id

        self.service._trace_manager = MagicMock()
        self.service._trace_manager.traces = {
            trace1.trace_id: mock_trace1,
            trace2.trace_id: mock_trace2,
        }

        # Get all traces
        traces = self.service.get_traces()

        # Verify traces
        assert len(traces) == 2
        assert traces[0].trace_id == trace1.trace_id
        assert traces[1].trace_id == trace2.trace_id

    def test_create_blame_node(self) -> None:
        """Test creating a blame node."""
        # Mock the blame graph
        mock_node = MagicMock()
        mock_node.name = "test_node"
        mock_node.component = "test_type"
        mock_node.attributes = {"node_attr": "node_value"}

        self.service.blame_graph = MagicMock()
        self.service.blame_graph.nodes = {}
        self.service.blame_graph.has_node.return_value = True
        self.service.blame_graph.get_node.return_value = mock_node

        # Create a blame node
        node_id = self.service.create_blame_node(
            name="test_node", node_type="test_type", attributes={"node_attr": "node_value"}
        )

        # Verify node
        assert node_id is not None
        assert node_id == "test_type.test_node"

    def test_add_blame_edge(self) -> None:
        """Test adding a blame edge."""
        # Mock the blame graph
        self.service.blame_graph = MagicMock()
        self.service.blame_graph.edges = {}

        # Define source and target IDs
        source_id = "source_type.source"
        target_id = "target_type.target"

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
        assert edge_id == f"edge-{source_id}-{target_id}-test_edge"

    def test_export_trace(self) -> None:
        """Test exporting a trace."""
        # Create a fixed trace ID for testing
        test_trace_id = "test-trace-id"

        # Mock the trace manager
        self.service._trace_manager = MagicMock()
        self.service._trace_manager.get_trace.return_value = MagicMock(trace_id=test_trace_id)

        # Export the trace
        with patch("builtins.open", MagicMock()) as mock_open:
            self.service.export_trace(test_trace_id, "test_trace.json")
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
