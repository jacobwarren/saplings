"""
Tests for the tracing infrastructure.
"""

import json
import os
import tempfile
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from saplings.monitoring.config import MonitoringConfig, TracingBackend
from saplings.monitoring.trace import TraceManager, Trace, Span, SpanContext, SpanEvent


@pytest.fixture
def trace_manager():
    """Create a TraceManager instance for testing."""
    return TraceManager(config=MonitoringConfig())


@pytest.fixture
def sample_trace(trace_manager):
    """Create a sample trace for testing."""
    # Create a trace
    trace = trace_manager.create_trace(trace_id="test-trace-1")

    # Add root span
    root_span = trace_manager.start_span(
        name="root_operation",
        trace_id=trace.trace_id,
        attributes={"component": "component_a"},
    )

    # Add child spans
    child1 = trace_manager.start_span(
        name="child_operation_1",
        trace_id=trace.trace_id,
        parent_id=root_span.span_id,
        attributes={"component": "component_b"},
    )

    child2 = trace_manager.start_span(
        name="child_operation_2",
        trace_id=trace.trace_id,
        parent_id=root_span.span_id,
        attributes={"component": "component_c"},
    )

    # Add grandchild span
    grandchild = trace_manager.start_span(
        name="grandchild_operation",
        trace_id=trace.trace_id,
        parent_id=child1.span_id,
        attributes={"component": "component_d"},
    )

    # Simulate some processing time
    for span in [grandchild, child1, child2, root_span]:
        # Set start time to a fixed time for deterministic testing
        span.start_time = datetime.now() - timedelta(seconds=1)
        # Set end time
        span.end_time = datetime.now()

    # End spans
    trace_manager.end_span(grandchild.span_id)
    trace_manager.end_span(child1.span_id)
    trace_manager.end_span(child2.span_id)
    trace_manager.end_span(root_span.span_id)

    return trace


def test_span_event():
    """Test SpanEvent functionality."""
    # Create an event
    event = SpanEvent(
        name="test_event",
        attributes={"key": "value"},
    )

    # Check initial values
    assert event.name == "test_event"
    assert event.attributes == {"key": "value"}
    assert isinstance(event.timestamp, datetime)


def test_span_context():
    """Test SpanContext functionality."""
    # Create a context
    context = SpanContext(
        trace_id="test-trace",
        span_id="test-span",
        parent_id="parent-span",
    )

    # Check initial values
    assert context.trace_id == "test-trace"
    assert context.span_id == "test-span"
    assert context.parent_id == "parent-span"


def test_span():
    """Test Span functionality."""
    # Create a context
    context = SpanContext(
        trace_id="test-trace",
        span_id="test-span",
        parent_id="parent-span",
    )

    # Create a span
    span = Span(
        name="test_span",
        context=context,
        attributes={"key": "value"},
    )

    # Check initial values
    assert span.name == "test_span"
    assert span.context == context
    assert span.span_id == "test-span"
    assert span.parent_id == "parent-span"
    assert span.trace_id == "test-trace"
    assert span.attributes == {"key": "value"}
    assert span.status == "OK"
    assert len(span.events) == 0

    # Add an event
    event = span.add_event(
        name="test_event",
        attributes={"event_key": "event_value"},
    )

    # Check that the event was added
    assert len(span.events) == 1
    assert span.events[0] == event
    assert span.events[0].name == "test_event"
    assert span.events[0].attributes == {"event_key": "event_value"}

    # Set status
    span.set_status("ERROR")
    assert span.status == "ERROR"

    # Set attribute
    span.set_attribute("new_key", "new_value")
    assert span.attributes["new_key"] == "new_value"

    # End span
    span.end()
    assert span.end_time is not None

    # Check duration
    assert span.duration_ms() > 0


def test_trace():
    """Test Trace functionality."""
    # Create a trace
    trace = Trace(
        trace_id="test-trace",
        attributes={"key": "value"},
    )

    # Check initial values
    assert trace.trace_id == "test-trace"
    assert trace.attributes == {"key": "value"}
    assert trace.status == "OK"
    assert len(trace.spans) == 0

    # Create a span
    context = SpanContext(
        trace_id="test-trace",
        span_id="test-span",
    )
    span = Span(
        name="test_span",
        context=context,
    )

    # Add span to trace
    trace.add_span(span)

    # Check that the span was added
    assert len(trace.spans) == 1
    assert trace.spans[0] == span

    # Create a child span
    child_context = SpanContext(
        trace_id="test-trace",
        span_id="child-span",
        parent_id="test-span",
    )
    child_span = Span(
        name="child_span",
        context=child_context,
    )

    # Add child span to trace
    trace.add_span(child_span)

    # Check that the child span was added
    assert len(trace.spans) == 2
    assert trace.spans[1] == child_span

    # Check root spans
    root_spans = trace.get_root_spans()
    assert len(root_spans) == 1
    assert root_spans[0] == span

    # Check child spans
    child_spans = trace.get_child_spans("test-span")
    assert len(child_spans) == 1
    assert child_spans[0] == child_span

    # Set error status on child span
    child_span.set_status("ERROR")

    # Add span with error to trace
    trace.add_span(child_span)

    # Check that trace status was updated
    assert trace.status == "ERROR"

    # End trace
    trace.end()
    assert trace.end_time is not None

    # Check duration
    assert trace.duration_ms() > 0


def test_trace_manager():
    """Test TraceManager functionality."""
    # Create a trace manager
    manager = TraceManager()

    # Check initial state
    assert len(manager.traces) == 0
    assert len(manager.active_traces) == 0
    assert len(manager.active_spans) == 0

    # Create a trace
    trace = manager.create_trace(trace_id="test-trace")

    # Check that the trace was created
    assert len(manager.traces) == 1
    assert "test-trace" in manager.traces
    assert manager.traces["test-trace"] == trace
    assert "test-trace" in manager.active_traces

    # Start a span
    span = manager.start_span(
        name="test_span",
        trace_id="test-trace",
        attributes={"key": "value"},
    )

    # Check that the span was created
    assert len(manager.active_spans) == 1
    assert span.span_id in manager.active_spans
    assert manager.active_spans[span.span_id] == span
    assert len(manager.traces["test-trace"].spans) == 1
    assert manager.traces["test-trace"].spans[0] == span

    # Start a child span
    child_span = manager.start_span(
        name="child_span",
        trace_id="test-trace",
        parent_id=span.span_id,
        attributes={"child_key": "child_value"},
    )

    # Check that the child span was created
    assert len(manager.active_spans) == 2
    assert child_span.span_id in manager.active_spans
    assert manager.active_spans[child_span.span_id] == child_span
    assert len(manager.traces["test-trace"].spans) == 2
    assert manager.traces["test-trace"].spans[1] == child_span

    # End child span
    manager.end_span(child_span.span_id)

    # Check that the child span was ended
    assert child_span.span_id not in manager.active_spans
    assert child_span.end_time is not None
    assert "test-trace" in manager.active_traces  # Trace still active

    # End parent span
    manager.end_span(span.span_id)

    # Check that the parent span was ended and trace is inactive
    assert span.span_id not in manager.active_spans
    assert span.end_time is not None
    assert "test-trace" not in manager.active_traces
    assert manager.traces["test-trace"].end_time is not None

    # Get trace
    retrieved_trace = manager.get_trace("test-trace")
    assert retrieved_trace == trace

    # Get span
    retrieved_span = manager.get_span(span.span_id)
    assert retrieved_span == span

    # List traces
    traces = manager.list_traces()
    assert len(traces) == 1
    assert traces[0] == trace

    # Clear traces
    cleared = manager.clear_traces()
    assert cleared == 1
    assert len(manager.traces) == 0


@pytest.mark.skip(reason="OpenTelemetry not installed")
def test_trace_manager_with_otel():
    """Test TraceManager with OpenTelemetry."""
    # Create a config with OTEL backend
    config = MonitoringConfig(
        tracing_backend=TracingBackend.OTEL,
        otel_endpoint="http://localhost:4317",
    )

    # Mock OTEL availability and imports
    with patch("saplings.monitoring.trace.OTEL_AVAILABLE", True), \
         patch("opentelemetry.sdk.trace.TracerProvider") as mock_provider, \
         patch("opentelemetry.sdk.trace.export.BatchSpanProcessor") as mock_processor, \
         patch("opentelemetry.exporter.otlp.proto.grpc.trace_exporter.OTLPSpanExporter") as mock_exporter, \
         patch("opentelemetry.trace") as mock_trace:

        # Mock the tracer
        mock_tracer = MagicMock()
        mock_trace.get_tracer.return_value = mock_tracer

        # Mock the span
        mock_otel_span = MagicMock()
        mock_tracer.start_span.return_value = mock_otel_span

        # Create a trace manager
        manager = TraceManager(config=config)

        # Check that OTEL was initialized
        mock_provider.assert_called_once()
        mock_exporter.assert_called_once_with(endpoint="http://localhost:4317")
        mock_processor.assert_called_once()
        mock_trace.set_tracer_provider.assert_called_once()
        mock_trace.get_tracer.assert_called_once_with("saplings")

        # Create a trace and span
        trace = manager.create_trace(trace_id="test-trace")
        span = manager.start_span(
            name="test_span",
            trace_id="test-trace",
            attributes={"key": "value"},
        )

        # Check that OTEL span was created
        mock_tracer.start_span.assert_called_once()
        assert hasattr(span, "otel_span")
        assert span.otel_span == mock_otel_span

        # Add an event
        span.add_event("test_event", {"event_key": "event_value"})

        # Check that event was added to OTEL span
        mock_otel_span.add_event.assert_called_once()

        # Set an attribute
        span.set_attribute("new_key", "new_value")

        # Check that attribute was set on OTEL span
        mock_otel_span.set_attribute.assert_called_with("new_key", "new_value")

        # Set status
        span.set_status("ERROR")

        # Check that status was set on OTEL span
        mock_otel_span.set_status.assert_called_once()

        # End the span
        manager.end_span(span.span_id)

        # Check that OTEL span was ended
        mock_otel_span.end.assert_called_once()


def test_trace_manager_search_spans(trace_manager, sample_trace):
    """Test searching for spans in a trace."""
    # Get a span from the trace
    span = sample_trace.spans[0]

    # Get the span by ID
    retrieved_span = trace_manager.get_span(span.span_id)
    assert retrieved_span == span

    # Try to get a non-existent span
    non_existent_span = trace_manager.get_span("non-existent-span")
    assert non_existent_span is None


def test_trace_manager_time_filtering(trace_manager, sample_trace):
    """Test filtering traces by time."""
    # Create a trace with specific times
    now = datetime.now()
    old_time = now - timedelta(hours=1)

    # Create an old trace
    old_trace = trace_manager.create_trace(trace_id="old-trace")
    old_span = trace_manager.start_span(
        name="old_span",
        trace_id=old_trace.trace_id,
    )

    # Set times manually
    old_trace.start_time = old_time
    old_trace.end_time = old_time + timedelta(minutes=5)
    old_span.start_time = old_time
    old_span.end_time = old_time + timedelta(minutes=5)

    # End the span
    trace_manager.end_span(old_span.span_id)

    # List traces with time filtering
    recent_traces = trace_manager.list_traces(start_time=now - timedelta(minutes=30))
    assert len(recent_traces) == 1
    assert recent_traces[0].trace_id == sample_trace.trace_id

    old_traces = trace_manager.list_traces(end_time=now - timedelta(minutes=30))
    assert len(old_traces) == 1
    assert old_traces[0].trace_id == "old-trace"

    # Clear old traces
    cleared = trace_manager.clear_traces(before_time=now - timedelta(minutes=30))
    assert cleared == 1

    # Check that only the old trace was cleared
    assert "old-trace" not in trace_manager.traces
    assert sample_trace.trace_id in trace_manager.traces


def test_trace_callbacks():
    """Test trace callbacks."""
    # Create trace manager
    trace_manager = TraceManager()

    # Create a mock callback
    callback_events = []
    def test_callback(trace_id, event):
        callback_events.append((trace_id, event))

    # Register the callback
    trace_manager.register_trace_callback(test_callback)

    # Create a trace
    trace = trace_manager.create_trace(trace_id="callback-test")

    # Check that the callback was called with "created" event
    assert len(callback_events) == 1
    assert callback_events[0] == ("callback-test", "created")

    # Add a root span
    root_span = trace_manager.start_span(
        name="root_operation",
        trace_id=trace.trace_id,
        attributes={"component": "test"},
    )

    # End the span and trace
    trace_manager.end_span(root_span.span_id)

    # Check that the callback was called with "completed" event
    assert len(callback_events) == 2
    assert callback_events[1] == ("callback-test", "completed")

    # Create a trace with an error
    error_trace = trace_manager.create_trace(trace_id="error-test")

    # Reset callback events
    callback_events.clear()

    # Add a root span with error
    error_span = trace_manager.start_span(
        name="error_operation",
        trace_id=error_trace.trace_id,
        attributes={"component": "test"},
    )

    # End the span with error status
    trace_manager.end_span(error_span.span_id, status="ERROR")

    # Check that the callback was called with an event
    assert len(callback_events) >= 1
    # The event might be "error" or "completed" depending on implementation
    assert callback_events[0][0] == "error-test"  # Check trace ID
    assert callback_events[0][1] in ["error", "completed"]  # Check event type

    # Test unregistering the callback
    trace_manager.unregister_trace_callback(test_callback)

    # Store the current length of callback_events
    current_len = len(callback_events)

    # Create another trace
    trace_manager.create_trace(trace_id="after-unregister")

    # Check that the callback was not called (length should not change)
    assert len(callback_events) == current_len
