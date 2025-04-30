"""
Tests for the TraceViewer component.
"""

import json
import os
import tempfile
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from saplings.monitoring.config import MonitoringConfig, VisualizationFormat
from saplings.monitoring.trace import TraceManager, Trace, Span, SpanContext
from saplings.monitoring.trace_viewer import TraceViewer


# Check if plotly is available
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


@pytest.fixture
def trace_manager():
    """Create a TraceManager instance for testing."""
    return TraceManager(config=MonitoringConfig())


@pytest.fixture
def trace_viewer(trace_manager):
    """Create a TraceViewer instance for testing."""
    config = MonitoringConfig(
        visualization_output_dir=tempfile.mkdtemp(),
        visualization_format=VisualizationFormat.HTML,
    )
    return TraceViewer(trace_manager=trace_manager, config=config)


@pytest.fixture
def sample_trace(trace_manager):
    """Create a sample trace for testing."""
    # Create a trace
    trace = trace_manager.create_trace(trace_id="test-trace-1")

    # Add root span
    root_span = trace_manager.start_span(
        name="root_operation",
        trace_id=trace.trace_id,
        attributes={"component": "test_component"},
    )

    # Add child spans
    child1 = trace_manager.start_span(
        name="child_operation_1",
        trace_id=trace.trace_id,
        parent_id=root_span.span_id,
        attributes={"component": "test_component"},
    )

    child2 = trace_manager.start_span(
        name="child_operation_2",
        trace_id=trace.trace_id,
        parent_id=root_span.span_id,
        attributes={"component": "test_component"},
    )

    # Add grandchild span
    grandchild = trace_manager.start_span(
        name="grandchild_operation",
        trace_id=trace.trace_id,
        parent_id=child1.span_id,
        attributes={"component": "test_component"},
    )

    # End spans
    trace_manager.end_span(grandchild.span_id)
    trace_manager.end_span(child1.span_id)
    trace_manager.end_span(child2.span_id)
    trace_manager.end_span(root_span.span_id)

    return trace


@pytest.fixture
def complex_trace(trace_manager):
    """Create a more complex trace for testing."""
    # Create a trace
    trace = trace_manager.create_trace(trace_id="complex-trace")

    # Add root span for the agent
    agent_span = trace_manager.start_span(
        name="agent_execution",
        trace_id=trace.trace_id,
        attributes={
            "component": "agent",
            "agent_type": "reasoning",
            "user_query": "What is the capital of France?"
        },
    )

    # Add planning span
    planning_span = trace_manager.start_span(
        name="planning",
        trace_id=trace.trace_id,
        parent_id=agent_span.span_id,
        attributes={
            "component": "planner",
            "plan_type": "sequential",
            "steps_count": 3
        },
    )

    # Add LLM call span for planning
    planning_llm_span = trace_manager.start_span(
        name="llm_call",
        trace_id=trace.trace_id,
        parent_id=planning_span.span_id,
        attributes={
            "component": "llm",
            "model": "gpt-4",
            "prompt": "Create a plan to answer: What is the capital of France?",
            "completion": "1. Search for 'capital of France'\n2. Process results\n3. Format answer",
            "tokens": 25,
            "temperature": 0.7
        },
    )

    # End planning LLM span
    trace_manager.end_span(planning_llm_span.span_id)

    # End planning span
    trace_manager.end_span(planning_span.span_id)

    # Add execution span
    execution_span = trace_manager.start_span(
        name="execution",
        trace_id=trace.trace_id,
        parent_id=agent_span.span_id,
        attributes={
            "component": "executor",
            "execution_mode": "sequential"
        },
    )

    # Add tool call span
    tool_span = trace_manager.start_span(
        name="tool_call",
        trace_id=trace.trace_id,
        parent_id=execution_span.span_id,
        attributes={
            "component": "tool",
            "tool_name": "search",
            "input": "capital of France",
            "output": "Paris is the capital of France."
        },
    )

    # End tool span
    trace_manager.end_span(tool_span.span_id)

    # Add LLM call span for answer generation
    answer_llm_span = trace_manager.start_span(
        name="llm_call",
        trace_id=trace.trace_id,
        parent_id=execution_span.span_id,
        attributes={
            "component": "llm",
            "model": "gpt-4",
            "prompt": "Format the answer based on search results: Paris is the capital of France.",
            "completion": "The capital of France is Paris.",
            "tokens": 15,
            "temperature": 0.7
        },
    )

    # End answer LLM span
    trace_manager.end_span(answer_llm_span.span_id)

    # End execution span
    trace_manager.end_span(execution_span.span_id)

    # End agent span
    trace_manager.end_span(agent_span.span_id)

    return trace


@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not installed")
def test_view_trace(trace_viewer, sample_trace):
    """Test viewing a trace."""
    with tempfile.NamedTemporaryFile(suffix=".html") as tmp:
        # View trace
        fig = trace_viewer.view_trace(
            trace_id=sample_trace.trace_id,
            output_path=tmp.name,
            show=False,
        )

        # Check that the file was created
        assert os.path.exists(tmp.name)

        # Check that a figure was returned
        assert fig is not None


@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not installed")
def test_view_complex_trace(trace_viewer, complex_trace):
    """Test viewing a complex trace."""
    with tempfile.NamedTemporaryFile(suffix=".html") as tmp:
        # View trace
        fig = trace_viewer.view_trace(
            trace_id=complex_trace.trace_id,
            output_path=tmp.name,
            show=False,
        )

        # Check that the file was created
        assert os.path.exists(tmp.name)

        # Check that a figure was returned
        assert fig is not None


@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not installed")
def test_view_span(trace_viewer, sample_trace):
    """Test viewing a specific span."""
    # Get a span ID
    span_id = sample_trace.spans[0].span_id

    with tempfile.NamedTemporaryFile(suffix=".html") as tmp:
        # View span
        fig = trace_viewer.view_span(
            trace_id=sample_trace.trace_id,
            span_id=span_id,
            output_path=tmp.name,
            show=False,
        )

        # Check that the file was created
        assert os.path.exists(tmp.name)

        # Check that a figure was returned
        assert fig is not None


def test_export_trace(trace_viewer, sample_trace):
    """Test exporting a trace to a file."""
    with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
        # Export trace
        success = trace_viewer.export_trace(
            trace_id=sample_trace.trace_id,
            output_path=tmp.name,
            format=VisualizationFormat.JSON,
        )

        # Check that the export was successful
        assert success

        # Check that the file was created
        assert os.path.exists(tmp.name)

        # Check that the file contains valid JSON
        with open(tmp.name, "r") as f:
            data = json.load(f)

        # Check that the trace ID is correct
        assert data["trace_id"] == sample_trace.trace_id

        # Check that all spans are included
        assert len(data["spans"]) == len(sample_trace.spans)


def test_search_traces(trace_viewer, sample_trace, complex_trace):
    """Test searching for traces."""
    # Search for the trace by ID
    results = trace_viewer.search_traces(query=sample_trace.trace_id)

    # Check that the trace was found
    assert len(results) == 1
    assert results[0]["trace_id"] == sample_trace.trace_id

    # Search for a span name
    results = trace_viewer.search_traces(query="child_operation")

    # Check that the trace was found
    assert len(results) == 1
    assert results[0]["trace_id"] == sample_trace.trace_id

    # Search for a component in the complex trace
    results = trace_viewer.search_traces(query="llm")

    # Check that the complex trace was found
    assert len(results) == 1
    assert results[0]["trace_id"] == complex_trace.trace_id

    # Search for a non-existent trace
    results = trace_viewer.search_traces(query="non_existent_trace")

    # Check that no traces were found
    assert len(results) == 0


def test_filter_traces_by_component(trace_viewer, sample_trace, complex_trace):
    """Test filtering traces by component."""
    # Filter traces by the 'llm' component
    results = trace_viewer.filter_traces_by_component(component="llm")

    # Check that only the complex trace was found (it has LLM spans)
    assert len(results) == 1
    assert results[0]["trace_id"] == complex_trace.trace_id

    # Filter traces by the 'test_component' component
    results = trace_viewer.filter_traces_by_component(component="test_component")

    # Check that only the sample trace was found
    assert len(results) == 1
    assert results[0]["trace_id"] == sample_trace.trace_id

    # Filter traces by a non-existent component
    results = trace_viewer.filter_traces_by_component(component="non_existent_component")

    # Check that no traces were found
    assert len(results) == 0


def test_trace_to_dict(trace_viewer, sample_trace):
    """Test converting a trace to a dictionary."""
    # Convert trace to dictionary
    trace_dict = trace_viewer._trace_to_dict(sample_trace)

    # Check that the trace ID is correct
    assert trace_dict["trace_id"] == sample_trace.trace_id

    # Check that all spans are included
    assert len(trace_dict["spans"]) == len(sample_trace.spans)

    # Check that span IDs are correct
    span_ids = [span["span_id"] for span in trace_dict["spans"]]
    for span in sample_trace.spans:
        assert span.span_id in span_ids
