"""
Tests for the LangSmith exporter.
"""

import json
import os
import tempfile
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from saplings.monitoring.config import MonitoringConfig
from saplings.monitoring.langsmith import LangSmithExporter
from saplings.monitoring.trace import Span, SpanContext, Trace, TraceManager

# Check if langsmith is available
try:
    import langsmith

    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False


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


@pytest.fixture
def llm_trace(trace_manager):
    """Create a trace with LLM operations for testing."""
    # Create a trace
    trace = trace_manager.create_trace(trace_id="llm-trace")

    # Add root span for the agent
    agent_span = trace_manager.start_span(
        name="agent_execution",
        trace_id=trace.trace_id,
        attributes={
            "component": "agent",
            "agent_type": "reasoning",
            "user_query": "What is the capital of France?",
        },
    )

    # Add LLM call span
    llm_span = trace_manager.start_span(
        name="llm_call",
        trace_id=trace.trace_id,
        parent_id=agent_span.span_id,
        attributes={
            "component": "llm",
            "model": "gpt-4",
            "prompt": "Answer the following question: What is the capital of France?",
            "completion": "The capital of France is Paris.",
            "tokens": 15,
            "temperature": 0.7,
        },
    )

    # Add tool call span
    tool_span = trace_manager.start_span(
        name="tool_call",
        trace_id=trace.trace_id,
        parent_id=agent_span.span_id,
        attributes={
            "component": "tool",
            "tool_name": "search",
            "input": "capital of France",
            "output": "Paris is the capital of France.",
        },
    )

    # Simulate some processing time
    for span in [tool_span, llm_span, agent_span]:
        # Set start time to a fixed time for deterministic testing
        span.start_time = datetime.now() - timedelta(seconds=1)
        # Set end time
        span.end_time = datetime.now()

    # End spans
    trace_manager.end_span(tool_span.span_id)
    trace_manager.end_span(llm_span.span_id)
    trace_manager.end_span(agent_span.span_id)

    return trace


@pytest.fixture
def mock_langsmith_client():
    """Create a mock LangSmith client."""
    mock_client = MagicMock()
    mock_client.create_run.return_value = "mock-run-id"
    return mock_client


@pytest.mark.skipif(not LANGSMITH_AVAILABLE, reason="LangSmith not installed")
def test_langsmith_exporter_initialization():
    """Test LangSmithExporter initialization."""
    # Create exporter with config
    config = MonitoringConfig(
        langsmith_api_key="test-api-key",
        langsmith_project="test-project",
    )

    with patch("langsmith.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client.list_projects.return_value = []
        mock_client_class.return_value = mock_client

        exporter = LangSmithExporter(config=config)

        # Check that client was initialized
        mock_client_class.assert_called_once_with(api_key="test-api-key")
        assert exporter.client is not None
        assert exporter.api_key == "test-api-key"
        assert exporter.project_name == "test-project"
        assert exporter.auto_export is False

        # Check that project creation was attempted
        mock_client.list_projects.assert_called_once()
        mock_client.create_project.assert_called_once_with("test-project")


@pytest.mark.skipif(not LANGSMITH_AVAILABLE, reason="LangSmith not installed")
def test_export_trace(trace_manager, sample_trace, mock_langsmith_client):
    """Test exporting a trace to LangSmith."""
    # Create exporter with mock client
    exporter = LangSmithExporter(
        trace_manager=trace_manager,
        config=MonitoringConfig(),
        api_key="test-api-key",
        project_name="test-project",
    )

    # Replace client with mock
    exporter.client = mock_langsmith_client

    # Export trace
    run_id = exporter.export_trace(sample_trace)

    # Check that client was called
    mock_langsmith_client.create_run.assert_called_once()

    # Check that run ID was returned
    assert run_id == "mock-run-id"


@pytest.mark.skipif(not LANGSMITH_AVAILABLE, reason="LangSmith not installed")
def test_export_trace_by_id(trace_manager, sample_trace, mock_langsmith_client):
    """Test exporting a trace to LangSmith by ID."""
    # Create exporter with mock client
    exporter = LangSmithExporter(
        trace_manager=trace_manager,
        config=MonitoringConfig(),
        api_key="test-api-key",
        project_name="test-project",
    )

    # Replace client with mock
    exporter.client = mock_langsmith_client

    # Export trace by ID
    run_id = exporter.export_trace(sample_trace.trace_id)

    # Check that client was called
    mock_langsmith_client.create_run.assert_called_once()

    # Check that run ID was returned
    assert run_id == "mock-run-id"


@pytest.mark.skipif(not LANGSMITH_AVAILABLE, reason="LangSmith not installed")
def test_export_traces(trace_manager, sample_trace, mock_langsmith_client):
    """Test exporting multiple traces to LangSmith."""
    # Create a second trace
    trace2 = trace_manager.create_trace(trace_id="test-trace-2")
    root_span = trace_manager.start_span(
        name="root_operation",
        trace_id=trace2.trace_id,
        attributes={"component": "component_a"},
    )
    trace_manager.end_span(root_span.span_id)

    # Create exporter with mock client
    exporter = LangSmithExporter(
        trace_manager=trace_manager,
        config=MonitoringConfig(),
        api_key="test-api-key",
        project_name="test-project",
    )

    # Replace client with mock
    exporter.client = mock_langsmith_client

    # Export traces
    run_ids = exporter.export_traces()

    # Check that client was called twice
    assert mock_langsmith_client.create_run.call_count == 2

    # Check that run IDs were returned
    assert len(run_ids) == 2
    assert run_ids == ["mock-run-id", "mock-run-id"]


@pytest.mark.skipif(not LANGSMITH_AVAILABLE, reason="LangSmith not installed")
def test_export_traces_with_ids(trace_manager, sample_trace, mock_langsmith_client):
    """Test exporting specific traces to LangSmith."""
    # Create a second trace
    trace2 = trace_manager.create_trace(trace_id="test-trace-2")
    root_span = trace_manager.start_span(
        name="root_operation",
        trace_id=trace2.trace_id,
        attributes={"component": "component_a"},
    )
    trace_manager.end_span(root_span.span_id)

    # Create exporter with mock client
    exporter = LangSmithExporter(
        trace_manager=trace_manager,
        config=MonitoringConfig(),
        api_key="test-api-key",
        project_name="test-project",
    )

    # Replace client with mock
    exporter.client = mock_langsmith_client

    # Export specific trace
    run_ids = exporter.export_traces(trace_ids=[sample_trace.trace_id])

    # Check that client was called once
    assert mock_langsmith_client.create_run.call_count == 1

    # Check that run ID was returned
    assert len(run_ids) == 1
    assert run_ids == ["mock-run-id"]


@pytest.mark.skipif(not LANGSMITH_AVAILABLE, reason="LangSmith not installed")
def test_convert_trace_to_langsmith(trace_manager, sample_trace):
    """Test converting a trace to LangSmith format."""
    # Create exporter
    exporter = LangSmithExporter(
        trace_manager=trace_manager,
        config=MonitoringConfig(),
        api_key="test-api-key",
        project_name="test-project",
    )

    # Convert trace
    runs = exporter._convert_trace_to_langsmith(sample_trace)

    # Check that runs were created
    assert len(runs) == 4

    # Check that each span has a corresponding run
    span_ids = [span.span_id for span in sample_trace.spans]
    run_ids = [run["id"] for run in runs]
    assert set(span_ids) == set(run_ids)

    # Check run structure
    for run in runs:
        assert "id" in run
        assert "name" in run
        assert "run_type" in run
        assert "inputs" in run
        assert "outputs" in run
        assert "start_time" in run
        assert "end_time" in run
        assert "extra" in run
        assert "trace_id" in run["extra"]


@pytest.mark.skipif(not LANGSMITH_AVAILABLE, reason="LangSmith not installed")
def test_convert_llm_trace_to_langsmith(trace_manager, llm_trace):
    """Test converting an LLM trace to LangSmith format."""
    # Create exporter
    exporter = LangSmithExporter(
        trace_manager=trace_manager,
        config=MonitoringConfig(),
        api_key="test-api-key",
        project_name="test-project",
    )

    # Convert trace
    runs = exporter._convert_trace_to_langsmith(llm_trace)

    # Check that runs were created
    assert len(runs) == 3  # agent, llm, tool

    # Check that each span has a corresponding run
    span_ids = [span.span_id for span in llm_trace.spans]
    run_ids = [run["id"] for run in runs]
    assert set(span_ids) == set(run_ids)

    # Find the LLM run
    llm_run = None
    for run in runs:
        if run["name"] == "llm_call":
            llm_run = run
            break

    assert llm_run is not None, "LLM run not found"

    # Check LLM run structure
    assert llm_run["run_type"] == "llm"
    assert "prompt" in llm_run["inputs"]
    assert "completion" in llm_run["outputs"]
    assert (
        llm_run["inputs"]["prompt"]
        == "Answer the following question: What is the capital of France?"
    )
    assert llm_run["outputs"]["completion"] == "The capital of France is Paris."

    # Find the tool run
    tool_run = None
    for run in runs:
        if run["name"] == "tool_call":
            tool_run = run
            break

    assert tool_run is not None, "Tool run not found"

    # Check tool run structure
    assert tool_run["run_type"] == "tool"
    assert "input" in tool_run["inputs"]
    assert "output" in tool_run["outputs"]
    assert tool_run["inputs"]["input"] == "capital of France"
    assert tool_run["outputs"]["output"] == "Paris is the capital of France."


@pytest.mark.skipif(not LANGSMITH_AVAILABLE, reason="LangSmith not installed")
def test_auto_export(trace_manager, mock_langsmith_client):
    """Test automatic export of traces."""
    # Create exporter with auto-export enabled
    exporter = LangSmithExporter(
        trace_manager=trace_manager,
        config=MonitoringConfig(),
        api_key="test-api-key",
        project_name="test-project",
        auto_export=True,
    )

    # Replace client with mock
    exporter.client = mock_langsmith_client

    # Create a trace
    trace = trace_manager.create_trace(trace_id="auto-export-test")

    # Add a root span
    root_span = trace_manager.start_span(
        name="root_operation",
        trace_id=trace.trace_id,
        attributes={"component": "agent"},
    )

    # End the span and trace
    trace_manager.end_span(root_span.span_id)

    # Check that the callback was triggered and the trace was exported
    assert mock_langsmith_client.create_run.call_count == 1

    # Check that the trace ID was added to exported traces
    assert trace.trace_id in exporter.exported_trace_ids


@pytest.mark.skipif(not LANGSMITH_AVAILABLE, reason="LangSmith not installed")
def test_check_connection(mock_langsmith_client):
    """Test checking the LangSmith connection."""
    # Create exporter
    exporter = LangSmithExporter(
        config=MonitoringConfig(),
        api_key="test-api-key",
        project_name="test-project",
    )

    # Replace client with mock
    exporter.client = mock_langsmith_client

    # Check connection
    assert exporter.check_connection() is True
    mock_langsmith_client.list_projects.assert_called_once()

    # Test failed connection
    mock_langsmith_client.list_projects.side_effect = Exception("Connection error")
    assert exporter.check_connection() is False
