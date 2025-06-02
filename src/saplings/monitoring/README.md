# Saplings Monitoring

This package provides monitoring capabilities for Saplings agents.

## API Structure

The monitoring module follows the Saplings API separation pattern:

1. **Public API**: Exposed through `saplings.api.monitoring` and re-exported at the top level of the `saplings` package
2. **Internal Implementation**: Located in the `_internal` directory

## Usage

To use the monitoring components, import them from the public API:

```python
# Recommended: Import from the top-level package
from saplings import (
    TraceManager,
    Span,
    BlameGraph,
    GASAHeatmap
)

# Alternative: Import directly from the API module
from saplings.api.monitoring import (
    TraceManager,
    Span,
    BlameGraph,
    GASAHeatmap
)
```

Do not import directly from the internal implementation:

```python
# Don't do this
from saplings.monitoring._internal import TraceManager  # Wrong
```

## Available Components

The following monitoring components are available:

- `TraceManager`: Manager for creating and managing traces
- `Span`: Span representing a unit of work in a trace
- `SpanContext`: Context for a span in a trace
- `BlameGraph`: Graph for identifying bottlenecks in agent execution
- `BlameNode`: Node in a blame graph representing a component
- `BlameEdge`: Edge in a blame graph connecting two nodes
- `GASAHeatmap`: Visualization of GASA attention patterns
- `PerformanceVisualizer`: Visualizer for agent performance metrics
- `TraceViewer`: Interface for exploring traces
- `LangSmithExporter`: Exporter for sending traces to LangSmith
- `MonitoringConfig`: Configuration for monitoring components

## Tracing

Saplings provides a comprehensive tracing infrastructure based on OpenTelemetry:

```python
from saplings import TraceManager, Span

# Create a trace manager
trace_manager = TraceManager()

# Create a root span
with trace_manager.start_span("agent_execution") as root_span:
    # Create a child span
    with trace_manager.start_span("retrieval", parent=root_span) as retrieval_span:
        # Record an event
        retrieval_span.record_event("retrieved_documents", {"count": 10})

        # Set attributes
        retrieval_span.set_attribute("query", "What is the capital of France?")

    # Create another child span
    with trace_manager.start_span("generation", parent=root_span) as generation_span:
        # Record an event
        generation_span.record_event("generated_response", {"tokens": 100})
```

## Blame Graph

The blame graph is a directed graph that helps identify bottlenecks in agent execution:

```python
from saplings import BlameGraph, BlameNode, BlameEdge

# Create a blame graph
graph = BlameGraph()

# Create nodes
retrieval_node = BlameNode("retrieval", "component", {"time": 1.5})
generation_node = BlameNode("generation", "component", {"time": 3.2})

# Add nodes to the graph
graph.add_node(retrieval_node)
graph.add_node(generation_node)

# Add an edge
graph.add_edge(BlameEdge(retrieval_node, generation_node, 0.8))

# Find the bottleneck
bottleneck = graph.find_bottleneck()
```

## Visualization

Saplings provides visualization tools for monitoring agent performance:

```python
from saplings import GASAHeatmap, PerformanceVisualizer

# Create a GASA heatmap
heatmap = GASAHeatmap()
heatmap.generate("path/to/output.png", attention_matrix)

# Create a performance visualizer
visualizer = PerformanceVisualizer()
visualizer.generate_timeline("path/to/output.png", spans)
```

## Implementation Details

The monitoring implementations are located in the `_internal` directory:

- `_internal/trace.py`: Implementation of the tracing infrastructure
- `_internal/blame_graph.py`: Implementation of the blame graph
- `_internal/visualization.py`: Implementation of the visualization tools
- `_internal/trace_viewer.py`: Implementation of the trace viewer
- `_internal/langsmith.py`: Implementation of the LangSmith exporter
- `_internal/config.py`: Monitoring configuration

These internal implementations are wrapped by the public API in `saplings.api.monitoring` to provide stability annotations and a consistent interface.
