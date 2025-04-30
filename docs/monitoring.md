# Monitoring in Saplings

The Saplings monitoring module provides comprehensive monitoring capabilities for tracking, analyzing, and visualizing the performance and behavior of your agents. This includes tracing infrastructure, causal blame graph for identifying bottlenecks, GASA heatmap visualization, and integration with external tools like LangSmith.

## Overview

The monitoring system in Saplings consists of several components:

1. **Tracing Infrastructure**: OpenTelemetry (OTEL) integration for distributed tracing
2. **Causal Blame Graph**: Identifies performance bottlenecks and error sources
3. **GASA Heatmap**: Visualizes Graph-Aligned Sparse Attention masks
4. **TraceViewer**: Interactive interface for exploring traces
5. **LangSmith Export**: Integration with LangSmith for advanced analysis

## Getting Started

### Basic Setup

```python
from saplings.monitoring import MonitoringConfig, TraceManager, BlameGraph, TraceViewer

# Create a monitoring configuration
config = MonitoringConfig(
    tracing_backend="console",  # Options: "none", "console", "otel", "langsmith"
    visualization_output_dir="./visualizations",
)

# Create a trace manager
trace_manager = TraceManager(config=config)

# Create a blame graph
blame_graph = BlameGraph(trace_manager=trace_manager, config=config)

# Create a trace viewer
trace_viewer = TraceViewer(trace_manager=trace_manager, config=config)
```

### Creating and Managing Traces

```python
# Create a trace
trace = trace_manager.create_trace()

# Start a root span
root_span = trace_manager.start_span(
    name="execute_task",
    trace_id=trace.trace_id,
    attributes={"component": "executor"},
)

# Start a child span
child_span = trace_manager.start_span(
    name="retrieve_documents",
    trace_id=trace.trace_id,
    parent_id=root_span.span_id,
    attributes={"component": "retriever"},
)

# Add an event to the span
child_span.add_event(
    name="documents_found",
    attributes={"count": 5},
)

# Set an attribute
child_span.set_attribute("query", "example query")

# End spans
trace_manager.end_span(child_span.span_id)
trace_manager.end_span(root_span.span_id)
```

### Analyzing Performance with Blame Graph

```python
# Process a trace to update the blame graph
blame_graph.process_trace(trace)

# Identify bottlenecks
bottlenecks = blame_graph.identify_bottlenecks(
    threshold_ms=100.0,  # Minimum average duration to consider
    min_call_count=5,    # Minimum number of calls to consider
)

# Identify error sources
error_sources = blame_graph.identify_error_sources(
    min_error_rate=0.1,  # Minimum error rate to consider
    min_call_count=5,    # Minimum number of calls to consider
)

# Get the critical path for a trace
critical_path = blame_graph.get_critical_path(trace.trace_id)

# Export the blame graph
blame_graph.export_graph("blame_graph.json")
```

### Visualizing Traces

```python
# View a trace
trace_viewer.view_trace(
    trace_id=trace.trace_id,
    output_path="trace_visualization.html",
    show=True,  # Open in browser
)

# View a specific span
trace_viewer.view_span(
    trace_id=trace.trace_id,
    span_id=child_span.span_id,
    output_path="span_visualization.html",
)

# Export a trace to JSON
trace_viewer.export_trace(
    trace_id=trace.trace_id,
    output_path="trace.json",
)
```

### GASA Heatmap Visualization

```python
from saplings.monitoring import GASAHeatmap
from saplings.gasa import MaskFormat, MaskType

# Create a GASA heatmap visualizer
heatmap = GASAHeatmap(config=config)

# Visualize a mask
heatmap.visualize_mask(
    mask=mask,  # Attention mask from MaskBuilder
    format=MaskFormat.DENSE,
    mask_type=MaskType.ATTENTION,
    output_path="gasa_mask.html",
    title="GASA Attention Mask",
    show=True,  # Open in browser
)

# Visualize mask sparsity
heatmap.visualize_mask_sparsity(
    mask=mask,
    format=MaskFormat.DENSE,
    mask_type=MaskType.ATTENTION,
    output_path="gasa_sparsity.html",
)

# Compare multiple masks
heatmap.visualize_mask_comparison(
    masks=[
        (mask1, MaskFormat.DENSE, MaskType.ATTENTION, "Mask 1"),
        (mask2, MaskFormat.DENSE, MaskType.ATTENTION, "Mask 2"),
    ],
    output_path="gasa_comparison.html",
)
```

### LangSmith Integration

```python
from saplings.monitoring import LangSmithExporter

# Create a LangSmith exporter
exporter = LangSmithExporter(
    trace_manager=trace_manager,
    config=config,
    api_key="your-langsmith-api-key",
    project_name="your-project-name",
)

# Export a trace to LangSmith
run_id = exporter.export_trace(trace.trace_id)

# Export multiple traces
run_ids = exporter.export_traces(
    trace_ids=[trace1.trace_id, trace2.trace_id],
)
```

## Configuration Options

The `MonitoringConfig` class provides various configuration options for the monitoring system:

```python
from saplings.monitoring import MonitoringConfig, TracingBackend, VisualizationFormat

config = MonitoringConfig(
    # General settings
    enabled=True,  # Enable/disable monitoring
    
    # Tracing settings
    tracing_backend=TracingBackend.OTEL,  # Tracing backend
    otel_endpoint="http://localhost:4317",  # OpenTelemetry endpoint
    trace_sampling_rate=1.0,  # Sampling rate (0.0 to 1.0)
    
    # LangSmith settings
    langsmith_api_key="your-api-key",  # LangSmith API key
    langsmith_project="your-project",  # LangSmith project name
    
    # Visualization settings
    visualization_format=VisualizationFormat.HTML,  # Output format
    visualization_output_dir="./visualizations",  # Output directory
    
    # Feature toggles
    enable_blame_graph=True,  # Enable causal blame graph
    enable_gasa_heatmap=True,  # Enable GASA heatmap visualization
    
    # Performance settings
    max_spans_per_trace=1000,  # Maximum spans per trace
)
```

## Security and Privacy Considerations

When using the monitoring system, consider the following security and privacy guidelines:

1. **API Keys**: Store API keys securely and never commit them to version control.

2. **Sensitive Data**: Avoid including sensitive data in span attributes or events. If necessary, implement data scrubbing before exporting traces.

3. **Data Retention**: Regularly clear old traces to manage storage and reduce exposure risk:

   ```python
   # Clear traces older than 7 days
   from datetime import datetime, timedelta
   trace_manager.clear_traces(before_time=datetime.now() - timedelta(days=7))
   ```

4. **Access Control**: When exporting visualizations or data, ensure they are stored in secure locations with appropriate access controls.

5. **Network Security**: When using OpenTelemetry with remote endpoints, ensure the connection is secure (TLS/SSL) and properly authenticated.

6. **Resource Limits**: Set appropriate limits on trace collection to prevent excessive resource usage:

   ```python
   config = MonitoringConfig(
       max_spans_per_trace=1000,  # Limit spans per trace
       trace_sampling_rate=0.1,   # Sample only 10% of traces
   )
   ```

## Advanced Usage

### Custom Span Processors

You can extend the monitoring system with custom span processors:

```python
from saplings.monitoring.trace import TraceManager, Span

class CustomSpanProcessor:
    def on_start(self, span: Span) -> None:
        # Process span start
        print(f"Span started: {span.name}")
    
    def on_end(self, span: Span) -> None:
        # Process span end
        print(f"Span ended: {span.name}, duration: {span.duration_ms()} ms")

# Register the processor with the trace manager
trace_manager.add_span_processor(CustomSpanProcessor())
```

### Integration with Executor

The monitoring system can be integrated with the Executor for automatic tracing:

```python
from saplings.executor import Executor, ExecutorConfig
from saplings.monitoring import TraceManager, MonitoringConfig

# Create a trace manager
trace_manager = TraceManager(config=MonitoringConfig())

# Create an executor with tracing
executor = Executor(
    model=model,
    config=ExecutorConfig(
        enable_tracing=True,
    ),
    trace_manager=trace_manager,
)

# Execute with tracing
result = await executor.execute(
    prompt="Example prompt",
    trace_id="custom-trace-id",  # Optional: provide a custom trace ID
)

# Get the trace
trace = trace_manager.get_trace(result.trace_id)
```

### Performance Visualization

```python
from saplings.monitoring import PerformanceVisualizer

# Create a performance visualizer
visualizer = PerformanceVisualizer(config=config)

# Visualize latency metrics
latencies = {
    "Retriever": [10.5, 12.3, 9.8, 11.2, 10.9],
    "GASA": [15.2, 14.8, 16.1, 15.5, 14.9],
    "Model": [85.3, 86.1, 85.8, 85.5, 86.0],
}

visualizer.visualize_latency(
    latencies=latencies,
    output_path="latency.html",
    title="Component Latency",
    show=True,
)
```

## Troubleshooting

### Common Issues

1. **Missing Dependencies**:
   - For visualization: `pip install matplotlib plotly`
   - For OpenTelemetry: `pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp`
   - For LangSmith: `pip install langsmith`
   - For graph analysis: `pip install networkx`

2. **Trace Not Found**: Ensure the trace ID is correct and the trace hasn't been cleared.

3. **Visualization Not Working**: Check that the output directory exists and is writable.

4. **OTEL Connection Failed**: Verify the endpoint URL and network connectivity.

### Logging

The monitoring system uses Python's logging module. To enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("saplings.monitoring")
```

## Examples

See the `examples/` directory for complete examples:

- `gasa_visualization_example.py`: Demonstrates GASA heatmap visualization
- `trace_viewer_example.py`: Shows how to use the TraceViewer
- `blame_graph_example.py`: Illustrates causal blame graph analysis
- `langsmith_export_example.py`: Demonstrates LangSmith integration
