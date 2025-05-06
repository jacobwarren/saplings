# Monitoring System

The Monitoring system in Saplings provides comprehensive tracing, visualization, and performance analysis capabilities to help understand and optimize agent behavior.

## Overview

The Monitoring system consists of several key components:

- **TraceManager**: Manages traces and spans for tracking execution
- **BlameGraph**: Identifies performance bottlenecks and error sources
- **TraceViewer**: Visualizes traces for analysis
- **GASAHeatmap**: Visualizes GASA attention masks
- **PerformanceVisualizer**: Visualizes performance metrics
- **MonitoringService**: Orchestrates monitoring operations

This system enables detailed tracking of agent execution, identification of performance bottlenecks, and visualization of execution patterns.

## Core Concepts

### Traces and Spans

Traces represent the execution of a task, while spans represent individual operations within that task:

- **Trace**: A collection of spans that represent a complete execution
- **Span**: An individual operation with a start time, end time, and attributes
- **SpanContext**: Context information for a span

Traces and spans form a hierarchical structure that captures the execution flow, enabling detailed analysis of performance and behavior.

### Blame Graph

The Blame Graph is a causal graph that identifies performance bottlenecks and error sources:

- **BlameNode**: Represents a component or operation in the graph
- **BlameEdge**: Represents a relationship between nodes
- **Metrics**: Performance metrics like duration, call count, and error rate

The Blame Graph analyzes trace data to identify which components are causing performance issues or errors, enabling targeted optimization.

### Visualization

The Monitoring system provides various visualization capabilities:

- **Trace Visualization**: Visualizes the execution flow of a trace
- **GASA Heatmap**: Visualizes GASA attention masks
- **Performance Visualization**: Visualizes performance metrics like latency and throughput

These visualizations help understand agent behavior, identify issues, and optimize performance.

## API Reference

### TraceManager

```python
class TraceManager:
    def __init__(
        self,
        config: Optional[MonitoringConfig] = None,
    ):
        """Initialize the trace manager."""

    def create_trace(
        self,
        trace_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Trace:
        """Create a new trace."""

    def start_span(
        self,
        name: str,
        trace_id: str,
        parent_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Span:
        """Start a new span."""

    def end_span(
        self,
        span_id: str,
        status: str = "OK",
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        """End a span."""

    def get_trace(self, trace_id: str) -> Optional[Trace]:
        """Get a trace by ID."""

    def get_span(self, span_id: str) -> Optional[Span]:
        """Get a span by ID."""

    def list_traces(self) -> List[str]:
        """List all trace IDs."""

    def clear_traces(self) -> None:
        """Clear all traces."""
```

### BlameGraph

```python
class BlameGraph:
    def __init__(
        self,
        trace_manager: Optional[TraceManager] = None,
        config: Optional[MonitoringConfig] = None,
    ):
        """Initialize the blame graph."""

    def process_trace(self, trace: Trace) -> None:
        """Process a trace to update the blame graph."""

    def identify_bottlenecks(
        self,
        threshold_ms: float = 100.0,
        min_call_count: int = 5,
    ) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks in the graph."""

    def identify_error_sources(
        self,
        min_error_rate: float = 0.1,
        min_call_count: int = 5,
    ) -> List[Dict[str, Any]]:
        """Identify error sources in the graph."""

    def export_graph(self, output_path: str) -> bool:
        """Export the blame graph to a file."""

    def export_graphml(self, output_path: str) -> bool:
        """Export the blame graph to GraphML format."""
```

### TraceViewer

```python
class TraceViewer:
    def __init__(
        self,
        trace_manager: Optional[TraceManager] = None,
        config: Optional[MonitoringConfig] = None,
    ):
        """Initialize the trace viewer."""

    def view_trace(
        self,
        trace_id: str,
        output_path: Optional[str] = None,
        format: Optional[VisualizationFormat] = None,
        show: bool = False,
    ) -> Optional[Any]:
        """View a trace."""

    def view_traces(
        self,
        trace_ids: List[str],
        output_path: Optional[str] = None,
        format: Optional[VisualizationFormat] = None,
        show: bool = False,
    ) -> Optional[Any]:
        """View multiple traces."""

    def view_span(
        self,
        trace_id: str,
        span_id: str,
        output_path: Optional[str] = None,
        format: Optional[VisualizationFormat] = None,
        show: bool = False,
    ) -> Optional[Any]:
        """View a specific span within a trace."""
```

### GASAHeatmap

```python
class GASAHeatmap:
    def __init__(
        self,
        config: Optional[MonitoringConfig] = None,
    ):
        """Initialize the GASA heatmap."""

    def visualize_mask(
        self,
        mask: np.ndarray,
        output_path: Optional[str] = None,
        format: Optional[VisualizationFormat] = None,
        mask_type: str = "binary",
        title: Optional[str] = None,
        show: bool = False,
        interactive: bool = True,
        token_labels: Optional[List[str]] = None,
        highlight_tokens: Optional[List[int]] = None,
    ) -> Optional[Any]:
        """Visualize a GASA mask."""
```

### PerformanceVisualizer

```python
class PerformanceVisualizer:
    def __init__(
        self,
        config: Optional[MonitoringConfig] = None,
    ):
        """Initialize the performance visualizer."""

    def visualize_latency(
        self,
        latencies: Dict[str, List[float]],
        output_path: Optional[str] = None,
        format: Optional[VisualizationFormat] = None,
        title: Optional[str] = None,
        show: bool = False,
        interactive: bool = True,
    ) -> Optional[Any]:
        """Visualize latency by component."""

    def visualize_throughput(
        self,
        throughputs: Dict[str, List[float]],
        output_path: Optional[str] = None,
        format: Optional[VisualizationFormat] = None,
        title: Optional[str] = None,
        show: bool = False,
        interactive: bool = True,
    ) -> Optional[Any]:
        """Visualize throughput by component."""

    def visualize_error_rate(
        self,
        error_rates: Dict[str, List[float]],
        output_path: Optional[str] = None,
        format: Optional[VisualizationFormat] = None,
        title: Optional[str] = None,
        show: bool = False,
        interactive: bool = True,
    ) -> Optional[Any]:
        """Visualize error rate by component."""
```

### MonitoringService

```python
class MonitoringService:
    def __init__(
        self,
        enabled: bool = True,
        output_dir: str = "./output",
    ):
        """Initialize the monitoring service."""

    async def create_trace(self, timeout: Optional[float] = None) -> Dict:
        """Create a new trace if monitoring is enabled."""

    async def start_span(
        self,
        name: str,
        trace_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        attributes: Optional[Dict] = None,
        timeout: Optional[float] = None,
    ):
        """Start a new span if monitoring is enabled."""

    async def end_span(self, span_id: str, timeout: Optional[float] = None):
        """End a span if monitoring is enabled."""

    async def process_trace(self, trace_id: str, timeout: Optional[float] = None):
        """Process a trace for blame graph if monitoring is enabled."""

    async def identify_bottlenecks(
        self,
        threshold_ms: float = 100.0,
        min_call_count: int = 1,
        timeout: Optional[float] = None,
    ) -> List:
        """Identify performance bottlenecks if monitoring is enabled."""

    async def identify_error_sources(
        self,
        min_error_rate: float = 0.1,
        min_call_count: int = 1,
        timeout: Optional[float] = None,
    ) -> List:
        """Identify error sources if monitoring is enabled."""

    async def visualize_trace(
        self,
        trace_id: str,
        output_path: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        """Visualize a trace if monitoring is enabled."""
```

### MonitoringConfig

```python
class MonitoringConfig(BaseModel):
    enabled: bool = True  # Whether monitoring is enabled
    tracing_backend: TracingBackend = TracingBackend.CONSOLE  # Tracing backend to use
    otel_endpoint: Optional[str] = None  # OpenTelemetry endpoint URL
    langsmith_api_key: Optional[str] = None  # LangSmith API key
    langsmith_project: Optional[str] = None  # LangSmith project name
    trace_sampling_rate: float = 1.0  # Sampling rate for traces (0.0 to 1.0)
    visualization_format: VisualizationFormat = VisualizationFormat.HTML  # Default format for visualizations
    visualization_output_dir: str = "./visualizations"  # Directory for visualization outputs
    enable_blame_graph: bool = True  # Whether to enable the causal blame graph
    enable_gasa_heatmap: bool = True  # Whether to enable GASA heatmap visualization
    max_spans_per_trace: int = 1000  # Maximum number of spans per trace
    metadata: Dict[str, str] = {}  # Additional metadata for traces
```

### Enums

```python
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
```

## Usage Examples

### Basic Tracing

```python
from saplings.monitoring import TraceManager, MonitoringConfig

# Create a trace manager
trace_manager = TraceManager(config=MonitoringConfig())

# Create a trace
trace = trace_manager.create_trace()
print(f"Created trace: {trace.trace_id}")

# Start a span
span = trace_manager.start_span(
    name="process_data",
    trace_id=trace.trace_id,
    attributes={"component": "data_processor"},
)
print(f"Started span: {span.span_id}")

# Simulate work
import time
time.sleep(0.5)

# End the span
trace_manager.end_span(span.span_id)
print(f"Ended span: {span.span_id}")

# Get the trace
trace = trace_manager.get_trace(trace.trace_id)
print(f"Trace duration: {trace.duration_ms()} ms")
```

### Nested Spans

```python
from saplings.monitoring import TraceManager, MonitoringConfig

# Create a trace manager
trace_manager = TraceManager(config=MonitoringConfig())

# Create a trace
trace = trace_manager.create_trace()

# Start a parent span
parent_span = trace_manager.start_span(
    name="process_request",
    trace_id=trace.trace_id,
    attributes={"component": "api_handler"},
)

# Simulate work
import time
time.sleep(0.2)

# Start a child span
child_span = trace_manager.start_span(
    name="query_database",
    trace_id=trace.trace_id,
    parent_id=parent_span.span_id,
    attributes={"component": "database"},
)

# Simulate work
time.sleep(0.3)

# End the child span
trace_manager.end_span(child_span.span_id)

# Start another child span
child_span2 = trace_manager.start_span(
    name="format_response",
    trace_id=trace.trace_id,
    parent_id=parent_span.span_id,
    attributes={"component": "formatter"},
)

# Simulate work
time.sleep(0.1)

# End the second child span
trace_manager.end_span(child_span2.span_id)

# End the parent span
trace_manager.end_span(parent_span.span_id)

# Get the trace
trace = trace_manager.get_trace(trace.trace_id)
print(f"Trace duration: {trace.duration_ms()} ms")
print(f"Number of spans: {len(trace.spans)}")
```

### Blame Graph Analysis

```python
from saplings.monitoring import TraceManager, BlameGraph, MonitoringConfig

# Create a trace manager
trace_manager = TraceManager(config=MonitoringConfig())

# Create a blame graph
blame_graph = BlameGraph(trace_manager=trace_manager)

# Create a trace
trace = trace_manager.create_trace()

# Create spans for different components
components = ["api", "database", "cache", "processor"]
spans = []

for component in components:
    # Start a span
    span = trace_manager.start_span(
        name=f"{component}_operation",
        trace_id=trace.trace_id,
        attributes={"component": component},
    )
    spans.append(span)

    # Simulate work with different durations
    import time
    import random
    duration = random.uniform(0.1, 0.5)
    time.sleep(duration)

    # End the span
    trace_manager.end_span(span.span_id)

# Process the trace
blame_graph.process_trace(trace_manager.get_trace(trace.trace_id))

# Identify bottlenecks
bottlenecks = blame_graph.identify_bottlenecks(threshold_ms=100.0, min_call_count=1)
print("Bottlenecks:")
for bottleneck in bottlenecks:
    print(f"  {bottleneck['component']}.{bottleneck['name']}: {bottleneck['avg_time_ms']:.2f} ms")

# Export the graph
blame_graph.export_graph("blame_graph.json")
```

### Trace Visualization

```python
from saplings.monitoring import TraceManager, TraceViewer, MonitoringConfig

# Create a trace manager
trace_manager = TraceManager(config=MonitoringConfig())

# Create a trace viewer
trace_viewer = TraceViewer(trace_manager=trace_manager)

# Create a trace
trace = trace_manager.create_trace()

# Create a complex span hierarchy
root_span = trace_manager.start_span(
    name="process_request",
    trace_id=trace.trace_id,
    attributes={"component": "api_handler"},
)

# Add child spans
for i in range(3):
    child_span = trace_manager.start_span(
        name=f"subtask_{i}",
        trace_id=trace.trace_id,
        parent_id=root_span.span_id,
        attributes={"component": f"processor_{i}"},
    )

    # Add grandchild spans
    for j in range(2):
        grandchild_span = trace_manager.start_span(
            name=f"operation_{j}",
            trace_id=trace.trace_id,
            parent_id=child_span.span_id,
            attributes={"component": f"worker_{j}"},
        )

        # Simulate work
        import time
        time.sleep(0.1)

        # End the grandchild span
        trace_manager.end_span(grandchild_span.span_id)

    # End the child span
    trace_manager.end_span(child_span.span_id)

# End the root span
trace_manager.end_span(root_span.span_id)

# Visualize the trace
trace_viewer.view_trace(
    trace_id=trace.trace_id,
    output_path="trace_visualization.html",
    show=True,
)
```

### Integration with Agent

```python
from saplings import Agent, AgentConfig
from saplings.monitoring import MonitoringConfig, TraceManager

# Create a monitoring configuration
monitoring_config = MonitoringConfig(
    visualization_output_dir="./visualizations",
)

# Create a trace manager
trace_manager = TraceManager(config=monitoring_config)

# Create an agent with monitoring
agent = Agent(
    config=AgentConfig(
        provider="openai",
        model_name="gpt-4o",
        enable_monitoring=True,
    )
)

# Set the trace manager
agent.trace_manager = trace_manager

# Run a task
import asyncio
result = asyncio.run(agent.run("Explain the concept of graph-based memory"))

# Get the trace ID from the result
trace_id = result.get("trace_id")

# Visualize the trace
from saplings.monitoring import TraceViewer
trace_viewer = TraceViewer(trace_manager=trace_manager)
trace_viewer.view_trace(
    trace_id=trace_id,
    output_path="agent_execution_trace.html",
    show=True,
)
```

## Advanced Features

### GASA Heatmap Visualization

```python
from saplings.monitoring import GASAHeatmap, MonitoringConfig
import numpy as np

# Create a GASA heatmap visualizer
heatmap = GASAHeatmap(config=MonitoringConfig())

# Create a sample mask
mask = np.zeros((10, 10))
for i in range(10):
    for j in range(10):
        if i <= j:  # Upper triangular matrix
            mask[i, j] = 1.0
        if abs(i - j) <= 2:  # Diagonal band
            mask[i, j] = 1.0

# Visualize the mask
heatmap.visualize_mask(
    mask=mask,
    output_path="gasa_heatmap.html",
    mask_type="binary",
    title="GASA Attention Mask",
    show=True,
    interactive=True,
    token_labels=[f"Token {i}" for i in range(10)],
    highlight_tokens=[0, 5, 9],
)
```

### Performance Visualization

```python
from saplings.monitoring import PerformanceVisualizer, MonitoringConfig
import numpy as np

# Create a performance visualizer
visualizer = PerformanceVisualizer(config=MonitoringConfig())

# Create sample latency data
latencies = {
    "api": [10.5, 12.3, 9.8, 11.2, 10.9],
    "database": [25.6, 28.1, 24.9, 26.3, 27.5],
    "cache": [5.2, 4.8, 5.5, 4.9, 5.1],
    "processor": [15.3, 16.2, 14.9, 15.8, 15.5],
}

# Visualize latency
visualizer.visualize_latency(
    latencies=latencies,
    output_path="latency_visualization.html",
    title="Component Latency",
    show=True,
    interactive=True,
)

# Create sample throughput data
throughputs = {
    "api": [100.5, 98.3, 102.1, 99.7, 101.2],
    "database": [50.6, 48.9, 51.3, 49.5, 50.2],
    "cache": [200.2, 198.5, 201.8, 199.3, 200.7],
    "processor": [75.3, 74.1, 76.2, 74.8, 75.5],
}

# Visualize throughput
visualizer.visualize_throughput(
    throughputs=throughputs,
    output_path="throughput_visualization.html",
    title="Component Throughput",
    show=True,
    interactive=True,
)

# Create sample error rate data
error_rates = {
    "api": [0.01, 0.02, 0.01, 0.03, 0.02],
    "database": [0.05, 0.04, 0.06, 0.05, 0.04],
    "cache": [0.001, 0.002, 0.001, 0.003, 0.002],
    "processor": [0.03, 0.02, 0.04, 0.03, 0.02],
}

# Visualize error rate
visualizer.visualize_error_rate(
    error_rates=error_rates,
    output_path="error_rate_visualization.html",
    title="Component Error Rate",
    show=True,
    interactive=True,
)
```

### OpenTelemetry Integration

```python
from saplings.monitoring import TraceManager, MonitoringConfig, TracingBackend

# Create a monitoring configuration with OpenTelemetry
config = MonitoringConfig(
    tracing_backend=TracingBackend.OTEL,
    otel_endpoint="http://localhost:4317",
)

# Create a trace manager
trace_manager = TraceManager(config=config)

# Create a trace
trace = trace_manager.create_trace()

# Start a span
span = trace_manager.start_span(
    name="process_data",
    trace_id=trace.trace_id,
    attributes={"component": "data_processor"},
)

# Simulate work
import time
time.sleep(0.5)

# End the span
trace_manager.end_span(span.span_id)

# The trace will be exported to the OpenTelemetry endpoint
```

### LangSmith Integration

```python
from saplings.monitoring import TraceManager, MonitoringConfig, TracingBackend

# Create a monitoring configuration with LangSmith
config = MonitoringConfig(
    tracing_backend=TracingBackend.LANGSMITH,
    langsmith_api_key="your_langsmith_api_key",
    langsmith_project="your_project_name",
)

# Create a trace manager
trace_manager = TraceManager(config=config)

# Create a trace
trace = trace_manager.create_trace()

# Start a span
span = trace_manager.start_span(
    name="process_data",
    trace_id=trace.trace_id,
    attributes={"component": "data_processor"},
)

# Simulate work
import time
time.sleep(0.5)

# End the span
trace_manager.end_span(span.span_id)

# The trace will be exported to LangSmith
```

## Implementation Details

### Trace Management Process

The trace management process works as follows:

1. **Trace Creation**: Create a trace with a unique ID
2. **Span Creation**: Create spans within the trace to represent operations
3. **Span Hierarchy**: Build a hierarchy of spans to represent the execution flow
4. **Span Completion**: Complete spans with duration and status information
5. **Trace Analysis**: Analyze the trace to identify performance issues and errors

### Blame Graph Process

The blame graph process works as follows:

1. **Node Creation**: Create nodes for each component or operation
2. **Edge Creation**: Create edges to represent relationships between nodes
3. **Metric Collection**: Collect performance metrics for nodes and edges
4. **Bottleneck Identification**: Identify nodes with high latency or error rates
5. **Graph Export**: Export the graph for visualization and analysis

### Visualization Process

The visualization process works as follows:

1. **Data Preparation**: Prepare trace or performance data for visualization
2. **Visualization Creation**: Create a visualization using Plotly or Matplotlib
3. **Interactive Elements**: Add interactive elements for exploration
4. **Output Generation**: Generate the visualization in the specified format
5. **Display or Save**: Display the visualization or save it to a file

## Extension Points

The Monitoring system is designed to be extensible:

### Custom Trace Backend

You can create a custom trace backend by extending the `TraceManager` class:

```python
from saplings.monitoring import TraceManager, MonitoringConfig

class CustomTraceManager(TraceManager):
    def __init__(self, config: Optional[MonitoringConfig] = None):
        super().__init__(config)
        # Initialize custom backend

    def create_trace(self, trace_id=None, attributes=None):
        # Create a trace in the custom backend
        trace = super().create_trace(trace_id, attributes)
        # Additional custom logic
        return trace

    def start_span(self, name, trace_id, parent_id=None, attributes=None):
        # Start a span in the custom backend
        span = super().start_span(name, trace_id, parent_id, attributes)
        # Additional custom logic
        return span

    def end_span(self, span_id, status="OK", attributes=None):
        # End a span in the custom backend
        super().end_span(span_id, status, attributes)
        # Additional custom logic
```

### Custom Visualization

You can create a custom visualization by extending the `TraceViewer` or `PerformanceVisualizer` classes:

```python
from saplings.monitoring import TraceViewer, MonitoringConfig

class CustomTraceViewer(TraceViewer):
    def __init__(self, trace_manager=None, config=None):
        super().__init__(trace_manager, config)
        # Initialize custom visualization

    def view_trace(self, trace_id, output_path=None, format=None, show=False):
        # Get the trace
        trace = self.trace_manager.get_trace(trace_id)
        if not trace:
            return None

        # Create a custom visualization
        # ...

        # Return the visualization
        return visualization
```

### Custom Blame Graph Analysis

You can create a custom blame graph analysis by extending the `BlameGraph` class:

```python
from saplings.monitoring import BlameGraph, MonitoringConfig

class CustomBlameGraph(BlameGraph):
    def __init__(self, trace_manager=None, config=None):
        super().__init__(trace_manager, config)
        # Initialize custom analysis

    def identify_bottlenecks(self, threshold_ms=100.0, min_call_count=5):
        # Custom bottleneck identification logic
        # ...

        # Return the bottlenecks
        return bottlenecks

    def identify_error_sources(self, min_error_rate=0.1, min_call_count=5):
        # Custom error source identification logic
        # ...

        # Return the error sources
        return error_sources
```

## Conclusion

The Monitoring system in Saplings provides comprehensive tracing, visualization, and performance analysis capabilities to help understand and optimize agent behavior. By tracking execution, identifying bottlenecks, and visualizing performance, it enables detailed analysis and optimization of agent performance.
