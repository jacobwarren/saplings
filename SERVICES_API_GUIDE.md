# Services API Guide

This guide provides comprehensive documentation for all public service APIs available in Saplings, including managers and specialized services that are exposed through the public API.

## Table of Contents

- [Overview](#overview)
- [Memory Manager](#memory-manager)
- [Model Services](#model-services)
- [Monitoring Service](#monitoring-service)
- [Service Integration](#service-integration)
- [Usage Patterns](#usage-patterns)
- [Best Practices](#best-practices)

## Overview

Saplings exposes several high-level service APIs that provide specific functionality for different aspects of the agent framework. These services are public APIs designed for direct use by developers.

### Public vs Internal APIs

**Public APIs** (documented here):
- `MemoryManager` - Memory and document management
- `ModelCachingService` - Model response caching
- `ModelInitializationService` - Model setup and configuration
- `MonitoringService` - Tracing and performance monitoring

**Internal APIs** (not for direct use):
- Internal service implementations under `_internal/` directories
- Service builders under `services/_internal/builders/`
- Interface implementations that are wrapped by public APIs

## Memory Manager

The `MemoryManager` is a public service for managing documents, vector storage, and dependency graphs.

### Basic Usage

```python
from saplings.api.services import MemoryManager, MemoryManagerBuilder

# Using the builder (recommended)
memory_manager = (MemoryManagerBuilder()
    .with_storage(vector_store)
    .with_indexer(indexer)
    .build())

# Direct instantiation
from saplings.api.memory import MemoryStore, MemoryConfig

config = MemoryConfig(chunk_size=1000, chunk_overlap=200)
memory_store = MemoryStore(config=config)
memory_manager = MemoryManager(
    memory_store=memory_store,
    trace_manager=trace_manager
)
```

### Key Methods

```python
# Add documents
documents = await memory_manager.add_documents_from_directory(
    directory="./documents",
    extension=".txt",
    use_indexer=True
)

# Retrieve documents
document = await memory_manager.get_document("doc_id")

# Store content
from saplings.api.memory import Document, DocumentMetadata

metadata = DocumentMetadata(title="Example", source="manual")
document = Document(content="Sample content", metadata=metadata)
await memory_manager.add_document(document)

# Access components
memory_store = memory_manager.memory_store
dependency_graph = memory_manager.dependency_graph
vector_store = memory_manager.vector_store
indexer = memory_manager.indexer
```

## Model Services

### Model Caching Service

The `ModelCachingService` provides intelligent caching of model responses to reduce costs and improve performance.

```python
from saplings.api.services import ModelCachingService

# Create with model initialization service
caching_service = ModelCachingService(
    model_initialization_service=model_service,
    cache_enabled=True,
    cache_namespace="model",
    cache_ttl=3600,  # 1 hour
    cache_provider="memory",
    cache_strategy="lru",
)

# Generate text with caching
response = await caching_service.generate_text_with_cache(
    prompt="What is the capital of France?",
    max_tokens=100,
    temperature=0.7,
)

# Generate full response with caching
llm_response = await caching_service.generate_response_with_cache(
    prompt="Explain quantum computing",
    max_tokens=500,
    temperature=0.3,
)

# Cost estimation
cost = await caching_service.estimate_cost(
    prompt_tokens=100,
    completion_tokens=50
)
```

### Model Initialization Service

The `ModelInitializationService` handles model setup, configuration, and lifecycle management.

```python
from saplings.api.services import ModelInitializationService

# Create service
model_service = ModelInitializationService(
    provider="openai",
    model_name="gpt-4o",
    api_key="your-api-key",
    model_parameters={
        "temperature": 0.7,
        "max_tokens": 1000,
    }
)

# Get the initialized model
model = await model_service.get_model()

# Get model metadata
metadata = await model_service.get_model_metadata()
print(f"Model: {metadata['model_name']}")
print(f"Provider: {metadata['provider']}")

# Initialize with timeout
model = await model_service.get_model(timeout=30.0)
```

## Monitoring Service

The `MonitoringService` provides comprehensive tracing, performance monitoring, and observability features.

### Basic Setup

```python
from saplings.api.services import MonitoringService
from saplings.monitoring import MonitoringConfig, TracingBackend, VisualizationFormat

# Create with default configuration
monitoring_service = MonitoringService(
    output_dir="./output",
    enabled=True,
)

# Create with custom configuration
config = MonitoringConfig(
    enabled=True,
    tracing_backend=TracingBackend.CONSOLE,
    otel_endpoint=None,
    langsmith_api_key=None,
    langsmith_project=None,
    trace_sampling_rate=1.0,
    visualization_format=VisualizationFormat.HTML,
    visualization_output_dir="./visualizations",
    enable_blame_graph=True,
    enable_gasa_heatmap=True,
    max_spans_per_trace=1000,
    metadata={"environment": "development"},
)

monitoring_service = MonitoringService(config=config)
```

### Tracing Operations

```python
# Create a trace
trace = monitoring_service.create_trace()

# Start a span
span = monitoring_service.start_span(
    name="execute_task",
    trace_id=trace.trace_id,
    attributes={"task": "summarize_document"},
)

# Add events and attributes to spans
monitoring_service.add_span_event(
    span.span_id, 
    "processing_started", 
    {"document_size": 1024}
)

monitoring_service.add_span_attribute(
    span.span_id, 
    "processing_time", 
    1.5
)

# End the span
monitoring_service.end_span(span.span_id)

# Process the trace for analysis
monitoring_service.process_trace(trace.trace_id)
```

### Performance Analysis

```python
# Identify performance bottlenecks
bottlenecks = monitoring_service.identify_bottlenecks(
    threshold_ms=100.0,
    min_call_count=1,
)

for bottleneck in bottlenecks:
    print(f"Bottleneck: {bottleneck['operation']}")
    print(f"Average duration: {bottleneck['avg_duration_ms']}ms")
    print(f"Call count: {bottleneck['call_count']}")

# Identify error sources
error_sources = monitoring_service.identify_error_sources(
    min_error_rate=0.1,
    min_call_count=1,
)

for error_source in error_sources:
    print(f"Error source: {error_source['operation']}")
    print(f"Error rate: {error_source['error_rate']:.1%}")
```

### Service Registration

```python
# Register services for monitoring
monitoring_service.register_service("memory_manager", memory_manager)
monitoring_service.register_service("model_service", model_service)

# Register hooks for service events
def on_memory_service_ready(service):
    print(f"Memory service is ready: {service}")

monitoring_service.register_service_hook("memory_manager", on_memory_service_ready)

# Get registered services
all_services = monitoring_service.get_registered_services()
memory_service = monitoring_service.get_registered_service("memory_manager")
```

## Service Integration

### Using Services Together

```python
from saplings import AgentBuilder
from saplings.api.services import (
    MemoryManager, 
    ModelCachingService, 
    ModelInitializationService,
    MonitoringService
)

# Create monitoring service first
monitoring_service = MonitoringService(enabled=True, output_dir="./monitoring")

# Create model services
model_service = ModelInitializationService(
    provider="openai",
    model_name="gpt-4o",
    model_parameters={"temperature": 0.7}
)

caching_service = ModelCachingService(
    model_initialization_service=model_service,
    cache_enabled=True,
    cache_ttl=3600
)

# Create memory manager
memory_manager = MemoryManager(
    memory_path="./agent_memory",
    trace_manager=monitoring_service.trace_manager
)

# Register services with monitoring
monitoring_service.register_service("model_service", model_service)
monitoring_service.register_service("caching_service", caching_service)
monitoring_service.register_service("memory_manager", memory_manager)

# Create agent with services
agent = (AgentBuilder.for_openai("gpt-4o")
    .with_memory_manager(memory_manager)
    .with_monitoring(monitoring_service)
    .build())
```

### Service Dependencies

```python
# Services have natural dependencies:
# ModelCachingService -> ModelInitializationService
# MemoryManager -> (optional) MonitoringService for tracing
# All services -> (optional) MonitoringService for observability

# Ensure proper initialization order
model_init_service = ModelInitializationService(...)
model_cache_service = ModelCachingService(model_init_service, ...)
monitoring_service = MonitoringService(...)
memory_manager = MemoryManager(trace_manager=monitoring_service.trace_manager)
```

## Usage Patterns

### Pattern 1: Agent with Full Observability

```python
# Set up monitoring first
monitoring = MonitoringService(
    enabled=True,
    output_dir="./monitoring",
    config=MonitoringConfig(
        enable_blame_graph=True,
        enable_gasa_heatmap=True,
        visualization_format=VisualizationFormat.HTML
    )
)

# Create agent with monitoring
agent = (AgentBuilder.for_openai("gpt-4o")
    .with_monitoring(monitoring)
    .with_memory()
    .with_tools([WebSearchTool()])
    .build())

# Execute with automatic tracing
response = agent.execute("Research the latest AI trends")

# Analyze performance
bottlenecks = monitoring.identify_bottlenecks(threshold_ms=50.0)
```

### Pattern 2: Cost-Optimized Agent

```python
# Set up model services with caching
model_service = ModelInitializationService(
    provider="openai",
    model_name="gpt-4o",
    model_parameters={"temperature": 0.3}  # Lower temperature for better caching
)

caching_service = ModelCachingService(
    model_initialization_service=model_service,
    cache_enabled=True,
    cache_ttl=3600,  # Cache for 1 hour
    cache_strategy="lru",
    cache_namespace="cost_optimized"
)

# Create agent with caching-enabled model service
agent = (AgentBuilder.for_openai("gpt-4o")
    .with_model_service(caching_service)
    .with_memory()
    .build())
```

### Pattern 3: Memory-Intensive Applications

```python
# Configure memory manager for large document processing
from saplings.api.memory import MemoryConfig

memory_config = MemoryConfig(
    chunk_size=2000,  # Larger chunks
    chunk_overlap=400,  # More overlap for better retrieval
    max_documents=10000  # Higher document limit
)

memory_store = MemoryStore(config=memory_config)
memory_manager = MemoryManager(memory_store=memory_store)

# Bulk document processing
documents = await memory_manager.add_documents_from_directory(
    directory="./large_corpus",
    extension=".pdf",
    use_indexer=True,
    timeout=300.0  # 5 minute timeout
)

# Create agent with configured memory
agent = (AgentBuilder.for_openai("gpt-4o")
    .with_memory_manager(memory_manager)
    .build())
```

## Best Practices

### 1. Service Lifecycle Management

```python
# Always initialize monitoring first if using multiple services
monitoring_service = MonitoringService(enabled=True)

# Initialize model services before dependent services
model_service = ModelInitializationService(...)
cache_service = ModelCachingService(model_service, ...)

# Register services for monitoring
monitoring_service.register_service("model", model_service)
monitoring_service.register_service("cache", cache_service)
```

### 2. Error Handling

```python
from saplings.core.exceptions import ModelError, MemoryError, MonitoringError

try:
    # Service operations
    model = await model_service.get_model(timeout=30.0)
    documents = await memory_manager.add_documents_from_directory("./docs")
except ModelError as e:
    logger.error(f"Model initialization failed: {e}")
except MemoryError as e:
    logger.error(f"Memory operation failed: {e}")
except MonitoringError as e:
    logger.error(f"Monitoring setup failed: {e}")
```

### 3. Configuration Management

```python
# Use configuration objects for complex setups
memory_config = MemoryConfig(chunk_size=1500, chunk_overlap=300)
monitoring_config = MonitoringConfig(
    enabled=True,
    tracing_backend=TracingBackend.OTEL,
    otel_endpoint="http://localhost:4317"
)

# Apply configurations consistently
memory_store = MemoryStore(config=memory_config)
monitoring_service = MonitoringService(config=monitoring_config)
```

### 4. Resource Cleanup

```python
# Services support async context managers where applicable
async with monitoring_service:
    # Service operations
    trace = monitoring_service.create_trace()
    # ... work with traces
    
# Or explicit cleanup
try:
    # Service operations
    pass
finally:
    await memory_manager.save("./backup")
    monitoring_service.process_all_traces()
```

### 5. Testing with Services

```python
# Use builders for easier testing
def create_test_memory_manager():
    return (MemoryManagerBuilder()
        .with_storage(MockVectorStore())
        .with_indexer(MockIndexer())
        .build())

def create_test_monitoring():
    return MonitoringService(
        enabled=False,  # Disable for tests
        output_dir="/tmp/test_monitoring"
    )

# Test service integration
memory_manager = create_test_memory_manager()
monitoring = create_test_monitoring()

agent = (AgentBuilder.minimal()
    .with_memory_manager(memory_manager)
    .with_monitoring(monitoring)
    .build())
```

This guide covers all the public service APIs that developers should use directly. Internal APIs under `_internal/` directories are implementation details and should not be used directly in application code.