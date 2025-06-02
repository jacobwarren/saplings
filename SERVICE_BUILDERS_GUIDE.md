# Service Builders Guide

This guide provides comprehensive documentation for all service builders in Saplings, their interactions, and how they work together to create the full agent architecture.

## Table of Contents

- [Overview](#overview)
- [Builder Architecture](#builder-architecture)
- [Individual Service Builders](#individual-service-builders)
- [Builder Interactions & Dependencies](#builder-interactions--dependencies)
- [AgentFacadeBuilder - The Orchestrator](#agentfacadebuilder---the-orchestrator)
- [Usage Patterns](#usage-patterns)
- [Best Practices](#best-practices)

## Overview

Saplings uses a comprehensive service-oriented architecture where each major functional area is handled by a dedicated service. Each service has its own builder that follows the Builder pattern to provide fluent configuration and dependency injection.

### Available Service Builders

Saplings provides the following service builders:

1. **ExecutionServiceBuilder** - Manages task execution with models and tools
2. **JudgeServiceBuilder** - Handles output quality evaluation
3. **MemoryManagerBuilder** - Manages document storage and retrieval 
4. **ModalityServiceBuilder** - Handles multi-modal content processing
5. **ModelInitializationServiceBuilder** - Manages model initialization and configuration
6. **MonitoringServiceBuilder** - Provides monitoring, tracing, and observability
7. **OrchestrationServiceBuilder** - Coordinates complex workflows
8. **PlannerServiceBuilder** - Creates and manages execution plans
9. **RetrievalServiceBuilder** - Handles document search and retrieval
10. **SelfHealingServiceBuilder** - Manages error recovery and learning
11. **ToolServiceBuilder** - Manages tool registration and execution
12. **ValidatorServiceBuilder** - Validates outputs and responses

### Base Builder Classes

- **ServiceBuilder<T>** - Base generic builder class
- **LazyServiceBuilder<T>** - Extends ServiceBuilder with lazy initialization
- **AgentFacadeBuilder** - Orchestrates all service builders

## Builder Architecture

### Base ServiceBuilder Pattern

All service builders extend the base `ServiceBuilder<T>` class:

```python
from saplings.core._internal.builder import ServiceBuilder

class ServiceBuilder(Generic[T]):
    """Base builder class for Saplings services."""
    
    def with_dependency(self, name: str, dependency: Any) -> ServiceBuilder[T]:
        """Add a dependency to the service."""
        
    def with_config(self, config: Dict[str, Any]) -> ServiceBuilder[T]:
        """Add configuration to the service."""
        
    def require_dependency(self, name: str) -> None:
        """Mark a dependency as required."""
        
    def build(self) -> T:
        """Build the service instance."""
```

### Dependency Injection

The builders use dependency injection through:

1. **Required Dependencies** - Must be provided via `require_dependency()`
2. **Optional Dependencies** - Can be provided via `with_dependency()`
3. **Configuration** - Provided via `with_config()`
4. **Lazy Loading** - Services can be initialized on-demand

## Individual Service Builders

### ExecutionServiceBuilder

Manages the core execution of tasks with language models.

```python
from saplings.services._internal.builders import ExecutionServiceBuilder

builder = ExecutionServiceBuilder()
service = builder.with_model(model) \
               .with_gasa(gasa_service) \
               .with_validator(validator_service) \
               .with_trace_manager(trace_manager) \
               .with_config({"validation_type": "standard"}) \
               .build()
```

**Dependencies:**
- **Optional**: `model` (supports lazy initialization)
- **Optional**: `gasa_service` (for GASA optimization)  
- **Optional**: `validator_service` (for output validation)
- **Optional**: `trace_manager` (for monitoring)

**Interactions:**
- Uses `validator_service` for output validation
- Uses `gasa_service` for attention optimization
- Publishes events to other services via event bus

### JudgeServiceBuilder

Creates services for evaluating output quality using rubrics.

```python
from saplings.services._internal.builders import JudgeServiceBuilder

builder = JudgeServiceBuilder()
service = builder.with_model(model) \
               .with_trace_manager(trace_manager) \
               .with_initialize_judge(True) \
               .with_rubric_path("./rubrics") \
               .build()
```

**Dependencies:**
- **Required**: `model` (LLM for judging)
- **Optional**: `trace_manager` (for monitoring)
- **Optional**: `rubric_path` (custom rubrics directory)

**Interactions:**
- Used by `ValidatorService` for output evaluation
- Can be used by `SelfHealingService` for quality assessment

### MemoryManagerBuilder

Manages document storage, vector stores, and dependency graphs.

```python
from saplings.services._internal.builders import MemoryManagerBuilder

builder = MemoryManagerBuilder()
manager = builder.with_memory_path("./agent_memory") \
                .with_trace_manager(trace_manager) \
                .with_memory_store(custom_store) \  # Optional pre-configured store
                .build()
```

**Dependencies:**
- **Required**: Either `memory_path` OR `memory_store`
- **Optional**: `trace_manager` (for monitoring)
- **Optional**: `memory_store` (pre-configured store)

**Interactions:**
- Used by `RetrievalService` for document search
- Used by `ExecutionService` for context management
- Provides dependency graph to `GASAService`

### ModalityServiceBuilder

Handles multi-modal content processing (text, images, audio, video).

```python
from saplings.services._internal.builders import ModalityServiceBuilder

builder = ModalityServiceBuilder()
service = builder.with_model(model) \
               .with_supported_modalities(["text", "image", "audio"]) \
               .with_trace_manager(trace_manager) \
               .build()
```

**Dependencies:**
- **Optional**: `model` (can use model_provider instead)
- **Optional**: `model_provider` (lazy model loading)
- **Optional**: `supported_modalities` (defaults to ["text"])
- **Optional**: `trace_manager` (for monitoring)

**Interactions:**
- Used by `ExecutionService` for multi-modal processing
- Can register custom modality handlers

### ModelInitializationServiceBuilder

Manages model creation, configuration, and parameter management.

```python
from saplings.services._internal.builders import ModelInitializationServiceBuilder

builder = ModelInitializationServiceBuilder()
service = builder.with_provider("openai") \
               .with_model_name("gpt-4o") \
               .with_model_parameters({"temperature": 0.7, "max_tokens": 2048}) \
               .build()
```

**Dependencies:**
- **Required**: `provider` (e.g., "openai", "anthropic", "vllm")
- **Required**: `model_name` (specific model identifier)
- **Optional**: Model parameters (temperature, max_tokens, etc.)

**Interactions:**
- Provides models to all other services requiring LLMs
- Used by `ExecutionService`, `JudgeService`, `PlannerService`

### MonitoringServiceBuilder

Provides comprehensive monitoring, tracing, and observability.

```python
from saplings.services._internal.builders import MonitoringServiceBuilder

builder = MonitoringServiceBuilder()
service = builder.with_output_dir("./monitoring") \
               .with_enabled(True) \
               .with_tracing_backend(TracingBackend.CONSOLE) \
               .with_enable_blame_graph(True) \
               .with_enable_gasa_heatmap(True) \
               .build()
```

**Dependencies:**
- **Optional**: `output_dir` (for visualization outputs)
- **Optional**: `enabled` (defaults to True)
- **Optional**: `tracing_backend` (CONSOLE, LANGSMITH, etc.)
- **Optional**: Visualization options

**Interactions:**
- Used by ALL other services for tracing and monitoring
- Provides trace managers to other builders
- Collects performance metrics across the system

### OrchestrationServiceBuilder

Coordinates complex workflows and service interactions.

```python
from saplings.services._internal.builders import OrchestrationServiceBuilder

builder = OrchestrationServiceBuilder()
service = builder.with_model(model) \
               .with_trace_manager(trace_manager) \
               .with_config(orchestration_config) \
               .build()
```

**Dependencies:**
- **Optional**: `model` (supports lazy initialization)  
- **Optional**: `trace_manager` (for monitoring)
- **Optional**: Configuration for workflow management

**Interactions:**
- Coordinates between multiple services
- Manages complex execution workflows
- Used by `AgentFacade` for high-level orchestration

### PlannerServiceBuilder

Creates and manages execution plans with budget management.

```python
from saplings.services._internal.builders import PlannerServiceBuilder

builder = PlannerServiceBuilder()
service = builder.with_model(model) \
               .with_budget_strategy("proportional") \
               .with_total_budget(5.0) \
               .with_max_steps(10) \
               .with_trace_manager(trace_manager) \
               .build()
```

**Dependencies:**
- **Optional**: `model` (supports lazy initialization)
- **Optional**: Budget and planning configuration
- **Optional**: `trace_manager` (for monitoring)

**Interactions:**
- Used by `ExecutionService` for task planning
- Works with `OrchestrationService` for complex workflows
- Provides plans to guide execution

### RetrievalServiceBuilder

Handles document search and retrieval with entropy-based filtering.

```python
from saplings.services._internal.builders import RetrievalServiceBuilder

builder = RetrievalServiceBuilder()
service = builder.with_memory_store(memory_store) \
               .with_entropy_threshold(0.1) \
               .with_max_documents(10) \
               .with_trace_manager(trace_manager) \
               .build()
```

**Dependencies:**
- **Required**: `memory_store` (document storage)
- **Optional**: Entropy and retrieval configuration
- **Optional**: `trace_manager` (for monitoring)

**Interactions:**
- Uses `MemoryManager` for document access
- Used by `ExecutionService` for context retrieval
- Provides documents to `GASAService` for attention optimization

### SelfHealingServiceBuilder

Manages error recovery, patch generation, and learning from failures.

```python
from saplings.services._internal.builders import SelfHealingServiceBuilder

builder = SelfHealingServiceBuilder()
service = builder.with_patch_generator(patch_generator) \
               .with_success_pair_collector(collector) \
               .with_enabled(True) \
               .with_trace_manager(trace_manager) \
               .build()
```

**Dependencies:**
- **Optional**: `patch_generator` (supports lazy initialization)
- **Optional**: `success_pair_collector` (for learning)
- **Optional**: `enabled` (defaults to True)
- **Optional**: `trace_manager` (for monitoring)

**Interactions:**
- Responds to failures from `ExecutionService`
- Can use `JudgeService` for quality assessment
- Learns from successful executions

### ToolServiceBuilder

Manages tool registration, execution, and sandboxing.

```python
from saplings.services._internal.builders import ToolServiceBuilder

builder = ToolServiceBuilder()
service = builder.with_executor(executor) \
               .with_allowed_imports(["os", "json"]) \
               .with_sandbox_enabled(True) \
               .with_registry(tool_registry) \
               .build()
```

**Dependencies:**
- **Optional**: `executor` (supports lazy initialization)
- **Optional**: Sandbox and security configuration
- **Optional**: `tool_registry` (custom tool registry)

**Interactions:**
- Used by `ExecutionService` for tool execution
- Can be used by multiple services requiring tool capabilities
- Manages security and sandboxing for tool execution

### ValidatorServiceBuilder

Validates outputs and responses for quality and correctness.

```python
from saplings.services._internal.builders import ValidatorServiceBuilder

builder = ValidatorServiceBuilder()
service = builder.with_model(model) \
               .with_judge_service(judge_service) \
               .with_validation_strategy(strategy) \
               .with_trace_manager(trace_manager) \
               .build()
```

**Dependencies:**
- **Optional**: `model` (for validation)
- **Optional**: `judge_service` (for quality evaluation)
- **Optional**: `validation_strategy` (custom validation logic)
- **Optional**: `trace_manager` (for monitoring)

**Interactions:**
- Used by `ExecutionService` for output validation
- Uses `JudgeService` for quality assessment
- Can trigger `SelfHealingService` on validation failures

## Builder Interactions & Dependencies

### Service Dependency Graph

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   Core Services     │    │   Processing        │    │   Quality & Safety  │
│                     │    │                     │    │                     │
│ • ModelInit         │───▶│ • Execution         │───▶│ • Validator         │
│ • MemoryManager     │    │ • Planner           │    │ • Judge             │
│ • Monitoring        │    │ • Tool              │    │ • SelfHealing       │
└─────────────────────┘    │ • Modality          │    └─────────────────────┘
          │                │ • Retrieval         │              │
          │                └─────────────────────┘              │
          │                          │                          │
          └──────────────────────────┼──────────────────────────┘
                                     │
         ┌─────────────────────────────────────────────────┐
         │              Orchestration                      │
         │                                                 │
         │ • OrchestrationService (coordinates all)       │
         │ • AgentFacade (high-level interface)           │
         └─────────────────────────────────────────────────┘
```

### Key Interactions

1. **ModelInitializationService** → Provides models to all LLM-requiring services
2. **MemoryManager** → Used by RetrievalService and ExecutionService
3. **MonitoringService** → Used by ALL services for tracing
4. **ExecutionService** → Central service using most other services
5. **ValidatorService** → Uses JudgeService, can trigger SelfHealingService
6. **OrchestrationService** → Coordinates multiple services for complex workflows

### Event-Based Communication

Services communicate through an event bus:

```python
# Services publish events
self._event_bus.publish(CoreEvent(
    event_type=CoreEventType.EXECUTION_COMPLETED,
    data={"result": result},
    source="ExecutionService"
))

# Other services subscribe to events
self._event_bus.subscribe(
    CoreEventType.VALIDATION_FAILED,
    self._handle_validation_failure
)
```

## AgentFacadeBuilder - The Orchestrator

The `AgentFacadeBuilder` orchestrates all individual service builders:

```python
from saplings._internal._agent_facade_builder import AgentFacadeBuilder

builder = AgentFacadeBuilder()
facade = builder.with_config(config) \
               .with_monitoring_service(monitoring_service) \
               .with_model_service(model_service) \
               .with_memory_manager(memory_manager) \
               .with_retrieval_service(retrieval_service) \
               .with_validator_service(validator_service) \
               .with_execution_service(execution_service) \
               .with_planner_service(planner_service) \
               .with_tool_service(tool_service) \
               .with_self_healing_service(self_healing_service) \
               .with_modality_service(modality_service) \
               .with_orchestration_service(orchestration_service) \
               .build()
```

### AgentFacadeBuilder Dependencies

The `AgentFacadeBuilder` requires ALL services to be provided:

- **Required**: `config` (AgentConfig)
- **Required**: All 11 service instances (or uses dependency injection container)

### How AgentBuilder Uses Service Builders

The `AgentBuilder` uses dependency injection to automatically create and wire services:

```python
from saplings import AgentBuilder

# AgentBuilder internally:
# 1. Configures dependency injection container
# 2. Registers all service builders with container
# 3. Creates services using builders
# 4. Wires dependencies automatically
# 5. Creates AgentFacade with all services

agent = (AgentBuilder
    .for_openai("gpt-4o")
    .with_memory_path("./memory")
    .with_tools(["PythonInterpreterTool"])
    .build())
```

### Dependency Injection Flow

```python
# 1. Container Configuration
configure_container(config)

# 2. Service Registration (using builders)
container.register(IExecutionService, 
    factory=lambda: ExecutionServiceBuilder().with_config(...).build())

# 3. Service Resolution
execution_service = container.resolve(IExecutionService)

# 4. AgentFacade Creation
facade = AgentFacade(config, execution_service=execution_service, ...)
```

## Usage Patterns

### 1. High-Level Builder Pattern (Recommended)

```python
from saplings import AgentBuilder

agent = (AgentBuilder
    .for_openai("gpt-4o")
    .with_memory_path("./memory")
    .with_tools(["WebSearchTool"])
    .build())
```

### 2. Service Builder Pattern (Advanced)

```python
from saplings.services._internal.builders import *

# Build individual services
execution_service = (ExecutionServiceBuilder()
    .with_model(model)
    .with_gasa(gasa_service)
    .build())

memory_manager = (MemoryManagerBuilder()
    .with_memory_path("./memory")
    .build())

# Create agent facade
facade = (AgentFacadeBuilder()
    .with_config(config)
    .with_execution_service(execution_service)
    .with_memory_manager(memory_manager)
    .build())
```

### 3. Dependency Injection Pattern (Expert)

```python
from saplings.api.container import container, configure_container

# Configure container
configure_container(config)

# Override specific services
container.register(IExecutionService, custom_execution_service)

# Create agent
agent = Agent(config)  # Uses container-resolved services
```

## Best Practices

### 1. Use AgentBuilder for Most Cases

The `AgentBuilder` provides the best developer experience and handles all service building automatically:

```python
# ✅ Recommended
agent = AgentBuilder.for_openai("gpt-4o").build()

# ❌ Usually unnecessary
execution_builder = ExecutionServiceBuilder()
# ... complex manual setup
```

### 2. Use Service Builders for Custom Services

When you need custom service implementations:

```python
# Create custom service
custom_execution = (ExecutionServiceBuilder()
    .with_model(custom_model)
    .with_validator(custom_validator)
    .build())

# Use in AgentFacadeBuilder
facade = (AgentFacadeBuilder()
    .with_config(config)
    .with_execution_service(custom_execution)
    .build())
```

### 3. Lazy Initialization for Performance

Many builders support lazy initialization:

```python
# Model will be created when first needed
service = (ExecutionServiceBuilder()
    .build())  # No model provided - lazy initialization

# Later, when model is needed:
service.initialize(model)
```

### 4. Error Handling

```python
try:
    service = (ExecutionServiceBuilder()
        .with_model(model)
        .build())
except InitializationError as e:
    logger.error(f"Failed to build service: {e}")
    # Handle gracefully
```

### 5. Testing with Builders

```python
# Use builders for easy test setup
test_service = (ExecutionServiceBuilder()
    .with_model(mock_model)
    .with_validator(mock_validator)
    .build())

# Test the service
result = test_service.execute(test_input)
assert result.success
```

### 6. Monitoring and Observability

Always include monitoring in production:

```python
monitoring_service = (MonitoringServiceBuilder()
    .with_enabled(True)
    .with_tracing_backend(TracingBackend.LANGSMITH)
    .build())

# Use monitoring service in other builders
execution_service = (ExecutionServiceBuilder()
    .with_trace_manager(monitoring_service.trace_manager)
    .build())
```

This comprehensive service builder architecture provides maximum flexibility while maintaining clean separation of concerns and enabling powerful features like dependency injection, lazy loading, and event-driven communication.