# Saplings Services

This package provides services for Saplings agents.

## API Structure

The services module follows the Saplings API separation pattern:

1. **Public API**: Exposed through `saplings.api.services` and re-exported at the top level of the `saplings` package
2. **Internal Implementation**: Located in the `_internal` directory

## Usage

To use the services, import them from the public API:

```python
# Recommended: Import from the top-level package
from saplings import ExecutionService, JudgeService, ExecutionServiceBuilder

# Alternative: Import directly from the API module
from saplings.api.services import ExecutionService, JudgeService, ExecutionServiceBuilder
```

Do not import directly from the internal implementation:

```python
# Don't do this
from saplings.services._internal import ExecutionService  # Wrong
```

## Available Services

The following services are available in the public API:

- `ExecutionService`: Service for executing tasks
- `JudgeService`: Service for judging outputs
- `MemoryManager`: Service for managing memory
- `ModalityService`: Service for handling different modalities
- `OrchestrationService`: Service for orchestrating the agent workflow
- `PlannerService`: Service for planning tasks
- `RetrievalService`: Service for retrieving documents
- `SelfHealingService`: Service for self-healing
- `ToolService`: Service for managing tools
- `ValidatorService`: Service for validating outputs

## Service Builders

Each service has a corresponding builder for creating instances:

- `ExecutionServiceBuilder`: Builder for creating ExecutionService instances
- `JudgeServiceBuilder`: Builder for creating JudgeService instances
- `MemoryManagerBuilder`: Builder for creating MemoryManager instances
- `ModalityServiceBuilder`: Builder for creating ModalityService instances
- `ModelServiceBuilder`: Builder for creating ModelService instances
- `OrchestrationServiceBuilder`: Builder for creating OrchestrationService instances
- `PlannerServiceBuilder`: Builder for creating PlannerService instances
- `RetrievalServiceBuilder`: Builder for creating RetrievalService instances
- `SelfHealingServiceBuilder`: Builder for creating SelfHealingService instances
- `ToolServiceBuilder`: Builder for creating ToolService instances
- `ValidatorServiceBuilder`: Builder for creating ValidatorService instances

## GASA Services

The following GASA-related services are available:

- `GASAConfigBuilder`: Builder for creating GASAConfig instances
- `GASAServiceBuilder`: Builder for creating GASAService instances

## Builder Pattern

The services module uses the builder pattern for creating instances:

```python
from saplings import ExecutionServiceBuilder, LLM

# Create an execution service with the builder
execution_service = (ExecutionServiceBuilder()
    .with_llm(llm)
    .with_tools([tool1, tool2])
    .with_validator(validator)
    .build())
```

## Implementation Details

The service implementations are located in the `_internal` directory:

- `_internal/execution_service.py`: Implementation of the execution service
- `_internal/judge_service.py`: Implementation of the judge service
- `_internal/memory_manager.py`: Implementation of the memory manager
- `_internal/modality_service.py`: Implementation of the modality service
- `_internal/orchestration_service.py`: Implementation of the orchestration service
- `_internal/planner_service.py`: Implementation of the planner service
- `_internal/retrieval_service.py`: Implementation of the retrieval service
- `_internal/self_healing_service.py`: Implementation of the self-healing service
- `_internal/tool_service.py`: Implementation of the tool service
- `_internal/validator_service.py`: Implementation of the validator service

These internal implementations are wrapped by the public API in `saplings.api.services` to provide stability annotations and a consistent interface.
