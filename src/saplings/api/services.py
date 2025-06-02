from __future__ import annotations

"""
Services API module for Saplings.

This module provides the public API for service builders and interfaces.
"""

# Import service builders at runtime to avoid circular imports
import importlib
from typing import Any, Dict, List

from saplings.api.gasa import (
    GASAConfigBuilder as _GASAConfigBuilder,
)
from saplings.api.gasa import (
    GASAServiceBuilder as _GASAServiceBuilder,
)

# Import service implementations from the service_impl module
from saplings.api.service_impl import (
    ExecutionService as _ExecutionService,
)
from saplings.api.service_impl import (
    JudgeService as _JudgeService,
)
from saplings.api.service_impl import (
    MemoryManager as _MemoryManager,
)
from saplings.api.service_impl import (
    ModalityService as _ModalityService,
)
from saplings.api.service_impl import (
    OrchestrationService as _OrchestrationService,
)
from saplings.api.service_impl import (
    PlannerService as _PlannerService,
)
from saplings.api.service_impl import (
    RetrievalService as _RetrievalService,
)
from saplings.api.service_impl import (
    SelfHealingService as _SelfHealingService,
)
from saplings.api.service_impl import (
    ToolService as _ToolService,
)
from saplings.api.service_impl import (
    ValidatorService as _ValidatorService,
)
from saplings.api.stability import beta, stable

# Import model caching service
from saplings.services._internal.managers.model_caching_service import (
    ModelCachingService as _ModelCachingService,
)

# Import monitoring service
from saplings.services._internal.managers.monitoring_service import (
    MonitoringService as _MonitoringService,
)


# Re-export the service implementations with their public APIs
@stable
class ExecutionService(_ExecutionService):
    """
    Service for executing tasks.

    This service provides functionality for executing tasks, including
    handling context, tools, and validation.
    """


# Re-export the service builders with their public APIs
@stable
class ExecutionServiceBuilder:
    """
    Builder for creating ExecutionService instances with a fluent interface.

    This builder provides a convenient way to configure and create ExecutionService
    instances with various options and dependencies.
    """

    def __init__(self):
        """Initialize the execution service builder."""
        # Import here to avoid circular imports
        service_module = importlib.import_module("saplings.api.service")
        self._builder = service_module.ExecutionServiceBuilder()

    def with_model(self, model: Any) -> "ExecutionServiceBuilder":
        """
        Set the model for the execution service.

        Args:
        ----
            model: Model to use for execution

        Returns:
        -------
            Self for method chaining

        """
        self._builder.with_model(model)
        return self

    def with_tools(self, tools: List[Any]) -> "ExecutionServiceBuilder":
        """
        Set the tools for the execution service.

        Args:
        ----
            tools: List of tools to use for execution

        Returns:
        -------
            Self for method chaining

        """
        self._builder.with_tools(tools)
        return self

    def with_validator(self, validator: Any) -> "ExecutionServiceBuilder":
        """
        Set the validator for the execution service.

        Args:
        ----
            validator: Validator to use for execution

        Returns:
        -------
            Self for method chaining

        """
        self._builder.with_validator(validator)
        return self

    def with_config(self, config: Dict[str, Any]) -> "ExecutionServiceBuilder":
        """
        Set the configuration for the execution service.

        Args:
        ----
            config: Configuration dictionary

        Returns:
        -------
            Self for method chaining

        """
        self._builder.with_config(config)
        return self

    def build(self) -> Any:
        """
        Build the execution service instance.

        Returns
        -------
            ExecutionService instance

        """
        return self._builder.build()


@beta
class GASAConfigBuilder(_GASAConfigBuilder):
    """
    Builder for creating GASAConfig instances with a fluent interface.

    This builder provides a convenient way to configure and create GASAConfig
    instances with various options and dependencies.
    """


@beta
class GASAServiceBuilder(_GASAServiceBuilder):
    """
    Builder for creating GASAService instances with a fluent interface.

    This builder provides a convenient way to configure and create GASAService
    instances with various options and dependencies.
    """


@stable
class JudgeService(_JudgeService):
    """
    Service for judging outputs.

    This service provides functionality for judging outputs, including
    evaluating quality, correctness, and other criteria.
    """


@stable
class JudgeServiceBuilder:
    """
    Builder for creating JudgeService instances with a fluent interface.

    This builder provides a convenient way to configure and create JudgeService
    instances with various options and dependencies.
    """

    def __init__(self):
        """Initialize the judge service builder."""
        # Import here to avoid circular imports
        service_module = importlib.import_module("saplings.api.service")
        self._builder = service_module.JudgeServiceBuilder()

    def with_model(self, model: Any) -> "JudgeServiceBuilder":
        """
        Set the model for the judge service.

        Args:
        ----
            model: Model to use for judging

        Returns:
        -------
            Self for method chaining

        """
        self._builder.with_model(model)
        return self

    def with_criteria(self, criteria: List[str]) -> "JudgeServiceBuilder":
        """
        Set the criteria for the judge service.

        Args:
        ----
            criteria: List of criteria to use for judging

        Returns:
        -------
            Self for method chaining

        """
        self._builder.with_criteria(criteria)
        return self

    def with_config(self, config: Dict[str, Any]) -> "JudgeServiceBuilder":
        """
        Set the configuration for the judge service.

        Args:
        ----
            config: Configuration dictionary

        Returns:
        -------
            Self for method chaining

        """
        self._builder.with_config(config)
        return self

    def build(self) -> Any:
        """
        Build the judge service instance.

        Returns
        -------
            JudgeService instance

        """
        return self._builder.build()


@stable
class MemoryManager(_MemoryManager):
    """
    Service for managing memory.

    This service provides functionality for managing memory, including
    adding documents, retrieving documents, and managing the dependency graph.
    """


@stable
class MemoryManagerBuilder:
    """
    Builder for creating MemoryManager instances with a fluent interface.

    This builder provides a convenient way to configure and create MemoryManager
    instances with various options and dependencies.
    """

    def __init__(self):
        """Initialize the memory manager builder."""
        # Import here to avoid circular imports
        service_module = importlib.import_module("saplings.api.service")
        self._builder = service_module.MemoryManagerBuilder()

    def with_storage(self, storage: Any) -> "MemoryManagerBuilder":
        """
        Set the storage for the memory manager.

        Args:
        ----
            storage: Storage to use for memory

        Returns:
        -------
            Self for method chaining

        """
        self._builder.with_storage(storage)
        return self

    def with_indexer(self, indexer: Any) -> "MemoryManagerBuilder":
        """
        Set the indexer for the memory manager.

        Args:
        ----
            indexer: Indexer to use for memory

        Returns:
        -------
            Self for method chaining

        """
        self._builder.with_indexer(indexer)
        return self

    def with_config(self, config: Dict[str, Any]) -> "MemoryManagerBuilder":
        """
        Set the configuration for the memory manager.

        Args:
        ----
            config: Configuration dictionary

        Returns:
        -------
            Self for method chaining

        """
        self._builder.with_config(config)
        return self

    def build(self) -> Any:
        """
        Build the memory manager instance.

        Returns
        -------
            MemoryManager instance

        """
        return self._builder.build()


@beta
class ModalityService(_ModalityService):
    """
    Service for handling different modalities.

    This service provides functionality for handling different modalities,
    including text, images, audio, and video.
    """


@beta
class ModalityServiceBuilder:
    """
    Builder for creating ModalityService instances with a fluent interface.

    This builder provides a convenient way to configure and create ModalityService
    instances with various options and dependencies.
    """

    def __init__(self):
        """Initialize the modality service builder."""
        # Import here to avoid circular imports
        service_module = importlib.import_module("saplings.api.service")
        self._builder = service_module.ModalityServiceBuilder()

    def with_handler(self, modality: str, handler: Any) -> "ModalityServiceBuilder":
        """
        Set a handler for a specific modality.

        Args:
        ----
            modality: Modality type (e.g., "image", "audio")
            handler: Handler for the modality

        Returns:
        -------
            Self for method chaining

        """
        self._builder.with_handler(modality, handler)
        return self

    def with_config(self, config: Dict[str, Any]) -> "ModalityServiceBuilder":
        """
        Set the configuration for the modality service.

        Args:
        ----
            config: Configuration dictionary

        Returns:
        -------
            Self for method chaining

        """
        self._builder.with_config(config)
        return self

    def build(self) -> Any:
        """
        Build the modality service instance.

        Returns
        -------
            ModalityService instance

        """
        return self._builder.build()


@stable
class ModelServiceBuilder:
    """
    Builder for creating ModelService instances with a fluent interface.

    This builder provides a convenient way to configure and create ModelService
    instances with various options and dependencies.

    Example:
    -------
    ```python
    # Create a model service builder
    builder = ModelServiceBuilder()

    # Configure the builder with dependencies and options
    model_service = builder.with_provider("openai") \\
                          .with_model_name("gpt-4o") \\
                          .with_retry_config({
                              "max_attempts": 3,
                              "initial_backoff": 1.0,
                              "max_backoff": 30.0,
                              "backoff_factor": 2.0,
                              "jitter": True,
                          }) \\
                          .with_circuit_breaker_config({
                              "failure_threshold": 5,
                              "recovery_timeout": 60.0,
                          }) \\
                          .with_model_parameters({
                              "temperature": 0.7,
                              "max_tokens": 2048,
                          }) \\
                          .build()
    ```

    """

    def __init__(self):
        """Initialize the model service builder."""
        # Import here to avoid circular imports
        service_module = importlib.import_module("saplings.api.service")
        self._builder = service_module.ModelServiceBuilder()

    def with_provider(self, provider: str) -> "ModelServiceBuilder":
        """
        Set the provider for the model service.

        Args:
        ----
            provider: Provider name (e.g., "openai", "anthropic")

        Returns:
        -------
            Self for method chaining

        """
        self._builder.with_provider(provider)
        return self

    def with_model_name(self, model_name: str) -> "ModelServiceBuilder":
        """
        Set the model name for the model service.

        Args:
        ----
            model_name: Name of the model to use

        Returns:
        -------
            Self for method chaining

        """
        self._builder.with_model_name(model_name)
        return self

    def with_parameters(self, parameters: Dict[str, Any]) -> "ModelServiceBuilder":
        """
        Set additional parameters for the model service.

        Args:
        ----
            parameters: Additional parameters

        Returns:
        -------
            Self for method chaining

        """
        self._builder.with_parameters(parameters)
        return self

    def build(self) -> Any:
        """
        Build the model service instance.

        Returns
        -------
            Model service instance

        """
        return self._builder.build()


@stable
class ModelCachingService(_ModelCachingService):
    """
    Service for caching model responses.

    This service provides functionality for caching model responses, reducing
    costs and improving performance by avoiding redundant API calls.

    Example:
    -------
    ```python
    # Create a model caching service
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
    ```

    """


@beta
class MonitoringService(_MonitoringService):
    """
    Service for monitoring and tracing.

    This service provides functionality for monitoring and tracing, including
    creating traces, recording spans, and analyzing performance bottlenecks.

    Example:
    -------
    ```python
    # Create a monitoring service with default configuration
    monitoring_service = MonitoringService(
        output_dir="./output",
        enabled=True,
    )

    # Create a monitoring service with custom configuration
    from saplings.monitoring import MonitoringConfig, TracingBackend, VisualizationFormat

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

    monitoring_service = MonitoringService(
        config=config,
    )

    # Create a trace
    trace = monitoring_service.create_trace()

    # Start a span
    span_id = monitoring_service.start_span(
        name="execute_task",
        trace_id=trace.trace_id,
        attributes={"task": "summarize_document"},
    )

    # End the span
    monitoring_service.end_span(span_id)

    # Process the trace for analysis
    monitoring_service.process_trace(trace.trace_id)

    # Identify performance bottlenecks
    bottlenecks = monitoring_service.identify_bottlenecks(
        threshold_ms=100.0,
        min_call_count=1,
    )

    # Identify error sources
    error_sources = monitoring_service.identify_error_sources(
        min_error_rate=0.1,
        min_call_count=1,
    )
    ```

    """


@stable
class OrchestrationService(_OrchestrationService):
    """
    Service for orchestrating the agent workflow.

    This service provides functionality for orchestrating the agent workflow,
    including retrieval, planning, execution, and validation.
    """


@stable
class OrchestrationServiceBuilder:
    """
    Builder for creating OrchestrationService instances with a fluent interface.

    This builder provides a convenient way to configure and create OrchestrationService
    instances with various options and dependencies.
    """

    def __init__(self):
        """Initialize the orchestration service builder."""
        # Import here to avoid circular imports
        service_module = importlib.import_module("saplings.api.service")
        self._builder = service_module.OrchestrationServiceBuilder()

    def with_retrieval_service(self, service: Any) -> "OrchestrationServiceBuilder":
        """
        Set the retrieval service for the orchestration service.

        Args:
        ----
            service: Retrieval service to use

        Returns:
        -------
            Self for method chaining

        """
        self._builder.with_retrieval_service(service)
        return self

    def with_planner_service(self, service: Any) -> "OrchestrationServiceBuilder":
        """
        Set the planner service for the orchestration service.

        Args:
        ----
            service: Planner service to use

        Returns:
        -------
            Self for method chaining

        """
        self._builder.with_planner_service(service)
        return self

    def with_execution_service(self, service: Any) -> "OrchestrationServiceBuilder":
        """
        Set the execution service for the orchestration service.

        Args:
        ----
            service: Execution service to use

        Returns:
        -------
            Self for method chaining

        """
        self._builder.with_execution_service(service)
        return self

    def with_validator_service(self, service: Any) -> "OrchestrationServiceBuilder":
        """
        Set the validator service for the orchestration service.

        Args:
        ----
            service: Validator service to use

        Returns:
        -------
            Self for method chaining

        """
        self._builder.with_validator_service(service)
        return self

    def with_config(self, config: Dict[str, Any]) -> "OrchestrationServiceBuilder":
        """
        Set the configuration for the orchestration service.

        Args:
        ----
            config: Configuration dictionary

        Returns:
        -------
            Self for method chaining

        """
        self._builder.with_config(config)
        return self

    def build(self) -> Any:
        """
        Build the orchestration service instance.

        Returns
        -------
            OrchestrationService instance

        """
        return self._builder.build()


@stable
class PlannerService(_PlannerService):
    """
    Service for planning tasks.

    This service provides functionality for planning tasks, including
    breaking down tasks into steps and allocating budget.
    """


@stable
class PlannerServiceBuilder:
    """
    Builder for creating PlannerService instances with a fluent interface.

    This builder provides a convenient way to configure and create PlannerService
    instances with various options and dependencies.
    """

    def __init__(self):
        """Initialize the planner service builder."""
        # Import here to avoid circular imports
        service_module = importlib.import_module("saplings.api.service")
        self._builder = service_module.PlannerServiceBuilder()

    def with_model(self, model: Any) -> "PlannerServiceBuilder":
        """
        Set the model for the planner service.

        Args:
        ----
            model: Model to use for planning

        Returns:
        -------
            Self for method chaining

        """
        self._builder.with_model(model)
        return self

    def with_budget(self, budget: float) -> "PlannerServiceBuilder":
        """
        Set the budget for the planner service.

        Args:
        ----
            budget: Budget for planning in USD

        Returns:
        -------
            Self for method chaining

        """
        self._builder.with_budget(budget)
        return self

    def with_config(self, config: Dict[str, Any]) -> "PlannerServiceBuilder":
        """
        Set the configuration for the planner service.

        Args:
        ----
            config: Configuration dictionary

        Returns:
        -------
            Self for method chaining

        """
        self._builder.with_config(config)
        return self

    def build(self) -> Any:
        """
        Build the planner service instance.

        Returns
        -------
            PlannerService instance

        """
        return self._builder.build()


@stable
class RetrievalService(_RetrievalService):
    """
    Service for retrieving documents.

    This service provides functionality for retrieving documents from memory,
    including semantic search and filtering.
    """


@stable
class RetrievalServiceBuilder:
    """
    Builder for creating RetrievalService instances with a fluent interface.

    This builder provides a convenient way to configure and create RetrievalService
    instances with various options and dependencies.
    """

    def __init__(self):
        """Initialize the retrieval service builder."""
        # Import here to avoid circular imports
        service_module = importlib.import_module("saplings.api.service")
        self._builder = service_module.RetrievalServiceBuilder()

    def with_memory_manager(self, memory_manager: Any) -> "RetrievalServiceBuilder":
        """
        Set the memory manager for the retrieval service.

        Args:
        ----
            memory_manager: Memory manager to use for retrieval

        Returns:
        -------
            Self for method chaining

        """
        self._builder.with_memory_manager(memory_manager)
        return self

    def with_top_k(self, top_k: int) -> "RetrievalServiceBuilder":
        """
        Set the top_k parameter for the retrieval service.

        Args:
        ----
            top_k: Number of top results to retrieve

        Returns:
        -------
            Self for method chaining

        """
        self._builder.with_top_k(top_k)
        return self

    def with_config(self, config: Dict[str, Any]) -> "RetrievalServiceBuilder":
        """
        Set the configuration for the retrieval service.

        Args:
        ----
            config: Configuration dictionary

        Returns:
        -------
            Self for method chaining

        """
        self._builder.with_config(config)
        return self

    def build(self) -> Any:
        """
        Build the retrieval service instance.

        Returns
        -------
            RetrievalService instance

        """
        return self._builder.build()


@beta
class SelfHealingService(_SelfHealingService):
    """
    Service for self-healing.

    This service provides functionality for self-healing, including
    detecting and fixing issues, and learning from past mistakes.
    """


@beta
class SelfHealingServiceBuilder:
    """
    Builder for creating SelfHealingService instances with a fluent interface.

    This builder provides a convenient way to configure and create SelfHealingService
    instances with various options and dependencies.
    """

    def __init__(self):
        """Initialize the self-healing service builder."""
        # Import here to avoid circular imports
        service_module = importlib.import_module("saplings.api.service")
        self._builder = service_module.SelfHealingServiceBuilder()

    def with_model(self, model: Any) -> "SelfHealingServiceBuilder":
        """
        Set the model for the self-healing service.

        Args:
        ----
            model: Model to use for self-healing

        Returns:
        -------
            Self for method chaining

        """
        self._builder.with_model(model)
        return self

    def with_memory_manager(self, memory_manager: Any) -> "SelfHealingServiceBuilder":
        """
        Set the memory manager for the self-healing service.

        Args:
        ----
            memory_manager: Memory manager to use for self-healing

        Returns:
        -------
            Self for method chaining

        """
        self._builder.with_memory_manager(memory_manager)
        return self

    def with_config(self, config: Dict[str, Any]) -> "SelfHealingServiceBuilder":
        """
        Set the configuration for the self-healing service.

        Args:
        ----
            config: Configuration dictionary

        Returns:
        -------
            Self for method chaining

        """
        self._builder.with_config(config)
        return self

    def build(self) -> Any:
        """
        Build the self-healing service instance.

        Returns
        -------
            SelfHealingService instance

        """
        return self._builder.build()


@stable
class ToolService(_ToolService):
    """
    Service for managing tools.

    This service provides functionality for managing tools, including
    registering tools, creating dynamic tools, and executing tools.
    """


@stable
class ToolServiceBuilder:
    """
    Builder for creating ToolService instances with a fluent interface.

    This builder provides a convenient way to configure and create ToolService
    instances with various options and dependencies.
    """

    def __init__(self):
        """Initialize the tool service builder."""
        # Import here to avoid circular imports
        service_module = importlib.import_module("saplings.api.service")
        self._builder = service_module.ToolServiceBuilder()

    def with_tools(self, tools: List[Any]) -> "ToolServiceBuilder":
        """
        Set the tools for the tool service.

        Args:
        ----
            tools: List of tools to use

        Returns:
        -------
            Self for method chaining

        """
        self._builder.with_tools(tools)
        return self

    def with_config(self, config: Dict[str, Any]) -> "ToolServiceBuilder":
        """
        Set the configuration for the tool service.

        Args:
        ----
            config: Configuration dictionary

        Returns:
        -------
            Self for method chaining

        """
        self._builder.with_config(config)
        return self

    def build(self) -> Any:
        """
        Build the tool service instance.

        Returns
        -------
            ToolService instance

        """
        return self._builder.build()


@stable
class ValidatorService(_ValidatorService):
    """
    Service for validating outputs.

    This service provides functionality for validating outputs, including
    checking for correctness, completeness, and other criteria.
    """


@stable
class ValidatorServiceBuilder:
    """
    Builder for creating ValidatorService instances with a fluent interface.

    This builder provides a convenient way to configure and create ValidatorService
    instances with various options and dependencies.
    """

    def __init__(self):
        """Initialize the validator service builder."""
        # Import here to avoid circular imports
        service_module = importlib.import_module("saplings.api.service")
        self._builder = service_module.ValidatorServiceBuilder()

    def with_validators(self, validators: List[Any]) -> "ValidatorServiceBuilder":
        """
        Set the validators for the validator service.

        Args:
        ----
            validators: List of validators to use

        Returns:
        -------
            Self for method chaining

        """
        self._builder.with_validators(validators)
        return self

    def with_model(self, model: Any) -> "ValidatorServiceBuilder":
        """
        Set the model for the validator service.

        Args:
        ----
            model: Model to use for validation

        Returns:
        -------
            Self for method chaining

        """
        self._builder.with_model(model)
        return self

    def with_config(self, config: Dict[str, Any]) -> "ValidatorServiceBuilder":
        """
        Set the configuration for the validator service.

        Args:
        ----
            config: Configuration dictionary

        Returns:
        -------
            Self for method chaining

        """
        self._builder.with_config(config)
        return self

    def build(self) -> Any:
        """
        Build the validator service instance.

        Returns
        -------
            ValidatorService instance

        """
        return self._builder.build()


# Define the public API
__all__ = [
    "ExecutionService",
    "ExecutionServiceBuilder",
    "GASAConfigBuilder",
    "GASAServiceBuilder",
    "JudgeService",
    "JudgeServiceBuilder",
    "MemoryManager",
    "MemoryManagerBuilder",
    "ModalityService",
    "ModalityServiceBuilder",
    "ModelServiceBuilder",
    "ModelCachingService",
    "MonitoringService",
    "OrchestrationService",
    "OrchestrationServiceBuilder",
    "PlannerService",
    "PlannerServiceBuilder",
    "RetrievalService",
    "RetrievalServiceBuilder",
    "SelfHealingService",
    "SelfHealingServiceBuilder",
    "ToolService",
    "ToolServiceBuilder",
    "ValidatorService",
    "ValidatorServiceBuilder",
]
