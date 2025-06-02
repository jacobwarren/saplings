from __future__ import annotations

"""
Service API module for Saplings.

This module provides the public API for service interfaces and builders.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from saplings.api.stability import beta, stable


# Base Service Builder
@stable
class ServiceBuilder(ABC):
    """
    Base class for service builders.

    This class provides a fluent interface for building service instances
    with proper configuration and dependency injection.
    """

    def __init__(self):
        """Initialize the service builder."""
        self._config = {}

    def with_config(self, config: Dict[str, Any]) -> "ServiceBuilder":
        """
        Set the configuration for the service.

        Args:
        ----
            config: Configuration dictionary

        Returns:
        -------
            Self for method chaining

        """
        self._config.update(config)
        return self

    @abstractmethod
    def build(self) -> Any:
        """
        Build the service instance.

        Returns
        -------
            Service instance

        """


# Execution Service Builder
@stable
class ExecutionServiceBuilder(ServiceBuilder):
    """
    Builder for creating ExecutionService instances with a fluent interface.

    This builder provides a convenient way to configure and create ExecutionService
    instances with various options and dependencies.
    """

    def __init__(self):
        """Initialize the execution service builder."""
        super().__init__()
        self._model = None
        self._tools = []
        self._validator = None

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
        self._model = model
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
        self._tools = tools
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
        self._validator = validator
        return self

    def build(self) -> Any:
        """
        Build the execution service instance.

        Returns
        -------
            ExecutionService instance

        """

        # Use a factory function to avoid circular imports
        def create_service():
            from saplings.api.services import ExecutionService

            return ExecutionService(
                model=self._model, tools=self._tools, validator=self._validator, **self._config
            )

        return create_service()


# Judge Service Builder
@stable
class JudgeServiceBuilder(ServiceBuilder):
    """
    Builder for creating JudgeService instances with a fluent interface.

    This builder provides a convenient way to configure and create JudgeService
    instances with various options and dependencies.
    """

    def __init__(self):
        """Initialize the judge service builder."""
        super().__init__()
        self._model = None
        self._criteria = []

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
        self._model = model
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
        self._criteria = criteria
        return self

    def build(self) -> Any:
        """
        Build the judge service instance.

        Returns
        -------
            JudgeService instance

        """

        # Use a factory function to avoid circular imports
        def create_service():
            from saplings.api.services import JudgeService

            return JudgeService(model=self._model, criteria=self._criteria, **self._config)

        return create_service()


# Memory Manager Builder
@stable
class MemoryManagerBuilder(ServiceBuilder):
    """
    Builder for creating MemoryManager instances with a fluent interface.

    This builder provides a convenient way to configure and create MemoryManager
    instances with various options and dependencies.
    """

    def __init__(self):
        """Initialize the memory manager builder."""
        super().__init__()
        self._storage = None
        self._indexer = None

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
        self._storage = storage
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
        self._indexer = indexer
        return self

    def build(self) -> Any:
        """
        Build the memory manager instance.

        Returns
        -------
            MemoryManager instance

        """

        # Use a factory function to avoid circular imports
        def create_service():
            from saplings.api.services import MemoryManager

            return MemoryManager(storage=self._storage, indexer=self._indexer, **self._config)

        return create_service()


# Modality Service Builder
@beta
class ModalityServiceBuilder(ServiceBuilder):
    """
    Builder for creating ModalityService instances with a fluent interface.

    This builder provides a convenient way to configure and create ModalityService
    instances with various options and dependencies.
    """

    def __init__(self):
        """Initialize the modality service builder."""
        super().__init__()
        self._handlers = {}

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
        self._handlers[modality] = handler
        return self

    def build(self) -> Any:
        """
        Build the modality service instance.

        Returns
        -------
            ModalityService instance

        """

        # Use a factory function to avoid circular imports
        def create_service():
            from saplings.api.services import ModalityService

            return ModalityService(handlers=self._handlers, **self._config)

        return create_service()


# Model Service Builder
@stable
class ModelServiceBuilder(ServiceBuilder):
    """
    Builder for creating ModelService instances with a fluent interface.

    This builder provides a convenient way to configure and create ModelService
    instances with various options and dependencies.
    """

    def __init__(self):
        """Initialize the model service builder."""
        super().__init__()
        self._provider = None
        self._model_name = None

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
        self._provider = provider
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
        self._model_name = model_name
        return self

    def build(self) -> Any:
        """
        Build the model service instance.

        Returns
        -------
            Model instance

        """
        from saplings.api.models import LLMBuilder

        builder = LLMBuilder()

        if self._provider:
            builder.with_provider(self._provider)

        if self._model_name:
            builder.with_model_name(self._model_name)

        # Add any additional configuration
        if self._config:
            builder.with_parameters(self._config)

        return builder.build()


# Orchestration Service Builder
@stable
class OrchestrationServiceBuilder(ServiceBuilder):
    """
    Builder for creating OrchestrationService instances with a fluent interface.

    This builder provides a convenient way to configure and create OrchestrationService
    instances with various options and dependencies.
    """

    def __init__(self):
        """Initialize the orchestration service builder."""
        super().__init__()
        self._retrieval_service = None
        self._planner_service = None
        self._execution_service = None
        self._validator_service = None

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
        self._retrieval_service = service
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
        self._planner_service = service
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
        self._execution_service = service
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
        self._validator_service = service
        return self

    def build(self) -> Any:
        """
        Build the orchestration service instance.

        Returns
        -------
            OrchestrationService instance

        """

        # Use a factory function to avoid circular imports
        def create_service():
            from saplings.api.services import OrchestrationService

            return OrchestrationService(
                retrieval_service=self._retrieval_service,
                planner_service=self._planner_service,
                execution_service=self._execution_service,
                validator_service=self._validator_service,
                **self._config,
            )

        return create_service()


# Planner Service Builder
@stable
class PlannerServiceBuilder(ServiceBuilder):
    """
    Builder for creating PlannerService instances with a fluent interface.

    This builder provides a convenient way to configure and create PlannerService
    instances with various options and dependencies.
    """

    def __init__(self):
        """Initialize the planner service builder."""
        super().__init__()
        self._model = None
        self._budget = None

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
        self._model = model
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
        self._budget = budget
        return self

    def build(self) -> Any:
        """
        Build the planner service instance.

        Returns
        -------
            PlannerService instance

        """

        # Use a factory function to avoid circular imports
        def create_service():
            from saplings.api.services import PlannerService

            return PlannerService(model=self._model, budget=self._budget, **self._config)

        return create_service()


# Retrieval Service Builder
@stable
class RetrievalServiceBuilder(ServiceBuilder):
    """
    Builder for creating RetrievalService instances with a fluent interface.

    This builder provides a convenient way to configure and create RetrievalService
    instances with various options and dependencies.
    """

    def __init__(self):
        """Initialize the retrieval service builder."""
        super().__init__()
        self._memory_manager = None
        self._top_k = 5

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
        self._memory_manager = memory_manager
        return self

    def with_top_k(self, top_k: int) -> "RetrievalServiceBuilder":
        """
        Set the top_k parameter for the retrieval service.

        Args:
        ----
            top_k: Number of results to retrieve

        Returns:
        -------
            Self for method chaining

        """
        self._top_k = top_k
        return self

    def build(self) -> Any:
        """
        Build the retrieval service instance.

        Returns
        -------
            RetrievalService instance

        """

        # Use a factory function to avoid circular imports
        def create_service():
            from saplings.api.services import RetrievalService

            return RetrievalService(
                memory_manager=self._memory_manager, top_k=self._top_k, **self._config
            )

        return create_service()


# Self-Healing Service Builder
@beta
class SelfHealingServiceBuilder(ServiceBuilder):
    """
    Builder for creating SelfHealingService instances with a fluent interface.

    This builder provides a convenient way to configure and create SelfHealingService
    instances with various options and dependencies.
    """

    def __init__(self):
        """Initialize the self-healing service builder."""
        super().__init__()
        self._model = None
        self._memory_manager = None

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
        self._model = model
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
        self._memory_manager = memory_manager
        return self

    def build(self) -> Any:
        """
        Build the self-healing service instance.

        Returns
        -------
            SelfHealingService instance

        """

        # Use a factory function to avoid circular imports
        def create_service():
            from saplings.api.services import SelfHealingService

            return SelfHealingService(
                model=self._model, memory_manager=self._memory_manager, **self._config
            )

        return create_service()


# Tool Service Builder
@stable
class ToolServiceBuilder(ServiceBuilder):
    """
    Builder for creating ToolService instances with a fluent interface.

    This builder provides a convenient way to configure and create ToolService
    instances with various options and dependencies.
    """

    def __init__(self):
        """Initialize the tool service builder."""
        super().__init__()
        self._tools = []

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
        self._tools = tools
        return self

    def with_tool(self, tool: Any) -> "ToolServiceBuilder":
        """
        Add a tool to the tool service.

        Args:
        ----
            tool: Tool to add

        Returns:
        -------
            Self for method chaining

        """
        self._tools.append(tool)
        return self

    def build(self) -> Any:
        """
        Build the tool service instance.

        Returns
        -------
            ToolService instance

        """

        # Use a factory function to avoid circular imports
        def create_service():
            from saplings.api.services import ToolService

            return ToolService(tools=self._tools, **self._config)

        return create_service()


# Validator Service Builder
@stable
class ValidatorServiceBuilder(ServiceBuilder):
    """
    Builder for creating ValidatorService instances with a fluent interface.

    This builder provides a convenient way to configure and create ValidatorService
    instances with various options and dependencies.
    """

    def __init__(self):
        """Initialize the validator service builder."""
        super().__init__()
        self._validators = []
        self._model = None

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
        self._validators = validators
        return self

    def with_validator(self, validator: Any) -> "ValidatorServiceBuilder":
        """
        Add a validator to the validator service.

        Args:
        ----
            validator: Validator to add

        Returns:
        -------
            Self for method chaining

        """
        self._validators.append(validator)
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
        self._model = model
        return self

    def build(self) -> Any:
        """
        Build the validator service instance.

        Returns
        -------
            ValidatorService instance

        """

        # Use a factory function to avoid circular imports
        def create_service():
            from saplings.api.services import ValidatorService

            return ValidatorService(validators=self._validators, model=self._model, **self._config)

        return create_service()


__all__ = [
    # Base builder
    "ServiceBuilder",
    # Service builders
    "ExecutionServiceBuilder",
    "JudgeServiceBuilder",
    "MemoryManagerBuilder",
    "ModalityServiceBuilder",
    "ModelServiceBuilder",
    "OrchestrationServiceBuilder",
    "PlannerServiceBuilder",
    "RetrievalServiceBuilder",
    "SelfHealingServiceBuilder",
    "ToolServiceBuilder",
    "ValidatorServiceBuilder",
]
