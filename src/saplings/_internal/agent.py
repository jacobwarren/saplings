from __future__ import annotations

"""
Agent module for Saplings.

This module provides a high-level Agent class that integrates all components
of the Saplings framework through composition and delegation to the AgentFacade.
The Agent class is the main entry point for using Saplings, providing a clean
and intuitive API.

Key features:
- Structural memory with vector and graph stores
- Cascaded, entropy-aware retrieval
- Guard-railed generation with planning and execution
- Judge and validator loop for self-improvement
- Graph-Aligned Sparse Attention (GASA) for efficient processing
- Comprehensive monitoring and tracing
- Explicit tool registration for function calling

Design principles:
- Dependency Inversion: Agent depends on interfaces, not concrete implementations
- Composition over inheritance: Uses internal AgentFacade instead of inheritance
- Single Responsibility: Each component has a clear, focused responsibility
- Interface Segregation: Clear, cohesive interfaces for each service
"""


import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from saplings.agent_config import AgentConfig

# Configure logging
logger = logging.getLogger(__name__)


class Agent:
    """
    High-level agent class that integrates all Saplings components.

    This class is implemented using composition rather than inheritance,
    delegating to an internal AgentFacade instance. This embraces the
    principle of "composition over inheritance" and allows for better
    flexibility and maintainability.

    The Agent class provides the main entry point for using Saplings with
    a clean, intuitive API that hides the complexity of the underlying
    components and their interactions.

    The Agent follows the Dependency Inversion Principle by depending on
    service interfaces rather than concrete implementations, allowing for
    easier testing, extension, and alternative implementations.

    For simplified initialization, use the AgentBuilder:
    ```python
    from saplings import AgentBuilder

    agent = AgentBuilder() \
        .with_provider("openai") \
        .with_model_name("gpt-4o") \
        .with_memory_path("./agent_memory") \
        .build()
    ```
    """

    def __init__(self, config: AgentConfig) -> None:
        """
        Initialize the agent with the provided configuration.

        For simplified initialization, use the AgentBuilder instead.

        Args:
        ----
            config: Agent configuration

        """
        # Import here to avoid circular imports
        from saplings.agent_facade import AgentFacade
        from saplings.api.container import configure_container, container
        from saplings.api.core.interfaces import (
            IExecutionService,
            IMemoryManager,
            IModalityService,
            IModelInitializationService,
            IMonitoringService,
            IOrchestrationService,
            IPlannerService,
            IRetrievalService,
            ISelfHealingService,
            IToolService,
            IValidatorService,
        )

        # Configure the container with our config
        # This ensures all services are properly registered with the correct configuration
        configure_container(config)

        # Store a reference to the container
        self._container = container

        # Resolve services from the container
        # This follows the Dependency Inversion Principle - depending on abstractions not concretions
        self._monitoring_service = container.resolve(IMonitoringService)
        self._model_service = container.resolve(IModelInitializationService)
        self._memory_manager = container.resolve(IMemoryManager)
        self._retrieval_service = container.resolve(IRetrievalService)
        self._validator_service = container.resolve(IValidatorService)
        self._execution_service = container.resolve(IExecutionService)
        self._planner_service = container.resolve(IPlannerService)
        self._tool_service = container.resolve(IToolService)
        self._self_healing_service = container.resolve(ISelfHealingService)
        self._modality_service = container.resolve(IModalityService)
        self._orchestration_service = container.resolve(IOrchestrationService)

        # Create the internal facade using the services from container
        # The facade handles the coordination between services
        self._facade = AgentFacade(
            config,
            monitoring_service=self._monitoring_service,
            model_service=self._model_service,
            memory_manager=self._memory_manager,
            retrieval_service=self._retrieval_service,
            validator_service=self._validator_service,
            execution_service=self._execution_service,
            planner_service=self._planner_service,
            tool_service=self._tool_service,
            self_healing_service=self._self_healing_service,
            modality_service=self._modality_service,
            orchestration_service=self._orchestration_service,
        )
        self.config = config

        logger.info("Agent initialized using dependency injection with interfaces")

    @classmethod
    def create(cls, provider: str, model_name: str, **kwargs) -> Agent:
        """
        Create an agent with the specified provider and model.

        This is a convenience factory method that uses the AgentBuilder internally.
        The framework automatically determines whether to use the container based on
        the requested features.

        Args:
        ----
            provider: Model provider (e.g., "openai", "anthropic", "vllm")
            model_name: Model name (e.g., "gpt-4o", "claude-3-opus")
            **kwargs: Additional configuration parameters

        Returns:
        -------
            Initialized Agent instance

        """
        # Import here to avoid circular imports
        from saplings.agent_builder import AgentBuilder

        # Create and configure the builder
        builder = AgentBuilder()
        builder.with_provider(provider)
        builder.with_model_name(model_name)

        # Apply any additional configuration
        if kwargs:
            builder.with_config(kwargs)

        # Build and return the agent with automatic container determination
        return builder.build()

    # Delegate all public methods to the facade

    async def add_document(self, content: str, metadata: dict[str, Any] | None = None):
        """Add a document to the agent's memory."""
        return await self._facade.add_document(content, metadata)

    async def execute_plan(self, plan, context=None, use_tools=True):
        """Execute a plan."""
        return await self._facade.execute_plan(plan, context, use_tools)

    def register_tool(self, tool):
        """Register a tool with the agent."""
        return self._facade.register_tool(tool)

    async def create_tool(self, name: str, description: str, code):
        """Create a dynamic tool."""
        return await self._facade.create_tool(name, description, code)

    async def judge_output(self, input_data, output_data, judgment_type="general"):
        """Judge an output using the JudgeAgent."""
        return await self._facade.judge_output(input_data, output_data, judgment_type)

    async def self_improve(self):
        """Improve the agent based on past performance."""
        return await self._facade.self_improve()

    async def run(
        self,
        task: str,
        input_modalities: list[str] | None = None,
        output_modalities: list[str] | None = None,
        use_tools: bool = True,
        skip_retrieval: bool = False,
        skip_planning: bool = False,
        skip_validation: bool = False,
        context: list[Any] | None = None,
        plan: list[Any] | None = None,
        timeout: float | None = None,
        save_results: bool = True,
    ) -> dict[str, Any] | str:
        """
        Run the agent on a task, handling the full lifecycle.

        This method orchestrates the agent workflow with optional components:
        1. Retrieve relevant context (optional)
        2. Create a plan (optional)
        3. Execute the plan or direct execution
        4. Validate and judge the results (optional)
        5. Collect success pairs for self-improvement (optional)

        Args:
        ----
            task: Task description
            input_modalities: Modalities of the input (default: ["text"])
            output_modalities: Expected modalities of the output (default: ["text"])
            use_tools: Whether to enable tool usage (default: True)
            skip_retrieval: Skip the retrieval step and use provided context (default: False)
            skip_planning: Skip the planning step and use direct execution (default: False)
            skip_validation: Skip the validation step (default: False)
            context: Pre-provided context documents (used if skip_retrieval=True)
            plan: Pre-provided plan steps (used if skip_planning=True)
            timeout: Maximum time in seconds for the entire operation (default: None)
            save_results: Whether to save results to disk (default: True)

        Returns:
        -------
            Dict[str, Any] | str: Results of the task execution. Returns a string if simple_output=True,
            otherwise returns a dictionary with detailed execution information.

        """
        result = await self._facade.run(
            task=task,
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            use_tools=use_tools,
            skip_retrieval=skip_retrieval,
            skip_planning=skip_planning,
            skip_validation=skip_validation,
            context=context,
            plan=plan,
            timeout=timeout,
            save_results=save_results,
        )

        # If the result is a dictionary with a final_result key, return that for simplicity
        if isinstance(result, dict) and "final_result" in result:
            return result["final_result"]

        return result

    async def add_documents_from_directory(self, directory: str, extension=".txt"):
        """Add documents from a directory."""
        return await self._facade.add_documents_from_directory(directory, extension)

    async def retrieve(self, query: str, limit=None, fast_mode=False):
        """
        Retrieve documents based on a query.

        Args:
        ----
            query: Query to retrieve documents
            limit: Maximum number of documents to retrieve (optional)
            fast_mode: Whether to use fast retrieval mode for better performance (optional)

        Returns:
        -------
            List of retrieved documents

        """
        return await self._facade.retrieve(query, limit, fast_mode=fast_mode)

    async def plan(self, task, context=None):
        """Create a plan for a task."""
        return await self._facade.plan(task, context)

    async def execute(self, prompt: str, context=None, use_tools=True):
        """Execute a prompt with the agent."""
        return await self._facade.execute(prompt, context, use_tools)
