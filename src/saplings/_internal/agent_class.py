from __future__ import annotations

"""
Agent class module for Saplings.

This module provides the Agent class implementation, which is the main entry point
for using Saplings. It integrates all components of the framework through composition
and delegation to specialized services.
"""

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from saplings._internal.agent_module import AgentConfig

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
    """

    def __init__(self, config: "AgentConfig | None" = None, **kwargs) -> None:
        """
        Initialize the agent with the provided configuration.

        Supports both explicit config object and simplified keyword arguments.

        Args:
        ----
            config: Agent configuration (optional if kwargs provided)
            **kwargs: Simplified configuration parameters (provider, model_name, etc.)

        Examples:
        --------
            # Simplified creation with keyword arguments
            agent = Agent(provider="openai", model_name="gpt-4o")

            # Traditional creation with config object
            config = AgentConfig(provider="openai", model_name="gpt-4o")
            agent = Agent(config=config)

        """
        # Initialize model factory if not already done
        self._ensure_model_factory_initialized()

        # Handle both config object and kwargs patterns
        if config is None and kwargs:
            # Create config from kwargs (simplified pattern)
            from saplings._internal.agent_module import AgentConfig

            config = AgentConfig(**kwargs)
        elif config is None:
            # Neither config nor kwargs provided
            raise ValueError(
                "Agent requires either a config object or keyword arguments. "
                "Examples:\n"
                "  Agent(provider='openai', model_name='gpt-4o')\n"
                "  Agent(config=AgentConfig(provider='openai', model_name='gpt-4o'))"
            )

        # Import here to avoid circular imports
        from saplings._internal.agent_facade import AgentFacade
        from saplings._internal.di import container

        # Import interfaces lazily to avoid circular imports
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
        # Force reset the container configuration state to ensure services are registered
        # This is needed because pytest may reset the container between tests
        from saplings.di import configure_container, reset_container_config

        reset_container_config()

        # Configure the container with services
        # Try configure_container first, but fall back to direct service configuration if needed
        try:
            configure_container(config)
        except Exception:
            # Fallback: configure services directly
            from saplings._internal.container_config import configure_services

            configure_services(config)

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

        # Track initialization state
        self._initialized = False

        logger.info("Agent initialized using dependency injection with interfaces")

    async def _initialize_async(self) -> None:
        """
        Initialize the agent asynchronously.

        This method handles the async initialization of services, particularly
        initializing the model and any services that depend on it.
        """
        if self._initialized:
            return
            
        # Initialize the model in the facade, which will also initialize
        # the ExecutionService with the model
        await self._facade.init_model()
        self._initialized = True
        logger.info("Agent async initialization completed")

    def _ensure_model_factory_initialized(self) -> None:
        """
        Ensure the model factory is initialized.

        This method checks if the model factory is set and initializes it if needed.
        """
        try:
            # Check if factory is already set by trying to create a dummy model
            from saplings.models._internal.interfaces import LLM

            if LLM._factory is None:
                # Initialize the model factory
                from saplings.models._internal.model_adapter import initialize_model_factory

                initialize_model_factory()
                logger.debug("Model factory initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize model factory: {e}")
            # Don't raise here as the agent might still work without models

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
        from saplings._internal.agent_builder_module import AgentBuilder

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

        This is an async method that must be awaited. For synchronous usage,
        use the run_sync() method instead.

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

        Example:
        -------
            # Async usage (recommended)
            agent = Agent(provider="openai", model_name="gpt-4o")
            result = await agent.run("What is 2+2?")

            # For sync usage, use run_sync() instead
            result = agent.run_sync("What is 2+2?")

        Note:
        ----
            This method is async and must be awaited. If you need synchronous execution,
            use the run_sync() method which handles the async execution internally.

        """
        # Ensure the agent is properly initialized
        await self._initialize_async()
        
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

    def run_sync(
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
        Synchronous wrapper for the run() method.

        This method provides a convenient synchronous interface for the async run() method.
        It handles the async execution internally using asyncio.

        Args:
        ----
            task: The task to execute
            input_modalities: Optional list of input modalities
            output_modalities: Optional list of output modalities
            use_tools: Whether to use tools during execution
            skip_retrieval: Whether to skip retrieval phase
            skip_planning: Whether to skip planning phase
            skip_validation: Whether to skip validation phase
            context: Optional context for the task
            plan: Optional pre-defined plan
            timeout: Optional timeout in seconds
            save_results: Whether to save results

        Returns:
        -------
            The result of the task execution (string or dict)

        Raises:
        ------
            RuntimeError: If called from within an async context
            Exception: Any exception raised by the underlying async method

        Example:
        -------
            agent = Agent(config)
            result = agent.run_sync("What is the capital of France?")
            print(result)  # "The capital of France is Paris."

        """
        import asyncio

        try:
            # Check if we're already in an async context
            loop = asyncio.get_running_loop()
            if loop.is_running():
                raise RuntimeError(
                    "Cannot call run_sync() from async context. Use 'await agent.run()' instead."
                )
        except RuntimeError:
            # No running loop, which is what we want for sync execution
            pass

        # Run the async method in a new event loop
        return asyncio.run(
            self.run(
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
        )

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
