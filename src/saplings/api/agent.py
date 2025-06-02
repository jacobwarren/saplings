"""
Agent API module for Saplings.

This module provides the public API for the Agent class and related components.
It follows the standardized API pattern using direct inheritance instead of
complex __new__ methods or dynamic imports.

The module exports the following classes:
- Agent: Main agent class for the Saplings framework
- AgentBuilder: Builder for creating Agent instances
- AgentConfig: Configuration class for Agent instances
- AgentFacade: Facade for Agent implementation (beta)
- AgentFacadeBuilder: Builder for creating AgentFacade instances (beta)

All classes use direct inheritance from internal implementations and include
proper stability annotations.
"""

from __future__ import annotations

from typing import Any

from saplings._internal._agent_facade import AgentFacade as _AgentFacade
from saplings._internal._agent_facade_builder import AgentFacadeBuilder as _AgentFacadeBuilder
from saplings._internal.agent_builder_module import AgentBuilder as _AgentBuilder

# Import internal implementations from the internal modules
# Import internal implementations directly
from saplings._internal.agent_class import Agent as _Agent
from saplings._internal.agent_config import AgentConfig as _AgentConfig

# Import stability annotations first to avoid circular imports
from saplings.api.stability import beta, stable


# Re-export the Agent class with its public API
@stable
class Agent(_Agent):
    """
    Agent class for the Saplings framework.

    This class provides a high-level interface for using Saplings agents.
    It integrates all components of the framework through composition and
    delegation to specialized services.

    The Agent class is the main entry point for using Saplings, providing
    a clean and intuitive API for running tasks, adding documents to memory,
    and managing the agent's lifecycle.

    Key Methods:
        run(task, **kwargs): Async method to execute a task and return results
        run_sync(task, **kwargs): Sync wrapper for run() method
        add_document(content, metadata): Add a document to agent's memory
        register_tool(tool): Register a tool with the agent

    Example Usage:
        # Async usage (recommended)
        agent = Agent(provider="openai", model_name="gpt-4o")
        result = await agent.run("What is 2+2?")

        # Sync usage (convenience wrapper)
        agent = Agent(provider="openai", model_name="gpt-4o")
        result = agent.run_sync("What is 2+2?")
    """

    __stability__ = "stable"

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
            agent = Agent(provider="openai", model_name="gpt-4o")
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


# Re-export the AgentBuilder class with its public API
@stable
class AgentBuilder(_AgentBuilder):
    """
    Builder for creating Agent instances with a fluent interface.

    This builder provides a convenient way to configure and create Agent
    instances with various options and dependencies.

    Example with container (default):
    ```python
    # Create a builder for Agent
    builder = AgentBuilder()

    # Configure the builder with dependencies and options
    agent = builder.with_provider("openai") \\
                  .with_model_name("gpt-4o") \\
                  .with_memory_path("./agent_memory") \\
                  .with_output_dir("./agent_output") \\
                  .with_gasa_enabled(True) \\
                  .with_monitoring_enabled(True) \\
                  .build()
    ```

    Example without container (simplified):
    ```python
    # Create a builder for Agent with simplified initialization
    builder = AgentBuilder()

    # Configure the builder with minimal options
    agent = builder.with_provider("openai") \\
                  .with_model_name("gpt-4o") \\
                  .with_tools([Calculator(), WebSearch()]) \\
                  .build(use_container=False)
    ```

    Example with advanced configuration:
    ```python
    # Create a builder with advanced configuration options
    agent = AgentBuilder() \\
        .with_provider("openai") \\
        .with_model_name("gpt-4o") \\
        .with_memory_path("./agent_memory") \\
        .with_output_dir("./agent_output") \\
        .with_gasa_enabled(True) \\
        .with_gasa_max_hops(3) \\
        .with_gasa_strategy("binary") \\
        .with_gasa_fallback("block_diagonal") \\
        .with_gasa_shadow_model(True) \\
        .with_gasa_shadow_model_name("Qwen/Qwen3-0.6B") \\
        .with_gasa_prompt_composer(True) \\
        .with_monitoring_enabled(True) \\
        .with_self_healing_enabled(True) \\
        .with_self_healing_max_retries(3) \\
        .with_tool_factory_enabled(True) \\
        .with_tool_factory_sandbox_enabled(True) \\
        .with_allowed_imports(["os", "json", "re", "math"]) \\
        .with_retrieval_entropy_threshold(0.1) \\
        .with_retrieval_max_documents(10) \\
        .with_planner_budget_strategy("token_count") \\
        .with_planner_total_budget(1.0) \\
        .with_planner_allow_budget_overflow(False) \\
        .with_planner_budget_overflow_margin(0.1) \\
        .with_executor_validation_type("execution") \\
        .with_model_parameters({"temperature": 0.7, "max_tokens": 2048}) \\
        .build()
    ```

    Factory Methods:
    ```python
    # Create an agent with minimal configuration
    agent = AgentBuilder.minimal("openai", "gpt-4o").build()

    # Create an agent with standard configuration
    agent = AgentBuilder.standard("openai", "gpt-4o").build()

    # Create an agent with full-featured configuration
    agent = AgentBuilder.full_featured("openai", "gpt-4o").build()

    # Create an agent optimized for OpenAI
    agent = AgentBuilder.for_openai("gpt-4o").build()

    # Create an agent optimized for Anthropic
    agent = AgentBuilder.for_anthropic("claude-3-opus").build()

    # Create an agent optimized for vLLM
    agent = AgentBuilder.for_vllm("Qwen/Qwen3-7B-Instruct").build()
    ```
    """

    __stability__ = "stable"


# Re-export the AgentConfig class with its public API
@stable
class AgentConfig(_AgentConfig):
    """
    Configuration class for Agent instances.

    This class provides a structured way to configure Agent instances
    with various options and dependencies.

    Args:
    ----
        provider: The model provider (e.g., "openai", "anthropic", "vllm")
        model_name: The name of the model to use
        memory_path: Path to the memory store directory
        output_dir: Path to the output directory
        enable_gasa: Whether to enable Graph-Aligned Sparse Attention
        enable_monitoring: Whether to enable monitoring
        enable_self_healing: Whether to enable self-healing
        self_healing_max_retries: Maximum number of retry attempts for self-healing operations
        enable_tool_factory: Whether to enable the tool factory
        max_tokens: Maximum number of tokens for model responses
        temperature: Temperature for model generation
        gasa_max_hops: Maximum number of hops for GASA mask
        gasa_strategy: Strategy for GASA mask (binary, soft, learned)
        gasa_fallback: Fallback strategy for GASA (block_diagonal, prompt_composer)
        gasa_shadow_model: Whether to use a shadow model for tokenization
        gasa_shadow_model_name: Name of the shadow model to use
        gasa_prompt_composer: Whether to use prompt composer for GASA
        retrieval_entropy_threshold: Entropy threshold for retrieval
        retrieval_max_documents: Maximum number of documents to retrieve
        planner_budget_strategy: Strategy for allocating budget to tasks (token_count, fixed, dynamic)
        planner_total_budget: Total budget for planning
        planner_allow_budget_overflow: Whether to allow budget overflow
        planner_budget_overflow_margin: Margin for budget overflow
        executor_validation_type: Type of validation to use for execution (basic, execution, judge)
        tool_factory_sandbox_enabled: Whether to enable sandboxing for tools
        allowed_imports: List of allowed imports for tools
        tools: List of tools to register with the agent
        supported_modalities: List of supported modalities
        model_parameters: Additional parameters for the model

    Factory Methods:
        minimal(provider, model_name, **kwargs): Create a minimal configuration
        standard(provider, model_name, **kwargs): Create a standard configuration
        full_featured(provider, model_name, **kwargs): Create a full-featured configuration
        for_openai(model_name, **kwargs): Create a configuration optimized for OpenAI
        for_anthropic(model_name, **kwargs): Create a configuration optimized for Anthropic
        for_vllm(model_name, **kwargs): Create a configuration optimized for vLLM

    """

    __stability__ = "stable"


# Re-export the AgentFacade class with its public API
@beta
class AgentFacade(_AgentFacade):
    """
    Facade for Agent implementation.

    This class provides a facade for the Agent implementation, exposing
    a simplified interface for using Saplings agents. The facade delegates
    to specialized service interfaces for each concern:

    - Memory (IMemoryManager)
    - Retrieval (IRetrievalService)
    - Planning (IPlannerService)
    - Execution (IExecutionService)
    - Validation (IValidatorService)
    - Self-healing (ISelfHealingService)
    - Tool management (IToolService)
    - Modality handling (IModalityService)
    - Monitoring (IMonitoringService)
    - Model management (IModelService)
    - Orchestration (IOrchestrationService)

    Methods
    -------
        add_document(content, metadata): Add a document to the agent's memory
        add_documents_from_directory(directory, extension): Add documents from a directory
        execute_plan(plan, context, use_tools): Execute a plan
        register_tool(tool): Register a tool with the agent
        create_tool(name, description, code): Create a dynamic tool
        judge_output(input_data, output_data, judgment_type): Judge an output
        self_improve(): Improve the agent based on past performance
        run(task, ...): Run the agent on a task, handling the full lifecycle
        retrieve(query, limit, fast_mode): Retrieve documents based on a query
        plan(task, context): Create a plan for a task
        execute(prompt, context, use_tools): Execute a prompt with the agent

    Note: This is a beta API and may change in future versions.

    """

    __stability__ = "beta"


# Re-export the AgentFacadeBuilder class with its public API
@beta
class AgentFacadeBuilder(_AgentFacadeBuilder):
    """
    Builder for creating AgentFacade instances with a fluent interface.

    This builder provides a convenient way to configure and create AgentFacade
    instances with various options and dependencies. It allows for direct
    customization of all services used by the AgentFacade.

    Example:
    -------
    ```python
    # Create a builder for AgentFacade
    builder = AgentFacadeBuilder()

    # Configure the builder with dependencies and options
    facade = builder.with_config(config) \\
                   .with_monitoring_service(monitoring_service) \\
                   .with_model_service(model_service) \\
                   .with_memory_manager(memory_manager) \\
                   .with_retrieval_service(retrieval_service) \\
                   .with_validator_service(validator_service) \\
                   .with_execution_service(execution_service) \\
                   .with_planner_service(planner_service) \\
                   .with_tool_service(tool_service) \\
                   .with_self_healing_service(self_healing_service) \\
                   .with_modality_service(modality_service) \\
                   .with_orchestration_service(orchestration_service) \\
                   .build()
    ```

    Methods:
    -------
        with_config(config): Set the agent configuration
        with_monitoring_service(service): Set the monitoring service
        with_model_service(service): Set the model service
        with_memory_manager(service): Set the memory manager
        with_retrieval_service(service): Set the retrieval service
        with_validator_service(service): Set the validator service
        with_execution_service(service): Set the execution service
        with_planner_service(service): Set the planner service
        with_tool_service(service): Set the tool service
        with_self_healing_service(service): Set the self-healing service
        with_modality_service(service): Set the modality service
        with_orchestration_service(service): Set the orchestration service
        build(): Build the AgentFacade instance

    Note: This is a beta API and may change in future versions.

    """

    __stability__ = "beta"


# Define what should be exported from this module
__all__ = [
    "Agent",
    "AgentBuilder",
    "AgentConfig",
    "AgentFacade",
    "AgentFacadeBuilder",
]
