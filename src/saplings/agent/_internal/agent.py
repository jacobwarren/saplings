from __future__ import annotations

"""
Agent module.

This module provides the Agent class for the Saplings framework.
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from saplings.agent._internal.types import AgentProtocol

# Configure logging
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from saplings.api.tools import Tool


class Agent(AgentProtocol):
    """
    Agent class for the Saplings framework.

    This class provides a high-level interface for using Saplings agents.
    It integrates all components of the framework through composition and
    delegation to specialized services.
    """

    def __init__(self, config: Any) -> None:
        """
        Initialize the agent with the provided configuration.

        For simplified initialization, use the AgentBuilder instead.

        Args:
        ----
            config: Agent configuration

        """
        # Import here to avoid circular imports
        from saplings.api.container import configure_container

        # Configure the container
        configure_container(config)

        # Store the config
        self.config = config

        # Create the facade using a factory function to avoid circular imports
        self._facade = self._create_facade(config)

    def _create_facade(self, config: Any) -> Any:
        """
        Create an AgentFacade instance.

        This method uses a factory approach to avoid circular imports.

        Args:
        ----
            config: Agent configuration

        Returns:
        -------
            AgentFacade: The created facade instance

        """
        # Use dynamic import to avoid circular imports
        import importlib

        # Import the builder dynamically
        facade_builder_module = importlib.import_module(
            "saplings.agent._internal.agent_facade_builder"
        )
        AgentFacadeBuilder = facade_builder_module.AgentFacadeBuilder

        # Create and return the facade
        return AgentFacadeBuilder().with_config(config).build()

    # Delegate all public methods to the facade
    async def add_document(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Any:
        """Add a document to the agent's memory."""
        return await self._facade.add_document(content, metadata)

    async def add_documents_from_directory(self, directory: str, extension: str = ".txt") -> Any:
        """Add documents from a directory."""
        # Delegate to memory manager through the facade
        memory_manager = getattr(self._facade, "_memory_manager", None)
        if memory_manager:
            return await memory_manager.add_documents_from_directory(directory, extension)
        logger.warning("Memory manager not available, cannot add documents from directory")
        return []

    async def run(self, task: str, **kwargs) -> Any:
        """Run a task with the agent."""
        return await self._facade.run(task, **kwargs)

    async def add_tool(self, tool: "Tool") -> None:
        """Add a tool to the agent."""
        return await self._facade.add_tool(tool)

    async def create_tool(self, name: str, description: str, code: str) -> Any:
        """Create a new tool dynamically."""
        return await self._facade.create_tool(name, description, code)

    async def retrieve(
        self, query: str, limit: Optional[int] = None, fast_mode: bool = False
    ) -> Any:
        """Retrieve documents based on a query."""
        # Delegate to retrieval service through the facade
        retrieval_service = getattr(self._facade, "_retrieval_service", None)
        if retrieval_service:
            # Pass fast_mode as a timeout parameter if supported
            timeout = 1.0 if fast_mode else None
            return await retrieval_service.retrieve(query, limit=limit, timeout=timeout)
        logger.warning("Retrieval service not available, cannot retrieve documents")
        return []

    async def plan(self, task: str, context: Optional[Any] = None) -> Any:
        """Create a plan for a task."""
        # Delegate to planner service through the facade
        planner_service = getattr(self._facade, "_planner_service", None)
        if planner_service:
            return await planner_service.create_plan(task, context)
        logger.warning("Planner service not available, cannot create plan")
        return []

    async def execute(
        self, prompt: str, context: Optional[Any] = None, use_tools: bool = True
    ) -> Any:
        """Execute a prompt with the agent."""
        # Delegate to execution service through the facade
        execution_service = getattr(self._facade, "_execution_service", None)
        if execution_service:
            functions = None
            if use_tools:
                # Try to get functions from the facade's run method
                try:
                    # This is a safer approach that doesn't rely on internal implementation details
                    # We're using the facade's existing functionality to get the functions
                    run_result = await self._facade.run(
                        f"Get functions for: {prompt}",
                        skip_retrieval=True,
                        skip_planning=True,
                        skip_validation=True,
                        use_tools=True,
                        _internal_get_functions_only=True,  # Special flag to just return functions
                    )
                    if isinstance(run_result, dict) and "functions" in run_result:
                        functions = run_result["functions"]
                except Exception as e:
                    logger.warning(f"Failed to get functions from facade: {e}")

            return await execution_service.execute(prompt, context, functions=functions)
        logger.warning("Execution service not available, cannot execute prompt")
        return {"result": "Execution service not available"}

    @classmethod
    def create(cls, provider: str, model_name: str, **kwargs):
        """
        Create an agent with the specified provider and model.

        This is a convenience method for creating an agent with minimal configuration.

        Args:
        ----
            provider: The model provider (e.g., "openai", "anthropic", "vllm")
            model_name: The name of the model to use
            **kwargs: Additional configuration options

        Returns:
        -------
            Initialized Agent instance

        """
        # Use a factory function to create the builder to avoid circular imports
        builder = cls._create_builder()

        # Configure the builder
        builder.with_provider(provider)
        builder.with_model_name(model_name)

        # Apply any additional configuration
        for key, value in kwargs.items():
            # Convert snake_case to camelCase for method names
            method_name = f"with_{key}"
            if hasattr(builder, method_name) and callable(getattr(builder, method_name)):
                getattr(builder, method_name)(value)

        # Build and return the agent
        return builder.build()

    @staticmethod
    def _create_builder():
        """
        Create an AgentBuilder instance.

        This method uses a factory approach to avoid circular imports.

        Returns
        -------
            AgentBuilder: The created builder instance

        """
        # Import here to avoid circular imports
        from saplings.agent._internal.agent_builder import AgentBuilder

        return AgentBuilder()
