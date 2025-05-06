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

from saplings.container import SaplingsContainer

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
    """

    def __init__(self, config: AgentConfig) -> None:
        """
        Initialize the agent with the provided configuration.

        Args:
        ----
            config: Agent configuration

        """
        # Import here to avoid circular imports
        from saplings.agent_facade import AgentFacade

        # Initialize the container with the configuration
        self._container = SaplingsContainer()
        self._container.config.override(config)

        # Get services from the container using interfaces
        # This follows the Dependency Inversion Principle - depending on abstractions not concretions
        self._monitoring_service = self._container.get_monitoring_service()
        self._model_service = self._container.get_model_service()
        self._memory_manager = self._container.get_memory_manager()
        self._retrieval_service = self._container.get_retrieval_service()
        self._validator_service = self._container.get_validator_service()
        self._execution_service = self._container.get_execution_service()
        self._planner_service = self._container.get_planner_service()
        self._tool_service = self._container.get_tool_service()
        self._self_healing_service = self._container.get_self_healing_service()
        self._modality_service = self._container.get_modality_service()
        self._orchestration_service = self._container.get_orchestration_service()

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

    async def run(self, task, input_modalities=None, output_modalities=None, use_tools=True):
        """Run the agent on a task, handling the full lifecycle."""
        return await self._facade.run(task, input_modalities, output_modalities, use_tools)

    async def add_documents_from_directory(self, directory: str, extension=".txt"):
        """Add documents from a directory."""
        return await self._facade.add_documents_from_directory(directory, extension)

    async def retrieve(self, query: str, limit=None):
        """Retrieve documents based on a query."""
        return await self._facade.retrieve(query, limit)

    async def plan(self, task, context=None):
        """Create a plan for a task."""
        return await self._facade.plan(task, context)

    async def execute(self, prompt: str, context=None, use_tools=True):
        """Execute a prompt with the agent."""
        return await self._facade.execute(prompt, context, use_tools)
