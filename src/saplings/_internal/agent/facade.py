from __future__ import annotations

"""
Agent facade module for Saplings.

This module provides the AgentFacade class, which is the internal implementation
of the Agent class. It handles the coordination between services and provides
the core functionality of the Agent.
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from saplings._internal.agent.config import AgentConfig
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

# Configure logging
logger = logging.getLogger(__name__)


class AgentFacade:
    """
    Facade for Agent implementation.

    This class provides the internal implementation of the Agent class,
    handling the coordination between services and providing the core
    functionality of the Agent.
    """

    def __init__(
        self,
        config: "AgentConfig",
        monitoring_service: "IMonitoringService",
        model_service: "IModelInitializationService",
        memory_manager: "IMemoryManager",
        retrieval_service: "IRetrievalService",
        validator_service: "IValidatorService",
        execution_service: "IExecutionService",
        planner_service: "IPlannerService",
        tool_service: "IToolService",
        self_healing_service: "ISelfHealingService",
        modality_service: "IModalityService",
        orchestration_service: "IOrchestrationService",
    ) -> None:
        """
        Initialize the agent facade with the provided services.

        Args:
        ----
            config: Agent configuration
            monitoring_service: Service for monitoring and tracing
            model_service: Service for model initialization
            memory_manager: Service for memory management
            retrieval_service: Service for document retrieval
            validator_service: Service for validation
            execution_service: Service for execution
            planner_service: Service for planning
            tool_service: Service for tool management
            self_healing_service: Service for self-healing
            modality_service: Service for modality handling
            orchestration_service: Service for orchestration

        """
        self.config = config
        self._monitoring_service = monitoring_service
        self._model_service = model_service
        self._memory_manager = memory_manager
        self._retrieval_service = retrieval_service
        self._validator_service = validator_service
        self._execution_service = execution_service
        self._planner_service = planner_service
        self._tool_service = tool_service
        self._self_healing_service = self_healing_service
        self._modality_service = modality_service
        self._orchestration_service = orchestration_service
        self._model = self._model_service.model

        logger.info("AgentFacade initialized with all services")

    async def add_document(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a document to the agent's memory."""
        return await self._memory_manager.add_document(content, metadata)

    async def execute_plan(self, plan, context=None, use_tools=True):
        """Execute a plan."""
        return await self._execution_service.execute_plan(plan, context, use_tools)

    def register_tool(self, tool):
        """Register a tool with the agent."""
        return self._tool_service.register_tool(tool)

    async def create_tool(self, name: str, description: str, code):
        """Create a dynamic tool."""
        return await self._tool_service.create_tool(name, description, code)

    async def judge_output(self, input_data, output_data, judgment_type="general"):
        """Judge an output using the JudgeAgent."""
        return await self._validator_service.judge_output(input_data, output_data, judgment_type)

    async def self_improve(self):
        """Improve the agent based on past performance."""
        return await self._self_healing_service.self_improve()

    async def run(
        self,
        task: str,
        input_modalities: Optional[List[str]] = None,
        output_modalities: Optional[List[str]] = None,
        use_tools: bool = True,
        skip_retrieval: bool = False,
        skip_planning: bool = False,
        skip_validation: bool = False,
        context: Optional[List[Any]] = None,
        plan: Optional[List[Any]] = None,
        timeout: Optional[float] = None,
        save_results: bool = True,
    ) -> Dict[str, Any] | str:
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
        # Use the orchestration service to run the task
        return await self._orchestration_service.run(
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

    async def add_documents_from_directory(self, directory: str, extension=".txt"):
        """Add documents from a directory."""
        return await self._memory_manager.add_documents_from_directory(directory, extension)

    async def retrieve(self, query: str, limit=None, fast_mode=False):
        """Retrieve documents based on a query."""
        return await self._retrieval_service.retrieve(query, limit, fast_mode=fast_mode)

    async def plan(self, task, context=None):
        """Create a plan for a task."""
        return await self._planner_service.plan(task, context)

    async def execute(self, prompt: str, context=None, use_tools=True):
        """Execute a prompt with the agent."""
        return await self._execution_service.execute(prompt, context, use_tools)
