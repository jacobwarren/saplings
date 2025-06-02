from __future__ import annotations

"""
Agent facade module.

This module provides a facade for the Agent implementation, exposing
a simplified interface for using Saplings agents.
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from saplings.agent._internal.types import AgentFacadeProtocol

# Configure logging
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    # Import interfaces from public API
    from saplings.agent._internal.agent_config import AgentConfig
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
    from saplings.api.memory import Document
    from saplings.api.tools import Tool


class AgentFacade(AgentFacadeProtocol):
    """
    Facade for Agent implementation.

    This class provides a facade for the Agent implementation, exposing
    a simplified interface for using Saplings agents.
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
        testing: bool = False,
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
            testing: Whether the agent is in testing mode

        """
        self.config = config
        self.testing = testing

        # Store services
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

        # Initialize services
        self._initialize_services()

    def _initialize_services(self) -> None:
        """Initialize all services."""
        logger.debug("Initializing agent services")

        # Register services with the monitoring service for event-based initialization
        # This allows the monitoring service to track and interact with other services
        if hasattr(self._monitoring_service, "register_service"):
            # Register all services with the monitoring service
            self._monitoring_service.register_service("model_service", self._model_service)
            self._monitoring_service.register_service("memory_manager", self._memory_manager)
            self._monitoring_service.register_service("retrieval_service", self._retrieval_service)
            self._monitoring_service.register_service("validator_service", self._validator_service)
            self._monitoring_service.register_service("execution_service", self._execution_service)
            self._monitoring_service.register_service("planner_service", self._planner_service)
            self._monitoring_service.register_service("tool_service", self._tool_service)
            self._monitoring_service.register_service(
                "self_healing_service", self._self_healing_service
            )
            self._monitoring_service.register_service("modality_service", self._modality_service)
            self._monitoring_service.register_service(
                "orchestration_service", self._orchestration_service
            )

            logger.debug("Services registered with monitoring service")
        else:
            logger.debug("Monitoring service does not support service registration")

        logger.debug("Agent services initialized")

    # Public API methods
    async def add_document(
        self, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> "Document":
        """
        Add a document to the agent's memory.

        Args:
        ----
            content: The content of the document
            metadata: Optional metadata for the document

        Returns:
        -------
            The added document

        """
        return await self._memory_manager.add_document(content, metadata or {})

    async def run(self, task: str, **kwargs) -> Any:
        """
        Run a task with the agent.

        Args:
        ----
            task: The task to run
            **kwargs: Additional arguments for the task

        Returns:
        -------
            The result of the task

        """
        # Create a trace for monitoring
        trace = self._monitoring_service.create_trace()
        trace_id = getattr(trace, "trace_id", None)

        # Start a span for the run
        span = None
        if trace_id:
            span = self._monitoring_service.start_span(
                name="agent.run", trace_id=trace_id, attributes={"task": task}
            )

        try:
            # Plan the task
            # The interface doesn't specify the exact method name, so we use a more generic approach
            if hasattr(self._planner_service, "plan"):
                plan = await self._planner_service.plan(task, **kwargs)
            elif hasattr(self._planner_service, "create_plan"):
                plan = await self._planner_service.create_plan(task, **kwargs)
            else:
                raise AttributeError("Planner service does not have plan or create_plan method")

            # Execute the plan
            # The interface doesn't specify the exact method name, so we use a more generic approach
            if hasattr(self._orchestration_service, "execute_plan"):
                result = await self._orchestration_service.execute_plan(plan)
            elif hasattr(self._orchestration_service, "execute"):
                result = await self._orchestration_service.execute(plan)
            else:
                raise AttributeError(
                    "Orchestration service does not have execute_plan or execute method"
                )

            # Validate the result
            if hasattr(self._validator_service, "validate"):
                validated_result = await self._validator_service.validate(result, task)
            else:
                # If no validation method is available, return the result as is
                validated_result = result

            return validated_result
        finally:
            # End the span if it was created
            if span and hasattr(span, "span_id"):
                self._monitoring_service.end_span(span.span_id)

    async def add_tool(self, tool: "Tool") -> None:
        """
        Add a tool to the agent.

        Args:
        ----
            tool: The tool to add

        """
        # Register the tool with the tool service
        # The interface doesn't specify if this is async, so we call it directly
        self._tool_service.register_tool(tool)

    async def register_tool(self, tool: "Tool") -> None:
        """
        Register a tool with the agent.

        This is an alias for add_tool to satisfy the AgentFacadeProtocol.

        Args:
        ----
            tool: The tool to register

        """
        # Delegate to add_tool
        await self.add_tool(tool)

    async def create_tool(self, name: str, description: str, code: str) -> Any:
        """
        Create a new tool dynamically.

        Args:
        ----
            name: The name of the tool
            description: The description of the tool
            code: The code for the tool

        Returns:
        -------
            The created tool

        """
        # The interface doesn't specify if this is async, so we handle both possibilities
        if hasattr(self._tool_service, "create_tool"):
            result = self._tool_service.create_tool(name, description, code)
            # If the result is awaitable, await it
            if hasattr(result, "__await__"):
                return await result
            return result
        else:
            raise AttributeError("Tool service does not have create_tool method")

    async def add_documents_from_directory(
        self, directory: str, extension: str = ".txt"
    ) -> List[Any]:
        """
        Add documents from a directory.

        Args:
        ----
            directory: The directory to add documents from
            extension: The file extension to filter by

        Returns:
        -------
            The added documents

        """
        if hasattr(self._memory_manager, "add_documents_from_directory"):
            return await self._memory_manager.add_documents_from_directory(directory, extension)
        else:
            logger.warning("Memory manager does not support adding documents from directory")
            return []

    async def execute_plan(
        self, plan: List[Any], context: Optional[List[Any]] = None, use_tools: bool = True
    ) -> Any:
        """
        Execute a plan.

        Args:
        ----
            plan: The plan to execute
            context: Optional context for execution
            use_tools: Whether to use tools during execution

        Returns:
        -------
            The result of executing the plan

        """
        # Use dynamic dispatch to handle different service interfaces
        if hasattr(self._orchestration_service, "execute_plan"):
            # Try to call with the expected signature
            try:
                return await self._orchestration_service.execute_plan(plan, context, use_tools)
            except TypeError:
                # Fall back to a simpler signature if needed
                return await self._orchestration_service.execute_plan(plan)
        elif hasattr(self._orchestration_service, "orchestrate"):
            # Try to call with the expected signature
            try:
                return await self._orchestration_service.orchestrate(
                    workflow={"plan": plan, "context": context, "use_tools": use_tools},
                    trace_id=None,
                )
            except TypeError:
                # Fall back to a simpler signature if needed
                return await self._orchestration_service.orchestrate({"plan": plan})
        else:
            logger.warning("Orchestration service does not support executing plans")
            return {"error": "Plan execution not supported"}

    async def judge_output(
        self, input_data: Dict[str, Any], output_data: Any, judgment_type: str = "general"
    ) -> Dict[str, Any]:
        """
        Judge an output.

        Args:
        ----
            input_data: The input data
            output_data: The output data to judge
            judgment_type: The type of judgment to perform

        Returns:
        -------
            The judgment result

        """
        if hasattr(self._validator_service, "judge_output"):
            return await self._validator_service.judge_output(
                input_data, output_data, judgment_type
            )
        else:
            logger.warning("Validator service does not support judging outputs")
            return {"valid": True, "score": 1.0, "feedback": "Judgment not supported"}

    async def self_improve(self) -> Dict[str, Any]:
        """
        Improve the agent based on past performance.

        Returns
        -------
            The improvement result

        """
        # Use reflection to find the appropriate method
        for method_name in ["improve", "self_improve", "heal", "self_heal"]:
            if hasattr(self._self_healing_service, method_name):
                method = getattr(self._self_healing_service, method_name)
                try:
                    result = method()
                    # If the result is awaitable, await it
                    if hasattr(result, "__await__"):
                        return await result
                    return result
                except Exception as e:
                    logger.warning(f"Error calling {method_name}: {e}")

        # If no method worked, return a default response
        logger.warning("Self-healing service does not support self-improvement")
        return {"improved": False, "reason": "Self-improvement not supported"}

    async def retrieve(
        self, query: str, limit: Optional[int] = None, fast_mode: bool = False
    ) -> List[Any]:
        """
        Retrieve documents based on a query.

        Args:
        ----
            query: The query to retrieve documents for
            limit: The maximum number of documents to retrieve
            fast_mode: Whether to use fast mode for retrieval

        Returns:
        -------
            The retrieved documents

        """
        timeout = 1.0 if fast_mode else None
        return await self._retrieval_service.retrieve(query, limit=limit, timeout=timeout)

    async def plan(self, task: str, context: Optional[List[Any]] = None) -> List[Any]:
        """
        Create a plan for a task.

        Args:
        ----
            task: The task to create a plan for
            context: Optional context for planning

        Returns:
        -------
            The created plan

        """
        # Convert context to a format the planner service can understand
        context_str = None
        if context:
            # Convert the context list to a string if needed
            try:
                context_str = "\n".join(str(item) for item in context)
            except Exception:
                logger.warning("Failed to convert context to string, using None")

        # Call the planner service
        plan_result = await self._planner_service.create_plan(task, context_str)

        # Convert the result to a list if it's not already
        if isinstance(plan_result, list):
            return plan_result
        # Check for common attributes that might contain the plan steps
        for attr in ["steps", "plan", "plan_steps", "tasks"]:
            if hasattr(plan_result, attr):
                attr_value = getattr(plan_result, attr)
                if isinstance(attr_value, list):
                    return attr_value

        # If we can't extract a list, return the result as a singleton list
        logger.warning("Could not extract plan steps as a list, returning as singleton")
        return [plan_result]

    async def execute(
        self, prompt: str, context: Optional[List[Any]] = None, use_tools: bool = True
    ) -> Any:
        """
        Execute a prompt with the agent.

        Args:
        ----
            prompt: The prompt to execute
            context: Optional context for execution
            use_tools: Whether to use tools during execution

        Returns:
        -------
            The execution result

        """
        functions = None
        if use_tools:
            # Get registered tools if available
            if hasattr(self._tool_service, "get_registered_tools"):
                try:
                    # Get the tools and ensure we have a list
                    tools_result = self._tool_service.get_registered_tools()
                    tools_list = []

                    # Handle different return types
                    if isinstance(tools_result, list):
                        tools_list = tools_result
                    elif tools_result is not None:
                        # Try to convert to a list if it's iterable
                        try:
                            if hasattr(tools_result, "__iter__") and not isinstance(
                                tools_result, (str, bytes, dict)
                            ):
                                tools_list = list(tools_result)
                            else:
                                # Single tool
                                tools_list = [tools_result]
                        except Exception:
                            # Single tool
                            tools_list = [tools_result]

                    # Convert tools to functions format
                    if tools_list:
                        functions = []
                        for tool in tools_list:
                            try:
                                if hasattr(tool, "to_dict"):
                                    functions.append(tool.to_dict())
                                elif hasattr(tool, "to_function_dict"):
                                    functions.append(tool.to_function_dict())
                                elif hasattr(tool, "to_openai_function"):
                                    functions.append(tool.to_openai_function())
                            except Exception as e:
                                logger.warning(f"Error converting tool to function: {e}")
                except Exception as e:
                    logger.warning(f"Error getting tools: {e}")

        # Convert context to a format the execution service can understand
        context_docs = None
        if context:
            # If context is a list of documents, use it directly
            if all(hasattr(item, "content") for item in context):
                context_docs = context
            else:
                # Otherwise, try to convert to a string
                try:
                    context_str = "\n".join(str(item) for item in context)
                    # Create a document-like object
                    context_docs = [{"content": context_str, "metadata": {}}]
                except Exception:
                    logger.warning("Failed to convert context to string, using None")

        # Call the execution service
        return await self._execution_service.execute(prompt, context_docs, functions=functions)
