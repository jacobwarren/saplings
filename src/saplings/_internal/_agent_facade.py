from __future__ import annotations

"""
Agent Facade module for Saplings.

This module provides a high-level AgentFacade class that integrates all components
of the Saplings framework through composition rather than direct implementation.
The facade delegates to specialized service interfaces for each concern:

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

Each service encapsulates a single area of responsibility, improving maintainability,
testability, and separation of concerns.

The implementation follows the Dependency Inversion Principle:
- High-level modules should not depend on low-level modules
- Both should depend on abstractions
- Abstractions should not depend on details
- Details should depend on abstractions
"""


import logging
import os
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable

# Import the AgentConfig
# Service interfaces
# Core components
from saplings.api.planner import PlanStep, PlanStepStatus

# Import Tool class for type annotations

if TYPE_CHECKING:
    from saplings._internal.agent_module import AgentConfig
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

# Configure logging
logger = logging.getLogger(__name__)


class AgentFacade:
    """
    A true facade for the Saplings framework components.

    This class follows the Facade design pattern by providing a simplified interface
    to a complex subsystem of services. It delegates calls to specific service
    interfaces without exposing their internal details.

    The facade pattern is fully implemented here:
    - All component access is through strictly encapsulated service interfaces
    - Internal implementation details are hidden from clients
    - Public API is focused on high-level operations
    - Each service encapsulates a single area of responsibility

    Key services with their interfaces:
    - IModelService: Model initialization and management
    - IMemoryManager: Document storage and indexing
    - IRetrievalService: Document retrieval
    - IPlannerService: Planning tasks
    - IExecutionService: Executing prompts with context
    - IValidatorService: Validating outputs
    - ISelfHealingService: Self-improvement capabilities
    - IToolService: Tool registration and dynamic creation
    - IModalityService: Multimodal support
    - IMonitoringService: Tracing and monitoring
    - IOrchestrationService: Workflow orchestration

    The facade exposes a high-level public API while keeping the implementation
    details completely decoupled through interfaces, following the Dependency
    Inversion Principle.
    """

    def __init__(
        self,
        config: AgentConfig,
        monitoring_service: IMonitoringService | None = None,
        testing: bool = False,
        model_service: IModelInitializationService | None = None,
        memory_manager: IMemoryManager | None = None,
        retrieval_service: IRetrievalService | None = None,
        validator_service: IValidatorService | None = None,
        execution_service: IExecutionService | None = None,
        planner_service: IPlannerService | None = None,
        tool_service: IToolService | None = None,
        self_healing_service: ISelfHealingService | None = None,
        modality_service: IModalityService | None = None,
        orchestration_service: IOrchestrationService | None = None,
    ) -> None:
        """
        Initialize the agent facade with the provided configuration and services.

        Args:
        ----
            config: Agent configuration
            monitoring_service: Monitoring service (optional)
            model_service: Model service (optional)
            memory_manager: Memory manager (optional)
            retrieval_service: Retrieval service (optional)
            validator_service: Validator service (optional)
            execution_service: Execution service (optional)
            planner_service: Planner service (optional)
            tool_service: Tool service (optional)
            self_healing_service: Self-healing service (optional)
            modality_service: Modality service (optional)
            orchestration_service: Orchestration service (optional)

        """
        self.config = config
        self.testing = testing

        # Set up directories
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.memory_path, exist_ok=True)

        # Use provided services or get from container
        if all(
            [
                monitoring_service,
                model_service,
                memory_manager,
                retrieval_service,
                validator_service,
                execution_service,
                planner_service,
                tool_service,
                self_healing_service,
                modality_service,
                orchestration_service,
            ]
        ):
            # Use provided services
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

            logger.info("AgentFacade initialized with explicitly provided services")
        else:
            # Use the global container instead of creating a new one
            # This ensures we're using the same container throughout the application
            from saplings.api.di import configure_container, container

            # Configure the container with our config
            # This ensures all services are properly registered with the correct configuration
            configure_container(config)

            # Store a reference to the container
            self._container = container

            # Resolve services from the container
            try:
                # Core interfaces
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

                # Resolve services with proper error handling
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

                logger.info("AgentFacade initialized with services from global container")
            except Exception as e:
                logger.error(f"Failed to resolve services from container: {e}")
                logger.warning("Some agent functionality may be limited")
                # Initialize services to None to avoid attribute errors
                self._monitoring_service = None
                self._model_service = None
                self._memory_manager = None
                self._retrieval_service = None
                self._validator_service = None
                self._execution_service = None
                self._planner_service = None
                self._tool_service = None
                self._self_healing_service = None
                self._modality_service = None
                self._orchestration_service = None

        # We'll initialize the model in a separate async method
        # This is a placeholder that will be set in _init_model_async
        self._model = None

        # Register tools if tool service is available
        if self._tool_service and self.config.tools:
            for tool in self.config.tools:
                self._tool_service.register_tool(tool)

        # Import and register other tools
        if self._tool_service:
            self._register_default_tools()

        logger.info(f"Agent Facade initialized with model: {config.provider}/{config.model_name}")
        logger.info(f"GASA enabled: {config.enable_gasa}")
        logger.info(f"Monitoring enabled: {config.enable_monitoring}")
        logger.info(f"Self-healing enabled: {config.enable_self_healing}")
        logger.info(f"Tool factory enabled: {config.enable_tool_factory}")
        registered_tools = self._tool_service.get_registered_tools() if self._tool_service else []
        logger.info(f"Registered tools: {len(registered_tools) if registered_tools else 0}")
        logger.info(f"Supported modalities: {self.config.supported_modalities}")

    async def init_model(self):
        """
        Initialize the model asynchronously.

        This method must be called after creating the AgentFacade instance
        and before using any methods that require the model.

        Returns
        -------
            self: The AgentFacade instance for method chaining

        """
        if self._model_service:
            self._model = await self._model_service.get_model()

            # Initialize the ExecutionService with the model
            if self._execution_service:
                try:
                    self._execution_service.initialize(self._model)
                    logger.info("ExecutionService initialized with model")
                except Exception as e:
                    logger.warning(f"Failed to initialize ExecutionService with model: {e}")

            # Initialize the judge for the validator service
            # This ensures the judge is available for validation
            if self._validator_service:
                await self._init_judge_for_validator()

        return self

    async def _init_judge_for_validator(self):
        """Initialize the judge for the validator service."""
        if not self._model:
            await self.init_model()

        if not self._validator_service:
            logger.warning("Validator service not available, judge initialization skipped")
            return

        try:
            from saplings.judge import JudgeAgent

            # Create a judge agent
            judge = JudgeAgent(
                model=self._model,
                rubric_path=None,  # Use default rubric
            )

            # Set judge on validator service
            try:
                # Use the interface method to set the judge if available
                # Note: This is a compatibility check for older validator service implementations
                if (
                    self._validator_service
                    and hasattr(self._validator_service, "set_judge")
                    and callable(self._validator_service.set_judge)
                ):
                    # Use dynamic attribute access to avoid type errors
                    await self._validator_service.set_judge(judge)
                    logger.info("Judge initialized for validator service using async method")
                else:
                    # This is expected in newer implementations where the judge is set during initialization
                    logger.info(
                        "Validator service does not have set_judge method, judge may already be initialized"
                    )
            except Exception as e:
                logger.warning(f"Error setting judge via async method: {e}")

            # Log success without testing the judge
            # We don't want to send test tasks to the judge as it wastes tokens
            logger.info("Judge initialized for validator service")
        except ImportError:
            logger.warning("JudgeAgent not available, some validation features may not work")
        except Exception as e:
            logger.exception(f"Failed to initialize judge: {e}")

    def _register_default_tools(self):
        """Register default tools with the agent."""
        if not self._tool_service:
            logger.warning("Tool service not available, skipping default tool registration")
            return

        try:
            # Import here to avoid circular imports
            from saplings.tools import get_registered_tools

            # Register tools that were registered via the @register_tool decorator
            registered_tools = get_registered_tools()
            if registered_tools:
                for tool in registered_tools.values():
                    if tool:
                        self._tool_service.register_tool(tool)
        except ImportError:
            logger.warning("Failed to import tools module, default tools not registered")
        except Exception as e:
            logger.exception(f"Error registering default tools: {e}")

    def _execute_in_process_tool(self, tool, tool_args):
        """
        Execute an in-process Python tool directly, avoiding JSON serialization.

        This method is optimized for Python functions that are called directly within
        the same process, avoiding the overhead of JSON serialization and deserialization.

        Args:
        ----
            tool: The tool object with a func attribute
            tool_args: The arguments for the tool (string or dict)

        Returns:
        -------
            The result of the function call

        """
        try:
            # Extract the raw function
            func = tool.func

            # Parse arguments if needed
            if isinstance(tool_args, str):
                import json

                try:
                    # We still need to parse JSON if it's a string, but we'll
                    # only do this conversion once
                    tool_args = json.loads(tool_args)
                except json.JSONDecodeError:
                    # If not valid JSON, pass as is
                    return func(tool_args)

            # Call function directly with arguments
            if isinstance(tool_args, dict):
                # For keyword arguments
                return func(**tool_args)
            # For positional arguments
            return func(tool_args)
        except Exception as e:
            # Log the error and return it
            logger.exception(f"Error executing in-process tool: {e}")
            return f"Error: {e!s}"

    # Public API methods
    # ===============================================================
    # The facade pattern dictates that we should expose only high-level operations,
    # hiding all implementation details and keeping dependencies properly encapsulated

    async def add_document(self, content: str, metadata: dict[str, Any] | None = None) -> Document:
        """
        Add a document to the agent's memory.

        Args:
        ----
            content: Document content
            metadata: Document metadata (optional)

        Returns:
        -------
            Document: The added document

        """
        trace_id = None
        span = None
        if (
            self._monitoring_service
            and hasattr(self._monitoring_service, "enabled")
            and self._monitoring_service.enabled
        ):
            if hasattr(self._monitoring_service, "create_trace"):
                trace = self._monitoring_service.create_trace()
                if trace and hasattr(trace, "get"):
                    trace_id = trace.get("trace_id")

                if hasattr(self._monitoring_service, "start_span"):
                    span = self._monitoring_service.start_span(
                        name="add_document",
                        trace_id=trace_id,
                        attributes={"component": "agent"},
                    )

        try:
            if not self._memory_manager:
                raise ValueError("Memory manager not available")

            document = await self._memory_manager.add_document(
                content=content,
                metadata=metadata,
            )

            logger.info(f"Added document: {document.id}")
            return document
        finally:
            if (
                trace_id
                and span
                and self._monitoring_service
                and hasattr(self._monitoring_service, "end_span")
            ):
                self._monitoring_service.end_span(span.span_id)

    async def execute_plan(
        self, plan: list[PlanStep], context: list[Document] | None = None, use_tools: bool = True
    ) -> dict[str, Any]:
        """
        Execute a plan.

        Args:
        ----
            plan: Plan steps to execute
            context: Context documents (optional)
            use_tools: Whether to enable tool usage (default: True)

        Returns:
        -------
            Dict[str, Any]: Execution results

        """
        trace_id = None
        span = None
        if (
            self._monitoring_service
            and hasattr(self._monitoring_service, "enabled")
            and self._monitoring_service.enabled
        ):
            if hasattr(self._monitoring_service, "create_trace"):
                trace = self._monitoring_service.create_trace()
                if trace and hasattr(trace, "get"):
                    trace_id = trace.get("trace_id")

                if hasattr(self._monitoring_service, "start_span"):
                    span = self._monitoring_service.start_span(
                        name="execute_plan",
                        trace_id=trace_id,
                        attributes={
                            "component": "agent",
                            "step_count": len(plan),
                        },
                    )

        try:
            results = []

            # Execute each step
            for i, step in enumerate(plan):
                step_span = None
                if (
                    self._monitoring_service
                    and hasattr(self._monitoring_service, "enabled")
                    and self._monitoring_service.enabled
                    and trace_id
                    and hasattr(self._monitoring_service, "start_span")
                ):
                    step_span = self._monitoring_service.start_span(
                        name=f"execute_step_{i + 1}",
                        trace_id=trace_id,
                        parent_id=span.span_id if span else None,
                        attributes={
                            "component": "agent",
                            "step_number": i + 1,
                            "step_description": step.task_description,
                        },
                    )

                # Execute step with tools if enabled
                # Prepare functions for tool calling
                functions = None
                if (
                    use_tools
                    and self._tool_service
                    and hasattr(self._tool_service, "prepare_functions_for_model")
                ):
                    functions = self._tool_service.prepare_functions_for_model()

                if not self._execution_service:
                    raise ValueError("Execution service not available")

                step_result = await self._execution_service.execute(
                    prompt=step.task_description if step.task_description else "",
                    documents=context,
                    functions=functions,
                    function_call="auto" if functions else None,
                    trace_id=trace_id,
                )

                # Update step status
                step.update_status(PlanStepStatus.COMPLETED)
                step.result = step_result.text

                # Update actual cost and token usage
                if step_result.response and step_result.response.usage:
                    # Get token usage
                    total_tokens = step_result.response.usage.get("total_tokens")
                    if total_tokens is not None:
                        step.actual_tokens = total_tokens

                    # Calculate actual cost
                    prompt_tokens = step_result.response.usage.get("prompt_tokens", 0)
                    completion_tokens = step_result.response.usage.get("completion_tokens", 0)

                    # Calculate cost using model service if available
                    # Note: This is a compatibility check for older model service implementations
                    if (
                        self._model_service
                        and hasattr(self._model_service, "estimate_cost")
                        and callable(self._model_service.estimate_cost)
                    ):
                        # Use dynamic attribute access to avoid type errors
                        cost = await self._model_service.estimate_cost(
                            prompt_tokens, completion_tokens
                        )
                        # Cost is already awaited, just assign it directly
                        step.actual_cost = cost

                # Add to results
                results.append(
                    {
                        "step": step,
                        "result": step_result.text,
                    }
                )

                # End step span
                if (
                    self._monitoring_service
                    and hasattr(self._monitoring_service, "enabled")
                    and self._monitoring_service.enabled
                    and trace_id
                    and step_span
                    and hasattr(self._monitoring_service, "end_span")
                ):
                    self._monitoring_service.end_span(step_span.span_id)

            # Process trace for blame graph if monitoring is enabled
            if (
                self._monitoring_service
                and hasattr(self._monitoring_service, "enabled")
                and self._monitoring_service.enabled
                and trace_id
                and hasattr(self._monitoring_service, "process_trace")
            ):
                self._monitoring_service.process_trace(trace_id)

            return {
                "results": results,
                "trace_id": trace_id,
            }
        finally:
            if (
                trace_id
                and span
                and self._monitoring_service
                and hasattr(self._monitoring_service, "end_span")
            ):
                self._monitoring_service.end_span(span.span_id)

    def _has_service_attr(self, service, attr_name):
        """
        Safely check if a service exists and has a specific attribute.

        Args:
        ----
            service: The service to check
            attr_name: The attribute name to check for

        Returns:
        -------
            bool: True if the service exists and has the attribute, False otherwise

        """
        return service is not None and hasattr(service, attr_name)

    def register_tool(self, tool: Tool | Callable) -> bool:
        """
        Register a tool with the agent.

        Args:
        ----
            tool: Tool to register (either a Tool instance or a callable)

        Returns:
        -------
            bool: True if registration was successful, False otherwise

        """
        if not self._tool_service:
            logger.warning("Tool service not available")
            return False

        if not hasattr(self._tool_service, "register_tool"):
            logger.warning("Tool service doesn't support register_tool")
            return False

        return self._tool_service.register_tool(tool)

    async def create_tool(self, name: str, description: str, code: str) -> Tool:
        """
        Create a dynamic tool.

        Args:
        ----
            name: Tool name
            description: Tool description
            code: Tool code

        Returns:
        -------
            Tool: Created tool

        """
        if not self._tool_service:
            logger.warning("Tool service not available")
            raise ValueError("Tool service not available")

        if not hasattr(self._tool_service, "create_tool"):
            logger.warning("Tool service doesn't support create_tool")
            raise ValueError("Tool service doesn't support create_tool method")

        trace_id = None
        if (
            self._monitoring_service
            and hasattr(self._monitoring_service, "enabled")
            and self._monitoring_service.enabled
        ):
            if hasattr(self._monitoring_service, "create_trace"):
                trace = self._monitoring_service.create_trace()
                if trace and hasattr(trace, "get"):
                    trace_id = trace.get("trace_id")

        return await self._tool_service.create_tool(
            name=name,
            description=description,
            code=code,
            trace_id=trace_id,
        )

    async def judge_output(
        self,
        input_data: dict[str, Any],
        output_data: str | dict[str, Any],
        judgment_type: str = "general",
    ) -> dict[str, Any]:
        """
        Judge an output using the JudgeAgent.

        Args:
        ----
            input_data: Input data
            output_data: Output data to judge
            judgment_type: Type of judgment

        Returns:
        -------
            Dict[str, Any]: Judgment result

        """
        if not self._validator_service:
            logger.warning("Validator service not available")
            return {
                "is_valid": False,
                "score": 0.0,
                "feedback": "Validation service not available",
                "trace_id": None,
            }

        if not hasattr(self._validator_service, "judge_output"):
            logger.warning("Validator service doesn't support judge_output")
            return {
                "is_valid": False,
                "score": 0.0,
                "feedback": "Validation service doesn't support judge_output",
                "trace_id": None,
            }

        trace_id = None
        if (
            self._monitoring_service
            and hasattr(self._monitoring_service, "enabled")
            and self._monitoring_service.enabled
        ):
            if hasattr(self._monitoring_service, "create_trace"):
                trace = self._monitoring_service.create_trace()
                if trace and hasattr(trace, "get"):
                    trace_id = trace.get("trace_id")

        result = await self._validator_service.judge_output(
            input_data=input_data,
            output_data=output_data,
            judgment_type=judgment_type,
            trace_id=trace_id,
        )

        # Add trace_id to result
        if isinstance(result, dict):
            result["trace_id"] = trace_id
        elif hasattr(result, "trace_id"):
            result.trace_id = trace_id

        return result

    async def self_improve(self):
        """
        Improve the agent based on past performance.

        Returns
        -------
            Dict[str, Any]: Improvement results

        """
        if not self.config.enable_self_healing:
            msg = "Self-healing is disabled"
            raise ValueError(msg)

        trace_id = None
        span = None
        if (
            self._monitoring_service
            and hasattr(self._monitoring_service, "enabled")
            and self._monitoring_service.enabled
        ):
            if hasattr(self._monitoring_service, "create_trace"):
                trace = self._monitoring_service.create_trace()
                if trace and hasattr(trace, "get"):
                    trace_id = trace.get("trace_id")

                    if hasattr(self._monitoring_service, "start_span"):
                        span = self._monitoring_service.start_span(
                            name="self_improve",
                            trace_id=trace_id,
                            attributes={"component": "agent"},
                        )

        try:
            # Get performance data
            performance_data = {}
            if (
                self._monitoring_service
                and hasattr(self._monitoring_service, "enabled")
                and self._monitoring_service.enabled
            ):
                if hasattr(self._monitoring_service, "identify_bottlenecks"):
                    bottlenecks = self._monitoring_service.identify_bottlenecks(
                        threshold_ms=100.0,
                        min_call_count=1,
                    )
                    performance_data["bottlenecks"] = bottlenecks

                if hasattr(self._monitoring_service, "identify_error_sources"):
                    error_sources = self._monitoring_service.identify_error_sources(
                        min_error_rate=0.1,
                        min_call_count=1,
                    )
                    performance_data["error_sources"] = error_sources

            # Get validation data
            validation_data = {}
            if self._validator_service and hasattr(
                self._validator_service, "get_validation_history"
            ):
                validation_data = self._validator_service.get_validation_history()

            # Get success pairs
            success_pairs = []
            if self._self_healing_service and hasattr(
                self._self_healing_service, "get_all_success_pairs"
            ):
                success_pairs = await self._self_healing_service.get_all_success_pairs(
                    trace_id=trace_id
                )

            # Create improvement prompt
            prompt = f"""
            Analyze the agent's performance and suggest improvements.

            Performance Data:
            {performance_data}

            Validation History:
            {validation_data}

            Success Pairs Count: {len(success_pairs)}

            Based on this data, suggest specific improvements to the agent's:
            1. Prompting strategies
            2. Document analysis approach
            3. Overall workflow

            Format your response as JSON with the following structure:
            {{
                "identified_issues": [list of issues identified],
                "improvement_suggestions": [list of specific improvement suggestions],
                "prompt_templates": {{
                    "analysis_prompt": "improved analysis prompt template",
                    "execution_prompt": "improved execution prompt template"
                }},
                "implementation_plan": [list of steps to implement improvements]
            }}
            """

            # Execute improvement analysis
            if not self._execution_service:
                raise ValueError("Execution service not available")

            if not hasattr(self._execution_service, "execute"):
                raise ValueError("Execution service does not support execute method")

            result = await self._execution_service.execute(
                prompt=prompt,
                trace_id=trace_id,
            )

            # Parse JSON response
            try:
                import json

                improvements = json.loads(result.text)
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON response, returning raw text")
                improvements = {"raw_text": result.text}

            # Judge improvements
            judgment = {
                "score": 0.5,
                "feedback": "No judgment performed",
                "strengths": [],
                "weaknesses": [],
            }
            if self._validator_service and hasattr(self._validator_service, "judge_output"):
                judgment = await self._validator_service.judge_output(
                    input_data={
                        "performance_data": performance_data,
                        "validation_data": validation_data,
                    },
                    output_data=improvements,
                    judgment_type="agent_improvement",
                    trace_id=trace_id,
                )

            # Add judgment to improvements
            if isinstance(judgment, dict):
                improvements["judgment"] = {
                    "score": judgment.get("score", 0.5),
                    "feedback": judgment.get("feedback", "No feedback available"),
                    "strengths": judgment.get("strengths", []),
                    "weaknesses": judgment.get("weaknesses", []),
                }
            else:
                improvements["judgment"] = {
                    "score": getattr(judgment, "score", 0.5),
                    "feedback": getattr(judgment, "feedback", "No feedback available"),
                    "strengths": getattr(judgment, "strengths", []),
                    "weaknesses": getattr(judgment, "weaknesses", []),
                }

            # Save improvements to output directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.config.output_dir, f"improvements_{timestamp}.json")

            with open(output_path, "w") as f:
                json.dump(improvements, f, indent=2)

            logger.info(f"Improvement analysis completed and saved to {output_path}")

            # Process trace for blame graph if monitoring is enabled
            if (
                self._monitoring_service
                and hasattr(self._monitoring_service, "enabled")
                and self._monitoring_service.enabled
                and trace_id
                and hasattr(self._monitoring_service, "process_trace")
            ):
                self._monitoring_service.process_trace(trace_id)

            return {
                "improvements": improvements,
                "trace_id": trace_id,
                "output_path": output_path,
            }
        finally:
            if (
                trace_id
                and span
                and self._monitoring_service
                and hasattr(self._monitoring_service, "end_span")
            ):
                self._monitoring_service.end_span(span.span_id)

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
        """
        # Initialize the model if not already initialized
        if not self._model:
            await self.init_model()

        # Initialize the judge for the validator service
        # This ensures the judge is available for validation
        if not skip_validation and self._validator_service:
            await self._init_judge_for_validator()

        # Continue with the agent workflow
        """
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

        Raises:
        ------
            ValueError: If modality is not supported
            TimeoutError: If the operation times out
            CancellationError: If the operation is cancelled
        """
        # Set default modalities if not provided
        input_modalities = input_modalities or ["text"]
        output_modalities = output_modalities or ["text"]

        # Validate input and output modalities
        if self._modality_service and hasattr(self._modality_service, "supported_modalities"):
            supported_modalities = self._modality_service.supported_modalities()
            if supported_modalities:  # Check if not None
                for modality in input_modalities + output_modalities:
                    if modality not in supported_modalities:
                        msg = (
                            f"Unsupported modality: {modality}. "
                            f"This agent supports: {supported_modalities}"
                        )
                        raise ValueError(msg)

        # Set up monitoring
        trace_id = None
        span = None
        if (
            self._monitoring_service
            and hasattr(self._monitoring_service, "enabled")
            and self._monitoring_service.enabled
        ):
            if hasattr(self._monitoring_service, "create_trace"):
                trace = self._monitoring_service.create_trace()
                if trace and hasattr(trace, "get"):
                    trace_id = trace.get("trace_id")

                    if hasattr(self._monitoring_service, "start_span"):
                        span = self._monitoring_service.start_span(
                            name="run",
                            trace_id=trace_id,
                            attributes={
                                "component": "agent",
                                "task": task,
                                "skip_retrieval": skip_retrieval,
                                "skip_planning": skip_planning,
                                "skip_validation": skip_validation,
                            },
                        )

        try:
            # Create the execution coroutine
            execution_coro = self._run_internal(
                task=task,
                input_modalities=input_modalities,
                output_modalities=output_modalities,
                use_tools=use_tools,
                skip_retrieval=skip_retrieval,
                skip_planning=skip_planning,
                skip_validation=skip_validation,
                context=context,
                plan=plan,
                trace_id=trace_id,
                span=span,
                save_results=save_results,
            )

            # Apply timeout if specified
            if timeout:
                from saplings.core.resilience import with_timeout

                # Use with_timeout for consistent timeout handling
                logger.debug(f"Running agent with timeout: {timeout}s")
                try:
                    result = await with_timeout(
                        execution_coro, timeout=timeout, operation_name="agent.run"
                    )
                    logger.debug("Agent run completed successfully within timeout")
                except Exception as e:
                    logger.error(f"Error during agent.run with timeout: {e}")
                    # Clean up any resources before re-raising
                    if (
                        trace_id
                        and span
                        and self._monitoring_service
                        and hasattr(self._monitoring_service, "end_span")
                    ):
                        self._monitoring_service.end_span(span.span_id)
                    raise
            else:
                # Run without timeout
                logger.debug("Running agent without timeout")
                result = await execution_coro
                logger.debug("Agent run completed successfully")

            # If the result is a dictionary with a final_result key, return that for simplicity
            if isinstance(result, dict) and "final_result" in result:
                return result["final_result"]

            return result
        except Exception as e:
            logger.exception(f"Error during agent.run: {e}")
            # Re-raise the exception to allow proper error handling
            raise
        finally:
            # Always ensure the span is closed
            if (
                trace_id
                and span
                and self._monitoring_service
                and hasattr(self._monitoring_service, "end_span")
            ):
                self._monitoring_service.end_span(span.span_id)

    async def _run_internal(
        self,
        task: str,
        input_modalities: list[str],
        output_modalities: list[str],
        use_tools: bool,
        skip_retrieval: bool,
        skip_planning: bool,
        skip_validation: bool,
        context: list[Any] | None,
        plan: list["PlanStep"] | None,
        trace_id: str | None,
        span: Any | None,
        save_results: bool,
    ) -> dict[str, Any] | str:
        """
        Internal implementation of the run method with all parameters.

        This separates the timeout handling from the actual implementation.
        """
        try:
            # Step 1: Retrieve context (if not skipped)
            if not skip_retrieval:
                context_span = None
                if (
                    self._monitoring_service
                    and hasattr(self._monitoring_service, "enabled")
                    and self._monitoring_service.enabled
                    and trace_id
                    and hasattr(self._monitoring_service, "start_span")
                ):
                    context_span = self._monitoring_service.start_span(
                        name="retrieve_context",
                        trace_id=trace_id,
                        parent_id=span.span_id if span else None,
                        attributes={"component": "retriever"},
                    )

                try:
                    context = await self.retrieve(task)
                except Exception as e:
                    logger.warning(f"Retrieval failed, continuing without context: {e}")
                    context = []

                if (
                    self._monitoring_service
                    and hasattr(self._monitoring_service, "enabled")
                    and self._monitoring_service.enabled
                    and trace_id
                    and context_span
                    and hasattr(self._monitoring_service, "end_span")
                ):
                    self._monitoring_service.end_span(context_span.span_id)
            elif context is None:
                # If retrieval is skipped but no context is provided, use empty list
                context = []

            # Step 2: Create plan (if not skipped)
            if not skip_planning:
                plan_span = None
                if (
                    self._monitoring_service
                    and hasattr(self._monitoring_service, "enabled")
                    and self._monitoring_service.enabled
                    and trace_id
                    and hasattr(self._monitoring_service, "start_span")
                ):
                    plan_span = self._monitoring_service.start_span(
                        name="create_plan",
                        trace_id=trace_id,
                        parent_id=span.span_id if span else None,
                        attributes={"component": "planner"},
                    )

                try:
                    plan = await self.plan(task, context)
                except Exception as e:
                    logger.warning(f"Planning failed, falling back to direct execution: {e}")
                    # If planning fails, we'll use direct execution
                    skip_planning = True

                if (
                    self._monitoring_service
                    and hasattr(self._monitoring_service, "enabled")
                    and self._monitoring_service.enabled
                    and trace_id
                    and plan_span
                    and hasattr(self._monitoring_service, "end_span")
                ):
                    self._monitoring_service.end_span(plan_span.span_id)

            # Step 3: Execute (either plan or direct)
            execution_span = None
            if (
                self._monitoring_service
                and hasattr(self._monitoring_service, "enabled")
                and self._monitoring_service.enabled
                and trace_id
                and hasattr(self._monitoring_service, "start_span")
            ):
                execution_span = self._monitoring_service.start_span(
                    name="execute",
                    trace_id=trace_id,
                    parent_id=span.span_id if span else None,
                    attributes={
                        "component": "executor",
                        "execution_type": "plan" if not skip_planning else "direct",
                    },
                )

            try:
                if skip_planning or plan is None:
                    # Direct execution
                    execution_result = await self.execute(task, context, use_tools=use_tools)
                    final_result = execution_result["text"]
                    execution_results = {"results": [{"result": final_result}]}
                else:
                    # Plan execution (we know plan is not None here)
                    execution_results = await self.execute_plan(plan, context, use_tools=use_tools)
                    final_result = "\n\n".join([r["result"] for r in execution_results["results"]])
            except Exception as e:
                logger.exception(f"Execution failed: {e}")
                final_result = f"Error during execution: {e!s}"
                execution_results = {"results": [{"result": final_result}]}

            if (
                self._monitoring_service
                and hasattr(self._monitoring_service, "enabled")
                and self._monitoring_service.enabled
                and trace_id
                and execution_span
                and hasattr(self._monitoring_service, "end_span")
            ):
                self._monitoring_service.end_span(execution_span.span_id)

            # Step 4: Judge results (if not skipped)
            judgment = {"is_valid": True, "score": 1.0, "feedback": "No validation performed"}
            if not skip_validation:
                validation_span = None
                if (
                    self._monitoring_service
                    and hasattr(self._monitoring_service, "enabled")
                    and self._monitoring_service.enabled
                    and trace_id
                    and hasattr(self._monitoring_service, "start_span")
                ):
                    validation_span = self._monitoring_service.start_span(
                        name="validate",
                        trace_id=trace_id,
                        parent_id=span.span_id if span else None,
                        attributes={"component": "validator"},
                    )

                try:
                    if self._validator_service and hasattr(self._validator_service, "judge_output"):
                        # Truncate context to avoid token limit errors
                        try:
                            # Use a simple truncation approach that doesn't rely on external imports
                            context_texts = [doc.content for doc in context]

                            # Limit to 3 most relevant documents (assuming they're already sorted by relevance)
                            if len(context_texts) > 3:
                                context_texts = context_texts[:3]

                            # Limit each document to 500 characters
                            truncated_context = [
                                text[:500] + ("..." if len(text) > 500 else "")
                                for text in context_texts
                            ]
                            logger.info(
                                f"Truncated context for validation from {len(context)} to {len(truncated_context)} items"
                            )
                        except Exception as e:
                            # Fall back to full context if truncation fails
                            logger.warning(f"Error truncating context: {e}")
                            truncated_context = [doc.content for doc in context]
                            logger.warning(
                                "Could not import context_utils, using full context for validation"
                            )

                        judgment = await self._validator_service.judge_output(
                            input_data={"task": task, "context": truncated_context},
                            output_data=final_result,
                            judgment_type="task_execution",
                            trace_id=trace_id,
                            # Note: timeout parameter is handled by the service internally
                        )
                except Exception as e:
                    logger.warning(f"Validation failed: {e}")

                if (
                    self._monitoring_service
                    and hasattr(self._monitoring_service, "enabled")
                    and self._monitoring_service.enabled
                    and trace_id
                    and validation_span
                    and hasattr(self._monitoring_service, "end_span")
                ):
                    self._monitoring_service.end_span(validation_span.span_id)

            # Step 5: Collect success pair if valid and self-healing is enabled
            judgment_score = (
                judgment.get("score", 0)
                if isinstance(judgment, dict)
                else getattr(judgment, "score", 0)
            )

            if (
                self.config.enable_self_healing
                and judgment_score >= 0.7
                and self._self_healing_service
                and hasattr(self._self_healing_service, "collect_success_pair")
            ):
                try:
                    await self._self_healing_service.collect_success_pair(
                        input_text=task,
                        output_text=final_result,
                        context=[doc.content for doc in context],
                        metadata={
                            "judgment_score": judgment_score,
                            "timestamp": datetime.now().isoformat(),
                        },
                        trace_id=trace_id,
                    )
                except Exception as e:
                    logger.warning(f"Failed to collect success pair: {e}")

            # Process modality outputs
            modality_outputs = {}
            if self._modality_service and hasattr(self._modality_service, "format_output"):
                for modality in output_modalities:
                    try:
                        modality_output = await self._modality_service.format_output(
                            content=final_result,
                            output_modality=modality,
                            trace_id=trace_id,
                        )
                        modality_outputs[modality] = modality_output
                    except Exception as e:
                        logger.exception(f"Error processing output for {modality} modality: {e}")
                        modality_outputs[modality] = None
            else:
                # Default to text if no modality service
                modality_outputs["text"] = final_result

            # Save results to output directory if requested
            output_path = None
            if save_results:
                try:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_path = os.path.join(
                        self.config.output_dir, f"task_results_{timestamp}.json"
                    )

                    with open(output_path, "w") as f:
                        import json

                        # Create a serializable version of modality outputs
                        serializable_modality_outputs = {}
                        for modality, output in modality_outputs.items():
                            if modality == "text":
                                serializable_modality_outputs[modality] = output
                            elif output is not None:
                                # For non-text modalities, store type information
                                if isinstance(output, dict) and "type" in output:
                                    if output["type"] == "url":
                                        serializable_modality_outputs[modality] = {
                                            "type": "url",
                                            "url": output["url"],
                                        }
                                    else:
                                        serializable_modality_outputs[modality] = {
                                            "type": output["type"]
                                        }

                        json.dump(
                            {
                                "task": task,
                                "plan": [step.__dict__ for step in plan]
                                if plan and not skip_planning
                                else [],
                                "results": execution_results["results"],
                                "judgment": judgment,
                                "input_modalities": input_modalities,
                                "output_modalities": output_modalities,
                                "modality_outputs": serializable_modality_outputs,
                            },
                            f,
                            indent=2,
                            default=str,
                        )

                    logger.info(f"Task execution completed and saved to {output_path}")
                except Exception as e:
                    logger.warning(f"Failed to save results: {e}")

            # Process trace for blame graph if monitoring is enabled
            if (
                self._monitoring_service
                and hasattr(self._monitoring_service, "enabled")
                and self._monitoring_service.enabled
                and trace_id
                and hasattr(self._monitoring_service, "process_trace")
            ):
                try:
                    self._monitoring_service.process_trace(trace_id)
                except Exception as e:
                    logger.warning(f"Failed to process trace: {e}")

            # Return the full result dictionary
            return {
                "task": task,
                "context": context,
                "plan": plan if not skip_planning else None,
                "results": execution_results["results"],
                "final_result": final_result,
                "modality_outputs": modality_outputs,
                "judgment": judgment,
                "trace_id": trace_id,
                "output_path": output_path,
                "input_modalities": input_modalities,
                "output_modalities": output_modalities,
            }
        finally:
            # Always ensure the span is closed
            if (
                trace_id
                and span
                and self._monitoring_service
                and hasattr(self._monitoring_service, "end_span")
            ):
                self._monitoring_service.end_span(span.span_id)

    async def add_documents_from_directory(
        self, directory: str, extension: str = ".txt"
    ) -> list[Document]:
        """
        Add documents from a directory.

        Args:
        ----
            directory: Directory containing documents
            extension: File extension to filter by

        Returns:
        -------
            List[Document]: Added documents

        """
        trace_id = None
        span = None
        if (
            self._monitoring_service
            and hasattr(self._monitoring_service, "enabled")
            and self._monitoring_service.enabled
        ):
            if hasattr(self._monitoring_service, "create_trace"):
                trace = self._monitoring_service.create_trace()
                if trace and hasattr(trace, "get"):
                    trace_id = trace.get("trace_id")

                if hasattr(self._monitoring_service, "start_span"):
                    span = self._monitoring_service.start_span(
                        name="add_documents_from_directory",
                        trace_id=trace_id,
                        attributes={
                            "component": "agent",
                            "directory": directory,
                            "extension": extension,
                        },
                    )

        try:
            if not self._memory_manager:
                raise ValueError("Memory manager not available")

            if not hasattr(self._memory_manager, "add_documents_from_directory"):
                raise ValueError("Memory manager does not support add_documents_from_directory")

            documents = await self._memory_manager.add_documents_from_directory(
                directory=directory,
                extension=extension,
            )

            # No need to check if already awaited result is awaitable

            logger.info(f"Added {len(documents)} documents from {directory}")
            return documents
        finally:
            if (
                trace_id
                and span
                and self._monitoring_service
                and hasattr(self._monitoring_service, "end_span")
            ):
                self._monitoring_service.end_span(span.span_id)

    async def retrieve(
        self, query: str, limit: int | None = None, fast_mode: bool = False
    ) -> list[Document]:
        """
        Retrieve documents based on a query.

        Args:
        ----
            query: Query to retrieve documents
            limit: Maximum number of documents to retrieve (optional)
            fast_mode: Whether to use fast retrieval mode (bypasses some steps for better performance)

        Returns:
        -------
            List[Document]: Retrieved documents

        """
        trace_id = None
        span = None
        if (
            self._monitoring_service
            and hasattr(self._monitoring_service, "enabled")
            and self._monitoring_service.enabled
        ):
            if hasattr(self._monitoring_service, "create_trace"):
                trace = self._monitoring_service.create_trace()
                if trace and hasattr(trace, "get"):
                    trace_id = trace.get("trace_id")

                if hasattr(self._monitoring_service, "start_span"):
                    span = self._monitoring_service.start_span(
                        name="retrieve",
                        trace_id=trace_id,
                        attributes={
                            "component": "agent",
                            "query": query,
                            "limit": limit or self.config.retrieval_max_documents,
                            "fast_mode": fast_mode,
                        },
                    )

        try:
            if not self._retrieval_service:
                raise ValueError("Retrieval service not available")

            if not hasattr(self._retrieval_service, "retrieve"):
                raise ValueError("Retrieval service does not support retrieve method")

            # Prepare the parameters for the retrieve method
            retrieve_params = {
                "query": query,
                "limit": limit,
            }

            # Check if the retrieval service supports fast_mode parameter
            supports_fast_mode = False
            if hasattr(self._retrieval_service, "retrieve"):
                # Check if the method is a callable with a signature we can inspect
                if callable(self._retrieval_service.retrieve):
                    # Try to inspect the function signature
                    try:
                        import inspect

                        sig = inspect.signature(self._retrieval_service.retrieve)
                        supports_fast_mode = "fast_mode" in sig.parameters
                    except (ValueError, TypeError):
                        # If we can't inspect the signature, try to check the code object
                        if hasattr(self._retrieval_service.retrieve, "__code__"):
                            supports_fast_mode = (
                                "fast_mode" in self._retrieval_service.retrieve.__code__.co_varnames
                            )

            # Add fast_mode parameter if supported and requested
            if fast_mode and supports_fast_mode:
                retrieve_params["fast_mode"] = fast_mode
                logger.debug(f"Using fast_mode={fast_mode} for retrieval")

            # Call the retrieve method with the appropriate parameters
            # Use dynamic attribute access to avoid type errors
            retrieve_method = self._retrieval_service.retrieve
            documents = await retrieve_method(**retrieve_params)

            logger.info(
                f"Retrieved {len(documents)} documents for query: {query} (fast_mode={fast_mode})"
            )
            return documents
        finally:
            if (
                trace_id
                and span
                and self._monitoring_service
                and hasattr(self._monitoring_service, "end_span")
            ):
                self._monitoring_service.end_span(span.span_id)

    async def plan(self, task: str, context: list[Document] | None = None) -> list[PlanStep]:
        """
        Create a plan for a task.

        Args:
        ----
            task: Task description
            context: Context documents (optional)

        Returns:
        -------
            List[PlanStep]: Plan steps

        """
        trace_id = None
        span = None
        if (
            self._monitoring_service
            and hasattr(self._monitoring_service, "enabled")
            and self._monitoring_service.enabled
        ):
            if hasattr(self._monitoring_service, "create_trace"):
                trace = self._monitoring_service.create_trace()
                if trace and hasattr(trace, "get"):
                    trace_id = trace.get("trace_id")

                if hasattr(self._monitoring_service, "start_span"):
                    span = self._monitoring_service.start_span(
                        name="plan",
                        trace_id=trace_id,
                        attributes={
                            "component": "agent",
                            "task": task,
                        },
                    )

        try:
            if not self._planner_service:
                raise ValueError("Planner service not available")

            if not hasattr(self._planner_service, "create_plan"):
                raise ValueError("Planner service does not support create_plan method")

            plan = await self._planner_service.create_plan(
                task=task,
                context=context,
                trace_id=trace_id,
            )

            # No need to check if already awaited result is awaitable

            logger.info(f"Created plan with {len(plan)} steps for task: {task}")
            return list(plan)
        finally:
            if (
                trace_id
                and span
                and self._monitoring_service
                and hasattr(self._monitoring_service, "end_span")
            ):
                self._monitoring_service.end_span(span.span_id)

    async def execute(
        self, prompt: str, context: list[Document] | None = None, use_tools: bool = True
    ) -> dict[str, Any]:
        """
        Execute a prompt with the agent.

        Args:
        ----
            prompt: Prompt to execute
            context: Context documents (optional)
            use_tools: Whether to enable tool usage (default: True)

        Returns:
        -------
            Dict[str, Any]: Execution result

        """
        trace_id = None
        span = None
        if (
            self._monitoring_service
            and hasattr(self._monitoring_service, "enabled")
            and self._monitoring_service.enabled
        ):
            if hasattr(self._monitoring_service, "create_trace"):
                trace = self._monitoring_service.create_trace()
                if trace and hasattr(trace, "get"):
                    trace_id = trace.get("trace_id")

                if hasattr(self._monitoring_service, "start_span"):
                    span = self._monitoring_service.start_span(
                        name="execute",
                        trace_id=trace_id,
                        attributes={
                            "component": "agent",
                            "prompt": prompt,
                            "use_tools": use_tools,
                        },
                    )

        try:
            # Retrieve context if not provided
            if context is None and prompt:
                context_span = None
                if (
                    self._monitoring_service
                    and hasattr(self._monitoring_service, "enabled")
                    and self._monitoring_service.enabled
                    and trace_id
                    and hasattr(self._monitoring_service, "start_span")
                ):
                    context_span = self._monitoring_service.start_span(
                        name="retrieve_context",
                        trace_id=trace_id,
                        parent_id=span.span_id if span else None,
                        attributes={"component": "retriever"},
                    )

                context = await self.retrieve(prompt)

                if (
                    self._monitoring_service
                    and hasattr(self._monitoring_service, "enabled")
                    and self._monitoring_service.enabled
                    and trace_id
                    and context_span
                    and hasattr(self._monitoring_service, "end_span")
                ):
                    self._monitoring_service.end_span(context_span.span_id)

            # Prepare functions for tool calling if enabled
            functions = None
            if use_tools and self._tool_service:
                functions_span = None
                if (
                    self._monitoring_service
                    and hasattr(self._monitoring_service, "enabled")
                    and self._monitoring_service.enabled
                    and trace_id
                    and hasattr(self._monitoring_service, "start_span")
                ):
                    functions_span = self._monitoring_service.start_span(
                        name="prepare_functions",
                        trace_id=trace_id,
                        parent_id=span.span_id if span else None,
                        attributes={"component": "tools"},
                    )

                # Get functions from tool service
                if hasattr(self._tool_service, "prepare_functions_for_model"):
                    functions = self._tool_service.prepare_functions_for_model()

                if (
                    self._monitoring_service
                    and hasattr(self._monitoring_service, "enabled")
                    and self._monitoring_service.enabled
                    and trace_id
                    and functions_span
                    and hasattr(self._monitoring_service, "end_span")
                ):
                    self._monitoring_service.end_span(functions_span.span_id)

            # Execute with execution service
            execution_span = None
            if (
                self._monitoring_service
                and hasattr(self._monitoring_service, "enabled")
                and self._monitoring_service.enabled
                and trace_id
                and hasattr(self._monitoring_service, "start_span")
            ):
                execution_span = self._monitoring_service.start_span(
                    name="execute_prompt",
                    trace_id=trace_id,
                    parent_id=span.span_id if span else None,
                    attributes={
                        "component": "executor",
                        "with_tools": bool(functions),
                    },
                )

            if not self._execution_service:
                raise ValueError("Execution service not available")

            if not hasattr(self._execution_service, "execute"):
                raise ValueError("Execution service does not support execute method")

            result = await self._execution_service.execute(
                prompt=prompt,
                documents=context,
                functions=functions,
                function_call="auto" if functions else None,
                trace_id=trace_id,
            )

            if (
                self._monitoring_service
                and hasattr(self._monitoring_service, "enabled")
                and self._monitoring_service.enabled
                and trace_id
                and execution_span
                and hasattr(self._monitoring_service, "end_span")
            ):
                self._monitoring_service.end_span(execution_span.span_id)

            # Handle tool calls if present
            tool_results = []
            if hasattr(result, "tool_calls") and result.tool_calls:
                tool_span = None
                if (
                    self._monitoring_service
                    and hasattr(self._monitoring_service, "enabled")
                    and self._monitoring_service.enabled
                    and trace_id
                    and hasattr(self._monitoring_service, "start_span")
                ):
                    tool_span = self._monitoring_service.start_span(
                        name="execute_tools",
                        trace_id=trace_id,
                        parent_id=span.span_id if span else None,
                        attributes={"component": "tools"},
                    )

                for tool_call in result.tool_calls:
                    tool_name = tool_call["function"]["name"]
                    tool_args = tool_call["function"]["arguments"]

                    # Log the tool call
                    logger.info(f"Tool call: {tool_name} with args: {tool_args}")

                    # Execute the tool if available
                    if self._tool_service and hasattr(self._tool_service, "tools"):
                        tools = self._tool_service.tools
                        if tools and tool_name in tools:
                            try:
                                # Get tool - check if it's a Python in-process function
                                tool = tools[tool_name]
                                is_in_process = hasattr(tool, "func") and callable(tool.func)

                                # For in-process functions, avoid serialization/deserialization
                                if is_in_process:
                                    tool_result = self._execute_in_process_tool(tool, tool_args)
                                else:
                                    # For external tools or non-Python functions, handle normally
                                    # Parse arguments if they're a string
                                    if isinstance(tool_args, str):
                                        import json

                                        try:
                                            tool_args = json.loads(tool_args)
                                        except json.JSONDecodeError:
                                            # If not valid JSON, pass as is
                                            pass

                                    # Call the tool with serialization/deserialization
                                    tool_result = (
                                        tool(**tool_args)
                                        if isinstance(tool_args, dict)
                                        else tool(tool_args)
                                    )

                                # Add to results
                                tool_results.append(
                                    {"tool": tool_name, "args": tool_args, "result": tool_result}
                                )

                                # Log the result
                                logger.info(f"Tool result: {tool_result}")

                            except Exception as e:
                                logger.exception(f"Error executing tool {tool_name}: {e}")
                                tool_results.append(
                                    {"tool": tool_name, "args": tool_args, "error": str(e)}
                                )

                if (
                    self._monitoring_service
                    and hasattr(self._monitoring_service, "enabled")
                    and self._monitoring_service.enabled
                    and trace_id
                    and tool_span
                    and hasattr(self._monitoring_service, "end_span")
                ):
                    self._monitoring_service.end_span(tool_span.span_id)

            # Validate result
            validation_span = None
            if (
                self._monitoring_service
                and hasattr(self._monitoring_service, "enabled")
                and self._monitoring_service.enabled
                and trace_id
                and hasattr(self._monitoring_service, "start_span")
            ):
                validation_span = self._monitoring_service.start_span(
                    name="validate_result",
                    trace_id=trace_id,
                    parent_id=span.span_id if span else None,
                    attributes={"component": "validator"},
                )

            validation_result = {
                "is_valid": True,
                "score": 1.0,
                "feedback": "No validation performed",
            }

            if self._validator_service and hasattr(self._validator_service, "validate"):
                validation_result = await self._validator_service.validate(
                    input_data={
                        "prompt": prompt,
                        "context": [doc.content for doc in context] if context else [],
                        "tool_results": tool_results,
                    },
                    output_data=result.text,
                    validation_type="execution",
                    trace_id=trace_id,
                )

            if (
                self._monitoring_service
                and hasattr(self._monitoring_service, "enabled")
                and self._monitoring_service.enabled
                and trace_id
                and validation_span
                and hasattr(self._monitoring_service, "end_span")
            ):
                self._monitoring_service.end_span(validation_span.span_id)

            # Collect success pair if valid
            is_valid = (
                validation_result.get("is_valid", False)
                if isinstance(validation_result, dict)
                else getattr(validation_result, "is_valid", False)
            )

            if (
                self.config.enable_self_healing
                and is_valid
                and self._self_healing_service
                and hasattr(self._self_healing_service, "collect_success_pair")
            ):
                score = (
                    validation_result.get("score", 0.0)
                    if isinstance(validation_result, dict)
                    else getattr(validation_result, "score", 0.0)
                )

                await self._self_healing_service.collect_success_pair(
                    input_text=prompt,
                    output_text=result.text,
                    context=[doc.content for doc in context] if context else [],
                    metadata={
                        "validation_score": score,
                        "timestamp": datetime.now().isoformat(),
                    },
                    trace_id=trace_id,
                )

            # Process trace for blame graph if monitoring is enabled
            if (
                self._monitoring_service
                and hasattr(self._monitoring_service, "enabled")
                and self._monitoring_service.enabled
                and trace_id
                and hasattr(self._monitoring_service, "process_trace")
            ):
                self._monitoring_service.process_trace(trace_id)

            # Prepare validation info for return
            if isinstance(validation_result, dict):
                validation_info = {
                    "is_valid": validation_result.get("is_valid", True),
                    "score": validation_result.get("score", 1.0),
                    "feedback": validation_result.get("feedback", "No feedback available"),
                }
            else:
                validation_info = {
                    "is_valid": getattr(validation_result, "is_valid", True),
                    "score": getattr(validation_result, "score", 1.0),
                    "feedback": getattr(validation_result, "feedback", "No feedback available"),
                }

            return {
                "text": result.text,
                "validation": validation_info,
                "tool_results": tool_results,
                "trace_id": trace_id,
            }
        finally:
            if (
                trace_id
                and span
                and self._monitoring_service
                and hasattr(self._monitoring_service, "end_span")
            ):
                self._monitoring_service.end_span(span.span_id)
