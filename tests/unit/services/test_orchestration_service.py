from __future__ import annotations

"""
Unit tests for the orchestration service.
"""


from unittest.mock import MagicMock

from saplings.core.interfaces import (
    IExecutionService,
    IModelService,
    IOrchestrationService,
    IPlannerService,
    IToolService,
    IValidatorService,
)
from saplings.orchestration.config import OrchestrationConfig
from saplings.planner import PlanStep, PlanStepStatus
from saplings.services.orchestration_service import OrchestrationService


class TestOrchestrationService:
    EXPECTED_COUNT_1 = 3

    """Test the orchestration service."""

    def setup_method(self) -> None:
        """Set up test environment."""
        # Create mock services
        self.mock_model_service = MagicMock(spec=IModelService)
        self.mock_planner_service = MagicMock(spec=IPlannerService)
        self.mock_execution_service = MagicMock(spec=IExecutionService)
        self.mock_tool_service = MagicMock(spec=IToolService)
        self.mock_validator_service = MagicMock(spec=IValidatorService)

        # Mock planner service create_plan
        self.mock_planner_service.create_plan.return_value = [
            PlanStep(
                id="step1",
                description="Search for information about France",
                tool="search",
                tool_input={"query": "France capital"},
                status=PlanStepStatus.PENDING,
            ),
            PlanStep(
                id="step2",
                description="Extract the capital city",
                tool="extract",
                tool_input={"text": "{{step1.result}}"},
                status=PlanStepStatus.PENDING,
            ),
            PlanStep(
                id="step3",
                description="Format the final answer",
                tool="final_answer",
                tool_input={"answer": "The capital of France is {{step2.result}}"},
                status=PlanStepStatus.PENDING,
            ),
        ]

        # Mock tool service
        self.mock_tool_service.get_tool_definitions.return_value = [
            {
                "name": "search",
                "description": "Search for information",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "Search query"}},
                    "required": ["query"],
                },
            },
            {
                "name": "extract",
                "description": "Extract information from text",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to extract from"}
                    },
                    "required": ["text"],
                },
            },
            {
                "name": "final_answer",
                "description": "Provide the final answer",
                "parameters": {
                    "type": "object",
                    "properties": {"answer": {"type": "string", "description": "The final answer"}},
                    "required": ["answer"],
                },
            },
        ]

        # Mock tool execution
        self.mock_tool_service.execute_tool.side_effect = [
            "France's capital is Paris.",  # search
            "Paris",  # extract
            "The capital of France is Paris.",  # final_answer
        ]

        # Mock execution service
        mock_response = MagicMock()
        mock_response.text = "The capital of France is Paris."
        self.mock_execution_service.execute.return_value = mock_response

        # Mock validator service
        mock_validation = MagicMock()
        mock_validation.valid = True
        mock_validation.score = 0.95
        self.mock_validator_service.validate_response.return_value = mock_validation

        # Create orchestration service
        self.config = OrchestrationConfig(
            max_iterations=10, enable_validation=True, validation_threshold=0.7
        )
        self.service = OrchestrationService(
            model_service=self.mock_model_service,
            planner_service=self.mock_planner_service,
            execution_service=self.mock_execution_service,
            tool_service=self.mock_tool_service,
            validator_service=self.mock_validator_service,
            config=self.config,
        )

    def test_initialization(self) -> None:
        """Test orchestration service initialization."""
        assert self.service.model_service is self.mock_model_service
        assert self.service.planner_service is self.mock_planner_service
        assert self.service.execution_service is self.mock_execution_service
        assert self.service.tool_service is self.mock_tool_service
        assert self.service.validator_service is self.mock_validator_service
        assert self.service.config is self.config

    def test_execute_plan(self) -> None:
        """Test executing a plan."""
        # Execute a plan
        result = self.service.execute_plan(
            goal="Find the capital of France", context=["France is a country in Europe."]
        )

        # Verify result
        assert result is not None
        assert result.final_answer == "The capital of France is Paris."
        assert result.completed is True
        assert len(result.steps) == self.EXPECTED_COUNT_1
        assert result.steps[0].status == PlanStepStatus.COMPLETED
        assert result.steps[1].status == PlanStepStatus.COMPLETED
        assert result.steps[2].status == PlanStepStatus.COMPLETED

        # Verify services were called
        self.mock_planner_service.create_plan.assert_called_once()
        assert self.mock_tool_service.execute_tool.call_count == 3
        self.mock_validator_service.validate_response.assert_called_once()

    def test_execute_plan_with_validation_failure(self) -> None:
        """Test executing a plan with validation failure."""
        # Mock validator service to fail
        mock_validation = MagicMock()
        mock_validation.valid = False
        mock_validation.score = 0.5
        self.mock_validator_service.validate_response.return_value = mock_validation

        # Mock planner service to revise the plan
        self.mock_planner_service.revise_plan.return_value = [
            PlanStep(
                id="step1",
                description="Search for information about France",
                tool="search",
                tool_input={"query": "France capital city"},
                status=PlanStepStatus.COMPLETED,
                result="France's capital is Paris.",
            ),
            PlanStep(
                id="step2",
                description="Verify the capital city",
                tool="search",
                tool_input={"query": "Paris capital of France confirm"},
                status=PlanStepStatus.PENDING,
            ),
            PlanStep(
                id="step3",
                description="Format the final answer",
                tool="final_answer",
                tool_input={
                    "answer": "The capital of France is Paris, confirmed by multiple sources."
                },
                status=PlanStepStatus.PENDING,
            ),
        ]

        # Update tool execution for the revised plan
        self.mock_tool_service.execute_tool.side_effect = [
            "France's capital is Paris.",  # search (step1)
            "Paris is indeed the capital of France.",  # search (step2)
            "The capital of France is Paris, confirmed by multiple sources.",  # final_answer
        ]

        # Mock validator service to pass on second attempt
        self.mock_validator_service.validate_response.side_effect = [
            mock_validation,  # First attempt fails
            MagicMock(valid=True, score=0.9),  # Second attempt passes
        ]

        # Execute a plan
        result = self.service.execute_plan(
            goal="Find the capital of France", context=["France is a country in Europe."]
        )

        # Verify result
        assert result is not None
        assert (
            result.final_answer == "The capital of France is Paris, confirmed by multiple sources."
        )
        assert result.completed is True
        assert len(result.steps) == self.EXPECTED_COUNT_1

        # Verify services were called
        self.mock_planner_service.create_plan.assert_called_once()
        self.mock_planner_service.revise_plan.assert_called_once()
        assert self.mock_validator_service.validate_response.call_count == 2

    def test_execute_plan_with_tool_error(self) -> None:
        """Test executing a plan with a tool error."""
        # Mock tool service to raise an error
        self.mock_tool_service.execute_tool.side_effect = [
            Exception("Tool execution failed"),  # First call fails
            "France's capital is Paris.",  # Second call succeeds
            "Paris",
            "The capital of France is Paris.",
        ]

        # Mock planner service to handle the error
        original_update_plan = self.mock_planner_service.update_plan

        def mock_update_plan(plan, step):
            # Update the step status based on the error
            if step.status == PlanStepStatus.ERROR:
                # Create a new step to retry
                new_step = PlanStep(
                    id=step.id,
                    description=step.description,
                    tool=step.tool,
                    tool_input=step.tool_input,
                    status=PlanStepStatus.PENDING,
                )
                return [new_step] + plan[1:]
            return original_update_plan(plan, step)

        self.mock_planner_service.update_plan = mock_update_plan

        # Execute a plan
        result = self.service.execute_plan(
            goal="Find the capital of France", context=["France is a country in Europe."]
        )

        # Verify result
        assert result is not None
        assert result.final_answer == "The capital of France is Paris."
        assert result.completed is True

        # Verify services were called
        self.mock_planner_service.create_plan.assert_called_once()
        assert self.mock_tool_service.execute_tool.call_count == 4  # Including the retry

    def test_execute_plan_with_max_iterations(self) -> None:
        """Test executing a plan with maximum iterations reached."""
        # Set a low max iterations
        self.service.config.max_iterations = 1

        # Mock validator service to always fail
        mock_validation = MagicMock()
        mock_validation.valid = False
        mock_validation.score = 0.5
        self.mock_validator_service.validate_response.return_value = mock_validation

        # Execute a plan
        result = self.service.execute_plan(
            goal="Find the capital of France", context=["France is a country in Europe."]
        )

        # Verify result
        assert result is not None
        assert result.completed is False  # Plan not completed due to max iterations
        assert result.error == "Maximum iterations reached"

    def test_interface_compliance(self) -> None:
        """Test that OrchestrationService implements IOrchestrationService."""
        assert isinstance(self.service, IOrchestrationService)

        # Check required methods
        assert hasattr(self.service, "execute_plan")
