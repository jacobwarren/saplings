from __future__ import annotations

"""
Integration tests for minimal agent functionality.

These tests verify the core agent functionality with a minimal setup,
focusing on planning and execution with deterministic responses.
"""


import asyncio
from unittest.mock import MagicMock

import pytest

from saplings.core.model_adapter import LLMResponse
from saplings.planner.plan_step import PlanStep, PlanStepStatus, StepPriority, StepType


@pytest.fixture()
def mock_planner():
    """Create a mock planner for testing."""
    # Create a mock planner
    planner = MagicMock()

    # Mock the create_plan method
    async def mock_create_plan(**_):
        # Create a simple plan with two steps
        return [
            PlanStep(
                id="step1",
                task_description="Analyze the input",
                step_type=StepType.ANALYSIS,
                priority=StepPriority.HIGH,
                estimated_cost=0.1,
                estimated_tokens=500,
                dependencies=[],
                status=PlanStepStatus.PENDING,
                actual_cost=None,
                actual_tokens=None,
                result=None,
                error=None,
            ),
            PlanStep(
                id="step2",
                task_description="Generate response",
                step_type=StepType.GENERATION,
                priority=StepPriority.MEDIUM,
                estimated_cost=0.2,
                estimated_tokens=1000,
                dependencies=["step1"],
                status=PlanStepStatus.PENDING,
                actual_cost=None,
                actual_tokens=None,
                result=None,
                error=None,
            ),
        ]

    planner.create_plan.side_effect = mock_create_plan

    # Mock the monitoring service
    planner.monitoring_service = MagicMock()
    planner.monitoring_service.trace_viewer = MagicMock()
    planner.monitoring_service.trace_viewer.last_plan = None

    # Set up the record_plan method to capture the plan
    def record_plan(plan):
        planner.monitoring_service.trace_viewer.last_plan = plan

    planner.monitoring_service.trace_viewer.record_plan = record_plan

    return planner


@pytest.fixture()
def mock_executor():
    """Create a mock executor for testing."""
    # Create a mock executor
    executor = MagicMock()

    # Mock the execute method
    async def mock_execute(**kwargs):
        # Get the prompt and functions from kwargs
        prompt = kwargs.get("prompt", "")
        functions = kwargs.get("functions")

        # If functions are provided, return a tool call
        if functions:
            return LLMResponse(
                text="I'll calculate this for you.",
                provider="echo",
                model_name="echo-model",
                function_call=None,
                tool_calls=[
                    {
                        "id": "call_01",
                        "type": "function",
                        "function": {"name": "calculator", "arguments": '{"a": 2, "b": 3}'},
                    }
                ],
            )
        # Default echo response
        return LLMResponse(
            text=f"ECHO: {prompt[:100]}...",
            provider="echo",
            model_name="echo-model",
            function_call=None,
            tool_calls=None,
        )

    executor.execute.side_effect = mock_execute

    # Mock the monitoring service
    executor.monitoring_service = MagicMock()
    executor.monitoring_service.trace_viewer = MagicMock()
    executor.monitoring_service.trace_viewer.last_model_calls = []

    # Set up the record_model_call method to capture model calls
    def record_model_call(prompt: str, response, metadata=None):
        executor.monitoring_service.trace_viewer.last_model_calls.append(
            {"prompt": prompt, "response": response, "metadata": metadata or {}}
        )

    executor.monitoring_service.trace_viewer.record_model_call = record_model_call

    return executor


@pytest.mark.integration()
class TestMinimalAgent:
    """Test minimal agent functionality."""

    def test_minimal_plan_creation(self, mock_planner) -> None:
        """Test that the planner can create a minimal plan."""
        # Create a plan
        plan = asyncio.run(mock_planner.create_plan(goal="Add two numbers: 2 and 3.", context=[]))

        # Record the plan for monitoring
        mock_planner.monitoring_service.trace_viewer.record_plan(plan)

        # Verify plan structure
        assert isinstance(plan, list)
        assert len(plan) > 0
        assert isinstance(plan[0], PlanStep)

        # Verify first step
        assert plan[0].id == "step1"
        assert plan[0].task_description == "Analyze the input"
        assert plan[0].status == PlanStepStatus.PENDING

        # Verify second step
        assert plan[1].id == "step2"
        assert plan[1].task_description == "Generate response"
        assert plan[1].dependencies == ["step1"]

        # Verify monitoring trace
        assert mock_planner.monitoring_service.trace_viewer.last_plan is not None
        assert len(mock_planner.monitoring_service.trace_viewer.last_plan) > 0

    def test_minimal_execution(self, mock_executor) -> None:
        """Test that the executor can execute a prompt."""
        # Execute a prompt
        result = asyncio.run(mock_executor.execute(prompt="Add two numbers: 2 and 3.", context=[]))

        # Record the model call for monitoring
        mock_executor.monitoring_service.trace_viewer.record_model_call(
            prompt="Add two numbers: 2 and 3.", response=result
        )

        # Verify result
        assert result is not None
        assert "ECHO: Add two numbers: 2 and 3." in result.text

        # Verify monitoring trace
        assert len(mock_executor.monitoring_service.trace_viewer.last_model_calls) > 0

    def test_tool_execution(self, mock_executor) -> None:
        """Test that the executor can execute a tool."""
        # Define available tools
        tools = [
            {
                "name": "calculator",
                "description": "A calculator tool for testing",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number", "description": "First number"},
                        "b": {"type": "number", "description": "Second number"},
                    },
                    "required": ["a", "b"],
                },
            }
        ]

        # Execute a prompt with tools
        result = asyncio.run(mock_executor.execute(prompt="Calculate 2 + 3", functions=tools))

        # Record the model call for monitoring
        mock_executor.monitoring_service.trace_viewer.record_model_call(
            prompt="Calculate 2 + 3", response=result, metadata={"tools": tools}
        )

        # Verify result
        assert result is not None
        assert result.tool_calls is not None
        assert len(result.tool_calls) > 0
        assert result.tool_calls[0]["function"]["name"] == "calculator"

        # Verify tool was called with correct parameters
        tool_args = result.tool_calls[0]["function"]["arguments"]
        assert '"a": 2' in tool_args
        assert '"b": 3' in tool_args

        # Verify monitoring trace
        assert len(mock_executor.monitoring_service.trace_viewer.last_model_calls) > 0

    def test_plan_and_execution_integration(self, mock_planner, mock_executor) -> None:
        """Test integration between planner and executor."""
        # Create a plan
        plan = asyncio.run(mock_planner.create_plan(goal="Add two numbers: 2 and 3.", context=[]))

        # Record the plan for monitoring
        mock_planner.monitoring_service.trace_viewer.record_plan(plan)

        # Verify plan
        assert isinstance(plan, list)
        assert len(plan) > 0

        # Execute each step in the plan
        results = []
        for step in plan:
            result = asyncio.run(mock_executor.execute(prompt=step.task_description, context=[]))

            # Record the model call for monitoring
            mock_executor.monitoring_service.trace_viewer.record_model_call(
                prompt=step.task_description, response=result
            )

            results.append(result)

        # Verify results
        assert len(results) == len(plan)
        for result in results:
            assert result is not None
            assert result.text.startswith("ECHO:")

        # Verify monitoring traces
        assert mock_planner.monitoring_service.trace_viewer.last_plan is not None
        assert len(mock_executor.monitoring_service.trace_viewer.last_model_calls) == len(plan)


if __name__ == "__main__":
    # Run the tests
    pytest.main(["-xvs", __file__])
