"""
Tests for task splitting in the planner module.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from saplings.core.model_adapter import LLM, LLMResponse, ModelMetadata, ModelRole
from saplings.planner.config import PlannerConfig
from saplings.planner.plan_step import PlanStep, PlanStepStatus, StepType
from saplings.planner.sequential_planner import SequentialPlanner


class TestTaskSplitting:
    """Tests for task splitting in the planner module."""

    def setup_method(self):
        """Set up test environment."""
        # Create a mock model
        self.model = MagicMock(spec=LLM)
        self.model.generate = AsyncMock()
        self.model.get_metadata.return_value = ModelMetadata(
            name="test-model",
            provider="test-provider",
            version="latest",
            roles=[ModelRole.PLANNER],
            context_window=4096,
            max_tokens_per_request=2048,
        )

        # Create a planner with a higher max_steps value
        self.config = PlannerConfig(max_steps=15)  # Increase max_steps to allow more steps
        self.planner = SequentialPlanner(config=self.config, model=self.model)

    @pytest.mark.asyncio
    async def test_create_plan_task_splitting(self):
        """Test that create_plan properly splits tasks."""
        # Configure mock response with a complex task split into multiple steps
        response = MagicMock(spec=LLMResponse)
        response.text = """
Here's a plan for the task:

```json
[
  {
    "task_description": "Research recent developments in AI",
    "step_type": "RETRIEVAL",
    "estimated_cost": 0.1,
    "estimated_tokens": 1000,
    "dependencies": []
  },
  {
    "task_description": "Research historical context of AI",
    "step_type": "RETRIEVAL",
    "estimated_cost": 0.1,
    "estimated_tokens": 1000,
    "dependencies": []
  },
  {
    "task_description": "Analyze technical aspects",
    "step_type": "ANALYSIS",
    "estimated_cost": 0.15,
    "estimated_tokens": 1500,
    "dependencies": [0]
  },
  {
    "task_description": "Analyze historical trends",
    "step_type": "ANALYSIS",
    "estimated_cost": 0.15,
    "estimated_tokens": 1500,
    "dependencies": [1]
  },
  {
    "task_description": "Synthesize technical and historical analysis",
    "step_type": "ANALYSIS",
    "estimated_cost": 0.2,
    "estimated_tokens": 2000,
    "dependencies": [2, 3]
  },
  {
    "task_description": "Generate comprehensive report",
    "step_type": "GENERATION",
    "estimated_cost": 0.3,
    "estimated_tokens": 3000,
    "dependencies": [4]
  }
]
```
"""
        self.model.generate.return_value = response

        # Create plan
        steps = await self.planner.create_plan("Write a comprehensive report on AI")

        # Check that model was called
        self.model.generate.assert_called_once()

        # Check steps
        assert len(steps) == 6

        # Check that tasks were properly split
        retrieval_steps = [step for step in steps if step.step_type == StepType.RETRIEVAL]
        analysis_steps = [step for step in steps if step.step_type == StepType.ANALYSIS]
        generation_steps = [step for step in steps if step.step_type == StepType.GENERATION]

        assert len(retrieval_steps) == 2  # Split into two retrieval tasks
        assert len(analysis_steps) == 3  # Split into three analysis tasks
        assert len(generation_steps) == 1  # One generation task

        # Check dependencies
        # The first two steps should have no dependencies
        assert steps[0].dependencies == []
        assert steps[1].dependencies == []

        # The analysis steps should depend on retrieval steps
        assert steps[2].dependencies == [steps[0].id]
        assert steps[3].dependencies == [steps[1].id]

        # The synthesis step should depend on both analysis steps
        assert steps[4].dependencies == [steps[2].id, steps[3].id]

        # The generation step should depend on the synthesis step
        assert steps[5].dependencies == [steps[4].id]

    def test_create_simple_plan_task_splitting(self):
        """Test that _create_simple_plan properly splits tasks."""
        # Create a simple plan
        steps = self.planner._create_simple_plan("Write a comprehensive report on AI")

        # Check steps
        assert len(steps) == 3

        # Check that tasks were properly split into retrieval, analysis, and generation
        assert steps[0].step_type == StepType.RETRIEVAL
        assert steps[1].step_type == StepType.ANALYSIS
        assert steps[2].step_type == StepType.GENERATION

        # Check dependencies
        assert steps[0].dependencies == []
        assert steps[1].dependencies == [steps[0].id]
        assert steps[2].dependencies == [steps[0].id, steps[1].id]

    @pytest.mark.asyncio
    async def test_complex_task_splitting(self):
        """Test splitting of a complex task with many subtasks."""
        # Configure mock response with many subtasks
        response = MagicMock(spec=LLMResponse)
        response.text = """
Here's a plan for the task:

```json
[
  {
    "task_description": "Research topic 1",
    "step_type": "RETRIEVAL",
    "estimated_cost": 0.05,
    "estimated_tokens": 500,
    "dependencies": []
  },
  {
    "task_description": "Research topic 2",
    "step_type": "RETRIEVAL",
    "estimated_cost": 0.05,
    "estimated_tokens": 500,
    "dependencies": []
  },
  {
    "task_description": "Research topic 3",
    "step_type": "RETRIEVAL",
    "estimated_cost": 0.05,
    "estimated_tokens": 500,
    "dependencies": []
  },
  {
    "task_description": "Analyze findings from topic 1",
    "step_type": "ANALYSIS",
    "estimated_cost": 0.1,
    "estimated_tokens": 1000,
    "dependencies": [0]
  },
  {
    "task_description": "Analyze findings from topic 2",
    "step_type": "ANALYSIS",
    "estimated_cost": 0.1,
    "estimated_tokens": 1000,
    "dependencies": [1]
  },
  {
    "task_description": "Analyze findings from topic 3",
    "step_type": "ANALYSIS",
    "estimated_cost": 0.1,
    "estimated_tokens": 1000,
    "dependencies": [2]
  },
  {
    "task_description": "Compare analyses from topics 1 and 2",
    "step_type": "ANALYSIS",
    "estimated_cost": 0.15,
    "estimated_tokens": 1500,
    "dependencies": [3, 4]
  },
  {
    "task_description": "Compare analyses from topics 2 and 3",
    "step_type": "ANALYSIS",
    "estimated_cost": 0.15,
    "estimated_tokens": 1500,
    "dependencies": [4, 5]
  },
  {
    "task_description": "Synthesize all analyses",
    "step_type": "ANALYSIS",
    "estimated_cost": 0.2,
    "estimated_tokens": 2000,
    "dependencies": [6, 7]
  },
  {
    "task_description": "Generate draft report",
    "step_type": "GENERATION",
    "estimated_cost": 0.2,
    "estimated_tokens": 2000,
    "dependencies": [8]
  },
  {
    "task_description": "Review and refine report",
    "step_type": "VERIFICATION",
    "estimated_cost": 0.1,
    "estimated_tokens": 1000,
    "dependencies": [9]
  },
  {
    "task_description": "Generate final report",
    "step_type": "GENERATION",
    "estimated_cost": 0.15,
    "estimated_tokens": 1500,
    "dependencies": [10]
  }
]
```
"""
        self.model.generate.return_value = response

        # Create plan
        steps = await self.planner.create_plan(
            "Write a comprehensive report comparing three topics"
        )

        # Check that model was called at least once
        assert self.model.generate.call_count >= 1

        # Check steps
        assert len(steps) == 12

        # Check step types
        retrieval_steps = [step for step in steps if step.step_type == StepType.RETRIEVAL]
        analysis_steps = [step for step in steps if step.step_type == StepType.ANALYSIS]
        generation_steps = [step for step in steps if step.step_type == StepType.GENERATION]
        verification_steps = [step for step in steps if step.step_type == StepType.VERIFICATION]

        assert len(retrieval_steps) == 3
        assert len(analysis_steps) == 6
        assert len(generation_steps) == 2
        assert len(verification_steps) == 1

        # Check execution order
        batches = self.planner.get_execution_order(steps)

        # First batch should be all retrieval steps (no dependencies)
        assert len(batches[0]) == 3
        assert all(step.step_type == StepType.RETRIEVAL for step in batches[0])

        # Last batch should be the final generation step
        assert len(batches[-1]) == 1
        assert batches[-1][0].step_type == StepType.GENERATION
        assert batches[-1][0].task_description == "Generate final report"

    @pytest.mark.asyncio
    async def test_task_splitting_with_tool_use(self):
        """Test splitting of a task that requires tool use."""
        # Configure mock response with tool use steps
        response = MagicMock(spec=LLMResponse)
        response.text = """
Here's a plan for the task:

```json
[
  {
    "task_description": "Search for current weather data",
    "step_type": "RETRIEVAL",
    "estimated_cost": 0.05,
    "estimated_tokens": 500,
    "dependencies": []
  },
  {
    "task_description": "Use weather API to get forecast",
    "step_type": "TOOL_USE",
    "estimated_cost": 0.1,
    "estimated_tokens": 1000,
    "dependencies": [0]
  },
  {
    "task_description": "Search for historical weather patterns",
    "step_type": "RETRIEVAL",
    "estimated_cost": 0.05,
    "estimated_tokens": 500,
    "dependencies": []
  },
  {
    "task_description": "Analyze current and historical data",
    "step_type": "ANALYSIS",
    "estimated_cost": 0.15,
    "estimated_tokens": 1500,
    "dependencies": [1, 2]
  },
  {
    "task_description": "Generate weather report",
    "step_type": "GENERATION",
    "estimated_cost": 0.2,
    "estimated_tokens": 2000,
    "dependencies": [3]
  }
]
```
"""
        self.model.generate.return_value = response

        # Create plan
        steps = await self.planner.create_plan("Create a weather report with forecast")

        # Check that model was called
        self.model.generate.assert_called_once()

        # Check steps
        assert len(steps) == 5

        # Check that there's a tool use step
        tool_use_steps = [step for step in steps if step.step_type == StepType.TOOL_USE]
        assert len(tool_use_steps) == 1
        assert tool_use_steps[0].task_description == "Use weather API to get forecast"

        # Check execution order
        batches = self.planner.get_execution_order(steps)

        # First batch should include both retrieval steps (no dependencies between them)
        first_batch_ids = [step.id for step in batches[0]]
        retrieval_steps = [step for step in steps if step.step_type == StepType.RETRIEVAL]
        for step in retrieval_steps:
            assert step.id in first_batch_ids

        # Tool use step should be in the second batch
        assert tool_use_steps[0].id in [step.id for step in batches[1]]
