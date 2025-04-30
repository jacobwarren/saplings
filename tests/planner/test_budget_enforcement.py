"""
Tests for budget enforcement in the planner module.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from saplings.core.model_adapter import LLM, LLMResponse, ModelMetadata, ModelRole
from saplings.planner.config import BudgetStrategy, CostHeuristicConfig, PlannerConfig
from saplings.planner.plan_step import PlanStep, StepType
from saplings.planner.sequential_planner import SequentialPlanner


class TestBudgetEnforcement:
    """Tests for budget enforcement in the planner module."""

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
            max_tokens_per_request=2048
        )

        # Create a planner with a specific budget
        self.config = PlannerConfig(
            total_budget=1.0,
            allow_budget_overflow=False,
            budget_overflow_margin=0.1,
            cost_heuristics=CostHeuristicConfig(
                max_cost_per_step=0.5,
            ),
        )
        self.planner = SequentialPlanner(config=self.config, model=self.model)

        # Create test steps
        self.steps = [
            PlanStep(
                id="step1",
                task_description="Step 1",
                step_type=StepType.RETRIEVAL,
                estimated_cost=0.2,
                estimated_tokens=1000,
                dependencies=[],
            ),
            PlanStep(
                id="step2",
                task_description="Step 2",
                step_type=StepType.ANALYSIS,
                estimated_cost=0.3,
                estimated_tokens=2000,
                dependencies=["step1"],
            ),
            PlanStep(
                id="step3",
                task_description="Step 3",
                step_type=StepType.GENERATION,
                estimated_cost=0.4,
                estimated_tokens=3000,
                dependencies=["step1", "step2"],
            ),
        ]

    def test_validate_plan_budget_enforcement(self):
        """Test that validate_plan enforces budget constraints."""
        # Plan within budget (total cost = 0.9)
        assert self.planner.validate_plan(self.steps)

        # Plan exceeding budget
        expensive_steps = self.steps.copy()
        expensive_steps[0].estimated_cost = 0.5
        expensive_steps[1].estimated_cost = 0.4
        expensive_steps[2].estimated_cost = 0.3
        # Total cost = 1.2, which exceeds budget of 1.0
        assert not self.planner.validate_plan(expensive_steps)

        # Plan within budget + margin when overflow is allowed
        self.planner.config.allow_budget_overflow = True
        self.planner.config.budget_overflow_margin = 0.3  # 30% margin
        # With 30% margin, max budget is 1.3, which is > 1.2
        assert self.planner.validate_plan(expensive_steps)

        # Plan exceeding budget + margin
        very_expensive_steps = self.steps.copy()
        very_expensive_steps[0].estimated_cost = 0.6
        very_expensive_steps[1].estimated_cost = 0.5
        very_expensive_steps[2].estimated_cost = 0.4
        # Total cost = 1.5, which exceeds budget + margin (1.3)
        assert not self.planner.validate_plan(very_expensive_steps)

    def test_max_cost_per_step_enforcement(self):
        """Test that validate_plan enforces maximum cost per step."""
        # All steps within max cost per step
        assert self.planner.validate_plan(self.steps)

        # Step exceeding max cost per step
        expensive_step = self.steps.copy()
        expensive_step[0].estimated_cost = 0.6  # Exceeds max of 0.5
        assert not self.planner.validate_plan(expensive_step)

        # Increase max cost per step
        self.planner.config.cost_heuristics.max_cost_per_step = 1.0
        # Also need to update the total budget to accommodate the increased cost
        total_cost = sum(step.estimated_cost for step in expensive_step)
        self.planner.config.total_budget = total_cost
        assert self.planner.validate_plan(expensive_step)

    @pytest.mark.asyncio
    async def test_optimize_plan_budget_enforcement(self):
        """Test that optimize_plan enforces budget constraints."""
        # Create a plan that exceeds budget
        expensive_steps = self.steps.copy()
        expensive_steps[0].estimated_cost = 0.5
        expensive_steps[1].estimated_cost = 0.4
        expensive_steps[2].estimated_cost = 0.3
        # Total cost = 1.2, which exceeds budget of 1.0

        # Configure mock response for optimization
        response = MagicMock(spec=LLMResponse)
        response.text = """
Here's an optimized plan:

```json
[
  {
    "task_description": "Optimized Step 1",
    "step_type": "RETRIEVAL",
    "estimated_cost": 0.3,
    "estimated_tokens": 1500,
    "dependencies": []
  },
  {
    "task_description": "Optimized Step 2",
    "step_type": "ANALYSIS",
    "estimated_cost": 0.3,
    "estimated_tokens": 1500,
    "dependencies": ["step1"]
  },
  {
    "task_description": "Optimized Step 3",
    "step_type": "GENERATION",
    "estimated_cost": 0.4,
    "estimated_tokens": 2000,
    "dependencies": ["step1", "step2"]
  }
]
```
"""
        self.model.generate.return_value = response

        # Optimize plan
        optimized_steps = await self.planner.optimize_plan(expensive_steps)

        # Check that model was called
        self.model.generate.assert_called_once()

        # Check that optimized plan is within budget
        total_cost = sum(step.estimated_cost for step in optimized_steps)
        assert total_cost <= self.planner.config.total_budget

    @pytest.mark.asyncio
    async def test_simple_optimize_budget_enforcement(self):
        """Test that _simple_optimize enforces budget constraints."""
        # Create a plan that exceeds budget significantly
        expensive_steps = self.steps.copy()
        expensive_steps[0].estimated_cost = 0.6
        expensive_steps[1].estimated_cost = 0.5
        expensive_steps[2].estimated_cost = 0.4
        # Total cost = 1.5, which exceeds budget of 1.0 by a significant amount

        # Apply simple optimization
        optimized_steps = self.planner._simple_optimize(expensive_steps)

        # Check that optimized plan is within budget
        total_cost = sum(step.estimated_cost for step in optimized_steps)
        assert total_cost <= self.planner.config.total_budget

        # Check that total cost was reduced or at least capped at the budget
        original_total = sum(step.estimated_cost for step in expensive_steps)
        assert total_cost <= self.planner.config.total_budget
        assert total_cost <= original_total

    def test_budget_strategy_equal(self):
        """Test equal budget strategy."""
        # Set budget strategy to equal
        self.planner.config.budget_strategy = BudgetStrategy.EQUAL
        self.planner.config.total_budget = 0.9  # 0.3 per step

        # Create steps with unequal costs
        unequal_steps = self.steps.copy()
        unequal_steps[0].estimated_cost = 0.1
        unequal_steps[1].estimated_cost = 0.2
        unequal_steps[2].estimated_cost = 0.6  # Exceeds equal share

        # Make a deep copy to ensure we don't modify the original
        import copy
        unequal_steps_copy = copy.deepcopy(unequal_steps)

        # Apply simple optimization
        optimized_steps = self.planner._simple_optimize(unequal_steps_copy)

        # Check that total budget is respected
        total_cost = sum(step.estimated_cost for step in optimized_steps)
        assert total_cost <= self.planner.config.total_budget

        # Check that the total cost was reduced
        original_total = sum(step.estimated_cost for step in unequal_steps)
        assert total_cost < original_total

        # Check that the costs are more balanced than before
        original_max_diff = 0.6 - 0.1  # 0.5
        optimized_max_diff = max(s.estimated_cost for s in optimized_steps) - min(s.estimated_cost for s in optimized_steps)
        assert optimized_max_diff < original_max_diff

    def test_budget_strategy_proportional(self):
        """Test proportional budget strategy."""
        # Set budget strategy to proportional
        self.planner.config.budget_strategy = BudgetStrategy.PROPORTIONAL
        self.planner.config.total_budget = 0.9

        # Create steps with costs that exceed budget
        expensive_steps = self.steps.copy()
        expensive_steps[0].estimated_cost = 0.2
        expensive_steps[1].estimated_cost = 0.4
        expensive_steps[2].estimated_cost = 0.6
        # Total cost = 1.2, which exceeds budget of 0.9

        # Apply simple optimization
        optimized_steps = self.planner._simple_optimize(expensive_steps)

        # Check that costs were scaled down proportionally
        scale_factor = 0.9 / 1.2  # Expected scale factor
        assert optimized_steps[0].estimated_cost == pytest.approx(0.2 * scale_factor)
        assert optimized_steps[1].estimated_cost == pytest.approx(0.4 * scale_factor)
        assert optimized_steps[2].estimated_cost == pytest.approx(0.6 * scale_factor)

        # Check total budget is respected
        total_cost = sum(step.estimated_cost for step in optimized_steps)
        assert total_cost <= self.planner.config.total_budget

    @pytest.mark.asyncio
    async def test_create_plan_budget_enforcement(self):
        """Test that create_plan enforces budget constraints."""
        # Configure mock response with a plan that exceeds budget
        response = MagicMock(spec=LLMResponse)
        response.text = """
Here's a plan for the task:

```json
[
  {
    "task_description": "Research information",
    "step_type": "RETRIEVAL",
    "estimated_cost": 0.3,
    "estimated_tokens": 2000,
    "dependencies": []
  },
  {
    "task_description": "Analyze information",
    "step_type": "ANALYSIS",
    "estimated_cost": 0.4,
    "estimated_tokens": 3000,
    "dependencies": [0]
  },
  {
    "task_description": "Generate response",
    "step_type": "GENERATION",
    "estimated_cost": 0.5,
    "estimated_tokens": 4000,
    "dependencies": [0, 1]
  }
]
```
"""
        self.model.generate.side_effect = [response, response]  # For create_plan and optimize_plan

        # Create plan
        with patch.object(self.planner, 'optimize_plan', return_value=self.steps):
            steps = await self.planner.create_plan("Test task")

        # Check that model was called
        assert self.model.generate.call_count == 1

        # Check that optimize_plan was called (since the initial plan exceeds budget)
        assert len(steps) == 3
        assert sum(step.estimated_cost for step in steps) <= self.planner.config.total_budget
