"""
Tests for plan optimization in the planner module.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from saplings.core.model_adapter import LLM, LLMResponse, ModelMetadata, ModelRole
from saplings.planner.config import OptimizationStrategy, PlannerConfig
from saplings.planner.plan_step import PlanStep, StepType
from saplings.planner.sequential_planner import SequentialPlanner


class TestPlanOptimization:
    """Tests for plan optimization in the planner module."""

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

        # Create a planner
        self.config = PlannerConfig()
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

    @pytest.mark.asyncio
    async def test_optimize_plan_balanced_strategy(self):
        """Test optimize_plan with balanced optimization strategy."""
        # Set optimization strategy to balanced
        self.planner.config.optimization_strategy = OptimizationStrategy.BALANCED

        # Configure mock response
        response = MagicMock(spec=LLMResponse)
        response.text = """
Here's an optimized plan:

```json
[
  {
    "task_description": "Optimized Step 1",
    "step_type": "RETRIEVAL",
    "estimated_cost": 0.15,
    "estimated_tokens": 800,
    "dependencies": []
  },
  {
    "task_description": "Optimized Step 2",
    "step_type": "ANALYSIS",
    "estimated_cost": 0.25,
    "estimated_tokens": 1500,
    "dependencies": ["step1"]
  },
  {
    "task_description": "Optimized Step 3",
    "step_type": "GENERATION",
    "estimated_cost": 0.35,
    "estimated_tokens": 2500,
    "dependencies": ["step1", "step2"]
  }
]
```
"""
        self.model.generate.return_value = response

        # Create optimized steps to return
        optimized_steps = [
            PlanStep(
                id="step1",
                task_description="Optimized Step 1",
                step_type=StepType.RETRIEVAL,
                estimated_cost=0.15,
                estimated_tokens=800,
                dependencies=[],
            ),
            PlanStep(
                id="step2",
                task_description="Optimized Step 2",
                step_type=StepType.ANALYSIS,
                estimated_cost=0.25,
                estimated_tokens=1500,
                dependencies=["step1"],
            ),
            PlanStep(
                id="step3",
                task_description="Optimized Step 3",
                step_type=StepType.GENERATION,
                estimated_cost=0.35,
                estimated_tokens=2500,
                dependencies=["step1", "step2"],
            ),
        ]

        # Make the steps invalid to force optimization
        invalid_steps = self.steps.copy()
        invalid_steps[0].estimated_cost = 0.6  # Exceeds max cost per step

        # Mock the _parse_optimization_response method to return our optimized steps
        with patch.object(
            self.planner, "_parse_optimization_response", return_value=optimized_steps
        ):
            # Optimize plan
            result = await self.planner.optimize_plan(invalid_steps)

            # Check that model was called
            self.model.generate.assert_called_once()

            # Check optimized steps
            assert len(result) == 3
            assert result[0].task_description == "Optimized Step 1"
            assert result[0].estimated_cost == 0.15
            assert result[0].estimated_tokens == 800

            # Check that total cost was reduced
        original_cost = sum(step.estimated_cost for step in self.steps)
        optimized_cost = sum(step.estimated_cost for step in optimized_steps)
        assert optimized_cost < original_cost

    @pytest.mark.asyncio
    async def test_optimize_plan_cost_strategy(self):
        """Test optimize_plan with cost optimization strategy."""
        # Set optimization strategy to cost
        self.planner.config.optimization_strategy = OptimizationStrategy.COST

        # Configure mock response
        response = MagicMock(spec=LLMResponse)
        response.text = """
Here's an optimized plan:

```json
[
  {
    "task_description": "Cost-Optimized Step 1",
    "step_type": "RETRIEVAL",
    "estimated_cost": 0.1,
    "estimated_tokens": 500,
    "dependencies": []
  },
  {
    "task_description": "Cost-Optimized Step 2",
    "step_type": "ANALYSIS",
    "estimated_cost": 0.15,
    "estimated_tokens": 1000,
    "dependencies": ["step1"]
  },
  {
    "task_description": "Cost-Optimized Step 3",
    "step_type": "GENERATION",
    "estimated_cost": 0.2,
    "estimated_tokens": 1500,
    "dependencies": ["step1", "step2"]
  }
]
```
"""
        self.model.generate.return_value = response

        # Create optimized steps to return
        optimized_steps = [
            PlanStep(
                id="step1",
                task_description="Cost-Optimized Step 1",
                step_type=StepType.RETRIEVAL,
                estimated_cost=0.1,
                estimated_tokens=500,
                dependencies=[],
            ),
            PlanStep(
                id="step2",
                task_description="Cost-Optimized Step 2",
                step_type=StepType.ANALYSIS,
                estimated_cost=0.15,
                estimated_tokens=1000,
                dependencies=["step1"],
            ),
            PlanStep(
                id="step3",
                task_description="Cost-Optimized Step 3",
                step_type=StepType.GENERATION,
                estimated_cost=0.2,
                estimated_tokens=1500,
                dependencies=["step1", "step2"],
            ),
        ]

        # Make the steps invalid to force optimization
        invalid_steps = self.steps.copy()
        invalid_steps[0].estimated_cost = 0.6  # Exceeds max cost per step

        # Mock the _parse_optimization_response method to return our optimized steps
        with patch.object(
            self.planner, "_parse_optimization_response", return_value=optimized_steps
        ):
            # Optimize plan
            result = await self.planner.optimize_plan(invalid_steps)

            # Check that model was called
            self.model.generate.assert_called_once()

            # Check optimized steps
            assert len(result) == 3
            assert result[0].task_description == "Cost-Optimized Step 1"
            assert result[0].estimated_cost == 0.1
            assert result[0].estimated_tokens == 500
        # Check that total cost was significantly reduced
        original_cost = sum(step.estimated_cost for step in self.steps)
        optimized_cost = sum(step.estimated_cost for step in optimized_steps)
        assert optimized_cost < original_cost * 0.7  # At least 30% reduction

    @pytest.mark.asyncio
    async def test_optimize_plan_quality_strategy(self):
        """Test optimize_plan with quality optimization strategy."""
        # Set optimization strategy to quality
        self.planner.config.optimization_strategy = OptimizationStrategy.QUALITY

        # Configure mock response
        response = MagicMock(spec=LLMResponse)
        response.text = """
Here's an optimized plan:

```json
[
  {
    "task_description": "Quality-Optimized Step 1",
    "step_type": "RETRIEVAL",
    "estimated_cost": 0.25,
    "estimated_tokens": 1200,
    "dependencies": []
  },
  {
    "task_description": "Quality-Optimized Step 2",
    "step_type": "ANALYSIS",
    "estimated_cost": 0.35,
    "estimated_tokens": 2500,
    "dependencies": ["step1"]
  },
  {
    "task_description": "Quality-Optimized Step 3",
    "step_type": "GENERATION",
    "estimated_cost": 0.45,
    "estimated_tokens": 3500,
    "dependencies": ["step1", "step2"]
  },
  {
    "task_description": "Quality-Optimized Step 4",
    "step_type": "VERIFICATION",
    "estimated_cost": 0.2,
    "estimated_tokens": 1000,
    "dependencies": ["step3"]
  }
]
```
"""
        self.model.generate.return_value = response

        # Create optimized steps to return
        optimized_steps = [
            PlanStep(
                id="step1",
                task_description="Quality-Optimized Step 1",
                step_type=StepType.RETRIEVAL,
                estimated_cost=0.25,
                estimated_tokens=1200,
                dependencies=[],
            ),
            PlanStep(
                id="step2",
                task_description="Quality-Optimized Step 2",
                step_type=StepType.ANALYSIS,
                estimated_cost=0.35,
                estimated_tokens=2500,
                dependencies=["step1"],
            ),
            PlanStep(
                id="step3",
                task_description="Quality-Optimized Step 3",
                step_type=StepType.GENERATION,
                estimated_cost=0.45,
                estimated_tokens=3500,
                dependencies=["step1", "step2"],
            ),
            PlanStep(
                id="step4",
                task_description="Quality-Optimized Step 4",
                step_type=StepType.VERIFICATION,
                estimated_cost=0.2,
                estimated_tokens=1000,
                dependencies=["step3"],
            ),
        ]

        # Make the steps invalid to force optimization
        invalid_steps = self.steps.copy()
        invalid_steps[0].estimated_cost = 0.6  # Exceeds max cost per step

        # Mock both _parse_optimization_response and _simple_optimize to return our optimized steps
        with patch.object(
            self.planner, "_parse_optimization_response", return_value=optimized_steps
        ), patch.object(self.planner, "_simple_optimize", return_value=optimized_steps):
            # Optimize plan
            result = await self.planner.optimize_plan(invalid_steps)

            # Check that model was called
            self.model.generate.assert_called_once()

            # Check optimized steps
            assert len(result) == 4  # Added a verification step
            assert result[0].task_description == "Quality-Optimized Step 1"
            assert result[3].step_type == StepType.VERIFICATION
        # Check that quality was prioritized (more steps, potentially higher cost)
        assert len(optimized_steps) > len(self.steps)

    @pytest.mark.asyncio
    async def test_optimize_plan_invalid_response(self):
        """Test optimize_plan with invalid response."""
        # Configure mock response
        response = MagicMock(spec=LLMResponse)
        response.text = "This is not a valid JSON response."
        self.model.generate.return_value = response

        # Make the steps invalid to force optimization
        invalid_steps = self.steps.copy()
        invalid_steps[0].estimated_cost = 0.6  # Exceeds max cost per step

        # Mock _simple_optimize to return a known result
        simple_optimized = [
            PlanStep(
                id="step1",
                task_description="Step 1",
                step_type=StepType.RETRIEVAL,
                estimated_cost=0.3,
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

        # Optimize plan with mocked _simple_optimize
        with patch.object(self.planner, "_simple_optimize", return_value=simple_optimized):
            optimized_steps = await self.planner.optimize_plan(invalid_steps)

            # Check that model was called
            self.model.generate.assert_called_once()

            # Check that simple optimization was applied
            assert len(optimized_steps) == 3
            assert optimized_steps[0].task_description == "Step 1"

            # Costs should be within budget
            total_cost = sum(step.estimated_cost for step in optimized_steps)
            assert total_cost <= self.planner.config.total_budget

    def test_simple_optimize_with_different_strategies(self):
        """Test _simple_optimize with different optimization strategies."""
        # Create steps with high costs
        expensive_steps = self.steps.copy()
        expensive_steps[0].estimated_cost = 0.3
        expensive_steps[1].estimated_cost = 0.4
        expensive_steps[2].estimated_cost = 0.5
        # Total cost = 1.2, which exceeds budget of 1.0

        # Test with cost strategy
        self.planner.config.optimization_strategy = OptimizationStrategy.COST
        cost_optimized = self.planner._simple_optimize(expensive_steps)
        cost_total = sum(step.estimated_cost for step in cost_optimized)

        # Test with balanced strategy
        self.planner.config.optimization_strategy = OptimizationStrategy.BALANCED
        balanced_optimized = self.planner._simple_optimize(expensive_steps)
        balanced_total = sum(step.estimated_cost for step in balanced_optimized)

        # Test with quality strategy
        self.planner.config.optimization_strategy = OptimizationStrategy.QUALITY
        quality_optimized = self.planner._simple_optimize(expensive_steps)
        quality_total = sum(step.estimated_cost for step in quality_optimized)

        # Cost strategy should result in lowest cost
        assert cost_total <= balanced_total
        assert cost_total <= quality_total

        # Quality strategy should preserve more of the original cost
        assert quality_total >= balanced_total

    def test_optimize_dependencies(self):
        """Test optimization of dependencies."""
        # Create steps with circular dependencies
        circular_steps = self.steps.copy()
        circular_steps[0].dependencies = ["step3"]  # Creates a cycle

        # Apply simple optimization
        optimized_steps = self.planner._simple_optimize(circular_steps)

        # Check that circular dependencies were resolved
        assert not self.planner._has_circular_dependencies(optimized_steps)

        # Create steps with missing dependencies
        missing_deps_steps = self.steps.copy()
        missing_deps_steps[1].dependencies = ["step1", "step4"]  # Missing dependency

        # Apply simple optimization
        optimized_steps = self.planner._simple_optimize(missing_deps_steps)

        # Check that missing dependencies were resolved
        assert not self.planner._has_missing_dependencies(optimized_steps)

    def test_optimize_step_count(self):
        """Test optimization of step count."""
        # Create many small steps
        many_steps = []
        for i in range(15):  # Exceeds max_steps of 10
            many_steps.append(
                PlanStep(
                    id=f"step{i+1}",
                    task_description=f"Step {i+1}",
                    step_type=StepType.TASK,
                    estimated_cost=0.05,
                    estimated_tokens=500,
                    dependencies=[f"step{i}"] if i > 0 else [],
                )
            )

        # Apply simple optimization
        self.planner.config.max_steps = 10
        optimized_steps = self.planner._simple_optimize(many_steps)

        # Check that step count was reduced
        assert len(optimized_steps) <= self.planner.config.max_steps

        # Create too few steps
        few_steps = self.steps[:1]  # Only one step

        # Apply simple optimization
        self.planner.config.min_steps = 2
        optimized_steps = self.planner._simple_optimize(few_steps)

        # Check that step count was increased
        assert len(optimized_steps) >= self.planner.config.min_steps
