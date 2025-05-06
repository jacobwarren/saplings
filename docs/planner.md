# Planning System

The planning system in Saplings breaks down complex tasks into manageable steps while respecting budget constraints and optimizing for efficiency.

## Overview

The planning system consists of several key components:

- **BasePlanner**: Abstract base class that defines the planner interface
- **SequentialPlanner**: Implementation that executes steps in sequence
- **PlanStep**: Represents a single step in a plan
- **PlannerConfig**: Configuration for the planning system
- **BudgetStrategy**: Strategies for budget allocation
- **OptimizationStrategy**: Approaches for optimizing plans

This system enables agents to tackle complex tasks by breaking them down into smaller, more manageable steps, while ensuring that resource constraints are respected.

## Core Concepts

### Plan Steps

A plan step represents a single unit of work in a plan. Each step has:

- **Description**: What needs to be done
- **Type**: The kind of operation (task, retrieval, generation, etc.)
- **Priority**: How important the step is
- **Cost Estimates**: Expected resource usage
- **Dependencies**: Other steps that must be completed first
- **Status**: Current state of the step (pending, in progress, completed, etc.)
- **Result**: Output of the step when completed

### Budget Strategies

Budget strategies determine how resources are allocated across steps:

- **Equal**: Each step gets an equal share of the budget
- **Proportional**: Budget is allocated proportionally to step complexity
- **Dynamic**: Budget is adjusted based on previous steps
- **Fixed**: Each step type has a fixed budget

### Optimization Strategies

Optimization strategies determine how plans are refined:

- **Cost**: Minimize resource usage
- **Quality**: Maximize output quality
- **Balanced**: Balance cost and quality
- **Speed**: Minimize execution time

### Plan Execution

Plan execution involves:

1. **Dependency Resolution**: Determining which steps can be executed
2. **Step Execution**: Running each step and capturing its output
3. **Result Aggregation**: Combining outputs into a final result

## API Reference

### BasePlanner

```python
class BasePlanner(ABC):
    def __init__(
        self,
        config: Optional[PlannerConfig] = None,
        model: Optional[LLM] = None,
        trace_manager: Optional["TraceManager"] = None,
    ):
        """Initialize the planner."""

    @abstractmethod
    async def create_plan(self, task: str, **kwargs) -> List[PlanStep]:
        """Create a plan for a task."""

    @abstractmethod
    async def optimize_plan(self, steps: List[PlanStep], **kwargs) -> List[PlanStep]:
        """Optimize a plan."""

    @abstractmethod
    async def execute_plan(self, steps: List[PlanStep], **kwargs) -> Tuple[bool, Any]:
        """Execute a plan."""

    def validate_plan(self, steps: List[PlanStep]) -> bool:
        """Validate a plan."""

    def get_execution_order(self, steps: List[PlanStep]) -> List[List[PlanStep]]:
        """Get the execution order for a plan."""

    def estimate_cost(self, steps: List[PlanStep]) -> float:
        """Estimate the total cost of a plan."""

    def estimate_tokens(self, steps: List[PlanStep]) -> int:
        """Estimate the total number of tokens required for a plan."""

    def get_step_by_id(self, step_id: str) -> Optional[PlanStep]:
        """Get a step by its ID."""
```

### SequentialPlanner

```python
class SequentialPlanner(BasePlanner):
    async def create_plan(self, task: str, **kwargs) -> List[PlanStep]:
        """Create a plan for a task."""

    async def optimize_plan(self, steps: List[PlanStep], **kwargs) -> List[PlanStep]:
        """Optimize a plan."""

    async def execute_plan(self, steps: List[PlanStep], **kwargs) -> Tuple[bool, Any]:
        """Execute a plan."""

    async def _create_planning_prompt(self, task: str) -> str:
        """Create a prompt for planning."""

    async def _parse_planning_response(self, response: LLMResponse, task: str) -> List[PlanStep]:
        """Parse a planning response into steps."""

    def _create_optimization_prompt(self, steps: List[PlanStep]) -> str:
        """Create a prompt for optimization."""

    def _parse_optimization_response(self, response: LLMResponse, original_steps: List[PlanStep]) -> List[PlanStep]:
        """Parse an optimization response into steps."""

    def _simple_optimize(self, steps: List[PlanStep]) -> List[PlanStep]:
        """Apply simple optimization to a plan."""

    async def _execute_step(self, step: PlanStep, results: Dict[str, Any]) -> Any:
        """Execute a single step."""
```

### PlanStep

```python
class PlanStep(BaseModel):
    id: str  # Unique ID for the step
    task_description: str  # Description of the task to perform
    step_type: StepType  # Type of the step
    priority: StepPriority  # Priority of the step
    estimated_cost: float  # Estimated cost of the step in USD
    actual_cost: Optional[float]  # Actual cost of the step in USD
    estimated_tokens: int  # Estimated number of tokens required
    actual_tokens: Optional[int]  # Actual number of tokens used
    dependencies: List[str]  # IDs of steps this step depends on
    status: PlanStepStatus  # Current status of the step
    result: Optional[Any]  # Result of the step execution
    error: Optional[str]  # Error message if the step failed
    metadata: Dict[str, Any]  # Additional metadata

    def is_complete(self) -> bool:
        """Check if this step is complete."""

    def is_successful(self) -> bool:
        """Check if this step completed successfully."""

    def is_failed(self) -> bool:
        """Check if this step failed."""

    def is_skipped(self) -> bool:
        """Check if this step was skipped."""

    def is_in_progress(self) -> bool:
        """Check if this step is in progress."""

    def is_pending(self) -> bool:
        """Check if this step is pending."""

    def update_status(self, status: PlanStepStatus) -> None:
        """Update the status of this step."""

    def complete(self, result: Any, actual_cost: float, actual_tokens: int) -> None:
        """Mark this step as completed."""

    def fail(self, error: str, actual_cost: Optional[float] = None, actual_tokens: Optional[int] = None) -> None:
        """Mark this step as failed."""

    def skip(self, reason: str) -> None:
        """Mark this step as skipped."""

    def start(self) -> None:
        """Mark this step as in progress."""

    def reset(self) -> None:
        """Reset this step to pending status."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert this step to a dictionary."""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlanStep":
        """Create a step from a dictionary."""
```

### PlannerConfig

```python
class PlannerConfig(BaseModel):
    budget_strategy: BudgetStrategy  # Strategy for budget allocation
    optimization_strategy: OptimizationStrategy  # Strategy for plan optimization
    max_steps: int  # Maximum number of steps in a plan
    min_steps: int  # Minimum number of steps in a plan
    total_budget: float  # Total budget for the plan in USD
    allow_budget_overflow: bool  # Whether to allow exceeding the budget
    budget_overflow_margin: float  # Margin by which the budget can be exceeded (as a fraction)
    cost_heuristics: CostHeuristicConfig  # Cost heuristic configuration
    enable_pruning: bool  # Whether to enable pruning of unnecessary steps
    enable_parallelization: bool  # Whether to enable parallel execution of independent steps
    enable_caching: bool  # Whether to enable caching of step results
    cache_dir: Optional[str]  # Directory to cache plan results

    @classmethod
    def default(cls) -> "PlannerConfig":
        """Create a default configuration."""

    @classmethod
    def minimal(cls) -> "PlannerConfig":
        """Create a minimal configuration with only essential features enabled."""

    @classmethod
    def comprehensive(cls) -> "PlannerConfig":
        """Create a comprehensive configuration with all features enabled."""

    @classmethod
    def from_cli_args(cls, args: Dict[str, Any]) -> "PlannerConfig":
        """Create a configuration from command-line arguments."""
```

### Enums

```python
class StepType(str, Enum):
    """Type of plan step."""
    TASK = "task"  # General task
    RETRIEVAL = "retrieval"  # Information retrieval
    GENERATION = "generation"  # Content generation
    ANALYSIS = "analysis"  # Data analysis
    TOOL_USE = "tool_use"  # Tool usage
    DECISION = "decision"  # Decision making
    VERIFICATION = "verification"  # Result verification

class StepPriority(str, Enum):
    """Priority of plan step."""
    LOW = "low"  # Low priority
    MEDIUM = "medium"  # Medium priority
    HIGH = "high"  # High priority
    CRITICAL = "critical"  # Critical priority

class PlanStepStatus(str, Enum):
    """Status of plan step."""
    PENDING = "pending"  # Not yet started
    IN_PROGRESS = "in_progress"  # Currently executing
    COMPLETED = "completed"  # Successfully completed
    FAILED = "failed"  # Failed to complete
    SKIPPED = "skipped"  # Skipped (e.g., due to dependency failure)

class BudgetStrategy(str, Enum):
    """Strategy for budget allocation."""
    EQUAL = "equal"  # Equal budget for all steps
    PROPORTIONAL = "proportional"  # Budget proportional to step complexity
    DYNAMIC = "dynamic"  # Dynamically adjust budget based on previous steps
    FIXED = "fixed"  # Fixed budget per step type

class OptimizationStrategy(str, Enum):
    """Strategy for plan optimization."""
    COST = "cost"  # Optimize for minimum cost
    QUALITY = "quality"  # Optimize for maximum quality
    BALANCED = "balanced"  # Balance cost and quality
    SPEED = "speed"  # Optimize for minimum execution time
```

## Usage Examples

### Basic Usage

```python
from saplings.core.model_adapter import LLM
from saplings.planner import SequentialPlanner, PlannerConfig

# Create a model
model = LLM.create("openai", "gpt-4o")

# Create a planner
planner = SequentialPlanner(
    model=model,
    config=PlannerConfig(
        total_budget=1.0,
        max_steps=10,
        min_steps=3,
    )
)

# Create a plan
import asyncio
steps = asyncio.run(planner.create_plan("Analyze the performance of our product and suggest improvements"))

# Print the plan
for i, step in enumerate(steps):
    print(f"Step {i+1}: {step.task_description}")
    print(f"  Type: {step.step_type}")
    print(f"  Estimated cost: ${step.estimated_cost:.4f}")
    print(f"  Estimated tokens: {step.estimated_tokens}")
    print(f"  Dependencies: {step.dependencies}")
    print()

# Execute the plan
success, result = asyncio.run(planner.execute_plan(steps))

# Print the result
if success:
    print("Plan executed successfully:")
    print(result)
else:
    print("Plan execution failed:")
    print(result)
```

### Custom Budget Strategy

```python
from saplings.core.model_adapter import LLM
from saplings.planner import SequentialPlanner, PlannerConfig, BudgetStrategy

# Create a model
model = LLM.create("openai", "gpt-4o")

# Create a planner with a custom budget strategy
planner = SequentialPlanner(
    model=model,
    config=PlannerConfig(
        total_budget=1.0,
        budget_strategy=BudgetStrategy.DYNAMIC,
        allow_budget_overflow=True,
        budget_overflow_margin=0.2,  # Allow 20% overflow
    )
)

# Create and execute a plan
import asyncio
steps = asyncio.run(planner.create_plan("Research the latest advancements in quantum computing"))
success, result = asyncio.run(planner.execute_plan(steps))

# Print the result
print(result)
```

### Custom Optimization Strategy

```python
from saplings.core.model_adapter import LLM
from saplings.planner import SequentialPlanner, PlannerConfig, OptimizationStrategy

# Create a model
model = LLM.create("openai", "gpt-4o")

# Create a planner with a custom optimization strategy
planner = SequentialPlanner(
    model=model,
    config=PlannerConfig(
        total_budget=1.0,
        optimization_strategy=OptimizationStrategy.QUALITY,
    )
)

# Create a plan
import asyncio
steps = asyncio.run(planner.create_plan("Write a comprehensive report on climate change"))

# Optimize the plan
optimized_steps = asyncio.run(planner.optimize_plan(steps))

# Print the optimized plan
for i, step in enumerate(optimized_steps):
    print(f"Step {i+1}: {step.task_description}")
    print(f"  Type: {step.step_type}")
    print(f"  Estimated cost: ${step.estimated_cost:.4f}")
    print(f"  Estimated tokens: {step.estimated_tokens}")
    print()

# Execute the optimized plan
success, result = asyncio.run(planner.execute_plan(optimized_steps))

# Print the result
print(result)
```

### Manual Plan Creation

```python
from saplings.core.model_adapter import LLM
from saplings.planner import SequentialPlanner, PlanStep, StepType, StepPriority, PlanStepStatus

# Create a model
model = LLM.create("openai", "gpt-4o")

# Create a planner
planner = SequentialPlanner(model=model)

# Create steps manually
steps = [
    PlanStep(
        id="step1",
        task_description="Gather data on user engagement",
        step_type=StepType.RETRIEVAL,
        priority=StepPriority.HIGH,
        estimated_cost=0.1,
        estimated_tokens=1000,
        dependencies=[],
    ),
    PlanStep(
        id="step2",
        task_description="Analyze engagement patterns",
        step_type=StepType.ANALYSIS,
        priority=StepPriority.MEDIUM,
        estimated_cost=0.2,
        estimated_tokens=2000,
        dependencies=["step1"],
    ),
    PlanStep(
        id="step3",
        task_description="Generate recommendations",
        step_type=StepType.GENERATION,
        priority=StepPriority.MEDIUM,
        estimated_cost=0.3,
        estimated_tokens=3000,
        dependencies=["step2"],
    ),
    PlanStep(
        id="step4",
        task_description="Verify recommendations",
        step_type=StepType.VERIFICATION,
        priority=StepPriority.LOW,
        estimated_cost=0.1,
        estimated_tokens=1000,
        dependencies=["step3"],
    ),
]

# Execute the plan
import asyncio
success, result = asyncio.run(planner.execute_plan(steps))

# Print the result
print(result)
```

### Integration with Agent

```python
from saplings import Agent, AgentConfig
from saplings.planner import PlannerConfig, BudgetStrategy

# Create an agent with custom planner configuration
agent = Agent(
    config=AgentConfig(
        provider="openai",
        model_name="gpt-4o",
        planner_config=PlannerConfig(
            total_budget=2.0,
            budget_strategy=BudgetStrategy.PROPORTIONAL,
            max_steps=15,
            enable_parallelization=True,
        ),
    )
)

# Run a complex task
import asyncio
result = asyncio.run(agent.run("Analyze the latest research papers on large language models and summarize the key findings"))

# Print the result
print(result)
```

## Advanced Features

### Budget Enforcement

The planner enforces budget constraints to prevent runaway costs:

```python
from saplings.core.model_adapter import LLM
from saplings.planner import SequentialPlanner, PlannerConfig, PlanStep, StepType

# Create a model
model = LLM.create("openai", "gpt-4o")

# Create a planner with a strict budget
planner = SequentialPlanner(
    model=model,
    config=PlannerConfig(
        total_budget=0.5,
        allow_budget_overflow=False,
    )
)

# Create steps that exceed the budget
expensive_steps = [
    PlanStep(
        id="step1",
        task_description="Expensive step 1",
        step_type=StepType.GENERATION,
        estimated_cost=0.3,
        estimated_tokens=3000,
        dependencies=[],
    ),
    PlanStep(
        id="step2",
        task_description="Expensive step 2",
        step_type=StepType.GENERATION,
        estimated_cost=0.3,
        estimated_tokens=3000,
        dependencies=[],
    ),
]

# Optimize the plan to fit within budget
import asyncio
optimized_steps = asyncio.run(planner.optimize_plan(expensive_steps))

# Print the optimized plan
total_cost = sum(step.estimated_cost for step in optimized_steps)
print(f"Total cost: ${total_cost:.4f} (budget: $0.50)")
for i, step in enumerate(optimized_steps):
    print(f"Step {i+1}: {step.task_description}")
    print(f"  Estimated cost: ${step.estimated_cost:.4f}")
    print(f"  Estimated tokens: {step.estimated_tokens}")
    print()
```

### Parallel Execution

The planner can execute independent steps in parallel:

```python
from saplings.core.model_adapter import LLM
from saplings.planner import SequentialPlanner, PlannerConfig, PlanStep, StepType

# Create a model
model = LLM.create("openai", "gpt-4o")

# Create a planner with parallelization enabled
planner = SequentialPlanner(
    model=model,
    config=PlannerConfig(
        enable_parallelization=True,
    )
)

# Create steps with independent branches
steps = [
    PlanStep(
        id="step1",
        task_description="Initial data gathering",
        step_type=StepType.RETRIEVAL,
        estimated_cost=0.1,
        estimated_tokens=1000,
        dependencies=[],
    ),
    PlanStep(
        id="step2a",
        task_description="Analyze financial data",
        step_type=StepType.ANALYSIS,
        estimated_cost=0.2,
        estimated_tokens=2000,
        dependencies=["step1"],
    ),
    PlanStep(
        id="step2b",
        task_description="Analyze user data",
        step_type=StepType.ANALYSIS,
        estimated_cost=0.2,
        estimated_tokens=2000,
        dependencies=["step1"],
    ),
    PlanStep(
        id="step3",
        task_description="Generate final report",
        step_type=StepType.GENERATION,
        estimated_cost=0.3,
        estimated_tokens=3000,
        dependencies=["step2a", "step2b"],
    ),
]

# Get the execution order
execution_order = planner.get_execution_order(steps)

# Print the execution order
print("Execution order:")
for i, batch in enumerate(execution_order):
    print(f"Batch {i+1}:")
    for step in batch:
        print(f"  - {step.task_description}")
    print()

# Execute the plan
import asyncio
success, result = asyncio.run(planner.execute_plan(steps))

# Print the result
print(result)
```

### Cost Heuristics

The planner uses cost heuristics to estimate resource usage:

```python
from saplings.core.model_adapter import LLM
from saplings.planner import SequentialPlanner, PlannerConfig, CostHeuristicConfig

# Create a model
model = LLM.create("openai", "gpt-4o")

# Create a planner with custom cost heuristics
planner = SequentialPlanner(
    model=model,
    config=PlannerConfig(
        cost_heuristics=CostHeuristicConfig(
            token_cost_multiplier=1.2,  # Increase token cost estimates by 20%
            base_cost_per_step=0.02,  # Base cost for each step
            complexity_factor=2.0,  # Factor for complexity scaling
            tool_use_cost=0.1,  # Additional cost for tool use
            retrieval_cost_per_doc=0.002,  # Cost per document retrieved
            max_cost_per_step=0.5,  # Maximum cost per step
        ),
    )
)

# Create a plan
import asyncio
steps = asyncio.run(planner.create_plan("Analyze customer feedback and generate product improvement recommendations"))

# Print the cost estimates
for i, step in enumerate(steps):
    print(f"Step {i+1}: {step.task_description}")
    print(f"  Type: {step.step_type}")
    print(f"  Estimated cost: ${step.estimated_cost:.4f}")
    print(f"  Estimated tokens: {step.estimated_tokens}")
    print()
```

## Implementation Details

### Plan Creation Process

The plan creation process works as follows:

1. **Prompt Generation**: Create a prompt for the model that describes the task and planning requirements
2. **Model Invocation**: Send the prompt to the model and get a response
3. **Response Parsing**: Parse the model's response into a list of plan steps
4. **Plan Validation**: Validate the plan to ensure it meets requirements
5. **Plan Optimization**: Optimize the plan if necessary

### Plan Optimization Process

The plan optimization process works as follows:

1. **Budget Check**: Check if the plan exceeds the budget
2. **Optimization Strategy**: Apply the selected optimization strategy
3. **Step Pruning**: Remove unnecessary steps if pruning is enabled
4. **Cost Adjustment**: Adjust step costs to fit within budget
5. **Dependency Validation**: Ensure dependencies are still valid

### Plan Execution Process

The plan execution process works as follows:

1. **Execution Order**: Determine the order in which steps should be executed
2. **Step Execution**: Execute each step in order, respecting dependencies
3. **Result Tracking**: Track the results of each step
4. **Error Handling**: Handle any errors that occur during execution
5. **Result Aggregation**: Combine the results of all steps into a final result

### Budget Allocation Strategies

The budget allocation strategies work as follows:

1. **Equal**: Divide the total budget equally among all steps
2. **Proportional**: Allocate budget proportionally to step complexity
3. **Dynamic**: Adjust budget based on the results of previous steps
4. **Fixed**: Allocate a fixed budget to each step type

## Extension Points

The planning system is designed to be extensible:

### Custom Planner

You can create a custom planner by extending the `BasePlanner` class:

```python
from saplings.planner import BasePlanner, PlanStep

class CustomPlanner(BasePlanner):
    async def create_plan(self, task: str, **kwargs) -> List[PlanStep]:
        # Custom plan creation logic
        # ...
        return steps

    async def optimize_plan(self, steps: List[PlanStep], **kwargs) -> List[PlanStep]:
        # Custom plan optimization logic
        # ...
        return optimized_steps

    async def execute_plan(self, steps: List[PlanStep], **kwargs) -> Tuple[bool, Any]:
        # Custom plan execution logic
        # ...
        return success, result
```

### Custom Budget Strategy

You can create a custom budget strategy by extending the `BudgetStrategy` enum and implementing the corresponding logic:

```python
from enum import Enum
from saplings.planner import BudgetStrategy, PlanStep

# Extend the BudgetStrategy enum
class CustomBudgetStrategy(str, Enum):
    ADAPTIVE = "adaptive"  # Adapt budget based on step importance

# Implement the strategy in your planner
class CustomPlanner(BasePlanner):
    def _apply_budget_strategy(self, steps: List[PlanStep]) -> List[PlanStep]:
        if self.config.budget_strategy == "adaptive":
            # Implement adaptive budget allocation
            # ...
            return adjusted_steps
        else:
            # Fall back to default strategies
            return super()._apply_budget_strategy(steps)
```

### Custom Optimization Strategy

You can create a custom optimization strategy by extending the `OptimizationStrategy` enum and implementing the corresponding logic:

```python
from enum import Enum
from saplings.planner import OptimizationStrategy, PlanStep

# Extend the OptimizationStrategy enum
class CustomOptimizationStrategy(str, Enum):
    HYBRID = "hybrid"  # Hybrid optimization approach

# Implement the strategy in your planner
class CustomPlanner(BasePlanner):
    async def optimize_plan(self, steps: List[PlanStep], **kwargs) -> List[PlanStep]:
        if self.config.optimization_strategy == "hybrid":
            # Implement hybrid optimization
            # ...
            return optimized_steps
        else:
            # Fall back to default optimization
            return await super().optimize_plan(steps, **kwargs)
```

## Conclusion

The planning system in Saplings provides a powerful foundation for breaking down complex tasks into manageable steps while respecting budget constraints and optimizing for efficiency. By combining budget-aware planning, cost estimation, and flexible execution strategies, it enables agents to tackle a wide range of tasks effectively.
