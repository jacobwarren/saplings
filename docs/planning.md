# Planning

## Overview

The Planning system in Saplings breaks down complex tasks into manageable steps. It uses a budget-aware approach to ensure that tasks are executed efficiently and within constraints.

## Key Components

The Planning system consists of several key components:

1. **SequentialPlanner**: Creates a sequential plan for a task
2. **PlanStep**: Represents a step in a plan
3. **PlannerConfig**: Configuration for the planner
4. **CostHeuristic**: Estimates the cost of executing a plan

## Basic Usage

```python
from saplings.planner import SequentialPlanner, PlannerConfig
from saplings.core.model_adapter import LLM
import asyncio

# Create a model
model = LLM.from_uri("openai://gpt-4")

# Create a planner configuration
config = PlannerConfig(
    budget_strategy="token_count",
    max_steps=10,
    cost_heuristic="linear",
)

# Create a planner
planner = SequentialPlanner(
    model=model,
    config=config,
)

# Create a plan
async def main():
    plan = await planner.create_plan(
        task="Analyze the performance of a sorting algorithm",
    )
    
    # Print plan steps
    for i, step in enumerate(plan):
        print(f"Step {i+1}: {step.description}")
        print(f"Estimated cost: {step.estimated_cost}")
        print()

# Run the async function
asyncio.run(main())
```

## Integration with Agent

The Planning system is integrated with the Agent class:

```python
from saplings import Agent, AgentConfig
import asyncio

# Create agent
agent = Agent(AgentConfig(model_uri="openai://gpt-4"))

# Create a plan
async def main():
    # Create a plan
    plan = await agent.plan("Analyze the performance of a sorting algorithm")
    
    # Print plan steps
    for i, step in enumerate(plan):
        print(f"Step {i+1}: {step.description}")
    
    # Execute the plan
    results = await agent.execute_plan(plan)
    
    # Print results
    for result in results["results"]:
        print(f"Step: {result['step'].description}")
        print(f"Result: {result['result']}")
        print()

# Run the async function
asyncio.run(main())
```

## Advanced Features

### Budget-Aware Planning

The SequentialPlanner uses a budget-aware approach to ensure that tasks are executed efficiently and within constraints:

```python
from saplings.planner import SequentialPlanner, PlannerConfig
from saplings.core.model_adapter import LLM
import asyncio

# Create a model
model = LLM.from_uri("openai://gpt-4")

# Create a planner configuration with a budget
config = PlannerConfig(
    budget_strategy="token_count",
    max_budget=10000,  # Maximum token budget
    max_steps=10,
    cost_heuristic="linear",
)

# Create a planner
planner = SequentialPlanner(
    model=model,
    config=config,
)

# Create a plan with a budget
async def main():
    plan = await planner.create_plan(
        task="Analyze the performance of a sorting algorithm",
        budget=5000,  # Budget for this specific plan
    )
    
    # Print plan steps and budget usage
    total_cost = 0
    for i, step in enumerate(plan):
        print(f"Step {i+1}: {step.description}")
        print(f"Estimated cost: {step.estimated_cost}")
        total_cost += step.estimated_cost
        print()
    
    print(f"Total estimated cost: {total_cost}")
    print(f"Budget: {5000}")
    print(f"Remaining budget: {5000 - total_cost}")

# Run the async function
asyncio.run(main())
```

### Cost Heuristics

The Planning system supports different cost heuristics to estimate the cost of executing a plan:

```python
from saplings.planner import CostHeuristic

# Create a linear cost heuristic
linear_heuristic = CostHeuristic.create("linear")

# Estimate the cost of a step
step_description = "Analyze the time complexity of the sorting algorithm"
estimated_cost = linear_heuristic.estimate_cost(step_description)

print(f"Estimated cost: {estimated_cost}")
```

### Plan Optimization

The SequentialPlanner can optimize plans to reduce cost and improve efficiency:

```python
from saplings.planner import SequentialPlanner, PlannerConfig
from saplings.core.model_adapter import LLM
import asyncio

# Create a model
model = LLM.from_uri("openai://gpt-4")

# Create a planner configuration with optimization
config = PlannerConfig(
    budget_strategy="token_count",
    max_steps=10,
    cost_heuristic="linear",
    optimize_plan=True,
)

# Create a planner
planner = SequentialPlanner(
    model=model,
    config=config,
)

# Create an optimized plan
async def main():
    plan = await planner.create_plan(
        task="Analyze the performance of a sorting algorithm",
    )
    
    # Print optimized plan steps
    for i, step in enumerate(plan):
        print(f"Step {i+1}: {step.description}")
        print(f"Estimated cost: {step.estimated_cost}")
        print()

# Run the async function
asyncio.run(main())
```

## Performance Considerations

- **Budget Strategy**: Different budget strategies can significantly impact plan creation
- **Cost Heuristics**: The accuracy of cost heuristics affects budget allocation
- **Plan Optimization**: Plan optimization can reduce cost but may increase planning time
- **Model Quality**: The quality of the model used for planning affects the quality of the plan

## Best Practices

- **Start with a Reasonable Budget**: Begin with a reasonable budget and adjust as needed
- **Use Appropriate Cost Heuristics**: Choose cost heuristics that match your use case
- **Monitor Plan Execution**: Track the actual cost of executing plans and adjust heuristics accordingly
- **Balance Optimization and Planning Time**: Optimize plans when the potential cost savings outweigh the additional planning time
