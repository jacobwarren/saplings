# Multi-Agent Orchestration

Saplings includes powerful multi-agent orchestration capabilities that allow you to coordinate multiple agents to solve complex tasks.

## Overview

The orchestration system consists of several components:

1. **GraphRunner**: Coordinates multiple agents in a graph structure
2. **AgentNode**: Represents an agent in the graph
3. **CommunicationChannel**: Represents a communication channel between agents
4. **Negotiation Strategies**: Debate and Contract-Net protocols for agent coordination

## GraphRunner

The `GraphRunner` class is the main coordinator for multi-agent workflows. It manages a graph of agents and their communication channels.

```python
from saplings import GraphRunner, GraphRunnerConfig, AgentNode, NegotiationStrategy
from saplings.core.model_adapter import LLM

# Create a graph runner
config = GraphRunnerConfig(
    negotiation_strategy=NegotiationStrategy.DEBATE,
    max_rounds=5,
    timeout_seconds=60,
    consensus_threshold=0.8,
)
graph_runner = GraphRunner(model=model, config=config)

# Register agents
planner = AgentNode(
    id="planner",
    name="Planner Agent",
    role="planner",
    description="An agent that creates plans",
    capabilities=["planning"],
)
executor = AgentNode(
    id="executor",
    name="Executor Agent",
    role="executor",
    description="An agent that executes plans",
    capabilities=["execution"],
)
graph_runner.register_agent(planner)
graph_runner.register_agent(executor)

# Create communication channels
graph_runner.create_channel(
    source_id="planner",
    target_id="executor",
    channel_type="plan",
    description="Channel for sending plans",
)
graph_runner.create_channel(
    source_id="executor",
    target_id="planner",
    channel_type="result",
    description="Channel for sending results",
)

# Run a negotiation
result = await graph_runner.negotiate(
    task="Solve this problem",
    context="This is a complex problem that requires planning and execution",
)
print(result)
```

## Negotiation Strategies

The `GraphRunner` supports two negotiation strategies:

### Debate

The debate strategy involves agents discussing and refining a solution until consensus is reached:

1. An initial agent proposes a solution
2. Other agents provide feedback
3. The proposal is refined based on feedback
4. The process repeats until consensus is reached or max rounds is hit

```python
config = GraphRunnerConfig(
    negotiation_strategy=NegotiationStrategy.DEBATE,
    max_rounds=5,
    consensus_threshold=0.8,
)
```

### Contract-Net

The Contract-Net Protocol (CNP) is a task allocation mechanism:

1. A manager agent breaks down a task into subtasks
2. Worker agents bid on subtasks they can perform
3. The manager allocates subtasks to the best bidders
4. Workers execute their assigned subtasks
5. The manager aggregates the results

```python
config = GraphRunnerConfig(
    negotiation_strategy=NegotiationStrategy.CONTRACT_NET,
    max_rounds=2,
)
```

## Agent Nodes

An `AgentNode` represents an agent in the graph:

```python
agent = AgentNode(
    id="critic",
    name="Critic Agent",
    role="critic",
    description="An agent that critiques solutions",
    capabilities=["critiquing", "evaluation"],
    metadata={"expertise_level": "expert"},
)
```

## Communication Channels

A `CommunicationChannel` represents a directed communication link between agents:

```python
channel = CommunicationChannel(
    source_id="planner",
    target_id="executor",
    channel_type="plan",
    description="Channel for sending plans",
    metadata={"priority": "high"},
)
```

## Integration with Other Components

The orchestration system integrates with other Saplings components:

### Integration with Executor

```python
from saplings import Executor, ExecutorConfig

# Create an executor
executor = Executor(model=model, config=ExecutorConfig())

# Register it as an agent
executor_node = AgentNode(
    id="executor",
    name="Executor Agent",
    role="executor",
    description="An agent that executes code",
    capabilities=["execution"],
)
graph_runner.register_agent(executor_node)

# Use the executor in a negotiation
# The executor will be called to execute code during the negotiation
```

### Integration with Planner

```python
from saplings import SequentialPlanner, PlannerConfig

# Create a planner
planner = SequentialPlanner(model=model, config=PlannerConfig())

# Register it as an agent
planner_node = AgentNode(
    id="planner",
    name="Planner Agent",
    role="planner",
    description="An agent that creates plans",
    capabilities=["planning"],
)
graph_runner.register_agent(planner_node)

# Use the planner in a negotiation
# The planner will be called to create plans during the negotiation
```

## Advanced Usage

### Custom Agent Implementations

You can create custom agent implementations by subclassing `AgentNode`:

```python
class CustomAgent(AgentNode):
    def __init__(self, id, name, role, description, capabilities=None, metadata=None):
        super().__init__(id, name, role, description, capabilities, metadata)
        # Custom initialization
        
    async def process_message(self, message, context=None):
        # Custom message processing
        return "Response to message"
```

### Tracking Agent Interactions

The `GraphRunner` keeps a history of agent interactions:

```python
# Get the interaction history
history = graph_runner.get_history()

# Print the history
for interaction in history:
    print(f"Agent {interaction['agent_id']} performed {interaction['action']}")
    print(f"Content: {interaction['content'][:100]}...")
    print(f"Round: {interaction['round']}")
    print()
```

### Timeout and Max Rounds

You can control the negotiation process with timeout and max rounds:

```python
# Set a timeout and max rounds in the config
config = GraphRunnerConfig(
    timeout_seconds=120,  # 2 minutes
    max_rounds=10,
)

# Or override them in the negotiate call
result = await graph_runner.negotiate(
    task="Solve this problem",
    context="This is a complex problem",
    max_rounds=3,
    timeout_seconds=30,
)
```

## Conclusion

The multi-agent orchestration capabilities in Saplings provide a powerful way to coordinate multiple agents to solve complex tasks. By defining agent relationships and communication channels, you can create sophisticated multi-agent workflows that leverage the strengths of different agent types.
