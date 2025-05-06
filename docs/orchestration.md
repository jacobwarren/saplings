# Orchestration

The Orchestration system in Saplings provides powerful capabilities for coordinating multiple agents to solve complex tasks through collaboration and negotiation.

## Overview

The Orchestration system consists of several key components:

- **GraphRunner**: Coordinates multiple agents in a graph structure
- **AgentNode**: Represents an agent in the graph
- **CommunicationChannel**: Enables communication between agents
- **NegotiationStrategy**: Implements different negotiation protocols
- **OrchestrationService**: Service interface for orchestration operations

This system enables the creation of multi-agent workflows where specialized agents collaborate to solve complex tasks.

## Core Concepts

### Agent Graph

The agent graph represents the structure of a multi-agent system:

- **Nodes**: Agents with specific roles and capabilities
- **Edges**: Communication channels between agents
- **Topology**: The structure of connections between agents

The graph can be configured to implement different collaboration patterns, such as hierarchical, peer-to-peer, or hybrid structures.

### Negotiation Strategies

Negotiation strategies define how agents collaborate to reach consensus:

- **Debate**: Agents engage in a structured debate to reach consensus
- **Contract-Net**: A manager agent delegates subtasks to worker agents
- **Voting**: Agents vote on proposals to reach a decision

These strategies enable different forms of collaboration depending on the task requirements.

### Communication Channels

Communication channels enable information exchange between agents:

- **Direct**: Direct communication between two agents
- **Broadcast**: One agent communicates with multiple agents
- **Moderated**: Communication is moderated by a central agent

Channels can be configured with different properties, such as bandwidth, latency, and reliability.

## API Reference

### GraphRunner

```python
class GraphRunner:
    def __init__(
        self,
        model: LLM,
        config: Optional[GraphRunnerConfig] = None,
    ):
        """Initialize the graph runner."""

    def register_agent(self, agent: AgentNode) -> None:
        """Register an agent in the graph."""

    def register_channel(self, channel: CommunicationChannel) -> None:
        """Register a communication channel."""

    async def run_debate(
        self,
        task: str,
        agent_ids: Optional[List[str]] = None,
        context: Optional[str] = None,
        max_rounds: int = 3,
        timeout_seconds: Optional[float] = None,
        trace_id: Optional[str] = None,
    ) -> str:
        """Run a debate between agents."""

    async def run_contract_net(
        self,
        task: str,
        manager_id: Optional[str] = None,
        worker_ids: Optional[List[str]] = None,
        context: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        trace_id: Optional[str] = None,
    ) -> str:
        """Run a contract-net protocol."""

    async def validate_output(
        self,
        output: str,
        task: str,
    ) -> Optional[Dict[str, Any]]:
        """Validate an output using the judge."""

    def collect_success_pair(
        self,
        task: str,
        output: str,
        score: float,
    ) -> None:
        """Collect a success pair for self-healing."""
```

### AgentNode

```python
class AgentNode(BaseModel):
    id: str = Field(..., description="Unique identifier for the agent")
    name: str = Field(..., description="Human-readable name for the agent")
    role: str = Field(..., description="Role of the agent (e.g., 'planner', 'executor')")
    description: str = Field(..., description="Description of the agent's purpose")
    capabilities: List[str] = Field(default_factory=list, description="List of agent capabilities")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    # Fields for model specification
    provider: Optional[str] = Field(
        None, description="Model provider (e.g., 'vllm', 'openai', 'anthropic')"
    )
    model: Optional[str] = Field(None, description="Model name")
    model_parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Additional model parameters"
    )

    # Fields for component integration
    memory_store: Optional[Any] = Field(default=None, description="Memory store for this agent")
    retriever: Optional[Any] = Field(default=None, description="Retriever for this agent")
    enable_gasa: bool = Field(default=False, description="Whether to enable GASA for this agent")
    gasa_config: Optional[Dict[str, Any]] = Field(
        default=None, description="GASA configuration for this agent"
    )

    # Field for agent composition
    agent: Optional[Any] = Field(default=None, description="Base Agent instance for this node")
```

### CommunicationChannel

```python
class CommunicationChannel(BaseModel):
    source_id: str = Field(..., description="ID of the source agent")
    target_id: str = Field(..., description="ID of the target agent")
    channel_type: str = Field("direct", description="Type of channel (direct, broadcast, etc.)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
```

### GraphRunnerConfig

```python
class GraphRunnerConfig(BaseModel):
    # Negotiation settings
    negotiation_strategy: NegotiationStrategy = Field(
        NegotiationStrategy.DEBATE,
        description="Strategy for agent negotiation",
    )
    max_rounds: int = Field(
        3,
        description="Maximum number of negotiation rounds",
    )
    consensus_threshold: float = Field(
        0.7,
        description="Threshold for reaching consensus (0.0 to 1.0)",
    )

    # Memory and retrieval
    memory_store: Optional[Any] = Field(
        default=None,
        description="Shared memory store for all agents",
    )

    # Monitoring and tracing
    enable_monitoring: bool = Field(
        default=False,
        description="Whether to enable monitoring and tracing",
    )
    trace_manager: Optional[Any] = Field(
        default=None,
        description="Trace manager for monitoring agent interactions",
    )
    blame_graph: Optional[Any] = Field(
        default=None,
        description="Blame graph for identifying performance bottlenecks",
    )

    # Validation and judging
    enable_validation: bool = Field(
        default=False,
        description="Whether to enable validation of agent outputs",
    )
    judge: Optional[Any] = Field(
        default=None,
        description="Judge agent for validating outputs",
    )

    # Self-healing
    enable_self_healing: bool = Field(
        default=False,
        description="Whether to enable self-healing capabilities",
    )
    success_pair_collector: Optional[Any] = Field(
        default=None,
        description="Collector for successful error-fix pairs",
    )
```

### NegotiationStrategy

```python
class NegotiationStrategy(str, Enum):
    """Strategies for agent negotiation."""

    DEBATE = "debate"  # Agents engage in a structured debate
    CONTRACT_NET = "contract_net"  # Manager delegates subtasks to workers
    VOTING = "voting"  # Agents vote on proposals
```

### OrchestrationService

```python
class OrchestrationService:
    def __init__(
        self,
        model: LLM,
        trace_manager: Optional["TraceManager"] = None,
    ):
        """Initialize the orchestration service."""

    async def run_workflow(
        self,
        workflow_definition: Dict,
        inputs: Dict,
        trace_id: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> Dict:
        """Run a workflow defined as a graph."""

    @property
    def inner_graph_runner(self) -> GraphRunner:
        """Get the underlying graph runner."""
```

## Usage Examples

### Basic Multi-Agent Debate

```python
from saplings import Agent, AgentConfig
from saplings.memory import MemoryStore
from saplings.orchestration import GraphRunner, GraphRunnerConfig, AgentNode
from saplings.core.model_adapter import LLM

# Create a model for orchestration
model = LLM.create(provider="openai", model_name="gpt-4o")

# Create a graph runner for agent coordination
graph_runner = GraphRunner(model=model)

# Create specialized agents
code_analyzer = Agent(AgentConfig(
    provider="openai",
    model_name="gpt-4o",
    name="CodeAnalyzer",
))
code_analyzer.memory_store = MemoryStore()

refactoring_expert = Agent(AgentConfig(
    provider="openai",
    model_name="gpt-4o",
    name="RefactoringExpert",
))
refactoring_expert.memory_store = MemoryStore()

documentation_writer = Agent(AgentConfig(
    provider="openai",
    model_name="gpt-4o",
    name="DocumentationWriter",
))
documentation_writer.memory_store = MemoryStore()

# Register agents with the graph runner
graph_runner.register_agent(AgentNode(
    id="code_analyzer",
    name="Code Analyzer",
    role="analyzer",
    description="Analyzes code for issues and improvement opportunities",
    agent=code_analyzer,
))

graph_runner.register_agent(AgentNode(
    id="refactoring_expert",
    name="Refactoring Expert",
    role="refactorer",
    description="Suggests code refactoring strategies",
    agent=refactoring_expert,
))

graph_runner.register_agent(AgentNode(
    id="documentation_writer",
    name="Documentation Writer",
    role="writer",
    description="Writes and improves documentation",
    agent=documentation_writer,
))

# Run a debate between agents
import asyncio
result = asyncio.run(graph_runner.run_debate(
    task="Improve the error handling in auth.py and update the documentation",
    agent_ids=["code_analyzer", "refactoring_expert", "documentation_writer"],
    max_rounds=3,
))

print(result)
```

### Contract-Net Protocol

```python
from saplings import Agent, AgentConfig
from saplings.memory import MemoryStore
from saplings.orchestration import GraphRunner, GraphRunnerConfig, AgentNode, NegotiationStrategy
from saplings.core.model_adapter import LLM

# Create a model for orchestration
model = LLM.create(provider="openai", model_name="gpt-4o")

# Create a graph runner with contract-net strategy
config = GraphRunnerConfig(
    negotiation_strategy=NegotiationStrategy.CONTRACT_NET,
)
graph_runner = GraphRunner(model=model, config=config)

# Create a manager agent
manager = Agent(AgentConfig(
    provider="openai",
    model_name="gpt-4o",
    name="ProjectManager",
))
manager.memory_store = MemoryStore()

# Create worker agents
developer = Agent(AgentConfig(
    provider="openai",
    model_name="gpt-4o",
    name="Developer",
))
developer.memory_store = MemoryStore()

tester = Agent(AgentConfig(
    provider="openai",
    model_name="gpt-4o",
    name="Tester",
))
tester.memory_store = MemoryStore()

designer = Agent(AgentConfig(
    provider="openai",
    model_name="gpt-4o",
    name="Designer",
))
designer.memory_store = MemoryStore()

# Register agents with the graph runner
graph_runner.register_agent(AgentNode(
    id="manager",
    name="Project Manager",
    role="manager",
    description="Manages the project and delegates tasks",
    agent=manager,
))

graph_runner.register_agent(AgentNode(
    id="developer",
    name="Developer",
    role="developer",
    description="Implements features and fixes bugs",
    agent=developer,
))

graph_runner.register_agent(AgentNode(
    id="tester",
    name="Tester",
    role="tester",
    description="Tests features and reports bugs",
    agent=tester,
))

graph_runner.register_agent(AgentNode(
    id="designer",
    name="Designer",
    role="designer",
    description="Designs user interfaces and experiences",
    agent=designer,
))

# Run a contract-net protocol
import asyncio
result = asyncio.run(graph_runner.run_contract_net(
    task="Implement a new user registration feature",
    manager_id="manager",
    worker_ids=["developer", "tester", "designer"],
))

print(result)
```

### Integration with Validation

```python
from saplings import Agent, AgentConfig
from saplings.memory import MemoryStore
from saplings.orchestration import GraphRunner, GraphRunnerConfig, AgentNode
from saplings.core.model_adapter import LLM
from saplings.judge import JudgeAgent, JudgeConfig

# Create a model for orchestration
model = LLM.create(provider="openai", model_name="gpt-4o")

# Create a judge for validation
judge = JudgeAgent(
    model=model,
    config=JudgeConfig(
        threshold=0.7,
        critique_format="structured",
    ),
)

# Create a graph runner with validation
config = GraphRunnerConfig(
    enable_validation=True,
    judge=judge,
)
graph_runner = GraphRunner(model=model, config=config)

# Create specialized agents
researcher = Agent(AgentConfig(
    provider="openai",
    model_name="gpt-4o",
    name="Researcher",
))
researcher.memory_store = MemoryStore()

writer = Agent(AgentConfig(
    provider="openai",
    model_name="gpt-4o",
    name="Writer",
))
writer.memory_store = MemoryStore()

# Register agents with the graph runner
graph_runner.register_agent(AgentNode(
    id="researcher",
    name="Researcher",
    role="researcher",
    description="Researches topics and gathers information",
    agent=researcher,
))

graph_runner.register_agent(AgentNode(
    id="writer",
    name="Writer",
    role="writer",
    description="Writes content based on research",
    agent=writer,
))

# Run a debate with validation
import asyncio
result = asyncio.run(graph_runner.run_debate(
    task="Write a comprehensive article about climate change",
    agent_ids=["researcher", "writer"],
    max_rounds=3,
))

print(result)
```

### Integration with Self-Healing

```python
from saplings import Agent, AgentConfig
from saplings.memory import MemoryStore
from saplings.orchestration import GraphRunner, GraphRunnerConfig, AgentNode
from saplings.core.model_adapter import LLM
from saplings.self_heal import SuccessPairCollector

# Create a model for orchestration
model = LLM.create(provider="openai", model_name="gpt-4o")

# Create a success pair collector
collector = SuccessPairCollector(output_dir="./success_pairs")

# Create a graph runner with self-healing
config = GraphRunnerConfig(
    enable_self_healing=True,
    success_pair_collector=collector,
)
graph_runner = GraphRunner(model=model, config=config)

# Create specialized agents
coder = Agent(AgentConfig(
    provider="openai",
    model_name="gpt-4o",
    name="Coder",
))
coder.memory_store = MemoryStore()

reviewer = Agent(AgentConfig(
    provider="openai",
    model_name="gpt-4o",
    name="Reviewer",
))
reviewer.memory_store = MemoryStore()

# Register agents with the graph runner
graph_runner.register_agent(AgentNode(
    id="coder",
    name="Coder",
    role="coder",
    description="Writes code to solve problems",
    agent=coder,
))

graph_runner.register_agent(AgentNode(
    id="reviewer",
    name="Reviewer",
    role="reviewer",
    description="Reviews code for quality and correctness",
    agent=reviewer,
))

# Run a debate with self-healing
import asyncio
result = asyncio.run(graph_runner.run_debate(
    task="Write a Python function to calculate the Fibonacci sequence",
    agent_ids=["coder", "reviewer"],
    max_rounds=3,
))

print(result)

# Export success pairs for training
collector.export_to_jsonl("./success_pairs.jsonl")
```

## Advanced Features

### Custom Negotiation Strategy

```python
from saplings.orchestration import GraphRunner, GraphRunnerConfig, AgentNode, NegotiationStrategy
from saplings.core.model_adapter import LLM
from enum import Enum

# Define a custom negotiation strategy
class CustomNegotiationStrategy(str, Enum):
    AUCTION = "auction"  # Agents bid on tasks

# Extend the GraphRunner class
class CustomGraphRunner(GraphRunner):
    async def run_auction(
        self,
        task: str,
        auctioneer_id: str,
        bidder_ids: List[str],
        context: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        trace_id: Optional[str] = None,
    ) -> str:
        """Run an auction protocol."""
        # Implementation of the auction protocol
        # ...

        # For this example, we'll just use a simplified version
        auctioneer = self.agents[auctioneer_id]
        bidders = [self.agents[bidder_id] for bidder_id in bidder_ids]

        # Step 1: Auctioneer announces the task
        announcement = f"Task: {task}\n\n{context or ''}"

        # Step 2: Bidders submit bids
        bids = []
        for bidder in bidders:
            bid_prompt = f"""
            You are {bidder.name}, a {bidder.role}. {bidder.description}

            The auctioneer has announced the following task:
            {announcement}

            Submit a bid for this task, including:
            1. Your proposed approach
            2. Your estimated time to complete
            3. Your confidence level (0-100%)

            Format your response as a JSON object with "approach", "time", and "confidence" fields.
            """

            # Use agent-specific model if available, otherwise use the default model
            model = self._get_agent_model(bidder)

            # Generate the bid
            bid_response = await model.generate(prompt=bid_prompt.strip())
            bid_text = bid_response.text

            # Record the bid in the history
            self._record_interaction(
                agent_id=bidder.id,
                action="bid",
                content=bid_text,
                round=1,
                trace_id=trace_id,
            )

            bids.append({
                "bidder_id": bidder.id,
                "bid": bid_text,
            })

        # Step 3: Auctioneer selects the winning bid
        selection_prompt = f"""
        You are {auctioneer.name}, a {auctioneer.role}. {auctioneer.description}

        You have received the following bids for this task:
        {task}

        Bids:
        {json.dumps(bids, indent=2)}

        Select the winning bid based on the approach, time, and confidence.
        Explain your decision and provide feedback to all bidders.
        """

        # Use agent-specific model if available, otherwise use the default model
        model = self._get_agent_model(auctioneer)

        # Generate the selection
        selection_response = await model.generate(prompt=selection_prompt.strip())
        selection_text = selection_response.text

        # Record the selection in the history
        self._record_interaction(
            agent_id=auctioneer_id,
            action="select",
            content=selection_text,
            round=2,
            trace_id=trace_id,
        )

        return selection_text

# Create a custom graph runner
model = LLM.create(provider="openai", model_name="gpt-4o")
graph_runner = CustomGraphRunner(model=model)

# Register agents
# ...

# Run an auction
import asyncio
result = asyncio.run(graph_runner.run_auction(
    task="Design a new logo for our company",
    auctioneer_id="client",
    bidder_ids=["designer1", "designer2", "designer3"],
))

print(result)
```

### Monitoring and Tracing

```python
from saplings.orchestration import GraphRunner, GraphRunnerConfig, AgentNode
from saplings.core.model_adapter import LLM
from saplings.monitoring import TraceManager, BlameGraph, MonitoringConfig

# Create monitoring components
trace_manager = TraceManager(config=MonitoringConfig())
blame_graph = BlameGraph(trace_manager=trace_manager)

# Create a graph runner with monitoring
config = GraphRunnerConfig(
    enable_monitoring=True,
    trace_manager=trace_manager,
    blame_graph=blame_graph,
)
graph_runner = GraphRunner(model=LLM.create(provider="openai", model_name="gpt-4o"), config=config)

# Register agents
# ...

# Run a debate with monitoring
import asyncio
trace_id = trace_manager.create_trace().trace_id
result = asyncio.run(graph_runner.run_debate(
    task="Solve the traveling salesman problem for 5 cities",
    agent_ids=["mathematician", "computer_scientist", "algorithm_expert"],
    trace_id=trace_id,
))

# Process the trace
blame_graph.process_trace(trace_manager.get_trace(trace_id))

# Identify bottlenecks
bottlenecks = blame_graph.identify_bottlenecks(threshold_ms=100.0)
print("Bottlenecks:")
for bottleneck in bottlenecks:
    print(f"  {bottleneck['component']}: {bottleneck['avg_time_ms']:.2f} ms")

# Visualize the trace
from saplings.monitoring import TraceViewer
trace_viewer = TraceViewer(trace_manager=trace_manager)
trace_viewer.view_trace(
    trace_id=trace_id,
    output_path="debate_trace.html",
    show=True,
)
```

### Shared Memory

```python
from saplings.orchestration import GraphRunner, GraphRunnerConfig, AgentNode
from saplings.core.model_adapter import LLM
from saplings.memory import MemoryStore, Document

# Create a shared memory store
memory_store = MemoryStore()

# Add documents to the shared memory
memory_store.add_document(
    content="The traveling salesman problem (TSP) asks the following question: Given a list of cities and the distances between each pair of cities, what is the shortest possible route that visits each city exactly once and returns to the origin city?",
    metadata={"source": "wikipedia", "topic": "tsp"},
)

memory_store.add_document(
    content="Dynamic programming is both a mathematical optimization method and a computer programming method. In both contexts it refers to simplifying a complicated problem by breaking it down into simpler sub-problems in a recursive manner.",
    metadata={"source": "wikipedia", "topic": "dynamic_programming"},
)

memory_store.add_document(
    content="A greedy algorithm is an algorithmic paradigm that follows the problem-solving heuristic of making the locally optimal choice at each stage with the hope of finding a global optimum.",
    metadata={"source": "wikipedia", "topic": "greedy_algorithm"},
)

# Create a graph runner with shared memory
config = GraphRunnerConfig(
    memory_store=memory_store,
)
graph_runner = GraphRunner(model=LLM.create(provider="openai", model_name="gpt-4o"), config=config)

# Register agents
# ...

# Run a debate with shared memory
import asyncio
result = asyncio.run(graph_runner.run_debate(
    task="Compare dynamic programming and greedy approaches for solving the traveling salesman problem",
    agent_ids=["mathematician", "computer_scientist", "algorithm_expert"],
))

print(result)
```

## Implementation Details

### Debate Protocol

The debate protocol works as follows:

1. **Initial Proposal**: The first agent makes an initial proposal
2. **Feedback**: Other agents provide feedback on the proposal
3. **Refinement**: The original proposer refines the proposal based on feedback
4. **Consensus Check**: Agents vote on whether consensus has been reached
5. **Iteration**: If consensus is not reached, the process repeats with a new proposer

### Contract-Net Protocol

The contract-net protocol works as follows:

1. **Task Announcement**: The manager announces the task
2. **Subtask Creation**: The manager breaks down the task into subtasks
3. **Bid Submission**: Workers submit bids for subtasks
4. **Task Allocation**: The manager allocates subtasks to workers
5. **Task Execution**: Workers execute their allocated subtasks
6. **Result Integration**: The manager integrates the results

### Agent Selection

Agents are selected for tasks based on their capabilities and roles:

1. **Capability Matching**: Tasks are matched with agents based on required capabilities
2. **Role Assignment**: Agents are assigned roles based on their expertise
3. **Load Balancing**: Tasks are distributed to balance the load across agents
4. **Specialization**: Agents specialize in specific types of tasks

### Result Integration

Results from multiple agents are integrated using various strategies:

1. **Concatenation**: Results are simply concatenated
2. **Summarization**: Results are summarized to extract key points
3. **Consensus Building**: Results are combined to build consensus
4. **Voting**: Results are selected based on voting

## Extension Points

The Orchestration system is designed to be extensible:

### Custom GraphRunner

You can create a custom graph runner by extending the `GraphRunner` class:

```python
from saplings.orchestration import GraphRunner, GraphRunnerConfig
from saplings.core.model_adapter import LLM
from typing import List, Optional, Dict, Any

class CustomGraphRunner(GraphRunner):
    def __init__(
        self,
        model: LLM,
        config: Optional[GraphRunnerConfig] = None,
    ):
        """Initialize the custom graph runner."""
        super().__init__(model, config)
        self.custom_state = {}

    async def run_custom_protocol(
        self,
        task: str,
        agent_ids: List[str],
        context: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> str:
        """Run a custom protocol."""
        # Implementation of the custom protocol
        # ...

        return "Custom protocol result"

    async def _generate_proposal(self, agent, task, context):
        """Custom proposal generation."""
        # Override the proposal generation logic
        # ...

        # Call the parent method
        proposal = await super()._generate_proposal(agent, task, context)

        # Modify the proposal
        enhanced_proposal = f"Enhanced: {proposal}"

        return enhanced_proposal
```

### Custom AgentNode

You can create a custom agent node by extending the `AgentNode` class:

```python
from saplings.orchestration import AgentNode
from pydantic import Field
from typing import List, Dict, Any, Optional

class EnhancedAgentNode(AgentNode):
    expertise_level: int = Field(1, description="Expertise level (1-5)")
    specializations: List[str] = Field(default_factory=list, description="Areas of specialization")
    learning_rate: float = Field(0.1, description="Rate at which the agent learns")

    def get_expertise_score(self, task_type: str) -> float:
        """Get the expertise score for a task type."""
        base_score = self.expertise_level / 5.0

        # Bonus for specialization
        specialization_bonus = 0.0
        if task_type in self.specializations:
            specialization_bonus = 0.2

        return base_score + specialization_bonus

    def learn_from_experience(self, task_type: str, success: bool) -> None:
        """Learn from experience."""
        if success:
            # Add to specializations if not already present
            if task_type not in self.specializations:
                self.specializations.append(task_type)

            # Increase expertise level
            if self.expertise_level < 5:
                self.expertise_level += self.learning_rate
        else:
            # Decrease expertise level
            if self.expertise_level > 1:
                self.expertise_level -= self.learning_rate
```

### Custom NegotiationStrategy

You can create a custom negotiation strategy by extending the `NegotiationStrategy` enum:

```python
from enum import Enum
from saplings.orchestration import NegotiationStrategy

class CustomNegotiationStrategy(str, Enum):
    # Include the original strategies
    DEBATE = NegotiationStrategy.DEBATE
    CONTRACT_NET = NegotiationStrategy.CONTRACT_NET
    VOTING = NegotiationStrategy.VOTING

    # Add custom strategies
    AUCTION = "auction"  # Agents bid on tasks
    HIERARCHY = "hierarchy"  # Hierarchical decision-making
    CONSENSUS = "consensus"  # Consensus-based decision-making
```

## Conclusion

The Orchestration system in Saplings provides powerful capabilities for coordinating multiple agents to solve complex tasks through collaboration and negotiation. By using different negotiation strategies, communication channels, and agent configurations, it enables the creation of sophisticated multi-agent workflows tailored to specific task requirements.
