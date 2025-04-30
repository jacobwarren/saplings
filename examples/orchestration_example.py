"""
Example of using the multi-agent orchestration capabilities in Saplings.

This example demonstrates how to use the GraphRunner, AgentNode, and
CommunicationChannel classes to coordinate multiple agents.
"""

import asyncio
import os
from saplings import (
    GraphRunner,
    GraphRunnerConfig,
    AgentNode,
    CommunicationChannel,
    NegotiationStrategy,
)
from saplings.core.model_adapter import LLM, LLMResponse, ModelMetadata, ModelRole

# Mock LLM for demonstration purposes
class MockLLM(LLM):
    """Mock LLM for demonstration purposes."""
    
    async def generate(self, prompt, **kwargs):
        """Generate a response."""
        # In a real application, this would call an actual LLM
        if "plan" in prompt.lower():
            return LLMResponse(
                text="1. Analyze the problem\n2. Break it down into steps\n3. Implement each step\n4. Test the solution",
                model_uri="mock://model",
                usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
                metadata={"model": "mock-model"},
            )
        elif "critique" in prompt.lower():
            return LLMResponse(
                text="The plan is good, but it lacks detail on testing. I suggest adding more specific testing steps.",
                model_uri="mock://model",
                usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
                metadata={"model": "mock-model"},
            )
        elif "refine" in prompt.lower() or "consensus" in prompt.lower():
            return LLMResponse(
                text="0.9",
                model_uri="mock://model",
                usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
                metadata={"model": "mock-model"},
            )
        else:
            return LLMResponse(
                text="I'll help solve this problem by following a systematic approach.",
                model_uri="mock://model",
                usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
                metadata={"model": "mock-model"},
            )
    
    def get_metadata(self):
        """Get metadata about the model."""
        return ModelMetadata(
            name="mock-model",
            provider="mock-provider",
            version="1.0",
            capabilities=[],
            roles=[ModelRole.GENERAL],
            context_window=4096,
            max_tokens_per_request=1024,
        )


async def run_debate_example():
    """Run an example of the debate negotiation strategy."""
    print("=== Debate Negotiation Example ===")
    
    # Create a mock LLM
    model = MockLLM()
    
    # Create a graph runner with debate strategy
    config = GraphRunnerConfig(
        negotiation_strategy=NegotiationStrategy.DEBATE,
        max_rounds=3,
        timeout_seconds=10,
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
    critic = AgentNode(
        id="critic",
        name="Critic Agent",
        role="critic",
        description="An agent that critiques plans",
        capabilities=["critiquing"],
    )
    implementer = AgentNode(
        id="implementer",
        name="Implementer Agent",
        role="implementer",
        description="An agent that implements plans",
        capabilities=["implementation"],
    )
    
    graph_runner.register_agent(planner)
    graph_runner.register_agent(critic)
    graph_runner.register_agent(implementer)
    
    # Create communication channels
    graph_runner.create_channel(
        source_id="planner",
        target_id="critic",
        channel_type="plan",
        description="Channel for sending plans",
    )
    graph_runner.create_channel(
        source_id="critic",
        target_id="planner",
        channel_type="critique",
        description="Channel for sending critiques",
    )
    graph_runner.create_channel(
        source_id="planner",
        target_id="implementer",
        channel_type="final_plan",
        description="Channel for sending final plans",
    )
    graph_runner.create_channel(
        source_id="implementer",
        target_id="planner",
        channel_type="implementation",
        description="Channel for sending implementations",
    )
    
    # Run the negotiation
    result = await graph_runner.negotiate(
        task="Create a plan to build a web application",
        context="The web application should have a frontend and backend",
    )
    
    print(f"Result: {result}")
    
    # Print the interaction history
    print("\nInteraction History:")
    for interaction in graph_runner.get_history():
        print(f"Round {interaction['round']}: Agent {interaction['agent_id']} performed {interaction['action']}")
        print(f"Content: {interaction['content'][:50]}...")
        print()


async def run_contract_net_example():
    """Run an example of the contract-net negotiation strategy."""
    print("\n=== Contract-Net Negotiation Example ===")
    
    # Create a mock LLM
    model = MockLLM()
    
    # Create a graph runner with contract-net strategy
    config = GraphRunnerConfig(
        negotiation_strategy=NegotiationStrategy.CONTRACT_NET,
        max_rounds=2,
        timeout_seconds=10,
    )
    graph_runner = GraphRunner(model=model, config=config)
    
    # Register agents
    manager = AgentNode(
        id="manager",
        name="Manager Agent",
        role="manager",
        description="An agent that manages tasks",
        capabilities=["managing"],
    )
    frontend_dev = AgentNode(
        id="frontend_dev",
        name="Frontend Developer Agent",
        role="frontend_developer",
        description="An agent that develops frontend code",
        capabilities=["frontend_development"],
    )
    backend_dev = AgentNode(
        id="backend_dev",
        name="Backend Developer Agent",
        role="backend_developer",
        description="An agent that develops backend code",
        capabilities=["backend_development"],
    )
    
    graph_runner.register_agent(manager)
    graph_runner.register_agent(frontend_dev)
    graph_runner.register_agent(backend_dev)
    
    # Create communication channels
    graph_runner.create_channel(
        source_id="manager",
        target_id="frontend_dev",
        channel_type="task",
        description="Channel for sending tasks",
    )
    graph_runner.create_channel(
        source_id="manager",
        target_id="backend_dev",
        channel_type="task",
        description="Channel for sending tasks",
    )
    graph_runner.create_channel(
        source_id="frontend_dev",
        target_id="manager",
        channel_type="result",
        description="Channel for sending results",
    )
    graph_runner.create_channel(
        source_id="backend_dev",
        target_id="manager",
        channel_type="result",
        description="Channel for sending results",
    )
    
    # Run the negotiation
    result = await graph_runner.negotiate(
        task="Build a web application",
        context="The web application should have a frontend and backend",
    )
    
    print(f"Result: {result}")
    
    # Print the interaction history
    print("\nInteraction History:")
    for interaction in graph_runner.get_history():
        print(f"Round {interaction['round']}: Agent {interaction['agent_id']} performed {interaction['action']}")
        print(f"Content: {interaction['content'][:50]}...")
        print()


async def main():
    """Run the examples."""
    await run_debate_example()
    await run_contract_net_example()


if __name__ == "__main__":
    asyncio.run(main())
