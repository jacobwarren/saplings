"""
Enhanced orchestration example for Saplings.

This example demonstrates the enhanced GraphRunner and AgentNode capabilities,
including memory, monitoring, validation, and self-healing.
"""

import asyncio
import logging
from typing import Dict, List, Optional

from saplings import Agent, AgentConfig
from saplings.core.model_adapter import LLM
from saplings.memory import MemoryStore, Document, DocumentMetadata
from saplings.monitoring import MonitoringConfig, TraceManager, BlameGraph
from saplings.judge import JudgeAgent, JudgeConfig
from saplings.self_heal import SuccessPairCollector
from saplings.orchestration import GraphRunner, GraphRunnerConfig, AgentNode, NegotiationStrategy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def run_enhanced_debate_example():
    """Run an example of the enhanced debate negotiation strategy."""
    print("=== Enhanced Debate Negotiation Example ===")
    
    # Create a model
    model = LLM.create(provider="openai", model="gpt-4o")
    
    # Create memory store
    memory_store = MemoryStore()
    
    # Add some documents to the memory store
    memory_store.add(Document(
        id="doc1",
        content="Saplings is a graphs-first self-improving agent framework.",
        metadata=DocumentMetadata(
            source="README.md",
            title="Saplings Overview",
        ),
    ))
    
    memory_store.add(Document(
        id="doc2",
        content="Graph-Aligned Sparse Attention (GASA) injects learned binary attention masks derived from retrieval dependency graphs into transformer layers.",
        metadata=DocumentMetadata(
            source="docs/gasa.md",
            title="GASA Overview",
        ),
    ))
    
    # Create monitoring components
    trace_manager = TraceManager()
    blame_graph = BlameGraph(trace_manager=trace_manager)
    
    # Create judge agent
    judge = JudgeAgent(model=model)
    
    # Create success pair collector
    success_pair_collector = SuccessPairCollector()
    
    # Create a graph runner with enhanced capabilities
    config = GraphRunnerConfig(
        negotiation_strategy=NegotiationStrategy.DEBATE,
        max_rounds=3,
        timeout_seconds=60,
        consensus_threshold=0.8,
        memory_store=memory_store,
        enable_monitoring=True,
        trace_manager=trace_manager,
        blame_graph=blame_graph,
        enable_validation=True,
        judge=judge,
        enable_self_healing=True,
        success_pair_collector=success_pair_collector,
    )
    graph_runner = GraphRunner(model=model, config=config)
    
    # Create a base Agent for composition
    researcher_config = AgentConfig(
        provider="openai",
        model_name="gpt-4o",
        name="ResearcherAgent",
    )
    researcher_agent = Agent(config=researcher_config)
    
    # Register agents
    graph_runner.register_agent(AgentNode(
        id="researcher",
        name="Research Agent",
        role="researcher",
        description="An agent that researches and analyzes information",
        capabilities=["research", "analysis"],
        agent=researcher_agent,  # Use the base Agent
    ))
    
    graph_runner.register_agent(AgentNode(
        id="writer",
        name="Writer Agent",
        role="writer",
        description="An agent that writes clear, concise content",
        capabilities=["writing", "editing"],
        provider="openai",
        model="gpt-4o",
        enable_gasa=True,  # Enable GASA for this agent
    ))
    
    graph_runner.register_agent(AgentNode(
        id="critic",
        name="Critic Agent",
        role="critic",
        description="An agent that critiques and improves content",
        capabilities=["critiquing", "improvement"],
        provider="openai",
        model="gpt-4o",
    ))
    
    # Run a negotiation
    task = "Explain the key benefits of Graph-Aligned Sparse Attention (GASA) in the context of the Saplings framework"
    
    # Add relevant information to memory before running the task
    graph_runner.add_to_memory(Document(
        id="doc3",
        content="GASA provides benefits including reduced token usage, improved performance, and better focus on relevant information.",
        metadata=DocumentMetadata(
            source="docs/benefits.md",
            title="GASA Benefits",
        ),
    ))
    
    # Run the negotiation
    result = await graph_runner.negotiate(task)
    print(f"\nResult:\n{result}")
    
    # Show the history
    print("\nInteraction History:")
    for interaction in graph_runner.get_history():
        print(f"Round {interaction['round']}: {interaction['agent_id']} {interaction['action']}")


async def run_enhanced_contract_net_example():
    """Run an example of the enhanced contract-net negotiation strategy."""
    print("\n=== Enhanced Contract-Net Negotiation Example ===")
    
    # Create a model
    model = LLM.create(provider="openai", model="gpt-4o")
    
    # Create memory store
    memory_store = MemoryStore()
    
    # Add some documents to the memory store
    memory_store.add(Document(
        id="doc1",
        content="Saplings is a graphs-first self-improving agent framework.",
        metadata=DocumentMetadata(
            source="README.md",
            title="Saplings Overview",
        ),
    ))
    
    # Create monitoring components
    trace_manager = TraceManager()
    
    # Create a graph runner with enhanced capabilities
    config = GraphRunnerConfig(
        negotiation_strategy=NegotiationStrategy.CONTRACT_NET,
        max_rounds=2,
        timeout_seconds=60,
        memory_store=memory_store,
        enable_monitoring=True,
        trace_manager=trace_manager,
    )
    graph_runner = GraphRunner(model=model, config=config)
    
    # Register agents
    graph_runner.register_agent(AgentNode(
        id="manager",
        name="Manager Agent",
        role="manager",
        description="An agent that manages and coordinates tasks",
        capabilities=["management", "coordination"],
        provider="openai",
        model="gpt-4o",
    ))
    
    graph_runner.register_agent(AgentNode(
        id="researcher",
        name="Research Agent",
        role="researcher",
        description="An agent that researches and analyzes information",
        capabilities=["research", "analysis"],
        provider="openai",
        model="gpt-4o",
    ))
    
    graph_runner.register_agent(AgentNode(
        id="writer",
        name="Writer Agent",
        role="writer",
        description="An agent that writes clear, concise content",
        capabilities=["writing", "editing"],
        provider="openai",
        model="gpt-4o",
    ))
    
    # Run a negotiation
    task = "Create a comprehensive guide to the Saplings framework"
    
    # Run the negotiation
    result = await graph_runner.negotiate(task)
    print(f"\nResult:\n{result}")
    
    # Show the history
    print("\nInteraction History:")
    for interaction in graph_runner.get_history():
        print(f"Round {interaction['round']}: {interaction['agent_id']} {interaction['action']}")


async def main():
    """Run the examples."""
    await run_enhanced_debate_example()
    await run_enhanced_contract_net_example()


if __name__ == "__main__":
    asyncio.run(main())
