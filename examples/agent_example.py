"""
Example demonstrating the high-level Agent class.

This example shows how to:
1. Initialize an agent with a configuration
2. Add documents to the agent's memory
3. Retrieve relevant documents for a query
4. Create a plan for a task
5. Execute the plan
6. Run the agent on a complete task
"""

import asyncio
import os
import logging
from typing import List

from saplings import Agent, AgentConfig, Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def add_sample_documents(agent: Agent) -> List[Document]:
    """Add sample documents to the agent's memory."""
    documents = []
    
    # Add a few sample documents
    doc1 = await agent.add_document(
        content="Saplings is a graphs-first, self-improving agent framework.",
        metadata={"source": "readme", "document_id": "doc1"}
    )
    documents.append(doc1)
    
    doc2 = await agent.add_document(
        content="Graph-Aligned Sparse Attention (GASA) injects learned binary attention masks derived from retrieval dependency graphs into transformer layers.",
        metadata={"source": "technical_doc", "document_id": "doc2"}
    )
    documents.append(doc2)
    
    doc3 = await agent.add_document(
        content="The planner component creates a sequence of steps to accomplish a task, with budget enforcement and optimization strategies.",
        metadata={"source": "api_doc", "document_id": "doc3"}
    )
    documents.append(doc3)
    
    logger.info(f"Added {len(documents)} sample documents")
    return documents


async def main():
    """Run the agent example."""
    # Create output directory
    output_dir = os.path.join(os.getcwd(), "agent_output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create agent configuration
    config = AgentConfig(
        model_uri="openai:gpt-4",  # Replace with your preferred model
        memory_path=os.path.join(output_dir, "memory"),
        output_dir=output_dir,
        enable_gasa=True,
        enable_monitoring=True,
        enable_self_healing=True,
        enable_tool_factory=True,
        max_tokens=1024,
        temperature=0.7,
    )
    
    # Initialize agent
    logger.info("Initializing agent...")
    agent = Agent(config=config)
    
    # Add sample documents
    documents = await add_sample_documents(agent)
    
    # Retrieve documents for a query
    logger.info("Retrieving documents...")
    query = "How does GASA work in Saplings?"
    retrieved_docs = await agent.retrieve(query)
    
    logger.info(f"Retrieved {len(retrieved_docs)} documents")
    for i, doc in enumerate(retrieved_docs):
        logger.info(f"Document {i+1}: {doc.content[:100]}...")
    
    # Create a plan for a task
    logger.info("Creating plan...")
    task = "Explain how Saplings uses GASA for efficient attention in LLMs"
    plan = await agent.plan(task, retrieved_docs)
    
    logger.info(f"Created plan with {len(plan)} steps")
    for i, step in enumerate(plan):
        logger.info(f"Step {i+1}: {step.description}")
    
    # Execute the plan
    logger.info("Executing plan...")
    execution_results = await agent.execute_plan(plan, retrieved_docs)
    
    logger.info("Plan execution results:")
    for i, result in enumerate(execution_results["results"]):
        logger.info(f"Step {i+1} result: {result['result'][:100]}...")
    
    # Run the agent on a complete task
    logger.info("Running agent on a task...")
    task_result = await agent.run("Create a summary of how Saplings implements self-improving agents")
    
    logger.info(f"Task result: {task_result['final_result'][:200]}...")
    logger.info(f"Judgment score: {task_result['judgment']['score']}")
    logger.info(f"Output saved to: {task_result['output_path']}")
    
    # Self-improvement
    logger.info("Running self-improvement...")
    improvement_result = await agent.self_improve()
    
    logger.info(f"Improvement suggestions: {len(improvement_result['improvements'].get('improvement_suggestions', []))}")
    logger.info(f"Output saved to: {improvement_result['output_path']}")


if __name__ == "__main__":
    asyncio.run(main())
