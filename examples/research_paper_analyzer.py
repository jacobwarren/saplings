"""
This example demonstrates using Saplings to analyze research papers,
build a knowledge graph, and answer complex questions spanning multiple papers.
"""

from __future__ import annotations

import asyncio
import os

from saplings import Agent, AgentConfig
from saplings.core.model_adapter import LLM
from saplings.memory import DependencyGraph, MemoryStore
from saplings.memory.paper_chunker import PaperChunker


async def main():
    # Create model
    model = LLM.create("openai", "gpt-4o")

    # Create memory components
    memory = MemoryStore()
    graph = DependencyGraph()
    chunker = PaperChunker()

    # Directory with PDF research papers
    # Replace with your own papers directory
    papers_dir = "./papers"

    # Create papers directory if it doesn't exist
    if not os.path.exists(papers_dir):
        os.makedirs(papers_dir)
        print(f"Created papers directory at {papers_dir}")
        print("Please add PDF research papers to this directory and run the script again.")
        return

    # Check if there are any PDFs in the directory
    pdf_files = [f for f in os.listdir(papers_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"No PDF files found in {papers_dir}")
        print("Please add PDF research papers to this directory and run the script again.")
        return

    print(f"Found {len(pdf_files)} PDF files. Processing...")

    # Process each paper
    for filename in pdf_files:
        paper_path = os.path.join(papers_dir, filename)
        print(f"Processing: {filename}")

        # Extract and chunk paper content
        chunks = chunker.chunk_pdf(paper_path)

        # Add each chunk to memory
        for i, chunk in enumerate(chunks):
            await memory.add_document(
                content=chunk.text,
                metadata={
                    "source": filename,
                    "section": chunk.section,
                    "page": chunk.page,
                    "chunk_id": f"{filename}-{i}",
                },
            )

    # Build dependency graph based on citations, topics, etc.
    print("Building dependency graph...")
    graph.build_from_memory(memory)

    # Create agent
    agent = Agent(
        config=AgentConfig(
            provider="anthropic",
            model_name="claude-3-opus-20240229",
            enable_gasa=True,
            memory_path="./research_assistant_memory",
        )
    )

    # Set memory components
    agent.memory_store = memory
    agent.dependency_graph = graph

    # Research questions
    questions = [
        "What are the key innovations in these papers?",
        "How do the methodologies in these papers compare and contrast?",
        "What open problems or future directions are identified across these papers?",
        "Synthesize the findings from all papers into a coherent research direction",
    ]

    # Run each research question
    for question in questions:
        print(f"\n--- Question: {question} ---\n")
        result = await agent.run(question)
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
