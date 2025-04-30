# Memory System

## Overview

The Memory System in Saplings provides a comprehensive solution for storing, indexing, and retrieving documents and their relationships. It combines vector storage for semantic similarity search with graph-based memory for representing complex relationships between documents and entities.

## Key Components

The Memory System consists of several key components:

1. **MemoryStore**: The main component that combines vector storage and graph-based memory
2. **Document**: Represents a document in the memory store
3. **DependencyGraph**: Represents relationships between documents and entities
4. **VectorStore**: Stores document embeddings for efficient retrieval
5. **Indexer**: Extracts entities and relationships from documents

## Detailed Documentation

For detailed documentation on the Memory System, see:

- [Memory Store and Graph Capabilities](./memory_store.md)

## Basic Usage

```python
from saplings.memory import MemoryStore, Document, DocumentMetadata
import numpy as np

# Create a memory store
memory_store = MemoryStore()

# Add a document
document = memory_store.add_document(
    content="Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed.",
    metadata={"source": "textbook", "author": "John Doe", "date": "2023-01-01"},
)

# Add another document
document2 = memory_store.add_document(
    content="Deep learning is a subset of machine learning that uses neural networks with multiple layers.",
    metadata={"source": "article", "author": "Jane Smith", "date": "2023-02-01"},
)

# Search for documents
query_embedding = np.random.rand(768)  # Replace with actual embedding
results = memory_store.search(
    query_embedding=query_embedding,
    limit=5,
    include_graph_results=True,
)

# Print results
for doc, score in results:
    print(f"Document: {doc.id}, Score: {score}")
    print(f"Content: {doc.content}")
    print(f"Metadata: {doc.metadata}")
    print()

# Save the memory store
memory_store.save("memory_store_data")

# Load the memory store
new_memory_store = MemoryStore()
new_memory_store.load("memory_store_data")
```

## Integration with Agent

The Memory System is integrated with the Agent class:

```python
from saplings import Agent, AgentConfig
from saplings.memory import MemoryStore, Document, DocumentMetadata
import asyncio

# Create agent
agent = Agent(AgentConfig(model_uri="openai://gpt-4"))

# Add a document
document = Document(
    content="Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed.",
    metadata=DocumentMetadata(
        source="textbook",
        author="John Doe",
        date="2023-01-01"
    ),
)

# Add to memory store
agent.memory_store.add_document(document)

# Run a task that uses the document
async def main():
    result = await agent.run("Explain machine learning")
    print(result["final_result"])

# Run the async function
asyncio.run(main())
```

## Advanced Features

The Memory System supports advanced features such as:

- **Entity and Relationship Extraction**: Automatically extract entities and relationships from documents
- **Graph-Based Retrieval**: Expand search results using the graph structure
- **Security and Privacy**: Support for different privacy levels
- **Custom Vector Stores**: Support for different vector store implementations

For more information on these advanced features, see the [Memory Store and Graph Capabilities](./memory_store.md) documentation.
