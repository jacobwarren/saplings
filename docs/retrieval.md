# Retrieval

## Overview

The Retrieval system in Saplings provides efficient and context-aware retrieval of documents from the memory store. It combines multiple retrieval strategies to provide the most relevant documents for a given query.

## Key Components

The Retrieval system consists of several key components:

1. **CascadeRetriever**: Combines multiple retrieval strategies in a cascade
2. **TFIDFRetriever**: Retrieves documents based on TF-IDF similarity
3. **EmbeddingRetriever**: Retrieves documents based on embedding similarity
4. **GraphExpander**: Expands retrieval results using the dependency graph
5. **EntropyCalculator**: Calculates entropy to determine when to stop retrieval

## Basic Usage

```python
from saplings.retrieval import CascadeRetriever, RetrievalConfig
from saplings.memory import MemoryStore
import asyncio

# Create a memory store
memory_store = MemoryStore()

# Add documents to the memory store
# ...

# Create a retrieval configuration
config = RetrievalConfig(
    entropy_threshold=0.1,
    max_documents=10,
    enable_graph_expansion=True,
)

# Create a cascade retriever
retriever = CascadeRetriever(
    memory_store=memory_store,
    config=config,
)

# Retrieve documents
async def main():
    documents = await retriever.retrieve(
        query="What is machine learning?",
        limit=5,
    )
    
    # Print retrieved documents
    for doc in documents:
        print(f"Document: {doc.id}")
        print(f"Content: {doc.content}")
        print(f"Metadata: {doc.metadata}")
        print()

# Run the async function
asyncio.run(main())
```

## Integration with Agent

The Retrieval system is integrated with the Agent class:

```python
from saplings import Agent, AgentConfig
import asyncio

# Create agent
agent = Agent(AgentConfig(model_uri="openai://gpt-4"))

# Retrieve documents
async def main():
    # Retrieve documents related to a query
    documents = await agent.retrieve("machine learning")
    
    # Print retrieved documents
    for doc in documents:
        print(f"Document: {doc.id}")
        print(f"Content: {doc.content}")
        print(f"Metadata: {doc.metadata}")
        print()

# Run the async function
asyncio.run(main())
```

## Advanced Features

### Cascade Retrieval

The CascadeRetriever combines multiple retrieval strategies in a cascade:

1. **TF-IDF Retrieval**: First, documents are retrieved based on TF-IDF similarity
2. **Embedding Retrieval**: Then, documents are retrieved based on embedding similarity
3. **Graph Expansion**: Finally, the results are expanded using the dependency graph

This cascade approach provides a balance between speed and accuracy.

### Entropy-Aware Retrieval

The EntropyCalculator calculates the entropy of the retrieval results to determine when to stop retrieval:

```python
from saplings.retrieval import EntropyCalculator
import numpy as np

# Create an entropy calculator
calculator = EntropyCalculator()

# Calculate entropy
embeddings = [np.random.rand(768) for _ in range(10)]
entropy = calculator.calculate_entropy(embeddings)

# Check if entropy is below threshold
threshold = 0.1
if entropy < threshold:
    print("Entropy is below threshold, stop retrieval")
else:
    print("Entropy is above threshold, continue retrieval")
```

### Graph Expansion

The GraphExpander expands retrieval results using the dependency graph:

```python
from saplings.retrieval import GraphExpander
from saplings.memory import DependencyGraph

# Create a dependency graph
graph = DependencyGraph()

# Add nodes and edges to the graph
# ...

# Create a graph expander
expander = GraphExpander(graph=graph)

# Expand results
expanded_docs = expander.expand(
    doc_ids=["doc1", "doc2"],
    max_hops=2,
)

# Print expanded documents
for doc_id in expanded_docs:
    print(f"Document: {doc_id}")
```

## Performance Considerations

- **Speed vs. Accuracy**: The cascade approach balances speed and accuracy
- **Entropy Threshold**: A lower threshold results in more documents being retrieved
- **Graph Expansion**: Graph expansion can significantly increase the number of retrieved documents
- **Memory Usage**: Retrieving a large number of documents can consume significant memory

## Best Practices

- **Start with a Small Number of Documents**: Begin with a small number of documents and increase as needed
- **Tune the Entropy Threshold**: Adjust the entropy threshold to balance between retrieving too few or too many documents
- **Use Graph Expansion Judiciously**: Graph expansion can be powerful but can also introduce noise
- **Monitor Retrieval Performance**: Use the monitoring tools to track retrieval performance and adjust parameters as needed
