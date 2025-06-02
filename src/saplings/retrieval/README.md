# Saplings Retrieval

This package provides retrieval functionality for Saplings agents.

## API Structure

The retrieval module follows the Saplings API separation pattern:

1. **Public API**: Exposed through `saplings.api.retrieval` and re-exported at the top level of the `saplings` package
2. **Internal Implementation**: Located in the `_internal` directory

## Usage

To use the retrieval components, import them from the public API:

```python
# Recommended: Import from the top-level package
from saplings import (
    CascadeRetriever,
    EmbeddingRetriever,
    TFIDFRetriever,
    GraphExpander
)

# Alternative: Import directly from the API module
from saplings.api.retrieval import (
    CascadeRetriever,
    EmbeddingRetriever,
    TFIDFRetriever,
    GraphExpander
)
```

Do not import directly from the internal implementation:

```python
# Don't do this
from saplings.retrieval._internal import CascadeRetriever  # Wrong
```

## Available Components

The following retrieval components are available:

- `CascadeRetriever`: Orchestrates the retrieval pipeline with multiple stages
- `EmbeddingRetriever`: Retriever that uses embeddings for similarity search
- `TFIDFRetriever`: Retriever that uses TF-IDF for initial filtering
- `GraphExpander`: Expander that uses the dependency graph to find related documents
- `EntropyCalculator`: Calculator for determining when to stop retrieval
- `RetrievalConfig`: Configuration for retrieval components

## Retrieval Pipeline

The retrieval pipeline in Saplings is designed to be efficient and context-aware, with a cascaded approach that progressively refines results:

1. **TF-IDF Initial Filtering**: Quickly filter documents based on keyword matching
2. **Embedding-based Similarity Search**: Find semantically similar documents
3. **Graph Expansion**: Follow edges in the dependency graph to find related documents
4. **Entropy-based Termination**: Stop retrieval when additional results would not add significant information

## Example Usage

```python
from saplings import CascadeRetriever, MemoryStore, RetrievalConfig

# Create a memory store
memory_store = MemoryStore()

# Create a retrieval configuration
config = RetrievalConfig(
    tfidf_threshold=0.2,
    embedding_threshold=0.7,
    max_results=10,
    max_hops=2,
    entropy_threshold=0.1
)

# Create a cascade retriever
retriever = CascadeRetriever(
    memory_store=memory_store,
    config=config
)

# Retrieve documents
results = retriever.retrieve("What is the capital of France?")
```

## Implementation Details

The retrieval implementations are located in the `_internal` directory:

- `_internal/cascade_retriever.py`: Implementation of the cascade retriever
- `_internal/embedding_retriever.py`: Implementation of the embedding retriever
- `_internal/tfidf_retriever.py`: Implementation of the TF-IDF retriever
- `_internal/graph_expander.py`: Implementation of the graph expander
- `_internal/entropy_calculator.py`: Implementation of the entropy calculator
- `_internal/config.py`: Retrieval configuration

These internal implementations are wrapped by the public API in `saplings.api.retrieval` to provide stability annotations and a consistent interface.
