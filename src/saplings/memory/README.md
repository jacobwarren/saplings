# Saplings Memory

This package provides memory management functionality for Saplings agents.

## API Structure

The memory module follows the Saplings API separation pattern:

1. **Public API**: Exposed through `saplings.api.memory` and re-exported at the top level of the `saplings` package
2. **Internal Implementation**: Located in the `_internal` directory

## Usage

To use the memory components, import them from the public API:

```python
# Recommended: Import from the top-level package
from saplings import MemoryStore, DependencyGraph, MemoryStoreBuilder

# Alternative: Import directly from the API module
from saplings.api.memory import MemoryStore, DependencyGraph, MemoryStoreBuilder
```

Do not import directly from the internal implementation:

```python
# Don't do this
from saplings.memory._internal import MemoryStore  # Wrong
```

## Available Components

The following components are available in the public API:

- `MemoryStore`: Store for documents with support for embeddings and retrieval
- `MemoryStoreBuilder`: Builder for creating MemoryStore instances
- `DependencyGraph`: Graph for tracking dependencies between documents
- `DependencyGraphBuilder`: Builder for creating DependencyGraph instances

## Memory Store

The `MemoryStore` is the main component for storing and retrieving documents:

```python
from saplings import MemoryStore, Document

# Create a memory store
memory_store = MemoryStore()

# Add a document
document = Document.create("This is a test document")
memory_store.add_document(document)

# Search for documents
results = memory_store.search("test document")
```

## Dependency Graph

The `DependencyGraph` is used to track dependencies between documents:

```python
from saplings import DependencyGraph, DocumentNode, Document

# Create a dependency graph
graph = DependencyGraph()

# Add nodes
doc1 = Document.create("Document 1")
doc2 = Document.create("Document 2")
node1 = DocumentNode.from_document(doc1)
node2 = DocumentNode.from_document(doc2)
graph.add_node(node1)
graph.add_node(node2)

# Add an edge
graph.add_edge(doc1.metadata.id, doc2.metadata.id)

# Get neighbors
neighbors = graph.get_neighbors(doc1.metadata.id)
```

## Builder Pattern

The memory module uses the builder pattern for creating instances:

```python
from saplings import MemoryStoreBuilder

# Create a memory store with the builder
memory_store = (MemoryStoreBuilder()
    .with_memory_path("./memory")
    .with_embedding_model("sentence-transformers/all-MiniLM-L6-v2")
    .with_embeddings_enabled(True)
    .with_dependency_graph_enabled(True)
    .build())
```

## Implementation Details

The memory implementations are located in the `_internal` directory:

- `_internal/memory_store.py`: Implementation of the memory store
- `_internal/dependency_graph.py`: Implementation of the dependency graph
- `_internal/document.py`: Implementation of the document class
- `_internal/indexer.py`: Implementation of the indexer interface

These internal implementations are wrapped by the public API in `saplings.api.memory` to provide stability annotations and a consistent interface.
