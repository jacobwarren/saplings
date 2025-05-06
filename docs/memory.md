# Memory and Graph Storage

The memory system in Saplings combines vector storage with graph-based memory to create a rich representation of information. This document explains the memory components and how they work together.

## Overview

The memory system is designed to store, index, and retrieve documents efficiently while maintaining relationships between them. It consists of several key components:

- **MemoryStore**: The central component that manages documents and their relationships
- **Document**: The basic unit of information, containing content and metadata
- **DependencyGraph**: Represents relationships between documents, entities, and concepts
- **VectorStore**: Enables efficient similarity search using embeddings
- **Indexer**: Extracts entities and relationships from documents

## Core Concepts

### Memory Store

The `MemoryStore` is the main entry point for the memory system. It combines vector storage and graph-based memory to provide efficient and context-aware retrieval of documents. It handles:

- Adding and updating documents
- Indexing documents to extract entities and relationships
- Storing and retrieving document embeddings
- Persisting memory to disk and loading it back

### Documents

A `Document` is the basic unit of storage in the memory system. It contains:

- Content: The actual text or data
- Metadata: Information about the document (source, creation time, etc.)
- Embedding: Vector representation for similarity search
- Chunks: Optional subdivisions of the document for more granular retrieval

### Dependency Graph

The `DependencyGraph` represents relationships between documents and entities. It's a directed graph where:

- Nodes represent documents and entities
- Edges represent relationships between nodes
- Relationships have types and weights
- The graph can be traversed to find related information

### Vector Store

The `VectorStore` enables efficient similarity search using embeddings. It:

- Stores document embeddings
- Provides similarity search functionality
- Supports filtering based on metadata
- Can be implemented with different backends (in-memory, FAISS, etc.)

### Indexer

The `Indexer` extracts entities and relationships from documents. It:

- Identifies entities (people, organizations, locations, concepts)
- Discovers relationships between entities
- Creates a knowledge graph from the extracted information

## API Reference

### MemoryStore

```python
class MemoryStore:
    def __init__(self, config: Optional[MemoryConfig] = None):
        """Initialize the memory store."""

    def add_document(
        self,
        content: str,
        metadata: Optional[Union[Dict[str, Any], DocumentMetadata]] = None,
        document_id: Optional[str] = None,
        embedding: Optional[List[float]] = None,
    ) -> Document:
        """Add a document to the memory store."""

    def add_documents(self, documents: List[Document], index: bool = True) -> List[Document]:
        """Add multiple documents to the memory store."""

    def get_document(self, document_id: str) -> Optional[Document]:
        """Get a document by ID."""

    def update_document(
        self,
        document_id: str,
        content: Optional[str] = None,
        metadata: Optional[Union[Dict[str, Any], DocumentMetadata]] = None,
        embedding: Optional[List[float]] = None,
    ) -> Document:
        """Update a document in the memory store."""

    def delete_document(self, document_id: str) -> bool:
        """Delete a document from the memory store."""

    def clear(self) -> None:
        """Clear all data from the memory store."""

    def save(self, directory: str) -> None:
        """Save the memory store to disk."""

    def load(self, directory: str) -> None:
        """Load the memory store from disk."""
```

### Document

```python
class DocumentMetadata(BaseModel):
    source: str  # Source of the document (e.g., file path, URL)
    created_at: datetime  # Creation time
    updated_at: datetime  # Last update time
    content_type: str  # Content type (e.g., text/plain, text/markdown)
    language: Optional[str]  # Language of the document
    author: Optional[str]  # Author of the document
    tags: List[str]  # Tags associated with the document
    custom: Dict[str, Any]  # Custom metadata fields

    def update(self, **kwargs) -> None:
        """Update metadata fields."""

@dataclass
class Document:
    id: str  # Unique identifier
    content: str  # Document content
    metadata: DocumentMetadata  # Document metadata
    embedding: Optional[np.ndarray] = None  # Embedding vector
    chunks: List["Document"] = field(default_factory=list)  # Document chunks

    def update_embedding(self, embedding: Union[List[float], np.ndarray]) -> None:
        """Update the document embedding."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert the document to a dictionary."""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        """Create a document from a dictionary."""
```

### DependencyGraph

```python
class DependencyGraph:
    def __init__(self, config: Optional[MemoryConfig] = None):
        """Initialize the dependency graph."""

    def add_node(self, node: Node) -> None:
        """Add a node to the graph."""

    def add_document_node(self, document: Document) -> DocumentNode:
        """Add a document node to the graph."""

    def add_entity_node(self, entity: Entity) -> EntityNode:
        """Add an entity node to the graph."""

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Edge:
        """Add an edge to the graph."""

    def add_relationship(self, relationship: Relationship) -> Edge:
        """Add a relationship to the graph."""

    def get_node(self, node_id: str) -> Optional[Node]:
        """Get a node by ID."""

    def get_edge(
        self, source_id: str, target_id: str, relationship_type: str
    ) -> Optional[Edge]:
        """Get an edge by source, target, and relationship type."""

    def get_neighbors(
        self,
        node_id: str,
        relationship_types: Optional[List[str]] = None,
        direction: str = "both",
    ) -> List[Node]:
        """Get neighbors of a node."""

    def get_subgraph(
        self,
        node_ids: List[str],
        max_hops: int = 1,
        relationship_types: Optional[List[str]] = None,
    ) -> "DependencyGraph":
        """Get a subgraph centered on the specified nodes."""

    def save(self, directory: str) -> None:
        """Save the graph to disk."""

    def load(self, directory: str) -> None:
        """Load the graph from disk."""
```

### VectorStore

```python
class VectorStore(ABC):
    @abstractmethod
    def add_document(self, document: Document) -> None:
        """Add a document to the vector store."""

    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        """Add multiple documents to the vector store."""

    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        limit: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """Search for similar documents."""

    @abstractmethod
    def update(self, document: Document) -> None:
        """Update a document in the vector store."""

    @abstractmethod
    def delete(self, document_id: str) -> bool:
        """Delete a document from the vector store."""

    @abstractmethod
    def clear(self) -> None:
        """Clear all data from the vector store."""

    @abstractmethod
    def save(self, directory: str) -> None:
        """Save the vector store to disk."""

    @abstractmethod
    def load(self, directory: str) -> None:
        """Load the vector store from disk."""
```

### Indexer

```python
class Indexer(ABC):
    def __init__(self, config: Optional[MemoryConfig] = None):
        """Initialize the indexer."""

    @abstractmethod
    def extract_entities(self, document: Document) -> List[Entity]:
        """Extract entities from a document."""

    @abstractmethod
    def extract_relationships(
        self, document: Document, entities: List[Entity]
    ) -> List[Relationship]:
        """Extract relationships between entities in a document."""

    def index_document(self, document: Document) -> IndexingResult:
        """Index a document."""
```

## Configuration

The memory system can be configured using the `MemoryConfig` class:

```python
class MemoryConfig(BaseModel):
    vector_store: VectorStoreConfig  # Vector store configuration
    graph: GraphConfig  # Graph configuration
    secure_store: SecureStoreConfig  # SecureStore configuration
    chunk_size: int = 1000  # Default chunk size for documents in characters
    chunk_overlap: int = 200  # Default chunk overlap for documents in characters

    @classmethod
    def default(cls) -> "MemoryConfig":
        """Create a default configuration."""

    @classmethod
    def with_faiss(cls, use_gpu: bool = False) -> "MemoryConfig":
        """Create a configuration with FAISS vector store."""

    @classmethod
    def minimal(cls) -> "MemoryConfig":
        """Create a minimal configuration with only essential features enabled."""

    @classmethod
    def secure(cls) -> "MemoryConfig":
        """Create a configuration with security features enabled."""
```

## Usage Examples

### Basic Usage

```python
from saplings.memory import MemoryStore, Document, DocumentMetadata

# Create a memory store
memory = MemoryStore()

# Add a document
document = memory.add_document(
    content="Saplings is a graph-first, self-improving agent framework.",
    metadata={"source": "README.md", "author": "Saplings Team"}
)

# Get a document
retrieved_document = memory.get_document(document.id)

# Update a document
updated_document = memory.update_document(
    document_id=document.id,
    content="Saplings is a graph-first, self-improving agent framework that takes root in your repository.",
    metadata={"tags": ["framework", "agent", "graph"]}
)

# Save the memory store
memory.save("./memory_store")

# Load the memory store
new_memory = MemoryStore()
new_memory.load("./memory_store")
```

### Working with the Dependency Graph

```python
from saplings.memory import MemoryStore, DependencyGraph
from saplings.memory.indexer import Entity, Relationship

# Create a memory store
memory = MemoryStore()

# Add documents
doc1 = memory.add_document(
    content="John works for Acme Corporation in New York.",
    metadata={"source": "employee_records.txt"}
)

doc2 = memory.add_document(
    content="Acme Corporation is headquartered in New York and was founded in 1990.",
    metadata={"source": "company_info.txt"}
)

# Get the dependency graph
graph = memory.graph

# Get neighbors of a document
neighbors = graph.get_neighbors(doc1.id)

# Get a subgraph
subgraph = graph.get_subgraph(
    node_ids=[doc1.id],
    max_hops=2,
    relationship_types=["mentions", "located_in"]
)
```

### Custom Indexer

```python
from saplings.memory import MemoryStore, Document
from saplings.memory.indexer import Indexer, Entity, Relationship, IndexingResult

class CustomIndexer(Indexer):
    def extract_entities(self, document: Document) -> List[Entity]:
        # Custom entity extraction logic
        entities = []

        # Example: Extract company names
        if "company" in document.content.lower():
            entity = Entity(
                name="Company",
                entity_type="organization",
                metadata={"source_document": document.id}
            )
            entities.append(entity)

        return entities

    def extract_relationships(
        self, document: Document, entities: List[Entity]
    ) -> List[Relationship]:
        # Custom relationship extraction logic
        relationships = []

        # Example: Create relationships between entities
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                relationship = Relationship(
                    source_id=f"entity:{entity1.entity_type}:{entity1.name}",
                    target_id=f"entity:{entity2.entity_type}:{entity2.name}",
                    relationship_type="related_to",
                    metadata={"confidence": 0.8}
                )
                relationships.append(relationship)

        return relationships

# Create a memory store with the custom indexer
memory = MemoryStore()
memory.indexer = CustomIndexer()

# Add a document
document = memory.add_document(
    content="The company is expanding its operations.",
    metadata={"source": "news.txt"}
)
```

## Advanced Features

### Secure Memory Store

Saplings provides security features for the memory store:

```python
from saplings.memory import MemoryStore
from saplings.memory.config import MemoryConfig, PrivacyLevel

# Create a secure memory store
config = MemoryConfig.secure()
memory = MemoryStore(config=config)

# Add a document
document = memory.add_document(
    content="This is sensitive information.",
    metadata={"source": "confidential.txt"}
)
```

The secure memory store provides:

- Hash-based protection for document IDs and metadata
- Differential privacy noise for embeddings
- Secure storage and retrieval

### FAISS Vector Store

For larger document collections, you can use FAISS for efficient similarity search:

```python
from saplings.memory import MemoryStore
from saplings.memory.config import MemoryConfig

# Create a memory store with FAISS
config = MemoryConfig.with_faiss(use_gpu=False)
memory = MemoryStore(config=config)

# Add documents
for i in range(1000):
    memory.add_document(
        content=f"Document {i} with some content for testing.",
        metadata={"source": f"test_{i}.txt"}
    )
```

FAISS provides:

- Efficient similarity search for large collections
- GPU acceleration (optional)
- Approximate nearest neighbor search

## Implementation Details

### Document Chunking

Documents can be automatically chunked for more granular retrieval:

```python
from saplings.memory import MemoryStore
from saplings.memory.config import MemoryConfig

# Configure chunking
config = MemoryConfig(
    chunk_size=500,  # Characters per chunk
    chunk_overlap=100  # Overlap between chunks
)
memory = MemoryStore(config=config)

# Add a document (will be automatically chunked)
document = memory.add_document(
    content="This is a long document that will be split into chunks. " * 20,
    metadata={"source": "long_document.txt"}
)

# Access chunks
for chunk in document.chunks:
    print(f"Chunk ID: {chunk.id}, Content: {chunk.content[:50]}...")
```

### Entity and Relationship Extraction

The indexer extracts entities and relationships from documents:

```python
from saplings.memory import MemoryStore, Document
from saplings.memory.indexer import SimpleIndexer

# Create a memory store with a simple indexer
memory = MemoryStore()
memory.indexer = SimpleIndexer()

# Add a document
document = memory.add_document(
    content="John works for Acme Corporation in New York.",
    metadata={"source": "employee_records.txt"}
)

# Get entities
entities = memory.indexer.extract_entities(document)
for entity in entities:
    print(f"Entity: {entity.name}, Type: {entity.entity_type}")

# Get relationships
relationships = memory.indexer.extract_relationships(document, entities)
for relationship in relationships:
    print(f"Relationship: {relationship.source_id} {relationship.relationship_type} {relationship.target_id}")
```

## Extension Points

The memory system is designed to be extensible:

### Custom Vector Store

You can create a custom vector store by implementing the `VectorStore` interface:

```python
from saplings.memory import VectorStore, Document
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

class CustomVectorStore(VectorStore):
    def __init__(self, config=None):
        self.documents = {}
        self.embeddings = {}

    def add_document(self, document: Document) -> None:
        if document.embedding is None:
            raise ValueError(f"Document {document.id} has no embedding")
        self.documents[document.id] = document
        self.embeddings[document.id] = document.embedding

    def add_documents(self, documents: List[Document]) -> None:
        for document in documents:
            self.add_document(document)

    def search(
        self,
        query_embedding: np.ndarray,
        limit: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        # Custom search implementation
        results = []
        for doc_id, embedding in self.embeddings.items():
            similarity = np.dot(query_embedding, embedding)
            results.append((self.documents[doc_id], similarity))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    # Implement other required methods...
```

### Custom Indexer

You can create a custom indexer by implementing the `Indexer` interface:

```python
from saplings.memory import Indexer, Document
from saplings.memory.indexer import Entity, Relationship
from typing import List

class CustomIndexer(Indexer):
    def extract_entities(self, document: Document) -> List[Entity]:
        # Custom entity extraction logic
        entities = []
        # ...
        return entities

    def extract_relationships(
        self, document: Document, entities: List[Entity]
    ) -> List[Relationship]:
        # Custom relationship extraction logic
        relationships = []
        # ...
        return relationships
```

## Conclusion

The memory system in Saplings provides a powerful foundation for storing, indexing, and retrieving documents while maintaining relationships between them. By combining vector storage with graph-based memory, it enables more contextual and accurate retrieval than traditional vector stores alone.
