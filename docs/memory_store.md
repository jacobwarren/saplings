# Memory Store and Graph Capabilities

## Overview

The Memory Store is a comprehensive system for storing, indexing, and retrieving documents and their relationships. It combines vector storage for semantic similarity search with graph-based memory for representing complex relationships between documents and entities.

Key features of the Memory Store include:
- **Vector storage**: Efficient storage and retrieval of document embeddings
- **Graph-based memory**: Representation of relationships between documents and entities
- **Indexing**: Automatic extraction of entities and relationships from documents
- **Security**: Optional encryption and privacy controls
- **Persistence**: Ability to save and load the memory store

## Key Components

### MemoryStore

The `MemoryStore` class is the main component that combines vector storage and graph-based memory:

```python
class MemoryStore:
    def __init__(self, config: Optional[MemoryConfig] = None):
        # Initialize the memory store
        self.config = config or MemoryConfig.default()
        self.vector_store = get_vector_store(config=self.config)
        self.graph = DependencyGraph(config=self.config)
        self.indexer = get_indexer(config=self.config)
        self.secure_mode = self.config.secure_store.privacy_level != PrivacyLevel.NONE
```

### Document

The `Document` class represents a document in the memory store:

```python
class Document:
    def __init__(
        self,
        id: Optional[str] = None,
        content: str = "",
        metadata: Optional[Union[Dict[str, Any], DocumentMetadata]] = None,
        embedding: Optional[np.ndarray] = None,
    ):
        # Initialize the document
```

### DependencyGraph

The `DependencyGraph` class represents relationships between documents and entities:

```python
class DependencyGraph:
    def __init__(self, config: Optional[MemoryConfig] = None):
        # Initialize the dependency graph
        self.config = config or MemoryConfig.default()
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[Tuple[str, str, str], Edge] = {}
```

### Node Types

The graph supports different types of nodes:

```python
class Node(ABC):
    """Base class for nodes in the dependency graph."""
    
    @property
    @abstractmethod
    def id(self) -> str:
        """Get the node ID."""
        pass
    
    @property
    @abstractmethod
    def type(self) -> str:
        """Get the node type."""
        pass
    
    @property
    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """Get the node metadata."""
        pass


class DocumentNode(Node):
    """Node representing a document."""
    
    def __init__(self, document: Document):
        self.document = document
    
    @property
    def id(self) -> str:
        return self.document.id
    
    @property
    def type(self) -> str:
        return "document"
    
    @property
    def metadata(self) -> Dict[str, Any]:
        return self.document.metadata.to_dict()


class EntityNode(Node):
    """Node representing an entity."""
    
    def __init__(self, entity: Entity):
        self.entity = entity
    
    @property
    def id(self) -> str:
        return self.entity.id
    
    @property
    def type(self) -> str:
        return "entity"
    
    @property
    def metadata(self) -> Dict[str, Any]:
        return self.entity.metadata
```

### Edge

The `Edge` class represents a relationship between nodes:

```python
class Edge:
    """Edge in the dependency graph."""
    
    def __init__(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        # Initialize the edge
```

## How the Memory Store Works

### 1. Adding Documents

The `add_document` method adds a document to the memory store:

```python
def add_document(
    self,
    content: str,
    metadata: Optional[Union[Dict[str, Any], DocumentMetadata]] = None,
    document_id: Optional[str] = None,
    embedding: Optional[np.ndarray] = None,
) -> Document:
    # Add a document to the memory store
```

The process involves:
1. Creating a document with the given content, metadata, and ID
2. Applying security measures if enabled
3. Adding the document to the vector store
4. Adding the document to the graph
5. Indexing the document to extract entities and relationships

### 2. Indexing Documents

The `_index_document` method extracts entities and relationships from a document:

```python
def _index_document(self, document: Document) -> None:
    # Extract entities and relationships
    indexing_result = self.indexer.index_document(document)
    
    # Add entities to graph
    for entity in indexing_result.entities:
        entity_node = self.graph.add_entity_node(entity)
    
    # Add relationships to graph
    for relationship in indexing_result.relationships:
        try:
            self.graph.add_relationship(relationship)
        except ValueError as e:
            logger.warning(f"Failed to add relationship: {e}")
```

### 3. Searching Documents

The `search` method searches for documents similar to a query embedding:

```python
def search(
    self,
    query_embedding: np.ndarray,
    limit: int = 10,
    filter_dict: Optional[Dict[str, Any]] = None,
    include_graph_results: bool = True,
    max_graph_hops: int = 1,
) -> List[Tuple[Document, float]]:
    # Search for documents similar to the query embedding
```

The search process:
1. Searches in the vector store for documents similar to the query embedding
2. Optionally expands the results using the graph
3. Combines and ranks the results

### 4. Graph Operations

The `DependencyGraph` class provides methods for graph operations:

```python
# Add a document node
document_node = graph.add_document_node(document)

# Add an entity node
entity_node = graph.add_entity_node(entity)

# Add an edge
edge = graph.add_edge(
    source_id=source_id,
    target_id=target_id,
    relationship_type=relationship_type,
    weight=weight,
    metadata=metadata,
)

# Get a subgraph
subgraph = graph.get_subgraph(
    node_ids=node_ids,
    max_hops=max_hops,
    relationship_types=relationship_types,
)

# Find paths between nodes
paths = graph.find_paths(
    source_id=source_id,
    target_id=target_id,
    max_hops=max_hops,
    relationship_types=relationship_types,
)
```

### 5. Persistence

The `save` and `load` methods save and load the memory store:

```python
def save(self, directory: str) -> None:
    # Save the memory store to disk
    directory_path = Path(directory)
    directory_path.mkdir(parents=True, exist_ok=True)
    
    # Save vector store
    vector_store_dir = directory_path / "vector_store"
    self.vector_store.save(str(vector_store_dir))
    
    # Save graph
    graph_dir = directory_path / "graph"
    self.graph.save(str(graph_dir))
    
    # Save configuration
    with open(directory_path / "config.json", "w") as f:
        json.dump(self.config.model_dump(), f)


def load(self, directory: str) -> None:
    # Load the memory store from disk
    directory_path = Path(directory)
    
    # Load configuration
    config_path = directory_path / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            config_data = json.load(f)
            self.config = MemoryConfig(**config_data)
    
    # Load vector store
    vector_store_dir = directory_path / "vector_store"
    if vector_store_dir.exists():
        self.vector_store = get_vector_store(config=self.config)
        self.vector_store.load(str(vector_store_dir))
    
    # Load graph
    graph_dir = directory_path / "graph"
    if graph_dir.exists():
        self.graph = DependencyGraph(config=self.config)
        self.graph.load(str(graph_dir))
```

## Example Usage

### Basic Usage

```python
from saplings.memory import MemoryStore, Document

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
import numpy as np
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

### Working with the Graph

```python
from saplings.memory import MemoryStore, Document, DependencyGraph
from saplings.memory.indexer import Entity, Relationship

# Create a memory store
memory_store = MemoryStore()

# Add documents
doc1 = memory_store.add_document(
    content="Machine learning is a field of study in artificial intelligence.",
    metadata={"source": "textbook"},
)

doc2 = memory_store.add_document(
    content="Deep learning is a subset of machine learning.",
    metadata={"source": "article"},
)

# Get the graph
graph = memory_store.graph

# Add an entity manually
entity = Entity(
    id="entity:artificial_intelligence",
    name="Artificial Intelligence",
    type="concept",
    metadata={"description": "The simulation of human intelligence in machines"},
)
entity_node = graph.add_entity_node(entity)

# Add a relationship manually
relationship = Relationship(
    source_id=doc1.id,
    target_id="entity:artificial_intelligence",
    relationship_type="mentions",
    metadata={"confidence": 0.9},
)
graph.add_relationship(relationship)

# Get a subgraph
subgraph = graph.get_subgraph(
    node_ids=[doc1.id],
    max_hops=2,
)

# Print subgraph nodes
for node_id, node in subgraph.nodes.items():
    print(f"Node: {node_id}, Type: {node.type}")
    if node.type == "document":
        print(f"Content: {node.document.content}")
    elif node.type == "entity":
        print(f"Name: {node.entity.name}, Type: {node.entity.type}")
    print()

# Find paths between nodes
paths = graph.find_paths(
    source_id=doc1.id,
    target_id=doc2.id,
    max_hops=2,
)

# Print paths
for path in paths:
    print("Path:")
    for node_id in path:
        node = graph.get_node(node_id)
        print(f"  {node_id} ({node.type})")
    print()
```

### Custom Configuration

```python
from saplings.memory import MemoryStore, MemoryConfig
from saplings.memory.config import VectorStoreConfig, GraphConfig, IndexerConfig, SecureStoreConfig, PrivacyLevel

# Create a custom configuration
config = MemoryConfig(
    vector_store=VectorStoreConfig(
        type="in_memory",
        dimension=768,
        similarity_metric="cosine",
    ),
    graph=GraphConfig(
        enable_graph=True,
        max_entities_per_document=50,
        max_relationships_per_document=100,
    ),
    indexer=IndexerConfig(
        type="basic",
        extract_entities=True,
        extract_relationships=True,
        entity_types=["person", "organization", "location", "concept"],
    ),
    secure_store=SecureStoreConfig(
        privacy_level=PrivacyLevel.METADATA,
        encryption_key="your-encryption-key",
    ),
)

# Create a memory store with the custom configuration
memory_store = MemoryStore(config=config)
```

## Advanced Features

### Entity and Relationship Extraction

The Memory Store can automatically extract entities and relationships from documents:

```python
# Configure the indexer
config = MemoryConfig(
    indexer=IndexerConfig(
        type="spacy",  # Use spaCy for entity extraction
        extract_entities=True,
        extract_relationships=True,
        entity_types=["person", "organization", "location", "concept"],
        relationship_types=["mentions", "contains", "references"],
        spacy_model="en_core_web_lg",  # Use a larger spaCy model
    ),
)

# Create a memory store with the configuration
memory_store = MemoryStore(config=config)

# Add a document (entities and relationships will be extracted automatically)
document = memory_store.add_document(
    content="Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976. The company is headquartered in Cupertino, California.",
)

# Get entities and relationships
entities = memory_store.graph.get_entities()
relationships = memory_store.graph.get_relationships()

# Print entities
for entity in entities:
    print(f"Entity: {entity.name}, Type: {entity.type}")

# Print relationships
for relationship in relationships:
    source = memory_store.graph.get_node(relationship.source_id)
    target = memory_store.graph.get_node(relationship.target_id)
    print(f"Relationship: {source.name} --[{relationship.relationship_type}]--> {target.name}")
```

### Security and Privacy

The Memory Store supports different privacy levels:

```python
from saplings.memory.config import SecureStoreConfig, PrivacyLevel

# Configure security
config = MemoryConfig(
    secure_store=SecureStoreConfig(
        privacy_level=PrivacyLevel.FULL,  # Encrypt both content and metadata
        encryption_key="your-encryption-key",
        hashing_algorithm="sha256",
    ),
)

# Create a secure memory store
secure_memory_store = MemoryStore(config=config)

# Add a document (content and metadata will be encrypted)
document = secure_memory_store.add_document(
    content="Sensitive information that should be encrypted.",
    metadata={"classification": "confidential"},
)

# The document is encrypted in storage but can be used normally
results = secure_memory_store.search(query_embedding)
```

### Custom Vector Stores

The Memory Store supports different vector store implementations:

```python
from saplings.memory.config import VectorStoreConfig

# Configure a custom vector store
config = MemoryConfig(
    vector_store=VectorStoreConfig(
        type="faiss",  # Use FAISS for vector storage
        dimension=768,
        similarity_metric="cosine",
        index_type="IVF100,PQ16",  # FAISS-specific configuration
    ),
)

# Create a memory store with the custom vector store
memory_store = MemoryStore(config=config)
```

## Performance Considerations

- **Scalability**: The vector store and graph can handle large numbers of documents.
- **Memory Usage**: The in-memory vector store and graph can consume significant memory for large document sets.
- **Indexing Performance**: Entity and relationship extraction can be computationally expensive.
- **Search Performance**: Vector search is optimized for fast retrieval.

## Limitations

- **Entity Extraction Quality**: The quality of entity extraction depends on the indexer implementation.
- **Graph Complexity**: Very complex graphs can impact performance.
- **Embedding Quality**: The quality of document embeddings affects search results.

## Future Directions

- **Distributed Storage**: Support for distributed vector stores and graphs.
- **Incremental Indexing**: More efficient indexing of document updates.
- **Advanced Entity Linking**: Improved entity resolution and linking.
- **Temporal Graphs**: Support for temporal aspects of relationships.
- **Multi-modal Memory**: Support for images, audio, and other modalities.
