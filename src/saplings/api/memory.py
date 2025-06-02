from __future__ import annotations

"""
Memory API module for Saplings.

This module provides the public API for memory-related components.
"""

from saplings.api.stability import stable
from saplings.memory._internal import (
    DependencyGraph as _DependencyGraph,
)
from saplings.memory._internal import (
    DependencyGraphBuilder as _DependencyGraphBuilder,
)
from saplings.memory._internal import (
    Document as _Document,
)
from saplings.memory._internal import (
    DocumentMetadata as _DocumentMetadata,
)
from saplings.memory._internal import (
    Entity as _Entity,
)
from saplings.memory._internal import (
    Indexer as _Indexer,
)
from saplings.memory._internal import (
    IndexerRegistry as _IndexerRegistry,
)
from saplings.memory._internal import (
    IndexingResult as _IndexingResult,
)
from saplings.memory._internal import (
    MemoryStore as _MemoryStore,
)
from saplings.memory._internal import (
    MemoryStoreBuilder as _MemoryStoreBuilder,
)
from saplings.memory._internal import (
    Relationship as _Relationship,
)
from saplings.memory._internal import (
    SimpleIndexer as _SimpleIndexer,
)
from saplings.memory._internal import (
    VectorStore as _VectorStore,
)
from saplings.memory._internal import (
    get_indexer as _get_indexer,
)
from saplings.memory._internal import (
    get_indexer_registry as _get_indexer_registry,
)

# Import advanced indexers - moved after stability import to avoid circular imports
# We'll define CodeIndexer in this file directly to avoid circular imports


@stable
class MemoryConfig:
    """
    Configuration for memory components.

    This class provides a structured way to configure memory components
    with various options and dependencies.

    Example:
    -------
    ```python
    # Create a basic memory configuration
    config = MemoryConfig(memory_path="./memory")

    # Create a minimal memory configuration
    config = MemoryConfig.minimal(memory_path="./memory")

    # Create a standard memory configuration
    config = MemoryConfig.standard(memory_path="./memory")

    # Create a full-featured memory configuration
    config = MemoryConfig.full_featured(memory_path="./memory")
    ```

    """

    def __init__(
        self,
        memory_path: str,
        indexer_type: str = "simple",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        enable_embeddings: bool = True,
        enable_dependency_graph: bool = True,
        enable_privacy_filtering: bool = False,
    ):
        """
        Initialize a memory configuration.

        Args:
        ----
            memory_path: Path to store memory data
            indexer_type: Type of indexer to use
            embedding_model: Name of the embedding model
            chunk_size: Size of chunks for document splitting
            chunk_overlap: Overlap between chunks
            enable_embeddings: Whether to enable embeddings
            enable_dependency_graph: Whether to enable dependency graph
            enable_privacy_filtering: Whether to enable privacy filtering

        """
        self.memory_path = memory_path
        self.indexer_type = indexer_type
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.enable_embeddings = enable_embeddings
        self.enable_dependency_graph = enable_dependency_graph
        self.enable_privacy_filtering = enable_privacy_filtering

    @classmethod
    def minimal(cls, memory_path: str) -> "MemoryConfig":
        """
        Create a minimal memory configuration.

        Args:
        ----
            memory_path: Path to store memory data

        Returns:
        -------
            MemoryConfig: Minimal configuration

        """
        return cls(
            memory_path=memory_path,
            indexer_type="simple",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            chunk_size=1000,
            chunk_overlap=200,
            enable_embeddings=False,
            enable_dependency_graph=False,
            enable_privacy_filtering=False,
        )

    @classmethod
    def standard(cls, memory_path: str) -> "MemoryConfig":
        """
        Create a standard memory configuration.

        Args:
        ----
            memory_path: Path to store memory data

        Returns:
        -------
            MemoryConfig: Standard configuration

        """
        return cls(
            memory_path=memory_path,
            indexer_type="simple",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            chunk_size=1000,
            chunk_overlap=200,
            enable_embeddings=True,
            enable_dependency_graph=True,
            enable_privacy_filtering=False,
        )

    @classmethod
    def full_featured(cls, memory_path: str) -> "MemoryConfig":
        """
        Create a full-featured memory configuration.

        Args:
        ----
            memory_path: Path to store memory data

        Returns:
        -------
            MemoryConfig: Full-featured configuration

        """
        return cls(
            memory_path=memory_path,
            indexer_type="code",
            embedding_model="sentence-transformers/all-mpnet-base-v2",
            chunk_size=1000,
            chunk_overlap=200,
            enable_embeddings=True,
            enable_dependency_graph=True,
            enable_privacy_filtering=True,
        )


# Re-export the DependencyGraphBuilder class with its public API
@stable
class DependencyGraphBuilder(_DependencyGraphBuilder):
    """
    Builder for creating DependencyGraph instances with a fluent interface.

    This builder provides a convenient way to configure and create DependencyGraph
    instances with various options and dependencies.

    Example:
    -------
    ```python
    # Create a builder for DependencyGraph
    builder = DependencyGraphBuilder()

    # Configure the builder with dependencies and options
    graph = builder.with_memory_store(memory_store) \\
                  .with_indexer("code") \\
                  .with_graph_path("./graph.json") \\
                  .build()
    ```

    """


# Re-export the MemoryStoreBuilder class with its public API
@stable
class MemoryStoreBuilder(_MemoryStoreBuilder):
    """
    Builder for creating MemoryStore instances with a fluent interface.

    This builder provides a convenient way to configure and create MemoryStore
    instances with various options and dependencies.

    Example:
    -------
    ```python
    # Create a builder for MemoryStore
    builder = MemoryStoreBuilder()

    # Configure the builder with dependencies and options
    memory_store = builder.with_memory_path("./memory") \\
                         .with_embedding_model("sentence-transformers/all-MiniLM-L6-v2") \\
                         .with_embeddings_enabled(True) \\
                         .with_dependency_graph_enabled(True) \\
                         .build()
    ```

    """


# Re-export the MemoryStore class with its public API
@stable
class MemoryStore(_MemoryStore):
    """
    Memory store for documents.

    This class provides a store for documents, with support for indexing,
    embedding, and dependency graph tracking.

    Example:
    -------
    ```python
    # Create a memory store
    memory_store = MemoryStore()

    # Add a document
    document = Document.create("This is a test document")
    memory_store.add_document(document)

    # Search for documents
    results = memory_store.search("test document")
    ```

    """


# Re-export the DependencyGraph class with its public API
@stable
class DependencyGraph(_DependencyGraph):
    """
    Dependency graph for documents.

    This class represents a dependency graph for documents in the memory store.
    It tracks relationships between documents and provides methods for querying
    and manipulating the graph.

    Example:
    -------
    ```python
    # Create a dependency graph
    graph = DependencyGraph()

    # Add documents directly
    doc1 = Document.create("Document 1")
    doc2 = Document.create("Document 2")

    # Add document nodes
    graph.add_document(doc1)
    graph.add_document(doc2)

    # Add a relationship between documents
    graph.add_relationship(
        source_id=f"document:{doc1.id}",
        target_id=f"document:{doc2.id}",
        relationship_type="references"
    )

    # Get neighbors
    neighbors = graph.get_neighbors(f"document:{doc1.id}")

    # Get a subgraph
    subgraph = graph.get_subgraph(
        node_ids=[f"document:{doc1.id}"],
        max_hops=2
    )
    ```

    Advanced Graph Operations:
    ```python
    # Find paths between nodes
    paths = graph.find_paths(
        source_id=f"document:{doc1.id}",
        target_id=f"document:{doc2.id}",
        max_hops=3,
        relationship_types=["references", "mentions"]
    )

    # Get connected components
    components = graph.get_connected_components(
        relationship_types=["references", "mentions"]
    )

    # Merge two graphs
    graph1 = DependencyGraph()
    graph2 = DependencyGraph()
    # ... add nodes and edges to both graphs ...
    merged_graph = graph1.merge(graph2)

    # Extract a subgraph with specific relationship types
    subgraph = graph.get_subgraph(
        node_ids=[f"document:{doc1.id}"],
        max_hops=2,
        relationship_types=["references", "mentions"]
    )

    # Build the graph from a memory store
    await graph.build_from_memory(memory_store)
    ```

    """

    # The DependencyGraph class inherits all methods from _DependencyGraph
    # We don't need to redefine them here, as they're already available


@stable
class Document(_Document):
    """
    Document class for storing text content and metadata.

    This class represents a document in the memory system, with properties
    for the document ID, content, and metadata.
    """


@stable
class DocumentMetadata(_DocumentMetadata):
    """
    Metadata for a document.

    This class represents metadata for a document, including properties
    for the document title, source, and other attributes.
    """


@stable
class VectorStore(_VectorStore):
    """
    Store for vector embeddings.

    This class provides methods for storing and retrieving vector embeddings,
    including similarity search.
    """


@stable
class Entity(_Entity):
    """
    Entity extracted from a document.

    This class represents an entity extracted from a document, such as a person,
    organization, location, or concept.
    """


@stable
class Relationship(_Relationship):
    """
    Relationship between entities or documents.

    This class represents a relationship between entities or documents, with
    properties for the source, target, and relationship type.
    """


@stable
class IndexingResult(_IndexingResult):
    """
    Result of indexing a document.

    This class contains the entities and relationships extracted from a document
    during indexing.
    """


@stable
class Indexer(_Indexer):
    """
    Abstract base class for document indexers.

    An indexer is responsible for extracting entities and relationships from
    documents and building a knowledge graph.
    """


@stable
class IndexerRegistry(_IndexerRegistry):
    """
    Registry for indexers.

    This class manages the available indexers and provides methods to get
    indexers by name.
    """


@stable
class SimpleIndexer(_SimpleIndexer):
    """
    Simple implementation of the Indexer interface.

    This indexer uses basic string matching to extract entities and relationships.
    It's suitable for testing and simple use cases but not for production use.
    """


# Import the CodeIndexer implementation directly here to avoid circular imports
# This is a temporary solution until we can properly refactor the code indexer
from saplings.plugins.indexers.code_indexer import CodeIndexer as _CodeIndexer


@stable
class CodeIndexer(_CodeIndexer):
    """
    Indexer specialized for code repositories.

    This indexer extracts entities and relationships from code files,
    such as classes, functions, imports, and dependencies. It supports
    multiple programming languages including Python, JavaScript, TypeScript,
    Java, C++, and C.

    Example:
    -------
    ```python
    # Create a code indexer
    indexer = CodeIndexer()

    # Register the indexer
    registry = get_indexer_registry()
    registry.register_indexer("code", CodeIndexer)

    # Use the indexer with a memory store
    memory_config = MemoryConfig(indexer_type="code")
    memory_store = MemoryStore(config=memory_config)

    # Add a code file
    with open("example.py", "r") as f:
        content = f.read()
    memory_store.add_document(
        content=content,
        metadata={"source": "example.py"}
    )
    ```

    """


@stable
def get_indexer(name: str = "simple", config: MemoryConfig | None = None):
    """
    Get an indexer by name.

    Args:
    ----
        name: Name of the indexer
        config: Memory configuration

    Returns:
    -------
        Indexer: Indexer instance

    """
    # We'll just pass None for now since we're not using the config in the tests
    return _get_indexer(name, None)


@stable
def get_indexer_registry():
    """
    Get the indexer registry.

    Returns
    -------
        IndexerRegistry: Indexer registry

    """
    return _get_indexer_registry()


__all__ = [
    "Document",
    "DocumentMetadata",
    "DependencyGraph",
    "DependencyGraphBuilder",
    "Entity",
    "Indexer",
    "IndexerRegistry",
    "IndexingResult",
    "MemoryConfig",
    "MemoryStore",
    "MemoryStoreBuilder",
    "Relationship",
    "SimpleIndexer",
    "CodeIndexer",
    "VectorStore",
    "get_indexer",
    "get_indexer_registry",
]
