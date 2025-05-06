# Retrieval System

The retrieval system in Saplings is designed to efficiently find relevant information from the memory store. It uses a cascaded approach that progressively refines results through multiple stages.

## Overview

The retrieval system consists of several key components:

- **CascadeRetriever**: Orchestrates the entire retrieval pipeline
- **TFIDFRetriever**: Performs initial filtering using TF-IDF
- **EmbeddingRetriever**: Finds similar documents using embeddings
- **GraphExpander**: Expands results using the dependency graph
- **EntropyCalculator**: Determines when to stop retrieval based on information gain

This cascaded approach combines the strengths of different retrieval methods to provide more accurate and contextually relevant results than any single method alone.

## Core Concepts

### Cascade Retrieval

The cascade retrieval process follows these steps:

1. **TF-IDF Filtering**: Initial lexical filtering to quickly identify potentially relevant documents
2. **Embedding-based Retrieval**: Semantic search to find documents with similar meaning
3. **Graph Expansion**: Exploration of related documents through the dependency graph
4. **Entropy-based Termination**: Determination of when sufficient information has been retrieved

This process may iterate multiple times, expanding the search with each iteration until the entropy calculator determines that sufficient information has been retrieved.

### TF-IDF Retrieval

TF-IDF (Term Frequency-Inverse Document Frequency) is a statistical measure used to evaluate the importance of a word in a document relative to a collection of documents. The TFIDFRetriever:

- Builds an index of document terms
- Converts queries to the same TF-IDF space
- Calculates similarity scores between the query and documents
- Returns the top-k most similar documents

### Embedding-based Retrieval

Embedding-based retrieval uses vector representations of text to find semantically similar documents. The EmbeddingRetriever:

- Embeds the query using a pre-trained model
- Calculates similarity scores between the query embedding and document embeddings
- Returns the top-k most similar documents

### Graph Expansion

Graph expansion uses the dependency graph to find related documents. The GraphExpander:

- Takes the documents from the embedding retriever
- Finds related documents through graph traversal
- Calculates scores based on graph distance and edge weights
- Returns an expanded set of documents

### Entropy-based Termination

Entropy-based termination determines when sufficient information has been retrieved. The EntropyCalculator:

- Calculates the information entropy of the current set of documents
- Tracks changes in entropy across iterations
- Terminates retrieval when entropy stabilizes or other criteria are met

## API Reference

### CascadeRetriever

```python
class CascadeRetriever:
    def __init__(
        self,
        memory_store: MemoryStore,
        config: Optional[RetrievalConfig] = None,
    ):
        """Initialize the cascade retriever."""

    def retrieve(
        self,
        query: str,
        filter_dict: Optional[Dict[str, Any]] = None,
        max_documents: Optional[int] = None,
    ) -> RetrievalResult:
        """Retrieve documents relevant to the query."""

    def save(self, directory: str) -> None:
        """Save the retriever to disk."""

    def load(self, directory: str) -> None:
        """Load the retriever from disk."""
```

### TFIDFRetriever

```python
class TFIDFRetriever:
    def __init__(
        self,
        memory_store: MemoryStore,
        config: Optional[Union[RetrievalConfig, TFIDFConfig]] = None,
    ):
        """Initialize the TF-IDF retriever."""

    def build_index(self) -> None:
        """Build the TF-IDF index."""

    def retrieve(
        self, query: str, k: Optional[int] = None, filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """Retrieve documents similar to the query using TF-IDF."""

    def save(self, directory: str) -> None:
        """Save the TF-IDF retriever to disk."""

    def load(self, directory: str) -> None:
        """Load the TF-IDF retriever from disk."""
```

### EmbeddingRetriever

```python
class EmbeddingRetriever:
    def __init__(
        self,
        memory_store: MemoryStore,
        config: Optional[Union[RetrievalConfig, EmbeddingConfig]] = None,
    ):
        """Initialize the embedding retriever."""

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query string."""

    def retrieve(
        self,
        query: str,
        documents: List[Document],
        k: Optional[int] = None,
    ) -> List[Tuple[Document, float]]:
        """Retrieve documents similar to the query using embeddings."""

    def save(self, directory: str) -> None:
        """Save the embedding retriever to disk."""

    def load(self, directory: str) -> None:
        """Load the embedding retriever from disk."""
```

### GraphExpander

```python
class GraphExpander:
    def __init__(
        self,
        memory_store: MemoryStore,
        config: Optional[Union[RetrievalConfig, GraphConfig]] = None,
    ):
        """Initialize the graph expander."""

    def expand(
        self,
        documents: List[Document],
        scores: Optional[List[float]] = None,
    ) -> List[Tuple[Document, float]]:
        """Expand retrieval results using the dependency graph."""

    def save(self, directory: str) -> None:
        """Save the graph expander to disk."""

    def load(self, directory: str) -> None:
        """Load the graph expander from disk."""
```

### EntropyCalculator

```python
class EntropyCalculator:
    def __init__(
        self,
        config: Optional[Union[RetrievalConfig, EntropyConfig]] = None,
    ):
        """Initialize the entropy calculator."""

    def calculate_entropy(self, documents: List[Document]) -> float:
        """Calculate the information entropy of a set of documents."""

    def calculate_entropy_change(self, documents: List[Document]) -> float:
        """Calculate the change in entropy compared to previous iterations."""

    def should_terminate(self, documents: List[Document], iteration: int) -> bool:
        """Determine if the retrieval process should terminate."""

    def reset(self) -> None:
        """Reset the entropy calculator."""

    def save(self, directory: str) -> None:
        """Save the entropy calculator to disk."""

    def load(self, directory: str) -> None:
        """Load the entropy calculator from disk."""
```

### RetrievalResult

```python
class RetrievalResult:
    def __init__(
        self,
        documents: List[Document],
        scores: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize a retrieval result."""

    def __len__(self) -> int:
        """Get the number of documents in the result."""

    def __getitem__(self, index: int) -> Tuple[Document, float]:
        """Get a document and its score by index."""

    def get_documents(self) -> List[Document]:
        """Get all documents in the result."""

    def get_scores(self) -> List[float]:
        """Get all scores in the result."""

    def get_metadata(self) -> Dict[str, Any]:
        """Get the metadata for the result."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary."""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetrievalResult":
        """Create a retrieval result from a dictionary."""
```

## Configuration

The retrieval system can be configured using the `RetrievalConfig` class:

```python
class RetrievalConfig(BaseModel):
    tfidf: TFIDFConfig  # TF-IDF configuration
    embedding: EmbeddingConfig  # Embedding configuration
    graph: GraphConfig  # Graph configuration
    entropy: EntropyConfig  # Entropy configuration
    cache: CacheConfig  # Cache configuration

    @classmethod
    def default(cls) -> "RetrievalConfig":
        """Create a default configuration."""

    @classmethod
    def minimal(cls) -> "RetrievalConfig":
        """Create a minimal configuration with only essential features enabled."""

    @classmethod
    def comprehensive(cls) -> "RetrievalConfig":
        """Create a comprehensive configuration with all features enabled."""

    @classmethod
    def no_cache(cls) -> "RetrievalConfig":
        """Create a configuration with caching disabled."""
```

### TFIDFConfig

```python
class TFIDFConfig(BaseModel):
    initial_k: int = 100  # Initial number of documents to retrieve
    max_features: int = 10000  # Maximum number of features for the vectorizer
    ngram_range: Tuple[int, int] = (1, 2)  # Range of n-grams to consider
    min_df: int = 2  # Minimum document frequency for a term to be included
    use_idf: bool = True  # Whether to use inverse document frequency
    sublinear_tf: bool = True  # Whether to use sublinear term frequency scaling
```

### EmbeddingConfig

```python
class EmbeddingConfig(BaseModel):
    model_name: str = "all-MiniLM-L6-v2"  # Name of the embedding model to use
    embedding_dimension: int = 384  # Dimension of the embedding vectors
    batch_size: int = 32  # Batch size for embedding generation
    similarity_top_k: int = 20  # Number of documents to retrieve using embedding similarity
    similarity_cutoff: Optional[float] = 0.7  # Minimum similarity score for a document to be included
    use_existing_embeddings: bool = True  # Whether to use existing embeddings if available
```

### GraphConfig

```python
class GraphConfig(BaseModel):
    max_hops: int = 2  # Maximum number of hops to traverse in the graph
    max_nodes: int = 50  # Maximum number of nodes to include in the expanded results
    min_edge_weight: float = 0.5  # Minimum edge weight for traversal
    relationship_types: Optional[List[str]] = None  # Types of relationships to traverse
    include_entity_nodes: bool = True  # Whether to include entity nodes in the traversal
    score_decay_factor: float = 0.8  # Factor by which to decay scores with each hop
```

### EntropyConfig

```python
class EntropyConfig(BaseModel):
    threshold: float = 0.1  # Entropy threshold for termination
    max_iterations: int = 3  # Maximum number of iterations
    min_documents: int = 5  # Minimum number of documents to retrieve
    max_documents: int = 50  # Maximum number of documents to retrieve
    use_normalized_entropy: bool = True  # Whether to use normalized entropy
    window_size: int = 3  # Window size for entropy calculation
```

## Usage Examples

### Basic Usage

```python
from saplings.memory import MemoryStore
from saplings.retrieval import CascadeRetriever, RetrievalConfig

# Create a memory store
memory = MemoryStore()

# Add documents to the memory store
for i in range(100):
    memory.add_document(
        content=f"Document {i} about machine learning and artificial intelligence.",
        metadata={"source": f"doc_{i}.txt"}
    )

# Create a cascade retriever
retriever = CascadeRetriever(memory_store=memory)

# Retrieve documents
result = retriever.retrieve(query="What is machine learning?")

# Print the results
for doc, score in result:
    print(f"Document: {doc.id}, Score: {score:.4f}")
    print(f"Content: {doc.content[:100]}...")
    print()
```

### Custom Configuration

```python
from saplings.memory import MemoryStore
from saplings.retrieval import CascadeRetriever, RetrievalConfig
from saplings.retrieval.config import TFIDFConfig, EmbeddingConfig, GraphConfig, EntropyConfig

# Create a custom configuration
config = RetrievalConfig(
    tfidf=TFIDFConfig(
        initial_k=200,
        max_features=20000,
        ngram_range=(1, 3),
    ),
    embedding=EmbeddingConfig(
        model_name="all-MiniLM-L6-v2",
        similarity_top_k=50,
        similarity_cutoff=0.6,
    ),
    graph=GraphConfig(
        max_hops=3,
        max_nodes=100,
        include_entity_nodes=True,
    ),
    entropy=EntropyConfig(
        threshold=0.05,
        max_iterations=5,
        max_documents=100,
    ),
)

# Create a cascade retriever with the custom configuration
retriever = CascadeRetriever(memory_store=memory, config=config)

# Retrieve documents
result = retriever.retrieve(query="What is machine learning?")
```

### Filtering Results

```python
from saplings.memory import MemoryStore
from saplings.retrieval import CascadeRetriever

# Create a memory store
memory = MemoryStore()

# Add documents to the memory store
memory.add_document(
    content="Python is a programming language.",
    metadata={"category": "programming", "language": "python"}
)
memory.add_document(
    content="Java is a programming language.",
    metadata={"category": "programming", "language": "java"}
)
memory.add_document(
    content="Machine learning is a field of artificial intelligence.",
    metadata={"category": "ai", "topic": "machine learning"}
)

# Create a cascade retriever
retriever = CascadeRetriever(memory_store=memory)

# Retrieve documents with a filter
result = retriever.retrieve(
    query="programming language",
    filter_dict={"category": "programming", "language": "python"}
)

# Print the results
for doc, score in result:
    print(f"Document: {doc.id}, Score: {score:.4f}")
    print(f"Content: {doc.content}")
    print(f"Metadata: {doc.metadata}")
    print()
```

### Using Individual Components

```python
from saplings.memory import MemoryStore
from saplings.retrieval import TFIDFRetriever, EmbeddingRetriever, GraphExpander

# Create a memory store
memory = MemoryStore()

# Add documents to the memory store
for i in range(100):
    memory.add_document(
        content=f"Document {i} about machine learning and artificial intelligence.",
        metadata={"source": f"doc_{i}.txt"}
    )

# Create individual retrievers
tfidf_retriever = TFIDFRetriever(memory_store=memory)
embedding_retriever = EmbeddingRetriever(memory_store=memory)
graph_expander = GraphExpander(memory_store=memory)

# Build the TF-IDF index
tfidf_retriever.build_index()

# Retrieve documents using TF-IDF
tfidf_results = tfidf_retriever.retrieve(query="machine learning", k=20)

# Retrieve documents using embeddings
embedding_results = embedding_retriever.retrieve(
    query="machine learning",
    documents=[doc for doc, _ in tfidf_results],
    k=10
)

# Expand results using the graph
graph_results = graph_expander.expand(
    documents=[doc for doc, _ in embedding_results],
    scores=[score for _, score in embedding_results]
)

# Print the final results
for doc, score in graph_results:
    print(f"Document: {doc.id}, Score: {score:.4f}")
    print(f"Content: {doc.content[:100]}...")
    print()
```

## Advanced Features

### Entropy-based Termination

The entropy calculator determines when to stop retrieval based on the information entropy of the retrieved documents:

```python
from saplings.memory import MemoryStore, Document
from saplings.retrieval import EntropyCalculator, EntropyConfig

# Create an entropy calculator with custom configuration
config = EntropyConfig(
    threshold=0.05,
    max_iterations=5,
    min_documents=10,
    max_documents=100,
    use_normalized_entropy=True,
    window_size=3,
)
calculator = EntropyCalculator(config=config)

# Create some test documents
documents = [
    Document(
        id="1",
        content="Machine learning is a field of artificial intelligence.",
        metadata={"source": "doc_1.txt"}
    ),
    Document(
        id="2",
        content="Deep learning is a subset of machine learning.",
        metadata={"source": "doc_2.txt"}
    ),
    Document(
        id="3",
        content="Neural networks are used in deep learning.",
        metadata={"source": "doc_3.txt"}
    ),
]

# Calculate entropy
entropy = calculator.calculate_entropy(documents)
print(f"Entropy: {entropy:.4f}")

# Calculate entropy change
entropy_change = calculator.calculate_entropy_change(documents)
print(f"Entropy change: {entropy_change:.4f}")

# Check if retrieval should terminate
should_terminate = calculator.should_terminate(documents, iteration=1)
print(f"Should terminate: {should_terminate}")
```

### Saving and Loading

The retrieval components can be saved to disk and loaded back:

```python
from saplings.memory import MemoryStore
from saplings.retrieval import CascadeRetriever

# Create a memory store and retriever
memory = MemoryStore()
retriever = CascadeRetriever(memory_store=memory)

# Add documents and build indices
# ...

# Save the retriever
retriever.save("./retriever")

# Load the retriever
new_retriever = CascadeRetriever(memory_store=memory)
new_retriever.load("./retriever")
```

## Implementation Details

### Cascade Retrieval Process

The cascade retrieval process works as follows:

1. **Initialization**:
   - Reset the entropy calculator
   - Initialize metadata for tracking performance

2. **Iteration Loop**:
   - Perform TF-IDF retrieval (expanding in subsequent iterations)
   - Perform embedding-based retrieval on the TF-IDF results
   - Expand results using the dependency graph
   - Merge new results with existing results
   - Calculate entropy and check termination criteria
   - If termination criteria are met, exit the loop
   - Otherwise, continue to the next iteration

3. **Result Preparation**:
   - Sort the final results by score
   - Limit to the maximum number of documents
   - Return the results with metadata

### Entropy Calculation

The entropy calculator uses information theory to measure the diversity of information in the retrieved documents:

1. **Term Frequency Analysis**:
   - Extract terms from documents using a count vectorizer
   - Calculate term frequencies across all documents

2. **Probability Distribution**:
   - Convert term frequencies to probabilities
   - Filter out zero probabilities

3. **Entropy Calculation**:
   - Calculate Shannon entropy: -Σ(p * log2(p))
   - Optionally normalize by the maximum possible entropy

4. **Termination Decision**:
   - Track entropy changes across iterations
   - Terminate when entropy stabilizes (change below threshold)
   - Also consider minimum/maximum documents and iterations

## Extension Points

The retrieval system is designed to be extensible:

### Custom TF-IDF Retriever

You can create a custom TF-IDF retriever by extending the `TFIDFRetriever` class:

```python
from saplings.retrieval import TFIDFRetriever
from saplings.memory import MemoryStore
from typing import Any, Dict, List, Optional, Tuple

class CustomTFIDFRetriever(TFIDFRetriever):
    def __init__(self, memory_store: MemoryStore, config=None):
        super().__init__(memory_store, config)
        # Custom initialization

    def build_index(self) -> None:
        # Custom index building
        super().build_index()
        # Additional processing

    def retrieve(
        self, query: str, k: Optional[int] = None, filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        # Custom retrieval logic
        results = super().retrieve(query, k, filter_dict)
        # Additional processing
        return results
```

### Custom Embedding Retriever

You can create a custom embedding retriever by extending the `EmbeddingRetriever` class:

```python
from saplings.retrieval import EmbeddingRetriever
from saplings.memory import MemoryStore, Document
from typing import List, Optional, Tuple
import numpy as np

class CustomEmbeddingRetriever(EmbeddingRetriever):
    def __init__(self, memory_store: MemoryStore, config=None):
        super().__init__(memory_store, config)
        # Custom initialization

    def embed_query(self, query: str) -> np.ndarray:
        # Custom query embedding
        embedding = super().embed_query(query)
        # Additional processing
        return embedding

    def retrieve(
        self,
        query: str,
        documents: List[Document],
        k: Optional[int] = None,
    ) -> List[Tuple[Document, float]]:
        # Custom retrieval logic
        results = super().retrieve(query, documents, k)
        # Additional processing
        return results
```

### Custom Graph Expander

You can create a custom graph expander by extending the `GraphExpander` class:

```python
from saplings.retrieval import GraphExpander
from saplings.memory import MemoryStore, Document
from typing import List, Optional, Tuple

class CustomGraphExpander(GraphExpander):
    def __init__(self, memory_store: MemoryStore, config=None):
        super().__init__(memory_store, config)
        # Custom initialization

    def expand(
        self,
        documents: List[Document],
        scores: Optional[List[float]] = None,
    ) -> List[Tuple[Document, float]]:
        # Custom expansion logic
        results = super().expand(documents, scores)
        # Additional processing
        return results
```

## Conclusion

The retrieval system in Saplings provides a powerful and flexible way to find relevant information from the memory store. By combining multiple retrieval methods in a cascaded approach with entropy-based termination, it can efficiently retrieve contextually relevant documents for a wide range of queries.
