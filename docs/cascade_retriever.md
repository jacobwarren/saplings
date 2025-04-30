# Cascade Retriever with Entropy-Aware Loop

## Overview

The Cascade Retriever is a sophisticated retrieval system that orchestrates a multi-stage retrieval pipeline with an entropy-aware termination mechanism. It combines the strengths of different retrieval methods (TF-IDF, embeddings, and graph-based) while dynamically determining when sufficient information has been retrieved.

The key features of the Cascade Retriever include:
- **Multi-stage retrieval pipeline**: TF-IDF → embeddings → graph expansion
- **Entropy-based termination**: Automatically determines when to stop retrieving
- **Iterative refinement**: Expands the search in subsequent iterations
- **Configurable thresholds**: Customizable parameters for different use cases

## Key Components

### CascadeRetriever

The `CascadeRetriever` class is the main component that orchestrates the retrieval pipeline:

```python
class CascadeRetriever:
    def __init__(
        self,
        memory_store: MemoryStore,
        config: Optional[RetrievalConfig] = None,
    ):
        # Initialize the cascade retriever
```

The `CascadeRetriever` takes a memory store and configuration as input, and provides methods for retrieving documents.

### RetrievalResult

The `RetrievalResult` class represents the result of a retrieval operation:

```python
class RetrievalResult:
    def __init__(
        self,
        documents: List[Document],
        scores: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        # Initialize the retrieval result
```

### EntropyCalculator

The `EntropyCalculator` class calculates the information entropy of retrieval results to determine when sufficient information has been retrieved:

```python
class EntropyCalculator:
    def __init__(
        self,
        config: Optional[Union[RetrievalConfig, EntropyConfig]] = None,
    ):
        # Initialize the entropy calculator
```

### RetrievalConfig

The `RetrievalConfig` class provides configuration options for the retrieval pipeline:

```python
class RetrievalConfig(BaseModel):
    tfidf: TFIDFConfig = Field(default_factory=TFIDFConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    entropy: EntropyConfig = Field(default_factory=EntropyConfig)
```

## How the Cascade Retriever Works

### 1. Retrieval Pipeline

The `retrieve` method in `CascadeRetriever` is the main entry point for retrieving documents:

```python
def retrieve(
    self,
    query: str,
    filter_dict: Optional[Dict[str, Any]] = None,
    max_documents: Optional[int] = None,
) -> RetrievalResult:
    # Retrieve documents relevant to the query
```

The retrieval process involves multiple stages:

#### Stage 1: TF-IDF Retrieval

The first stage uses TF-IDF (Term Frequency-Inverse Document Frequency) to quickly filter documents based on keyword matching:

```python
# Step 1: TF-IDF retrieval
tfidf_start = time.time()
if iteration == 1:
    # Build TF-IDF index if not already built
    if not self.tfidf_retriever.is_built:
        self.tfidf_retriever.build_index()
    
    # Initial TF-IDF retrieval
    tfidf_results = self.tfidf_retriever.retrieve(
        query=query,
        filter_dict=filter_dict,
    )
else:
    # Expand TF-IDF retrieval in subsequent iterations
    tfidf_results = self.tfidf_retriever.retrieve(
        query=query,
        k=self.config.tfidf.initial_k * iteration,
        filter_dict=filter_dict,
    )
```

#### Stage 2: Embedding-based Retrieval

The second stage uses embeddings to find semantically similar documents:

```python
# Step 2: Embedding-based retrieval
embedding_start = time.time()
embedding_results = self.embedding_retriever.retrieve(
    query=query,
    documents=tfidf_docs,
    k=self.config.embedding.similarity_top_k,
)
```

#### Stage 3: Graph Expansion

The third stage uses the dependency graph to find related documents:

```python
# Step 3: Graph expansion
graph_start = time.time()
graph_results = self.graph_expander.expand(
    documents=embedding_docs,
    scores=embedding_scores,
)
```

### 2. Entropy-Aware Termination

The Cascade Retriever uses an entropy-based mechanism to determine when to stop retrieving:

```python
# Step 4: Check termination condition
entropy_start = time.time()
should_terminate = self.entropy_calculator.should_terminate(
    documents=all_documents,
    iteration=iteration,
)
```

The `EntropyCalculator` calculates the information entropy of the retrieved documents and determines if the entropy has stabilized, indicating that sufficient information has been retrieved.

#### Entropy Calculation

The entropy is calculated based on the term frequency distribution in the retrieved documents:

```python
def calculate_entropy(self, documents: List[Document]) -> float:
    # Extract document contents
    contents = [doc.content for doc in documents]
    
    # Calculate term frequencies
    term_counts = self.vectorizer.fit_transform(contents)
    term_freqs = term_counts.sum(axis=0).A1
    
    # Calculate probability distribution
    total_terms = term_freqs.sum()
    probabilities = term_freqs / total_terms
    
    # Filter out zero probabilities
    probabilities = probabilities[probabilities > 0]
    
    # Calculate entropy
    entropy = -np.sum(probabilities * np.log2(probabilities))
    
    # Normalize if configured
    if self.config.use_normalized_entropy:
        max_entropy = math.log2(len(probabilities))
        if max_entropy > 0:
            entropy /= max_entropy
    
    return float(entropy)
```

#### Termination Condition

The termination condition is based on the change in entropy over iterations:

```python
def should_terminate(
    self, documents: List[Document], iteration: int
) -> bool:
    # Check if we have enough documents
    if len(documents) < self.config.min_documents:
        return False
    
    # Check if we've reached the maximum number of iterations
    if iteration >= self.config.max_iterations:
        return True
    
    # Calculate entropy change
    entropy_change = self.calculate_entropy_change(documents)
    
    # Check if entropy change is below threshold
    if abs(entropy_change) <= self.config.threshold:
        return True
    
    # Check if entropy is decreasing
    if self.config.terminate_on_decreasing_entropy and entropy_change < 0:
        return True
    
    return False
```

### 3. Iterative Refinement

If the termination condition is not met, the Cascade Retriever expands the search in subsequent iterations:

```python
# Main retrieval loop
iteration = 0
while True:
    iteration += 1
    metadata["iterations"] = iteration
    
    # ... (retrieval stages) ...
    
    # Check termination condition
    should_terminate = self.entropy_calculator.should_terminate(
        documents=all_documents,
        iteration=iteration,
    )
    
    if should_terminate:
        break
```

In each iteration, the TF-IDF retriever expands its search to include more documents:

```python
# Expand TF-IDF retrieval in subsequent iterations
tfidf_results = self.tfidf_retriever.retrieve(
    query=query,
    k=self.config.tfidf.initial_k * iteration,
    filter_dict=filter_dict,
)
```

## Example Usage

### Basic Usage

```python
from saplings.memory import MemoryStore
from saplings.retrieval import CascadeRetriever, RetrievalConfig

# Create a memory store
memory_store = MemoryStore()

# Add documents to the memory store
# ...

# Create a cascade retriever
retriever = CascadeRetriever(memory_store=memory_store)

# Retrieve documents
result = retriever.retrieve(
    query="What is machine learning?",
    max_documents=10,
)

# Access the retrieved documents
for doc, score in zip(result.documents, result.scores):
    print(f"Document: {doc.id}, Score: {score}")
    print(f"Content: {doc.content[:100]}...")
    print()

# Access metadata
metadata = result.get_metadata()
print(f"Iterations: {metadata['iterations']}")
print(f"Total time: {metadata['total_time']:.2f} seconds")
print(f"Final entropy: {metadata['final_entropy']:.2f}")
```

### Custom Configuration

```python
from saplings.retrieval import RetrievalConfig, TFIDFConfig, EmbeddingConfig, GraphConfig, EntropyConfig

# Create a custom configuration
config = RetrievalConfig(
    tfidf=TFIDFConfig(
        initial_k=100,
        max_features=10000,
        ngram_range=(1, 2),
    ),
    embedding=EmbeddingConfig(
        similarity_top_k=20,
        similarity_cutoff=0.7,
    ),
    graph=GraphConfig(
        max_hops=2,
        max_nodes=50,
        score_decay_factor=0.7,
    ),
    entropy=EntropyConfig(
        threshold=0.05,
        max_iterations=3,
        min_documents=5,
        max_documents=50,
        window_size=2,
        use_normalized_entropy=True,
        terminate_on_decreasing_entropy=True,
    ),
)

# Create a cascade retriever with the custom configuration
retriever = CascadeRetriever(
    memory_store=memory_store,
    config=config,
)
```

### Filtering Documents

```python
# Retrieve documents with filtering
result = retriever.retrieve(
    query="What is machine learning?",
    filter_dict={
        "source": "textbook",
        "date": {"$gte": "2020-01-01"},
    },
)
```

### Saving and Loading

```python
# Save the retriever
retriever.save("retriever_data")

# Load the retriever
new_retriever = CascadeRetriever(memory_store=memory_store)
new_retriever.load("retriever_data")
```

## Advanced Configuration

### TF-IDF Configuration

```python
tfidf_config = TFIDFConfig(
    initial_k=200,  # Initial number of documents to retrieve
    max_features=20000,  # Maximum number of features for TF-IDF
    ngram_range=(1, 3),  # Use unigrams, bigrams, and trigrams
    use_idf=True,  # Use inverse document frequency
    sublinear_tf=True,  # Apply sublinear scaling to term frequencies
)
```

### Embedding Configuration

```python
embedding_config = EmbeddingConfig(
    similarity_top_k=50,  # Number of documents to retrieve
    similarity_cutoff=0.6,  # Minimum similarity score
    embedding_model="text-embedding-ada-002",  # Embedding model to use
)
```

### Graph Configuration

```python
graph_config = GraphConfig(
    max_hops=3,  # Maximum number of hops to traverse
    max_nodes=100,  # Maximum number of nodes to include
    min_edge_weight=0.3,  # Minimum edge weight for traversal
    relationship_types=["contains", "references", "similar_to"],  # Types of relationships to traverse
    include_entity_nodes=True,  # Include entity nodes in traversal
    score_decay_factor=0.8,  # Factor by which to decay scores with each hop
)
```

### Entropy Configuration

```python
entropy_config = EntropyConfig(
    threshold=0.05,  # Threshold for entropy change
    max_iterations=5,  # Maximum number of iterations
    min_documents=5,  # Minimum number of documents to retrieve
    max_documents=100,  # Maximum number of documents to retrieve
    window_size=3,  # Window size for entropy history
    use_normalized_entropy=True,  # Use normalized entropy
    terminate_on_decreasing_entropy=True,  # Terminate when entropy decreases
)
```

## Performance Considerations

- **Efficiency**: The cascade approach is more efficient than applying all retrieval methods to the entire document set.
- **Quality**: The multi-stage pipeline improves retrieval quality by combining different retrieval methods.
- **Adaptivity**: The entropy-aware termination adapts to the query and document set.
- **Scalability**: The TF-IDF initial filtering makes the approach scalable to large document sets.

## Limitations

- **Computational Cost**: The multi-stage pipeline can be computationally expensive for large document sets.
- **Parameter Sensitivity**: Performance can be sensitive to configuration parameters.
- **Cold Start**: The entropy-based termination may not work well with very small document sets.

## Future Directions

- **Learning to Rank**: Incorporating learning-to-rank techniques to improve ranking.
- **Query Expansion**: Automatically expanding queries to improve recall.
- **Personalization**: Adapting retrieval to user preferences and history.
- **Multi-modal Retrieval**: Extending to retrieve images, audio, and other modalities.
