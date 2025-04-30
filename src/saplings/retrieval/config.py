"""
Configuration module for Saplings retrieval.

This module defines the configuration classes for the retrieval module.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class TFIDFConfig(BaseModel):
    """Configuration for TF-IDF retrieval."""
    
    min_df: float = Field(
        0.01, description="Minimum document frequency for a term to be included in the vocabulary"
    )
    max_df: float = Field(
        0.95, description="Maximum document frequency for a term to be included in the vocabulary"
    )
    max_features: Optional[int] = Field(
        10000, description="Maximum number of features to include in the vocabulary"
    )
    ngram_range: tuple = Field(
        (1, 2), description="Range of n-grams to include in the vocabulary"
    )
    initial_k: int = Field(
        100, description="Initial number of documents to retrieve using TF-IDF"
    )
    use_idf: bool = Field(
        True, description="Whether to use inverse document frequency weighting"
    )
    norm: str = Field(
        "l2", description="Norm used to normalize term vectors"
    )
    analyzer: str = Field(
        "word", description="Whether to use word or character n-grams"
    )
    stop_words: Optional[Union[str, List[str]]] = Field(
        "english", description="Stop words to remove from the vocabulary"
    )


class EmbeddingConfig(BaseModel):
    """Configuration for embedding-based retrieval."""
    
    model_name: str = Field(
        "all-MiniLM-L6-v2", description="Name of the embedding model to use"
    )
    embedding_dimension: int = Field(
        384, description="Dimension of the embedding vectors"
    )
    batch_size: int = Field(
        32, description="Batch size for embedding generation"
    )
    similarity_top_k: int = Field(
        20, description="Number of documents to retrieve using embedding similarity"
    )
    similarity_cutoff: Optional[float] = Field(
        0.7, description="Minimum similarity score for a document to be included in results"
    )
    use_existing_embeddings: bool = Field(
        True, description="Whether to use existing embeddings if available"
    )


class GraphConfig(BaseModel):
    """Configuration for graph-based retrieval."""
    
    max_hops: int = Field(
        2, description="Maximum number of hops to traverse in the graph"
    )
    max_nodes: int = Field(
        50, description="Maximum number of nodes to include in the expanded results"
    )
    min_edge_weight: float = Field(
        0.5, description="Minimum edge weight for traversal"
    )
    relationship_types: Optional[List[str]] = Field(
        None, description="Types of relationships to traverse"
    )
    include_entity_nodes: bool = Field(
        True, description="Whether to include entity nodes in the traversal"
    )
    score_decay_factor: float = Field(
        0.8, description="Factor by which to decay scores with each hop"
    )


class EntropyConfig(BaseModel):
    """Configuration for entropy-based termination."""
    
    threshold: float = Field(
        0.1, description="Entropy threshold for termination"
    )
    max_iterations: int = Field(
        3, description="Maximum number of iterations"
    )
    min_documents: int = Field(
        5, description="Minimum number of documents to retrieve"
    )
    max_documents: int = Field(
        50, description="Maximum number of documents to retrieve"
    )
    use_normalized_entropy: bool = Field(
        True, description="Whether to use normalized entropy"
    )
    window_size: int = Field(
        3, description="Window size for entropy calculation"
    )


class RetrievalConfig(BaseModel):
    """Configuration for the retrieval module."""
    
    tfidf: TFIDFConfig = Field(
        default_factory=TFIDFConfig, description="TF-IDF configuration"
    )
    embedding: EmbeddingConfig = Field(
        default_factory=EmbeddingConfig, description="Embedding configuration"
    )
    graph: GraphConfig = Field(
        default_factory=GraphConfig, description="Graph configuration"
    )
    entropy: EntropyConfig = Field(
        default_factory=EntropyConfig, description="Entropy configuration"
    )
    
    @classmethod
    def default(cls) -> "RetrievalConfig":
        """
        Create a default configuration.
        
        Returns:
            RetrievalConfig: Default configuration
        """
        return cls()
    
    @classmethod
    def minimal(cls) -> "RetrievalConfig":
        """
        Create a minimal configuration with only essential features enabled.
        
        Returns:
            RetrievalConfig: Minimal configuration
        """
        return cls(
            tfidf=TFIDFConfig(
                initial_k=50,
                max_features=5000,
            ),
            embedding=EmbeddingConfig(
                similarity_top_k=10,
            ),
            graph=GraphConfig(
                max_hops=1,
                max_nodes=20,
            ),
            entropy=EntropyConfig(
                max_iterations=2,
                max_documents=20,
            ),
        )
    
    @classmethod
    def comprehensive(cls) -> "RetrievalConfig":
        """
        Create a comprehensive configuration with all features enabled.
        
        Returns:
            RetrievalConfig: Comprehensive configuration
        """
        return cls(
            tfidf=TFIDFConfig(
                initial_k=200,
                max_features=20000,
                ngram_range=(1, 3),
            ),
            embedding=EmbeddingConfig(
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
