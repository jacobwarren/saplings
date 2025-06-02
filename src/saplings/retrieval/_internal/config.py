from __future__ import annotations

"""
Configuration module for Saplings retrieval.

This module defines the configuration classes for the retrieval module.
"""


from pydantic import BaseModel, Field

from saplings.core.caching.interface import CacheStrategy
from saplings.retrieval._internal.cache_config import CacheConfig


class TFIDFConfig(BaseModel):
    """Configuration for TF-IDF retrieval."""

    min_df: float = Field(
        0.01, description="Minimum document frequency for a term to be included in the vocabulary"
    )
    max_df: float = Field(
        0.95, description="Maximum document frequency for a term to be included in the vocabulary"
    )
    max_features: int | None = Field(
        10000, description="Maximum number of features to include in the vocabulary"
    )
    ngram_range: tuple = Field((1, 2), description="Range of n-grams to include in the vocabulary")
    initial_k: int = Field(100, description="Initial number of documents to retrieve using TF-IDF")
    use_idf: bool = Field(True, description="Whether to use inverse document frequency weighting")
    norm: str = Field("l2", description="Norm used to normalize term vectors")
    analyzer: str = Field("word", description="Whether to use word or character n-grams")
    stop_words: str | list[str] | None = Field(
        "english", description="Stop words to remove from the vocabulary"
    )


class EmbeddingConfig(BaseModel):
    """Configuration for embedding-based retrieval."""

    model_name: str = Field("all-MiniLM-L6-v2", description="Name of the embedding model to use")
    embedding_dimension: int = Field(384, description="Dimension of the embedding vectors")
    batch_size: int = Field(32, description="Batch size for embedding generation")
    similarity_top_k: int = Field(
        20, description="Number of documents to retrieve using embedding similarity"
    )
    similarity_cutoff: float | None = Field(
        0.7, description="Minimum similarity score for a document to be included in results"
    )
    use_existing_embeddings: bool = Field(
        True, description="Whether to use existing embeddings if available"
    )


class GraphConfig(BaseModel):
    """Configuration for graph-based retrieval."""

    max_hops: int = Field(2, description="Maximum number of hops to traverse in the graph")
    max_nodes: int = Field(
        50, description="Maximum number of nodes to include in the expanded results"
    )
    min_edge_weight: float = Field(0.5, description="Minimum edge weight for traversal")
    relationship_types: list[str] | None = Field(
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

    threshold: float = Field(0.1, description="Entropy threshold for termination")
    max_iterations: int = Field(3, description="Maximum number of iterations")
    min_documents: int = Field(5, description="Minimum number of documents to retrieve")
    max_documents: int = Field(50, description="Maximum number of documents to retrieve")
    use_normalized_entropy: bool = Field(True, description="Whether to use normalized entropy")
    window_size: int = Field(3, description="Window size for entropy calculation")


class RetrievalConfig(BaseModel):
    """Configuration for the retrieval module."""

    tfidf: TFIDFConfig = Field(
        default_factory=lambda: TFIDFConfig(
            min_df=0.01,
            max_df=0.95,
            max_features=10000,
            ngram_range=(1, 2),
            initial_k=100,
            use_idf=True,
            norm="l2",
            analyzer="word",
            stop_words="english",
        ),
        description="TF-IDF configuration",
    )
    embedding: EmbeddingConfig = Field(
        default_factory=lambda: EmbeddingConfig(
            model_name="all-MiniLM-L6-v2",
            embedding_dimension=384,
            batch_size=32,
            similarity_top_k=20,
            similarity_cutoff=0.7,
            use_existing_embeddings=True,
        ),
        description="Embedding configuration",
    )
    graph: GraphConfig = Field(
        default_factory=lambda: GraphConfig(
            max_hops=2,
            max_nodes=50,
            min_edge_weight=0.5,
            relationship_types=None,
            include_entity_nodes=True,
            score_decay_factor=0.8,
        ),
        description="Graph configuration",
    )
    entropy: EntropyConfig = Field(
        default_factory=lambda: EntropyConfig(
            threshold=0.1,
            max_iterations=3,
            min_documents=5,
            max_documents=50,
            use_normalized_entropy=True,
            window_size=3,
        ),
        description="Entropy configuration",
    )
    cache: CacheConfig = Field(
        default_factory=lambda: CacheConfig(
            enabled=True,
            namespace="vector",
            ttl=3600,
            provider="memory",
            strategy=CacheStrategy.LRU,
            embedding_ttl=86400,
            retrieval_ttl=3600,
            max_size=1000,
            persist=False,
            persist_path=None,
        ),
        description="Cache configuration",
    )

    @classmethod
    def default(cls):
        """
        Create a default configuration.

        Returns
        -------
            RetrievalConfig: Default configuration

        """
        return cls()

    @classmethod
    def minimal(cls):
        """
        Create a minimal configuration with only essential features enabled.

        Returns
        -------
            RetrievalConfig: Minimal configuration

        """
        return cls(
            tfidf=TFIDFConfig(
                min_df=0.01,
                max_df=0.95,
                initial_k=50,
                max_features=5000,
                ngram_range=(1, 1),
                use_idf=True,
                norm="l2",
                analyzer="word",
                stop_words="english",
            ),
            embedding=EmbeddingConfig(
                model_name="all-MiniLM-L6-v2",
                embedding_dimension=384,
                batch_size=32,
                similarity_top_k=10,
                similarity_cutoff=0.7,
                use_existing_embeddings=True,
            ),
            graph=GraphConfig(
                max_hops=1,
                max_nodes=20,
                min_edge_weight=0.5,
                relationship_types=None,
                include_entity_nodes=True,
                score_decay_factor=0.8,
            ),
            entropy=EntropyConfig(
                threshold=0.1,
                max_iterations=2,
                min_documents=3,
                max_documents=20,
                use_normalized_entropy=True,
                window_size=2,
            ),
            cache=CacheConfig.memory_efficient(),
        )

    @classmethod
    def comprehensive(cls):
        """
        Create a comprehensive configuration with all features enabled.

        Returns
        -------
            RetrievalConfig: Comprehensive configuration

        """
        return cls(
            tfidf=TFIDFConfig(
                min_df=0.005,
                max_df=0.98,
                initial_k=200,
                max_features=20000,
                ngram_range=(1, 3),
                use_idf=True,
                norm="l2",
                analyzer="word",
                stop_words="english",
            ),
            embedding=EmbeddingConfig(
                model_name="all-MiniLM-L6-v2",
                embedding_dimension=384,
                batch_size=64,
                similarity_top_k=50,
                similarity_cutoff=0.6,
                use_existing_embeddings=True,
            ),
            graph=GraphConfig(
                max_hops=3,
                max_nodes=100,
                min_edge_weight=0.3,
                relationship_types=None,
                include_entity_nodes=True,
                score_decay_factor=0.7,
            ),
            entropy=EntropyConfig(
                threshold=0.05,
                max_iterations=5,
                min_documents=10,
                max_documents=100,
                use_normalized_entropy=True,
                window_size=5,
            ),
            cache=CacheConfig.high_performance(),
        )

    @classmethod
    def no_cache(cls):
        """
        Create a configuration with caching disabled.

        Returns
        -------
            RetrievalConfig: Configuration with no caching

        """
        return cls(cache=CacheConfig.disabled())
