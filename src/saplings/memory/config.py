"""
Configuration module for Saplings memory.

This module defines the configuration classes for the memory module.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class VectorStoreType(str, Enum):
    """Types of vector stores supported by Saplings."""

    IN_MEMORY = "in_memory"
    FAISS = "faiss"
    QDRANT = "qdrant"
    PINECONE = "pinecone"
    CUSTOM = "custom"


class SimilarityMetric(str, Enum):
    """Similarity metrics for vector search."""

    COSINE = "cosine"
    DOT_PRODUCT = "dot_product"
    EUCLIDEAN = "euclidean"


class PrivacyLevel(str, Enum):
    """Privacy levels for SecureStore."""

    NONE = "none"  # No privacy measures
    HASH_ONLY = "hash_only"  # Hash document IDs and metadata
    HASH_AND_DP = "hash_and_dp"  # Hash IDs and add differential privacy noise to embeddings


class VectorStoreConfig(BaseModel):
    """Configuration for vector stores."""

    store_type: VectorStoreType = Field(
        VectorStoreType.IN_MEMORY, description="Type of vector store to use"
    )
    similarity_metric: SimilarityMetric = Field(
        SimilarityMetric.COSINE, description="Similarity metric for vector search"
    )
    embedding_dimension: int = Field(1536, description="Dimension of embedding vectors")
    index_name: str = Field("default", description="Name of the vector index")
    persist_directory: Optional[str] = Field(
        None, description="Directory to persist vector store data"
    )
    custom_config: Dict[str, Any] = Field(
        default_factory=dict, description="Custom configuration for specific vector stores"
    )


class GraphConfig(BaseModel):
    """Configuration for dependency graphs."""

    enable_graph: bool = Field(True, description="Whether to enable the dependency graph")
    max_connections_per_node: int = Field(50, description="Maximum number of connections per node")
    min_similarity_threshold: float = Field(
        0.7, description="Minimum similarity threshold for automatic connections"
    )
    enable_entity_extraction: bool = Field(
        True, description="Whether to extract entities from documents"
    )
    entity_types: List[str] = Field(
        ["person", "organization", "location", "concept"],
        description="Types of entities to extract",
    )
    persist_directory: Optional[str] = Field(None, description="Directory to persist graph data")


class SecureStoreConfig(BaseModel):
    """Configuration for SecureStore."""

    privacy_level: PrivacyLevel = Field(
        PrivacyLevel.NONE, description="Privacy level for the store"
    )
    hash_salt: Optional[str] = Field(None, description="Salt for hashing document IDs and metadata")
    dp_epsilon: float = Field(1.0, description="Epsilon parameter for differential privacy")
    dp_delta: float = Field(1e-5, description="Delta parameter for differential privacy")
    dp_sensitivity: float = Field(0.1, description="Sensitivity parameter for differential privacy")


class MemoryConfig(BaseModel):
    """Configuration for the memory store."""

    vector_store: VectorStoreConfig = Field(
        default_factory=VectorStoreConfig, description="Vector store configuration"
    )
    graph: GraphConfig = Field(default_factory=GraphConfig, description="Graph configuration")
    secure_store: SecureStoreConfig = Field(
        default_factory=SecureStoreConfig, description="SecureStore configuration"
    )
    chunk_size: int = Field(1000, description="Default chunk size for documents in characters")
    chunk_overlap: int = Field(200, description="Default chunk overlap for documents in characters")

    @classmethod
    def default(cls) -> "MemoryConfig":
        """
        Create a default configuration.

        Returns:
            MemoryConfig: Default configuration
        """
        return cls()

    @classmethod
    def minimal(cls) -> "MemoryConfig":
        """
        Create a minimal configuration with only essential features enabled.

        Returns:
            MemoryConfig: Minimal configuration
        """
        return cls(
            vector_store=VectorStoreConfig(store_type=VectorStoreType.IN_MEMORY),
            graph=GraphConfig(
                enable_graph=False,
                enable_entity_extraction=False,
            ),
            secure_store=SecureStoreConfig(privacy_level=PrivacyLevel.NONE),
            chunk_size=1000,
            chunk_overlap=0,
        )

    @classmethod
    def secure(cls) -> "MemoryConfig":
        """
        Create a configuration with security features enabled.

        Returns:
            MemoryConfig: Secure configuration
        """
        return cls(
            vector_store=VectorStoreConfig(store_type=VectorStoreType.IN_MEMORY),
            graph=GraphConfig(enable_graph=True),
            secure_store=SecureStoreConfig(
                privacy_level=PrivacyLevel.HASH_AND_DP,
                hash_salt="saplings-secure",  # This should be overridden in production
            ),
            chunk_size=1000,
            chunk_overlap=200,
        )
