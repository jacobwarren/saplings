from __future__ import annotations

"""
In-memory vector store implementation for Saplings.

This module provides a simple in-memory vector store implementation for Saplings.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Import the abstract base class at runtime to avoid circular imports
from saplings.vector_store._internal.base.vector_store_abc import VectorStoreABC


class InMemoryVectorStore(VectorStoreABC):
    """
    In-memory vector store implementation.

    This vector store keeps all vectors in memory, which is useful for
    testing and small-scale applications.
    """

    def __init__(self) -> None:
        """Initialize the in-memory vector store."""
        self._vectors: Dict[str, np.ndarray] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._next_id = 0

    def add_vectors(
        self,
        vectors: List[np.ndarray],
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Add vectors to the store.

        Args:
        ----
            vectors: List of vectors to add
            metadata: Optional metadata for each vector
            ids: Optional IDs for each vector

        Returns:
        -------
            List[str]: IDs of the added vectors

        """
        if metadata is None:
            metadata = [{} for _ in vectors]

        if ids is None:
            ids = [f"id_{self._next_id + i}" for i in range(len(vectors))]
            self._next_id += len(vectors)

        for i, (vector, meta, id) in enumerate(zip(vectors, metadata, ids)):
            self._vectors[id] = vector
            self._metadata[id] = meta

        return ids

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for similar vectors.

        Args:
        ----
            query_vector: Vector to search for
            k: Number of results to return
            filter: Optional filter to apply to the search

        Returns:
        -------
            List[Tuple[str, float, Dict[str, Any]]]: List of (id, score, metadata) tuples

        """
        results = []

        for id, vector in self._vectors.items():
            # Apply filter if provided
            if filter is not None:
                metadata = self._metadata[id]
                if not self._matches_filter(metadata, filter):
                    continue

            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_vector, vector)
            results.append((id, similarity, self._metadata[id]))

        # Sort by similarity (highest first) and take top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def get_vector(self, id: str) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Get a vector by ID.

        Args:
        ----
            id: ID of the vector to get

        Returns:
        -------
            Optional[Tuple[np.ndarray, Dict[str, Any]]]: Vector and metadata if found

        """
        if id not in self._vectors:
            return None

        return (self._vectors[id], self._metadata[id])

    def delete_vector(self, id: str) -> bool:
        """
        Delete a vector by ID.

        Args:
        ----
            id: ID of the vector to delete

        Returns:
        -------
            bool: Whether the vector was deleted

        """
        if id not in self._vectors:
            return False

        del self._vectors[id]
        del self._metadata[id]
        return True

    def clear(self) -> None:
        """Clear all vectors from the store."""
        self._vectors.clear()
        self._metadata.clear()
        self._next_id = 0

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
        ----
            a: First vector
            b: Second vector

        Returns:
        -------
            float: Cosine similarity

        """
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def _matches_filter(self, metadata: Dict[str, Any], filter: Dict[str, Any]) -> bool:
        """
        Check if metadata matches a filter.

        Args:
        ----
            metadata: Metadata to check
            filter: Filter to apply

        Returns:
        -------
            bool: Whether the metadata matches the filter

        """
        for key, value in filter.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True
