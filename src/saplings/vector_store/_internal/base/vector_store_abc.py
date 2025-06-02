from __future__ import annotations

"""
Vector store abstract base class for Saplings.

This module provides the abstract base class for vector stores to avoid circular imports.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class VectorStoreABC(ABC):
    """
    Abstract base class for vector stores.

    Vector stores are used to store and retrieve vectors, typically for
    similarity search operations.
    """

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
    def clear(self) -> None:
        """Clear all vectors from the store."""
