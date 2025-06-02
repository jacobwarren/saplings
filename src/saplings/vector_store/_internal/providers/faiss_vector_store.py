from __future__ import annotations

"""
FAISS vector store implementation for Saplings.

This module provides a high-performance vector store implementation using Facebook AI
Similarity Search (FAISS) library. FAISS is optimized for fast similarity searches on
high-dimensional vectors and supports both CPU and GPU execution.
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

# Import the abstract base class to avoid circular imports
from saplings.vector_store._internal.base.vector_store_abc import VectorStoreABC

# For type checking
if TYPE_CHECKING:
    import faiss  # type: ignore[import-not-found]

# Runtime import
try:
    import faiss  # type: ignore[import]
except ImportError:
    faiss = None


def _require_faiss() -> Any:
    """
    Ensure FAISS is available and return the module.

    Raises
    ------
        ImportError: If FAISS is not installed

    Returns
    -------
        The FAISS module

    """
    if faiss is None:
        raise ImportError("FAISS is optional – install with `pip install saplings[faiss]`.")
    return faiss


logger = logging.getLogger(__name__)


class FaissVectorStore(VectorStoreABC):
    """
    FAISS-based implementation of the VectorStore interface.

    This implementation uses FAISS for high-performance vector similarity search,
    with support for both CPU and GPU acceleration. FAISS is highly optimized for
    large scale similarity search and clustering of dense vectors.
    """

    def __init__(self, use_gpu: bool = False) -> None:
        """
        Initialize the FAISS vector store.

        Args:
        ----
            use_gpu: Whether to use GPU acceleration (requires faiss-gpu)

        Raises:
        ------
            ImportError: If FAISS is not installed

        """
        # Check if FAISS is available
        _require_faiss()

        self.use_gpu = use_gpu
        self._index = None
        self._vectors: Dict[str, np.ndarray] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._id_to_index: Dict[str, int] = {}
        self._index_to_id: Dict[int, str] = {}
        self._next_index = 0

    def _initialize_index(self, dim: int) -> None:
        """
        Initialize the FAISS index with the specified dimension.

        Args:
        ----
            dim: Vector dimension

        """
        if self._index is not None:
            return

        # Ensure FAISS is available
        faiss_module = _require_faiss()

        logger.info(f"Initializing FAISS index with dimension {dim}")

        # Create index for inner product (cosine similarity with normalized vectors)
        self._index = faiss_module.IndexFlatIP(dim)

        # Move to GPU if requested and available
        if self.use_gpu:
            try:
                logger.info("Moving FAISS index to GPU")
                res = faiss_module.StandardGpuResources()
                self._index = faiss_module.index_cpu_to_gpu(res, 0, self._index)
            except Exception as e:
                logger.warning(f"Failed to use GPU for FAISS: {e}. Falling back to CPU.")
                self.use_gpu = False

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
        if not vectors:
            return []

        # Initialize index if not already done
        if self._index is None:
            self._initialize_index(vectors[0].shape[0])

        # Prepare metadata
        if metadata is None:
            metadata = [{} for _ in vectors]

        # Prepare IDs
        if ids is None:
            ids = [f"id_{self._next_index + i}" for i in range(len(vectors))]

        # Prepare vectors for FAISS
        faiss_vectors = np.vstack([v.astype(np.float32).reshape(1, -1) for v in vectors])

        # Normalize vectors for cosine similarity
        _require_faiss().normalize_L2(faiss_vectors)

        # Add to FAISS index
        if self._index is not None:
            self._index.add(faiss_vectors)

        # Store vectors and metadata
        for i, (vector, meta, id) in enumerate(zip(vectors, metadata, ids)):
            self._vectors[id] = vector
            self._metadata[id] = meta
            self._id_to_index[id] = self._next_index + i
            self._index_to_id[self._next_index + i] = id

        self._next_index += len(vectors)
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
        if not self._vectors or self._index is None:
            return []

        # Prepare the query vector
        query_vector = query_vector.astype(np.float32).reshape(1, -1)

        # Normalize for cosine similarity
        _require_faiss().normalize_L2(query_vector)

        # If we have filters, we need to get more results and post-filter
        actual_k = k
        if filter:
            # Get more results to account for filtering
            actual_k = min(len(self._vectors), max(k * 3, k + 50))

        # Perform the search
        distances, indices = self._index.search(query_vector, actual_k)

        # Process results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:  # FAISS may return -1 for padding when not enough results
                continue

            id = self._index_to_id.get(idx)
            if id is None:
                continue

            metadata = self._metadata.get(id, {})

            # Apply filter if provided
            if filter and not self._matches_filter(metadata, filter):
                continue

            # Add to results
            results.append((id, float(distances[0][i]), metadata))

            # Check if we have enough results after filtering
            if len(results) >= k:
                break

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

        # Remove from vectors and metadata
        del self._vectors[id]
        del self._metadata[id]

        # FAISS doesn't support direct deletion, so we need to rebuild the index
        if id in self._id_to_index:
            # Remove from mappings
            index_to_remove = self._id_to_index[id]
            del self._id_to_index[id]
            del self._index_to_id[index_to_remove]

            # If this was the only vector, just clear everything
            if not self._vectors:
                self._index = None
                self._next_index = 0
                return True

            # Otherwise, rebuild the index with remaining vectors
            self._rebuild_index()

        return True

    def clear(self) -> None:
        """Clear all vectors from the store."""
        self._vectors.clear()
        self._metadata.clear()
        self._id_to_index.clear()
        self._index_to_id.clear()
        self._next_index = 0
        self._index = None

    def _rebuild_index(self) -> None:
        """Rebuild the FAISS index from scratch using current vectors."""
        # Get the dimension from any vector
        for vector in self._vectors.values():
            dim = vector.shape[0]
            # Reinitialize the index
            self._initialize_index(dim)
            break
        else:
            # If no vectors, we can't rebuild the index
            logger.warning("No vectors found, can't rebuild index")
            return

        # Clear mappings
        self._id_to_index.clear()
        self._index_to_id.clear()
        self._next_index = 0

        # Add all vectors to the index
        vectors = []
        ids = []
        metadata = []

        for id, vector in self._vectors.items():
            vectors.append(vector)
            ids.append(id)
            metadata.append(self._metadata[id])

        # Use add_vectors to rebuild
        self.add_vectors(vectors, metadata, ids)

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
