from __future__ import annotations

"""
FAISS vector store implementation for Saplings.

This module provides a high-performance vector store implementation using Facebook AI
Similarity Search (FAISS) library. FAISS is optimized for fast similarity searches on
high-dimensional vectors and supports both CPU and GPU execution.
"""


import datetime
import json
import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

# Import from public API instead of internal implementation
from saplings.api.memory import MemoryConfig
from saplings.api.memory.document import Document
from saplings.api.vector_store import VectorStore

# Import SimilarityMetric from memory config
# This is needed because it's an enum that might not be exposed in the public API
from saplings.memory._internal.config import SimilarityMetric

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


if TYPE_CHECKING:
    import builtins

logger = logging.getLogger(__name__)


class FaissVectorStore(VectorStore):
    """
    FAISS-based implementation of the VectorStore interface.

    This implementation uses FAISS for high-performance vector similarity search,
    with support for both CPU and GPU acceleration. FAISS is highly optimized for
    large scale similarity search and clustering of dense vectors.
    """

    def __init__(self, config: MemoryConfig | None = None, use_gpu: bool = False) -> None:
        """
        Initialize the FAISS vector store.

        Args:
        ----
            config: Memory configuration
            use_gpu: Whether to use GPU acceleration (requires faiss-gpu)

        Raises:
        ------
            ImportError: If FAISS is not installed

        """
        # Check if FAISS is available
        _require_faiss()

        self.config = config or MemoryConfig.default()
        self.documents: dict[str, Document] = {}
        self.use_gpu = use_gpu
        self._index = None
        self._id_to_index: dict[str, int] = {}
        self._index_to_id: dict[int, str] = {}
        self._next_index = 0
        self.similarity_metric = self.config.vector_store.similarity_metric

        # Initialize the index lazily when the first document is added
        # This allows us to determine the vector dimension from the document

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

        # Create index based on similarity metric
        if self.similarity_metric == SimilarityMetric.COSINE:
            # For cosine similarity, we need to normalize vectors before adding them
            # and use the inner product (dot product) as the similarity measure
            self._index = faiss_module.IndexFlatIP(dim)
        elif self.similarity_metric == SimilarityMetric.DOT_PRODUCT:
            # Dot product is directly supported by FAISS
            self._index = faiss_module.IndexFlatIP(dim)
        elif self.similarity_metric == SimilarityMetric.EUCLIDEAN:
            # For Euclidean distance, use L2 distance
            self._index = faiss_module.IndexFlatL2(dim)
        else:
            # Default to inner product
            logger.warning(
                f"Unknown similarity metric: {self.similarity_metric}, defaulting to dot product"
            )
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

    def add_document(self, document: Document) -> None:
        """
        Add a document to the vector store.

        Args:
        ----
            document: Document to add

        """
        if document.embedding is None:
            msg = f"Document {document.id} has no embedding"
            raise ValueError(msg)

        # Initialize index with first document dimension if not already done
        if self._index is None:
            self._initialize_index(document.embedding.shape[0])

        # Store the document
        self.documents[document.id] = document

        # Get the embedding and prepare for FAISS
        embedding = document.embedding.astype(np.float32).reshape(1, -1)

        # Normalize for cosine similarity if needed
        if self.similarity_metric == SimilarityMetric.COSINE:
            _require_faiss().normalize_L2(embedding)

        # Add to FAISS index
        if self._index is not None:
            self._index.add(embedding)

        # Map document ID to index
        self._id_to_index[document.id] = self._next_index
        self._index_to_id[self._next_index] = document.id
        self._next_index += 1

        # Also add any chunks
        for chunk in document.chunks:
            if chunk.embedding is not None:
                # Store the chunk document
                self.documents[chunk.id] = chunk

                # Get the embedding and prepare for FAISS
                chunk_embedding = chunk.embedding.astype(np.float32).reshape(1, -1)

                # Normalize for cosine similarity if needed
                if self.similarity_metric == SimilarityMetric.COSINE:
                    _require_faiss().normalize_L2(chunk_embedding)

                # Add to FAISS index
                if self._index is not None:
                    self._index.add(chunk_embedding)

                # Map chunk ID to index
                self._id_to_index[chunk.id] = self._next_index
                self._index_to_id[self._next_index] = chunk.id
                self._next_index += 1

    def add_documents(self, documents: builtins.list[Document]) -> None:
        """
        Add multiple documents to the vector store.

        Args:
        ----
            documents: Documents to add

        """
        if not documents:
            return

        # Add documents one by one
        # This could be optimized for batch insertion in the future
        for document in documents:
            self.add_document(document)

    def search(
        self,
        query_embedding: np.ndarray,
        limit: int = 10,
        filter_dict: dict[str, Any] | None = None,
    ) -> builtins.list[tuple[Document, float]]:
        """
        Search for similar documents.

        Args:
        ----
            query_embedding: Query embedding vector
            limit: Maximum number of results
            filter_dict: Optional filter criteria

        Returns:
        -------
            List[Tuple[Document, float]]: List of (document, similarity_score) tuples

        """
        if not self.documents or self._index is None:
            return []

        # Prepare the query embedding
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)

        # Normalize for cosine similarity if needed
        if self.similarity_metric == SimilarityMetric.COSINE:
            _require_faiss().normalize_L2(query_embedding)

        # If we have filters, we need to get more results and post-filter
        actual_limit = limit
        if filter_dict:
            # Get more results to account for filtering
            actual_limit = min(len(self.documents), max(limit * 3, limit + 50))

        # Perform the search
        distances, indices = self._index.search(query_embedding, actual_limit)

        # Process results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:  # FAISS may return -1 for padding when not enough results
                continue

            doc_id = self._index_to_id.get(idx)
            if doc_id is None:
                continue

            document = self.documents.get(doc_id)
            if document is None:
                continue

            # Apply post-filtering if needed
            if filter_dict and not self._matches_filter(document, filter_dict):
                continue

            # For inner product, distance is already the similarity
            # For L2 distance, we need to convert to similarity
            similarity = distances[0][i]
            if self.similarity_metric == SimilarityMetric.EUCLIDEAN:
                # Convert L2 distance to similarity (1 / (1 + distance))
                similarity = 1.0 / (1.0 + similarity)

            results.append((document, float(similarity)))

            # Check if we have enough results after filtering
            if len(results) >= limit:
                break

        # Sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:limit]

    def delete(self, document_id: str) -> bool:
        """
        Delete a document from the vector store.

        Args:
        ----
            document_id: ID of the document to delete

        Returns:
        -------
            bool: True if the document was deleted, False otherwise

        """
        if document_id not in self.documents:
            return False

        # Remove from documents dict
        del self.documents[document_id]

        # FAISS doesn't support direct deletion, so we need to rebuild the index
        # This is an inefficient operation but happens infrequently
        if document_id in self._id_to_index:
            # Remove from mappings
            index_to_remove = self._id_to_index[document_id]
            del self._id_to_index[document_id]
            del self._index_to_id[index_to_remove]

            # If this was the only document, just clear everything
            if not self.documents:
                self._index = None
                self._next_index = 0
                return True

            # Otherwise, rebuild the index with remaining documents
            self._rebuild_index()

        return True

    def update(self, document: Document) -> None:
        """
        Update a document in the vector store.

        Args:
        ----
            document: Updated document

        """
        if document.id not in self.documents:
            msg = f"Document {document.id} not found"
            raise ValueError(msg)

        if document.embedding is None:
            msg = f"Document {document.id} has no embedding"
            raise ValueError(msg)

        # FAISS doesn't support updates, so we delete and re-add
        self.delete(document.id)
        self.add_document(document)

    def get(self, document_id: str) -> Document | None:
        """
        Get a document by ID.

        Args:
        ----
            document_id: ID of the document to get

        Returns:
        -------
            Optional[Document]: Document if found, None otherwise

        """
        return self.documents.get(document_id)

    def list(
        self, limit: int = 100, filter_dict: dict[str, Any] | None = None
    ) -> builtins.list[Document]:
        """
        List documents in the vector store.

        Args:
        ----
            limit: Maximum number of documents to return
            filter_dict: Optional filter criteria

        Returns:
        -------
            List[Document]: List of documents

        """
        if filter_dict:
            docs = [
                doc for doc in self.documents.values() if self._matches_filter(doc, filter_dict)
            ]
        else:
            docs = list(self.documents.values())

        return docs[:limit]

    def count(self, filter_dict: dict[str, Any] | None = None) -> int:
        """
        Count documents in the vector store.

        Args:
        ----
            filter_dict: Optional filter criteria

        Returns:
        -------
            int: Number of documents

        """
        if filter_dict:
            return len(
                [doc for doc in self.documents.values() if self._matches_filter(doc, filter_dict)]
            )
        return len(self.documents)

    def clear(self):
        """Clear all documents from the vector store."""
        self.documents.clear()
        self._id_to_index.clear()
        self._index_to_id.clear()
        self._next_index = 0
        self._index = None

    def save(self, directory: str) -> None:
        """
        Save the vector store to disk.

        Args:
        ----
            directory: Directory to save to

        """
        directory_path = Path(directory)
        directory_path.mkdir(parents=True, exist_ok=True)

        # Save documents
        documents_data = {doc_id: doc.to_dict() for doc_id, doc in self.documents.items()}

        # Custom JSON encoder to handle datetime objects
        class DateTimeEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, datetime.datetime):
                    return obj.isoformat()
                return super().default(obj)

        with open(directory_path / "documents.json", "w") as f:
            json.dump(documents_data, f, cls=DateTimeEncoder)

        # Save mappings
        with open(directory_path / "mappings.pkl", "wb") as f:
            pickle.dump(
                {
                    "id_to_index": self._id_to_index,
                    "index_to_id": self._index_to_id,
                    "next_index": self._next_index,
                },
                f,
            )

        # Save FAISS index if it exists
        if self._index is not None:
            faiss_module = _require_faiss()
            # Move to CPU if on GPU before saving
            index_to_save = self._index
            if self.use_gpu:
                index_to_save = faiss_module.index_gpu_to_cpu(self._index)

            faiss_module.write_index(index_to_save, str(directory_path / "faiss.index"))

        # Save config
        with open(directory_path / "config.json", "w") as f:
            json.dump(self.config.model_dump(), f)

    def load(self, directory: str) -> None:
        """
        Load the vector store from disk.

        Args:
        ----
            directory: Directory to load from

        """
        directory_path = Path(directory)

        # Load config
        config_path = directory_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config_data = json.load(f)
                self.config = MemoryConfig(**config_data)

        # Load documents
        documents_path = directory_path / "documents.json"
        if documents_path.exists():
            with open(documents_path) as f:
                documents_data = json.load(f)
                self.documents = {
                    doc_id: Document.from_dict(doc_data)
                    for doc_id, doc_data in documents_data.items()
                }

        # Load mappings
        mappings_path = directory_path / "mappings.pkl"
        if mappings_path.exists():
            with open(mappings_path, "rb") as f:
                mappings = pickle.load(f)
                self._id_to_index = mappings["id_to_index"]
                self._index_to_id = mappings["index_to_id"]
                self._next_index = mappings["next_index"]

        # Load FAISS index
        index_path = directory_path / "faiss.index"
        if index_path.exists():
            faiss_module = _require_faiss()
            self._index = faiss_module.read_index(str(index_path))

            # Move to GPU if requested
            if self.use_gpu:
                try:
                    res = faiss_module.StandardGpuResources()
                    self._index = faiss_module.index_cpu_to_gpu(res, 0, self._index)
                except Exception as e:
                    logger.warning(f"Failed to use GPU for FAISS: {e}. Falling back to CPU.")
                    self.use_gpu = False

    def _rebuild_index(self):
        """Rebuild the FAISS index from scratch using current documents."""
        # Get the dimension from any document with an embedding
        for doc in self.documents.values():
            if doc.embedding is not None:
                dim = doc.embedding.shape[0]
                # Reinitialize the index
                self._initialize_index(dim)
                break
        else:
            # If no document has an embedding, we can't rebuild the index
            logger.warning("No documents with embeddings found, can't rebuild index")

        # Clear mappings
        self._id_to_index.clear()
        self._index_to_id.clear()
        self._next_index = 0

        # Add all documents to the index
        for doc_id, document in self.documents.items():
            if document.embedding is None:
                continue

            # Add the document embedding
            embedding = document.embedding.astype(np.float32).reshape(1, -1)

            # Normalize for cosine similarity if needed
            if self.similarity_metric == SimilarityMetric.COSINE:
                _require_faiss().normalize_L2(embedding)

            # Add to FAISS index
            if self._index is not None:
                self._index.add(embedding)

            # Update mappings
            self._id_to_index[doc_id] = self._next_index
            self._index_to_id[self._next_index] = doc_id
            self._next_index += 1

    def _matches_filter(self, document: Document, filter_dict: dict[str, Any]) -> bool:
        """
        Check if a document matches a filter.

        Args:
        ----
            document: Document to check
            filter_dict: Filter criteria

        Returns:
        -------
            bool: True if the document matches the filter, False otherwise

        """
        for key, value in filter_dict.items():
            # Check document ID
            if key == "id" and document.id != value:
                return False

            # Check metadata fields
            if key.startswith("metadata."):
                field = key[len("metadata.") :]

                # Handle nested fields with dot notation
                if "." in field:
                    parts = field.split(".")
                    obj = document.metadata
                    for part in parts[:-1]:
                        if hasattr(obj, part):
                            obj = getattr(obj, part)
                        elif isinstance(obj, dict) and part in obj:
                            obj = obj[part]
                        else:
                            return False

                    last_part = parts[-1]
                    if hasattr(obj, last_part):
                        field_value = getattr(obj, last_part)
                    elif isinstance(obj, dict) and last_part in obj:
                        field_value = obj[last_part]
                    else:
                        return False

                # Handle direct metadata fields
                elif document.metadata is not None:
                    if hasattr(document.metadata, field):
                        field_value = getattr(document.metadata, field)
                    elif hasattr(document.metadata, "custom"):
                        # Handle DocumentMetadata object with custom attribute
                        custom = getattr(document.metadata, "custom", {})
                        if isinstance(custom, dict) and field in custom:
                            field_value = custom[field]
                        else:
                            return False
                    elif isinstance(document.metadata, dict):
                        # Handle metadata as a dictionary
                        if field in document.metadata:
                            field_value = document.metadata[field]
                        elif "custom" in document.metadata and isinstance(
                            document.metadata["custom"], dict
                        ):
                            custom = document.metadata["custom"]
                            if field in custom:
                                field_value = custom[field]
                            else:
                                return False
                        else:
                            return False
                    else:
                        return False
                else:
                    return False

                # Check if the field value matches the filter value
                if isinstance(value, list):
                    if field_value not in value:
                        return False
                elif field_value != value:
                    return False

            # Check content (substring match)
            elif key == "content" and value not in document.content:
                return False

        return True
