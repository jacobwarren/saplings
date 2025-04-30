"""
TF-IDF retriever module for Saplings.

This module provides the TF-IDF retriever for initial document filtering.
"""

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from saplings.memory.document import Document
from saplings.memory.memory_store import MemoryStore
from saplings.retrieval.config import RetrievalConfig, TFIDFConfig

logger = logging.getLogger(__name__)


class TFIDFRetriever:
    """
    TF-IDF retriever for initial document filtering.
    
    This class uses TF-IDF (Term Frequency-Inverse Document Frequency) to
    perform initial filtering of documents based on lexical similarity.
    """
    
    def __init__(
        self,
        memory_store: MemoryStore,
        config: Optional[Union[RetrievalConfig, TFIDFConfig]] = None,
    ):
        """
        Initialize the TF-IDF retriever.
        
        Args:
            memory_store: Memory store containing the documents
            config: Retrieval or TF-IDF configuration
        """
        self.memory_store = memory_store
        
        # Extract TF-IDF config from RetrievalConfig if needed
        if config is None:
            self.config = TFIDFConfig()
        elif isinstance(config, RetrievalConfig):
            self.config = config.tfidf
        else:
            self.config = config
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            min_df=self.config.min_df,
            max_df=self.config.max_df,
            max_features=self.config.max_features,
            ngram_range=self.config.ngram_range,
            use_idf=self.config.use_idf,
            norm=self.config.norm,
            analyzer=self.config.analyzer,
            stop_words=self.config.stop_words,
        )
        
        # Initialize document mapping
        self.doc_id_to_index: Dict[str, int] = {}
        self.index_to_doc_id: Dict[int, str] = {}
        
        # Initialize TF-IDF matrix
        self.tfidf_matrix = None
        
        # Flag to track if the index is built
        self.is_built = False
    
    def build_index(self, documents: Optional[List[Document]] = None) -> None:
        """
        Build the TF-IDF index.
        
        Args:
            documents: Documents to index (if None, all documents in the memory store are used)
        """
        # Get documents if not provided
        if documents is None:
            documents = self.memory_store.vector_store.list()
        
        if not documents:
            logger.warning("No documents to index")
            return
        
        # Extract document contents and IDs
        contents = []
        doc_ids = []
        
        for i, doc in enumerate(documents):
            contents.append(doc.content)
            doc_ids.append(doc.id)
            self.doc_id_to_index[doc.id] = i
            self.index_to_doc_id[i] = doc.id
        
        # Build TF-IDF matrix
        self.tfidf_matrix = self.vectorizer.fit_transform(contents)
        
        logger.info(f"Built TF-IDF index with {len(documents)} documents")
        self.is_built = True
    
    def update_index(self, documents: List[Document]) -> None:
        """
        Update the TF-IDF index with new documents.
        
        Args:
            documents: Documents to add to the index
        """
        if not self.is_built:
            self.build_index(documents)
            return
        
        if not documents:
            return
        
        # Extract document contents and IDs
        contents = []
        doc_ids = []
        
        for doc in documents:
            contents.append(doc.content)
            doc_ids.append(doc.id)
        
        # Transform new documents
        new_tfidf = self.vectorizer.transform(contents)
        
        # Update document mapping
        offset = self.tfidf_matrix.shape[0]
        for i, doc_id in enumerate(doc_ids):
            self.doc_id_to_index[doc_id] = offset + i
            self.index_to_doc_id[offset + i] = doc_id
        
        # Concatenate with existing matrix
        if self.tfidf_matrix is not None:
            self.tfidf_matrix = np.vstack([self.tfidf_matrix, new_tfidf])
        else:
            self.tfidf_matrix = new_tfidf
        
        logger.info(f"Updated TF-IDF index with {len(documents)} new documents")
    
    def remove_from_index(self, doc_ids: List[str]) -> None:
        """
        Remove documents from the TF-IDF index.
        
        Args:
            doc_ids: IDs of documents to remove
        """
        if not self.is_built or self.tfidf_matrix is None:
            logger.warning("TF-IDF index not built yet")
            return
        
        if not doc_ids:
            return
        
        # Get indices to keep
        indices_to_remove = set()
        for doc_id in doc_ids:
            if doc_id in self.doc_id_to_index:
                indices_to_remove.add(self.doc_id_to_index[doc_id])
        
        if not indices_to_remove:
            return
        
        # Create a mask for rows to keep
        keep_mask = np.ones(self.tfidf_matrix.shape[0], dtype=bool)
        for idx in indices_to_remove:
            keep_mask[idx] = False
        
        # Filter the TF-IDF matrix
        self.tfidf_matrix = self.tfidf_matrix[keep_mask]
        
        # Update document mapping
        new_doc_id_to_index = {}
        new_index_to_doc_id = {}
        
        new_idx = 0
        for old_idx, doc_id in self.index_to_doc_id.items():
            if old_idx not in indices_to_remove:
                new_doc_id_to_index[doc_id] = new_idx
                new_index_to_doc_id[new_idx] = doc_id
                new_idx += 1
        
        self.doc_id_to_index = new_doc_id_to_index
        self.index_to_doc_id = new_index_to_doc_id
        
        logger.info(f"Removed {len(indices_to_remove)} documents from TF-IDF index")
    
    def retrieve(
        self, query: str, k: Optional[int] = None, filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve documents similar to the query using TF-IDF.
        
        Args:
            query: Query string
            k: Number of documents to retrieve (if None, uses config.initial_k)
            filter_dict: Optional filter criteria
            
        Returns:
            List[Tuple[Document, float]]: List of (document, similarity_score) tuples
        """
        if not self.is_built or self.tfidf_matrix is None:
            logger.warning("TF-IDF index not built yet, building now...")
            self.build_index()
            
            if not self.is_built or self.tfidf_matrix is None:
                logger.error("Failed to build TF-IDF index")
                return []
        
        # Use default k if not provided
        if k is None:
            k = self.config.initial_k
        
        # Transform query to TF-IDF space
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarity scores
        similarity_scores = (self.tfidf_matrix @ query_vector.T).toarray().flatten()
        
        # Get top-k indices
        if filter_dict:
            # Apply filter
            filtered_indices = []
            for idx, doc_id in self.index_to_doc_id.items():
                doc = self.memory_store.get_document(doc_id)
                if doc and self._matches_filter(doc, filter_dict):
                    filtered_indices.append(idx)
            
            if not filtered_indices:
                return []
            
            # Sort filtered indices by score
            filtered_indices = sorted(
                filtered_indices, key=lambda idx: similarity_scores[idx], reverse=True
            )
            top_indices = filtered_indices[:k]
        else:
            # Get top-k indices without filtering
            top_indices = np.argsort(similarity_scores)[::-1][:k]
        
        # Get documents and scores
        results = []
        for idx in top_indices:
            doc_id = self.index_to_doc_id.get(idx)
            if doc_id:
                doc = self.memory_store.get_document(doc_id)
                if doc:
                    score = float(similarity_scores[idx])
                    results.append((doc, score))
        
        return results
    
    def save(self, directory: str) -> None:
        """
        Save the TF-IDF retriever to disk.
        
        Args:
            directory: Directory to save to
        """
        directory_path = Path(directory)
        directory_path.mkdir(parents=True, exist_ok=True)
        
        # Save vectorizer
        with open(directory_path / "vectorizer.pkl", "wb") as f:
            pickle.dump(self.vectorizer, f)
        
        # Save TF-IDF matrix
        if self.tfidf_matrix is not None:
            with open(directory_path / "tfidf_matrix.npz", "wb") as f:
                np.savez_compressed(f, data=self.tfidf_matrix.data, indices=self.tfidf_matrix.indices,
                                   indptr=self.tfidf_matrix.indptr, shape=self.tfidf_matrix.shape)
        
        # Save document mapping
        with open(directory_path / "doc_mapping.json", "w") as f:
            json.dump({
                "doc_id_to_index": {k: v for k, v in self.doc_id_to_index.items()},
                "index_to_doc_id": {str(k): v for k, v in self.index_to_doc_id.items()},
            }, f)
        
        # Save config
        with open(directory_path / "config.json", "w") as f:
            json.dump(self.config.model_dump(), f)
        
        logger.info(f"Saved TF-IDF retriever to {directory}")
    
    def load(self, directory: str) -> None:
        """
        Load the TF-IDF retriever from disk.
        
        Args:
            directory: Directory to load from
        """
        directory_path = Path(directory)
        
        # Load vectorizer
        vectorizer_path = directory_path / "vectorizer.pkl"
        if vectorizer_path.exists():
            with open(vectorizer_path, "rb") as f:
                self.vectorizer = pickle.load(f)
        
        # Load TF-IDF matrix
        tfidf_matrix_path = directory_path / "tfidf_matrix.npz"
        if tfidf_matrix_path.exists():
            with open(tfidf_matrix_path, "rb") as f:
                loader = np.load(f)
                from scipy.sparse import csr_matrix
                self.tfidf_matrix = csr_matrix(
                    (loader["data"], loader["indices"], loader["indptr"]), shape=tuple(loader["shape"])
                )
        
        # Load document mapping
        doc_mapping_path = directory_path / "doc_mapping.json"
        if doc_mapping_path.exists():
            with open(doc_mapping_path, "r") as f:
                mapping = json.load(f)
                self.doc_id_to_index = {k: int(v) for k, v in mapping["doc_id_to_index"].items()}
                self.index_to_doc_id = {int(k): v for k, v in mapping["index_to_doc_id"].items()}
        
        # Load config
        config_path = directory_path / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                config_data = json.load(f)
                self.config = TFIDFConfig(**config_data)
        
        self.is_built = self.tfidf_matrix is not None
        logger.info(f"Loaded TF-IDF retriever from {directory}")
    
    def _matches_filter(self, document: Document, filter_dict: Dict[str, Any]) -> bool:
        """
        Check if a document matches a filter.
        
        Args:
            document: Document to check
            filter_dict: Filter criteria
            
        Returns:
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
                elif hasattr(document.metadata, field):
                    field_value = getattr(document.metadata, field)
                elif field in document.metadata.custom:
                    field_value = document.metadata.custom[field]
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
