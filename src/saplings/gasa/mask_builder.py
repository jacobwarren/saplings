"""
Mask builder module for Graph-Aligned Sparse Attention (GASA).

This module provides the core functionality for building sparse attention masks
based on document dependency graphs.
"""

import hashlib
import json
import logging
import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import scipy.sparse as sp

from saplings.gasa.config import GASAConfig
from saplings.memory.document import Document
from saplings.memory.graph import DependencyGraph, DocumentNode, Node

logger = logging.getLogger(__name__)


class MaskFormat(str, Enum):
    """Format of attention masks."""

    DENSE = "dense"  # Dense matrix (numpy array)
    SPARSE = "sparse"  # Sparse matrix (scipy.sparse)
    BLOCK_SPARSE = "block_sparse"  # Block-sparse format (list of blocks)


class MaskType(str, Enum):
    """Type of attention masks."""

    ATTENTION = "attention"  # Regular attention mask (0 = masked, 1 = attend)
    GLOBAL_ATTENTION = "global_attention"  # Global attention mask (1 = global attention)


class ChunkInfo:
    """Information about a document chunk."""

    def __init__(
        self,
        chunk_id: str,
        document_id: str,
        start_token: int,
        end_token: int,
        node_id: Optional[str] = None,
    ):
        """
        Initialize chunk information.

        Args:
            chunk_id: ID of the chunk
            document_id: ID of the parent document
            start_token: Start token index (inclusive)
            end_token: End token index (exclusive)
            node_id: ID of the corresponding node in the dependency graph
        """
        self.chunk_id = chunk_id
        self.document_id = document_id
        self.start_token = start_token
        self.end_token = end_token
        self.node_id = node_id or chunk_id

    def __repr__(self) -> str:
        return f"ChunkInfo(chunk_id={self.chunk_id}, start={self.start_token}, end={self.end_token})"

    def contains_token(self, token_idx: int) -> bool:
        """
        Check if the chunk contains a token.

        Args:
            token_idx: Token index

        Returns:
            bool: True if the chunk contains the token, False otherwise
        """
        return self.start_token <= token_idx < self.end_token

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "start_token": self.start_token,
            "end_token": self.end_token,
            "node_id": self.node_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkInfo":
        """
        Create from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            ChunkInfo: Chunk information
        """
        return cls(
            chunk_id=data["chunk_id"],
            document_id=data["document_id"],
            start_token=data["start_token"],
            end_token=data["end_token"],
            node_id=data.get("node_id"),
        )


class MaskBuilder:
    """
    Builder for Graph-Aligned Sparse Attention (GASA) masks.

    This class builds sparse attention masks based on document dependency graphs,
    allowing the model to focus on relevant context and reduce computational cost.
    """

    def __init__(
        self,
        graph: DependencyGraph,
        config: Optional[GASAConfig] = None,
        tokenizer: Optional[Any] = None,
    ):
        """
        Initialize the mask builder.

        Args:
            graph: Dependency graph
            config: GASA configuration
            tokenizer: Tokenizer for converting text to tokens
        """
        self.graph = graph
        self.config = config or GASAConfig()
        self.tokenizer = tokenizer

        # Cache for distance calculations
        self._distance_cache: Dict[Tuple[str, str], int] = {}

        # Cache for generated masks
        self._mask_cache: Dict[str, Any] = {}

    def build_mask(
        self,
        documents: List[Document],
        prompt: str,
        format: MaskFormat = MaskFormat.DENSE,
        mask_type: MaskType = MaskType.ATTENTION,
    ) -> Union[np.ndarray, sp.spmatrix, List[Dict[str, Any]]]:
        """
        Build an attention mask based on the document dependency graph.

        Args:
            documents: Documents used in the prompt
            prompt: Prompt text
            format: Output format for the mask
            mask_type: Type of attention mask

        Returns:
            Union[np.ndarray, sp.spmatrix, List[Dict[str, Any]]]: Attention mask
        """
        if not self.config.enabled:
            # Return a dense mask of all ones if GASA is disabled
            if self.tokenizer is None:
                raise ValueError("Tokenizer is required when GASA is disabled")

            tokens = self.tokenizer(prompt, return_tensors="pt")
            # Handle different tokenizer return types
            if hasattr(tokens.input_ids, "shape"):
                # PyTorch tensor
                seq_len = tokens.input_ids.shape[1]
            elif isinstance(tokens.input_ids, list) and len(tokens.input_ids) > 0:
                # List of lists
                if isinstance(tokens.input_ids[0], list):
                    seq_len = len(tokens.input_ids[0])
                else:
                    # Single list
                    seq_len = len(tokens.input_ids)

            if format == MaskFormat.DENSE:
                return np.ones((seq_len, seq_len), dtype=np.int32)
            elif format == MaskFormat.SPARSE:
                return sp.csr_matrix(np.ones((seq_len, seq_len), dtype=np.int32))
            else:
                raise ValueError(f"Unsupported format for dense mask: {format}")

        # Check if we have a cached mask
        cache_key = self._get_cache_key(documents, prompt, format, mask_type)
        if self.config.cache_masks and cache_key in self._mask_cache:
            return self._mask_cache[cache_key]

        # Tokenize the prompt
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required to build masks")

        tokens = self.tokenizer(prompt, return_tensors="pt")
        input_ids = tokens.input_ids[0].tolist()
        seq_len = len(input_ids)

        # Map tokens to chunks
        chunk_infos = self._map_tokens_to_chunks(documents, prompt, input_ids)

        # Build the adjacency matrix for chunks
        chunk_adjacency = self._build_chunk_adjacency(chunk_infos)

        # Expand to token-level mask
        token_mask = self._expand_to_token_mask(chunk_adjacency, chunk_infos, seq_len)

        # Handle global tokens
        token_mask = self._handle_global_tokens(token_mask, input_ids)

        # Convert to the requested format
        result = self._convert_mask_format(token_mask, format, mask_type)

        # Cache the result
        if self.config.cache_masks:
            self._mask_cache[cache_key] = result

        return result

    def _get_cache_key(
        self,
        documents: List[Document],
        prompt: str,
        format: MaskFormat,
        mask_type: MaskType,
    ) -> str:
        """
        Generate a cache key for a mask.

        Args:
            documents: Documents used in the prompt
            prompt: Prompt text
            format: Output format for the mask
            mask_type: Type of attention mask

        Returns:
            str: Cache key
        """
        # Create a string representation of the inputs
        doc_ids = sorted([doc.id for doc in documents])
        doc_ids_str = ",".join(doc_ids)

        # Hash the prompt to avoid long keys
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()

        # Combine all components
        key = f"{doc_ids_str}_{prompt_hash}_{format.value}_{mask_type.value}_{self.config.max_hops}"

        return key

    def _map_tokens_to_chunks(
        self,
        documents: List[Document],
        prompt: str,
        input_ids: List[int],
    ) -> List[ChunkInfo]:
        """
        Map tokens to document chunks.

        Args:
            documents: Documents used in the prompt
            prompt: Prompt text
            input_ids: Token IDs

        Returns:
            List[ChunkInfo]: Chunk information for each token
        """
        # This is a simplified implementation that assumes the prompt is constructed
        # by concatenating document chunks. In a real implementation, you would need
        # to track the mapping between tokens and document chunks more carefully.

        chunk_infos = []
        current_pos = 0

        for doc in documents:
            # Get or create chunks for the document
            if hasattr(doc, "chunks") and doc.chunks:
                chunks = doc.chunks
            else:
                chunks = doc.create_chunks()

            for i, chunk in enumerate(chunks):
                # Find the chunk in the prompt
                chunk_text = chunk.content
                chunk_pos = prompt.find(chunk_text, current_pos)

                if chunk_pos >= 0:
                    # Tokenize the text before the chunk
                    prefix_text = prompt[current_pos:chunk_pos]
                    prefix_tokens = self.tokenizer(prefix_text, add_special_tokens=False).input_ids

                    # Tokenize the chunk
                    chunk_tokens = self.tokenizer(chunk_text, add_special_tokens=False).input_ids

                    # Calculate token positions
                    start_token = current_pos + len(prefix_tokens)
                    end_token = start_token + len(chunk_tokens)

                    # Create chunk info
                    chunk_info = ChunkInfo(
                        chunk_id=chunk.id,
                        document_id=doc.id,
                        start_token=start_token,
                        end_token=end_token,
                        node_id=f"document:{doc.id}",
                    )

                    chunk_infos.append(chunk_info)

                    # Update current position
                    current_pos = chunk_pos + len(chunk_text)

        return chunk_infos

    def _build_chunk_adjacency(self, chunk_infos: List[ChunkInfo]) -> np.ndarray:
        """
        Build an adjacency matrix for chunks based on the dependency graph.

        Args:
            chunk_infos: Chunk information

        Returns:
            np.ndarray: Adjacency matrix (1 = connected, 0 = not connected)
        """
        n_chunks = len(chunk_infos)
        adjacency = np.zeros((n_chunks, n_chunks), dtype=np.int32)

        # Set diagonal to 1 (each chunk is connected to itself)
        np.fill_diagonal(adjacency, 1)

        # Build adjacency based on graph distances
        for i in range(n_chunks):
            for j in range(i + 1, n_chunks):
                chunk_i = chunk_infos[i]
                chunk_j = chunk_infos[j]

                # Check if chunks are from the same document
                if chunk_i.document_id == chunk_j.document_id:
                    # Chunks from the same document are always connected
                    adjacency[i, j] = 1
                    adjacency[j, i] = 1
                else:
                    # Check graph distance
                    distance = self._get_graph_distance(chunk_i.node_id, chunk_j.node_id)

                    if distance <= self.config.max_hops:
                        adjacency[i, j] = 1
                        adjacency[j, i] = 1

        return adjacency

    def _get_graph_distance(self, node_id1: str, node_id2: str) -> int:
        """
        Get the distance between two nodes in the dependency graph.

        Args:
            node_id1: ID of the first node
            node_id2: ID of the second node

        Returns:
            int: Distance between the nodes (number of hops)
        """
        # Check cache
        cache_key = (node_id1, node_id2)
        if cache_key in self._distance_cache:
            return self._distance_cache[cache_key]

        # If nodes are the same, distance is 0
        if node_id1 == node_id2:
            return 0

        try:
            # Find paths between nodes
            paths = self.graph.find_paths(
                source_id=node_id1,
                target_id=node_id2,
                max_hops=self.config.max_hops,
            )

            if not paths:
                # No path found within max_hops
                distance = float("inf")
            else:
                # Get the shortest path
                distance = min(len(path) for path in paths)

            # Cache the result
            self._distance_cache[cache_key] = distance
            self._distance_cache[(node_id2, node_id1)] = distance  # Cache the reverse direction

            return distance

        except (ValueError, KeyError):
            # Nodes not in graph or other error
            return float("inf")

    def _expand_to_token_mask(
        self,
        chunk_adjacency: np.ndarray,
        chunk_infos: List[ChunkInfo],
        seq_len: int,
    ) -> np.ndarray:
        """
        Expand chunk-level adjacency to token-level mask.

        Args:
            chunk_adjacency: Adjacency matrix for chunks
            chunk_infos: Chunk information
            seq_len: Sequence length

        Returns:
            np.ndarray: Token-level attention mask
        """
        # Initialize token mask with zeros
        token_mask = np.zeros((seq_len, seq_len), dtype=np.int32)

        # Set diagonal to 1 (each token attends to itself)
        np.fill_diagonal(token_mask, 1)

        # Map chunk adjacency to token mask
        for i, chunk_i in enumerate(chunk_infos):
            for j, chunk_j in enumerate(chunk_infos):
                if chunk_adjacency[i, j] == 1:
                    # Set attention between all tokens in the connected chunks
                    token_mask[
                        chunk_i.start_token:chunk_i.end_token,
                        chunk_j.start_token:chunk_j.end_token,
                    ] = 1

        return token_mask

    def _handle_global_tokens(self, token_mask: np.ndarray, input_ids: List[int]) -> np.ndarray:
        """
        Handle global tokens that should attend to all other tokens.

        Args:
            token_mask: Token-level attention mask
            input_ids: Token IDs

        Returns:
            np.ndarray: Updated attention mask
        """
        if self.tokenizer is None:
            return token_mask

        # Get token IDs for global tokens
        global_token_ids = []
        for token_text in self.config.global_tokens:
            try:
                token_id = self.tokenizer.convert_tokens_to_ids(token_text)
                if token_id != self.tokenizer.unk_token_id:
                    global_token_ids.append(token_id)
            except (AttributeError, KeyError):
                # Token not in vocabulary
                pass

        # Find positions of global tokens
        global_positions = [i for i, token_id in enumerate(input_ids) if token_id in global_token_ids]

        # Set global attention
        for pos in global_positions:
            # Global token attends to all tokens
            token_mask[pos, :] = 1
            # All tokens attend to global token
            token_mask[:, pos] = 1

        # Add summary token if configured
        if self.config.add_summary_token and self.config.summary_token:
            # In a real implementation, you would add the summary token to the input
            # and update the mask accordingly. For now, we'll just use the first token
            # as a proxy for the summary token.
            summary_pos = 0

            # Connect unconnected token pairs through the summary token
            for i in range(token_mask.shape[0]):
                for j in range(token_mask.shape[1]):
                    if token_mask[i, j] == 0:
                        # Connect through summary token
                        token_mask[i, summary_pos] = 1
                        token_mask[summary_pos, j] = 1

        return token_mask

    def _convert_mask_format(
        self,
        mask: np.ndarray,
        format: MaskFormat,
        mask_type: MaskType,
    ) -> Union[np.ndarray, sp.spmatrix, List[Dict[str, Any]]]:
        """
        Convert mask to the requested format.

        Args:
            mask: Dense attention mask
            format: Output format
            mask_type: Type of attention mask

        Returns:
            Union[np.ndarray, sp.spmatrix, List[Dict[str, Any]]]: Converted mask
        """
        if mask_type == MaskType.GLOBAL_ATTENTION:
            # For global attention, 1 means global attention, 0 means local attention
            # We need to invert the mask: global_attention = 1 - attention
            mask = 1 - mask

        if format == MaskFormat.DENSE:
            return mask

        elif format == MaskFormat.SPARSE:
            return sp.csr_matrix(mask)

        elif format == MaskFormat.BLOCK_SPARSE:
            # Convert to block-sparse format
            blocks = []

            # Find contiguous blocks of 1s
            for i in range(0, mask.shape[0], self.config.block_size):
                for j in range(0, mask.shape[1], self.config.block_size):
                    # Get block
                    block_i_end = min(i + self.config.block_size, mask.shape[0])
                    block_j_end = min(j + self.config.block_size, mask.shape[1])
                    block = mask[i:block_i_end, j:block_j_end]

                    # Check if block contains any 1s
                    if np.any(block):
                        blocks.append({
                            "row": i,
                            "col": j,
                            "size_row": block.shape[0],
                            "size_col": block.shape[1],
                            "block": block.tolist(),
                        })

            return blocks

        else:
            raise ValueError(f"Unsupported format: {format}")

    def save_mask(
        self,
        mask: Union[np.ndarray, sp.spmatrix, List[Dict[str, Any]]],
        file_path: str,
        format: MaskFormat,
        mask_type: MaskType,
    ) -> None:
        """
        Save a mask to disk.

        Args:
            mask: Attention mask
            file_path: Path to save the mask
            format: Format of the mask
            mask_type: Type of attention mask
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        metadata = {
            "format": format.value,
            "mask_type": mask_type.value,
            "max_hops": self.config.max_hops,
        }

        if format == MaskFormat.DENSE:
            # Save as .npz file
            np.savez_compressed(
                path,
                mask=mask,
                metadata=json.dumps(metadata),
            )

        elif format == MaskFormat.SPARSE:
            # Save as .npz file with sparse matrix
            sp.save_npz(path, mask)

            # Save metadata separately
            with open(f"{path}.metadata.json", "w") as f:
                json.dump(metadata, f)

        elif format == MaskFormat.BLOCK_SPARSE:
            # Save as JSON file
            with open(path, "w") as f:
                json.dump({
                    "blocks": mask,
                    "metadata": metadata,
                }, f)

        else:
            raise ValueError(f"Unsupported format: {format}")

    def load_mask(
        self,
        file_path: str,
    ) -> Tuple[Union[np.ndarray, sp.spmatrix, List[Dict[str, Any]]], MaskFormat, MaskType]:
        """
        Load a mask from disk.

        Args:
            file_path: Path to load the mask from

        Returns:
            Tuple[Union[np.ndarray, sp.spmatrix, List[Dict[str, Any]]], MaskFormat, MaskType]:
                Mask, format, and type
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Mask file not found: {file_path}")

        if path.suffix == ".npz":
            # Try loading as dense .npz file
            try:
                data = np.load(path, allow_pickle=True)
                if "mask" in data:
                    # Dense format
                    mask = data["mask"]
                    metadata = json.loads(str(data["metadata"]))
                    format = MaskFormat(metadata["format"])
                    mask_type = MaskType(metadata["mask_type"])
                    return mask, format, mask_type
                else:
                    # Sparse format
                    mask = sp.load_npz(path)

                    # Load metadata
                    metadata_path = f"{path}.metadata.json"
                    if os.path.exists(metadata_path):
                        with open(metadata_path, "r") as f:
                            metadata = json.load(f)
                        format = MaskFormat(metadata["format"])
                        mask_type = MaskType(metadata["mask_type"])
                    else:
                        # Default to sparse format and attention mask type
                        format = MaskFormat.SPARSE
                        mask_type = MaskType.ATTENTION

                    return mask, format, mask_type

            except Exception as e:
                raise ValueError(f"Failed to load mask from {file_path}: {e}")

        elif path.suffix == ".json":
            # Load as JSON file (block-sparse format)
            try:
                with open(path, "r") as f:
                    data = json.load(f)

                blocks = data["blocks"]
                metadata = data["metadata"]
                format = MaskFormat(metadata["format"])
                mask_type = MaskType(metadata["mask_type"])

                return blocks, format, mask_type

            except Exception as e:
                raise ValueError(f"Failed to load mask from {file_path}: {e}")

        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    def clear_cache(self) -> None:
        """Clear the mask cache."""
        self._distance_cache.clear()
        self._mask_cache.clear()
