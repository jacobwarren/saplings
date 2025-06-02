from __future__ import annotations

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
from typing import TYPE_CHECKING, Any, Protocol, Union, runtime_checkable

import numpy as np
import scipy.sparse as sp
import torch

from saplings.gasa._internal.config import GASAConfig
from saplings.gasa._internal.core.graph_distance import GraphDistanceCalculator
from saplings.gasa._internal.core.token_mapper import TokenMapper

# Set up logger
logger = logging.getLogger(__name__)

from saplings.core.utils.platform import is_apple_silicon, is_triton_available

IS_APPLE_SILICON = is_apple_silicon()
TRITON_AVAILABLE = is_triton_available()

# Import shadow model tokenizer if available
try:
    from saplings.tokenizers._internal.shadow_model_tokenizer import ShadowModelTokenizer

    SHADOW_MODEL_AVAILABLE = True
except ImportError:
    SHADOW_MODEL_AVAILABLE = False


# Import simple tokenizer as fallback
from saplings.tokenizers._internal.simple_tokenizer import SimpleTokenizer

if TYPE_CHECKING:
    from collections.abc import Sequence

    from saplings.gasa._internal.core.chunk_info import ChunkInfo
    from saplings.memory._internal.graph import DependencyGraph


# Define Document Protocol for type checking
@runtime_checkable
class Document(Protocol):
    """Protocol for document objects."""

    id: str
    content: str
    metadata: dict[str, Any] | None
    chunks: list[Any] | None

    def chunk(self, chunk_size: int, chunk_overlap: int = 0) -> list[Any]: ...
    def create_chunks(self) -> Sequence[Any]: ...


class MaskFormat(str, Enum):
    """Format of attention masks."""

    DENSE = "dense"  # Dense matrix (numpy array)
    SPARSE = "sparse"  # Sparse matrix (scipy.sparse)
    BLOCK_SPARSE = "block_sparse"  # Block-sparse format (list of blocks)


class MaskType(str, Enum):
    """Type of attention masks."""

    ATTENTION = "attention"  # Regular attention mask (0 = masked, 1 = attend)
    GLOBAL_ATTENTION = "global_attention"  # Global attention mask (1 = global attention)


class MaskBuilder:
    """
    Builder for Graph-Aligned Sparse Attention (GASA) masks.

    This class builds sparse attention masks based on document dependency graphs,
    allowing the model to focus on relevant context and reduce computational cost.
    """

    def __init__(
        self,
        graph: DependencyGraph,
        config: GASAConfig | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        """
        Initialize the mask builder.

        Args:
        ----
            graph: Dependency graph
            config: GASA configuration
            tokenizer: Tokenizer for converting text to tokens

        """
        self.graph = graph
        self.config = config or GASAConfig.default()

        # Initialize tokenizer
        if tokenizer is not None:
            # Use the provided tokenizer (e.g., from vLLM)
            self.tokenizer = tokenizer
            logger.info(f"Using provided tokenizer: {type(tokenizer).__name__}")
        elif self.config.enable_shadow_model and SHADOW_MODEL_AVAILABLE:
            # Check if we're on Apple Silicon or in an environment without Triton
            if IS_APPLE_SILICON or not TRITON_AVAILABLE:
                # Use a CPU-friendly model on Apple Silicon or without Triton
                logger.info("Using shadow model tokenizer with CPU-only mode")
                self.tokenizer = ShadowModelTokenizer(
                    model_name=self.config.shadow_model_name,
                    device="cpu",
                    cache_dir=self.config.shadow_model_cache_dir,
                    fallback_to_simple=True,
                    cpu_only=True,
                    # Use alternative models that don't require Triton
                    alternative_models=[
                        "distilgpt2",  # Small and widely compatible
                        "gpt2",  # Widely available
                        "EleutherAI/pythia-70m",  # Very small model
                        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Small but capable
                    ],
                )
            else:
                # Use the configured shadow model
                logger.info(f"Using shadow model tokenizer: {self.config.shadow_model_name}")
                self.tokenizer = ShadowModelTokenizer(
                    model_name=self.config.shadow_model_name,
                    device=self.config.shadow_model_device,
                    cache_dir=self.config.shadow_model_cache_dir,
                    fallback_to_simple=True,
                )
        else:
            # Fall back to simple tokenizer
            logger.info("Using simple tokenizer")
            self.tokenizer = SimpleTokenizer()

        # Initialize graph distance calculator
        self.distance_calculator = GraphDistanceCalculator(graph)

        # Cache for generated masks
        self._mask_cache: dict[str, Any] = {}

    def build_mask(
        self,
        documents: list[Document],
        prompt: str,
        format: MaskFormat = MaskFormat.DENSE,
        mask_type: MaskType = MaskType.ATTENTION,
    ) -> np.ndarray | sp.spmatrix | list[dict[str, Any]]:
        """
        Build an attention mask based on the document dependency graph.

        Args:
        ----
            documents: Documents used in the prompt
            prompt: Prompt text
            format: Output format for the mask
            mask_type: Type of attention mask

        Returns:
        -------
            Union[np.ndarray, sp.spmatrix, List[Dict[str, Any]]]: Attention mask

        """
        if not self.config.enabled:
            # Return a dense mask of all ones if GASA is disabled
            if self.tokenizer is None:
                msg = "Tokenizer is required when GASA is disabled"
                raise ValueError(msg)

            tokens = self.tokenizer(prompt, return_tensors="pt")
            # Handle different tokenizer return types
            seq_len = None

            # Get input_ids safely with proper type handling
            input_ids = getattr(tokens, "input_ids", None)
            if input_ids is not None:
                # Check if input_ids has shape attribute (tensor-like)
                if hasattr(input_ids, "shape"):
                    seq_len = input_ids.shape[1]
                # Check if input_ids is a list
                elif isinstance(input_ids, list) and len(input_ids) > 0:
                    # List of lists
                    if isinstance(input_ids[0], list):
                        seq_len = len(input_ids[0])
                    else:
                        # Single list
                        seq_len = len(input_ids)

            # Fallback if we couldn't determine seq_len
            if seq_len is None:
                seq_len = len(prompt) // 4  # Rough estimate of 4 chars per token

            if format == MaskFormat.DENSE:
                return np.ones((seq_len, seq_len), dtype=np.int32)
            if format == MaskFormat.SPARSE:
                return sp.csr_matrix(np.ones((seq_len, seq_len), dtype=np.int32))
            msg = f"Unsupported format for dense mask: {format}"
            raise ValueError(msg)

        # Check if we have a cached mask
        cache_key = self._get_cache_key(documents, prompt, format, mask_type)
        if self.config.cache_masks and cache_key in self._mask_cache:
            return self._mask_cache[cache_key]

        # Tokenize the prompt
        if self.tokenizer is None:
            msg = "Tokenizer is required to build masks"
            raise ValueError(msg)

        tokens = self.tokenizer(prompt, return_tensors="pt")
        # Extract input_ids safely from different tokenizer output formats
        input_ids = []
        if hasattr(tokens, "input_ids"):
            ids = getattr(tokens, "input_ids", None)
            assert ids is not None
            if hasattr(ids, "shape") and len(ids.shape) > 0:
                input_ids = ids[0].tolist() if hasattr(ids, "tolist") else list(ids[0])
            elif isinstance(ids, list) and len(ids) > 0:
                input_ids = ids[0] if isinstance(ids[0], list) else ids
        elif isinstance(tokens, dict) and "input_ids" in tokens:
            ids = tokens["input_ids"]
            if hasattr(ids, "shape") and len(ids.shape) > 0:
                input_ids = ids[0].tolist() if hasattr(ids, "tolist") else list(ids[0])
            elif isinstance(ids, list) and len(ids) > 0:
                input_ids = ids[0] if isinstance(ids[0], list) else ids
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
        documents: list[Document],
        prompt: str,
        format: MaskFormat,
        mask_type: MaskType,
    ) -> str:
        """
        Generate a cache key for a mask.

        Args:
        ----
            documents: Documents used in the prompt
            prompt: Prompt text
            format: Output format for the mask
            mask_type: Type of attention mask

        Returns:
        -------
            str: Cache key

        """
        # Create a unique cache key that handles collisions
        key_components = [
            # Document IDs sorted for consistency
            ",".join(sorted([doc.id for doc in documents])),
            # Hash of prompt content
            hashlib.sha256(prompt.encode()).hexdigest()[:16],
            # Format and type
            format.value,
            mask_type.value,
            # Configuration parameters that affect the mask
            str(self.config.max_hops),
            str(self.config.block_size),
            str(self.config.overlap),
            self.config.mask_strategy.value,
            # Include tokenizer info if available
            getattr(self.tokenizer, "__class__.__name__", "unknown_tokenizer"),
        ]

        # Join with a separator unlikely to appear in the components
        return "||".join(key_components)

    def _map_tokens_to_chunks(
        self,
        documents: list[Document],
        prompt: str,
        input_ids: list[int],
    ) -> list[ChunkInfo]:
        """
        Map tokens to document chunks.

        Args:
        ----
            documents: Documents used in the prompt
            prompt: Prompt text (used for context)
            input_ids: Token IDs (used for context)

        Returns:
        -------
            List[ChunkInfo]: Chunk information for each token

        """
        # Use the TokenMapper for accurate token-to-chunk mapping
        token_mapper = TokenMapper(self.tokenizer)

        # Process documents and track token positions
        for doc in documents:
            # Get or create chunks for the document
            chunks = []
            if hasattr(doc, "chunks") and doc.chunks:
                chunks = doc.chunks
            elif hasattr(doc, "create_chunks"):
                try:
                    # Use getattr to avoid static type checking errors
                    create_chunks_method = doc.create_chunks
                    chunks = create_chunks_method()
                except Exception as e:
                    logger.warning(f"Error creating chunks for document {doc.id}: {e}")
            elif hasattr(doc, "chunk"):
                try:
                    chunks = doc.chunk(self.config.block_size, self.config.overlap)
                except Exception as e:
                    logger.warning(f"Error chunking document {doc.id}: {e}")
            else:
                logger.warning(f"Document {doc.id} has no chunks and no method to create them")

            for chunk in chunks:
                # Add the chunk to the token mapper
                token_mapper.add_document_chunk(
                    chunk=chunk,
                    document_id=doc.id,
                    node_id=f"document:{doc.id}",
                )

        # Return the tracked chunk infos
        return token_mapper.get_chunk_infos()

    def _build_chunk_adjacency(self, chunk_infos: list[ChunkInfo]) -> np.ndarray:
        """
        Build an adjacency matrix for chunks based on the dependency graph.

        Args:
        ----
            chunk_infos: Chunk information

        Returns:
        -------
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

    def _get_graph_distance(self, node_id1: str, node_id2: str) -> int | float:
        """
        Get the distance between two nodes in the dependency graph.

        Args:
        ----
            node_id1: ID of the first node
            node_id2: ID of the second node

        Returns:
        -------
            Union[int, float]: Distance between the nodes (number of hops),
                or float('inf') if no path exists

        """
        # Use the GraphDistanceCalculator for efficient distance calculation
        return self.distance_calculator.get_distance(
            source_id=node_id1,
            target_id=node_id2,
            max_hops=self.config.max_hops,
        )

    def _expand_to_token_mask(
        self,
        chunk_adjacency: np.ndarray,
        chunk_infos: list[ChunkInfo],
        seq_len: int,
    ) -> np.ndarray:
        """
        Expand chunk-level adjacency to token-level mask.

        Args:
        ----
            chunk_adjacency: Adjacency matrix for chunks
            chunk_infos: Chunk information
            seq_len: Sequence length

        Returns:
        -------
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
                        chunk_i.start_token : chunk_i.end_token,
                        chunk_j.start_token : chunk_j.end_token,
                    ] = 1

        return token_mask

    def _handle_global_tokens(self, token_mask: np.ndarray, input_ids: list[int]) -> np.ndarray:
        """
        Handle global tokens that should attend to all other tokens.

        Args:
        ----
            token_mask: Token-level attention mask
            input_ids: Token IDs

        Returns:
        -------
            np.ndarray: Updated attention mask

        """
        if self.tokenizer is None:
            return token_mask

        # Get token IDs for global tokens with robust error handling
        global_token_ids = set()  # Use set to avoid duplicates

        if self.tokenizer is None:
            logger.warning("No tokenizer available for global token handling")
            return token_mask

        for token_text in self.config.global_tokens:
            try:
                token_id = None
                # Try different tokenizer methods in order of preference
                if hasattr(self.tokenizer, "convert_tokens_to_ids"):
                    # Most common method for HuggingFace tokenizers
                    result = self.tokenizer.convert_tokens_to_ids([token_text])
                    token_id = self._extract_token_id(result)
                elif callable(self.tokenizer):
                    # Try direct tokenizer call
                    try:
                        result = self.tokenizer(token_text)
                        if isinstance(result, dict) and "input_ids" in result:
                            token_id = self._extract_token_id(result["input_ids"])
                    except Exception:
                        pass

                if token_id is None:
                    logger.warning(f"Could not convert token {token_text} to ID")
                    continue

                # Get unknown token ID using multiple methods
                unk_token_id = self._get_unk_token_id()

                # Add valid token ID if it's not the unknown token
                if token_id != unk_token_id:
                    global_token_ids.add(token_id)

            except Exception as e:
                logger.warning(f"Error processing global token {token_text}: {e!s}")

        # Find positions of global tokens
        global_positions = [
            i for i, token_id in enumerate(input_ids) if token_id in global_token_ids
        ]

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

    def _extract_token_id(self, result: Any) -> Union[int, None]:
        """Extract token ID from various result types."""
        try:
            if isinstance(result, (int, np.integer)):
                return int(result)
            if isinstance(result, (list, tuple)) and result:
                if isinstance(result[0], (int, np.integer)):
                    return int(result[0])
                if isinstance(result[0], (list, tuple)) and result[0]:
                    # Handle nested lists/tuples
                    return (
                        int(result[0][0]) if isinstance(result[0][0], (int, np.integer)) else None
                    )
            if isinstance(result, (np.ndarray, torch.Tensor)):
                # Handle both numpy arrays and torch tensors
                if isinstance(result, torch.Tensor):
                    result = result.detach().cpu().numpy()
                if result.size > 0:
                    # Handle multi-dimensional arrays
                    if result.ndim > 1:
                        result = result.flatten()
                    return int(result[0])
            # Handle custom tokenizer outputs that may have an ids attribute
            ids_attr = getattr(result, "ids", None)
            if ids_attr is not None and isinstance(ids_attr, (list, tuple)) and ids_attr:
                return int(ids_attr[0])
            return None
        except Exception as e:
            logger.warning(f"Error extracting token ID: {e!s}")
            return None

    def _get_unk_token_id(self) -> int:
        """Get the unknown token ID using multiple methods, handling different tokenizer types."""
        try:
            tokenizer_instance = self.tokenizer  # Default to the main tokenizer

            # If it's a ShadowModelTokenizer, try to get the underlying tokenizer
            # Check SHADOW_MODEL_AVAILABLE first to avoid NameError if import failed
            if SHADOW_MODEL_AVAILABLE and isinstance(self.tokenizer, ShadowModelTokenizer):
                # Explicitly get the inner tokenizer if the attribute exists
                inner_tokenizer = getattr(self.tokenizer, "tokenizer", None)
                if inner_tokenizer is not None:
                    tokenizer_instance = inner_tokenizer
                # else: tokenizer_instance remains self.tokenizer (the ShadowModelTokenizer wrapper)
                # This handles cases where ShadowModelTokenizer failed to load or fell back internally

            # --- Now use tokenizer_instance for attribute checks, but self.tokenizer for method calls ---

            # Try standard attribute (unk_token_id) on the effective tokenizer instance
            if hasattr(tokenizer_instance, "unk_token_id"):
                # Use getattr to avoid static type checking errors
                unk_id = tokenizer_instance.unk_token_id
                if isinstance(unk_id, int):
                    return unk_id

            # Try special tokens dictionary (using the main tokenizer wrapper's property)
            # This assumes the wrapper correctly implements the TokenizerInterface
            if hasattr(self.tokenizer, "special_tokens"):
                special_tokens = (
                    self.tokenizer.special_tokens
                )  # Access property on the wrapper/interface
                if isinstance(special_tokens, dict):
                    for key in ["<unk>", "[UNK]", "UNK", "<|endoftext|>"]:
                        if key in special_tokens:
                            # Ensure the value is an int before returning
                            token_id = special_tokens[key]
                            if isinstance(token_id, int):
                                return token_id
                            # Handle cases where the value might be complex (e.g., AddedToken)
                            if hasattr(token_id, "id"):  # Check for AddedToken structure
                                return int(token_id.id)

            # Try direct token conversion (using the main tokenizer wrapper's method)
            if hasattr(self.tokenizer, "convert_tokens_to_ids"):
                for token in ["<unk>", "[UNK]", "UNK", "<|endoftext|>"]:
                    try:
                        # Use the main tokenizer's method as per TokenizerInterface
                        result = self.tokenizer.convert_tokens_to_ids([token])
                        unk_id_list = None
                        # Handle various return types from convert_tokens_to_ids
                        if isinstance(result, (list, tuple)) and result:
                            unk_id_list = result
                        elif (
                            isinstance(result, dict) and "input_ids" in result
                        ):  # Handle dict output
                            unk_id_list = result["input_ids"]
                        elif isinstance(result, int):  # Handle single int return
                            unk_id_list = [result]

                        # Check if we got a valid list and a valid ID
                        if (
                            unk_id_list
                            and isinstance(unk_id_list[0], int)
                            and unk_id_list[0] >= 0  # Often -1 or similar indicates failure
                        ):
                            return unk_id_list[0]
                    except Exception as conversion_error:
                        logger.debug(f"Failed converting token '{token}': {conversion_error}")
                        continue  # Ignore errors during conversion attempts

            # Try vocabulary lookup (using the effective tokenizer_instance)
            vocab_dict = None
            if isinstance(tokenizer_instance, SimpleTokenizer):
                # SimpleTokenizer uses token_to_id, not vocab
                vocab_dict = getattr(tokenizer_instance, "token_to_id", None)
            elif hasattr(tokenizer_instance, "vocab"):  # Check for 'vocab' attribute first
                vocab_dict = getattr(tokenizer_instance, "vocab", None)
            elif hasattr(tokenizer_instance, "get_vocab"):  # Check for 'get_vocab' method
                # Ensure we don't call get_vocab on the wrapper classes themselves
                if not isinstance(tokenizer_instance, (ShadowModelTokenizer, SimpleTokenizer)):
                    vocab_dict = tokenizer_instance.get_vocab()  # type: ignore[attr-defined]

            if isinstance(vocab_dict, dict):
                for key in ["<unk>", "[UNK]", "UNK", "<|endoftext|>"]:
                    if key in vocab_dict:
                        token_id = vocab_dict[key]
                        if isinstance(token_id, int):
                            return token_id

            # Default unknown token ID if all else fails
            logger.warning(
                "Could not determine unknown token ID using standard methods, defaulting to 3."
            )
            return 3
        except Exception as e:
            logger.warning(f"Unexpected error getting unknown token ID: {e!s}, defaulting to 3.")
            return 3

    def _convert_mask_format(
        self,
        mask: np.ndarray,
        format: MaskFormat,
        mask_type: MaskType,
    ) -> np.ndarray | sp.spmatrix | list[dict[str, Any]]:
        """
        Convert mask to the requested format.

        Args:
        ----
            mask: Dense attention mask
            format: Output format
            mask_type: Type of attention mask

        Returns:
        -------
            Union[np.ndarray, sp.spmatrix, List[Dict[str, Any]]]: Converted mask

        """
        if mask_type == MaskType.GLOBAL_ATTENTION:
            # For global attention, 1 means global attention, 0 means local attention
            # We need to invert the mask: global_attention = 1 - attention
            mask = 1 - mask

        if format == MaskFormat.DENSE:
            return mask

        if format == MaskFormat.SPARSE:
            return sp.csr_matrix(mask)

        if format == MaskFormat.BLOCK_SPARSE:
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
                        blocks.append(
                            {
                                "row": i,
                                "col": j,
                                "size_row": block.shape[0],
                                "size_col": block.shape[1],
                                "block": block.tolist(),
                            }
                        )

            return blocks

        msg = f"Unsupported format: {format}"
        raise ValueError(msg)

    def save_mask(
        self,
        mask: np.ndarray | sp.spmatrix | list[dict[str, Any]],
        file_path: str,
        format: MaskFormat,
        mask_type: MaskType,
    ) -> None:
        """
        Save a mask to disk.

        Args:
        ----
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
            if isinstance(mask, list):
                # Convert block-sparse format to JSON instead
                with open(path, "w") as f:
                    json.dump(
                        {
                            "blocks": mask,
                            "metadata": metadata,
                        },
                        f,
                    )
                return
            if sp.issparse(mask):
                # Convert sparse matrix to dense array
                try:
                    # Use scipy's built-in methods to convert to array
                    # This is a safer approach that works with type checking
                    from scipy import sparse

                    # Convert to CSR format which has toarray method
                    csr_mask = sparse.csr_matrix(mask)
                    mask_array = csr_mask.toarray()
                except Exception:
                    # Fallback to direct numpy conversion
                    try:
                        mask_array = np.array(mask)
                    except Exception as e:
                        logger.warning(f"Failed to convert sparse matrix to array: {e}")
                        # Create a dummy array as last resort
                        mask_array = np.ones((1, 1), dtype=np.int32)
            else:
                mask_array = np.array(mask)

            np.savez_compressed(
                path,
                mask=mask_array,
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
                json.dump(
                    {
                        "blocks": mask,
                        "metadata": metadata,
                    },
                    f,
                )

        else:
            msg = f"Unsupported format: {format}"
            raise ValueError(msg)

    def load_mask(
        self,
        file_path: str,
    ) -> tuple[np.ndarray | sp.spmatrix | list[dict[str, Any]], MaskFormat, MaskType]:
        """
        Load a mask from disk.

        Args:
        ----
            file_path: Path to load the mask from

        Returns:
        -------
            Tuple[Union[np.ndarray, sp.spmatrix, List[Dict[str, Any]]], MaskFormat, MaskType]:
                Mask, format, and type

        """
        path = Path(file_path)

        if not path.exists():
            msg = f"Mask file not found: {file_path}"
            raise FileNotFoundError(msg)

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
                # Sparse format
                mask = sp.load_npz(path)

                # Load metadata
                metadata_path = f"{path}.metadata.json"
                if os.path.exists(metadata_path):
                    with open(metadata_path) as f:
                        metadata = json.load(f)
                    format = MaskFormat(metadata["format"])
                    mask_type = MaskType(metadata["mask_type"])
                else:
                    # Default to sparse format and attention mask type
                    format = MaskFormat.SPARSE
                    mask_type = MaskType.ATTENTION

                return mask, format, mask_type

            except Exception as e:
                msg = f"Failed to load mask from {file_path}: {e}"
                raise ValueError(msg)

        elif path.suffix == ".json":
            # Load as JSON file (block-sparse format)
            try:
                with open(path) as f:
                    data = json.load(f)

                blocks = data["blocks"]
                metadata = data["metadata"]
                format = MaskFormat(metadata["format"])
                mask_type = MaskType(metadata["mask_type"])

                return blocks, format, mask_type

            except Exception as e:
                msg = f"Failed to load mask from {file_path}: {e}"
                raise ValueError(msg)

        else:
            msg = f"Unsupported file format: {path.suffix}"
            raise ValueError(msg)

    def clear_cache(self):
        """Clear the mask cache."""
        self.distance_calculator.distance_cache.clear()
        self._mask_cache.clear()
