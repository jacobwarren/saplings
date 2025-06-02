from __future__ import annotations

"""StandardMaskBuilder implementation."""

"""
Standard mask builder implementation for Graph-Aligned Sparse Attention (GASA).

This module provides a standard implementation of the MaskBuilderInterface that builds
attention masks based on document dependency graphs using a chunk-based approach.
"""

# Standard library imports
import hashlib
import importlib.util
import json
import logging
import platform
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    TypeGuard,
    runtime_checkable,
)

# Third-party imports
import numpy as np
import scipy.sparse as sp

# Local application imports
from saplings.gasa._internal.config import GASAConfig
from saplings.gasa._internal.core.chunk_info import ChunkInfo
from saplings.gasa._internal.core.interfaces import MaskBuilderInterface
from saplings.gasa._internal.core.types import MaskFormat, MaskType
from saplings.tokenizers._internal.simple_tokenizer import SimpleTokenizer

# Conditional imports and availability checks
try:
    from saplings.tokenizers._internal.shadow_model_tokenizer import ShadowModelTokenizer

    SHADOW_MODEL_AVAILABLE = True
except ImportError:
    SHADOW_MODEL_AVAILABLE = False
    ShadowModelTokenizer = None  # Define as None if not available

TRITON_AVAILABLE = False
IS_APPLE_SILICON = platform.system() == "Darwin" and platform.machine().startswith("arm")
if not IS_APPLE_SILICON:
    if importlib.util.find_spec("triton") is not None:
        TRITON_AVAILABLE = True
    else:
        logger = logging.getLogger(__name__)  # Setup logger here if needed before warning
        logger.warning("Triton not installed. Some GPU acceleration features may not be available.")

# Imports for type checking
if TYPE_CHECKING:
    from collections.abc import Sequence  # Moved here

    import torch
    from numpy.typing import NDArray  # Moved here

# Set up logger (moved after conditional imports that might log)
logger = logging.getLogger(__name__)


@runtime_checkable
class DependencyGraph(Protocol):
    """Protocol for dependency graph objects."""

    def get_neighbors(self, node_id: str) -> list[str]: ...
    def get_distance(self, source_id: str, target_id: str) -> int | float: ...
    def get_subgraph(self, node_ids: list[str], max_hops: int = 2) -> "DependencyGraph": ...
    def add_node(self, node_id: str, metadata: dict[str, Any] | None = None) -> None: ...
    def add_edge(
        self, source_id: str, target_id: str, metadata: dict[str, Any] | None = None
    ) -> None: ...
    def __getattr__(self, name: str) -> Any: ...


# Type guards
def is_tensor(x: Any) -> TypeGuard[torch.Tensor | NDArray[np.int32]]:
    """Check if x is a tensor-like object."""
    return hasattr(x, "shape") and hasattr(x, "tolist")


def is_sparse_matrix(x: Any) -> TypeGuard[sp.csr_matrix | sp.csc_matrix | sp.coo_matrix]:
    """Check if x is a sparse matrix."""
    return sp.issparse(x) and isinstance(x, (sp.csr_matrix, sp.csc_matrix, sp.coo_matrix))


def is_tokenizer_output(x: Any) -> TypeGuard[TokenizerOutput]:
    """Check if x is a tokenizer output."""
    return hasattr(x, "input_ids") or (isinstance(x, dict) and "input_ids" in x)


def get_input_ids(output: Any) -> list[int]:
    """Extract input IDs from tokenizer output."""
    if is_tokenizer_output(output):
        ids = output.input_ids if hasattr(output, "input_ids") else output["input_ids"]
        if is_tensor(ids):
            # Handle multi-dimensional tensors (e.g., shape [1, seq_len])
            if len(ids.shape) > 1 and ids.shape[0] == 1:
                return ids[0].tolist()
            # Handle single-dimensional tensors
            return ids.tolist()
        if isinstance(ids, list):
            # Handle list of lists (e.g., [[id1, id2, ...]])
            if len(ids) > 0 and isinstance(ids[0], list):
                return ids[0]
            # Handle flat list of ints
            return ids
    return []


def to_dense(mask: np.ndarray | sp.spmatrix) -> NDArray[np.int32]:
    """Convert mask to dense numpy array."""
    try:
        if sp.issparse(mask):
            # Use toarray() if available, it's the most standard method
            if hasattr(mask, "toarray"):
                # Ignore Pylance error: hasattr check ensures safety. ndarray doesn't have toarray.
                return np.asarray(mask.toarray(), dtype=np.int32)  # type: ignore[attr-defined]
            # Log a warning if toarray is missing for a sparse matrix
            logger.warning(
                f"Sparse matrix type {type(mask)} lacks toarray method. Conversion might be incomplete."
            )
            # Return an empty array matching shape as a safe fallback
            shape = mask.shape if hasattr(mask, "shape") else (0, 0)
            return np.zeros(shape, dtype=np.int32)
        # Handle dense arrays (ensure correct dtype)
        return np.asarray(mask, dtype=np.int32)
    except Exception as e:
        logger.warning(f"Error converting mask to dense array: {e}")
        # Fallback to an empty array on error
        shape = mask.shape if hasattr(mask, "shape") else (0, 0)
        return np.zeros(shape, dtype=np.int32)


@runtime_checkable
@runtime_checkable
class TokenizerOutput(Protocol):
    """Protocol for tokenizer outputs."""

    input_ids: list[int] | NDArray[np.int32] | torch.Tensor

    def __getitem__(self, key: str) -> Any: ...


@runtime_checkable
class SparseMatrix(Protocol):
    """Protocol for sparse matrices."""

    shape: tuple[int, ...]

    def toarray(self) -> NDArray[np.int32]: ...
    def todense(self) -> NDArray[np.int32]: ...
    def __setitem__(self, key: tuple[int, int | slice], value: int) -> None: ...
    def __getitem__(self, key: tuple[int, int | slice]) -> NDArray[np.int32]: ...
    def get_shape(self) -> tuple[int, ...]: ...

    def astype(self, dtype: Any) -> "SparseMatrix": ...


@runtime_checkable
@runtime_checkable
class Document(Protocol):
    """Protocol for document objects."""

    id: str
    content: str
    metadata: dict[str, Any] | None = None
    chunks: Sequence[Any] | None = None

    def create_chunks(self) -> Sequence[Any]: ...
    def __getattr__(self, name: str) -> Any: ...


class StandardMaskBuilder(MaskBuilderInterface):
    """
    Standard implementation of a mask builder for GASA.

    This class builds sparse attention masks based on document dependency graphs,
    allowing the model to focus on relevant context and reduce computational cost.
    """

    def __init__(
        self,
        graph: Any,
        config: GASAConfig | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        """
        Initialize the mask builder.

        Args:
        ----
            graph: Dependency graph (implements DependencyGraph protocol)
            config: GASA configuration
            tokenizer: Tokenizer for converting text to tokens

        """
        self.graph = graph
        # Create a default config if none is provided
        if config is None:
            # Create with default values
            self.config = GASAConfig.default()
        else:
            self.config = config

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
                if ShadowModelTokenizer is not None:  # Explicit check
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
                    # Fallback if import failed unexpectedly despite SHADOW_MODEL_AVAILABLE check
                    logger.warning("ShadowModelTokenizer is None, falling back to SimpleTokenizer.")
                    self.tokenizer = SimpleTokenizer()
            else:  # This corresponds to the case where Triton IS available (or not Apple Silicon)
                # Use the configured shadow model
                logger.info(f"Using shadow model tokenizer: {self.config.shadow_model_name}")
                if ShadowModelTokenizer is not None:  # Explicit check
                    self.tokenizer = ShadowModelTokenizer(
                        model_name=self.config.shadow_model_name,
                        device=self.config.shadow_model_device,
                        cache_dir=self.config.shadow_model_cache_dir,
                        fallback_to_simple=True,
                        # cpu_only=True and alternative_models removed as they likely belong to the CPU-only case
                    )
                else:
                    # Fallback if import failed unexpectedly despite SHADOW_MODEL_AVAILABLE check
                    logger.warning("ShadowModelTokenizer is None, falling back to SimpleTokenizer.")
                    self.tokenizer = SimpleTokenizer()
        else:
            # Fall back to simple tokenizer
            logger.info("Using simple tokenizer")
            self.tokenizer = SimpleTokenizer()

        # Cache for distance calculations
        self._distance_cache: dict[tuple[str, str], float | int] = {}

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

            # Estimate sequence length using various methods
            try:
                # Try to tokenize the prompt
                seq_len = None
                tokenizer_output = self.tokenizer(prompt, return_tensors="pt")
                input_ids = get_input_ids(tokenizer_output)
                seq_len = len(input_ids)

                # If we still don't have a sequence length, use a fallback
                if seq_len is None or seq_len <= 0:
                    # Fallback to estimating sequence length
                    seq_len = len(prompt) // 4  # Rough estimate of 4 chars per token

            except Exception as e:
                logger.warning(f"Error estimating sequence length: {e}. Using fallback.")
                # Fallback to estimating sequence length
                seq_len = len(prompt) // 4  # Rough estimate of 4 chars per token

            # Ensure we have a valid sequence length
            if seq_len <= 0:
                seq_len = 1  # Avoid zero-length masks

            # Create the appropriate mask format
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

        # Get token IDs using a robust approach
        try:
            # Try to tokenize the prompt
            input_ids = []
            seq_len = 0

            # Try different tokenization approaches
            try:
                tokenizer_output = self.tokenizer(prompt, return_tensors="pt")
                input_ids = get_input_ids(tokenizer_output)
            except Exception as e:
                logger.warning(f"Error with primary tokenization approach: {e}")

            # If we still don't have input_ids, try a simpler approach
            if not input_ids:
                try:
                    # Try direct tokenization without return_tensors
                    tokenizer_output = self.tokenizer(prompt)
                    input_ids = get_input_ids(tokenizer_output)
                except Exception as e:
                    logger.warning(f"Error with secondary tokenization approach: {e}")

            # If we still don't have input_ids, use a fallback
            if not input_ids:
                # Fallback to simple tokenization
                input_ids = list(range(len(prompt.split())))

            # Get sequence length
            seq_len = len(input_ids)
            if seq_len <= 0:
                seq_len = 1  # Avoid zero-length masks

        except Exception as e:
            logger.warning(f"Error tokenizing prompt: {e}. Using fallback tokenization.")
            # Fallback to simple tokenization
            input_ids = list(range(len(prompt.split())))
            seq_len = len(input_ids)
            if seq_len <= 0:
                seq_len = 1  # Avoid zero-length masks

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
            try:
                # Ensure mask is ndarray or spmatrix before calling to_dense
                if not isinstance(mask, (np.ndarray, sp.spmatrix)):
                    msg = f"Invalid mask type for DENSE format: {type(mask)}"
                    raise TypeError(msg)
                mask_array = to_dense(mask)
                np.savez_compressed(
                    path,
                    mask=mask_array,
                    metadata=json.dumps(metadata),
                )
            except Exception as e:
                logger.warning(f"Error saving dense mask: {e}")

        elif format == MaskFormat.SPARSE:
            # Save as .npz file with sparse matrix
            try:
                if not sp.issparse(mask):
                    mask = sp.csr_matrix(mask)
                sp.save_npz(path, mask)

                # Save metadata separately
                with open(f"{path}.metadata.json", "w") as f:
                    json.dump(metadata, f)
            except Exception as e:
                logger.warning(f"Error saving sparse mask: {e}")

        elif format == MaskFormat.BLOCK_SPARSE:
            # Save as JSON file
            try:
                # Ensure mask is list for block sparse format
                if not isinstance(mask, list):
                    msg = f"Invalid mask type for BLOCK_SPARSE format: {type(mask)}"
                    raise TypeError(msg)
                with open(path, "w") as f:
                    json.dump(
                        {
                            "blocks": mask,
                            "metadata": metadata,
                        },
                        f,
                    )
            except Exception as e:
                logger.warning(f"Error saving block sparse mask: {e}")

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
                if "mask" in data and "metadata" in data:
                    # Dense format
                    mask = data["mask"]
                    metadata = json.loads(str(data["metadata"]))
                    format = MaskFormat(metadata["format"])
                    mask_type = MaskType(metadata["mask_type"])
                    return mask, format, mask_type
                # Sparse format
                mask = sp.load_npz(path)

                # Load metadata
                metadata_path = path.with_suffix(".metadata.json")
                if metadata_path.exists():
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
        self._distance_cache.clear()
        self._mask_cache.clear()

    # Private helper methods

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
        # Create a string representation of the inputs
        doc_ids = sorted([doc.id for doc in documents])
        doc_ids_str = ",".join(doc_ids)

        # Hash the prompt to avoid long keys
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()

        # Combine all components
        return (
            f"{doc_ids_str}_{prompt_hash}_{format.value}_{mask_type.value}_{self.config.max_hops}"
        )

    def _map_tokens_to_chunks(
        self,
        documents: list[Document],
        prompt: str,
        input_ids: list[int] | None = None,
    ) -> list[ChunkInfo]:
        """
        Map tokens to document chunks.

        Args:
        ----
            documents: Documents used in the prompt
            prompt: Prompt text
            input_ids: Token IDs

        Returns:
        -------
            List[ChunkInfo]: Chunk information for each token

        """
        # This is a simplified implementation that assumes the prompt is constructed
        # by concatenating document chunks. In a real implementation, you would need
        # to track the mapping between tokens and document chunks more carefully.

        chunk_infos = []
        current_token_pos = 0
        current_char_pos = 0

        for doc in documents:
            # Get chunks for the document
            chunks: Sequence[Any] = []
            if hasattr(doc, "chunks") and doc.chunks:
                chunks = doc.chunks
            elif hasattr(doc, "create_chunks"):
                try:
                    # Use getattr to avoid static type checking errors
                    create_chunks_method = doc.create_chunks
                    chunks = create_chunks_method()
                except Exception as e:
                    logger.warning(f"Error creating chunks for document {doc.id}: {e}")

            # If we still don't have chunks, create a simple chunk-like object from the document content
            if not chunks and hasattr(doc, "content"):
                try:
                    # Create a simple chunk-like object with id and content attributes
                    class SimpleChunk:
                        def __init__(self, id: str, content: str) -> None:
                            self.id = id
                            self.content = content

                    chunks = [SimpleChunk(id=f"{doc.id}_chunk_0", content=doc.content)]
                except Exception as e:
                    logger.warning(f"Error creating fallback chunk for document {doc.id}: {e}")

            for i, chunk in enumerate(chunks):
                if not hasattr(chunk, "content") or not hasattr(chunk, "id"):
                    logger.warning(f"Skipping invalid chunk {i} in document {doc.id}")
                    continue

                chunk_text = chunk.content
                chunk_char_pos = prompt.find(chunk_text, current_char_pos)

                if chunk_char_pos != -1:
                    # Tokenize the text before the chunk
                    prefix_text = prompt[current_char_pos:chunk_char_pos]
                    prefix_tokens = (
                        get_input_ids(self.tokenizer(prefix_text)) if prefix_text else []
                    )

                    # Tokenize the chunk
                    chunk_tokens = get_input_ids(self.tokenizer(chunk_text)) if chunk_text else []

                    # Calculate token positions
                    start_token = current_token_pos + len(prefix_tokens)
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

                    # Update current positions
                    current_char_pos = chunk_char_pos + len(chunk_text)
                    current_token_pos = end_token
                else:
                    logger.warning(
                        f"Chunk {chunk.id} not found in prompt after position {current_char_pos}"
                    )
                    # Attempt to recover by tokenizing the rest of the prompt
                    remaining_prompt = prompt[current_char_pos:]
                    remaining_tokens = get_input_ids(self.tokenizer(remaining_prompt))
                    current_token_pos += len(remaining_tokens)
                    current_char_pos = len(prompt)  # Move to end

        return chunk_infos

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

    def _get_graph_distance(self, node_id1: str, node_id2: str) -> float | int:
        """
        Get the distance between two nodes in the dependency graph.

        Args:
        ----
            node_id1: ID of the first node
            node_id2: ID of the second node

        Returns:
        -------
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
        chunk_infos: list[ChunkInfo],
        seq_len: int,
    ) -> sp.coo_matrix:
        """
        Expand chunk-level adjacency to token-level mask.

        Args:
        ----
            chunk_adjacency: Adjacency matrix for chunks
            chunk_infos: Chunk information
            seq_len: Sequence length

        Returns:
        -------
            sp.coo_matrix: Token-level attention mask (sparse COO matrix)

        """
        # Use sparse COO matrix for better memory efficiency
        # Initialize with diagonal elements (each token attends to itself)
        diag_i = np.arange(seq_len)
        diag_j = np.arange(seq_len)
        diag_data = np.ones(seq_len, dtype=np.int32)

        # Collect coordinates and data for non-zero entries
        rows = list(diag_i)
        cols = list(diag_j)
        data = list(diag_data)

        # Map chunk adjacency to token mask
        for i, chunk_i in enumerate(chunk_infos):
            for j, chunk_j in enumerate(chunk_infos):
                if chunk_adjacency[i, j] == 1:
                    # Add all token pairs between connected chunks
                    # Ensure indices are within bounds
                    start_i = max(0, chunk_i.start_token)
                    end_i = min(seq_len, chunk_i.end_token)
                    start_j = max(0, chunk_j.start_token)
                    end_j = min(seq_len, chunk_j.end_token)

                    for ti in range(start_i, end_i):
                        for tj in range(start_j, end_j):
                            # Skip if it's already on the diagonal
                            if ti != tj:
                                rows.append(ti)
                                cols.append(tj)
                                data.append(1)

        # Create sparse COO matrix
        return sp.coo_matrix((data, (rows, cols)), shape=(seq_len, seq_len), dtype=np.int32)

    def _handle_global_tokens(
        self, token_mask: sp.coo_matrix, input_ids: list[int]
    ) -> Any:  # Use Any to accommodate both spmatrix and csr_array
        """
        Handle global tokens that should attend to all other tokens.

        Args:
        ----
            token_mask: Token-level attention mask (sparse COO)
            input_ids: Token IDs

        Returns:
        -------
            sp.csr_matrix or csr_array: Updated attention mask (CSR format)

        """
        if self.tokenizer is None:
            # Should not happen if build_mask checks correctly, but guard anyway
            if isinstance(token_mask, sp.csr_matrix):
                return token_mask
            return token_mask.tocsr()

        seq_len = token_mask.shape[0]
        if seq_len == 0:
            # Return an empty CSR matrix if seq_len is 0
            return sp.csr_matrix((seq_len, seq_len), dtype=np.int32)

        # Get token IDs for global tokens - simplified approach
        global_positions = []
        if seq_len > 0:
            global_positions.append(0)  # First token
        if seq_len > 1:
            global_positions.append(seq_len - 1)  # Last token

        # Convert to LIL format for efficient modification
        try:
            lil_mask = token_mask.tolil()

            for pos in global_positions:
                if 0 <= pos < seq_len:
                    # Global token attends to all tokens
                    lil_mask.rows[pos] = list(range(seq_len))
                    lil_mask.data[pos] = [1] * seq_len

                    # All tokens attend to global token
                    for i in range(seq_len):
                        if pos not in lil_mask.rows[i]:
                            lil_mask.rows[i].append(pos)
                            lil_mask.data[i].append(1)
                        else:
                            # Ensure the value is 1 if it exists
                            idx = lil_mask.rows[i].index(pos)
                            lil_mask.data[i][idx] = 1

            # Convert back to CSR format for efficiency
            csr_mask = lil_mask.tocsr()
            # Return without explicit casting to allow for both spmatrix and csr_array
            return csr_mask
        except Exception as e:  # Consider more specific exceptions if possible
            logger.warning(f"Error handling global tokens: {e}. Returning original mask.")
            # Ensure return type consistency even on error
            if isinstance(token_mask, sp.csr_matrix):
                return token_mask  # Already CSR, no need to cast
            # Attempt conversion if possible
            if hasattr(token_mask, "tocsr"):
                try:
                    # Return without explicit casting to allow for both spmatrix and csr_array
                    return token_mask.tocsr()
                except Exception as conversion_error:  # Consider more specific exceptions
                    logger.error(
                        f"Failed to convert mask to CSR in exception handler: {conversion_error}"
                    )

            # Fallback: return original coo_matrix if conversion fails - This path is problematic for type hint
            # If we reach here, we can't satisfy the spmatrix hint easily without conversion.
            # Returning an empty csr_matrix is safer than returning the coo_matrix.
            logger.error(
                "Could not convert mask back to CSR format after handling global tokens. Returning empty CSR matrix."
            )
            # Return an empty csr_matrix matching the function signature
            shape = token_mask.shape if hasattr(token_mask, "shape") else (0, 0)
            # Return without explicit casting to allow for both spmatrix and csr_array
            return sp.csr_matrix(
                shape, dtype=token_mask.dtype if hasattr(token_mask, "dtype") else np.int32
            )

    def _convert_mask_format(
        self,
        mask: sp.spmatrix,  # Expect sparse matrix
        format: MaskFormat,
        mask_type: MaskType,
    ) -> np.ndarray | sp.spmatrix | list[dict[str, Any]]:
        """
        Convert mask to the requested format.

        Args:
        ----
            mask: Attention mask (sparse format)
            format: Output format
            mask_type: Type of attention mask

        Returns:
        -------
            Union[np.ndarray, sp.spmatrix, List[Dict[str, Any]]]: Converted mask

        """
        try:
            dense_mask = to_dense(mask)

            if mask_type == MaskType.GLOBAL_ATTENTION:
                # For global attention, 1 means global attention, 0 means local attention
                # We need to invert the mask: global_attention = 1 - attention
                dense_mask = 1 - dense_mask

            if format == MaskFormat.DENSE:
                return dense_mask

            if format == MaskFormat.SPARSE:
                # Convert to CSR (more efficient for row slicing operations)
                # Return without explicit casting to allow for both spmatrix and csr_array
                return sp.csr_matrix(dense_mask)

            if format == MaskFormat.BLOCK_SPARSE:
                # Convert to block-sparse format
                blocks = []
                seq_len = dense_mask.shape[0]

                for i in range(0, seq_len, self.config.block_size):
                    for j in range(0, seq_len, self.config.block_size):
                        # Get block
                        block_i_end = min(i + self.config.block_size, seq_len)
                        block_j_end = min(j + self.config.block_size, seq_len)
                        block = dense_mask[i:block_i_end, j:block_j_end]

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

        except Exception as e:  # Consider more specific exceptions
            logger.warning(f"Error converting mask format: {e}. Returning original mask.")
            # Return in the most compatible format (dense) if conversion fails
            return to_dense(mask)
