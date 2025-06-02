from __future__ import annotations

"""
Block-diagonal packer implementation for Graph-Aligned Sparse Attention (GASA).

This module provides the BlockDiagonalPacker class, which implements token reordering
to enable GASA functionality on models that don't natively support sparse attention.
"""


import logging
from typing import Any, Dict, Protocol, Union, cast, runtime_checkable

import numpy as np

from saplings.gasa._internal.config import GASAConfig
from saplings.gasa._internal.core.graph_distance import GraphDistanceCalculator
from saplings.gasa._internal.core.token_mapper import TokenMapper

logger = logging.getLogger(__name__)


@runtime_checkable
class Document(Protocol):
    """Protocol for document objects."""

    id: str
    content: str
    metadata: dict[str, Any] | None
    chunks: list[Any] | None

    def __getattr__(self, name: str) -> Any: ...


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


class BlockDiagonalPacker:
    """
    Block-diagonal packer for Graph-Aligned Sparse Attention (GASA).

    This class implements the block-diagonal packing fallback for models that
    don't support sparse attention masks. It reorders tokens so that related
    tokens are close to each other, allowing the model to use its limited
    attention window effectively.
    """

    def __init__(
        self,
        graph: Any,
        config: GASAConfig | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        """
        Initialize the block-diagonal packer.

        Args:
        ----
            graph: Dependency graph (implements DependencyGraph protocol)
            config: GASA configuration
            tokenizer: Tokenizer for converting text to tokens

        """
        self.graph = graph
        # Use provided config or create a default one with explicit values
        if config is None:
            from saplings.gasa._internal.config import FallbackStrategy, MaskStrategy

            self.config = GASAConfig(
                enabled=True,  # Default GASA settings for packer context
                max_hops=2,
                mask_strategy=MaskStrategy.BINARY,
                fallback_strategy=FallbackStrategy.BLOCK_DIAGONAL,
                global_tokens=["[CLS]", "[SEP]", "<s>", "</s>", "[SUM]"],
                summary_token="[SUM]",
                add_summary_token=True,
                block_size=512,
                overlap=64,
                soft_mask_temperature=0.1,
                cache_masks=False,  # Caching not relevant for packer itself
                cache_dir=None,
                visualize=False,
                visualization_dir=None,
                enable_shadow_model=False,
                shadow_model_name="Qwen/Qwen3-1.8B",  # Match default from config.py
                shadow_model_device="cpu",
                shadow_model_cache_dir=None,
                enable_prompt_composer=False,
                focus_tags=False,
                core_tag="[CORE_CTX]",
                near_tag="[NEAR_CTX]",
                summary_tag="[SUMMARY_CTX]",
            )
        else:
            self.config = config
        self.tokenizer = tokenizer

        # Initialize graph distance calculator for efficient distance calculations
        self.distance_calculator = GraphDistanceCalculator(graph)

    def reorder_tokens(
        self,
        documents: list[Any],
        prompt: str,
        input_ids: list[int],
        attention_mask: np.ndarray | list[int] | None = None,
    ) -> tuple[list[int], Union[np.ndarray, list[int], None], Dict[int, int]]:
        """
        Reorder tokens to create a block-diagonal structure.

        Args:
        ----
            documents: Documents used in the prompt (implement Document protocol)
            prompt: Prompt text
            input_ids: Token IDs
            attention_mask: Attention mask (optional)

        Returns:
        -------
            Tuple[List[int], List[int], Dict[int, int]]:
                Reordered token IDs, reordered attention mask, and mapping from
                original to reordered positions

        """
        # Map tokens to chunks using the TokenMapper
        chunk_infos = self._map_tokens_to_chunks(documents, prompt, input_ids)

        # Convert ChunkInfo to the format expected by the rest of the packer
        chunk_mapping = {}
        for chunk_info in chunk_infos:
            for pos in range(chunk_info.start_token, chunk_info.end_token):
                chunk_mapping[pos] = (chunk_info.document_id, chunk_info.chunk_id)

        # Group chunks by document and graph distance
        chunk_groups = self._group_chunks_by_distance(documents, chunk_mapping)

        # Create reordering based on chunk groups
        reordering = self._create_reordering(chunk_groups, len(input_ids))

        # Apply reordering to input IDs
        reordered_input_ids = [input_ids[i] for i in reordering]

        # Apply reordering to attention mask if provided
        reordered_attention_mask = None
        if attention_mask is not None:
            if isinstance(attention_mask, np.ndarray):
                # Reorder 2D attention mask
                reordered_attention_mask = attention_mask[reordering][:, reordering]
            else:
                # Reorder 1D attention mask (typically used for padding)
                reordered_attention_mask = [attention_mask[i] for i in reordering]

        # Create mapping from original to reordered positions
        position_mapping = {orig: reordered for reordered, orig in enumerate(reordering)}

        return reordered_input_ids, reordered_attention_mask, position_mapping

    def _map_tokens_to_chunks(
        self,
        documents: list[Any],
        prompt: str,  # pylint: disable=unused-argument
        input_ids: list[int],  # pylint: disable=unused-argument
    ) -> list[Any]:
        """
        Map tokens to document chunks using the TokenMapper.

        Args:
        ----
            documents: Documents used in the prompt (implement Document protocol)
            prompt: Prompt text
            input_ids: Token IDs

        Returns:
        -------
            List[Any]: Information about document chunks (ChunkInfo objects)

        """
        # Use the TokenMapper for accurate token-to-chunk mapping
        token_mapper = TokenMapper(self.tokenizer)

        # Process prompt parts and track token positions
        for doc in documents:
            # Get or create chunks for the document
            chunks = []
            if hasattr(doc, "chunks") and doc.chunks:
                chunks = doc.chunks
            elif hasattr(doc, "chunk"):
                try:
                    # Use the chunk method to create chunks with default parameters
                    # The Document.chunk method returns a list of Document objects
                    chunk_method = doc.chunk
                    # Use default chunk size of 1000 and overlap of 100 if not specified in config
                    chunk_size = getattr(self.config, "block_size", 1000)
                    chunk_overlap = getattr(self.config, "overlap", 100)
                    chunks = chunk_method(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                except Exception as e:
                    logger.warning(f"Error creating chunks for document {doc.id}: {e}")
            elif hasattr(doc, "content"):  # Fallback if no chunk methods
                logger.warning(
                    f"Document {doc.id} has no chunks or chunk method. Treating content as single chunk."
                )

                # Define SimpleChunk providing id and content for TokenMapper
                class SimpleChunk:
                    def __init__(self, id: str, content: str) -> None:
                        self.id = id
                        self.content = content
                        # Add other potential Document attributes as needed, or leave them out if TokenMapper doesn't use them
                        # self.metadata: dict[str, Any] = {}

                chunks = [SimpleChunk(id=f"{doc.id}_chunk_0", content=doc.content)]

            for chunk in chunks:
                if not hasattr(chunk, "content") or not hasattr(chunk, "id"):
                    logger.warning(f"Skipping invalid chunk in document {doc.id}")
                    continue
                # Pass attributes needed by TokenMapper instead of the whole chunk object
                # Cast to Any to bypass strict type checking for this specific call,
                # as we know TokenMapper only needs .id and .content at runtime.
                chunk_data = cast("Any", {"id": chunk.id, "content": chunk.content})
                token_mapper.add_document_chunk(
                    chunk=chunk_data,
                    document_id=doc.id,
                    node_id=f"document:{doc.id}",
                )

        # Return the tracked chunk infos
        return token_mapper.get_chunk_infos()

    def _group_chunks_by_distance(
        self,
        documents: list[Any],  # pylint: disable=unused-argument
        chunk_mapping: dict[int, tuple[str, str]],
    ) -> list[list[int]]:
        """
        Group chunks by document and graph distance.

        Args:
        ----
            documents: Documents used in the prompt (implement Document protocol)
            chunk_mapping: Mapping from token position to (document_id, chunk_id)

        Returns:
        -------
            List[List[int]]: Groups of token positions

        """
        # Create a mapping from (document_id, chunk_id) to token positions
        chunk_to_tokens = {}
        for pos, (doc_id, chunk_id) in chunk_mapping.items():
            key = (doc_id, chunk_id)
            if key not in chunk_to_tokens:
                chunk_to_tokens[key] = []
            chunk_to_tokens[key].append(pos)

        # Create a mapping from document_id to chunk keys
        doc_to_chunks = {}
        for doc_id, chunk_id in chunk_to_tokens:
            if doc_id not in doc_to_chunks:
                doc_to_chunks[doc_id] = []
            doc_to_chunks[doc_id].append((doc_id, chunk_id))

        # If we don't have a graph or max_hops is 0, just group by document
        if self.graph is None or self.config.max_hops <= 0:
            # Group chunks by document only
            groups = []
            for doc_id, chunk_keys in doc_to_chunks.items():
                # Get all token positions for this document
                doc_tokens = []
                for key in chunk_keys:
                    doc_tokens.extend(chunk_to_tokens[key])

                # Sort token positions
                doc_tokens.sort()

                # Add to groups
                groups.append(doc_tokens)

            return groups

        # Group chunks based on graph distance
        groups = []
        processed_chunks = set()

        # Process each document
        for doc_id, chunk_keys in doc_to_chunks.items():
            # Process each chunk in the document
            for start_chunk in chunk_keys:
                # Skip if already processed
                if start_chunk in processed_chunks:
                    continue

                # Mark as processed
                processed_chunks.add(start_chunk)

                # Find all chunks within max_hops of this chunk
                related_chunks = self._find_related_chunks(start_chunk, set(chunk_to_tokens.keys()))

                # Mark all related chunks as processed
                processed_chunks.update(related_chunks)

                # Get all token positions for this group
                group_tokens = []
                for chunk in related_chunks:
                    group_tokens.extend(chunk_to_tokens[chunk])

                # Sort token positions
                group_tokens.sort()

                # Add to groups if not empty
                if group_tokens:
                    groups.append(group_tokens)

        # Handle any remaining chunks that weren't processed
        remaining_chunks = set(chunk_to_tokens.keys()) - processed_chunks
        if remaining_chunks:
            remaining_tokens = []
            for chunk in remaining_chunks:
                remaining_tokens.extend(chunk_to_tokens[chunk])

            # Sort token positions
            remaining_tokens.sort()

            # Add to groups if not empty
            if remaining_tokens:
                groups.append(remaining_tokens)

        return groups

    def _find_related_chunks(
        self, start_chunk: tuple[str, str], all_chunks: set[tuple[str, str]]
    ) -> set[tuple[str, str]]:
        """
        Find all chunks within max_hops of the start chunk.

        Args:
        ----
            start_chunk: Starting chunk (document_id, chunk_id)
            all_chunks: All available chunks

        Returns:
        -------
            Set[Tuple[str, str]]: Set of related chunks including the start chunk

        """
        # Use the GraphDistanceCalculator for efficient distance queries
        doc_id, chunk_id = start_chunk  # pylint: disable=unused-variable
        start_node_id = f"document:{doc_id}"

        # Initialize result with the start chunk
        related_chunks = {start_chunk}

        # Check each potential chunk's distance
        for other_chunk in all_chunks:
            if other_chunk == start_chunk:
                continue

            other_doc_id, other_chunk_id = other_chunk  # pylint: disable=unused-variable
            other_node_id = f"document:{other_doc_id}"

            # Get distance between the document nodes
            distance = self.distance_calculator.get_distance(
                start_node_id, other_node_id, self.config.max_hops
            )

            # If within max_hops, add to related chunks
            if distance <= self.config.max_hops:
                related_chunks.add(other_chunk)

        return related_chunks

    def _create_reordering(
        self,
        chunk_groups: list[list[int]],
        seq_len: int,
    ) -> list[int]:
        """
        Create a reordering based on chunk groups.

        Args:
        ----
            chunk_groups: Groups of token positions
            seq_len: Sequence length

        Returns:
        -------
            List[int]: Reordering (list of original positions in new order)

        """
        # Flatten groups - this gives us the new token order
        # (original position -> new position in the reordered sequence)
        reordering = []
        for group in chunk_groups:
            reordering.extend(group)

        # Add any positions that weren't mapped to chunks
        unmapped = set(range(seq_len)) - set(reordering)
        reordering.extend(sorted(unmapped))

        return reordering

    def restore_order(
        self,
        reordered_output: list[Any],
        position_mapping: dict[int, int],
    ) -> list[Any]:
        """
        Restore the original order of tokens.

        Args:
        ----
            reordered_output: Output with reordered tokens
            position_mapping: Mapping from original to reordered positions

        Returns:
        -------
            List[Any]: Output with original token order

        """
        # Create inverse mapping from reordered position to original position
        # The position_mapping goes from original -> reordered
        inverse_mapping = {reordered: orig for orig, reordered in position_mapping.items()}

        # Restore original order
        original_order = [None] * len(reordered_output)
        for reordered_pos, value in enumerate(reordered_output):
            if reordered_pos in inverse_mapping:
                orig_pos = inverse_mapping[reordered_pos]
                original_order[orig_pos] = value

        return original_order
