"""
Block-diagonal packing module for Graph-Aligned Sparse Attention (GASA).

This module provides the block-diagonal packing fallback for models that don't
support sparse attention masks.
"""

import logging
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from saplings.gasa.config import GASAConfig
from saplings.memory.document import Document
from saplings.memory.graph import DependencyGraph

logger = logging.getLogger(__name__)


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
        graph: DependencyGraph,
        config: Optional[GASAConfig] = None,
        tokenizer: Optional[Any] = None,
    ):
        """
        Initialize the block-diagonal packer.

        Args:
            graph: Dependency graph
            config: GASA configuration
            tokenizer: Tokenizer for converting text to tokens
        """
        self.graph = graph
        self.config = config or GASAConfig()
        self.tokenizer = tokenizer

    def reorder_tokens(
        self,
        documents: List[Document],
        prompt: str,
        input_ids: List[int],
        attention_mask: Optional[np.ndarray] = None,
    ) -> Tuple[List[int], List[int], Dict[int, int]]:
        """
        Reorder tokens to create a block-diagonal structure.

        Args:
            documents: Documents used in the prompt
            prompt: Prompt text
            input_ids: Token IDs
            attention_mask: Attention mask (optional)

        Returns:
            Tuple[List[int], List[int], Dict[int, int]]:
                Reordered token IDs, reordered attention mask, and mapping from
                original to reordered positions
        """
        # Map tokens to chunks
        chunk_mapping = self._map_tokens_to_chunks(documents, prompt, input_ids)

        # Group chunks by document and graph distance
        chunk_groups = self._group_chunks_by_distance(documents, chunk_mapping)

        # Create reordering based on chunk groups
        reordering = self._create_reordering(chunk_groups, len(input_ids))

        # Apply reordering to input IDs
        reordered_input_ids = [input_ids[i] for i in reordering]

        # Apply reordering to attention mask if provided
        reordered_attention_mask = None
        if attention_mask is not None:
            reordered_attention_mask = [attention_mask[i] for i in reordering]

        # Create mapping from original to reordered positions
        position_mapping = {orig: reordered for orig, reordered in enumerate(reordering)}

        return reordered_input_ids, reordered_attention_mask, position_mapping

    def _map_tokens_to_chunks(
        self,
        documents: List[Document],
        prompt: str,
        input_ids: List[int],
    ) -> Dict[int, Tuple[str, str]]:
        """
        Map tokens to document chunks.

        Args:
            documents: Documents used in the prompt
            prompt: Prompt text
            input_ids: Token IDs

        Returns:
            Dict[int, Tuple[str, str]]: Mapping from token position to (document_id, chunk_id)
        """
        # This is a simplified implementation that assumes the prompt is constructed
        # by concatenating document chunks. In a real implementation, you would need
        # to track the mapping between tokens and document chunks more carefully.

        chunk_mapping = {}
        current_pos = 0

        for doc in documents:
            # Get or create chunks for the document
            if hasattr(doc, "chunks") and doc.chunks:
                chunks = doc.chunks
            else:
                chunks = doc.create_chunks()

            for chunk in chunks:
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

                    # Map tokens to chunk
                    for pos in range(start_token, end_token):
                        chunk_mapping[pos] = (doc.id, chunk.id)

                    # Update current position
                    current_pos = chunk_pos + len(chunk_text)

        return chunk_mapping

    def _group_chunks_by_distance(
        self,
        documents: List[Document],
        chunk_mapping: Dict[int, Tuple[str, str]],
    ) -> List[List[int]]:
        """
        Group chunks by document and graph distance.

        Args:
            documents: Documents used in the prompt
            chunk_mapping: Mapping from token position to (document_id, chunk_id)

        Returns:
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
        for doc_id, chunk_id in chunk_to_tokens.keys():
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
                related_chunks = self._find_related_chunks(start_chunk, chunk_to_tokens.keys())

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
        self, start_chunk: Tuple[str, str], all_chunks: Set[Tuple[str, str]]
    ) -> Set[Tuple[str, str]]:
        """
        Find all chunks within max_hops of the start chunk.

        Args:
            start_chunk: Starting chunk (document_id, chunk_id)
            all_chunks: All available chunks

        Returns:
            Set[Tuple[str, str]]: Set of related chunks including the start chunk
        """
        # Initialize result with the start chunk
        related_chunks = {start_chunk}

        # Use breadth-first search to find related chunks
        queue = deque([(start_chunk, 0)])  # (chunk, distance)
        visited = {start_chunk}

        while queue:
            current_chunk, distance = queue.popleft()
            doc_id, chunk_id = current_chunk

            # If we've reached max_hops, don't explore further
            if distance >= self.config.max_hops:
                continue

            # Get neighbors from the graph
            try:
                # Get document node from the graph
                doc_node = self.graph.get_node(doc_id)
                if doc_node is None:
                    continue

                # Get chunk node from the document
                chunk_node = doc_node.get_chunk(chunk_id)
                if chunk_node is None:
                    continue

                # Get neighbors
                neighbors = self.graph.get_neighbors(doc_id, chunk_id)

                # Add neighbors to the queue
                for neighbor in neighbors:
                    neighbor_chunk = (neighbor.document_id, neighbor.chunk_id)

                    # Skip if already visited or not in our chunk set
                    if neighbor_chunk in visited or neighbor_chunk not in all_chunks:
                        continue

                    # Add to visited and queue
                    visited.add(neighbor_chunk)
                    queue.append((neighbor_chunk, distance + 1))

                    # Add to related chunks
                    related_chunks.add(neighbor_chunk)
            except Exception as e:
                logger.warning(f"Error finding neighbors for chunk {current_chunk}: {e}")

        return related_chunks

    def _create_reordering(
        self,
        chunk_groups: List[List[int]],
        seq_len: int,
    ) -> List[int]:
        """
        Create a reordering based on chunk groups.

        Args:
            chunk_groups: Groups of token positions
            seq_len: Sequence length

        Returns:
            List[int]: Reordering (list of original positions in new order)
        """
        # Flatten groups
        reordering = []
        for group in chunk_groups:
            reordering.extend(group)

        # Add any positions that weren't mapped to chunks
        unmapped = set(range(seq_len)) - set(reordering)
        reordering.extend(sorted(unmapped))

        return reordering

    def restore_order(
        self,
        reordered_output: List[Any],
        position_mapping: Dict[int, int],
    ) -> List[Any]:
        """
        Restore the original order of tokens.

        Args:
            reordered_output: Output with reordered tokens
            position_mapping: Mapping from original to reordered positions

        Returns:
            List[Any]: Output with original token order
        """
        # Create inverse mapping
        inverse_mapping = {reordered: orig for orig, reordered in position_mapping.items()}

        # Restore original order
        original_order = [None] * len(reordered_output)
        for reordered_pos, value in enumerate(reordered_output):
            if reordered_pos in inverse_mapping:
                orig_pos = inverse_mapping[reordered_pos]
                original_order[orig_pos] = value

        return original_order
