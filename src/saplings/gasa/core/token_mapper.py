from __future__ import annotations

"""
Token mapping for Graph-Aligned Sparse Attention (GASA).

This module provides utilities for tracking the exact token positions
during prompt composition to ensure accurate mask alignment.
"""


import logging
from typing import TYPE_CHECKING, Any

from saplings.gasa.core.chunk_info import ChunkInfo

if TYPE_CHECKING:
    from saplings.memory.document import Document

logger = logging.getLogger(__name__)


class TokenMapper:
    """
    Tracks token positions during prompt composition for accurate GASA masks.

    This class provides bookkeeping for mapping between tokens and document chunks
    during prompt composition, ensuring that token-to-chunk mapping remains accurate
    even when prompt processing (like adding headers or newline normalization) is applied.
    """

    def __init__(self, tokenizer: Any) -> None:
        """
        Initialize the token mapper.

        Args:
        ----
            tokenizer: Tokenizer to use for token counting

        """
        self.tokenizer = tokenizer
        self.chunk_infos: list[ChunkInfo] = []
        self.current_pos = 0

    def add_text(self, text: str) -> tuple[int, int]:
        """
        Add plain text to the prompt and track its token position.

        Args:
        ----
            text: Text to add

        Returns:
        -------
            Tuple[int, int]: Start and end token positions

        """
        # Count the tokens in the text
        tokens = self.tokenizer(text, add_special_tokens=False).input_ids
        token_count = len(tokens)

        # Record positions
        start_pos = self.current_pos
        end_pos = start_pos + token_count

        # Update current position
        self.current_pos = end_pos

        return start_pos, end_pos

    def add_document_chunk(
        self, chunk: Document, document_id: str, node_id: str | None = None
    ) -> ChunkInfo:
        """
        Add a document chunk to the prompt and track its token position.

        Args:
        ----
            chunk: Document chunk to add
            document_id: ID of the document
            node_id: ID of the corresponding node in the dependency graph

        Returns:
        -------
            ChunkInfo: Information about the chunk's position

        """
        # Count the tokens in the chunk
        content = chunk.content
        tokens = self.tokenizer(content, add_special_tokens=False).input_ids
        token_count = len(tokens)

        # Record positions
        start_pos = self.current_pos
        end_pos = start_pos + token_count

        # Create chunk info
        chunk_info = ChunkInfo(
            chunk_id=chunk.id,
            document_id=document_id,
            start_token=start_pos,
            end_token=end_pos,
            node_id=node_id or chunk.id,
        )

        # Add to list of chunks
        self.chunk_infos.append(chunk_info)

        # Update current position
        self.current_pos = end_pos

        return chunk_info

    def add_separator(self, separator: str) -> tuple[int, int]:
        """
        Add a separator to the prompt and track its token position.

        Args:
        ----
            separator: Separator text

        Returns:
        -------
            Tuple[int, int]: Start and end token positions

        """
        return self.add_text(separator)

    def get_chunk_infos(self):
        """
        Get all chunk infos tracked by this mapper.

        Returns
        -------
            List[ChunkInfo]: List of chunk infos

        """
        return self.chunk_infos

    def get_token_count(self):
        """
        Get the total number of tokens tracked.

        Returns
        -------
            int: Total number of tokens

        """
        return self.current_pos
