from __future__ import annotations

"""
Chunk information for Graph-Aligned Sparse Attention (GASA).

This module provides the ChunkInfo class, which represents information about
a document chunk for token-to-chunk mapping in GASA.
"""


from typing import Any


class ChunkInfo:
    """Information about a document chunk."""

    def __init__(
        self,
        chunk_id: str,
        document_id: str,
        start_token: int,
        end_token: int,
        node_id: str | None = None,
    ) -> None:
        """
        Initialize chunk information.

        Args:
        ----
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

    def __repr__(self):
        return (
            f"ChunkInfo(chunk_id={self.chunk_id}, start={self.start_token}, end={self.end_token})"
        )

    def contains_token(self, token_idx: int) -> bool:
        """
        Check if the chunk contains a token.

        Args:
        ----
            token_idx: Token index

        Returns:
        -------
            bool: True if the chunk contains the token, False otherwise

        """
        return self.start_token <= token_idx < self.end_token

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary.

        Returns
        -------
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
    def from_dict(cls, data: dict[str, Any]) -> "ChunkInfo":
        """
        Create from dictionary.

        Args:
        ----
            data: Dictionary representation

        Returns:
        -------
            ChunkInfo: Chunk information

        """
        return cls(
            chunk_id=data["chunk_id"],
            document_id=data["document_id"],
            start_token=data["start_token"],
            end_token=data["end_token"],
            node_id=data.get("node_id"),
        )
