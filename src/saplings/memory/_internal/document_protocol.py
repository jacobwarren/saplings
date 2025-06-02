from __future__ import annotations

"""
Document protocol module for Saplings memory.

This module defines the Document protocol for type checking.
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Document(Protocol):
    """Protocol for document objects."""

    id: str
    content: str
    metadata: dict[str, Any] | None
    chunks: list[Any] | None

    def chunk(self, chunk_size: int, chunk_overlap: int = 0) -> list[Any]: ...
    def create_chunks(self) -> list[Any]: ...
