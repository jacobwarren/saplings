from __future__ import annotations

"""
Paper processing module for memory components.

This module provides paper processing functionality for the Saplings framework.
"""

from saplings.memory._internal.paper.paper_chunker import build_section_relationships, chunk_paper
from saplings.memory._internal.paper.paper_processor import process_paper

__all__ = [
    "chunk_paper",
    "build_section_relationships",
    "process_paper",
]
