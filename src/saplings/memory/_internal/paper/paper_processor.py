from __future__ import annotations

"""
Paper processor module for Saplings memory.

This module provides specialized functions for processing research papers
and integrating them with GASA.
"""


import logging
from typing import TYPE_CHECKING, Any

from saplings.memory._internal.document import Document, DocumentMetadata
from saplings.memory._internal.paper.paper_chunker import build_section_relationships, chunk_paper

if TYPE_CHECKING:
    # Import from API modules to avoid circular imports
    from saplings.memory._internal.graph.dependency_graph import DependencyGraph
    from saplings.memory._internal.memory_store import MemoryStore

    # Use forward references for retrieval types to avoid circular imports
    RetrievalConfig = Any
    EmbeddingRetriever = Any

logger = logging.getLogger(__name__)


def process_paper(
    paper_content: str | dict[str, str],
    paper_id: str,
    memory_store: "MemoryStore",
    dependency_graph: "DependencyGraph | None" = None,
    metadata: dict[str, Any] | None = None,
    max_chunk_size: int = 1000,
) -> tuple[Document, list[Document]]:
    """
    Process a research paper for GASA integration.

    Args:
    ----
        paper_content: Paper content or dictionary with paper content
        paper_id: ID of the paper
        memory_store: Memory store for storing documents
        dependency_graph: Dependency graph for storing relationships
        metadata: Additional metadata for the paper
        max_chunk_size: Maximum size of each chunk in characters

    Returns:
    -------
        Tuple[Document, List[Document]]: Parent document and list of chunk documents

    """
    # Extract content from dictionary if needed
    if isinstance(paper_content, dict):
        content = paper_content.get("content", "")
        if not content and "abstract" in paper_content:
            # Build content from components
            title = paper_content.get("title", "")
            authors = paper_content.get("authors", "")
            abstract = paper_content.get("abstract", "")
            paper_text = paper_content.get("text", "")

            content = (
                f"Title: {title}\n\nAuthors: {authors}\n\nAbstract: {abstract}\n\n{paper_text}"
            )
    else:
        content = paper_content

    # Create metadata if not provided
    if metadata is None:
        metadata = {}

    # Create document metadata
    doc_metadata = DocumentMetadata(
        source=metadata.get("source", f"paper:{paper_id}"),
        content_type=metadata.get("content_type", "text/plain"),
        language=metadata.get("language", "en"),
        author=metadata.get("author", ""),
        tags=list(metadata.get("tags", [])) if isinstance(metadata.get("tags"), list) else [],
        custom={
            "paper_id": paper_id,
            "title": metadata.get("title", ""),
            "abstract": metadata.get("abstract", ""),
            **{
                k: v
                for k, v in metadata.items()
                if k not in ["source", "content_type", "language", "author", "tags"]
            },
        },
    )

    # Create parent document
    parent_doc = Document(
        id=f"paper_{paper_id}",
        content=content,
        metadata=doc_metadata,
        embedding=None,
    )

    # Chunk the document
    chunks = chunk_paper(parent_doc, max_chunk_size)

    # Add parent document to memory store
    memory_store.add_document(parent_doc)

    # Add chunks to memory store
    for chunk in chunks:
        memory_store.add_document(chunk)

    # Build relationships between chunks if dependency graph is provided
    if dependency_graph is not None:
        build_section_relationships(chunks, dependency_graph)

    return parent_doc, chunks
