from __future__ import annotations

"""
Paper processor module for Saplings memory.

This module provides specialized functions for processing research papers
and integrating them with GASA.
"""


import logging
from typing import Any, cast

from saplings.api.gasa import GASAConfig

# Import retrieval components to avoid circular imports
from saplings.api.retrieval import EmbeddingRetriever, RetrievalConfig
from saplings.memory._internal.document import Document, DocumentMetadata
from saplings.memory._internal.graph import DependencyGraph
from saplings.memory._internal.memory_store import MemoryStore
from saplings.memory._internal.paper_chunker import build_section_relationships, chunk_paper

logger = logging.getLogger(__name__)


class PaperProcessor:
    """
    Processor for research papers that integrates with GASA.

    This class handles the chunking, embedding, and graph building for research papers,
    ensuring they are properly integrated with GASA.
    """

    def __init__(
        self,
        memory_store: MemoryStore,
        dependency_graph: DependencyGraph | None = None,
        embedding_retriever: EmbeddingRetriever | None = None,
        gasa_config: GASAConfig | None = None,
        max_chunk_size: int = 1000,
    ) -> None:
        """
        Initialize the paper processor.

        Args:
        ----
            memory_store: Memory store for storing documents
            dependency_graph: Dependency graph for storing relationships
            embedding_retriever: Embedding retriever for generating embeddings
            gasa_config: GASA configuration
            max_chunk_size: Maximum size of each chunk in characters

        """
        self.memory_store = memory_store
        self.dependency_graph = dependency_graph or DependencyGraph()

        # Create embedding retriever if not provided
        if embedding_retriever is None:
            self.embedding_retriever = EmbeddingRetriever(
                cast(Any, memory_store), RetrievalConfig()
            )
        else:
            self.embedding_retriever = embedding_retriever

        self.gasa_config = gasa_config or GASAConfig.default()
        self.max_chunk_size = max_chunk_size

    def process_paper(
        self,
        paper_content: str | dict[str, str],
        paper_id: str,
        metadata: dict[str, str] | None = None,
    ) -> tuple[Document, list[Document]]:
        """
        Process a research paper for GASA integration.

        Args:
        ----
            paper_content: Paper content or dictionary with paper content
            paper_id: ID of the paper
            metadata: Additional metadata for the paper

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
        metadata_obj = DocumentMetadata(
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
            metadata=metadata_obj,
            embedding=None,
        )

        # Chunk the document
        chunks = chunk_paper(parent_doc, self.max_chunk_size)

        # Add parent document to memory store
        self.memory_store.add_document(
            content=parent_doc.content, metadata=parent_doc.metadata, document_id=parent_doc.id
        )

        # Add chunks to memory store
        for chunk in chunks:
            self.memory_store.add_document(
                content=chunk.content, metadata=chunk.metadata, document_id=chunk.id
            )

        # Build relationships between chunks
        relationships = build_section_relationships(chunks)

        # Add relationships to the dependency graph
        for rel in relationships:
            self.dependency_graph.add_relationship(
                rel["source_id"],  # source_id_or_relationship
                rel["target_id"],  # target_id
                rel["relationship_type"],  # relationship_type
                1.0,  # weight
                rel["metadata"],  # metadata
            )

        return parent_doc, chunks
