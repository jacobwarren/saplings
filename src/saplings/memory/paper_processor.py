from __future__ import annotations

"""
Paper processor module for Saplings memory.

This module provides specialized functions for processing research papers
and integrating them with GASA.
"""


import logging
from typing import TYPE_CHECKING

from saplings.gasa import GASAConfig
from saplings.memory import DependencyGraph, Document, DocumentMetadata, MemoryStore
from saplings.memory.graph import DocumentNode
from saplings.memory.paper_chunker import build_section_relationships, chunk_paper
from saplings.retrieval.config import RetrievalConfig
from saplings.retrieval.embedding_retriever import EmbeddingRetriever

if TYPE_CHECKING:
    from saplings.memory.indexer import Entity

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
            self.embedding_retriever = EmbeddingRetriever(memory_store, RetrievalConfig())
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
            id=f"paper_{paper_id}", content=content, metadata=doc_metadata, embedding=None
        )

        # Chunk the paper into logical sections
        chunks = chunk_paper(parent_doc, self.max_chunk_size)
        logger.info("Chunked paper into %d sections", len(chunks))

        # Generate embeddings for each chunk
        for chunk in chunks:
            try:
                # Make sure the content is not empty
                if not chunk.content.strip():
                    logger.warning(
                        "Chunk %s has empty content, skipping embedding generation", chunk.id
                    )
                    continue

                # Generate embedding
                embedding = self.embedding_retriever.embed_query(chunk.content)

                # Update the embedding
                chunk.update_embedding(embedding)

                # Verify the embedding was set
                if chunk.embedding is None:
                    logger.warning("Failed to set embedding for chunk %s", chunk.id)
            except Exception as e:  # noqa: BLE001
                logger.warning("Error generating embedding for chunk %s: %s", chunk.id, e)

        # Generate embedding for parent document
        try:
            parent_embedding = self.embedding_retriever.embed_query(content)
            parent_doc.update_embedding(parent_embedding)

            # Verify the embedding was set
            if parent_doc.embedding is None:
                logger.warning("Failed to set embedding for parent document %s", parent_doc.id)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Error generating embedding for parent document %s: %s", parent_doc.id, e
            )

        # Add parent document to memory store
        # Use add_documents method which accepts Document objects
        self.memory_store.add_documents([parent_doc])

        # Add chunks to memory store
        self.memory_store.add_documents(chunks)

        # First, add all document nodes to the dependency graph
        # This is crucial - we need to add nodes before adding relationships
        self.dependency_graph.add_document_node(parent_doc)

        # Add all chunk nodes to the dependency graph
        for chunk in chunks:
            self.dependency_graph.add_document_node(chunk)

        logger.info("Added %d nodes to dependency graph", len(chunks) + 1)

        # Build relationships between sections
        relationships = build_section_relationships(chunks)

        # Add relationships to dependency graph
        successful_relationships = 0
        for rel_data in relationships:
            try:
                # Make sure both source and target nodes exist
                source_id = rel_data["source_id"]
                target_id = rel_data["target_id"]

                if self.dependency_graph.get_node(source_id) and self.dependency_graph.get_node(
                    target_id
                ):
                    self.dependency_graph.add_edge(
                        source_id=source_id,
                        target_id=target_id,
                        relationship_type=rel_data["relationship_type"],
                        weight=rel_data["metadata"].get("confidence", 1.0),
                        metadata=rel_data["metadata"],
                    )
                    successful_relationships += 1
            except ValueError as e:
                logger.warning(f"Failed to add relationship: {e}")

        # Connect parent document to all chunks
        for chunk in chunks:
            try:
                # Parent to chunk relationship
                self.dependency_graph.add_edge(
                    source_id=parent_doc.id,
                    target_id=chunk.id,
                    relationship_type="contains",
                    weight=1.0,
                    metadata={"confidence": 1.0},
                )
                successful_relationships += 1

                # Chunk to parent relationship
                self.dependency_graph.add_edge(
                    source_id=chunk.id,
                    target_id=parent_doc.id,
                    relationship_type="part_of",
                    weight=1.0,
                    metadata={"confidence": 1.0},
                )
                successful_relationships += 1
            except ValueError as e:
                logger.warning(f"Failed to add relationship: {e}")

        logger.info(f"Added {successful_relationships} relationships to dependency graph")

        return parent_doc, chunks

    def extract_entities_from_paper(self, chunks: list[Document]) -> list[Entity]:
        """
        Extract entities from paper chunks.

        Args:
        ----
            chunks: List of document chunks

        Returns:
        -------
            List[Entity]: List of extracted entities

        """
        entities = []

        # Extract entities from each chunk using the memory store's indexer
        for chunk in chunks:
            try:
                indexing_result = self.memory_store.indexer.index_document(chunk)
                entities.extend(indexing_result.entities)

                # Add entity relationships to dependency graph
                for relationship in indexing_result.relationships:
                    try:
                        self.dependency_graph.add_relationship(relationship)
                    except ValueError as e:
                        logger.warning(f"Failed to add relationship: {e}")
            except Exception as e:
                logger.warning(f"Failed to extract entities from chunk {chunk.id}: {e}")

        return entities

    def get_gasa_context(
        self, query: str, max_chunks: int = 5, max_hops: int = 2
    ) -> list[Document]:
        """
        Get GASA-optimized context for a query.

        Args:
        ----
            query: Query string
            max_chunks: Maximum number of chunks to return
            max_hops: Maximum number of hops in the graph

        Returns:
        -------
            List[Document]: List of document chunks for context

        """
        # Check if memory store is empty
        all_docs = self.memory_store.vector_store.list()
        if not all_docs:
            logger.warning("Memory store is empty, no documents to retrieve")
            return []

        # Embed the query
        query_embedding = self.embedding_retriever.embed_query(query)

        # Get initial results from vector search
        initial_results = self.memory_store.search(
            query_embedding=query_embedding, limit=max_chunks, include_graph_results=False
        )

        # If no results from vector search, return some documents from the memory store
        if not initial_results:
            logger.warning("No results from vector search, returning random documents")
            return all_docs[:max_chunks]

        # Extract document IDs
        doc_ids = [doc.id for doc, _ in initial_results]

        # Check if dependency graph has nodes
        if len(self.dependency_graph.nodes) == 0:
            logger.warning("Dependency graph is empty, using vector search results only")
            return [doc for doc, _ in initial_results]

        # Expand using graph
        if doc_ids:
            try:
                # First check if the nodes exist in the graph
                existing_nodes = [
                    node_id for node_id in doc_ids if node_id in self.dependency_graph.nodes
                ]

                if not existing_nodes:
                    logger.warning(
                        "None of the retrieved documents exist in the graph, using vector search results only"
                    )
                    return [doc for doc, _ in initial_results]

                # Get subgraph centered on the initial results
                subgraph = self.dependency_graph.get_subgraph(
                    node_ids=existing_nodes, max_hops=max_hops
                )

                # Get additional documents from the subgraph
                expanded_docs = set(doc_ids)
                additional_docs = []

                for node_id in subgraph.nodes:
                    if node_id not in expanded_docs:
                        node = self.dependency_graph.get_node(node_id)
                        # Check if it's a DocumentNode which has a document attribute
                        if node and isinstance(node, DocumentNode) and hasattr(node, "document"):
                            additional_docs.append(node.document)
                            expanded_docs.add(node_id)

                # Combine initial and additional results
                all_docs = [doc for doc, _ in initial_results] + additional_docs
                return all_docs[:max_chunks]
            except Exception as e:
                logger.warning(f"Failed to expand results using graph: {e}")
                return [doc for doc, _ in initial_results]
        else:
            return [doc for doc, _ in initial_results]
