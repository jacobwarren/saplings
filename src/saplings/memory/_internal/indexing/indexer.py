from __future__ import annotations

"""
Indexer module for Saplings memory.

This module defines the Indexer abstract base class for document indexers.
"""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

# Import from other internal modules
from saplings.memory._internal.config import MemoryConfig
from saplings.memory._internal.graph.relationship import Relationship
from saplings.memory._internal.indexing.entity import Entity
from saplings.memory._internal.indexing.indexing_result import IndexingResult

if TYPE_CHECKING:
    from saplings.memory._internal.document import Document

# Configure logging
logger = logging.getLogger(__name__)


class Indexer(ABC):
    """
    Abstract base class for document indexers.

    An indexer is responsible for extracting entities and relationships from documents
    and building a knowledge graph.
    """

    def __init__(self, config: MemoryConfig | None = None) -> None:
        """
        Initialize the indexer.

        Args:
        ----
            config: Memory configuration

        """
        self.config = config or MemoryConfig.default()

    @abstractmethod
    def extract_entities(self, document: "Document") -> list[Entity]:
        """
        Extract entities from a document.

        Args:
        ----
            document: Document to extract entities from

        Returns:
        -------
            List[Entity]: Extracted entities

        """

    @abstractmethod
    def extract_relationships(
        self, document: "Document", entities: list[Entity]
    ) -> list[Relationship]:
        """
        Extract relationships from a document.

        Args:
        ----
            document: Document to extract relationships from
            entities: Entities extracted from the document

        Returns:
        -------
            List[Relationship]: Extracted relationships

        """

    def index_document(self, document: "Document") -> IndexingResult:
        """
        Index a document.

        Args:
        ----
            document: Document to index

        Returns:
        -------
            IndexingResult: Indexing result

        """
        result = IndexingResult(document_id=document.id)

        # Extract entities
        entities = self.extract_entities(document)
        for entity in entities:
            result.add_entity(entity)

            # Create relationship between document and entity
            relationship = Relationship(
                source_id=document.id,
                target_id=f"entity:{entity.entity_type}:{entity.name}",
                relationship_type="mentions",
                metadata={"confidence": 1.0},
            )
            result.add_relationship(relationship)

        # Extract relationships between entities
        relationships = self.extract_relationships(document, entities)
        for relationship in relationships:
            result.add_relationship(relationship)

        return result
