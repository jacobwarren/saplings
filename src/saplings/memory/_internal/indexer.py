from __future__ import annotations

"""
Indexer module for Saplings memory.

This module defines the Indexer abstract base class for document indexers.
"""

import logging
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

# Import from other internal modules
from saplings.memory._internal.config import MemoryConfig
from saplings.memory._internal.entity import Entity
from saplings.memory._internal.indexing_result import IndexingResult
from saplings.memory._internal.relationship import Relationship

if TYPE_CHECKING:
    from saplings.memory._internal.document_protocol import Document
    from saplings.memory._internal.entity import Entity

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
    def extract_entities(self, document: "Document") -> list["Entity"]:
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
        self, document: "Document", entities: list["Entity"]
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


class SimpleIndexer(Indexer):
    """
    Simple indexer implementation.

    This indexer uses basic text processing to extract entities and relationships.
    """

    def __init__(self, config: MemoryConfig | None = None) -> None:
        """
        Initialize the simple indexer.

        Args:
        ----
            config: Memory configuration

        """
        super().__init__(config)
        self._entity_patterns = {
            "person": r"\b[A-Z][a-z]+ [A-Z][a-z]+\b",  # Simple pattern for person names
            "organization": r"\b[A-Z][a-zA-Z]+ (Inc|Corp|LLC|Ltd)\b",  # Simple pattern for organizations
            "location": r"\b[A-Z][a-z]+ (City|Town|Village|County|State|Province|Country)\b",  # Simple pattern for locations
            "date": r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",  # Simple pattern for dates
        }

    def extract_entities(self, document: "Document") -> list["Entity"]:
        """
        Extract entities from a document.

        Args:
        ----
            document: Document to extract entities from

        Returns:
        -------
            List[Entity]: Extracted entities

        """
        entities = []
        text = document.content

        # Extract entities using patterns
        for entity_type, pattern in self._entity_patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                entity_text = match.group(0)
                entity = Entity(
                    name=entity_text,
                    entity_type=entity_type,
                    metadata={
                        "document_id": document.id,
                        "start_pos": match.start(),
                        "end_pos": match.end(),
                    },
                )
                entities.append(entity)

        return entities

    def extract_relationships(
        self, document: "Document", entities: list["Entity"]
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
        relationships = []

        # Create relationships between entities that are close to each other in the text
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities):
                if i != j:
                    # Get positions from metadata
                    pos1 = entity1.metadata.get("start_pos", 0)
                    pos2 = entity2.metadata.get("start_pos", 0)

                    # Calculate distance between entities
                    distance = abs(pos1 - pos2)

                    # Create relationship if entities are close
                    if distance < 100:  # Arbitrary threshold
                        relationship = Relationship(
                            source_id=f"entity:{entity1.entity_type}:{entity1.name}",
                            target_id=f"entity:{entity2.entity_type}:{entity2.name}",
                            relationship_type="related_to",
                            metadata={
                                "document_id": document.id,
                                "distance": distance,
                                "confidence": 1.0
                                - (distance / 100),  # Higher confidence for closer entities
                            },
                        )
                        relationships.append(relationship)

        return relationships
