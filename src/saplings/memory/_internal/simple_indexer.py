from __future__ import annotations

"""
Simple indexer module for Saplings memory.

This module defines the SimpleIndexer class for basic entity and relationship extraction.
"""

import logging
from typing import TYPE_CHECKING

# Import from other internal modules
from saplings.memory._internal.config import MemoryConfig
from saplings.memory._internal.entity import Entity
from saplings.memory._internal.indexer import Indexer
from saplings.memory._internal.relationship import Relationship

if TYPE_CHECKING:
    from saplings.memory._internal.document_protocol import Document

# Configure logging
logger = logging.getLogger(__name__)


class SimpleIndexer(Indexer):
    """
    Simple implementation of the Indexer interface.

    This indexer uses basic string matching to extract entities and relationships.
    It's suitable for testing and simple use cases but not for production use.
    """

    def __init__(self, config: MemoryConfig | None = None) -> None:
        """
        Initialize the simple indexer.

        Args:
        ----
            config: Memory configuration

        """
        super().__init__(config)
        self.entity_types = ["person", "organization", "location", "concept"]

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
        # This is a very simple implementation that just extracts capitalized words
        # In a real implementation, you would use NLP techniques
        entities = []
        words = document.content.split()
        for word in words:
            # Skip short words and words that don't start with a capital letter
            if len(word) < 3 or not word[0].isupper():
                continue

            # Clean up the word
            clean_word = word.strip(".,;:!?()[]{}\"'")
            if not clean_word:
                continue

            # Determine the entity type (very simplistic)
            entity_type = "concept"  # Default
            if clean_word.endswith("Corp") or clean_word.endswith("Inc"):
                entity_type = "organization"
            elif clean_word in ["New York", "London", "Paris", "Tokyo"]:
                entity_type = "location"
            elif clean_word in ["John", "Jane", "Bob", "Alice"]:
                entity_type = "person"

            # Create the entity
            entity = Entity(
                name=clean_word,
                entity_type=entity_type,
                metadata={"source": document.id, "confidence": 0.8},
            )
            entities.append(entity)

        return entities

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
        # This is a very simple implementation that just creates relationships
        # between entities that appear close to each other in the text
        # In a real implementation, you would use NLP techniques
        relationships = []
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities):
                if i == j:
                    continue

                # Create a relationship if the entities are of different types
                if entity1.entity_type != entity2.entity_type:
                    relationship = Relationship(
                        source_id=f"entity:{entity1.entity_type}:{entity1.name}",
                        target_id=f"entity:{entity2.entity_type}:{entity2.name}",
                        relationship_type="related",
                        weight=0.5,
                        metadata={"source": document.id, "confidence": 0.5},
                    )
                    relationships.append(relationship)

        return relationships
