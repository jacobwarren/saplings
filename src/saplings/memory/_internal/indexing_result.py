from __future__ import annotations

"""
Indexing result module for Saplings memory.

This module defines the IndexingResult class for representing the result of indexing a document.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

# Import from other internal modules
if TYPE_CHECKING:
    from saplings.memory._internal.entity import Entity
    from saplings.memory._internal.relationship import Relationship


@dataclass
class IndexingResult:
    """Result of indexing a document."""

    document_id: str
    """ID of the indexed document."""

    entities: list["Entity"] = field(default_factory=list)
    """Entities extracted from the document."""

    relationships: list["Relationship"] = field(default_factory=list)
    """Relationships extracted from the document."""

    def add_entity(self, entity: "Entity") -> None:
        """
        Add an entity to the result.

        Args:
        ----
            entity: Entity to add

        """
        self.entities.append(entity)

    def add_relationship(self, relationship: "Relationship") -> None:
        """
        Add a relationship to the result.

        Args:
        ----
            relationship: Relationship to add

        """
        self.relationships.append(relationship)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the indexing result to a dictionary.

        Returns
        -------
            Dict[str, Any]: Dictionary representation

        """
        return {
            "document_id": self.document_id,
            "entities": [entity.to_dict() for entity in self.entities],
            "relationships": [relationship.to_dict() for relationship in self.relationships],
        }
