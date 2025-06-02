from __future__ import annotations

"""
Relationship module for Saplings memory.

This module defines the Relationship class for representing relationships between entities or documents.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Relationship:
    """Relationship between entities or documents."""

    source_id: str
    """ID of the source entity or document."""

    target_id: str
    """ID of the target entity or document."""

    relationship_type: str
    """Type of the relationship."""

    weight: float = 1.0
    """Weight of the relationship."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata for the relationship."""

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the relationship to a dictionary.

        Returns
        -------
            Dict[str, Any]: Dictionary representation

        """
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relationship_type": self.relationship_type,
            "weight": self.weight,
            "metadata": self.metadata,
        }
