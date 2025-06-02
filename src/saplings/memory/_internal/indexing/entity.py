from __future__ import annotations

"""
Entity module for Saplings memory.

This module defines the Entity class for representing entities extracted from documents.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Entity:
    """Entity extracted from a document."""

    name: str
    """Name of the entity."""

    entity_type: str
    """Type of the entity."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata for the entity."""

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the entity to a dictionary.

        Returns
        -------
            Dict[str, Any]: Dictionary representation

        """
        return {
            "name": self.name,
            "entity_type": self.entity_type,
            "metadata": self.metadata,
        }
