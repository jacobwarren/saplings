from __future__ import annotations

"""
Indexer module for Saplings memory.

This module defines the Indexer abstract base class and implementations.
"""


import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, cast

from saplings.memory.config import MemoryConfig
from saplings.memory.document import Document

if TYPE_CHECKING:
    from saplings.core.plugin import IndexerPlugin

logger = logging.getLogger(__name__)


class Entity:
    """
    Entity class for representing entities extracted from documents.

    An entity is a named object or concept mentioned in a document,
    such as a person, organization, location, or technical term.
    """

    def __init__(
        self,
        name: str,
        entity_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize an entity.

        Args:
        ----
            name: Name of the entity
            entity_type: Type of the entity (e.g., person, organization, location)
            metadata: Additional metadata about the entity

        """
        self.name = name
        self.entity_type = entity_type
        self.metadata = metadata or {}

    def __eq__(self, other: object) -> bool:
        """
        Check if two entities are equal.

        Args:
        ----
            other: Other entity

        Returns:
        -------
            bool: True if the entities are equal, False otherwise

        """
        if not isinstance(other, Entity):
            return False
        return self.name == other.name and self.entity_type == other.entity_type

    def __hash__(self):
        """
        Get the hash of the entity.

        Returns
        -------
            int: Hash value

        """
        return hash((self.name, self.entity_type))

    def __str__(self):
        """
        Get a string representation of the entity.

        Returns
        -------
            str: String representation

        """
        return f"{self.name} ({self.entity_type})"

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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Entity":
        """
        Create an entity from a dictionary.

        Args:
        ----
            data: Dictionary representation

        Returns:
        -------
            Entity: Entity instance

        """
        return cls(
            name=data["name"],
            entity_type=data["entity_type"],
            metadata=data.get("metadata", {}),
        )


class Relationship:
    """
    Relationship class for representing relationships between entities or documents.

    A relationship connects two nodes (entities or documents) with a specific type
    and optional metadata.
    """

    def __init__(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize a relationship.

        Args:
        ----
            source_id: ID of the source node
            target_id: ID of the target node
            relationship_type: Type of the relationship
            metadata: Additional metadata about the relationship

        """
        self.source_id = source_id
        self.target_id = target_id
        self.relationship_type = relationship_type
        self.metadata = metadata or {}

    def __eq__(self, other: object) -> bool:
        """
        Check if two relationships are equal.

        Args:
        ----
            other: Other relationship

        Returns:
        -------
            bool: True if the relationships are equal, False otherwise

        """
        if not isinstance(other, Relationship):
            return False
        return (
            self.source_id == other.source_id
            and self.target_id == other.target_id
            and self.relationship_type == other.relationship_type
        )

    def __hash__(self):
        """
        Get the hash of the relationship.

        Returns
        -------
            int: Hash value

        """
        return hash((self.source_id, self.target_id, self.relationship_type))

    def __str__(self):
        """
        Get a string representation of the relationship.

        Returns
        -------
            str: String representation

        """
        return f"{self.source_id} --[{self.relationship_type}]--> {self.target_id}"

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
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Relationship":
        """
        Create a relationship from a dictionary.

        Args:
        ----
            data: Dictionary representation

        Returns:
        -------
            Relationship: Relationship instance

        """
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            relationship_type=data["relationship_type"],
            metadata=data.get("metadata", {}),
        )


class IndexingResult:
    """
    Result of indexing a document.

    This class contains the entities and relationships extracted from a document.
    """

    def __init__(
        self,
        document_id: str,
        entities: list[Entity] | None = None,
        relationships: list[Relationship] | None = None,
    ) -> None:
        """
        Initialize an indexing result.

        Args:
        ----
            document_id: ID of the indexed document
            entities: Entities extracted from the document
            relationships: Relationships extracted from the document

        """
        self.document_id = document_id
        self.entities = entities or []
        self.relationships = relationships or []

    def add_entity(self, entity: Entity) -> None:
        """
        Add an entity to the result.

        Args:
        ----
            entity: Entity to add

        """
        if entity not in self.entities:
            self.entities.append(entity)

    def add_relationship(self, relationship: Relationship) -> None:
        """
        Add a relationship to the result.

        Args:
        ----
            relationship: Relationship to add

        """
        if relationship not in self.relationships:
            self.relationships.append(relationship)

    def merge(self, other: "IndexingResult") -> None:
        """
        Merge another indexing result into this one.

        Args:
        ----
            other: Other indexing result

        """
        for entity in other.entities:
            self.add_entity(entity)

        for relationship in other.relationships:
            self.add_relationship(relationship)

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
            "relationships": [rel.to_dict() for rel in self.relationships],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IndexingResult":
        """
        Create an indexing result from a dictionary.

        Args:
        ----
            data: Dictionary representation

        Returns:
        -------
            IndexingResult: Indexing result instance

        """
        result = cls(document_id=data["document_id"])

        for entity_data in data.get("entities", []):
            result.add_entity(Entity.from_dict(entity_data))

        for rel_data in data.get("relationships", []):
            result.add_relationship(Relationship.from_dict(rel_data))

        return result


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
    def extract_entities(self, document: Document) -> list[Entity]:
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
        self, document: Document, entities: list[Entity]
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

    def index_document(self, document: Document) -> IndexingResult:
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

        # Define entity patterns (simple keyword lists for demonstration)
        self.entity_patterns = {
            "person": [
                "Alice",
                "Bob",
                "Charlie",
                "David",
                "Eve",
                "John",
                "Jane",
                "Michael",
                "Sarah",
                "Tom",
            ],
            "organization": [
                "Google",
                "Microsoft",
                "Apple",
                "Amazon",
                "Facebook",
                "IBM",
                "Intel",
                "Oracle",
                "Tesla",
                "Twitter",
            ],
            "location": [
                "New York",
                "London",
                "Paris",
                "Tokyo",
                "Berlin",
                "San Francisco",
                "Beijing",
                "Moscow",
                "Sydney",
                "Toronto",
            ],
            "concept": [
                "algorithm",
                "database",
                "network",
                "security",
                "privacy",
                "machine learning",
                "artificial intelligence",
                "blockchain",
                "cloud computing",
                "data science",
            ],
        }

        # Define relationship patterns (simple rules for demonstration)
        self.relationship_patterns = [
            ("person", "works_for", "organization", ["works for", "employed by", "joined"]),
            ("person", "lives_in", "location", ["lives in", "resides in", "moved to"]),
            (
                "organization",
                "located_in",
                "location",
                ["located in", "headquartered in", "based in"],
            ),
            ("person", "knows", "person", ["knows", "met", "collaborated with"]),
        ]

    def extract_entities(self, document: Document) -> list[Entity]:
        """
        Extract entities from a document using simple string matching.

        Args:
        ----
            document: Document to extract entities from

        Returns:
        -------
            List[Entity]: Extracted entities

        """
        entities = []

        # Handle nested Document objects
        if isinstance(document.content, Document):
            content = document.content.content
        else:
            content = document.content

        # Convert to string and lowercase
        content = str(content).lower()

        # Only process entity types that are enabled in the configuration
        enabled_entity_types = self.config.graph.entity_types

        for entity_type, patterns in self.entity_patterns.items():
            if entity_type not in enabled_entity_types:
                continue

            for pattern in patterns:
                pattern_lower = pattern.lower()
                if pattern_lower in content:
                    entity = Entity(
                        name=pattern,
                        entity_type=entity_type,
                        metadata={
                            "source_document": document.id,
                            "confidence": 1.0,
                        },
                    )
                    entities.append(entity)

        return entities

    def extract_relationships(
        self, document: Document, entities: list[Entity]
    ) -> list[Relationship]:
        """
        Extract relationships from a document using simple pattern matching.

        Args:
        ----
            document: Document to extract relationships from
            entities: Entities extracted from the document

        Returns:
        -------
            List[Relationship]: Extracted relationships

        """
        relationships = []

        # Handle nested Document objects
        if isinstance(document.content, Document):
            content = document.content.content
        else:
            content = document.content

        # Convert to string and lowercase
        content = str(content).lower()

        # Create a dictionary of entities by type for easier lookup
        entities_by_type = {}
        for entity in entities:
            if entity.entity_type not in entities_by_type:
                entities_by_type[entity.entity_type] = []
            entities_by_type[entity.entity_type].append(entity)

        # Check for relationships based on patterns
        for source_type, rel_type, target_type, patterns in self.relationship_patterns:
            if source_type not in entities_by_type or target_type not in entities_by_type:
                continue

            for source_entity in entities_by_type[source_type]:
                for target_entity in entities_by_type[target_type]:
                    if source_entity == target_entity:
                        continue

                    # Check if any relationship pattern is present in the content
                    for pattern in patterns:
                        pattern_lower = pattern.lower()
                        if pattern_lower in content:
                            # Check if the pattern is between the source and target entities
                            source_pos = content.find(source_entity.name.lower())
                            target_pos = content.find(target_entity.name.lower())
                            pattern_pos = content.find(pattern_lower)

                            # Simple heuristic: pattern should be between the entities
                            # or close to them in the text
                            if (
                                (source_pos < pattern_pos < target_pos)
                                or (target_pos < pattern_pos < source_pos)
                                or (
                                    abs(source_pos - pattern_pos) < 50
                                    and abs(target_pos - pattern_pos) < 50
                                )
                            ):
                                relationship = Relationship(
                                    source_id=f"entity:{source_entity.entity_type}:{source_entity.name}",
                                    target_id=f"entity:{target_entity.entity_type}:{target_entity.name}",
                                    relationship_type=rel_type,
                                    metadata={
                                        "source_document": document.id,
                                        "confidence": 0.8,
                                        "pattern": pattern,
                                    },
                                )
                                relationships.append(relationship)

        return relationships


class IndexerRegistry:
    """
    Registry for indexers.

    This class manages the available indexers and provides methods to get
    indexers by name.
    """

    def __init__(self) -> None:
        """Initialize the indexer registry."""
        self._indexers = {}

    def register_indexer(self, name: str, indexer_class: type) -> None:
        """
        Register an indexer.

        Args:
        ----
            name: Name of the indexer
            indexer_class: Indexer class

        """
        if not issubclass(indexer_class, Indexer):
            msg = f"Indexer class must be a subclass of Indexer, got {indexer_class}"
            raise TypeError(msg)

        self._indexers[name] = indexer_class
        logger.info(f"Registered indexer: {name}")

    def get_indexer(self, name: str, config: MemoryConfig | None = None) -> Indexer:
        """
        Get an indexer by name.

        Args:
        ----
            name: Name of the indexer
            config: Memory configuration

        Returns:
        -------
            Indexer: Indexer instance

        Raises:
        ------
            ValueError: If the indexer is not found

        """
        if name not in self._indexers:
            msg = f"Indexer not found: {name}"
            raise ValueError(msg)

        indexer_class = self._indexers[name]
        return indexer_class(config)

    def list_indexers(self):
        """
        List available indexers.

        Returns
        -------
            List[str]: List of indexer names

        """
        return list(self._indexers.keys())


def get_indexer_registry():
    """
    Get the indexer registry.

    This function is maintained for backward compatibility.
    New code should use constructor injection via the DI container.

    Returns
    -------
        IndexerRegistry: Indexer registry from the DI container

    """
    try:
        from saplings.di import container

        return cast("IndexerRegistry", container.resolve(IndexerRegistry))
    except Exception as e:
        # If the container is not initialized yet, create a new registry
        # This is a fallback for tests and initialization
        logger.warning(f"Failed to resolve IndexerRegistry from container: {e}")
        return IndexerRegistry()


# Register the SimpleIndexer
def register_simple_indexer():
    """Register the SimpleIndexer."""
    try:
        registry = get_indexer_registry()
        registry.register_indexer("simple", SimpleIndexer)
    except Exception as e:
        # Log the error but don't fail - this will be retried later
        logger.warning(f"Failed to register SimpleIndexer: {e}")


# We'll register the SimpleIndexer lazily when it's first needed
# This avoids circular dependencies during initialization
_simple_indexer_registered = False


def get_indexer(name: str = "simple", config: MemoryConfig | None = None) -> Indexer:
    """
    Get an indexer by name.

    Args:
    ----
        name: Name of the indexer
        config: Memory configuration

    Returns:
    -------
        Indexer: Indexer instance

    """
    global _simple_indexer_registered

    # Register the SimpleIndexer if it hasn't been registered yet
    if not _simple_indexer_registered and name == "simple":
        register_simple_indexer()
        _simple_indexer_registered = True

    # First, try to get the indexer from the registry
    try:
        return get_indexer_registry().get_indexer(name, config)
    except ValueError:
        # If not found in the registry, check for plugin-based indexers
        try:
            from saplings.core.plugin import PluginType, get_plugins_by_type

            # Get all indexer plugins
            indexer_plugins = get_plugins_by_type(PluginType.INDEXER)

            # Look for a plugin with a matching name
            for plugin_name, plugin_class in indexer_plugins.items():
                if plugin_name.lower() == name.lower() or plugin_name.lower().replace(
                    "_", ""
                ) == name.lower().replace("_", ""):
                    # Create an instance of the plugin
                    # Cast to ensure it's an IndexerPlugin that inherits from Indexer
                    indexer_plugin = cast("type[IndexerPlugin]", plugin_class)
                    return cast("Indexer", indexer_plugin())

            # If we're looking for a code indexer, try to find a code indexer plugin
            if "code" in name.lower():
                for plugin_name, plugin_class in indexer_plugins.items():
                    if "code" in plugin_name.lower():
                        # Create an instance of the code indexer plugin
                        # Cast to ensure it's an IndexerPlugin that inherits from Indexer
                        indexer_plugin = cast("type[IndexerPlugin]", plugin_class)
                        return cast("Indexer", indexer_plugin())

        except ImportError:
            # Plugin system not available
            pass

        # If still not found, raise the original error
        msg = f"Indexer not found: {name}"
        raise ValueError(msg)
