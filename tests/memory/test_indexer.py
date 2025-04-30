"""
Tests for the indexer module.
"""

import pytest

from saplings.memory.config import MemoryConfig
from saplings.memory.document import Document, DocumentMetadata
from saplings.memory.indexer import (
    Entity,
    Indexer,
    IndexerRegistry,
    IndexingResult,
    Relationship,
    SimpleIndexer,
    get_indexer,
)


class TestEntity:
    """Tests for the Entity class."""

    def test_create_entity(self):
        """Test creating an entity."""
        entity = Entity(
            name="John Doe",
            entity_type="person",
            metadata={"confidence": 0.9},
        )

        assert entity.name == "John Doe"
        assert entity.entity_type == "person"
        assert entity.metadata == {"confidence": 0.9}

    def test_entity_equality(self):
        """Test entity equality."""
        entity1 = Entity(name="John Doe", entity_type="person")
        entity2 = Entity(name="John Doe", entity_type="person")
        entity3 = Entity(name="John Doe", entity_type="organization")

        assert entity1 == entity2
        assert entity1 != entity3
        assert hash(entity1) == hash(entity2)
        assert hash(entity1) != hash(entity3)

    def test_to_dict_and_from_dict(self):
        """Test converting an entity to and from a dictionary."""
        entity = Entity(
            name="John Doe",
            entity_type="person",
            metadata={"confidence": 0.9},
        )

        # Convert to dictionary
        entity_dict = entity.to_dict()

        assert entity_dict["name"] == "John Doe"
        assert entity_dict["entity_type"] == "person"
        assert entity_dict["metadata"] == {"confidence": 0.9}

        # Convert back to entity
        new_entity = Entity.from_dict(entity_dict)

        assert new_entity.name == entity.name
        assert new_entity.entity_type == entity.entity_type
        assert new_entity.metadata == entity.metadata


class TestRelationship:
    """Tests for the Relationship class."""

    def test_create_relationship(self):
        """Test creating a relationship."""
        relationship = Relationship(
            source_id="entity:person:John Doe",
            target_id="entity:organization:Acme Corp",
            relationship_type="works_for",
            metadata={"confidence": 0.8},
        )

        assert relationship.source_id == "entity:person:John Doe"
        assert relationship.target_id == "entity:organization:Acme Corp"
        assert relationship.relationship_type == "works_for"
        assert relationship.metadata == {"confidence": 0.8}

    def test_relationship_equality(self):
        """Test relationship equality."""
        rel1 = Relationship(
            source_id="entity:person:John Doe",
            target_id="entity:organization:Acme Corp",
            relationship_type="works_for",
        )
        rel2 = Relationship(
            source_id="entity:person:John Doe",
            target_id="entity:organization:Acme Corp",
            relationship_type="works_for",
        )
        rel3 = Relationship(
            source_id="entity:person:John Doe",
            target_id="entity:organization:Acme Corp",
            relationship_type="founded",
        )

        assert rel1 == rel2
        assert rel1 != rel3
        assert hash(rel1) == hash(rel2)
        assert hash(rel1) != hash(rel3)

    def test_to_dict_and_from_dict(self):
        """Test converting a relationship to and from a dictionary."""
        relationship = Relationship(
            source_id="entity:person:John Doe",
            target_id="entity:organization:Acme Corp",
            relationship_type="works_for",
            metadata={"confidence": 0.8},
        )

        # Convert to dictionary
        rel_dict = relationship.to_dict()

        assert rel_dict["source_id"] == "entity:person:John Doe"
        assert rel_dict["target_id"] == "entity:organization:Acme Corp"
        assert rel_dict["relationship_type"] == "works_for"
        assert rel_dict["metadata"] == {"confidence": 0.8}

        # Convert back to relationship
        new_rel = Relationship.from_dict(rel_dict)

        assert new_rel.source_id == relationship.source_id
        assert new_rel.target_id == relationship.target_id
        assert new_rel.relationship_type == relationship.relationship_type
        assert new_rel.metadata == relationship.metadata


class TestIndexingResult:
    """Tests for the IndexingResult class."""

    def test_create_indexing_result(self):
        """Test creating an indexing result."""
        result = IndexingResult(document_id="doc1")

        assert result.document_id == "doc1"
        assert result.entities == []
        assert result.relationships == []

    def test_add_entity(self):
        """Test adding an entity to an indexing result."""
        result = IndexingResult(document_id="doc1")
        entity = Entity(name="John Doe", entity_type="person")

        result.add_entity(entity)

        assert len(result.entities) == 1
        assert result.entities[0] == entity

        # Adding the same entity again should not duplicate it
        result.add_entity(entity)

        assert len(result.entities) == 1

    def test_add_relationship(self):
        """Test adding a relationship to an indexing result."""
        result = IndexingResult(document_id="doc1")
        relationship = Relationship(
            source_id="entity:person:John Doe",
            target_id="entity:organization:Acme Corp",
            relationship_type="works_for",
        )

        result.add_relationship(relationship)

        assert len(result.relationships) == 1
        assert result.relationships[0] == relationship

        # Adding the same relationship again should not duplicate it
        result.add_relationship(relationship)

        assert len(result.relationships) == 1

    def test_merge(self):
        """Test merging two indexing results."""
        result1 = IndexingResult(document_id="doc1")
        result1.add_entity(Entity(name="John Doe", entity_type="person"))
        result1.add_relationship(
            Relationship(
                source_id="entity:person:John Doe",
                target_id="entity:organization:Acme Corp",
                relationship_type="works_for",
            )
        )

        result2 = IndexingResult(document_id="doc2")
        result2.add_entity(Entity(name="Jane Smith", entity_type="person"))
        result2.add_relationship(
            Relationship(
                source_id="entity:person:Jane Smith",
                target_id="entity:organization:Acme Corp",
                relationship_type="works_for",
            )
        )

        # Merge result2 into result1
        result1.merge(result2)

        assert len(result1.entities) == 2
        assert len(result1.relationships) == 2

    def test_to_dict_and_from_dict(self):
        """Test converting an indexing result to and from a dictionary."""
        result = IndexingResult(document_id="doc1")
        result.add_entity(Entity(name="John Doe", entity_type="person"))
        result.add_relationship(
            Relationship(
                source_id="entity:person:John Doe",
                target_id="entity:organization:Acme Corp",
                relationship_type="works_for",
            )
        )

        # Convert to dictionary
        result_dict = result.to_dict()

        assert result_dict["document_id"] == "doc1"
        assert len(result_dict["entities"]) == 1
        assert len(result_dict["relationships"]) == 1

        # Convert back to indexing result
        new_result = IndexingResult.from_dict(result_dict)

        assert new_result.document_id == result.document_id
        assert len(new_result.entities) == 1
        assert len(new_result.relationships) == 1
        assert new_result.entities[0].name == "John Doe"
        assert new_result.entities[0].entity_type == "person"
        assert new_result.relationships[0].source_id == "entity:person:John Doe"
        assert new_result.relationships[0].target_id == "entity:organization:Acme Corp"
        assert new_result.relationships[0].relationship_type == "works_for"


class TestSimpleIndexer:
    """Tests for the SimpleIndexer class."""

    def setup_method(self):
        """Set up test environment."""
        self.config = MemoryConfig()
        self.indexer = SimpleIndexer(self.config)

    def test_extract_entities(self):
        """Test extracting entities from a document."""
        # Create a document with known entities
        document = Document(
            id="doc1",
            content="John Doe works for Google in New York. He is an expert in machine learning.",
            metadata=DocumentMetadata(source="test.txt"),
        )

        # Extract entities
        entities = self.indexer.extract_entities(document)

        # Check that the expected entities were extracted
        entity_info = [(e.name, e.entity_type) for e in entities]
        assert ("John", "person") in entity_info
        assert ("Google", "organization") in entity_info
        assert ("New York", "location") in entity_info
        assert ("machine learning", "concept") in entity_info

    def test_extract_relationships(self):
        """Test extracting relationships from a document."""
        # Create a document with known relationships
        document = Document(
            id="doc1",
            content="John Doe works for Google in New York. Alice knows Bob.",
            metadata=DocumentMetadata(source="test.txt"),
        )

        # Extract entities first
        entities = self.indexer.extract_entities(document)

        # Extract relationships
        relationships = self.indexer.extract_relationships(document, entities)

        # Check that the expected relationships were extracted
        rel_info = [(r.source_id, r.relationship_type, r.target_id) for r in relationships]

        # The exact relationships extracted will depend on the implementation details
        # of the SimpleIndexer, but we can check for some expected patterns
        person_works_for_org = False
        person_knows_person = False

        for source, rel_type, target in rel_info:
            if "person" in source and rel_type == "works_for" and "organization" in target:
                person_works_for_org = True
            elif "person" in source and rel_type == "knows" and "person" in target:
                person_knows_person = True

        assert person_works_for_org or person_knows_person

    def test_index_document(self):
        """Test indexing a document."""
        # Create a document
        document = Document(
            id="doc1",
            content="John Doe works for Google in New York. Alice knows Bob.",
            metadata=DocumentMetadata(source="test.txt"),
        )

        # Index the document
        result = self.indexer.index_document(document)

        # Check the indexing result
        assert result.document_id == "doc1"
        assert len(result.entities) > 0

        # Check that document-entity relationships were created
        doc_entity_rels = [
            r for r in result.relationships
            if r.source_id == "doc1" and r.relationship_type == "mentions"
        ]
        assert len(doc_entity_rels) > 0


class TestIndexerRegistry:
    """Tests for the IndexerRegistry class."""

    def setup_method(self):
        """Set up test environment."""
        # Clear the registry before each test
        IndexerRegistry()._indexers = {}

    def test_register_and_get_indexer(self):
        """Test registering and getting an indexer."""
        registry = IndexerRegistry()

        # Register the SimpleIndexer
        registry.register_indexer("test_indexer", SimpleIndexer)

        # Get the indexer
        indexer = registry.get_indexer("test_indexer")

        assert isinstance(indexer, SimpleIndexer)

        # Try to get a non-existent indexer
        with pytest.raises(ValueError):
            registry.get_indexer("non_existent")

    def test_list_indexers(self):
        """Test listing indexers."""
        registry = IndexerRegistry()

        # Register some indexers
        registry.register_indexer("test_indexer1", SimpleIndexer)
        registry.register_indexer("test_indexer2", SimpleIndexer)

        # List indexers
        indexers = registry.list_indexers()

        assert len(indexers) == 2
        assert "test_indexer1" in indexers
        assert "test_indexer2" in indexers


class TestGetIndexer:
    """Tests for the get_indexer function."""

    def setup_method(self):
        """Set up test environment."""
        # Register the SimpleIndexer for testing
        IndexerRegistry()._indexers = {}
        IndexerRegistry().register_indexer("simple", SimpleIndexer)

    def test_get_default_indexer(self):
        """Test getting the default indexer."""
        indexer = get_indexer()

        assert isinstance(indexer, SimpleIndexer)

    def test_get_specific_indexer(self):
        """Test getting a specific indexer."""
        # Register a test indexer
        IndexerRegistry().register_indexer("test_indexer", SimpleIndexer)

        # Get the indexer
        indexer = get_indexer("test_indexer")

        assert isinstance(indexer, SimpleIndexer)

    def test_get_with_config(self):
        """Test getting an indexer with a specific configuration."""
        config = MemoryConfig(
            graph={"entity_types": ["person", "organization"]}
        )

        indexer = get_indexer(config=config)

        assert isinstance(indexer, SimpleIndexer)
        assert indexer.config.graph.entity_types == ["person", "organization"]
