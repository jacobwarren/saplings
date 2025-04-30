"""
Tests for the plugin entrypoints.

This module tests that the plugin entrypoints are correctly registered and can be loaded.
"""

import pytest

pytestmark = pytest.mark.nocov

from saplings.core.plugin import PluginType, discover_plugins, get_plugins_by_type
from saplings.memory.config import MemoryConfig, PrivacyLevel, VectorStoreType
from saplings.memory.vector_store import get_vector_store
from saplings.memory.indexer import get_indexer


class TestPluginEntrypoints:
    """Tests for the plugin entrypoints."""

    def setup_method(self):
        """Set up test environment."""
        # Discover plugins
        discover_plugins()

    def test_memory_store_plugins(self):
        """Test that memory store plugins are registered."""
        # Get all memory store plugins
        memory_store_plugins = get_plugins_by_type(PluginType.MEMORY_STORE)

        # Check that the secure memory store plugin is registered
        assert "secure_memory_store" in memory_store_plugins

    def test_validator_plugins(self):
        """Test that validator plugins are registered."""
        # Get all validator plugins
        validator_plugins = get_plugins_by_type(PluginType.VALIDATOR)

        # Check that the code validator plugin is registered
        assert "code_validator" in validator_plugins

        # Check that the factual validator plugin is registered
        assert "factual_validator" in validator_plugins

    def test_indexer_plugins(self):
        """Test that indexer plugins are registered."""
        # Get all indexer plugins
        indexer_plugins = get_plugins_by_type(PluginType.INDEXER)

        # Check that the code indexer plugin is registered
        assert "code_indexer" in indexer_plugins

    def test_get_secure_memory_store(self):
        """Test getting a secure memory store."""
        # Create a configuration for a secure memory store
        config = MemoryConfig(
            vector_store={"store_type": VectorStoreType.CUSTOM},
            secure_store={"privacy_level": PrivacyLevel.HASH_AND_DP},
        )

        # Get the vector store
        store = get_vector_store(config)

        # Check that we got a secure memory store
        assert store.__class__.__name__ == "SecureMemoryStore"

    def test_get_code_indexer(self):
        """Test getting a code indexer."""
        # Get the code indexer
        indexer = get_indexer("code")

        # Check that we got a code indexer
        assert indexer.__class__.__name__ == "CodeIndexer"
