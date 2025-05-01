"""
Tests for the plugin module.
"""

import pytest

from saplings.core.plugin import (
    IndexerPlugin,
    MemoryStorePlugin,
    ModelAdapterPlugin,
    Plugin,
    PluginRegistry,
    PluginType,
    ToolPlugin,
    ValidatorPlugin,
    get_plugin,
    get_plugin_registry,
    get_plugins_by_type,
    register_plugin,
)


# Test plugin implementations
class TestModelAdapter(ModelAdapterPlugin):
    """Test model adapter plugin."""

    @property
    def name(self) -> str:
        return "test_model_adapter"

    @property
    def version(self) -> str:
        return "0.1.0"

    @property
    def description(self) -> str:
        return "Test model adapter plugin"


class TestMemoryStore(MemoryStorePlugin):
    """Test memory store plugin."""

    @property
    def name(self) -> str:
        return "test_memory_store"

    @property
    def version(self) -> str:
        return "0.1.0"

    @property
    def description(self) -> str:
        return "Test memory store plugin"


class TestValidator(ValidatorPlugin):
    """Test validator plugin."""

    @property
    def name(self) -> str:
        return "test_validator"

    @property
    def version(self) -> str:
        return "0.1.0"

    @property
    def description(self) -> str:
        return "Test validator plugin"


class TestIndexer(IndexerPlugin):
    """Test indexer plugin."""

    @property
    def name(self) -> str:
        return "test_indexer"

    @property
    def version(self) -> str:
        return "0.1.0"

    @property
    def description(self) -> str:
        return "Test indexer plugin"


class TestTool(ToolPlugin):
    """Test tool plugin."""

    @property
    def name(self) -> str:
        return "test_tool"

    @property
    def version(self) -> str:
        return "0.1.0"

    @property
    def description(self) -> str:
        return "Test tool plugin"


class TestPluginRegistry:
    """Tests for the PluginRegistry class."""

    def setup_method(self):
        """Set up the test environment."""
        # Clear the registry before each test
        PluginRegistry()._plugins = {plugin_type: {} for plugin_type in PluginType}

    def test_singleton(self):
        """Test that PluginRegistry is a singleton."""
        registry1 = PluginRegistry()
        registry2 = PluginRegistry()
        assert registry1 is registry2

    def test_register_plugin(self):
        """Test registering a plugin."""
        registry = PluginRegistry()
        registry.register_plugin(TestModelAdapter)

        assert (
            registry.get_plugin(PluginType.MODEL_ADAPTER, "test_model_adapter") is TestModelAdapter
        )

    def test_get_plugin(self):
        """Test getting a plugin."""
        registry = PluginRegistry()
        registry.register_plugin(TestModelAdapter)

        plugin_class = registry.get_plugin(PluginType.MODEL_ADAPTER, "test_model_adapter")
        assert plugin_class is TestModelAdapter

        # Test getting a non-existent plugin
        assert registry.get_plugin(PluginType.MODEL_ADAPTER, "non_existent") is None

    def test_get_plugins_by_type(self):
        """Test getting plugins by type."""
        registry = PluginRegistry()
        registry.register_plugin(TestModelAdapter)
        registry.register_plugin(TestMemoryStore)

        model_adapters = registry.get_plugins_by_type(PluginType.MODEL_ADAPTER)
        assert len(model_adapters) == 1
        assert model_adapters["test_model_adapter"] is TestModelAdapter

        memory_stores = registry.get_plugins_by_type(PluginType.MEMORY_STORE)
        assert len(memory_stores) == 1
        assert memory_stores["test_memory_store"] is TestMemoryStore

        # Test getting plugins of a type with no plugins
        validators = registry.get_plugins_by_type(PluginType.VALIDATOR)
        assert len(validators) == 0

    def test_clear(self):
        """Test clearing the registry."""
        registry = PluginRegistry()
        registry.register_plugin(TestModelAdapter)
        registry.register_plugin(TestMemoryStore)

        registry.clear()

        assert len(registry.get_plugins_by_type(PluginType.MODEL_ADAPTER)) == 0
        assert len(registry.get_plugins_by_type(PluginType.MEMORY_STORE)) == 0


class TestPluginHelpers:
    """Tests for the plugin helper functions."""

    def setup_method(self):
        """Set up the test environment."""
        # Clear the registry before each test
        PluginRegistry()._plugins = {plugin_type: {} for plugin_type in PluginType}

    def test_get_plugin_registry(self):
        """Test getting the plugin registry."""
        registry = get_plugin_registry()
        assert isinstance(registry, PluginRegistry)

    def test_register_plugin(self):
        """Test registering a plugin."""
        register_plugin(TestModelAdapter)

        registry = get_plugin_registry()
        assert (
            registry.get_plugin(PluginType.MODEL_ADAPTER, "test_model_adapter") is TestModelAdapter
        )

    def test_get_plugin(self):
        """Test getting a plugin."""
        register_plugin(TestModelAdapter)

        plugin_class = get_plugin(PluginType.MODEL_ADAPTER, "test_model_adapter")
        assert plugin_class is TestModelAdapter

    def test_get_plugins_by_type(self):
        """Test getting plugins by type."""
        register_plugin(TestModelAdapter)
        register_plugin(TestMemoryStore)

        model_adapters = get_plugins_by_type(PluginType.MODEL_ADAPTER)
        assert len(model_adapters) == 1
        assert model_adapters["test_model_adapter"] is TestModelAdapter
