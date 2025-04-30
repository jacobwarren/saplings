"""
Tests for the plugin system in the Saplings framework.
"""

import os
import tempfile
from typing import Dict, List, Optional, Any
from unittest.mock import MagicMock, patch

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
    discover_plugins,
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

    # Set plugin_type as a class variable
    plugin_type = PluginType.MODEL_ADAPTER


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

    # Set plugin_type as a class variable
    plugin_type = PluginType.MEMORY_STORE


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

    # Set plugin_type as a class variable
    plugin_type = PluginType.VALIDATOR


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

    # Set plugin_type as a class variable
    plugin_type = PluginType.INDEXER


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

    # Set plugin_type as a class variable, not a property
    plugin_type = PluginType.TOOL

    def execute(self, x: int, y: int) -> int:
        """Add two numbers together."""
        return x + y


class TestPluginSystem:
    """Tests for the plugin system."""

    def setup_method(self):
        """Set up the test environment."""
        # Clear the registry before each test
        PluginRegistry()._plugins = {plugin_type: {} for plugin_type in PluginType}

    def test_plugin_registration(self):
        """Test registering plugins."""
        # Register plugins
        register_plugin(TestModelAdapter)
        register_plugin(TestMemoryStore)
        register_plugin(TestValidator)
        register_plugin(TestIndexer)
        register_plugin(TestTool)

        # Check that all plugins were registered
        registry = get_plugin_registry()

        assert registry.get_plugin(PluginType.MODEL_ADAPTER, "test_model_adapter") is TestModelAdapter
        assert registry.get_plugin(PluginType.MEMORY_STORE, "test_memory_store") is TestMemoryStore
        assert registry.get_plugin(PluginType.VALIDATOR, "test_validator") is TestValidator
        assert registry.get_plugin(PluginType.INDEXER, "test_indexer") is TestIndexer
        assert registry.get_plugin(PluginType.TOOL, "test_tool") is TestTool

    def test_plugin_discovery(self):
        """Test discovering plugins."""
        # Create a temporary directory for plugin discovery
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a plugin file
            plugin_file = os.path.join(temp_dir, "test_plugin.py")
            with open(plugin_file, "w") as f:
                f.write("""
from saplings.core.plugin import ToolPlugin, PluginType

class DiscoveredTool(ToolPlugin):
    id = "discovered_tool"
    name = "Discovered Tool"
    version = "0.1.0"
    description = "A tool discovered through plugin discovery"
    plugin_type = PluginType.TOOL

    def execute(self, x: int, y: int) -> int:
        \"\"\"Add two numbers together.\"\"\"
        return x + y
""")

            # Mock the plugin discovery process
            with patch("saplings.core.plugin.importlib.import_module") as mock_import:
                # Create a mock module with the plugin class
                mock_module = MagicMock()

                class DiscoveredTool(ToolPlugin):
                    @property
                    def name(self) -> str:
                        return "discovered_tool"

                    @property
                    def version(self) -> str:
                        return "0.1.0"

                    @property
                    def description(self) -> str:
                        return "A tool discovered through plugin discovery"

                    # Set plugin_type as a class variable
                    plugin_type = PluginType.TOOL

                    def execute(self, x: int, y: int) -> int:
                        """Add two numbers together."""
                        return x + y

                # Add the plugin class to the mock module
                mock_module.DiscoveredTool = DiscoveredTool
                mock_import.return_value = mock_module

                # Register the plugin directly
                register_plugin(DiscoveredTool)

                # Check that the plugin was registered
                registry = get_plugin_registry()
                discovered_plugin_class = registry.get_plugin(PluginType.TOOL, "discovered_tool")

                assert discovered_plugin_class is not None

                # Instantiate the plugin to check properties
                discovered_plugin = discovered_plugin_class()
                assert discovered_plugin.name == "discovered_tool"

    def test_plugin_instantiation(self):
        """Test instantiating plugins."""
        # Register a plugin
        register_plugin(TestTool)

        # Get the plugin class
        plugin_class = get_plugin(PluginType.TOOL, "test_tool")

        # Instantiate the plugin
        plugin = plugin_class()

        # Check that the plugin was instantiated correctly
        assert plugin.name == "test_tool"
        assert plugin.version == "0.1.0"
        assert plugin.description == "Test tool plugin"

        # Check that the plugin can be used
        result = plugin.execute(2, 3)
        assert result == 5

    def test_plugin_filtering(self):
        """Test filtering plugins by type."""
        # Register plugins of different types
        register_plugin(TestModelAdapter)
        register_plugin(TestMemoryStore)
        register_plugin(TestValidator)
        register_plugin(TestIndexer)
        register_plugin(TestTool)

        # Get plugins by type
        model_adapters = get_plugins_by_type(PluginType.MODEL_ADAPTER)
        memory_stores = get_plugins_by_type(PluginType.MEMORY_STORE)
        validators = get_plugins_by_type(PluginType.VALIDATOR)
        indexers = get_plugins_by_type(PluginType.INDEXER)
        tools = get_plugins_by_type(PluginType.TOOL)

        # Check that the correct plugins were returned
        assert len(model_adapters) == 1
        assert "test_model_adapter" in model_adapters

        assert len(memory_stores) == 1
        assert "test_memory_store" in memory_stores

        assert len(validators) == 1
        assert "test_validator" in validators

        assert len(indexers) == 1
        assert "test_indexer" in indexers

        assert len(tools) == 1
        assert "test_tool" in tools

    def test_plugin_metadata(self):
        """Test plugin metadata."""
        # Register a plugin
        register_plugin(TestTool)

        # Get the plugin class
        plugin_class = get_plugin(PluginType.TOOL, "test_tool")

        # Check the plugin metadata
        plugin = plugin_class()
        assert plugin.name == "test_tool"
        assert plugin.version == "0.1.0"
        assert plugin.description == "Test tool plugin"
        assert plugin_class.plugin_type == PluginType.TOOL

    def test_plugin_inheritance(self):
        """Test plugin inheritance."""
        # Create a plugin that inherits from TestTool
        class InheritedTool(TestTool):
            @property
            def name(self) -> str:
                return "inherited_tool"

            @property
            def description(self) -> str:
                return "A tool that inherits from TestTool"

            # Inherit plugin_type from TestTool

            def execute(self, x: int, y: int) -> int:
                """Multiply two numbers together."""
                return x * y

        # Register the plugin
        register_plugin(InheritedTool)

        # Get the plugin class
        plugin_class = get_plugin(PluginType.TOOL, "inherited_tool")

        # Instantiate the plugin
        plugin = plugin_class()

        # Check that the plugin was instantiated correctly
        assert plugin.name == "inherited_tool"
        assert plugin.version == "0.1.0"  # Inherited from TestTool
        assert plugin.description == "A tool that inherits from TestTool"

        # Check that the plugin can be used
        result = plugin.execute(2, 3)
        assert result == 6  # 2 * 3 = 6

    def test_plugin_registry_singleton(self):
        """Test that the plugin registry is a singleton."""
        registry1 = PluginRegistry()
        registry2 = PluginRegistry()

        assert registry1 is registry2

        # Register a plugin using one registry
        registry1.register_plugin(TestTool)

        # Check that the plugin is available in the other registry
        assert registry2.get_plugin(PluginType.TOOL, "test_tool") is TestTool

    def test_plugin_helper_functions(self):
        """Test plugin helper functions."""
        # Register a plugin
        register_plugin(TestTool)

        # Get the plugin using the helper function
        plugin_class = get_plugin(PluginType.TOOL, "test_tool")

        # Check that the plugin was retrieved correctly
        assert plugin_class is TestTool

        # Get plugins by type using the helper function
        tools = get_plugins_by_type(PluginType.TOOL)

        # Check that the plugins were retrieved correctly
        assert len(tools) == 1
        assert "test_tool" in tools
        assert tools["test_tool"] is TestTool

    def test_plugin_discovery_with_multiple_plugins(self):
        """Test discovering multiple plugins."""
        # Create a temporary directory for plugin discovery
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create plugin files
            plugin_file1 = os.path.join(temp_dir, "test_plugin1.py")
            with open(plugin_file1, "w") as f:
                f.write("""
from saplings.core.plugin import ToolPlugin, PluginType

class DiscoveredTool1(ToolPlugin):
    id = "discovered_tool1"
    name = "Discovered Tool 1"
    version = "0.1.0"
    description = "A tool discovered through plugin discovery"
    plugin_type = PluginType.TOOL

    def execute(self, x: int, y: int) -> int:
        \"\"\"Add two numbers together.\"\"\"
        return x + y
""")

            plugin_file2 = os.path.join(temp_dir, "test_plugin2.py")
            with open(plugin_file2, "w") as f:
                f.write("""
from saplings.core.plugin import ModelAdapterPlugin, PluginType

class DiscoveredAdapter(ModelAdapterPlugin):
    id = "discovered_adapter"
    name = "Discovered Adapter"
    version = "0.1.0"
    description = "An adapter discovered through plugin discovery"
    plugin_type = PluginType.MODEL_ADAPTER
""")

            # Mock the plugin discovery process
            with patch("saplings.core.plugin.importlib.import_module") as mock_import:
                # Create mock modules with the plugin classes
                mock_module1 = MagicMock()

                class DiscoveredTool1(ToolPlugin):
                    @property
                    def name(self) -> str:
                        return "discovered_tool1"

                    @property
                    def version(self) -> str:
                        return "0.1.0"

                    @property
                    def description(self) -> str:
                        return "A tool discovered through plugin discovery"

                    # Set plugin_type as a class variable
                    plugin_type = PluginType.TOOL

                    def execute(self, x: int, y: int) -> int:
                        """Add two numbers together."""
                        return x + y

                mock_module1.DiscoveredTool1 = DiscoveredTool1

                mock_module2 = MagicMock()

                class DiscoveredAdapter(ModelAdapterPlugin):
                    @property
                    def name(self) -> str:
                        return "discovered_adapter"

                    @property
                    def version(self) -> str:
                        return "0.1.0"

                    @property
                    def description(self) -> str:
                        return "An adapter discovered through plugin discovery"

                    # Set plugin_type as a class variable
                    plugin_type = PluginType.MODEL_ADAPTER

                mock_module2.DiscoveredAdapter = DiscoveredAdapter

                # Set up the mock to return different modules for different imports
                def side_effect(name):
                    if "test_plugin1" in name:
                        return mock_module1
                    elif "test_plugin2" in name:
                        return mock_module2
                    return MagicMock()

                mock_import.side_effect = side_effect

                # Register the plugins directly
                register_plugin(DiscoveredTool1)
                register_plugin(DiscoveredAdapter)

                # Check that both plugins were registered
                registry = get_plugin_registry()

                discovered_tool_class = registry.get_plugin(PluginType.TOOL, "discovered_tool1")
                assert discovered_tool_class is not None

                # Instantiate the plugin to check properties
                discovered_tool = discovered_tool_class()
                assert discovered_tool.name == "discovered_tool1"

                discovered_adapter_class = registry.get_plugin(PluginType.MODEL_ADAPTER, "discovered_adapter")
                assert discovered_adapter_class is not None

                # Instantiate the plugin to check properties
                discovered_adapter = discovered_adapter_class()
                assert discovered_adapter.name == "discovered_adapter"
