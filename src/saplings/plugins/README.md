# Saplings Plugins

This directory contains plugins for the Saplings framework. Plugins extend the functionality of Saplings by providing additional components that can be used by the framework.

## Available Plugins

### Memory Stores

- **SecureMemoryStore**: A memory store that provides privacy protection through hash-key protection and differential privacy noise.

### Validators

- **CodeValidator**: A validator that checks code outputs for syntax errors, security issues, and code quality.
- **FactualValidator**: A validator that checks outputs for factual accuracy against reference documents.

### Indexers

- **CodeIndexer**: An indexer that extracts entities and relationships from code files.

## Using Plugins

Plugins can be used in two ways:

1. **Direct usage**: Import the plugin class and use it directly.

```python
from saplings.plugins.memory_stores.secure_memory_store import SecureMemoryStore
from saplings.memory.config import MemoryConfig, PrivacyLevel

config = MemoryConfig(
    secure_store={"privacy_level": PrivacyLevel.HASH_AND_DP},
)
store = SecureMemoryStore(config)
```

2. **Plugin system**: Use the plugin system to get a plugin by name.

```python
from saplings.memory.config import MemoryConfig, PrivacyLevel, VectorStoreType
from saplings.memory.vector_store import get_vector_store

config = MemoryConfig(
    vector_store={"store_type": VectorStoreType.CUSTOM},
    secure_store={"privacy_level": PrivacyLevel.HASH_AND_DP},
)
store = get_vector_store(config)
```

## Creating Plugins

To create a new plugin, follow these steps:

1. Create a new Python file in the appropriate subdirectory (e.g., `memory_stores`, `validators`, `indexers`).
2. Define a class that inherits from the appropriate plugin base class (e.g., `MemoryStorePlugin`, `ValidatorPlugin`, `IndexerPlugin`).
3. Implement the required methods and properties.
4. Register the plugin in the `__init__.py` file of the subdirectory.

Example:

```python
from saplings.core.plugin import MemoryStorePlugin, PluginType
from saplings.memory.vector_store import VectorStore

class MyMemoryStore(MemoryStorePlugin, VectorStore):
    """My custom memory store."""

    @property
    def name(self) -> str:
        """Name of the plugin."""
        return "my_memory_store"

    @property
    def version(self) -> str:
        """Version of the plugin."""
        return "0.1.0"

    @property
    def description(self) -> str:
        """Description of the plugin."""
        return "My custom memory store"

    @property
    def plugin_type(self) -> PluginType:
        """Type of the plugin."""
        return PluginType.MEMORY_STORE

    # Implement the required methods...
```

Then, register the plugin in the `__init__.py` file:

```python
from saplings.core.plugin import RegistrationMode, register_plugin
from saplings.plugins.memory_stores.my_memory_store import MyMemoryStore

# Use SKIP mode to avoid warnings if the plugin is already registered
register_plugin(MyMemoryStore, mode=RegistrationMode.SKIP)
```

## Component-Specific Registries

For more advanced use cases, you can create isolated registries for different components:

```python
from saplings.core.plugin import PluginRegistryManager, register_plugin
from my_plugin import MyPlugin

# Create a registry manager
manager = PluginRegistryManager()

# Get component-specific registries
memory_registry = manager.get_registry("memory")

# Register plugins in specific registries
register_plugin(MyPlugin, registry=memory_registry, mode=RegistrationMode.SKIP)
```

This approach helps avoid plugin conflicts between different components.

## Plugin Entry Points

Plugins can also be registered using entry points in the `pyproject.toml` file:

```toml
[tool.poetry.plugins."saplings.memory_stores"]
my_memory_store = "my_package.my_module:MyMemoryStore"
```

This allows plugins to be discovered and loaded automatically when the Saplings framework is imported.
