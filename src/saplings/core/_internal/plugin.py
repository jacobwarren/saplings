from __future__ import annotations

"""
Plugin system for Saplings.

This module provides the plugin discovery and loading mechanism for Saplings.
"""


import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, Set, TypeVar

from importlib_metadata import entry_points

logger = logging.getLogger(__name__)


class PluginType(str, Enum):
    """Types of plugins supported by Saplings."""

    MODEL_ADAPTER = "model_adapter"
    MEMORY_STORE = "memory_store"
    VALIDATOR = "validator"
    INDEXER = "indexer"
    TOOL = "tool"


class RegistrationMode(str, Enum):
    """Plugin registration modes."""

    STRICT = "strict"  # Raise error if plugin already exists
    WARN = "warn"  # Log warning if plugin already exists (current behavior)
    SILENT = "silent"  # Silently overwrite existing plugin
    SKIP = "skip"  # Skip registration if plugin already exists


class Plugin(ABC):
    """Base class for all plugins."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the plugin."""

    @property
    @abstractmethod
    def version(self) -> str:
        """Version of the plugin."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of the plugin."""

    @property
    @abstractmethod
    def plugin_type(self) -> PluginType:
        """Type of the plugin."""


T = TypeVar("T", bound=Plugin)


class PluginRegistry:
    """Registry for plugins."""

    def __init__(self) -> None:
        """Initialize the plugin registry."""
        self._plugins: dict[PluginType, dict[str, type[Plugin]]] = {}
        for plugin_type in PluginType:
            self._plugins[plugin_type] = {}

    def register_plugin(
        self, plugin_class: type[Plugin], mode: RegistrationMode = RegistrationMode.WARN
    ) -> None:
        """
        Register a plugin class with configurable behavior for duplicates.

        Args:
        ----
            plugin_class: The plugin class to register
            mode: How to handle duplicate registrations

        """
        # Create a temporary instance to access property values
        temp_instance = plugin_class()
        plugin_type = temp_instance.plugin_type
        plugin_name = temp_instance.name

        # Check if plugin is already registered
        if plugin_name in self._plugins[plugin_type]:
            existing_plugin = self._plugins[plugin_type][plugin_name]

            # Handle based on mode
            if mode == RegistrationMode.STRICT:
                msg = f"Plugin {plugin_name} of type {plugin_type} is already registered"
                raise ValueError(msg)
            elif mode == RegistrationMode.WARN:
                logger.warning(
                    "Plugin %s of type %s is already registered. Overwriting.",
                    plugin_name,
                    plugin_type,
                )
            elif mode == RegistrationMode.SKIP:
                logger.debug(
                    "Plugin %s of type %s is already registered. Skipping.",
                    plugin_name,
                    plugin_type,
                )
                return
            # SILENT mode just continues without logging

        # Register the plugin
        if plugin_type not in self._plugins:
            self._plugins[plugin_type] = {}

        self._plugins[plugin_type][plugin_name] = plugin_class
        logger.debug("Registered plugin %s of type %s", plugin_name, plugin_type)

    def get_plugin(self, plugin_type: PluginType, plugin_name: str) -> type[Plugin] | None:
        """
        Get a plugin by type and name.

        Args:
        ----
            plugin_type: Type of the plugin
            plugin_name: Name of the plugin

        Returns:
        -------
            Optional[Type[Plugin]]: The plugin class if found, None otherwise

        """
        return self._plugins.get(plugin_type, {}).get(plugin_name)

    def get_plugins_by_type(self, plugin_type: PluginType) -> dict[str, type[Plugin]]:
        """
        Get all plugins of a specific type.

        Args:
        ----
            plugin_type: Type of plugins to get

        Returns:
        -------
            Dict[str, Type[Plugin]]: Dictionary of plugin name to plugin class

        """
        return self._plugins.get(plugin_type, {})

    def get_plugin_types(self) -> list[PluginType]:
        """
        Get all plugin types.

        Returns
        -------
            List[PluginType]: List of plugin types

        """
        return list(PluginType)

    def register_plugin_with_version(
        self,
        plugin_class: type[Plugin],
        version: str,
        mode: RegistrationMode = RegistrationMode.WARN,
    ) -> None:
        """
        Register a plugin with version information.

        Args:
        ----
            plugin_class: The plugin class to register
            version: Version string (e.g., "1.0.0")
            mode: How to handle duplicate registrations

        """
        # Create a temporary instance to access property values
        temp_instance = plugin_class()
        plugin_type = temp_instance.plugin_type
        plugin_name = f"{temp_instance.name}@{version}"

        # Create a wrapper class with the versioned name
        class VersionedPlugin(plugin_class):  # type: ignore
            @property
            def name(self) -> str:
                return plugin_name

        self.register_plugin(VersionedPlugin, mode)

    def clear(self) -> None:
        """Clear all registered plugins."""
        for plugin_type in PluginType:
            self._plugins[plugin_type] = {}
        logger.info("Cleared all registered plugins")


class PluginRegistryManager:
    """Manager for component-specific plugin registries."""

    def __init__(self) -> None:
        """Initialize the registry manager."""
        self._component_registries: Dict[str, PluginRegistry] = {}
        self._global_registry = PluginRegistry()

    def get_registry(self, component_name: str) -> PluginRegistry:
        """
        Get a component-specific registry.

        Args:
        ----
            component_name: Name of the component

        Returns:
        -------
            Component-specific plugin registry

        """
        if component_name not in self._component_registries:
            # Create a new registry for this component
            self._component_registries[component_name] = PluginRegistry()

        return self._component_registries[component_name]

    def get_global_registry(self) -> PluginRegistry:
        """
        Get the global plugin registry.

        Returns
        -------
            Global plugin registry

        """
        return self._global_registry

    def create_child_registry(
        self, parent_registry: PluginRegistry, inherit: bool = True
    ) -> PluginRegistry:
        """
        Create a child registry that can inherit from a parent.

        Args:
        ----
            parent_registry: Parent registry to inherit from
            inherit: Whether to inherit plugins from parent

        Returns:
        -------
            Child registry

        """
        child_registry = PluginRegistry()

        # Inherit plugins from parent if requested
        if inherit:
            for plugin_type in PluginType:
                for plugin_name, plugin_class in parent_registry._plugins.get(
                    plugin_type, {}
                ).items():
                    child_registry.register_plugin(plugin_class, mode=RegistrationMode.SILENT)

        return child_registry


class LazyPluginRegistry:
    """Registry that lazily registers default plugins."""

    def __init__(self) -> None:
        """Initialize the lazy registry."""
        self._registry = PluginRegistry()
        self._default_plugins: Dict[PluginType, Dict[str, Callable[[], type[Plugin]]]] = {}

    def register_default_plugin(
        self, plugin_type: PluginType, plugin_name: str, factory: Callable[[], type[Plugin]]
    ) -> None:
        """
        Register a default plugin factory.

        Args:
        ----
            plugin_type: Type of the plugin
            plugin_name: Name of the plugin
            factory: Factory function that returns the plugin class

        """
        if plugin_type not in self._default_plugins:
            self._default_plugins[plugin_type] = {}

        self._default_plugins[plugin_type][plugin_name] = factory

    def get_plugin(self, plugin_type: PluginType, plugin_name: str) -> type[Plugin] | None:
        """
        Get a plugin, lazily registering default plugins if needed.

        Args:
        ----
            plugin_type: Type of the plugin
            plugin_name: Name of the plugin

        Returns:
        -------
            Plugin class if found, None otherwise

        """
        # Check if plugin is already registered
        plugin = self._registry.get_plugin(plugin_type, plugin_name)
        if plugin is not None:
            return plugin

        # Check if we have a default plugin factory
        if (
            plugin_type in self._default_plugins
            and plugin_name in self._default_plugins[plugin_type]
        ):
            # Create and register the plugin
            factory = self._default_plugins[plugin_type][plugin_name]
            plugin_class = factory()
            self._registry.register_plugin(plugin_class, mode=RegistrationMode.SILENT)
            return plugin_class

        return None

    def get_registry(self) -> PluginRegistry:
        """
        Get the underlying registry.

        Returns
        -------
            The plugin registry

        """
        return self._registry


class DefaultPluginConfig:
    """Configuration for default plugins."""

    def __init__(self) -> None:
        """Initialize the configuration."""
        self.enabled_plugins: Dict[PluginType, Set[str]] = {}
        self.disabled_plugins: Dict[PluginType, Set[str]] = {}

    def enable_plugin(self, plugin_type: PluginType, plugin_name: str) -> None:
        """
        Enable a default plugin.

        Args:
        ----
            plugin_type: Type of the plugin
            plugin_name: Name of the plugin

        """
        if plugin_type not in self.enabled_plugins:
            self.enabled_plugins[plugin_type] = set()

        self.enabled_plugins[plugin_type].add(plugin_name)

        # Remove from disabled plugins if present
        if (
            plugin_type in self.disabled_plugins
            and plugin_name in self.disabled_plugins[plugin_type]
        ):
            self.disabled_plugins[plugin_type].remove(plugin_name)

    def disable_plugin(self, plugin_type: PluginType, plugin_name: str) -> None:
        """
        Disable a default plugin.

        Args:
        ----
            plugin_type: Type of the plugin
            plugin_name: Name of the plugin

        """
        if plugin_type not in self.disabled_plugins:
            self.disabled_plugins[plugin_type] = set()

        self.disabled_plugins[plugin_type].add(plugin_name)

        # Remove from enabled plugins if present
        if plugin_type in self.enabled_plugins and plugin_name in self.enabled_plugins[plugin_type]:
            self.enabled_plugins[plugin_type].remove(plugin_name)

    def is_plugin_enabled(self, plugin_type: PluginType, plugin_name: str) -> bool:
        """
        Check if a plugin is enabled.

        Args:
        ----
            plugin_type: Type of the plugin
            plugin_name: Name of the plugin

        Returns:
        -------
            True if the plugin is enabled, False otherwise

        """
        # Explicitly disabled plugins
        if (
            plugin_type in self.disabled_plugins
            and plugin_name in self.disabled_plugins[plugin_type]
        ):
            return False

        # Explicitly enabled plugins
        if plugin_type in self.enabled_plugins and plugin_name in self.enabled_plugins[plugin_type]:
            return True

        # Default to enabled
        return True


def discover_plugins(registry: PluginRegistry | None = None):
    """
    Discover and register plugins from entry points.

    This function looks for entry points in the following groups:
    - saplings.model_adapters
    - saplings.memory_stores
    - saplings.validators
    - saplings.indexers
    - saplings.tools

    Args:
    ----
        registry: Optional registry to use for registration

    """
    if registry is None:
        registry = PluginRegistry()

    # Map entry point groups to plugin types
    entry_point_groups = {
        "saplings.model_adapters": PluginType.MODEL_ADAPTER,
        "saplings.memory_stores": PluginType.MEMORY_STORE,
        "saplings.validators": PluginType.VALIDATOR,
        "saplings.indexers": PluginType.INDEXER,
        "saplings.tools": PluginType.TOOL,
    }

    # Discover plugins from entry points
    for group_name, plugin_type in entry_point_groups.items():
        for entry_point in entry_points(group=group_name):
            try:
                plugin_class = entry_point.load()
                if not issubclass(plugin_class, Plugin):
                    logger.warning(
                        "Entry point %s in group %s does not provide a Plugin subclass",
                        entry_point.name,
                        group_name,
                    )
                    continue

                # Verify that the plugin type matches the entry point group
                if plugin_class.plugin_type != plugin_type:
                    logger.warning(
                        "Plugin %s has type %s but was registered in group %s for type %s",
                        plugin_class.name,
                        plugin_class.plugin_type,
                        group_name,
                        plugin_type,
                    )
                    continue

                registry.register_plugin(plugin_class, mode=RegistrationMode.SILENT)
            except Exception:
                logger.exception(
                    "Error loading plugin %s from group %s", entry_point.name, group_name
                )

    logger.info("Plugin discovery completed")


def get_plugin_registry(registry: PluginRegistry | None = None) -> PluginRegistry:
    """
    Get the plugin registry.

    This function supports both service locator and container-based approaches.
    It also accepts an explicit registry parameter for direct injection.

    Args:
    ----
        registry: Optional explicit registry to use

    Returns:
    -------
        PluginRegistry: The plugin registry

    """
    # If an explicit registry is provided, use it
    if registry is not None:
        return registry

    # Try to get the registry from the service locator
    try:
        from saplings.core.service_locator import ServiceLocator

        locator = ServiceLocator.get_instance()
        if locator.has("plugin_registry"):
            return locator.get("plugin_registry")
    except ImportError:
        # Service locator not available
        logger.debug("Service locator not available")
    except Exception as e:
        logger.debug(f"Error accessing service locator: {e}")

    # Skip container-based approach to avoid circular imports
    # The plugin system should be independent of the DI container

    # If we get here, create a new instance
    result = PluginRegistry()

    # Try to register with service locator
    try:
        from saplings.core.service_locator import ServiceLocator

        locator = ServiceLocator.get_instance()
        locator.register("plugin_registry", result)
    except ImportError:
        pass

    # Skip container registration to avoid circular imports
    # The plugin system should be independent of the DI container

    return result


def get_plugin(
    plugin_type: PluginType, plugin_name: str, registry: PluginRegistry | None = None
) -> type[Plugin] | None:
    """
    Get a plugin by type and name.

    Args:
    ----
        plugin_type: Type of the plugin
        plugin_name: Name of the plugin
        registry: Optional plugin registry to use

    Returns:
    -------
        Optional[Type[Plugin]]: The plugin class if found, None otherwise

    """
    return get_plugin_registry(registry).get_plugin(plugin_type, plugin_name)


def get_plugins_by_type(
    plugin_type: PluginType, registry: PluginRegistry | None = None
) -> dict[str, type[Plugin]]:
    """
    Get all plugins of a specific type.

    Args:
    ----
        plugin_type: Type of plugins to get
        registry: Optional plugin registry to use

    Returns:
    -------
        Dict[str, Type[Plugin]]: Dictionary of plugin name to plugin class

    """
    return get_plugin_registry(registry).get_plugins_by_type(plugin_type)


def register_plugin(
    plugin_class: type[Plugin],
    registry: PluginRegistry | None = None,
    mode: RegistrationMode = RegistrationMode.WARN,
) -> None:
    """
    Register a plugin class.

    Args:
    ----
        plugin_class: The plugin class to register
        registry: Optional plugin registry to use
        mode: How to handle duplicate registrations

    """
    try:
        # Get the plugin registry and register the plugin
        plugin_registry = get_plugin_registry(registry)
        plugin_registry.register_plugin(plugin_class, mode=mode)
    except Exception as e:
        # Log the error but don't crash the application
        logger.error(f"Failed to register plugin {plugin_class.__name__}: {e}")
        logger.debug("Stack trace:", exc_info=True)


# Type-specific plugin base classes


class ModelAdapterPlugin(Plugin):
    """Base class for model adapter plugins."""

    @property
    def plugin_type(self) -> PluginType:
        """Type of the plugin."""
        return PluginType.MODEL_ADAPTER

    def create_adapter(self, _provider: str, _model_name: str, **_kwargs: Any) -> object:
        """
        Create a model adapter instance.

        Args:
        ----
            _provider: The model provider
            _model_name: The model name
            **_kwargs: Additional arguments for the adapter

        Returns:
        -------
            Any: The model adapter instance

        """
        msg = "Subclasses must implement create_adapter method"
        raise NotImplementedError(msg)


class MemoryStorePlugin(Plugin):
    """Base class for memory store plugins."""

    @property
    def plugin_type(self) -> PluginType:
        """Type of the plugin."""
        return PluginType.MEMORY_STORE


class ValidatorPlugin(Plugin):
    """Base class for validator plugins."""

    @property
    def plugin_type(self) -> PluginType:
        """Type of the plugin."""
        return PluginType.VALIDATOR


class IndexerPlugin(Plugin):
    """Base class for indexer plugins."""

    @property
    def plugin_type(self) -> PluginType:
        """Type of the plugin."""
        return PluginType.INDEXER


class ToolPlugin(Plugin):
    """Base class for tool plugins."""

    @property
    def plugin_type(self) -> PluginType:
        """Type of the plugin."""
        return PluginType.TOOL
