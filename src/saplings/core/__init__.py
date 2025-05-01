"""Core modules for the Saplings framework."""

from saplings.core.model_adapter import (
    LLM,
    LLMResponse,
    ModelCapability,
    ModelMetadata,
    ModelRole,
    ModelURI,
)
from saplings.core.plugin import (
    IndexerPlugin,
    MemoryStorePlugin,
    ModelAdapterPlugin,
    Plugin,
    PluginRegistry,
    PluginType,
    ToolPlugin,
    ValidatorPlugin,
    discover_plugins,
    get_plugin,
    get_plugin_registry,
    get_plugins_by_type,
    register_plugin,
)

__all__ = [
    # Model adapter
    "LLM",
    "LLMResponse",
    "ModelCapability",
    "ModelMetadata",
    "ModelRole",
    "ModelURI",
    # Plugin system
    "Plugin",
    "PluginRegistry",
    "PluginType",
    "ModelAdapterPlugin",
    "MemoryStorePlugin",
    "ValidatorPlugin",
    "IndexerPlugin",
    "ToolPlugin",
    "discover_plugins",
    "get_plugin",
    "get_plugin_registry",
    "get_plugins_by_type",
    "register_plugin",
]
