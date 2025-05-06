from __future__ import annotations

"""Core modules for the Saplings framework."""

# Import core models
# Import configuration

from saplings.core.config import Config, ConfigValue

# Import exceptions
from saplings.core.exceptions import (
    ConfigurationError,
    ModelError,
    ProviderError,
    ResourceExhaustedError,
    SaplingsError,
)

# Import interfaces
from saplings.core.interfaces import (
    IExecutionService,
    IMemoryManager,
    IModalityService,
    IModelService,
    IMonitoringService,
    IOrchestrationService,
    IPlannerService,
    IRetrievalService,
    ISelfHealingService,
    IToolService,
    IValidatorService,
)
from saplings.core.model_adapter import (
    LLM,
    LLMResponse,
    ModelCapability,
    ModelMetadata,
    ModelRole,
)

# Import plugin system
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

# Import utility functions
from saplings.core.utils import (
    count_tokens,
    get_tokens_remaining,
    split_text_by_tokens,
    truncate_text_tokens,
)

__all__ = [
    # Model adapter
    "LLM",
    # Configuration
    "Config",
    "ConfigValue",
    "ConfigurationError",
    "IExecutionService",
    "IMemoryManager",
    "IModalityService",
    # Interfaces
    "IModelService",
    "IMonitoringService",
    "IOrchestrationService",
    "IPlannerService",
    "IRetrievalService",
    "ISelfHealingService",
    "IToolService",
    "IValidatorService",
    "IndexerPlugin",
    "LLMResponse",
    "MemoryStorePlugin",
    "ModelAdapterPlugin",
    "ModelCapability",
    "ModelError",
    "ModelMetadata",
    "ModelRole",
    # Plugin system
    "Plugin",
    "PluginRegistry",
    "PluginType",
    "ProviderError",
    "ResourceExhaustedError",
    # Exceptions
    "SaplingsError",
    "ToolPlugin",
    "ValidatorPlugin",
    # Utilities
    "count_tokens",
    "discover_plugins",
    "get_plugin",
    "get_plugin_registry",
    "get_plugins_by_type",
    "get_tokens_remaining",
    "register_plugin",
    "split_text_by_tokens",
    "truncate_text_tokens",
]
