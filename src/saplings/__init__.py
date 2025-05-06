from __future__ import annotations

"""Saplings - A graphs-first, self-improving agent framework.

This package provides the core functionality for the Saplings agent framework.

Saplings is a lightweight (≤ 1.2 k LOC core) framework for building domain-aware,
self-critiquing autonomous agents.

Key pillars:
1. Structural Memory — vector + graph store per corpus.
2. Cascaded, Entropy-Aware Retrieval — TF-IDF → embeddings → graph expansion.
3. Guard-railed Generation — Planner with budget, Executor with speculative draft/verify.
4. Judge & Validator Loop — reflexive scoring, self-healing patches.
5. Extensibility — hot-pluggable models, tools, validators.
6. Graph-Aligned Sparse Attention (GASA) — graph-conditioned attention masks for faster, better-grounded reasoning.
"""

# Standard library imports
import logging  # noqa: E402
from importlib import import_module  # noqa: E402

# Local imports
from saplings.core import (  # noqa: E402
    # Core types and interfaces
    LLM,
    # Configuration
    Config,
    ConfigurationError,
    ConfigValue,
    IExecutionService,
    IMemoryManager,
    IModalityService,
    # Interfaces
    IModelService,
    IMonitoringService,
    IOrchestrationService,
    IPlannerService,
    IRetrievalService,
    ISelfHealingService,
    IToolService,
    IValidatorService,
    LLMResponse,
    ModelCapability,
    ModelError,
    ModelMetadata,
    ModelRole,
    Plugin,
    PluginRegistry,
    PluginType,
    ProviderError,
    ResourceExhaustedError,
    # Exceptions
    SaplingsError,
    # Utils
    count_tokens,
    discover_plugins,
    get_tokens_remaining,
    split_text_by_tokens,
    truncate_text_tokens,
)
from saplings.security.log_filter import install_global_filter  # noqa: E402
from saplings.version import __version__  # noqa: E402

# Create a module-specific logger
logger = logging.getLogger(__name__)

# Install security features on startup
install_global_filter()
logger.info("Saplings initialized with security filters")

# By default, only expose core modules to keep the API surface minimal
# Higher-level modules can be imported explicitly
# These imports allow for backwards compatibility but aren't exposed in __all__

# fmt: off
__all__ = [
    # Core types and interfaces
    "Config",
    "ConfigurationError",
    "ConfigValue",
    "IExecutionService",
    "IMemoryManager",
    "IModalityService",
    "IModelService",
    "IMonitoringService",
    "IOrchestrationService",
    "IPlannerService",
    "IRetrievalService",
    "ISelfHealingService",
    "IToolService",
    "IValidatorService",
    "LLM",
    "LLMResponse",
    "ModelCapability",
    "ModelError",
    "ModelMetadata",
    "ModelRole",
    "Plugin",
    "PluginRegistry",
    "PluginType",
    "ProviderError",
    "ResourceExhaustedError",
    "SaplingsError",
    # Utils
    "count_tokens",
    "discover_plugins",
    "get_tokens_remaining",
    "split_text_by_tokens",
    "truncate_text_tokens",
    # Version
    "__version__",
]
# fmt: on

# Lazy imports to avoid circular dependencies
# These will be imported when accessed but aren't loaded immediately


def __getattr__(name: str) -> object:
    """Lazy import mechanism for modules to avoid circular imports."""
    # Map attribute names to their module and attribute
    lazy_imports = {
        "Agent": ("saplings.agent", "Agent"),
        "AgentConfig": ("saplings.agent_config", "AgentConfig"),
        "AgentFacade": ("saplings.agent_facade", "AgentFacade"),
        "AnthropicAdapter": ("saplings.adapters", "AnthropicAdapter"),
        "OpenAIAdapter": ("saplings.adapters", "OpenAIAdapter"),
        "HuggingFaceAdapter": ("saplings.adapters", "HuggingFaceAdapter"),
        "VLLMAdapter": ("saplings.adapters", "VLLMAdapter"),
    }

    if name in lazy_imports:
        module_name, attr_name = lazy_imports[name]
        module = import_module(module_name)
        return getattr(module, attr_name)

    msg = f"module 'saplings' has no attribute '{name}'"
    raise AttributeError(msg)


# Note: These classes are already included in __all__ for easier importing
