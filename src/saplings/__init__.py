"""
Saplings: A powerful AI agent framework for building intelligent applications.

This package provides a comprehensive framework for creating AI agents with
advanced capabilities including memory management, tool integration, and
multi-modal processing.

Example:
-------
    >>> from saplings import AgentConfig, Agent
    >>> config = AgentConfig(
    ...     provider="openai",
    ...     model_name="gpt-4o",
    ...     api_key="your-api-key"
    ... )
    >>> agent = Agent(config)
    >>> result = await agent.run("Calculate the factorial of 5")

"""

from __future__ import annotations

# Import core configuration immediately as it's needed early
from saplings._internal.agent_config import AgentConfig

# Lazy import cache to avoid repeated imports
_lazy_cache = {}


def __getattr__(name: str):
    """
    Lazy import function to load API components when accessed.

    This avoids circular imports by only importing when actually needed.
    """
    if name in _lazy_cache:
        return _lazy_cache[name]

    # Version info
    if name == "__version__":
        from saplings.api.version import __version__

        _lazy_cache[name] = __version__
        return __version__

    # Core Agent functionality
    elif name == "Agent":
        from saplings.api.agent import Agent

        _lazy_cache[name] = Agent
        return Agent
    elif name == "AgentBuilder":
        from saplings.api.agent import AgentBuilder

        _lazy_cache[name] = AgentBuilder
        return AgentBuilder
    elif name == "AgentFacade":
        from saplings.api.agent import AgentFacade

        _lazy_cache[name] = AgentFacade
        return AgentFacade
    elif name == "AgentFacadeBuilder":
        from saplings.api.agent import AgentFacadeBuilder

        _lazy_cache[name] = AgentFacadeBuilder
        return AgentFacadeBuilder

    # Container and Dependency Injection
    elif name in (
        "Container",
        "container",
        "reset_container",
        "configure_container",
        "reset_container_config",
    ):
        from saplings.api.di import (
            Container,
            configure_container,
            container,
            reset_container,
            reset_container_config,
        )

        _lazy_cache.update(
            {
                "Container": Container,
                "container": container,
                "reset_container": reset_container,
                "configure_container": configure_container,
                "reset_container_config": reset_container_config,
            }
        )
        return _lazy_cache[name]

    # Tools
    elif name == "tool":
        from saplings.api.tools import tool

        _lazy_cache[name] = tool
        return tool

    # For any other attribute, raise AttributeError immediately
    # This prevents circular imports that would occur if we tried to import from saplings.api
    # Users should import other components from saplings.api directly
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Define a minimal __all__ for the most commonly used items
__all__ = [
    # Core essentials
    "__version__",
    "AgentConfig",
    # Core Agent functionality (lazy loaded)
    "Agent",
    "AgentBuilder",
    "AgentFacade",
    "AgentFacadeBuilder",
    # Tools (lazy loaded)
    "tool",
    # Container and Dependency Injection (lazy loaded)
    "Container",
    "container",
    "reset_container",
    "configure_container",
    "reset_container_config",
]
