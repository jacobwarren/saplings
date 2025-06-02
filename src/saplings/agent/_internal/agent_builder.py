from __future__ import annotations

"""
Agent builder module.

This module provides a builder for Agent instances.
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from saplings.agent._internal.types import AgentBuilderProtocol

# Configure logging
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from saplings.agent._internal.agent_config import AgentConfig


class AgentBuilder(AgentBuilderProtocol):
    """
    Builder for creating Agent instances with a fluent interface.

    This builder provides a convenient way to configure and create Agent
    instances with various options and dependencies.
    """

    def __init__(self) -> None:
        """Initialize the agent builder."""
        self._config_dict: Dict[str, Any] = {}
        self._use_container: bool = True

    def with_provider(self, provider: str) -> "AgentBuilder":
        """
        Set the model provider.

        Args:
        ----
            provider: The model provider (e.g., "openai", "anthropic", "vllm")

        Returns:
        -------
            The builder instance for method chaining

        """
        self._config_dict["provider"] = provider
        return self

    def with_model_name(self, model_name: str) -> "AgentBuilder":
        """
        Set the model name.

        Args:
        ----
            model_name: The name of the model to use

        Returns:
        -------
            The builder instance for method chaining

        """
        self._config_dict["model_name"] = model_name
        return self

    def with_memory_path(self, memory_path: str) -> "AgentBuilder":
        """
        Set the memory path.

        Args:
        ----
            memory_path: Path to the memory store directory

        Returns:
        -------
            The builder instance for method chaining

        """
        self._config_dict["memory_path"] = memory_path
        return self

    def with_output_dir(self, output_dir: str) -> "AgentBuilder":
        """
        Set the output directory.

        Args:
        ----
            output_dir: Path to the output directory

        Returns:
        -------
            The builder instance for method chaining

        """
        self._config_dict["output_dir"] = output_dir
        return self

    def with_gasa_enabled(self, enabled: bool) -> "AgentBuilder":
        """
        Enable or disable Graph-Aligned Sparse Attention.

        Args:
        ----
            enabled: Whether to enable GASA

        Returns:
        -------
            The builder instance for method chaining

        """
        self._config_dict["enable_gasa"] = enabled
        return self

    def with_monitoring_enabled(self, enabled: bool) -> "AgentBuilder":
        """
        Enable or disable monitoring.

        Args:
        ----
            enabled: Whether to enable monitoring

        Returns:
        -------
            The builder instance for method chaining

        """
        self._config_dict["enable_monitoring"] = enabled
        return self

    def with_tools(self, tools: List[Any]) -> "AgentBuilder":
        """
        Set the tools to register with the agent.

        Args:
        ----
            tools: List of tools to register

        Returns:
        -------
            The builder instance for method chaining

        """
        self._config_dict["tools"] = tools
        return self

    def with_config(self, config: Union[Dict[str, Any], "AgentConfig"]) -> "AgentBuilder":
        """
        Set additional configuration options.

        Args:
        ----
            config: Additional configuration options as a dictionary or AgentConfig

        Returns:
        -------
            The builder instance for method chaining

        """
        if isinstance(config, dict):
            self._config_dict.update(config)
        else:
            # Extract all attributes from the config object
            for key, value in vars(config).items():
                self._config_dict[key] = value
        return self

    def with_self_healing_enabled(self, enabled: bool) -> "AgentBuilder":
        """
        Enable or disable self-healing.

        Args:
        ----
            enabled: Whether to enable self-healing

        Returns:
        -------
            The builder instance for method chaining

        """
        self._config_dict["enable_self_healing"] = enabled
        return self

    def with_tool_factory_enabled(self, enabled: bool) -> "AgentBuilder":
        """
        Enable or disable the tool factory.

        Args:
        ----
            enabled: Whether to enable the tool factory

        Returns:
        -------
            The builder instance for method chaining

        """
        self._config_dict["enable_tool_factory"] = enabled
        return self

    def build(
        self, use_container: Optional[bool] = None
    ) -> Any:  # Return type is Any to avoid circular imports
        """
        Build the agent with the configured options.

        Args:
        ----
            use_container: Whether to use the dependency injection container.
                           If None, uses the value set with with_use_container().

        Returns:
        -------
            The initialized agent instance

        """
        # Validate required configuration
        if "provider" not in self._config_dict:
            raise ValueError("Provider is required")
        if "model_name" not in self._config_dict:
            raise ValueError("Model name is required")

        # Set use_container if provided
        if use_container is not None:
            self._use_container = use_container

        # Create the agent using a factory function to avoid circular imports
        return self._create_agent(self._config_dict, self._use_container)

    @staticmethod
    def _create_agent(config_dict: Dict[str, Any], use_container: bool) -> Any:
        """
        Create an Agent instance.

        This method uses a factory approach to avoid circular imports.

        Args:
        ----
            config_dict: Configuration dictionary
            use_container: Whether to use the dependency injection container

        Returns:
        -------
            Agent: The created agent instance

        """
        # Use dynamic import to avoid circular imports
        import importlib

        # Import the config module dynamically
        config_module = importlib.import_module("saplings.agent._internal.agent_config")
        AgentConfig = config_module.AgentConfig

        # Create a copy of the config dictionary
        config_dict_copy = config_dict.copy()

        # Add use_container to the config dictionary
        config_dict_copy["_use_container"] = use_container

        # Create the configuration object
        config = AgentConfig(**config_dict_copy)

        # Import the Agent class dynamically to avoid circular imports
        agent_module = importlib.import_module("saplings.agent._internal.agent")
        Agent = agent_module.Agent

        # Create and return the agent
        return Agent(config)
