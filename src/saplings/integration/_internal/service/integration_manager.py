from __future__ import annotations

"""
Integration manager for Saplings.

This module provides the integration manager for connecting different components
of the Saplings framework.
"""


import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from saplings.integration._internal.service.events import Event, EventSystem, EventType
from saplings.integration._internal.service.hot_loader import HotLoader, HotLoaderConfig
from saplings.integration._internal.service.secure_hot_loader import (
    SecureHotLoader,
    SecureHotLoaderConfig,
)

if TYPE_CHECKING:
    from saplings.core._internal.plugin import ToolPlugin
    from saplings.executor._internal import Executor
    from saplings.orchestration._internal import AgentNode, GraphRunner
    from saplings.planner._internal import BasePlanner

logger = logging.getLogger(__name__)


class ToolLifecycleManager:
    """
    Manager for tool lifecycle events.

    This class provides a simplified interface for managing tool lifecycle events,
    including loading, unloading, and updating tools.
    """

    def __init__(self, event_system: Optional[EventSystem] = None) -> None:
        """
        Initialize the tool lifecycle manager.

        Args:
        ----
            event_system: Event system to use, or None to create a new one

        """
        self.event_system = event_system or EventSystem()
        self.tools: Dict[str, Any] = {}

    def register_tool(self, tool_class: Any) -> None:
        """
        Register a tool.

        Args:
        ----
            tool_class: Tool class to register

        """
        # Create a temporary instance to get the tool ID
        temp_instance = tool_class()
        tool_id = getattr(temp_instance, "id", tool_class.__name__)

        # Store the tool
        self.tools[tool_id] = tool_class

        # Emit event
        self.event_system.emit(
            Event(
                type=EventType.TOOL_LOADED,
                source="tool_lifecycle_manager",
                data={
                    "tool_id": tool_id,
                    "tool_class": tool_class,
                },
            )
        )

        logger.info(f"Registered tool: {tool_id}")

    def unregister_tool(self, tool_id: str) -> bool:
        """
        Unregister a tool.

        Args:
        ----
            tool_id: ID of the tool to unregister

        Returns:
        -------
            bool: True if the tool was unregistered, False if not found

        """
        if tool_id not in self.tools:
            logger.warning(f"Cannot unregister tool {tool_id}: not registered")
            return False

        # Get the tool class before removing it
        tool_class = self.tools[tool_id]

        # Remove the tool
        del self.tools[tool_id]

        # Emit event
        self.event_system.emit(
            Event(
                type=EventType.TOOL_UNLOADED,
                source="tool_lifecycle_manager",
                data={
                    "tool_id": tool_id,
                    "tool_class": tool_class,
                },
            )
        )

        logger.info(f"Unregistered tool: {tool_id}")
        return True

    def update_tool(self, tool_class: Any) -> None:
        """
        Update a tool.

        Args:
        ----
            tool_class: Updated tool class

        """
        # Create a temporary instance to get the tool ID
        temp_instance = tool_class()
        tool_id = getattr(temp_instance, "id", tool_class.__name__)

        # Update the tool
        self.tools[tool_id] = tool_class

        # Emit event
        self.event_system.emit(
            Event(
                type=EventType.TOOL_UPDATED,
                source="tool_lifecycle_manager",
                data={
                    "tool_id": tool_id,
                    "tool_class": tool_class,
                },
            )
        )

        logger.info(f"Updated tool: {tool_id}")

    def get_tool(self, tool_id: str) -> Optional[Any]:
        """
        Get a tool by ID.

        Args:
        ----
            tool_id: ID of the tool to get

        Returns:
        -------
            Optional[Any]: Tool class if found, None otherwise

        """
        return self.tools.get(tool_id)


class IntegrationManager:
    """
    Integration manager for Saplings.

    This class provides integration between different components of the Saplings
    framework, including:
    - Hot-loading of tools
    - Integration with executor and planner
    - Event-based communication
    """

    def __init__(
        self,
        executor: Executor | None = None,
        planner: BasePlanner | None = None,
        graph_runner: GraphRunner | None = None,
        hot_loader: HotLoader | SecureHotLoader | None = None,
        hot_loader_config: HotLoaderConfig | SecureHotLoaderConfig | None = None,
    ) -> None:
        """
        Initialize the integration manager.

        Args:
        ----
            executor: Executor instance
            planner: Planner instance
            graph_runner: GraphRunner instance
            hot_loader: HotLoader or SecureHotLoader instance
            hot_loader_config: Configuration for the hot-loader

        """
        self.executor = executor
        self.planner = planner
        self.graph_runner = graph_runner

        # Create a hot loader if not provided - use SecureHotLoader by default
        if hot_loader is None:
            if isinstance(hot_loader_config, SecureHotLoaderConfig):
                self.hot_loader = SecureHotLoader(config=hot_loader_config)
            elif isinstance(hot_loader_config, HotLoaderConfig):
                # Regular HotLoaderConfig - but still use SecureHotLoader with security disabled
                secure_config = SecureHotLoaderConfig(
                    watch_directories=hot_loader_config.watch_directories,
                    auto_reload=hot_loader_config.auto_reload,
                    reload_interval=hot_loader_config.reload_interval,
                    tool_discovery_method=hot_loader_config.tool_discovery_method,
                    on_tool_load_callback=hot_loader_config.on_tool_load_callback,
                    on_tool_unload_callback=hot_loader_config.on_tool_unload_callback,
                    # Security settings - disabled by default for backward compatibility
                    enable_sandboxing=False,
                )
                self.hot_loader = SecureHotLoader(config=secure_config)
            else:
                # No config provided, create with defaults
                self.hot_loader = SecureHotLoader(
                    config=SecureHotLoaderConfig(enable_sandboxing=True)
                )
        else:
            self.hot_loader = hot_loader

        # Set up event system
        self.event_system = EventSystem()

        # Register event listeners
        self._register_event_listeners()

        logger.info("Initialized IntegrationManager")

    def _register_event_listeners(self):
        """Register event listeners."""
        # Register tool events
        self.event_system.add_listener(
            EventType.TOOL_LOADED,
            self._on_tool_loaded,
        )
        self.event_system.add_listener(
            EventType.TOOL_UNLOADED,
            self._on_tool_unloaded,
        )
        self.event_system.add_listener(
            EventType.TOOL_UPDATED,
            self._on_tool_updated,
        )

        # Register agent events
        self.event_system.add_listener(
            EventType.AGENT_REGISTERED,
            self._on_agent_registered,
        )
        self.event_system.add_listener(
            EventType.AGENT_UNREGISTERED,
            self._on_agent_unregistered,
        )

        logger.info("Registered event listeners")

    def _on_tool_loaded(self, event: Event) -> None:
        """
        Handle tool loaded event.

        Args:
        ----
            event: Event data

        """
        tool_class = event.data.get("tool_class")
        if tool_class is not None:
            # Register the tool with the executor and planner
            self.register_tool_with_executor(tool_class)
            self.register_tool_with_planner(tool_class)
            self.register_tool_with_graph_runner(tool_class)

            logger.info(f"Registered tool {tool_class.__name__} with components")

    def _on_tool_unloaded(self, event: Event) -> None:
        """
        Handle tool unloaded event.

        Args:
        ----
            event: Event data

        """
        tool_id = event.data.get("tool_id")
        if tool_id is not None:
            # Unregister the tool from the executor and planner
            self.unregister_tool_from_executor(tool_id)
            self.unregister_tool_from_planner(tool_id)
            self.unregister_tool_from_graph_runner(tool_id)

            logger.info(f"Unregistered tool {tool_id} from components")

    def _on_tool_updated(self, event: Event) -> None:
        """
        Handle tool updated event.

        Args:
        ----
            event: Event data

        """
        tool_class = event.data.get("tool_class")
        if tool_class is not None:
            # Update the tool in the executor and planner
            self.register_tool_with_executor(tool_class)
            self.register_tool_with_planner(tool_class)
            self.register_tool_with_graph_runner(tool_class)

            logger.info(f"Updated tool {tool_class.__name__} in components")

    def _on_agent_registered(self, event: Event) -> None:
        """
        Handle agent registered event.

        Args:
        ----
            event: Event data

        """
        agent = event.data.get("agent")
        if agent is not None:
            # Register tools with the agent
            self.register_tools_with_agent(agent)

            logger.info(f"Registered tools with agent {agent.id}")

    def _on_agent_unregistered(self, event: Event) -> None:
        """
        Handle agent unregistered event.

        Args:
        ----
            event: Event data

        """
        agent_id = event.data.get("agent_id")
        if agent_id is not None:
            logger.info(f"Agent {agent_id} unregistered")

    def register_tool_with_executor(self, tool_class: type[ToolPlugin]) -> None:
        """
        Register a tool with the executor.

        Args:
        ----
            tool_class: Tool class to register

        """
        if self.executor is None:
            logger.warning("Cannot register tool with executor: executor not set")
            return

        # Create a temporary instance to get the tool ID
        temp_instance = tool_class()
        tool_id = getattr(temp_instance, "id", tool_class.__name__)

        # Add the tool to the executor
        if not hasattr(self.executor, "tools"):
            self.executor.tools = {}

        self.executor.tools[tool_id] = tool_class

        logger.info(f"Registered tool {tool_id} with executor")

    def unregister_tool_from_executor(self, tool_id: str) -> None:
        """
        Unregister a tool from the executor.

        Args:
        ----
            tool_id: ID of the tool to unregister

        """
        if self.executor is None:
            logger.warning("Cannot unregister tool from executor: executor not set")
            return

        # Remove the tool from the executor
        if hasattr(self.executor, "tools") and tool_id in self.executor.tools:
            del self.executor.tools[tool_id]
            logger.info(f"Unregistered tool {tool_id} from executor")
        else:
            logger.warning(f"Cannot unregister tool {tool_id} from executor: not registered")

    def register_tool_with_planner(self, tool_class: type[ToolPlugin]) -> None:
        """
        Register a tool with the planner.

        Args:
        ----
            tool_class: Tool class to register

        """
        if self.planner is None:
            logger.warning("Cannot register tool with planner: planner not set")
            return

        # Create a temporary instance to get the tool ID
        temp_instance = tool_class()
        tool_id = getattr(temp_instance, "id", tool_class.__name__)

        # Add the tool to the planner
        if not hasattr(self.planner, "tools"):
            self.planner.tools = {}

        self.planner.tools[tool_id] = tool_class

        logger.info(f"Registered tool {tool_id} with planner")

    def unregister_tool_from_planner(self, tool_id: str) -> None:
        """
        Unregister a tool from the planner.

        Args:
        ----
            tool_id: ID of the tool to unregister

        """
        if self.planner is None:
            logger.warning("Cannot unregister tool from planner: planner not set")
            return

        # Remove the tool from the planner
        if hasattr(self.planner, "tools") and tool_id in self.planner.tools:
            del self.planner.tools[tool_id]
            logger.info(f"Unregistered tool {tool_id} from planner")
        else:
            logger.warning(f"Cannot unregister tool {tool_id} from planner: not registered")

    def register_tool_with_graph_runner(self, tool_class: type[ToolPlugin]) -> None:
        """
        Register a tool with the graph runner.

        Args:
        ----
            tool_class: Tool class to register

        """
        if self.graph_runner is None:
            logger.warning("Cannot register tool with graph runner: graph runner not set")
            return

        # Register the tool with each agent
        for agent in self.graph_runner.agents.values():
            self.register_tool_with_agent(agent, tool_class)

    def unregister_tool_from_graph_runner(self, tool_id: str) -> None:
        """
        Unregister a tool from the graph runner.

        Args:
        ----
            tool_id: ID of the tool to unregister

        """
        if self.graph_runner is None:
            logger.warning("Cannot unregister tool from graph runner: graph runner not set")
            return

        # Unregister the tool from each agent
        for agent in self.graph_runner.agents.values():
            self.unregister_tool_from_agent(agent, tool_id)

    def register_tool_with_agent(
        self, agent: AgentNode, tool_class: type[ToolPlugin] | None = None
    ) -> None:
        """
        Register a tool with an agent.

        Args:
        ----
            agent: Agent to register the tool with
            tool_class: Tool class to register, or None to register all tools

        """
        # Make sure the agent has a metadata dictionary with a tools entry
        if agent.metadata is None:
            agent.metadata = {}
        if "tools" not in agent.metadata:
            agent.metadata["tools"] = {}

        if tool_class is not None:
            # Register a single tool
            temp_instance = tool_class()
            tool_id = getattr(temp_instance, "id", tool_class.__name__)
            agent.metadata["tools"][tool_id] = tool_class
            logger.info(f"Registered tool {tool_id} with agent {agent.id}")
        else:
            # Register all tools
            for tool_id, tool_class in self.hot_loader.get_all_tools().items():
                agent.metadata["tools"][tool_id] = tool_class
                logger.info(f"Registered tool {tool_id} with agent {agent.id}")

    def unregister_tool_from_agent(self, agent: AgentNode, tool_id: str) -> None:
        """
        Unregister a tool from an agent.

        Args:
        ----
            agent: Agent to unregister the tool from
            tool_id: ID of the tool to unregister

        """
        if (
            agent.metadata is not None
            and "tools" in agent.metadata
            and tool_id in agent.metadata["tools"]
        ):
            del agent.metadata["tools"][tool_id]
            logger.info(f"Unregistered tool {tool_id} from agent {agent.id}")
        else:
            logger.warning(
                f"Cannot unregister tool {tool_id} from agent {agent.id}: not registered"
            )

    def register_tools_with_executor(self):
        """Register all tools with the executor."""
        if self.executor is None:
            logger.warning("Cannot register tools with executor: executor not set")
            return

        # Initialize tools attribute if it doesn't exist
        if not hasattr(self.executor, "tools"):
            self.executor.tools = {}

        # Register each tool
        for tool_id, tool_class in self.hot_loader.get_all_tools().items():
            self.executor.tools[tool_id] = tool_class

        logger.info(f"Registered {len(self.hot_loader.tools)} tools with executor")

    def register_tools_with_planner(self):
        """Register all tools with the planner."""
        if self.planner is None:
            logger.warning("Cannot register tools with planner: planner not set")
            return

        # Initialize tools attribute if it doesn't exist
        if not hasattr(self.planner, "tools"):
            self.planner.tools = {}

        # Register each tool
        for tool_id, tool_class in self.hot_loader.get_all_tools().items():
            self.planner.tools[tool_id] = tool_class

        logger.info(f"Registered {len(self.hot_loader.tools)} tools with planner")

    def register_tools_with_graph_runner(self):
        """Register all tools with the graph runner."""
        if self.graph_runner is None:
            logger.warning("Cannot register tools with graph runner: graph runner not set")
            return

        # Register tools with each agent
        for agent in self.graph_runner.agents.values():
            self.register_tools_with_agent(agent)

    def register_tools_with_agent(self, agent: AgentNode) -> None:
        """
        Register all tools with an agent.

        Args:
        ----
            agent: Agent to register tools with

        """
        # Make sure the agent has a metadata dictionary with a tools entry
        if agent.metadata is None:
            agent.metadata = {}
        if "tools" not in agent.metadata:
            agent.metadata["tools"] = {}

        # Register each tool
        for tool_id, tool_class in self.hot_loader.get_all_tools().items():
            agent.metadata["tools"][tool_id] = tool_class

        logger.info(f"Registered {len(self.hot_loader.tools)} tools with agent {agent.id}")

    def start(self):
        """Start the integration manager."""
        # Start the hot loader
        if self.hot_loader.config.auto_reload:
            self.hot_loader.start_auto_reload()

        # Register tools with components
        self.register_tools_with_executor()
        self.register_tools_with_planner()
        self.register_tools_with_graph_runner()

        logger.info("Started IntegrationManager")

    def stop(self):
        """Stop the integration manager."""
        # Stop the hot loader
        self.hot_loader.stop_auto_reload()

        logger.info("Stopped IntegrationManager")
