"""
Hot-loading system for Saplings.

This module provides the hot-loading mechanism for tools, allowing them to be
added, updated, or removed without restarting the system.
"""

import asyncio
import importlib
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Type, cast

from saplings.core.plugin import (
    Plugin,
    PluginType,
    ToolPlugin,
    discover_plugins,
    get_plugins_by_type,
)

logger = logging.getLogger(__name__)


@dataclass
class HotLoaderConfig:
    """Configuration for the hot-loading system."""
    
    watch_directories: List[str] = field(default_factory=list)
    """Directories to watch for tool changes."""
    
    auto_reload: bool = True
    """Whether to automatically reload tools when changes are detected."""
    
    reload_interval: float = 5.0
    """Interval in seconds between reload checks."""
    
    tool_discovery_method: str = "entry_points"
    """Method to discover tools. Options: 'entry_points', 'directory'."""
    
    on_tool_load_callback: Optional[Callable[[Type[ToolPlugin]], None]] = None
    """Callback to call when a tool is loaded."""
    
    on_tool_unload_callback: Optional[Callable[[str], None]] = None
    """Callback to call when a tool is unloaded."""


class ToolLifecycleManager:
    """
    Manager for tool lifecycle.
    
    This class handles the initialization, update, and retirement of tools.
    """
    
    def __init__(self):
        """Initialize the tool lifecycle manager."""
        self.initialized_tools: Dict[str, Type[ToolPlugin]] = {}
        self.retired_tools: Set[str] = set()
        
        logger.info("Initialized ToolLifecycleManager")
    
    def initialize_tool(self, tool_class: Type[ToolPlugin]) -> None:
        """
        Initialize a tool.
        
        Args:
            tool_class: Tool class to initialize
        """
        # Create a temporary instance to get the tool ID
        temp_instance = tool_class()
        tool_id = getattr(temp_instance, "id", tool_class.__name__)
        
        # Add to initialized tools
        self.initialized_tools[tool_id] = tool_class
        
        # Remove from retired tools if present
        if tool_id in self.retired_tools:
            self.retired_tools.remove(tool_id)
        
        logger.info(f"Initialized tool: {tool_id}")
    
    def update_tool(self, tool_class: Type[ToolPlugin]) -> None:
        """
        Update a tool.
        
        Args:
            tool_class: Tool class to update
        """
        # Create a temporary instance to get the tool ID
        temp_instance = tool_class()
        tool_id = getattr(temp_instance, "id", tool_class.__name__)
        
        # Check if the tool is already initialized
        if tool_id in self.initialized_tools:
            # Update the tool
            self.initialized_tools[tool_id] = tool_class
            logger.info(f"Updated tool: {tool_id}")
        else:
            # Initialize the tool
            self.initialize_tool(tool_class)
    
    def retire_tool(self, tool_id: str) -> None:
        """
        Retire a tool.
        
        Args:
            tool_id: ID of the tool to retire
        """
        # Check if the tool is initialized
        if tool_id in self.initialized_tools:
            # Remove from initialized tools
            del self.initialized_tools[tool_id]
            
            # Add to retired tools
            self.retired_tools.add(tool_id)
            
            logger.info(f"Retired tool: {tool_id}")
        else:
            logger.warning(f"Cannot retire tool {tool_id}: not initialized")
    
    def get_tool(self, tool_id: str) -> Optional[Type[ToolPlugin]]:
        """
        Get a tool by ID.
        
        Args:
            tool_id: ID of the tool
            
        Returns:
            Optional[Type[ToolPlugin]]: Tool class if found, None otherwise
        """
        return self.initialized_tools.get(tool_id)
    
    def get_all_tools(self) -> Dict[str, Type[ToolPlugin]]:
        """
        Get all initialized tools.
        
        Returns:
            Dict[str, Type[ToolPlugin]]: Dictionary of tool ID to tool class
        """
        return self.initialized_tools.copy()
    
    def is_retired(self, tool_id: str) -> bool:
        """
        Check if a tool is retired.
        
        Args:
            tool_id: ID of the tool
            
        Returns:
            bool: True if the tool is retired, False otherwise
        """
        return tool_id in self.retired_tools


class HotLoader:
    """
    Hot-loading system for tools.
    
    This class provides the hot-loading mechanism for tools, allowing them to be
    added, updated, or removed without restarting the system.
    """
    
    def __init__(self, config: Optional[HotLoaderConfig] = None):
        """
        Initialize the hot-loading system.
        
        Args:
            config: Configuration for the hot-loading system
        """
        self.config = config or HotLoaderConfig()
        self.tools: Dict[str, Type[ToolPlugin]] = {}
        self.lifecycle_manager = ToolLifecycleManager()
        self._auto_reload_task: Optional[asyncio.Task] = None
        self._last_reload_time = 0.0
        
        # Create watch directories if they don't exist
        for directory in self.config.watch_directories:
            os.makedirs(directory, exist_ok=True)
        
        logger.info(f"Initialized HotLoader with config: {self.config}")
    
    def load_tool(self, tool_class: Type[ToolPlugin]) -> Type[ToolPlugin]:
        """
        Load a tool.
        
        Args:
            tool_class: Tool class to load
            
        Returns:
            Type[ToolPlugin]: Loaded tool class
        """
        # Create a temporary instance to get the tool ID
        temp_instance = tool_class()
        tool_id = getattr(temp_instance, "id", tool_class.__name__)
        
        # Add to tools dictionary
        self.tools[tool_id] = tool_class
        
        # Initialize the tool
        self.lifecycle_manager.initialize_tool(tool_class)
        
        # Call the callback if provided
        if self.config.on_tool_load_callback is not None:
            self.config.on_tool_load_callback(tool_class)
        
        logger.info(f"Loaded tool: {tool_id}")
        
        return tool_class
    
    def unload_tool(self, tool_id: str) -> None:
        """
        Unload a tool.
        
        Args:
            tool_id: ID of the tool to unload
        """
        # Check if the tool is loaded
        if tool_id in self.tools:
            # Remove from tools dictionary
            del self.tools[tool_id]
            
            # Retire the tool
            self.lifecycle_manager.retire_tool(tool_id)
            
            # Call the callback if provided
            if self.config.on_tool_unload_callback is not None:
                self.config.on_tool_unload_callback(tool_id)
            
            logger.info(f"Unloaded tool: {tool_id}")
        else:
            logger.warning(f"Cannot unload tool {tool_id}: not loaded")
    
    def reload_tools(self) -> None:
        """Reload all tools."""
        # Record the reload time
        self._last_reload_time = time.time()
        
        # Discover plugins
        discover_plugins()
        
        # Get all tool plugins
        tool_plugins = get_plugins_by_type(PluginType.TOOL)
        
        # Track which tools were found
        found_tools = set()
        
        # Load or update each tool
        for tool_id, tool_class in tool_plugins.items():
            found_tools.add(tool_id)
            
            if tool_id in self.tools:
                # Update the tool
                self.tools[tool_id] = tool_class
                self.lifecycle_manager.update_tool(tool_class)
                logger.info(f"Updated tool: {tool_id}")
            else:
                # Load the tool
                self.load_tool(tool_class)
        
        # Unload tools that were not found
        for tool_id in list(self.tools.keys()):
            if tool_id not in found_tools:
                self.unload_tool(tool_id)
        
        logger.info(f"Reloaded tools: {len(self.tools)} active, {len(self.lifecycle_manager.retired_tools)} retired")
    
    def get_tool(self, tool_id: str) -> Optional[Type[ToolPlugin]]:
        """
        Get a tool by ID.
        
        Args:
            tool_id: ID of the tool
            
        Returns:
            Optional[Type[ToolPlugin]]: Tool class if found, None otherwise
        """
        return self.tools.get(tool_id)
    
    def get_all_tools(self) -> Dict[str, Type[ToolPlugin]]:
        """
        Get all loaded tools.
        
        Returns:
            Dict[str, Type[ToolPlugin]]: Dictionary of tool ID to tool class
        """
        return self.tools.copy()
    
    async def _auto_reload_loop(self) -> None:
        """Auto-reload loop."""
        while True:
            try:
                # Reload tools
                self.reload_tools()
                
                # Wait for the next reload
                await asyncio.sleep(self.config.reload_interval)
            except asyncio.CancelledError:
                logger.info("Auto-reload loop cancelled")
                break
            except Exception as e:
                logger.exception(f"Error in auto-reload loop: {e}")
                await asyncio.sleep(self.config.reload_interval)
    
    def start_auto_reload(self) -> None:
        """Start the auto-reload loop."""
        if self._auto_reload_task is None:
            self._auto_reload_task = asyncio.create_task(self._auto_reload_loop())
            logger.info("Started auto-reload loop")
        else:
            logger.warning("Auto-reload loop already running")
    
    def stop_auto_reload(self) -> None:
        """Stop the auto-reload loop."""
        if self._auto_reload_task is not None:
            self._auto_reload_task.cancel()
            self._auto_reload_task = None
            logger.info("Stopped auto-reload loop")
        else:
            logger.warning("Auto-reload loop not running")
    
    def scan_directory(self, directory: str) -> List[str]:
        """
        Scan a directory for Python files.
        
        Args:
            directory: Directory to scan
            
        Returns:
            List[str]: List of Python file paths
        """
        python_files = []
        
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".py"):
                    python_files.append(os.path.join(root, file))
        
        return python_files
    
    def load_module_from_file(self, file_path: str) -> Optional[Any]:
        """
        Load a module from a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Optional[Any]: Loaded module if successful, None otherwise
        """
        try:
            # Get the module name from the file path
            module_name = os.path.splitext(os.path.basename(file_path))[0]
            
            # Add the directory to sys.path if it's not already there
            directory = os.path.dirname(file_path)
            if directory not in sys.path:
                sys.path.insert(0, directory)
            
            # Import the module
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None:
                logger.warning(f"Failed to get spec for {file_path}")
                return None
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            return module
        except Exception as e:
            logger.exception(f"Error loading module from {file_path}: {e}")
            return None
