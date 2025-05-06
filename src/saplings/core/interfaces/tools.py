from __future__ import annotations

"""
Tool service interface for Saplings.

This module defines the interface for tool operations that manage function-calling
tools. This is a pure interface with no implementation details or dependencies
outside of the core types.
"""


from abc import ABC, abstractmethod
from typing import Any


class IToolService(ABC):
    """Interface for tool operations."""

    @abstractmethod
    def register_tool(self, tool: Any) -> bool:
        """
        Register a tool with the service.

        Args:
        ----
            tool: Tool to register

        Returns:
        -------
            bool: Whether the tool was successfully registered

        """

    @abstractmethod
    def prepare_functions_for_model(self):
        """
        Prepare tool definitions for the model.

        Returns
        -------
            List[Dict[str, Any]]: Tool definitions in the format expected by LLMs

        """

    @abstractmethod
    async def create_tool(
        self, name: str, description: str, code: str, trace_id: str | None = None
    ) -> Any:
        """
        Create a dynamic tool.

        Args:
        ----
            name: Tool name
            description: Tool description
            code: Tool implementation code
            trace_id: Optional trace ID for monitoring

        Returns:
        -------
            Any: The created tool

        """

    @abstractmethod
    def get_registered_tools(self):
        """
        Get all registered tools.

        Returns
        -------
            Dict[str, Any]: Registered tools

        """

    @property
    @abstractmethod
    def tools(self):
        """
        Get all tools.

        Returns
        -------
            Dict[str, Any]: All tools

        """

    @abstractmethod
    async def execute_tool(
        self, tool_name: str, parameters: dict[str, Any], trace_id: str | None = None
    ) -> Any:
        """
        Execute a tool.

        Args:
        ----
            tool_name: Name of the tool to execute
            parameters: Tool parameters
            trace_id: Optional trace ID for monitoring

        Returns:
        -------
            Any: Tool execution result

        """
