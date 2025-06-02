from __future__ import annotations

"""
MCP (Machine Control Protocol) client for Saplings.

This module provides a client for connecting to MCP servers and making their tools
available to Saplings agents. It is based on the mcpadapt library.
"""


import logging
from typing import Any

from saplings._internal.tools.base import Tool

logger = logging.getLogger(__name__)

# Check if MCP tools are available
try:
    import mcpadapt  # noqa: F401

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    logger.warning(
        "MCP support not available. Please install the mcpadapt library: pip install saplings[mcp]"
    )


def is_mcp_available() -> bool:
    """
    Check if MCP tools are available.

    Returns
    -------
        True if MCP tools are available, False otherwise

    """
    return MCP_AVAILABLE


# Runtime imports
if MCP_AVAILABLE:
    try:
        from mcpadapt.core import MCPAdapt, StdioServerParameters  # type: ignore[import]
        from mcpadapt.smolagents_adapter import SmolAgentsAdapter  # type: ignore[import]
    except ImportError:
        MCPAdapt = None
        StdioServerParameters = None
        SmolAgentsAdapter = None
        MCP_AVAILABLE = False
        logger.warning(
            "MCP support not available. Please install the mcpadapt library: pip install saplings[mcp]"
        )
else:
    MCPAdapt = None
    StdioServerParameters = None
    SmolAgentsAdapter = None


class MCPTool(Tool):
    """
    Tool that wraps a SmolAgents tool from an MCP server.

    This tool provides a bridge between Saplings and SmolAgents tools
    provided by MCP servers.
    """

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        output_type: str,
        mcp_tool: Any,
    ) -> None:
        """
        Initialize an MCP tool.

        Args:
        ----
            name: Name of the tool
            description: Description of the tool
            parameters: Tool parameters
            output_type: Type of the tool's output
            mcp_tool: The SmolAgents tool to wrap

        """
        self.name = name
        self.description = description
        self.parameters = parameters
        self.output_type = output_type
        self.mcp_tool = mcp_tool
        self.is_initialized = True

    def forward(self, **kwargs) -> Any:
        """
        Execute the MCP tool.

        Args:
        ----
            **kwargs: Tool parameters

        Returns:
        -------
            Any: Tool execution result

        """
        try:
            return self.mcp_tool(**kwargs)
        except Exception as e:
            logger.exception(f"Error executing MCP tool {self.name}: {e}")
            return f"Error executing MCP tool {self.name}: {e}"


class MCPClient:
    """
    Manages the connection to MCP servers and makes their tools available to Saplings.

    This client can connect to one or more MCP servers and expose their tools to Saplings agents.
    It supports both stdio and SSE server types.

    Note: tools can only be accessed after the connection has been started with the
    `connect()` method, done during the init. If you don't use the context manager
    we strongly encourage to use "try ... finally" to ensure the connection is cleaned up.

    Args:
    ----
        server_parameters: MCP server parameters (stdio or sse). Can be a list if you want to connect multiple MCPs at once.

    """

    def __init__(self, server_parameters: Any) -> None:
        """
        Initialize the MCP client.

        Args:
        ----
            server_parameters: MCP server parameters (stdio or sse)

        """
        if not MCP_AVAILABLE:
            msg = "MCP support not available. Please install the mcpadapt library: pip install saplings[mcp]"
            raise ImportError(msg)

        if MCPAdapt is None:
            msg = "MCPAdapt not available. Please install the mcpadapt library: pip install saplings[mcp]"
            raise ImportError(msg)

        self._adapter = MCPAdapt(server_parameters, SmolAgentsAdapter())
        self._tools: list[Any] | None = None
        self._saplings_tools: list[Tool] | None = None
        self.connect()

    def connect(self):
        """Connect to the MCP server and initialize the tools."""
        self._tools = self._adapter.__enter__()
        self._saplings_tools = self._convert_tools_to_saplings()
        tools_count = len(self._tools) if self._tools is not None else 0
        logger.info(f"Connected to MCP server(s) with {tools_count} tools")

    def disconnect(self):
        """Disconnect from the MCP server."""
        if hasattr(self, "_adapter") and self._adapter is not None:
            self._adapter.__exit__(None, None, None)
            self._tools = None
            self._saplings_tools = None
            logger.info("Disconnected from MCP server(s)")

    def __enter__(self):
        """Enter context manager."""
        return self._saplings_tools

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        self.disconnect()

    def _convert_tools_to_saplings(self):
        """
        Convert SmolAgents tools to Saplings tools.

        Returns
        -------
            List[Tool]: The converted tools

        """
        if self._tools is None:
            msg = "MCP tools not initialized"
            raise ValueError(msg)

        saplings_tools = []
        for tool in self._tools:
            # Create a Saplings tool that wraps the SmolAgents tool
            saplings_tool = MCPTool(
                name=tool.name,
                description=tool.description,
                parameters=self._convert_parameters(tool),
                output_type="any",
                mcp_tool=tool,
            )
            saplings_tools.append(saplings_tool)

        return saplings_tools

    def _convert_parameters(self, tool):
        """
        Convert SmolAgents tool parameters to Saplings tool parameters.

        Args:
        ----
            tool: SmolAgents tool

        Returns:
        -------
            Dict: Saplings tool parameters

        """
        parameters = {}
        if hasattr(tool, "parameters") and isinstance(tool.parameters, dict):
            for param_name, param_info in tool.parameters.items():
                param_type = param_info.get("type", "string")
                param_desc = param_info.get("description", f"Parameter {param_name}")
                param_required = param_info.get("required", True)

                parameters[param_name] = {
                    "type": param_type,
                    "description": param_desc,
                    "required": param_required,
                }

        return parameters
