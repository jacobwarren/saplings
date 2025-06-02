from __future__ import annotations

"""
MCP (Machine Control Protocol) client for Saplings.

This module provides a client for connecting to MCP servers and making their tools
available to Saplings agents. It is based on the mcpadapt library.
"""


import logging
from typing import TYPE_CHECKING, Any

from saplings.tools._internal.base import Tool

if TYPE_CHECKING:
    from types import TracebackType

    from mcpadapt.core import MCPAdapt, StdioServerParameters  # type: ignore[import-not-found]
    from mcpadapt.smolagents_adapter import SmolAgentsAdapter  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)

# Runtime imports
try:
    from mcpadapt.core import MCPAdapt, StdioServerParameters  # type: ignore[import]
    from mcpadapt.smolagents_adapter import SmolAgentsAdapter  # type: ignore[import]

    MCP_AVAILABLE = True
except ImportError:
    MCPAdapt = None
    StdioServerParameters = None
    SmolAgentsAdapter = None
    MCP_AVAILABLE = False
    logger.warning(
        "MCP support not available. Please install the mcpadapt library: pip install saplings[mcp]"
    )


def is_mcp_available() -> bool:
    """
    Check if MCP support is available.

    Returns
    -------
        bool: True if MCP support is available, False otherwise

    """
    return MCP_AVAILABLE


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

    Example:
    -------
        ```python
        # fully managed context manager + stdio
        with MCPClient(...) as tools:
            # tools are now available
            agent = Agent(
                config=AgentConfig(
                    model_uri="openai://gpt-4o",
                    tools=tools
                )
            )

        # context manager + sse
        with MCPClient({"url": "http://localhost:8000/sse"}) as tools:
            # tools are now available

        # manually manage the connection via the mcp_client object:
        try:
            mcp_client = MCPClient(...)
            tools = mcp_client.get_tools()

            # use your tools here.
        finally:
            mcp_client.disconnect()
        ```

    """

    def __init__(
        self,
        server_parameters: Any,
    ) -> None:
        """
        Initialize the MCP client.

        Args:
        ----
            server_parameters: MCP server parameters (stdio or sse). Can be a list if you want to connect multiple MCPs at once.
                Can be a StdioServerParameters object, a dict, or a list of either.

        """
        if not MCP_AVAILABLE:
            msg = (
                "MCP support not available. Please install the mcpadapt library: "
                "pip install saplings[mcp]"
            )
            raise ImportError(msg)

        # Ensure MCPAdapt and SmolAgentsAdapter are available
        if MCPAdapt is None or SmolAgentsAdapter is None:
            msg = "MCPAdapt or SmolAgentsAdapter is not available"
            raise ImportError(msg)

        self._adapter = MCPAdapt(server_parameters, SmolAgentsAdapter())
        self._tools: list[Tool] | None = None
        self._saplings_tools: list[Tool] | None = None
        self.connect()

    def connect(self):
        """Connect to the MCP server and initialize the tools."""
        self._tools = self._adapter.__enter__()
        self._saplings_tools = self._convert_tools_to_saplings()
        tools_count = len(self._tools) if self._tools is not None else 0
        logger.info(f"Connected to MCP server(s) with {tools_count} tools")

    def disconnect(
        self,
        exc_type: type[BaseException] | None = None,
        exc_value: BaseException | None = None,
        exc_traceback: TracebackType | None = None,
    ) -> None:
        """
        Disconnect from the MCP server.

        Args:
        ----
            exc_type: Exception type if an exception was raised
            exc_value: Exception value if an exception was raised
            exc_traceback: Exception traceback if an exception was raised

        """
        self._adapter.__exit__(exc_type, exc_value, exc_traceback)
        logger.info("Disconnected from MCP server(s)")

    def get_tools(self):
        """
        Get the Saplings tools available from the MCP server.

        Note: for now, this always returns the tools available at the creation of the session,
        but it will in a future release return also new tools available from the MCP server if
        any at call time.

        Raises
        ------
            ValueError: If the MCP server tools is None (usually assuming the server is not started).

        Returns
        -------
            List[Tool]: The Saplings tools available from the MCP server.

        """
        if self._saplings_tools is None:
            msg = "Couldn't retrieve tools from MCP server, run `mcp_client.connect()` first before accessing `tools`"
            raise ValueError(msg)
        return self._saplings_tools

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

    def _convert_parameters(self, tool: Any) -> dict[str, Any]:
        """
        Convert SmolAgents tool parameters to Saplings tool parameters.

        Args:
        ----
            tool: SmolAgents tool

        Returns:
        -------
            Dict[str, Any]: Saplings tool parameters

        """
        parameters = {}

        # SmolAgents tools have a 'parameters' attribute with parameter information
        if hasattr(tool, "parameters"):
            for param_name, param_info in tool.parameters.items():
                parameters[param_name] = {
                    "type": param_info.get("type", "any"),
                    "description": param_info.get("description", f"Parameter {param_name}"),
                    "required": param_info.get("required", True),
                }

        return parameters

    def __enter__(self):
        """
        Connect to the MCP server and return the tools directly.

        Note that because of the `.connect` in the init, the mcp_client
        is already connected at this point.

        Returns
        -------
            List[Tool]: The Saplings tools available from the MCP server

        """
        return self.get_tools()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_traceback: TracebackType | None,
    ):
        """
        Disconnect from the MCP server.

        Args:
        ----
            exc_type: Exception type if an exception was raised
            exc_value: Exception value if an exception was raised
            exc_traceback: Exception traceback if an exception was raised

        """
        self.disconnect(exc_type, exc_value, exc_traceback)


class MCPTool(Tool):
    """
    A Saplings tool that wraps an MCP tool.

    This class provides a bridge between MCP tools and Saplings tools.
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
            parameters: Parameters of the tool
            output_type: Output type of the tool
            mcp_tool: The underlying MCP tool

        """
        self.name = name
        self.description = description
        self.parameters = parameters
        self.output_type = output_type
        self.mcp_tool = mcp_tool
        self.is_initialized = True

    def forward(self, *args, **kwargs):
        """
        Call the MCP tool.

        Args:
        ----
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
        -------
            Any: The result of the tool call

        """
        return self.mcp_tool(*args, **kwargs)
