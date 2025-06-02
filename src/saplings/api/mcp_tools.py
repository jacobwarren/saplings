from __future__ import annotations

"""
MCP Tools API module for Saplings.

This module provides the public API for MCP-related tools.
"""


from saplings._internal.tools.mcp_client import (
    MCPClient as _MCPClient,
)
from saplings._internal.tools.mcp_client import (
    MCPTool as _MCPTool,
)
from saplings.api.stability import beta


@beta
class MCPClient:
    """
    Client for interacting with the MCP (Model Control Protocol) server.

    This client provides methods for sending requests to and receiving responses
    from an MCP server.
    """

    def __init__(self, *args, **kwargs):
        self._client = _MCPClient(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._client, name)


@beta
class MCPTool:
    """
    Tool for interacting with the MCP (Model Control Protocol) server.

    This tool allows an agent to send requests to and receive responses from
    an MCP server.
    """

    def __init__(self, *args, **kwargs):
        self._tool = _MCPTool(*args, **kwargs)
        self.name = self._tool.name
        self.description = self._tool.description
        self.parameters = self._tool.parameters

    def __call__(self, *args, **kwargs):
        return self._tool(*args, **kwargs)


__all__ = [
    "MCPClient",
    "MCPTool",
]
