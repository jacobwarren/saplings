# MCP Client

The MCP (Machine Control Protocol) client in Saplings allows agents to connect to MCP servers and use their tools. This document explains how to use the MCP client and integrate it with Saplings agents.

## Overview

The MCP client provides a bridge between Saplings agents and MCP servers, enabling agents to use tools provided by these servers. Key features include:

- **Multiple Server Support**: Connect to one or more MCP servers simultaneously
- **Tool Conversion**: Convert MCP tools to Saplings tools
- **Context Manager**: Easy-to-use context manager interface
- **Server Types**: Support for both stdio and SSE server types

## Core Concepts

### MCP Servers

MCP (Machine Control Protocol) servers provide tools that can be used by agents. These servers can be:

1. **Stdio Servers**: Command-line programs that communicate via standard input/output
2. **SSE Servers**: Web servers that communicate via Server-Sent Events

### Tool Conversion

The MCP client converts tools from MCP servers to Saplings tools, making them available to agents:

1. **Tool Discovery**: The client discovers tools available on the MCP server
2. **Parameter Mapping**: Tool parameters are mapped between MCP and Saplings formats
3. **Execution Bridge**: Tool execution is bridged between Saplings and MCP

## API Reference

### MCPClient

```python
class MCPClient:
    def __init__(
        self,
        server_parameters: Union[
            "StdioServerParameters",
            Dict[str, Any],
            List[Union["StdioServerParameters", Dict[str, Any]]]
        ],
    ):
        """
        Initialize the MCP client.

        Args:
            server_parameters: MCP server parameters (stdio or sse). Can be a list if you want to connect multiple MCPs at once.
        """

    def connect(self):
        """Connect to the MCP server and initialize the tools."""

    def disconnect(
        self,
        exc_type: Optional[type[BaseException]] = None,
        exc_value: Optional[BaseException] = None,
        exc_traceback: Optional[TracebackType] = None,
    ):
        """
        Disconnect from the MCP server.

        Args:
            exc_type: Exception type if an exception was raised
            exc_value: Exception value if an exception was raised
            exc_traceback: Exception traceback if an exception was raised
        """

    def get_tools(self) -> List[Tool]:
        """
        Get the Saplings tools available from the MCP server.

        Returns:
            List[Tool]: The Saplings tools available from the MCP server.
        """

    def __enter__(self) -> List[Tool]:
        """
        Connect to the MCP server and return the tools directly.

        Returns:
            List[Tool]: The Saplings tools available from the MCP server
        """

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        exc_traceback: Optional[TracebackType],
    ):
        """
        Disconnect from the MCP server.

        Args:
            exc_type: Exception type if an exception was raised
            exc_value: Exception value if an exception was raised
            exc_traceback: Exception traceback if an exception was raised
        """
```

### MCPTool

```python
class MCPTool(Tool):
    """
    A Saplings tool that wraps an MCP tool.

    This class provides a bridge between MCP tools and Saplings tools.
    """

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        output_type: str,
        mcp_tool: Any
    ):
        """
        Initialize an MCP tool.

        Args:
            name: Name of the tool
            description: Description of the tool
            parameters: Parameters of the tool
            output_type: Output type of the tool
            mcp_tool: The underlying MCP tool
        """

    def forward(self, *args, **kwargs):
        """
        Call the MCP tool.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Any: The result of the tool call
        """
```

## Usage Examples

### Basic Usage

```python
from saplings import Agent, AgentConfig
from saplings.tools import MCPClient

# Connect to an MCP server using stdio
with MCPClient({"command": "path/to/mcp/server"}) as mcp_tools:
    # Create an agent with the MCP tools
    agent = Agent(
        config=AgentConfig(
            provider="openai",
            model_name="gpt-4o",
            tools=mcp_tools
        )
    )

    # Run a task that uses the MCP tools
    import asyncio
    result = asyncio.run(agent.run(
        "Use the MCP tools to perform a task."
    ))
    print(result)
```

### Multiple MCP Servers

```python
from saplings import Agent, AgentConfig
from saplings.tools import MCPClient

# Connect to multiple MCP servers
server_parameters = [
    {"command": "path/to/first/mcp/server"},
    {"url": "http://localhost:8000/sse"},  # SSE server
]

with MCPClient(server_parameters) as mcp_tools:
    # Create an agent with tools from both MCP servers
    agent = Agent(
        config=AgentConfig(
            provider="openai",
            model_name="gpt-4o",
            tools=mcp_tools
        )
    )

    # Run a task that uses the MCP tools
    import asyncio
    result = asyncio.run(agent.run(
        "Use the MCP tools to perform a task."
    ))
    print(result)
```

### Manual Connection Management

```python
from saplings import Agent, AgentConfig
from saplings.tools import MCPClient

# Manually manage the connection
try:
    mcp_client = MCPClient({"command": "path/to/mcp/server"})
    mcp_tools = mcp_client.get_tools()

    # Create an agent with the MCP tools
    agent = Agent(
        config=AgentConfig(
            provider="openai",
            model_name="gpt-4o",
            tools=mcp_tools
        )
    )

    # Run a task that uses the MCP tools
    import asyncio
    result = asyncio.run(agent.run(
        "Use the MCP tools to perform a task."
    ))
    print(result)
finally:
    mcp_client.disconnect()
```

### Custom MCP Client

```python
from saplings.tools.mcp_client import MCPClient, MCPTool
from typing import Any, Dict, List, Optional, Union

class CustomMCPClient(MCPClient):
    """A custom MCP client with additional features."""

    def __init__(
        self,
        server_parameters: Union[
            "StdioServerParameters",
            Dict[str, Any],
            List[Union["StdioServerParameters", Dict[str, Any]]]
        ],
        auth_token: Optional[str] = None
    ):
        """
        Initialize the custom MCP client.

        Args:
            server_parameters: MCP server parameters
            auth_token: Authentication token
        """
        self.auth_token = auth_token
        super().__init__(server_parameters)

    def _convert_tools_to_saplings(self) -> List[MCPTool]:
        """
        Convert MCP tools to Saplings tools with authentication.

        Returns:
            List[MCPTool]: The converted tools
        """
        tools = super()._convert_tools_to_saplings()

        # Add authentication to all tools
        if self.auth_token:
            for tool in tools:
                tool.auth_token = self.auth_token
                original_forward = tool.forward

                def authenticated_forward(self, *args, **kwargs):
                    # Add authentication to the tool call
                    kwargs["auth_token"] = self.auth_token
                    return original_forward(*args, **kwargs)

                tool.forward = authenticated_forward.__get__(tool)

        return tools

# Use the custom MCP client
with CustomMCPClient(
    {"command": "path/to/mcp/server"},
    auth_token="your-auth-token"
) as mcp_tools:
    # Create an agent with the authenticated MCP tools
    agent = Agent(
        config=AgentConfig(
            provider="openai",
            model_name="gpt-4o",
            tools=mcp_tools
        )
    )

    # Run a task that uses the MCP tools
    import asyncio
    result = asyncio.run(agent.run(
        "Use the MCP tools to perform a task."
    ))
    print(result)
```

## Server Configuration

### Stdio Server

For stdio servers, provide the command to start the server:

```python
server_parameters = {"command": "path/to/mcp/server"}
```

### SSE Server

For SSE servers, provide the URL:

```python
server_parameters = {"url": "http://localhost:8000/sse"}
```

## Tool Usage

MCP tools are used like any other Saplings tools:

```python
# Get a specific tool
calculator_tool = next((tool for tool in mcp_tools if tool.name == "calculator"), None)

# Use the tool directly
result = calculator_tool(operation="add", a=5, b=3)
print(result)  # Output: 8

# Use the tool through an agent
import asyncio
result = asyncio.run(agent.run(
    "Calculate 5 + 3 using the calculator tool."
))
print(result)
```

## Error Handling

The MCP client provides error handling for various scenarios:

```python
from saplings.tools import MCPClient

try:
    with MCPClient({"command": "path/to/mcp/server"}) as mcp_tools:
        # Use the tools
        pass
except ImportError:
    print("MCP support not available. Please install the mcpadapt library.")
except ValueError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Best Practices

1. **Use Context Manager**: Prefer the context manager interface for automatic connection management
2. **Error Handling**: Implement robust error handling for MCP server failures
3. **Tool Documentation**: Document the available MCP tools for agent developers
4. **Authentication**: Use custom MCP clients for authentication if needed
5. **Server Selection**: Choose the appropriate server type (stdio or SSE) based on your needs

## Conclusion

The MCP client provides a powerful way to extend Saplings agents with tools from MCP servers. By connecting to one or more MCP servers, agents can access a wide range of capabilities beyond what's built into Saplings.
