# Saplings Tools

This package provides tools for use with Saplings agents.

## API Structure

The tools module follows the Saplings API separation pattern:

1. **Public API**: Exposed through `saplings.api.tools` and re-exported at the top level of the `saplings` package
2. **Internal Implementation**: Located in the `_internal` directory

## Usage

To use the tools, import them from the public API:

```python
# Recommended: Import from the top-level package
from saplings import PythonInterpreterTool, DuckDuckGoSearchTool, tool

# Alternative: Import directly from the API module
from saplings.api.tools import PythonInterpreterTool, DuckDuckGoSearchTool, tool
```

Do not import directly from the internal implementation:

```python
# Don't do this
from saplings.tools._internal import PythonInterpreterTool  # Wrong
```

## Available Tools

The following tools are available:

- `DuckDuckGoSearchTool`: Tool for searching the web using DuckDuckGo
- `FinalAnswerTool`: Tool for providing a final answer to a task
- `GoogleSearchTool`: Tool for searching the web using Google
- `PythonInterpreterTool`: Tool for executing Python code
- `SpeechToTextTool`: Tool for transcribing speech to text
- `UserInputTool`: Tool for getting input from the user
- `VisitWebpageTool`: Tool for visiting a webpage
- `WikipediaSearchTool`: Tool for searching Wikipedia

## Browser Tools

The following browser tools are available when the browser dependencies are installed:

- `ClickTool`: Tool for clicking on elements in a webpage
- `ClosePopupsTool`: Tool for closing popups in a webpage
- `GetPageTextTool`: Tool for getting the text of a webpage
- `GoBackTool`: Tool for navigating back in the browser
- `GoToTool`: Tool for navigating to a URL
- `ScrollTool`: Tool for scrolling in a webpage
- `SearchTextTool`: Tool for searching for text in a webpage
- `WaitTool`: Tool for waiting for a specified amount of time

## MCP Tools

The following MCP tools are available when the MCP dependencies are installed:

- `MCPClient`: Client for interacting with the MCP server
- `MCPTool`: Tool for executing MCP commands

## Tool Registration

Tools can be registered with the global registry using the `register_tool` function:

```python
from saplings import register_tool, Tool

class MyTool(Tool):
    name = "my_tool"
    description = "My custom tool"

    def execute(self, **kwargs):
        return "Hello from my tool!"

register_tool(MyTool())
```

Alternatively, tools can be created using the `tool` decorator:

```python
from saplings import tool

@tool(name="my_tool", description="My custom tool")
def my_tool(**kwargs):
    return "Hello from my tool!"
```

## Implementation Details

The tool implementations are located in the `_internal` directory:

- `_internal/base.py`: Base classes for tools
- `_internal/browser_tools.py`: Browser tools implementation
- `_internal/mcp_client.py`: MCP client implementation
- `_internal/python_interpreter.py`: Python interpreter tool implementation
- `_internal/search_tools.py`: Search tools implementation
- `_internal/tool_registry.py`: Tool registry implementation

These internal implementations are wrapped by the public API in `saplings.api.tools` to provide stability annotations and a consistent interface.
