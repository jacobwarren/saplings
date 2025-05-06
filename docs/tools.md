# Tools System

The tools system in Saplings enables agents to interact with the world by providing a standardized interface for defining, registering, and using tools.

## Overview

The tools system consists of several key components:

- **Tool**: Base class for all tools
- **ToolRegistry**: Registry for managing tools
- **ToolCollection**: Groups related tools together
- **Default Tools**: Built-in tools for common tasks
- **MCPClient**: Client for Machine Control Protocol servers
- **Tool Decorator**: Decorator for easily creating tools

This system provides a flexible and extensible way to define tools that agents can use to perform tasks like executing Python code, searching the web, or interacting with external systems.

## Core Concepts

### Tools

A tool is a function that can be called by an agent to perform a specific task. Each tool has:

- **Name**: Unique identifier for the tool
- **Description**: Explanation of what the tool does
- **Parameters**: Information about the tool's inputs
- **Output Type**: Type of the tool's output

Tools can be created by:

- Extending the `Tool` class
- Using the `@tool` decorator
- Registering a function with `register_tool`

### Tool Registry

The tool registry manages the registration and retrieval of tools. It:

- Stores all registered tools
- Provides methods for retrieving tools by name
- Converts tools to OpenAI function specifications

### Tool Service

The ToolService manages tool registration, preparation, and creation:

- **Tool Registration**: Validates tools before registration and handles duplicates
- **Function Preparation**: Converts tools to the appropriate format for different LLM providers
- **Dynamic Tool Creation**: Creates tools from code with proper error handling and validation

```python
from saplings.services import ToolService
from saplings.tools import Tool

# Create a tool service
tool_service = ToolService()

# Register a tool
class CalculatorTool(Tool):
    def __init__(self):
        self.name = "calculator"
        self.description = "Performs basic arithmetic operations"
        self.parameters = {
            "expression": {
                "type": "string",
                "description": "The mathematical expression to evaluate",
                "required": True
            }
        }
        self.output_type = "number"
        self.is_initialized = True

    def forward(self, expression: str) -> float:
        return eval(expression)

tool_service.register_tool(CalculatorTool())

# Create a tool dynamically
import asyncio
new_tool = asyncio.run(tool_service.create_tool(
    name="string_formatter",
    description="Formats a string with variables",
    code="""
def forward(template, **kwargs):
    return template.format(**kwargs)
"""
))

# Prepare functions for the model
functions = tool_service.prepare_functions_for_model()
```

### Tool Collections

Tool collections group related tools together and provide methods for:

- Loading tools from directories
- Loading tools from packages
- Saving and loading tool definitions
- Managing groups of related tools

### Default Tools

Saplings provides several default tools:

- **PythonInterpreterTool**: Executes Python code
- **FinalAnswerTool**: Provides a final answer to a question
- **UserInputTool**: Gets input from the user
- **DuckDuckGoSearchTool**: Searches the web using DuckDuckGo
- **GoogleSearchTool**: Searches the web using Google
- **VisitWebpageTool**: Visits a webpage and extracts content
- **WikipediaSearchTool**: Searches Wikipedia
- **SpeechToTextTool**: Transcribes speech to text

### MCP Client

The MCP (Machine Control Protocol) client connects to MCP servers and makes their tools available to Saplings agents. It:

- Connects to one or more MCP servers
- Converts MCP tools to Saplings tools
- Provides a context manager for easy use

## API Reference

### Tool

```python
class Tool:
    def __init__(
        self,
        func: Callable = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        output_type: Optional[str] = "any"
    ):
        """Initialize a tool."""

    def __call__(self, *args, **kwargs):
        """Call the tool's function."""

    def forward(self, *args, **kwargs):
        """Implement the tool's functionality."""

    def setup(self):
        """Perform any necessary setup before the tool is used."""

    def to_openai_function(self) -> Dict[str, Any]:
        """Convert the tool to an OpenAI function specification."""
```

### ToolRegistry

```python
class ToolRegistry:
    def __init__(self):
        """Initialize a tool registry."""

    def register(self, func_or_tool: Union[Callable, Tool]) -> None:
        """Register a function or Tool instance."""

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""

    def list(self) -> List[str]:
        """List all registered tool names."""

    def to_openai_functions(self) -> List[Dict[str, Any]]:
        """Convert all tools to OpenAI function specifications."""

    def __len__(self) -> int:
        """Get the number of registered tools."""

    def __contains__(self, name: str) -> bool:
        """Check if a tool with the given name is registered."""
```

### ToolCollection

```python
class ToolCollection:
    def __init__(self, tools: Optional[List[Tool]] = None):
        """Initialize a tool collection."""

    def add_tool(self, tool: Tool) -> None:
        """Add a tool to the collection."""

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""

    def get_tools(self) -> List[Tool]:
        """Get all tools in the collection."""

    def to_dict(self) -> Dict:
        """Convert the collection to a dictionary."""

    def save(self, path: Union[str, Path]) -> None:
        """Save the collection to a file."""

    @classmethod
    def from_directory(cls, directory: Union[str, Path]) -> "ToolCollection":
        """Load tools from Python files in a directory."""

    @classmethod
    def from_package(cls, package_name: str) -> "ToolCollection":
        """Load tools from a Python package."""

    @classmethod
    def from_registered_tools(cls) -> "ToolCollection":
        """Create a collection from all registered tools."""
```

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
        """Initialize the MCP client."""

    def connect(self):
        """Connect to the MCP server and initialize the tools."""

    def disconnect(
        self,
        exc_type: Optional[type[BaseException]] = None,
        exc_value: Optional[BaseException] = None,
        exc_traceback: Optional[TracebackType] = None,
    ):
        """Disconnect from the MCP server."""

    def get_tools(self) -> List[Tool]:
        """Get the Saplings tools available from the MCP server."""

    def __enter__(self) -> List[Tool]:
        """Connect to the MCP server and return the tools directly."""

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        exc_traceback: Optional[TracebackType],
    ):
        """Disconnect from the MCP server."""
```

### Tool Decorator

```python
def tool(name: Optional[str] = None, description: Optional[str] = None) -> Callable:
    """
    Decorator to convert a function into a Tool.

    Args:
        name: Optional name for the tool (defaults to function name)
        description: Optional description for the tool (defaults to function docstring)

    Returns:
        A decorator function
    """
```

### Helper Functions

```python
def register_tool(
    name: Optional[str] = None,
    description: Optional[str] = None
) -> Callable:
    """
    Decorator to register a function as a tool.

    Args:
        name: The name of the tool (defaults to the function name)
        description: A description of what the tool does

    Returns:
        Callable: Decorator function
    """

def get_registered_tools() -> Dict[str, Tool]:
    """
    Get all registered tools.

    Returns:
        Dict[str, Tool]: Dictionary of registered tools
    """

def get_default_tool(tool_name: str, **kwargs) -> Tool:
    """
    Get a default tool by name.

    Args:
        tool_name: Name of the tool to get
        **kwargs: Additional arguments to pass to the tool constructor

    Returns:
        Tool: The requested tool
    """

def get_all_default_tools() -> Dict[str, Tool]:
    """
    Get all default tools.

    Returns:
        Dict[str, Tool]: Dictionary mapping tool names to tool instances
    """
```

## Usage Examples

### Basic Tool Usage

```python
from saplings import Agent, AgentConfig
from saplings.tools import PythonInterpreterTool, WikipediaSearchTool

# Create tools
python_tool = PythonInterpreterTool()
wiki_tool = WikipediaSearchTool()

# Create an agent with tools
agent = Agent(
    config=AgentConfig(
        provider="openai",
        model_name="gpt-4o",
        tools=[python_tool, wiki_tool],
    )
)

# Run a task that requires using tools
import asyncio
result = asyncio.run(agent.run(
    "Search for information about Graph Attention Networks on Wikipedia, "
    "then write a Python function that creates a simple representation of "
    "a graph attention mechanism."
))
print(result)
```

### Creating a Custom Tool

```python
from saplings.tools import Tool, register_tool

# Method 1: Using the Tool class
class CalculatorTool(Tool):
    def __init__(self):
        self.name = "calculator"
        self.description = "Performs basic arithmetic operations"
        self.parameters = {
            "expression": {
                "type": "string",
                "description": "The mathematical expression to evaluate",
                "required": True
            }
        }
        self.output_type = "number"
        self.is_initialized = True

    def forward(self, expression: str) -> float:
        # Use a safer alternative to eval
        import ast
        import operator

        def safe_eval(expr):
            # Define supported operations
            operators = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
                ast.Pow: operator.pow,
                ast.USub: operator.neg,
            }

            def _eval(node):
                if isinstance(node, ast.Num):
                    return node.n
                elif isinstance(node, ast.BinOp):
                    return operators[type(node.op)](_eval(node.left), _eval(node.right))
                elif isinstance(node, ast.UnaryOp):
                    return operators[type(node.op)](_eval(node.operand))
                else:
                    raise TypeError(f"Unsupported operation: {node}")

            return _eval(ast.parse(expr, mode='eval').body)

        return safe_eval(expression)

# Method 2: Using the register_tool decorator
@register_tool(name="calculator", description="Performs basic arithmetic operations")
def calculate(expression: str) -> float:
    """
    Calculate the result of a mathematical expression.

    Args:
        expression: The mathematical expression to evaluate

    Returns:
        The result of the calculation
    """
    # Use a safer alternative to eval
    import ast
    import operator

    def safe_eval(expr):
        # Define supported operations
        operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
        }

        def _eval(node):
            if isinstance(node, ast.Num):
                return node.n
            elif isinstance(node, ast.BinOp):
                return operators[type(node.op)](_eval(node.left), _eval(node.right))
            elif isinstance(node, ast.UnaryOp):
                return operators[type(node.op)](_eval(node.operand))
            else:
                raise TypeError(f"Unsupported operation: {node}")

        return _eval(ast.parse(expr, mode='eval').body)

    return safe_eval(expression)

# Method 3: Using the tool decorator
from saplings.tools import tool

@tool(name="calculator", description="Performs basic arithmetic operations")
def calculate(expression: str) -> float:
    """
    Calculate the result of a mathematical expression.

    Args:
        expression (str): The mathematical expression to evaluate

    Returns:
        float: The result of the calculation
    """
    # Use a safer alternative to eval
    import ast
    import operator

    def safe_eval(expr):
        # Define supported operations
        operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
        }

        def _eval(node):
            if isinstance(node, ast.Num):
                return node.n
            elif isinstance(node, ast.BinOp):
                return operators[type(node.op)](_eval(node.left), _eval(node.right))
            elif isinstance(node, ast.UnaryOp):
                return operators[type(node.op)](_eval(node.operand))
            else:
                raise TypeError(f"Unsupported operation: {node}")

        return _eval(ast.parse(expr, mode='eval').body)

    return safe_eval(expression)
```

### Using Tool Collections

```python
from saplings.tools import ToolCollection, Tool
from saplings import Agent, AgentConfig

# Create a tool collection
math_tools = ToolCollection()

# Create and add tools
class AddTool(Tool):
    def __init__(self):
        self.name = "add"
        self.description = "Adds two numbers"
        self.parameters = {
            "a": {"type": "number", "description": "First number", "required": True},
            "b": {"type": "number", "description": "Second number", "required": True}
        }
        self.output_type = "number"
        self.is_initialized = True

    def forward(self, a: float, b: float) -> float:
        return a + b

class SubtractTool(Tool):
    def __init__(self):
        self.name = "subtract"
        self.description = "Subtracts two numbers"
        self.parameters = {
            "a": {"type": "number", "description": "First number", "required": True},
            "b": {"type": "number", "description": "Second number", "required": True}
        }
        self.output_type = "number"
        self.is_initialized = True

    def forward(self, a: float, b: float) -> float:
        return a - b

# Add tools to the collection
math_tools.add_tool(AddTool())
math_tools.add_tool(SubtractTool())

# Save the collection
math_tools.save("./math_tools.json")

# Load tools from a directory
tools_from_dir = ToolCollection.from_directory("./tools_directory")

# Create an agent with tools from a collection
agent = Agent(
    config=AgentConfig(
        provider="openai",
        model_name="gpt-4o",
        tools=math_tools.get_tools(),
    )
)
```

### Using MCP Client

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

### Using Default Tools

```python
from saplings.tools import get_default_tool, get_all_default_tools
from saplings import Agent, AgentConfig

# Get a specific default tool
python_tool = get_default_tool("python_interpreter")
wiki_tool = get_default_tool("wikipedia_search")

# Get all default tools
all_tools = get_all_default_tools()

# Create an agent with default tools
agent = Agent(
    config=AgentConfig(
        provider="openai",
        model_name="gpt-4o",
        tools=[python_tool, wiki_tool],
    )
)

# Run a task that uses the default tools
import asyncio
result = asyncio.run(agent.run(
    "Search for information about neural networks on Wikipedia, "
    "then write a Python function to create a simple neural network structure."
))
print(result)
```

## Default Tools

### PythonInterpreterTool

Executes Python code in a sandboxed environment.

```python
from saplings.tools import PythonInterpreterTool

# Create the tool
python_tool = PythonInterpreterTool()

# Use the tool
result = python_tool(code="""
import math

def calculate_circle_area(radius):
    return math.pi * radius ** 2

area = calculate_circle_area(5)
print(f"The area of a circle with radius 5 is {area:.2f}")
""")

print(result)
```

### WikipediaSearchTool

Searches Wikipedia for information.

```python
from saplings.tools import WikipediaSearchTool

# Create the tool
wiki_tool = WikipediaSearchTool(max_results=3)

# Use the tool
result = wiki_tool(query="Graph Attention Networks")
print(result)
```

### DuckDuckGoSearchTool

Searches the web using DuckDuckGo.

```python
from saplings.tools import DuckDuckGoSearchTool

# Create the tool
search_tool = DuckDuckGoSearchTool(max_results=5)

# Use the tool
result = search_tool(query="Graph Attention Networks research papers")
print(result)
```

### VisitWebpageTool

Visits a webpage and extracts its content.

```python
from saplings.tools import VisitWebpageTool

# Create the tool
visit_tool = VisitWebpageTool()

# Use the tool
result = visit_tool(url="https://en.wikipedia.org/wiki/Graph_attention_network")
print(result)
```

### FinalAnswerTool

Provides a final answer to a question.

```python
from saplings.tools import FinalAnswerTool

# Create the tool
answer_tool = FinalAnswerTool()

# Use the tool
result = answer_tool(answer="The capital of France is Paris.")
print(result)
```

### UserInputTool

Gets input from the user.

```python
from saplings.tools import UserInputTool

# Create the tool
input_tool = UserInputTool()

# Use the tool
result = input_tool(prompt="Please enter your name:")
print(f"Hello, {result}!")
```

### GoogleSearchTool

Searches the web using Google.

```python
from saplings.tools import GoogleSearchTool

# Create the tool
google_tool = GoogleSearchTool(api_key="your_api_key", cse_id="your_cse_id")

# Use the tool
result = google_tool(query="Graph Attention Networks research papers")
print(result)
```

### SpeechToTextTool

Transcribes speech to text.

```python
from saplings.tools import SpeechToTextTool

# Create the tool
transcriber = SpeechToTextTool()

# Use the tool
result = transcriber(audio_path="path/to/audio/file.wav")
print(result)
```

## Advanced Features

### Tool Validation

Saplings provides functions for validating tools:

```python
from saplings.tools import validate_tool_attributes, validate_tool_parameters

# Create a tool
class MyTool(Tool):
    def __init__(self):
        self.name = "my_tool"
        self.description = "My custom tool"
        self.parameters = {
            "param1": {"type": "string", "description": "Parameter 1", "required": True},
            "param2": {"type": "number", "description": "Parameter 2", "required": False}
        }
        self.output_type = "string"
        self.is_initialized = True

    def forward(self, param1: str, param2: float = 0.0) -> str:
        return f"param1: {param1}, param2: {param2}"

# Validate tool attributes
tool = MyTool()
errors = validate_tool_attributes(tool)
if errors:
    print(f"Tool validation errors: {errors}")

# Validate tool parameters
params = {"param1": "value1", "param2": 2.5}
errors = validate_tool_parameters(tool, params)
if errors:
    print(f"Parameter validation errors: {errors}")
```

### Tool Registry Management

```python
from saplings.tools import ToolRegistry, Tool

# Create a tool registry
registry = ToolRegistry()

# Create tools
class AddTool(Tool):
    def __init__(self):
        self.name = "add"
        self.description = "Adds two numbers"
        self.parameters = {
            "a": {"type": "number", "description": "First number", "required": True},
            "b": {"type": "number", "description": "Second number", "required": True}
        }
        self.output_type = "number"
        self.is_initialized = True

    def forward(self, a: float, b: float) -> float:
        return a + b

class SubtractTool(Tool):
    def __init__(self):
        self.name = "subtract"
        self.description = "Subtracts two numbers"
        self.parameters = {
            "a": {"type": "number", "description": "First number", "required": True},
            "b": {"type": "number", "description": "Second number", "required": True}
        }
        self.output_type = "number"
        self.is_initialized = True

    def forward(self, a: float, b: float) -> float:
        return a - b

# Register tools
registry.register(AddTool())
registry.register(SubtractTool())

# Check if a tool is registered
if "add" in registry:
    print("Add tool is registered")

# Get a tool by name
add_tool = registry.get("add")
if add_tool:
    result = add_tool(a=2, b=3)
    print(f"2 + 3 = {result}")

# List all registered tools
tool_names = registry.list()
print(f"Registered tools: {tool_names}")

# Convert tools to OpenAI function specifications
openai_functions = registry.to_openai_functions()
print(f"OpenAI functions: {openai_functions}")
```

### Tool Setup and Initialization

```python
from saplings.tools import Tool

class ExpensiveSetupTool(Tool):
    def __init__(self):
        self.name = "expensive_setup_tool"
        self.description = "A tool with expensive setup"
        self.parameters = {
            "input": {"type": "string", "description": "Input data", "required": True}
        }
        self.output_type = "string"
        self.is_initialized = False
        self.model = None

    def setup(self):
        """Perform expensive initialization."""
        print("Loading large model...")
        # In a real implementation, this might load a large model
        self.model = {"name": "large_model", "parameters": 1000000}
        self.is_initialized = True

    def forward(self, input: str) -> str:
        if not self.is_initialized:
            self.setup()

        # Use the model to process the input
        return f"Processed '{input}' with {self.model['name']}"

# Create the tool
tool = ExpensiveSetupTool()

# The setup method will be called automatically when needed
result = tool(input="test data")
print(result)
```

## Implementation Details

### Tool Registration Process

The tool registration process works as follows:

1. **Tool Creation**: Create a tool by extending the `Tool` class, using the `@tool` decorator, or using the `register_tool` function
2. **Parameter Extraction**: Extract parameter information from the function signature and type hints
3. **Registry Storage**: Store the tool in the registry for later retrieval

### Tool Execution Process

The tool execution process works as follows:

1. **Tool Retrieval**: Get the tool from the registry or collection
2. **Parameter Validation**: Validate the parameters against the tool's parameter specifications
3. **Tool Setup**: Call the tool's `setup` method if it hasn't been initialized
4. **Tool Execution**: Call the tool's `forward` method or the wrapped function
5. **Result Processing**: Process and return the result

### MCP Client Process

The MCP client process works as follows:

1. **Server Connection**: Connect to one or more MCP servers
2. **Tool Discovery**: Discover the tools available from the MCP servers
3. **Tool Conversion**: Convert MCP tools to Saplings tools
4. **Tool Usage**: Use the converted tools with Saplings agents
5. **Server Disconnection**: Disconnect from the MCP servers when done

## Extension Points

The tools system is designed to be extensible:

### Custom Tool Types

You can create custom tool types by extending the `Tool` class:

```python
from saplings.tools import Tool
from typing import Dict, Any

class AsyncTool(Tool):
    """A tool that supports asynchronous execution."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        output_type: str = "any"
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.output_type = output_type
        self.is_initialized = True

    def forward(self, *args, **kwargs):
        """Synchronous execution (fallback)."""
        import asyncio
        return asyncio.run(self.async_forward(*args, **kwargs))

    async def async_forward(self, *args, **kwargs):
        """Asynchronous execution."""
        raise NotImplementedError("Subclasses must implement async_forward()")

    async def async_call(self, *args, **kwargs):
        """Asynchronous call method."""
        if not self.is_initialized:
            self.setup()
        return await self.async_forward(*args, **kwargs)
```

### Custom Tool Collections

You can create custom tool collections by extending the `ToolCollection` class:

```python
from saplings.tools import ToolCollection, Tool
from typing import List, Optional, Dict
import requests

class RemoteToolCollection(ToolCollection):
    """A tool collection that loads tools from a remote API."""

    def __init__(self, api_url: str, api_key: Optional[str] = None):
        super().__init__()
        self.api_url = api_url
        self.api_key = api_key

    def load_tools(self) -> None:
        """Load tools from the remote API."""
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = requests.get(f"{self.api_url}/tools", headers=headers)
        response.raise_for_status()

        tool_data = response.json()
        for tool_info in tool_data:
            tool = self._create_tool_from_data(tool_info)
            self.add_tool(tool)

    def _create_tool_from_data(self, data: Dict) -> Tool:
        """Create a tool from API data."""
        class RemoteTool(Tool):
            def __init__(self, tool_data):
                self.name = tool_data["name"]
                self.description = tool_data["description"]
                self.parameters = tool_data["parameters"]
                self.output_type = tool_data.get("output_type", "any")
                self.tool_id = tool_data["id"]
                self.api_url = self.api_url
                self.api_key = self.api_key
                self.is_initialized = True

            def forward(self, **kwargs):
                headers = {}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"

                response = requests.post(
                    f"{self.api_url}/tools/{self.tool_id}/execute",
                    headers=headers,
                    json=kwargs
                )
                response.raise_for_status()
                return response.json()["result"]

        return RemoteTool(data)
```

### Custom MCP Clients

You can create custom MCP clients by extending the `MCPClient` class:

```python
from saplings.tools import MCPClient, Tool
from typing import List, Dict, Any, Optional

class CustomMCPClient(MCPClient):
    """A custom MCP client with additional features."""

    def __init__(self, server_parameters, auth_token: Optional[str] = None):
        self.auth_token = auth_token
        super().__init__(server_parameters)

    def _convert_tools_to_saplings(self) -> List[Tool]:
        """Convert MCP tools to Saplings tools with authentication."""
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

    def refresh_tools(self) -> None:
        """Refresh the tools from the MCP server."""
        self.disconnect()
        self.connect()
```

## Conclusion

The tools system in Saplings provides a powerful and flexible way to define, register, and use tools that enable agents to interact with the world. By providing a standardized interface for tools, it makes it easy to create and use tools for a wide range of tasks, from executing Python code to searching the web to interacting with external systems through the MCP protocol.
