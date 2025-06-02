from __future__ import annotations

"""
saplings.services.tool_service.
=============================

Encapsulates all tool-related functionality:
- Tool registration
- Tool factory
- Dynamic tool creation
"""


import logging
from typing import Any

from saplings.api.core.interfaces.tools import IToolService
from saplings.core._internal.exceptions import ToolError

logger = logging.getLogger(__name__)


class ToolService(IToolService):
    """Service that manages tools and dynamic tool creation."""

    def __init__(
        self,
        executor=None,  # We use a type-less parameter to avoid circular imports
        allowed_imports: list[str] | None = None,
        sandbox_enabled: bool = True,
        enabled: bool = True,
        trace_manager: Any | None = None,
    ) -> None:
        self.enabled = enabled
        self._trace_manager = trace_manager
        self._tools: dict[str, Any] = {}
        self._tool_factory = None

        # Initialize tool factory if enabled and available
        if enabled and executor is not None:
            try:
                # Import here to avoid circular imports
                from saplings.tool_factory.tool_factory import ToolFactory, ToolFactoryConfig

                # Create tool factory config
                # Since we don't know the exact interface, create a minimal config
                tool_factory_config = ToolFactoryConfig()

                # Create tool factory
                self._tool_factory = ToolFactory(
                    executor=executor,
                    config=tool_factory_config,
                )

                logger.info(
                    "ToolService initialized (enabled=True, sandbox=%s, allowed_imports=%s)",
                    sandbox_enabled,
                    len(allowed_imports or []),
                )
            except ImportError:
                logger.warning("Tool factory not available. Dynamic tool creation disabled.")
                logger.info("ToolService initialized (enabled=True, dynamic_tools=False)")
        else:
            logger.info("ToolService initialized (enabled=False)")

    @property
    def tools(self):
        """
        Get all tools.

        Returns
        -------
            Dict[str, Any]: All tools

        """
        return self._tools

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
        try:
            # If it's a Tool instance with a name attribute
            if hasattr(tool, "name"):
                self._tools[tool.name] = tool
                logger.info("Registered tool: %s", tool.name)
                return True
            # If it's a callable with a name
            if callable(tool) and hasattr(tool, "__name__"):
                # Try to create a simple wrapper
                name = getattr(tool, "__name__", "tool")
                description = getattr(tool, "__doc__", "No description provided")

                # Create a simple tool wrapper
                wrapped_tool = type(
                    "SimpleTool",
                    (),
                    {
                        "name": name,
                        "description": description,
                        "function": tool,
                        "__call__": lambda self, *args, **kwargs: self.function(*args, **kwargs),
                        "to_openai_function": lambda self: {
                            "name": self.name,
                            "description": self.description,
                            "parameters": {"type": "object", "properties": {}, "required": []},
                        },
                    },
                )()

                self._tools[name] = wrapped_tool
                logger.info("Registered wrapped tool: %s", name)
                return True
            logger.warning("Failed to register tool: not a valid tool or callable")
            return False
        except Exception as e:
            logger.exception("Error registering tool: %s", e)
            return False

    def get_registered_tools(self):
        """
        Get all registered tools.

        Returns
        -------
            Dict[str, Any]: Registered tools

        """
        return self._tools.copy()  # Return a copy to avoid concurrent modification

    async def create_tool(
        self,
        name: str,
        description: str,
        code: str,
        trace_id: str | None = None,
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

        Raises:
        ------
            ValueError: If tool factory is disabled or if tool creation fails

        """
        # Validate inputs
        if not name or not isinstance(name, str):
            msg = "Tool name must be a non-empty string"
            raise ValueError(msg)
        if not description or not isinstance(description, str):
            msg = "Tool description must be a non-empty string"
            raise ValueError(msg)
        if not code or not isinstance(code, str):
            msg = "Tool code must be a non-empty string"
            raise ValueError(msg)

        if not self.enabled or not self._tool_factory:
            msg = "Tool factory is disabled or not available"
            raise ValueError(msg)

        span = None
        if self._trace_manager:
            span = self._trace_manager.start_span(
                name="ToolService.create_tool",
                trace_id=trace_id,
                attributes={"component": "tool_service", "tool_name": name},
            )

        try:
            # First try to use the tool factory if available
            if self._tool_factory and hasattr(self._tool_factory, "create_tool"):
                try:
                    # Create a tool using the factory
                    # We'll use a generic approach that should work with most implementations

                    # Get the signature of the create_tool method
                    import inspect

                    sig = inspect.signature(self._tool_factory.create_tool)

                    # Prepare arguments based on the signature
                    kwargs = {}
                    for param_name in sig.parameters:
                        if param_name == "self":
                            continue
                        if param_name in {"spec", "tool_spec"}:
                            # If the method expects a spec object, create a dictionary
                            kwargs[param_name] = {
                                "name": name,
                                "description": description,
                                "code": code,
                            }
                        elif param_name == "name":
                            kwargs[param_name] = name
                        elif param_name == "description":
                            kwargs[param_name] = description
                        elif param_name in {"code", "implementation"}:
                            kwargs[param_name] = code

                    # Call the method with the appropriate arguments
                    tool = await self._tool_factory.create_tool(**kwargs)

                    # Register the tool
                    if self.register_tool(tool):
                        logger.info("Created and registered tool using factory: %s", name)
                        return tool
                    msg = f"Failed to register tool: {name}"
                    raise ToolError(msg)
                except Exception as e:
                    logger.warning(
                        f"Tool factory failed, falling back to simple implementation: {e}"
                    )
                    # Fall back to simple implementation

            # Create a simple tool directly as fallback
            # Create a simple callable that executes the code
            tool_code = f"""
def {name}(**kwargs):
    \"\"\"
    {description}
    \"\"\"
    # Tool implementation
    {code}

    # Default implementation if no code is provided
    return kwargs
"""
            # Create a namespace for the tool
            tool_namespace = {}

            # Execute the code in the namespace
            exec(tool_code, tool_namespace)

            # Get the tool function
            tool_func = tool_namespace.get(name)

            if not tool_func:
                msg = f"Failed to create tool: {name}"
                raise ValueError(msg)

            # Create a simple tool wrapper
            tool = type(
                "DynamicTool",
                (),
                {
                    "name": name,
                    "description": description,
                    "function": tool_func,
                    "__call__": lambda self, **kwargs: self.function(**kwargs),
                    "to_openai_function": lambda self: {
                        "name": self.name,
                        "description": self.description,
                        "parameters": {"type": "object", "properties": {}, "required": []},
                    },
                },
            )()

            # Register the tool
            self.register_tool(tool)

            logger.info("Created and registered tool: %s", name)
            return tool
        except Exception as e:
            logger.exception(f"Error creating tool: {e}")
            raise
        finally:
            if span and self._trace_manager:
                self._trace_manager.end_span(span.span_id)

    def prepare_functions_for_model(self):
        """
        Prepare tool definitions for the model.

        Returns
        -------
            List[Dict[str, Any]]: Tool definitions in the format expected by LLMs

        """
        functions = []
        for tool in self._tools.values():
            if hasattr(tool, "to_openai_function"):
                functions.append(tool.to_openai_function())
            elif hasattr(tool, "name") and hasattr(tool, "description"):
                # Create a simple function definition
                function_def = {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {"type": "object", "properties": {}, "required": []},
                }

                # If the tool has parameters defined, use them
                if hasattr(tool, "parameters") and isinstance(tool.parameters, dict):
                    function_def["parameters"]["properties"] = tool.parameters
                    # Extract required parameters
                    required = []
                    for param_name, param_info in tool.parameters.items():
                        if isinstance(param_info, dict) and param_info.get("required", False):
                            required.append(param_name)
                    if required:
                        function_def["parameters"]["required"] = required

                functions.append(function_def)
        return functions

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
        if tool_name not in self._tools:
            msg = f"Tool not found: {tool_name}"
            raise KeyError(msg)

        tool = self._tools[tool_name]

        span = None
        if self._trace_manager:
            span = self._trace_manager.start_span(
                name="ToolService.execute_tool",
                trace_id=trace_id,
                attributes={"component": "tool_service", "tool_name": tool_name},
            )

        try:
            # Execute the tool
            if callable(tool) or callable(tool):
                result = tool(**parameters)
            elif hasattr(tool, "function") and callable(tool.function):
                result = tool.function(**parameters)
            else:
                msg = f"Tool is not callable: {tool_name}"
                raise ValueError(msg)

            return result
        except Exception as e:
            logger.exception(f"Error executing tool {tool_name}: {e}")
            raise
        finally:
            if span and self._trace_manager:
                self._trace_manager.end_span(span.span_id)

    # Additional methods for backward compatibility with tests

    def register_tool_instance(self, tool: Any) -> bool:
        """
        Register a tool instance.

        Args:
        ----
            tool: Tool instance to register

        Returns:
        -------
            bool: Whether the tool was successfully registered

        """
        return self.register_tool(tool)

    def get_tool(self, name: str) -> Any:
        """
        Get a tool by name.

        Args:
        ----
            name: Tool name

        Returns:
        -------
            Any: The tool

        Raises:
        ------
            KeyError: If the tool is not found

        """
        if name not in self._tools:
            msg = f"Tool not found: {name}"
            raise KeyError(msg)
        return self._tools[name]

    def get_tools(self):
        """
        Get all tools.

        Returns
        -------
            Dict[str, Any]: All tools

        """
        return self.get_registered_tools()

    def get_tool_definitions(self):
        """
        Get tool definitions for LLM.

        Returns
        -------
            List[Dict[str, Any]]: Tool definitions

        """
        return self.prepare_functions_for_model()
