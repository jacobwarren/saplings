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
from typing import Any, Dict, List, Optional

from saplings.api.core.interfaces import IToolService, ToolConfig, ToolResult
from saplings.api.tools import ToolRegistry
from saplings.core._internal.exceptions import ToolError

logger = logging.getLogger(__name__)


class ToolService(IToolService):
    """Service that manages tools and dynamic tool creation."""

    def __init__(
        self,
        executor=None,  # Optional, will be used for lazy initialization
        allowed_imports: list[str] | None = None,
        sandbox_enabled: bool = True,
        enabled: bool = True,
        trace_manager: Any | None = None,
        registry: Optional[ToolRegistry] = None,
        template_directories: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize the tool service.

        Args:
        ----
            executor: Optional executor for tool factory (can be provided later)
            allowed_imports: List of allowed imports for dynamic tools
            sandbox_enabled: Whether to enable sandbox for dynamic tools
            enabled: Whether the tool service is enabled
            trace_manager: Optional trace manager for monitoring
            registry: Optional tool registry (creates a new one if not provided)
            template_directories: Optional list of directories containing tool templates

        """
        self.enabled = enabled
        self._trace_manager = trace_manager
        self._executor = executor
        self._allowed_imports = allowed_imports or []
        self._sandbox_enabled = sandbox_enabled
        self._tool_factory = None
        self._is_initialized = False
        self._template_directories = template_directories or []

        # Use provided registry or create a new one
        self._registry = registry or ToolRegistry()
        self._tools = {}  # For backward compatibility

        logger.info(
            "ToolService initialized (enabled=%s, sandbox=%s, allowed_imports=%s, template_dirs=%s)",
            enabled,
            sandbox_enabled,
            len(self._allowed_imports),
            len(self._template_directories),
        )

    def _initialize_tool_factory(self):
        """
        Initialize the tool factory on-demand.

        This method lazily initializes the tool factory when needed, avoiding
        circular dependencies during initialization.

        Returns
        -------
            bool: True if initialization was successful, False otherwise

        """
        # Skip if already initialized or not enabled
        if self._is_initialized or not self.enabled:
            return self._tool_factory is not None

        # Skip if no executor is available
        if self._executor is None:
            logger.warning("Cannot initialize tool factory: executor not provided")
            return False

        try:
            # Import here to avoid circular imports
            from saplings.api.tool_factory import SandboxType, ToolFactory, ToolFactoryConfig

            # Create tool factory config
            tool_factory_config = ToolFactoryConfig(
                sandbox_type=SandboxType.NONE if not self._sandbox_enabled else SandboxType.DOCKER,
            )

            # Get or create the tool factory using the enhanced API
            # This uses lazy initialization internally
            self._tool_factory = ToolFactory.create(
                executor=self._executor,
                config=tool_factory_config,
                # Optionally add template directories from config
                template_directories=getattr(self, "_template_directories", None),
            )

            self._is_initialized = True
            logger.info("Tool factory initialized on-demand")
            return True
        except ImportError:
            logger.warning("Tool factory not available. Dynamic tool creation disabled.")
            return False
        except Exception as e:
            logger.exception(f"Error initializing tool factory: {e}")
            return False

    @property
    def tools(self):
        """
        Get all tools.

        Returns
        -------
            Dict[str, Any]: All tools

        """
        # For backward compatibility, convert registry to dict
        tools_dict = {tool.name: tool for tool in self._registry.tools.values()}
        # Add any tools directly registered with the service
        tools_dict.update(self._tools)
        return tools_dict

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
                # Register with both registry and legacy dict for backward compatibility
                self._registry.register(tool)
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

                # Register with both registry and legacy dict
                # Skip registry registration for simple wrappers as they may not be compatible
                # with the registry's requirements

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
        # Combine tools from registry and legacy dict
        result = {tool.name: tool for tool in self._registry.tools.values()}
        result.update(self._tools)
        return result  # Return a copy to avoid concurrent modification

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

        if not self.enabled:
            msg = "Tool service is disabled"
            raise ValueError(msg)

        # Initialize tool factory on-demand if needed
        if not self._tool_factory and not self._initialize_tool_factory():
            msg = "Tool factory initialization failed"
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

        # First try to use the registry's built-in method
        if hasattr(self._registry, "to_openai_functions"):
            try:
                registry_functions = self._registry.to_openai_functions()
                functions.extend(registry_functions)
            except Exception as e:
                logger.warning(f"Failed to get functions from registry: {e}")

        # Process tools from the legacy dict
        for tool in self._tools.values():
            # Skip tools that are already in the registry
            if hasattr(tool, "name") and tool.name in self._registry.tools:
                continue

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
        # First check the registry
        tool = None
        if tool_name in self._registry.tools:
            tool = self._registry.tools[tool_name]
        # Then check the legacy dict
        elif tool_name in self._tools:
            tool = self._tools[tool_name]
        else:
            msg = f"Tool not found: {tool_name}"
            raise KeyError(msg)

        span = None
        if self._trace_manager:
            span = self._trace_manager.start_span(
                name="ToolService.execute_tool",
                trace_id=trace_id,
                attributes={"component": "tool_service", "tool_name": tool_name},
            )

        try:
            # Execute the tool
            if callable(tool):
                result = tool(**parameters)
            elif hasattr(tool, "function") and callable(tool.function):
                result = tool.function(**parameters)
            elif callable(tool) and callable(tool.__call__):
                result = tool(**parameters)
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

    async def execute_tool_with_config(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        config: Optional[ToolConfig] = None,
        trace_id: Optional[str] = None,
    ) -> ToolResult:
        """
        Execute a tool with configuration.

        Args:
        ----
            tool_name: Name of the tool to execute
            parameters: Tool parameters
            config: Optional tool configuration
            trace_id: Optional trace ID for monitoring

        Returns:
        -------
            ToolResult: Tool execution result with metadata

        """
        import time

        start_time = time.time()
        config = config or ToolConfig()

        span = None
        if self._trace_manager:
            span = self._trace_manager.start_span(
                name="ToolService.execute_tool_with_config",
                trace_id=trace_id,
                attributes={
                    "component": "tool_service",
                    "tool_name": tool_name,
                    "config": str(config),
                },
            )

        try:
            # Execute the tool
            result = await self.execute_tool(tool_name, parameters, trace_id)
            execution_time = time.time() - start_time

            # Return success result
            return ToolResult(
                success=True,
                output=result,
                execution_time=execution_time,
                metadata={"tool_name": tool_name, "parameters": parameters},
            )
        except Exception as e:
            execution_time = time.time() - start_time
            logger.exception(f"Error executing tool {tool_name} with config: {e}")

            # Return error result
            return ToolResult(
                success=False,
                output=None,
                execution_time=execution_time,
                error_message=str(e),
                metadata={
                    "tool_name": tool_name,
                    "parameters": parameters,
                    "error_type": type(e).__name__,
                },
            )
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
