from __future__ import annotations

"""
Unit tests for the tool service.
"""


import pytest

from saplings.core.interfaces import IToolService
from saplings.services.tool_service import ToolService
from saplings.tools.base import Tool


class TestToolService:
    EXPECTED_COUNT_1 = 2

    """Test the tool service."""

    def setup_method(self) -> None:
        """Set up test environment."""
        # Create tool service
        self.service = ToolService()

    def test_initialization(self) -> None:
        """Test tool service initialization."""
        assert self.service.tools == {}

    def test_register_tool(self) -> None:
        """Test registering a tool."""

        # Create a simple tool
        def calculator(expression):
            return eval(expression)

        # Create a tool object
        calculator_tool = type(
            "CalculatorTool",
            (),
            {
                "name": "calculator",
                "description": "Calculate mathematical expressions",
                "function": staticmethod(
                    calculator
                ),  # Use staticmethod to preserve the original function
                "__call__": lambda _, *args, **kwargs: calculator(
                    *args, **kwargs
                ),  # Call the original function directly
            },
        )()

        # Register tool
        self.service.register_tool(calculator_tool)

        # Verify tool was registered
        assert "calculator" in self.service.tools
        assert self.service.tools["calculator"].name == "calculator"
        assert self.service.tools["calculator"].description == "Calculate mathematical expressions"
        assert self.service.tools["calculator"].function is calculator

    def test_register_tool_instance(self) -> None:
        """Test registering a tool instance."""

        # Create a tool instance
        class CalculatorTool(Tool):
            name = "calculator"
            description = "Calculate mathematical expressions"

            def __call__(self, expression):
                return eval(expression)

        tool = CalculatorTool()

        # Register tool instance
        self.service.register_tool_instance(tool)

        # Verify tool was registered
        assert "calculator" in self.service.tools
        assert self.service.tools["calculator"] is tool

    def test_get_tool(self) -> None:
        """Test getting a tool."""

        # Create and register a tool
        def calculator(expression):
            return eval(expression)

        # Create a tool object
        calculator_tool = type(
            "CalculatorTool",
            (),
            {
                "name": "calculator",
                "description": "Calculate mathematical expressions",
                "function": staticmethod(
                    calculator
                ),  # Use staticmethod to preserve the original function
                "__call__": lambda _, *args, **kwargs: calculator(
                    *args, **kwargs
                ),  # Call the original function directly
            },
        )()

        # Register tool
        self.service.register_tool(calculator_tool)

        # Get tool
        tool = self.service.get_tool("calculator")

        # Verify tool
        assert tool.name == "calculator"
        assert tool.description == "Calculate mathematical expressions"
        assert tool.function is calculator

        # Test getting non-existent tool
        with pytest.raises(KeyError):
            self.service.get_tool("non_existent")

    def test_get_tools(self) -> None:
        """Test getting all tools."""

        # Create and register tools
        def calculator(expression):
            return eval(expression)

        def translator(text: str, language):
            return f"Translated to {language}: {text}"

        # Create calculator tool object
        calculator_tool = type(
            "CalculatorTool",
            (),
            {
                "name": "calculator",
                "description": "Calculate mathematical expressions",
                "function": staticmethod(
                    calculator
                ),  # Use staticmethod to preserve the original function
                "__call__": lambda _, *args, **kwargs: calculator(
                    *args, **kwargs
                ),  # Call the original function directly
            },
        )()

        # Create translator tool object
        translator_tool = type(
            "TranslatorTool",
            (),
            {
                "name": "translator",
                "description": "Translate text to another language",
                "function": staticmethod(
                    translator
                ),  # Use staticmethod to preserve the original function
                "__call__": lambda _, *args, **kwargs: translator(
                    *args, **kwargs
                ),  # Call the original function directly
            },
        )()

        # Register tools
        self.service.register_tool(calculator_tool)
        self.service.register_tool(translator_tool)

        # Get all tools
        tools = self.service.get_tools()

        # Verify tools
        assert len(tools) == self.EXPECTED_COUNT_1
        assert "calculator" in tools
        assert "translator" in tools
        assert tools["calculator"].name == "calculator"
        assert tools["translator"].name == "translator"

    def test_get_tool_definitions(self) -> None:
        """Test getting tool definitions for LLM."""

        # Create and register tools
        def calculator(expression):
            return eval(expression)

        def translator(text: str, language):
            return f"Translated to {language}: {text}"

        # Create calculator tool object
        calculator_tool = type(
            "CalculatorTool",
            (),
            {
                "name": "calculator",
                "description": "Calculate mathematical expressions",
                "function": staticmethod(
                    calculator
                ),  # Use staticmethod to preserve the original function
                "__call__": lambda _, *args, **kwargs: calculator(
                    *args, **kwargs
                ),  # Call the original function directly
            },
        )()

        # Create translator tool object with parameters
        translator_tool = type(
            "TranslatorTool",
            (),
            {
                "name": "translator",
                "description": "Translate text to another language",
                "function": staticmethod(
                    translator
                ),  # Use staticmethod to preserve the original function
                "parameters": {
                    "text": {"type": "string", "description": "Text to translate"},
                    "language": {"type": "string", "description": "Target language"},
                },
                "__call__": lambda _, *args, **kwargs: translator(
                    *args, **kwargs
                ),  # Call the original function directly
            },
        )()

        # Register tools
        self.service.register_tool(calculator_tool)
        self.service.register_tool(translator_tool)

        # Get tool definitions
        definitions = self.service.get_tool_definitions()

        # Verify definitions
        assert len(definitions) == self.EXPECTED_COUNT_1

        # Check calculator definition
        calculator_def = next(d for d in definitions if d["name"] == "calculator")
        assert calculator_def["name"] == "calculator"
        assert calculator_def["description"] == "Calculate mathematical expressions"
        assert "parameters" in calculator_def

        # Check translator definition
        translator_def = next(d for d in definitions if d["name"] == "translator")
        assert translator_def["name"] == "translator"
        assert translator_def["description"] == "Translate text to another language"
        assert "parameters" in translator_def
        assert (
            translator_def["parameters"]["properties"]["text"]["description"] == "Text to translate"
        )
        assert (
            translator_def["parameters"]["properties"]["language"]["description"]
            == "Target language"
        )

    @pytest.mark.asyncio()
    async def test_execute_tool(self) -> None:
        """Test executing a tool."""

        # Create and register a tool
        def calculator(expression):
            return eval(expression)

        # Create a tool object
        calculator_tool = type(
            "CalculatorTool",
            (),
            {
                "name": "calculator",
                "description": "Calculate mathematical expressions",
                "function": staticmethod(
                    calculator
                ),  # Use staticmethod to preserve the original function
                "__call__": lambda _, *args, **kwargs: calculator(
                    *args, **kwargs
                ),  # Call the original function directly
            },
        )()

        # Register tool
        self.service.register_tool(calculator_tool)

        # Execute tool
        result = await self.service.execute_tool("calculator", {"expression": "2 + 2"})

        # Verify result
        assert result == 4

        # Test executing non-existent tool
        with pytest.raises(KeyError):
            await self.service.execute_tool("non_existent", {})

    @pytest.mark.asyncio()
    async def test_execute_tool_with_validation(self) -> None:
        """Test executing a tool with parameter validation."""

        # Create and register a tool with parameter schema
        def translator(text: str, language):
            return f"Translated to {language}: {text}"

        # Create translator tool object with parameters and enum
        translator_tool = type(
            "TranslatorTool",
            (),
            {
                "name": "translator",
                "description": "Translate text to another language",
                "function": staticmethod(
                    translator
                ),  # Use staticmethod to preserve the original function
                "parameters": {
                    "text": {"type": "string", "description": "Text to translate"},
                    "language": {
                        "type": "string",
                        "description": "Target language",
                        "enum": ["english", "spanish", "french"],
                    },
                },
                "__call__": lambda _, *args, **kwargs: translator(
                    *args, **kwargs
                ),  # Call the original function directly
            },
        )()

        # Register tool
        self.service.register_tool(translator_tool)

        # Execute tool with valid parameters
        result = await self.service.execute_tool(
            "translator", {"text": "Hello", "language": "spanish"}
        )

        # Verify result
        assert result == "Translated to spanish: Hello"

        # Execute tool with invalid parameters - note that the ToolService doesn't validate parameters
        # against the schema, so this should not raise an error
        result = await self.service.execute_tool(
            "translator", {"text": "Hello", "language": "invalid"}
        )
        assert result == "Translated to invalid: Hello"

    def test_interface_compliance(self) -> None:
        """Test that ToolService implements IToolService."""
        assert isinstance(self.service, IToolService)

        # Check required methods
        assert hasattr(self.service, "register_tool")
        assert hasattr(self.service, "register_tool_instance")
        assert hasattr(self.service, "get_tool")
        assert hasattr(self.service, "get_tools")
        assert hasattr(self.service, "get_tool_definitions")
        assert hasattr(self.service, "execute_tool")
