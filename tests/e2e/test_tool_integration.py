from __future__ import annotations

"""
End-to-end tests for tool integration.

These tests verify that tools can be integrated with agents and used correctly.
They require API keys for external services and will be skipped if they are not available.
"""


import asyncio
import os

import pytest

from saplings.agent import Agent
from saplings.agent_config import AgentConfig
from saplings.tools.base import Tool


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OpenAI API key not available")
class TestToolIntegration:
    """Test tool integration with agents."""

    def setup_method(self) -> None:
        """Set up test environment."""
        # Create test directory
        self.test_dir = os.path.join(os.path.dirname(__file__), "test_output")
        os.makedirs(self.test_dir, exist_ok=True)

        # Create agent
        self.agent = Agent(
            config=AgentConfig(
                provider="openai",
                model_name="gpt-3.5-turbo",
                memory_path=os.path.join(self.test_dir, "memory"),
                output_dir=os.path.join(self.test_dir, "output"),
                enable_tool_factory=True,
            )
        )

    def teardown_method(self) -> None:
        """Clean up after test."""
        # Clean up test directories
        import shutil

        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_python_interpreter_tool(self) -> None:
        """Test Python interpreter tool."""

        # Create a Python interpreter tool
        class PythonInterpreterTool(Tool):
            name = "python_interpreter"
            description = "Execute Python code and return the result"

            def __call__(self, code):
                # Create a dictionary to store local variables
                locals_dict = {}

                # Execute the code
                exec(code, globals(), locals_dict)

                # Return the result if available
                if "result" in locals_dict:
                    return locals_dict["result"]
                return "Code executed successfully, but no 'result' variable was defined."

        # Register tool
        self.agent.register_tool_instance(PythonInterpreterTool())

        # Run a query that requires the Python interpreter
        result = asyncio.run(
            self.agent.run(
                "Calculate the area of a circle with radius 5. Use the formula: area = pi * r^2"
            )
        )

        # Verify result
        assert result
        assert isinstance(result, str)
        assert len(result) > 0

        # The result should mention the area (approximately 78.5)
        assert "78.5" in result or "78.54" in result

    def test_calculator_tool(self) -> None:
        """Test calculator tool."""

        # Create a calculator tool
        def calculator(expression):
            return eval(expression)

        # Register tool
        self.agent.register_tool(
            name="calculator",
            description="Calculate mathematical expressions",
            function=calculator,
            parameters={
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate",
                }
            },
        )

        # Run a query that requires the calculator
        result = asyncio.run(self.agent.run("What is 123 * 456?"))

        # Verify result
        assert result
        assert isinstance(result, str)
        assert len(result) > 0

        # The result should mention the correct answer (56088)
        assert "56088" in result

    def test_weather_tool_with_mock(self) -> None:
        """Test weather tool with mock implementation."""

        # Create a weather tool with mock implementation
        def get_weather(location):
            # Mock implementation
            weather_data = {
                "New York": {"temperature": 72, "conditions": "Sunny"},
                "London": {"temperature": 65, "conditions": "Cloudy"},
                "Tokyo": {"temperature": 80, "conditions": "Rainy"},
                "Sydney": {"temperature": 85, "conditions": "Clear"},
            }

            # Default for unknown locations
            return weather_data.get(location, {"temperature": 70, "conditions": "Unknown"})

        # Register tool
        self.agent.register_tool(
            name="get_weather",
            description="Get the current weather for a location",
            function=get_weather,
            parameters={
                "location": {"type": "string", "description": "The location to get weather for"}
            },
        )

        # Run a query that requires the weather tool
        result = asyncio.run(self.agent.run("What's the weather in New York?"))

        # Verify result
        assert result
        assert isinstance(result, str)
        assert len(result) > 0

        # The result should mention the weather conditions
        assert "72" in result or "Sunny" in result

    def test_multiple_tools(self) -> None:
        """Test using multiple tools together."""

        # Create a calculator tool
        def calculator(expression):
            return eval(expression)

        # Create a unit converter tool
        def convert_units(value, from_unit, to_unit):
            # Simple conversion factors
            conversions = {
                ("feet", "meters"): 0.3048,
                ("meters", "feet"): 3.28084,
                ("miles", "kilometers"): 1.60934,
                ("kilometers", "miles"): 0.621371,
                ("celsius", "fahrenheit"): lambda c: c * 9 / 5 + 32,
                ("fahrenheit", "celsius"): lambda f: (f - 32) * 5 / 9,
            }

            # Get conversion factor
            factor = conversions.get((from_unit.lower(), to_unit.lower()))

            if factor is None:
                return f"Conversion from {from_unit} to {to_unit} is not supported"

            # Apply conversion
            result = factor(value) if callable(factor) else value * factor

            return f"{value} {from_unit} = {result:.2f} {to_unit}"

        # Register tools
        self.agent.register_tool(
            name="calculator",
            description="Calculate mathematical expressions",
            function=calculator,
            parameters={
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate",
                }
            },
        )

        self.agent.register_tool(
            name="convert_units",
            description="Convert between different units of measurement",
            function=convert_units,
            parameters={
                "value": {"type": "number", "description": "The value to convert"},
                "from_unit": {"type": "string", "description": "The source unit"},
                "to_unit": {"type": "string", "description": "The target unit"},
            },
        )

        # Run a query that requires both tools
        result = asyncio.run(
            self.agent.run(
                "If I have a room that is 15 feet long and 12 feet wide, what is the area in square meters?"
            )
        )

        # Verify result
        assert result
        assert isinstance(result, str)
        assert len(result) > 0

        # The result should mention the correct answer (approximately 16.72 square meters)
        assert "16.7" in result or "16.8" in result or "16.72" in result
