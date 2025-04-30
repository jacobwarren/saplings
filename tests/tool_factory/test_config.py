"""
Tests for the tool factory configuration module.
"""

import pytest
from pydantic import ValidationError

from saplings.tool_factory.config import (
    ToolSpecification,
    ToolFactoryConfig,
    ToolTemplate,
    SecurityLevel,
)


class TestToolFactoryConfig:
    """Tests for the tool factory configuration classes."""

    def test_tool_template(self):
        """Test ToolTemplate configuration."""
        # Test valid configuration
        template = ToolTemplate(
            id="math_tool",
            name="Math Tool",
            description="A tool for mathematical operations",
            template_code="""
def {{function_name}}({{parameters}}):
    \"\"\"{{description}}\"\"\"
    {{code_body}}
""",
            required_parameters=["function_name", "parameters", "description", "code_body"],
            metadata={"category": "math"},
        )
        assert template.id == "math_tool"
        assert template.name == "Math Tool"
        assert template.description == "A tool for mathematical operations"
        assert "{{function_name}}" in template.template_code
        assert template.required_parameters == ["function_name", "parameters", "description", "code_body"]
        assert template.metadata == {"category": "math"}

        # Test invalid ID (empty string)
        with pytest.raises(ValidationError):
            ToolTemplate(
                id="",
                name="Math Tool",
                description="A tool for mathematical operations",
                template_code="""
def {{function_name}}({{parameters}}):
    \"\"\"{{description}}\"\"\"
    {{code_body}}
""",
                required_parameters=["function_name", "parameters", "description", "code_body"],
            )

    def test_tool_specification(self):
        """Test ToolSpecification configuration."""
        # Test valid configuration
        spec = ToolSpecification(
            id="add_numbers",
            name="Add Numbers",
            description="A tool to add two numbers",
            template_id="math_tool",
            parameters={
                "function_name": "add_numbers",
                "parameters": "a: int, b: int",
                "description": "Add two numbers together",
                "code_body": "return a + b",
            },
            metadata={"category": "math"},
        )
        assert spec.id == "add_numbers"
        assert spec.name == "Add Numbers"
        assert spec.description == "A tool to add two numbers"
        assert spec.template_id == "math_tool"
        assert spec.parameters["function_name"] == "add_numbers"
        assert spec.parameters["parameters"] == "a: int, b: int"
        assert spec.parameters["description"] == "Add two numbers together"
        assert spec.parameters["code_body"] == "return a + b"
        assert spec.metadata == {"category": "math"}

        # Test invalid ID (empty string)
        with pytest.raises(ValidationError):
            ToolSpecification(
                id="",
                name="Add Numbers",
                description="A tool to add two numbers",
                template_id="math_tool",
                parameters={
                    "function_name": "add_numbers",
                    "parameters": "a: int, b: int",
                    "description": "Add two numbers together",
                    "code_body": "return a + b",
                },
            )

        # Test invalid template_id (empty string)
        with pytest.raises(ValidationError):
            ToolSpecification(
                id="add_numbers",
                name="Add Numbers",
                description="A tool to add two numbers",
                template_id="",
                parameters={
                    "function_name": "add_numbers",
                    "parameters": "a: int, b: int",
                    "description": "Add two numbers together",
                    "code_body": "return a + b",
                },
            )

    def test_tool_factory_config(self):
        """Test ToolFactoryConfig configuration."""
        # Test default configuration
        config = ToolFactoryConfig()
        assert config.output_dir == "tools"
        assert config.security_level == SecurityLevel.MEDIUM
        assert config.enable_code_signing is False
        assert config.sandbox_type == "none"
        assert config.metadata == {}

        # Test custom configuration
        config = ToolFactoryConfig(
            output_dir="custom_tools",
            security_level=SecurityLevel.HIGH,
            enable_code_signing=True,
            sandbox_type="docker",
            metadata={"key": "value"},
        )
        assert config.output_dir == "custom_tools"
        assert config.security_level == SecurityLevel.HIGH
        assert config.enable_code_signing is True
        assert config.sandbox_type == "docker"
        assert config.metadata == {"key": "value"}

        # Test invalid output_dir (empty string)
        with pytest.raises(ValidationError):
            ToolFactoryConfig(output_dir="")
