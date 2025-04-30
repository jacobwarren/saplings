"""
Tests for the hot-loading system and integration with executor/planner.
"""

import asyncio
import os
import tempfile
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from saplings.core.model_adapter import LLM, LLMResponse, ModelMetadata, ModelRole
from saplings.core.plugin import PluginType, ToolPlugin
from saplings.executor import Executor, ExecutorConfig
from saplings.orchestration import GraphRunner, GraphRunnerConfig, AgentNode
from saplings.planner import SequentialPlanner, PlannerConfig
from saplings.tool_factory import (
    ToolFactory,
    ToolFactoryConfig,
    ToolTemplate,
    ToolSpecification,
    SecurityLevel,
)
from saplings.integration import (
    HotLoader,
    HotLoaderConfig,
    ToolLifecycleManager,
    IntegrationManager,
)


class TestHotLoading:
    """Tests for the hot-loading system."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        mock = MagicMock(spec=LLM)
        mock.generate.return_value = LLMResponse(
            text="def add_numbers(a: int, b: int) -> int:\n    return a + b",
            model_uri="test://model",
            usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            metadata={"model": "test-model"},
        )
        mock.get_metadata.return_value = ModelMetadata(
            name="test-model",
            provider="test-provider",
            version="1.0",
            capabilities=[],
            roles=[ModelRole.GENERAL],
            context_window=4096,
            max_tokens_per_request=1024,
        )
        return mock

    @pytest.fixture
    def tool_factory(self, mock_llm):
        """Create a ToolFactory instance for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ToolFactoryConfig(
                output_dir=temp_dir,
                security_level=SecurityLevel.MEDIUM,
                enable_code_signing=False,
            )
            yield ToolFactory(model=mock_llm, config=config)

    @pytest.fixture
    def hot_loader(self, tool_factory):
        """Create a HotLoader instance for testing."""
        config = HotLoaderConfig(
            watch_directories=[tool_factory.config.output_dir],
            auto_reload=True,
            reload_interval=1.0,
        )
        return HotLoader(config=config)

    @pytest.fixture
    def executor(self, mock_llm):
        """Create an Executor instance for testing."""
        config = ExecutorConfig()
        return Executor(model=mock_llm, config=config)

    @pytest.fixture
    def planner(self, mock_llm):
        """Create a SequentialPlanner instance for testing."""
        config = PlannerConfig()
        return SequentialPlanner(model=mock_llm, config=config)

    @pytest.fixture
    def graph_runner(self, mock_llm):
        """Create a GraphRunner instance for testing."""
        config = GraphRunnerConfig()
        return GraphRunner(model=mock_llm, config=config)

    @pytest.fixture
    def integration_manager(self, executor, planner, graph_runner, hot_loader):
        """Create an IntegrationManager instance for testing."""
        return IntegrationManager(
            executor=executor,
            planner=planner,
            graph_runner=graph_runner,
            hot_loader=hot_loader,
        )

    def test_initialization(self, hot_loader):
        """Test initialization of HotLoader."""
        assert hot_loader.config.auto_reload is True
        assert hot_loader.config.reload_interval == 1.0
        assert len(hot_loader.config.watch_directories) == 1
        assert hot_loader.tools == {}
        assert hot_loader.lifecycle_manager is not None

    @pytest.mark.asyncio
    async def test_load_tool(self, hot_loader, tool_factory):
        """Test loading a tool."""
        # Register a template
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
        tool_factory.register_template(template)

        # Create a tool specification
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

        # Create the tool
        with patch.object(tool_factory, "_validate_tool_code", return_value=(True, "")):
            tool_class = await tool_factory.create_tool(spec)

        # Load the tool
        loaded_tool = hot_loader.load_tool(tool_class)

        # Check that the tool was loaded
        assert loaded_tool == tool_class
        assert spec.id in hot_loader.tools
        assert hot_loader.tools[spec.id] == tool_class

    @pytest.mark.asyncio
    async def test_unload_tool(self, hot_loader, tool_factory):
        """Test unloading a tool."""
        # Register a template
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
        tool_factory.register_template(template)

        # Create a tool specification
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

        # Create the tool
        with patch.object(tool_factory, "_validate_tool_code", return_value=(True, "")):
            tool_class = await tool_factory.create_tool(spec)

        # Load the tool
        hot_loader.load_tool(tool_class)

        # Unload the tool
        hot_loader.unload_tool(spec.id)

        # Check that the tool was unloaded
        assert spec.id not in hot_loader.tools

    @pytest.mark.asyncio
    async def test_reload_tools(self, hot_loader, tool_factory):
        """Test reloading tools."""
        # Register a template
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
        tool_factory.register_template(template)

        # Create a tool specification
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

        # Create the tool
        with patch.object(tool_factory, "_validate_tool_code", return_value=(True, "")):
            tool_class = await tool_factory.create_tool(spec)

        # Mock the discover_plugins function
        with patch("saplings.integration.hot_loader.discover_plugins") as mock_discover:
            # Mock the get_plugins_by_type function
            with patch("saplings.integration.hot_loader.get_plugins_by_type") as mock_get_plugins:
                mock_get_plugins.return_value = {spec.id: tool_class}

                # Reload tools
                hot_loader.reload_tools()

                # Check that discover_plugins was called
                mock_discover.assert_called_once()

                # Check that the tool was loaded
                assert spec.id in hot_loader.tools
                assert hot_loader.tools[spec.id] == tool_class

    @pytest.mark.asyncio
    async def test_start_stop_auto_reload(self, hot_loader):
        """Test starting and stopping auto-reload."""
        # Start auto-reload
        hot_loader.start_auto_reload()

        # Check that auto-reload is running
        assert hot_loader._auto_reload_task is not None

        # Stop auto-reload
        hot_loader.stop_auto_reload()

        # Check that auto-reload is stopped
        assert hot_loader._auto_reload_task is None

    @pytest.mark.asyncio
    async def test_integration_with_executor(self, integration_manager, tool_factory):
        """Test integration with executor."""
        # Register a template
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
        tool_factory.register_template(template)

        # Create a tool specification
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

        # Create the tool
        with patch.object(tool_factory, "_validate_tool_code", return_value=(True, "")):
            tool_class = await tool_factory.create_tool(spec)

        # Load the tool
        integration_manager.hot_loader.load_tool(tool_class)

        # Register the tool with the executor
        integration_manager.register_tools_with_executor()

        # Check that the tool was registered with the executor
        assert hasattr(integration_manager.executor, "tools")
        assert spec.id in integration_manager.executor.tools

    @pytest.mark.asyncio
    async def test_integration_with_planner(self, integration_manager, tool_factory):
        """Test integration with planner."""
        # Register a template
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
        tool_factory.register_template(template)

        # Create a tool specification
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

        # Create the tool
        with patch.object(tool_factory, "_validate_tool_code", return_value=(True, "")):
            tool_class = await tool_factory.create_tool(spec)

        # Load the tool
        integration_manager.hot_loader.load_tool(tool_class)

        # Register the tool with the planner
        integration_manager.register_tools_with_planner()

        # Check that the tool was registered with the planner
        assert hasattr(integration_manager.planner, "tools")
        assert spec.id in integration_manager.planner.tools

    @pytest.mark.asyncio
    async def test_integration_with_graph_runner(self, integration_manager, tool_factory):
        """Test integration with graph runner."""
        # Register a template
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
        tool_factory.register_template(template)

        # Create a tool specification
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

        # Create the tool
        with patch.object(tool_factory, "_validate_tool_code", return_value=(True, "")):
            tool_class = await tool_factory.create_tool(spec)

        # Load the tool
        integration_manager.hot_loader.load_tool(tool_class)

        # Register the tool with the graph runner
        integration_manager.register_tools_with_graph_runner()

        # Create a tool agent
        agent = AgentNode(
            id="tool_agent",
            name="Tool Agent",
            role="tool_user",
            description="An agent that uses tools",
            capabilities=["using_tools"],
            metadata={"tools": {}},
        )
        integration_manager.graph_runner.register_agent(agent)

        # Register tools with the agent
        integration_manager.register_tool_with_agent(agent, tool_class)

        # Check that the tool was registered with the agent
        assert "tools" in agent.metadata
        assert spec.id in agent.metadata["tools"]

    @pytest.mark.asyncio
    async def test_hot_loading_with_file_watcher(self, integration_manager, tool_factory):
        """Test hot-loading with file watcher."""
        # Create a temporary tool file
        tool_dir = tool_factory.config.output_dir
        tool_file_path = os.path.join(tool_dir, "test_hot_load_tool.py")

        # Simple tool code
        tool_code = """
from saplings.core.plugin import ToolPlugin, PluginType

class HotLoadedTool(ToolPlugin):
    name = "Hot Loaded Tool"
    description = "A tool that was hot-loaded from disk"
    plugin_type = PluginType.TOOL
    version = "1.0.0"

    def execute(self, x: int, y: int) -> int:
        \"\"\"Add two numbers together.\"\"\"
        return x + y
"""

        # Write the tool file
        with open(tool_file_path, "w") as f:
            f.write(tool_code)

        try:
            # Create a mock tool class
            class MockHotLoadedTool(ToolPlugin):
                id = "hot_loaded_tool"  # Add explicit ID
                name = "Hot Loaded Tool"
                description = "A tool that was hot-loaded from disk"
                plugin_type = PluginType.TOOL
                version = "1.0.0"

                def execute(self, x: int, y: int) -> int:
                    """Add two numbers together."""
                    return x + y

            # Directly load the tool into the hot loader
            integration_manager.hot_loader.load_tool(MockHotLoadedTool)

            # Check that the tool was loaded
            assert "hot_loaded_tool" in integration_manager.hot_loader.tools
            assert integration_manager.hot_loader.tools["hot_loaded_tool"] == MockHotLoadedTool

            # Register with executor and planner
            integration_manager.register_tools_with_executor()
            integration_manager.register_tools_with_planner()

            # Check that the tool was registered with executor and planner
            assert hasattr(integration_manager.executor, "tools")
            assert "hot_loaded_tool" in integration_manager.executor.tools

            assert hasattr(integration_manager.planner, "tools")
            assert "hot_loaded_tool" in integration_manager.planner.tools

            # Simulate tool update
            updated_tool_code = """
from saplings.core.plugin import ToolPlugin, PluginType

class HotLoadedTool(ToolPlugin):
    name = "Hot Loaded Tool (Updated)"
    description = "An updated tool that was hot-loaded from disk"
    plugin_type = PluginType.TOOL
    version = "1.1.0"

    def execute(self, x: int, y: int) -> int:
        \"\"\"Add two numbers together with improved performance.\"\"\"
        return x + y
"""

            # Update the tool file
            with open(tool_file_path, "w") as f:
                f.write(updated_tool_code)

            # Create an updated mock tool class
            class UpdatedMockHotLoadedTool(ToolPlugin):
                id = "hot_loaded_tool"  # Add explicit ID
                name = "Hot Loaded Tool (Updated)"
                description = "An updated tool that was hot-loaded from disk"
                plugin_type = PluginType.TOOL
                version = "1.1.0"

                def execute(self, x: int, y: int) -> int:
                    """Add two numbers together with improved performance."""
                    return x + y

            # Update the tool directly
            integration_manager.hot_loader.load_tool(UpdatedMockHotLoadedTool)

            # Check that the tool was updated
            assert "hot_loaded_tool" in integration_manager.hot_loader.tools
            assert integration_manager.hot_loader.tools["hot_loaded_tool"] == UpdatedMockHotLoadedTool
            assert integration_manager.hot_loader.tools["hot_loaded_tool"].name == "Hot Loaded Tool (Updated)"
            assert integration_manager.hot_loader.tools["hot_loaded_tool"].version == "1.1.0"

        finally:
            # Stop the hot-loader
            integration_manager.hot_loader.stop_auto_reload()

            # Clean up the temporary file
            if os.path.exists(tool_file_path):
                os.remove(tool_file_path)

    @pytest.mark.asyncio
    async def test_end_to_end_integration(self, integration_manager, tool_factory):
        """Test end-to-end integration between hot-loading, tool factory, executor, and planner."""
        # Register a template
        template = ToolTemplate(
            id="data_tool",
            name="Data Processing Tool",
            description="A tool for processing data",
            template_code="""
def {{function_name}}({{parameters}}):
    \"\"\"{{description}}\"\"\"
    {{code_body}}
""",
            required_parameters=["function_name", "parameters", "description", "code_body"],
            metadata={"category": "data"},
        )
        tool_factory.register_template(template)

        # Create multiple tool specifications
        specs = [
            ToolSpecification(
                id="sum_numbers",
                name="Sum Numbers",
                description="A tool to sum a list of numbers",
                template_id="data_tool",
                parameters={
                    "function_name": "sum_numbers",
                    "parameters": "numbers: list[float]",
                    "description": "Sum a list of numbers",
                    "code_body": "return sum(numbers)",
                },
                metadata={"category": "data"},
            ),
            ToolSpecification(
                id="average_numbers",
                name="Average Numbers",
                description="A tool to calculate the average of a list of numbers",
                template_id="data_tool",
                parameters={
                    "function_name": "average_numbers",
                    "parameters": "numbers: list[float]",
                    "description": "Calculate the average of a list of numbers",
                    "code_body": "return sum(numbers) / len(numbers) if numbers else 0",
                },
                metadata={"category": "data"},
            ),
            ToolSpecification(
                id="filter_positive",
                name="Filter Positive Numbers",
                description="A tool to filter positive numbers from a list",
                template_id="data_tool",
                parameters={
                    "function_name": "filter_positive",
                    "parameters": "numbers: list[float]",
                    "description": "Filter positive numbers from a list",
                    "code_body": "return [n for n in numbers if n > 0]",
                },
                metadata={"category": "data"},
            ),
        ]

        # Create and load all tools
        tool_classes = []
        for spec in specs:
            with patch.object(tool_factory, "_validate_tool_code", return_value=(True, "")), \
                 patch.object(tool_factory, "_perform_security_checks", return_value=(True, "")):
                tool_class = await tool_factory.create_tool(spec)
                tool_classes.append(tool_class)
                integration_manager.hot_loader.load_tool(tool_class)

        # Register tools with all components
        integration_manager.register_tools_with_executor()
        integration_manager.register_tools_with_planner()
        integration_manager.register_tools_with_graph_runner()

        # Create a tool agent
        agent = AgentNode(
            id="data_agent",
            name="Data Processing Agent",
            role="data_processor",
            description="An agent that processes data",
            capabilities=["data_processing"],
            metadata={"tools": {}},
        )
        integration_manager.graph_runner.register_agent(agent)

        # Register tools with the agent
        for tool_class in tool_classes:
            integration_manager.register_tool_with_agent(agent, tool_class)

        # Mock the executor's execute method
        async def mock_execute(task, **kwargs):
            return f"Executed task: {task}"

        with patch.object(integration_manager.executor, "execute", side_effect=mock_execute):
            # Execute a task using the executor
            result = await integration_manager.executor.execute(
                "Process the data and calculate statistics",
                tools=integration_manager.executor.tools
            )

            assert "Executed task" in result

        # Mock the planner's create_plan method
        async def mock_create_plan(task, **kwargs):
            return f"Created plan for: {task}"

        with patch.object(integration_manager.planner, "create_plan", side_effect=mock_create_plan):
            # Create a plan using the planner
            result = await integration_manager.planner.create_plan(
                "Process the data and calculate statistics",
                tools=integration_manager.planner.tools
            )

            assert "Created plan for" in result

        # Mock the graph runner's negotiate method
        async def mock_negotiate(task, **kwargs):
            return f"Negotiated solution for: {task}"

        with patch.object(integration_manager.graph_runner, "negotiate", side_effect=mock_negotiate):
            # Run a negotiation using the graph runner
            result = await integration_manager.graph_runner.negotiate(
                "Process the data and calculate statistics",
                context="Using data processing tools"
            )

            assert "Negotiated solution for" in result

        # Verify that all tools are properly registered
        for spec in specs:
            # Check executor
            assert spec.id in integration_manager.executor.tools

            # Check planner
            assert spec.id in integration_manager.planner.tools

            # Check agent in graph runner
            assert spec.id in agent.metadata["tools"]

    @pytest.mark.asyncio
    async def test_tool_lifecycle_management(self, hot_loader, tool_factory):
        """Test tool lifecycle management."""
        # Register a template
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
        tool_factory.register_template(template)

        # Create a tool specification
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

        # Create the tool
        with patch.object(tool_factory, "_validate_tool_code", return_value=(True, "")):
            tool_class = await tool_factory.create_tool(spec)

        # Initialize the tool
        lifecycle_manager = hot_loader.lifecycle_manager
        lifecycle_manager.initialize_tool(tool_class)

        # Check that the tool was initialized
        assert spec.id in lifecycle_manager.initialized_tools
        assert lifecycle_manager.initialized_tools[spec.id] == tool_class

        # Update the tool
        lifecycle_manager.update_tool(tool_class)

        # Check that the tool was updated
        assert spec.id in lifecycle_manager.initialized_tools
        assert lifecycle_manager.initialized_tools[spec.id] == tool_class

        # Retire the tool
        lifecycle_manager.retire_tool(spec.id)

        # Check that the tool was retired
        assert spec.id not in lifecycle_manager.initialized_tools
        assert spec.id in lifecycle_manager.retired_tools
