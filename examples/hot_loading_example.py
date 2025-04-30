"""
Example of using the hot-loading system in Saplings.

This example demonstrates how to use the HotLoader, ToolLifecycleManager, and
IntegrationManager classes to dynamically load and use tools.
"""

import asyncio
import os
import tempfile
from typing import Dict, List, Optional

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
    IntegrationManager,
    EventSystem,
    EventType,
    Event,
)


# Mock LLM for demonstration purposes
class MockLLM(LLM):
    """Mock LLM for demonstration purposes."""
    
    async def generate(self, prompt, **kwargs):
        """Generate a response."""
        # In a real application, this would call an actual LLM
        if "add" in prompt.lower():
            return LLMResponse(
                text="return a + b",
                model_uri="mock://model",
                usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
                metadata={"model": "mock-model"},
            )
        elif "multiply" in prompt.lower():
            return LLMResponse(
                text="return a * b",
                model_uri="mock://model",
                usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
                metadata={"model": "mock-model"},
            )
        elif "subtract" in prompt.lower():
            return LLMResponse(
                text="return a - b",
                model_uri="mock://model",
                usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
                metadata={"model": "mock-model"},
            )
        else:
            return LLMResponse(
                text="return a / b if b != 0 else 0",
                model_uri="mock://model",
                usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
                metadata={"model": "mock-model"},
            )
    
    def get_metadata(self):
        """Get metadata about the model."""
        return ModelMetadata(
            name="mock-model",
            provider="mock-provider",
            version="1.0",
            capabilities=[],
            roles=[ModelRole.GENERAL],
            context_window=4096,
            max_tokens_per_request=1024,
        )


async def run_hot_loading_example():
    """Run an example of the hot-loading system."""
    print("=== Hot-Loading System Example ===")
    
    # Create a mock LLM
    model = MockLLM()
    
    # Create a temporary directory for tools
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a tool factory
        tool_factory_config = ToolFactoryConfig(
            output_dir=temp_dir,
            security_level=SecurityLevel.MEDIUM,
            enable_code_signing=False,
        )
        tool_factory = ToolFactory(model=model, config=tool_factory_config)
        
        # Create a hot loader
        hot_loader_config = HotLoaderConfig(
            watch_directories=[temp_dir],
            auto_reload=True,
            reload_interval=1.0,
        )
        hot_loader = HotLoader(config=hot_loader_config)
        
        # Create an executor
        executor_config = ExecutorConfig()
        executor = Executor(model=model, config=executor_config)
        
        # Create a planner
        planner_config = PlannerConfig()
        planner = SequentialPlanner(model=model, config=planner_config)
        
        # Create a graph runner
        graph_runner_config = GraphRunnerConfig()
        graph_runner = GraphRunner(model=model, config=graph_runner_config)
        
        # Create an integration manager
        integration_manager = IntegrationManager(
            executor=executor,
            planner=planner,
            graph_runner=graph_runner,
            hot_loader=hot_loader,
        )
        
        # Register an agent with the graph runner
        agent = AgentNode(
            id="tool_agent",
            name="Tool Agent",
            role="tool_user",
            description="An agent that uses tools",
            capabilities=["using_tools"],
        )
        graph_runner.register_agent(agent)
        
        # Start the integration manager
        integration_manager.start()
        
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
        
        print("Registered template: math_tool")
        
        # Create a tool specification
        add_spec = ToolSpecification(
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
        add_tool = await tool_factory.create_tool(add_spec)
        print(f"Created tool: {add_tool.name}")
        
        # Load the tool
        hot_loader.load_tool(add_tool)
        print(f"Loaded tool: {add_spec.id}")
        
        # Wait for a moment to allow the tool to be registered
        await asyncio.sleep(1)
        
        # Check that the tool was registered with the executor
        if hasattr(executor, "tools") and add_spec.id in executor.tools:
            print(f"Tool {add_spec.id} registered with executor")
        else:
            print(f"Tool {add_spec.id} not registered with executor")
        
        # Check that the tool was registered with the planner
        if hasattr(planner, "tools") and add_spec.id in planner.tools:
            print(f"Tool {add_spec.id} registered with planner")
        else:
            print(f"Tool {add_spec.id} not registered with planner")
        
        # Check that the tool was registered with the agent
        if hasattr(agent, "tools") and add_spec.id in agent.tools:
            print(f"Tool {add_spec.id} registered with agent")
        else:
            print(f"Tool {add_spec.id} not registered with agent")
        
        # Create another tool specification
        multiply_spec = ToolSpecification(
            id="multiply_numbers",
            name="Multiply Numbers",
            description="A tool to multiply two numbers",
            template_id="math_tool",
            parameters={
                "function_name": "multiply_numbers",
                "parameters": "a: int, b: int",
                "description": "Multiply two numbers together",
                "code_body": "return a * b",
            },
            metadata={"category": "math"},
        )
        
        # Create the tool
        multiply_tool = await tool_factory.create_tool(multiply_spec)
        print(f"Created tool: {multiply_tool.name}")
        
        # Load the tool
        hot_loader.load_tool(multiply_tool)
        print(f"Loaded tool: {multiply_spec.id}")
        
        # Wait for a moment to allow the tool to be registered
        await asyncio.sleep(1)
        
        # Check that the tool was registered with the executor
        if hasattr(executor, "tools") and multiply_spec.id in executor.tools:
            print(f"Tool {multiply_spec.id} registered with executor")
        else:
            print(f"Tool {multiply_spec.id} not registered with executor")
        
        # Unload the first tool
        hot_loader.unload_tool(add_spec.id)
        print(f"Unloaded tool: {add_spec.id}")
        
        # Wait for a moment to allow the tool to be unregistered
        await asyncio.sleep(1)
        
        # Check that the tool was unregistered from the executor
        if hasattr(executor, "tools") and add_spec.id not in executor.tools:
            print(f"Tool {add_spec.id} unregistered from executor")
        else:
            print(f"Tool {add_spec.id} still registered with executor")
        
        # Stop the integration manager
        integration_manager.stop()
        print("Stopped integration manager")


async def main():
    """Run the example."""
    await run_hot_loading_example()


if __name__ == "__main__":
    asyncio.run(main())
