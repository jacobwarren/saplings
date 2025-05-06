"""
Example of using the secure hot-loading system in Saplings.

This example demonstrates how to use the SecureHotLoader, which provides
sandboxing for dynamically loaded tools.
"""

from __future__ import annotations

import asyncio
import tempfile
from typing import Any

from saplings.core.model_adapter import LLM, LLMResponse, ModelMetadata, ModelRole
from saplings.executor import Executor, ExecutorConfig
from saplings.integration import (
    IntegrationManager,
    SecureHotLoader,
    SecureHotLoaderConfig,
)
from saplings.orchestration import AgentNode, GraphRunner, GraphRunnerConfig
from saplings.planner import PlannerConfig, SequentialPlanner
from saplings.tool_factory import (
    SandboxType,
    SecurityLevel,
    ToolFactory,
    ToolFactoryConfig,
    ToolSpecification,
    ToolTemplate,
)


# Mock LLM for demonstration purposes
class MockLLM(LLM):
    """Mock LLM for demonstration purposes."""

    async def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Generate a response."""
        # In a real application, this would call an actual LLM
        # Acknowledge kwargs to avoid unused argument warning
        _ = kwargs
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


async def run_secure_hot_loading_example():
    """Run an example of the secure hot-loading system."""
    print("=== Secure Hot-Loading System Example ===")

    # Create a mock LLM
    model = MockLLM()

    # Create a temporary directory for tools
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a tool factory
        tool_factory_config = ToolFactoryConfig(
            output_dir=temp_dir,
            security_level=SecurityLevel.HIGH,  # Higher security level
            enable_code_signing=True,  # Enable code signing
            sandbox_type=SandboxType.LOCAL,  # Use local sandbox for example (Docker or E2B would be better for production)
            blocked_imports=["os", "subprocess", "sys"],  # Block dangerous imports
        )
        tool_factory = ToolFactory(model=model, config=tool_factory_config)

        # Create a secure hot loader with sandboxing
        secure_hot_loader_config = SecureHotLoaderConfig(
            watch_directories=[temp_dir],
            auto_reload=True,
            reload_interval=1.0,
            enable_sandboxing=True,  # Enable sandboxing for dynamically loaded code
            sandbox_type=SandboxType.LOCAL,  # Use local sandbox for example
            blocked_imports=["os", "subprocess", "sys"],  # Block dangerous imports
        )
        hot_loader = SecureHotLoader(config=secure_hot_loader_config)

        # Create an executor
        executor_config = ExecutorConfig()
        executor = Executor(model=model, config=executor_config)

        # Create a planner
        planner_config = PlannerConfig()
        planner = SequentialPlanner(model=model, config=planner_config)

        # Create a graph runner
        graph_runner_config = GraphRunnerConfig()
        graph_runner = GraphRunner(model=model, config=graph_runner_config)

        # Create an integration manager with the secure hot loader
        integration_manager = IntegrationManager(
            executor=executor,
            planner=planner,
            graph_runner=graph_runner,
            hot_loader=hot_loader,  # Use SecureHotLoader here
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

        # Load the tool into the secure hot loader
        hot_loader.load_tool(add_tool)
        print(f"Loaded tool: {add_spec.id}")

        # Wait for a moment to allow the tool to be registered
        await asyncio.sleep(1)

        # Check that the tool was registered with the executor
        if hasattr(executor, "tools") and add_spec.id in executor.tools:
            print(f"Tool {add_spec.id} registered with executor")
        else:
            print(f"Tool {add_spec.id} not registered with executor")

        print("\n=== Loading a tool with potentially harmful code ===")

        # Create a tool specification with potentially harmful code
        harmful_spec = ToolSpecification(
            id="harmful_tool",
            name="Harmful Tool",
            description="A tool that tries to access the system",
            template_id="math_tool",
            parameters={
                "function_name": "harmful_access",
                "parameters": "command: str",
                "description": "Execute a command (will be blocked)",
                # This code would be dangerous without sandboxing
                "code_body": 'try:\n    import os\n    return os.system(command)\nexcept ImportError:\n    return "Blocked: OS module not allowed"',
            },
            metadata={"category": "system"},
        )

        try:
            # Create the tool
            harmful_tool = await tool_factory.create_tool(harmful_spec)
            print(f"Created tool: {harmful_tool.name}")

            # Load the tool into the secure hot loader
            hot_loader.load_tool(harmful_tool)
            print(f"Loaded tool: {harmful_spec.id}")

            # Wait for a moment to allow the tool to be registered
            await asyncio.sleep(1)

            # Create an instance of the harmful tool
            tool_instance = harmful_tool()

            # Try to execute it with a command (will be sandboxed)
            try:
                result = tool_instance.execute("echo hacked")
                print(f"Tool execution result: {result}")
            except Exception as e:
                print(f"Tool execution blocked as expected: {e}")
        except Exception as e:
            print(f"Tool creation or loading blocked as expected: {e}")

        print("\n=== Creating a tool that tries to read files ===")

        # Create a tool specification that tries to read files
        file_spec = ToolSpecification(
            id="file_reader",
            name="File Reader",
            description="A tool that tries to read files",
            template_id="math_tool",
            parameters={
                "function_name": "read_file",
                "parameters": "file_path: str",
                "description": "Read a file (will be blocked)",
                # This code would be dangerous without sandboxing
                "code_body": 'try:\n    with open(file_path, "r") as f:\n        return f.read()\nexcept Exception as e:\n    return f"Blocked: {str(e)}"',
            },
            metadata={"category": "file"},
        )

        try:
            # Create the tool
            file_tool = await tool_factory.create_tool(file_spec)
            print(f"Created tool: {file_tool.name}")

            # Load the tool into the secure hot loader
            hot_loader.load_tool(file_tool)
            print(f"Loaded tool: {file_spec.id}")

            # Wait for a moment to allow the tool to be registered
            await asyncio.sleep(1)

            # Create an instance of the file tool
            tool_instance = file_tool()

            # Try to execute it with a file path (will be sandboxed)
            try:
                result = tool_instance.execute("/etc/passwd")
                print(f"Tool execution result: {result}")
            except Exception as e:
                print(f"Tool execution blocked as expected: {e}")
        except Exception as e:
            print(f"Tool creation or loading blocked as expected: {e}")

        # Stop the integration manager
        integration_manager.stop()
        print("Stopped integration manager")

        # Cleanup the hot loader
        hot_loader.cleanup()
        print("Cleaned up hot loader")


async def main():
    """Run the example."""
    await run_secure_hot_loading_example()


if __name__ == "__main__":
    asyncio.run(main())
