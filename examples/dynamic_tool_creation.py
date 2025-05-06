"""
This example demonstrates creating tools dynamically at runtime,
with secure sandboxing provided by Saplings' hot-loading system.
"""

from __future__ import annotations

import asyncio
import os

from saplings import Agent, AgentConfig
from saplings.core.model_adapter import LLM
from saplings.executor import Executor, ExecutorConfig
from saplings.integration import SecureHotLoader, SecureHotLoaderConfig
from saplings.tool_factory import (
    SandboxType,
    SecurityLevel,
    ToolFactory,
    ToolFactoryConfig,
    ToolSpecification,
    ToolTemplate,
)


async def main():
    # Create a model
    print("Creating model...")
    model = LLM.create("openai", "gpt-4o")

    # Create a temporary directory for tools
    tool_dir = "./dynamic_tools"
    if not os.path.exists(tool_dir):
        os.makedirs(tool_dir)
        print(f"Created directory: {tool_dir}")

    print(f"Using tool directory: {tool_dir}")

    try:
        # Configure tool factory with high security
        print("Configuring tool factory...")
        tool_factory_config = ToolFactoryConfig(
            output_dir=tool_dir,
            security_level=SecurityLevel.HIGH,
            enable_code_signing=True,
            sandbox_type=SandboxType.LOCAL,
            blocked_imports=["os", "subprocess", "sys"],
        )
        tool_factory = ToolFactory(model=model, config=tool_factory_config)

        # Configure secure hot loader
        print("Configuring secure hot loader...")
        hot_loader_config = SecureHotLoaderConfig(
            watch_directories=[tool_dir],
            auto_reload=True,
            reload_interval=1.0,
            enable_sandboxing=True,
            sandbox_type=SandboxType.LOCAL,
            blocked_imports=["os", "subprocess", "sys"],
        )
        hot_loader = SecureHotLoader(config=hot_loader_config)

        # Create executor
        executor_config = ExecutorConfig()
        executor = Executor(model=model, config=executor_config)

        # Register a template for data analysis tools
        print("Registering tool template...")
        analysis_template = ToolTemplate(
            id="data_analysis_tool",
            name="Data Analysis Tool",
            description="A tool for analyzing data",
            template_code="""
def {{function_name}}({{parameters}}):
    \"\"\"{{description}}\"\"\"
    {{code_body}}
""",
            required_parameters=["function_name", "parameters", "description", "code_body"],
            metadata={"category": "data_analysis"},
        )
        tool_factory.register_template(analysis_template)

        # Create a tool specification for a correlation analyzer
        print("Creating correlation analyzer tool specification...")
        correlation_spec = ToolSpecification(
            id="correlation_analyzer",
            name="Correlation Analyzer",
            description="Analyzes correlations between variables in a dataset",
            template_id="data_analysis_tool",
            parameters={
                "function_name": "analyze_correlations",
                "parameters": "data: dict, variables: list = None",
                "description": "Calculate correlation coefficients between variables",
                "code_body": """
# This is a simple implementation - in a real tool, you might use more advanced methods
import numpy as np
import pandas as pd

# Convert input to DataFrame
df = pd.DataFrame(data)

# Use specified variables or all
if variables is None:
    variables = df.columns.tolist()

# Calculate correlation matrix
corr_matrix = df[variables].corr().to_dict()

return {
    "correlation_matrix": corr_matrix,
    "summary": {
        var: {
            "most_correlated": max(
                [(other_var, coef) for other_var, coef in var_corrs.items() if other_var != var],
                key=lambda x: abs(x[1])
            ) if len(var_corrs) > 1 else None
        }
        for var, var_corrs in corr_matrix.items()
    }
}
""",
            },
            metadata={"category": "data_analysis"},
        )

        # Create the tool
        print("Creating correlation analyzer tool...")
        correlation_tool = await tool_factory.create_tool(correlation_spec)
        print(f"Created tool: {correlation_tool.name}")

        # Load the tool into the secure hot loader
        print("Loading tool into secure hot loader...")
        hot_loader.load_tool(correlation_tool)
        print(f"Loaded tool: {correlation_spec.id}")

        # Wait for tool to be registered
        print("Waiting for tool registration...")
        await asyncio.sleep(1)

        # Create agent
        print("Creating agent...")
        agent = Agent(config=AgentConfig(provider="openai", model_name="gpt-4o"))

        # Register the tool with the agent
        print("Registering tool with agent...")
        agent.register_tool(correlation_tool)

        # Sample dataset for the tool to analyze
        sample_data = {
            "height": [170, 175, 160, 185, 155, 190, 178],
            "weight": [65, 72, 58, 80, 52, 88, 74],
            "age": [25, 30, 28, 35, 22, 40, 32],
            "salary": [50000, 70000, 48000, 90000, 45000, 100000, 75000],
        }

        # Run the agent with a task that uses the dynamic tool
        print("\nRunning agent with dynamic tool task...")
        prompt = f"""
Analyze this dataset for correlations:
{sample_data}

What variables have the strongest correlations? What insights can you provide?
"""
        result = await agent.run(prompt)
        print("\nAgent Result:")
        print(result)

        # Now create a tool that tries to do something potentially harmful
        print("\n=== Demonstrating security features ===")
        print("Creating a potentially harmful tool specification...")
        harmful_spec = ToolSpecification(
            id="harmful_tool",
            name="Harmful Tool",
            description="A tool that tries to access the system",
            template_id="data_analysis_tool",
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
            print("Attempting to create harmful tool...")
            harmful_tool = await tool_factory.create_tool(harmful_spec)
            print(f"Created tool: {harmful_tool.name}")

            # Load the tool into the secure hot loader
            print("Loading harmful tool into secure hot loader...")
            hot_loader.load_tool(harmful_tool)
            print(f"Loaded tool: {harmful_spec.id}")

            # Wait for a moment to allow the tool to be registered
            await asyncio.sleep(1)

            # Create an instance of the harmful tool
            print("Creating instance of harmful tool...")
            tool_instance = harmful_tool()

            # Try to execute it with a command (will be sandboxed)
            print("Attempting to execute harmful command...")
            try:
                result = tool_instance.execute("echo hacked")
                print(f"Tool execution result: {result}")
            except Exception as e:
                print(f"✅ Tool execution blocked as expected: {e}")
        except Exception as e:
            print(f"✅ Tool creation or loading blocked as expected: {e}")

        print("\nThis demonstration shows how Saplings provides secure sandboxing")
        print("for dynamic tools, preventing potentially harmful operations.")

    finally:
        # Clean up
        print("\nCleaning up...")
        if "hot_loader" in locals():
            hot_loader.cleanup()
            print("Hot loader cleaned up")


if __name__ == "__main__":
    asyncio.run(main())
