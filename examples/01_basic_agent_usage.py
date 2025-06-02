#!/usr/bin/env python3
"""
Basic Agent Usage Example

This example demonstrates the simplest way to create and use a Saplings agent
for basic text generation and question answering tasks.
"""

import asyncio
import os
from saplings import AgentBuilder


async def basic_usage_example():
    """Demonstrate basic agent creation and usage."""
    print("=== Basic Agent Usage Example ===\n")
    
    # Method 1: Using preset factory method (recommended)
    print("1. Creating agent with preset factory method...")
    agent = AgentBuilder.for_openai(
        "gpt-4o", 
        api_key=os.getenv("OPENAI_API_KEY"),
        # Use minimal validation to avoid validator errors
        executor_validation_type="basic"
    ).build()
    
    # Simple question answering
    print("\n2. Simple question answering:")
    response = await agent.run("What is the capital of France?")
    print(f"Q: What is the capital of France?")
    print(f"A: {response}")
    
    # Text generation task
    print("\n3. Text generation:")
    story = await agent.run(
        "Write a short story about a robot learning to paint, in exactly 3 sentences."
    )
    print(f"Generated story:\n{story}")
    
    # Method 3: Synchronous usage
    print("\n4. Synchronous usage (for non-async environments):")
    sync_response = agent.run_sync("Explain quantum computing in one sentence.")
    print(f"Sync response: {sync_response}")


async def main():
    """Run all basic usage examples."""
    await basic_usage_example()
    
    print("\n=== Basic Usage Examples Complete ===")
    print("\nNext steps:")
    print("- Check out 02_advanced_agent_usage.py for more complex configurations")
    print("- See 13_existing_tool_usage.py for using tools with agents")
    print("- Review 03_gasa_openai_example.py for performance optimization")


if __name__ == "__main__":
    asyncio.run(main())