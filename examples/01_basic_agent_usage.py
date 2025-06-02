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
    agent = AgentBuilder.for_openai("gpt-4o", api_key=os.getenv("OPENAI_API_KEY")).build()
    
    # Simple question answering
    print("\n2. Simple question answering:")
    response = await agent.run("What is the capital of France?")
    print(f"Q: What is the capital of France?")
    print(f"A: {response}")
    
    # Method 2: Using minimal preset
    print("\n3. Creating minimal agent...")
    minimal_agent = AgentBuilder.minimal(
        "openai", 
        "gpt-4o", 
        api_key=os.getenv("OPENAI_API_KEY")
    ).build()
    
    # Text generation task
    print("\n4. Text generation:")
    story = await minimal_agent.run(
        "Write a short story about a robot learning to paint, in exactly 3 sentences."
    )
    print(f"Generated story:\n{story}")
    
    # Method 3: Synchronous usage
    print("\n5. Synchronous usage (for non-async environments):")
    sync_response = agent.run_sync("Explain quantum computing in one sentence.")
    print(f"Sync response: {sync_response}")


async def different_providers_example():
    """Demonstrate using different model providers."""
    print("\n=== Different Providers Example ===\n")
    
    # OpenAI
    if os.getenv("OPENAI_API_KEY"):
        print("1. OpenAI GPT-4o:")
        openai_agent = AgentBuilder.for_openai(
            "gpt-4o", 
            api_key=os.getenv("OPENAI_API_KEY")
        ).build()
        
        response = await openai_agent.run("Name three programming languages")
        print(f"OpenAI response: {response}")
    
    # Anthropic
    if os.getenv("ANTHROPIC_API_KEY"):
        print("\n2. Anthropic Claude:")
        anthropic_agent = AgentBuilder.for_anthropic(
            "claude-3-opus", 
            api_key=os.getenv("ANTHROPIC_API_KEY")
        ).build()
        
        response = await anthropic_agent.run("Name three programming languages")
        print(f"Anthropic response: {response}")
    
    # Local vLLM (if available)
    print("\n3. Local vLLM (requires local vLLM server):")
    try:
        vllm_agent = AgentBuilder.for_vllm("Qwen/Qwen3-7B-Instruct").build()
        
        response = await vllm_agent.run("Name three programming languages")
        print(f"vLLM response: {response}")
    except Exception as e:
        print(f"vLLM not available: {e}")


async def builder_configuration_example():
    """Demonstrate detailed builder configuration."""
    print("\n=== Builder Configuration Example ===\n")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Skipping - OPENAI_API_KEY not set")
        return
    
    # Method 4: Detailed builder configuration
    print("4. Detailed builder configuration:")
    detailed_agent = (AgentBuilder()
        .with_provider("openai")
        .with_model_name("gpt-4o")
        .with_model_parameters({
            "api_key": os.getenv("OPENAI_API_KEY"),
            "temperature": 0.3,
            "max_tokens": 500
        })
        .with_memory_path("./basic_agent_memory")
        .build())
    
    response = await detailed_agent.run("Explain the difference between AI and machine learning")
    print(f"Detailed agent response: {response}")


async def error_handling_example():
    """Demonstrate basic error handling."""
    print("\n=== Error Handling Example ===\n")
    
    try:
        # Create agent with invalid API key to demonstrate error handling
        agent = AgentBuilder.for_openai("gpt-4o", api_key="invalid-key").build()
        
        response = await agent.run("This will fail")
        print(response)
        
    except Exception as e:
        print(f"Expected error caught: {type(e).__name__}: {e}")
        
        # Create agent with proper configuration
        print("\nRetrying with proper configuration...")
        if os.getenv("OPENAI_API_KEY"):
            agent = AgentBuilder.for_openai(
                "gpt-4o", 
                api_key=os.getenv("OPENAI_API_KEY")
            ).build()
            
            response = await agent.run("Now this should work!")
            print(f"Success: {response}")


async def main():
    """Run all basic usage examples."""
    await basic_usage_example()
    await different_providers_example()
    await builder_configuration_example()
    await error_handling_example()
    
    print("\n=== Basic Usage Examples Complete ===")
    print("\nNext steps:")
    print("- Check out 02_advanced_agent_usage.py for more complex configurations")
    print("- See 13_existing_tool_usage.py for using tools with agents")
    print("- Review 03_gasa_openai_example.py for performance optimization")


if __name__ == "__main__":
    asyncio.run(main())