#!/usr/bin/env python3
"""
Advanced Agent Usage Example

This example demonstrates advanced configurations and features of Saplings agents,
including memory management, tool integration, monitoring, and self-healing capabilities.
"""

import asyncio
import os
from saplings import AgentBuilder
from saplings.api.tools import tool


@tool(name="calculator", description="Performs mathematical calculations")
def calculate(expression: str) -> float:
    """
    Calculate the result of a mathematical expression.
    
    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 3 * 4")
    
    Returns:
        The numerical result of the calculation
    """
    # Safe evaluation for basic math expressions
    import ast
    import operator
    
    # Define allowed operations
    ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
    }
    
    def eval_expr(node):
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            return ops[type(node.op)](eval_expr(node.left), eval_expr(node.right))
        else:
            raise ValueError(f"Unsupported operation: {type(node)}")
    
    return eval_expr(ast.parse(expression, mode='eval').body)


async def advanced_configuration_example():
    """Demonstrate advanced agent configuration with all features enabled."""
    print("=== Advanced Configuration Example ===\n")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Skipping - OPENAI_API_KEY not set")
        return
    
    print("1. Creating full-featured agent with advanced configuration...")
    
    # Create agent with comprehensive configuration
    agent = (AgentBuilder()
        .with_provider("openai")
        .with_model_name("gpt-4o")
        .with_model_parameters({
            "api_key": os.getenv("OPENAI_API_KEY"),
            "temperature": 0.7,
            "max_tokens": 2048,
        })
        .with_memory_path("./advanced_agent_memory")
        .with_output_dir("./advanced_agent_output")
        .with_gasa_enabled(True)
        .with_monitoring_enabled(True)
        .with_tools([calculate])
        .build())
    
    # Test the advanced agent with a complex query
    print("\n2. Testing with complex reasoning task...")
    response = await agent.run("""
    I need help with a multi-step problem:
    1. Calculate the area of a circle with radius 5
    2. If that area represents square meters of farmland, and I can plant 3 tomato plants per square meter
    3. How many tomato plants can I plant in total?
    
    Please show your calculations step by step.
    """)
    
    print(f"Agent response:\n{response}")


async def memory_and_retrieval_example():
    """Demonstrate memory management and document retrieval."""
    print("\n=== Memory and Retrieval Example ===\n")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Skipping - OPENAI_API_KEY not set")
        return
    
    print("1. Creating agent with enhanced memory configuration...")
    
    # Create agent optimized for memory and retrieval
    agent = (AgentBuilder.standard("openai", "gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))
        .with_memory_path("./memory_agent_store")
        .build())
    
    # Add some documents to memory
    print("\n2. Adding documents to agent memory...")
    
    # Add contextual information
    await agent.add_document(
        content="""
        Quantum computing is a type of computation that harnesses quantum mechanics 
        to process information. Unlike classical computers that use bits (0 or 1), 
        quantum computers use quantum bits or qubits that can exist in superposition.
        """,
        metadata={"topic": "quantum_computing", "type": "definition"}
    )
    
    await agent.add_document(
        content="""
        Machine learning is a subset of artificial intelligence that enables computers 
        to learn and improve from experience without being explicitly programmed. 
        It uses algorithms to analyze data, identify patterns, and make predictions.
        """,
        metadata={"topic": "machine_learning", "type": "definition"}
    )
    
    await agent.add_document(
        content="""
        Python is a high-level, interpreted programming language known for its 
        simplicity and readability. It's widely used in data science, web development, 
        automation, and artificial intelligence applications.
        """,
        metadata={"topic": "python", "type": "definition"}
    )
    
    # Test retrieval-augmented generation
    print("\n3. Testing retrieval-augmented responses...")
    
    query = "Compare quantum computing and machine learning. How might they work together?"
    response = await agent.run(query)
    
    print(f"Query: {query}")
    print(f"Response: {response}")


async def gasa_optimization_example():
    """Demonstrate GASA (Graph-Aligned Sparse Attention) optimization."""
    print("\n=== GASA Optimization Example ===\n")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Skipping - OPENAI_API_KEY not set")
        return
    
    print("1. Creating agent with GASA optimization...")
    
    # Create agent with GASA optimizations
    gasa_agent = (AgentBuilder()
        .with_provider("openai")
        .with_model_name("gpt-4o")
        .with_model_parameters({
            "api_key": os.getenv("OPENAI_API_KEY"),
            "temperature": 0.5,
        })
        .with_gasa_enabled(True)
        .with_memory_path("./gasa_agent_memory")
        .build())
    
    print("\n2. Adding related documents for GASA to optimize...")
    
    # Add interconnected documents
    await gasa_agent.add_document(
        content="Neural networks are inspired by biological neural networks and consist of interconnected nodes called neurons.",
        metadata={"topic": "neural_networks", "category": "ai"}
    )
    
    await gasa_agent.add_document(
        content="Deep learning uses multiple layers of neural networks to learn complex patterns in data.",
        metadata={"topic": "deep_learning", "category": "ai"}
    )
    
    await gasa_agent.add_document(
        content="Convolutional neural networks (CNNs) are particularly effective for image recognition tasks.",
        metadata={"topic": "cnn", "category": "ai"}
    )
    
    print("\n3. Testing GASA-optimized reasoning...")
    
    query = "Explain how deep learning builds upon neural networks and give an example with CNNs"
    response = await gasa_agent.run(query)
    
    print(f"Query: {query}")
    print(f"GASA-optimized response: {response}")


async def monitoring_and_observability_example():
    """Demonstrate monitoring and observability features."""
    print("\n=== Monitoring and Observability Example ===\n")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Skipping - OPENAI_API_KEY not set")
        return
    
    print("1. Creating agent with comprehensive monitoring...")
    
    # Create agent with full monitoring enabled
    monitored_agent = (AgentBuilder.full_featured(
            "openai", 
            "gpt-4o", 
            api_key=os.getenv("OPENAI_API_KEY")
        )
        .with_output_dir("./monitored_agent_output")
        .build())
    
    print("\n2. Performing monitored operations...")
    
    # Run a series of operations that will be monitored
    tasks = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "What are the applications of deep learning?",
    ]
    
    for i, task in enumerate(tasks, 1):
        print(f"\n   Task {i}: {task}")
        response = await monitored_agent.run(task)
        print(f"   Response: {response[:100]}...")
    
    print("\n3. Monitoring data will be available in ./monitored_agent_output/")


async def custom_tool_integration_example():
    """Demonstrate custom tool integration."""
    print("\n=== Custom Tool Integration Example ===\n")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Skipping - OPENAI_API_KEY not set")
        return
    
    # Define additional custom tools
    @tool(name="text_analyzer", description="Analyzes text statistics")
    def analyze_text(text: str) -> dict:
        """Analyze text and return statistics."""
        words = text.split()
        sentences = text.split('.')
        return {
            "word_count": len(words),
            "sentence_count": len([s for s in sentences if s.strip()]),
            "character_count": len(text),
            "average_word_length": sum(len(word) for word in words) / len(words) if words else 0
        }
    
    @tool(name="unit_converter", description="Converts between units")
    def convert_units(value: float, from_unit: str, to_unit: str) -> str:
        """Convert between common units."""
        # Temperature conversions
        if from_unit.lower() == "celsius" and to_unit.lower() == "fahrenheit":
            result = (value * 9/5) + 32
            return f"{value}°C = {result}°F"
        elif from_unit.lower() == "fahrenheit" and to_unit.lower() == "celsius":
            result = (value - 32) * 5/9
            return f"{value}°F = {result}°C"
        
        # Length conversions
        elif from_unit.lower() == "meters" and to_unit.lower() == "feet":
            result = value * 3.28084
            return f"{value}m = {result:.2f}ft"
        elif from_unit.lower() == "feet" and to_unit.lower() == "meters":
            result = value / 3.28084
            return f"{value}ft = {result:.2f}m"
        
        return f"Conversion from {from_unit} to {to_unit} not supported"
    
    print("1. Creating agent with multiple custom tools...")
    
    # Create agent with custom tools
    tool_agent = (AgentBuilder.for_openai("gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))
        .with_tools([calculate, analyze_text, convert_units])
        .build())
    
    print("\n2. Testing multi-tool workflow...")
    
    response = await tool_agent.run("""
    I have a text passage: "Artificial intelligence is transforming the world. It enables machines to learn and make decisions."
    
    Please:
    1. Analyze the statistics of this text
    2. Calculate how many characters per sentence on average
    3. Convert the character count to feet (assuming 1 character = 1 inch)
    """)
    
    print(f"Multi-tool response:\n{response}")


async def self_healing_example():
    """Demonstrate self-healing capabilities."""
    print("\n=== Self-Healing Example ===\n")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Skipping - OPENAI_API_KEY not set")
        return
    
    print("1. Creating agent with self-healing enabled...")
    
    # Create agent with self-healing capabilities
    healing_agent = (AgentBuilder.standard("openai", "gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))
        .build())  # Self-healing is enabled by default in standard config
    
    print("\n2. Testing resilient operation...")
    
    # This will demonstrate the agent's ability to handle and recover from issues
    response = await healing_agent.run("""
    Please help me understand this concept, but simulate as if you encountered 
    some processing difficulties and had to recover. Explain: What is the difference 
    between supervised and unsupervised learning in machine learning?
    """)
    
    print(f"Self-healing response: {response}")


async def main():
    """Run all advanced usage examples."""
    await advanced_configuration_example()
    await memory_and_retrieval_example()
    await gasa_optimization_example()
    await monitoring_and_observability_example()
    await custom_tool_integration_example()
    await self_healing_example()
    
    print("\n=== Advanced Usage Examples Complete ===")
    print("\nNext steps:")
    print("- Explore 03_gasa_openai_example.py for detailed GASA usage")
    print("- Check 07_faiss_vector_store_example.py for advanced memory configurations")
    print("- Review 10_huggingface_research_analyzer.py for real-world applications")


if __name__ == "__main__":
    asyncio.run(main())