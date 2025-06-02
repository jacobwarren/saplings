#!/usr/bin/env python3
"""
GASA (Graph-Aligned Sparse Attention) with OpenAI Example

This example demonstrates how to use GASA to optimize performance and reduce costs
when working with OpenAI models by focusing attention on relevant context.
"""

import asyncio
import os
from saplings import AgentBuilder


async def basic_gasa_setup():
    """Demonstrate basic GASA configuration with OpenAI."""
    print("=== Basic GASA Setup with OpenAI ===\n")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Skipping - OPENAI_API_KEY not set")
        return
    
    print("1. Creating agent with GASA optimization...")
    
    # Create agent with GASA enabled for OpenAI
    agent = AgentBuilder.for_openai(
        "gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
        # GASA is enabled by default in for_openai preset
        gasa_strategy="binary",
        gasa_fallback="block_diagonal",
        gasa_max_hops=3,
    ).build()
    
    print("2. Adding structured documents to demonstrate GASA effectiveness...")
    
    # Add interconnected documents that GASA can optimize
    await agent.add_document(
        content="Python is a programming language created by Guido van Rossum in 1991.",
        metadata={"topic": "python_basics", "type": "fact"}
    )
    
    await agent.add_document(
        content="Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
        metadata={"topic": "python_paradigms", "type": "concept", "relates_to": "python_basics"}
    )
    
    await agent.add_document(
        content="NumPy is a Python library for numerical computing that provides support for large multi-dimensional arrays.",
        metadata={"topic": "numpy", "type": "tool", "relates_to": "python_basics"}
    )
    
    await agent.add_document(
        content="Pandas is built on top of NumPy and provides data structures for data analysis.",
        metadata={"topic": "pandas", "type": "tool", "relates_to": ["python_basics", "numpy"]}
    )
    
    print("\n3. Testing GASA-optimized retrieval and reasoning...")
    
    # This query should trigger GASA to focus on related documents
    query = "How do NumPy and Pandas work together in Python data analysis?"
    response = await agent.run(query)
    
    print(f"Query: {query}")
    print(f"GASA-optimized response: {response}")


async def gasa_shadow_model_example():
    """Demonstrate GASA with shadow model for enhanced optimization."""
    print("\n=== GASA with Shadow Model Example ===\n")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Skipping - OPENAI_API_KEY not set")
        return
    
    print("1. Creating agent with GASA shadow model...")
    
    # Create agent with shadow model for GASA optimization
    shadow_agent = AgentBuilder.for_openai(
        "gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
        gasa_shadow_model=True,
        gasa_shadow_model_name="Qwen/Qwen3-0.6B",  # Lightweight model for attention guidance
        gasa_strategy="binary",
        gasa_max_hops=2,
    ).build()
    
    print("2. Adding a larger knowledge base...")
    
    # Add multiple documents about AI/ML topics
    ai_topics = [
        ("Machine Learning is a subset of AI that enables computers to learn without explicit programming.", "ml_definition"),
        ("Supervised learning uses labeled data to train models that can make predictions.", "supervised_learning"),
        ("Unsupervised learning finds patterns in data without labeled examples.", "unsupervised_learning"),
        ("Deep learning uses neural networks with multiple layers to learn complex patterns.", "deep_learning"),
        ("Neural networks are inspired by biological neurons and consist of interconnected nodes.", "neural_networks"),
        ("Convolutional Neural Networks (CNNs) are effective for image processing tasks.", "cnn"),
        ("Recurrent Neural Networks (RNNs) are designed for sequential data processing.", "rnn"),
        ("Transformers use attention mechanisms and have revolutionized natural language processing.", "transformers"),
        ("BERT is a transformer-based model that understands context in both directions.", "bert"),
        ("GPT models are transformer-based generative models for text generation.", "gpt"),
    ]
    
    for content, topic in ai_topics:
        await shadow_agent.add_document(
            content=content,
            metadata={"topic": topic, "category": "ai_ml"}
        )
    
    print("\n3. Testing shadow model optimization...")
    
    # Complex query that requires understanding relationships between concepts
    query = "Compare supervised and unsupervised learning, then explain how deep learning and transformers relate to these paradigms."
    response = await shadow_agent.run(query)
    
    print(f"Query: {query}")
    print(f"Shadow model response: {response}")


async def gasa_performance_comparison():
    """Compare performance with and without GASA."""
    print("\n=== GASA Performance Comparison ===\n")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Skipping - OPENAI_API_KEY not set")
        return
    
    print("1. Creating two agents: one with GASA, one without...")
    
    # Agent with GASA enabled
    gasa_agent = AgentBuilder.for_openai(
        "gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
        gasa_enabled=True,
        gasa_strategy="binary",
        gasa_max_hops=3,
    ).build()
    
    # Agent without GASA (minimal configuration)
    standard_agent = AgentBuilder.minimal(
        "openai",
        "gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
        gasa_enabled=False,
    ).build()
    
    print("2. Adding identical knowledge base to both agents...")
    
    # Knowledge about programming languages
    knowledge_base = [
        "Python is known for its simplicity and readability, making it popular for beginners.",
        "JavaScript is the primary language for web development and runs in browsers.",
        "Java is a statically typed language known for its 'write once, run anywhere' philosophy.",
        "C++ provides low-level control and is used for system programming and game development.",
        "Rust focuses on memory safety without garbage collection.",
        "Go was designed for simplicity and efficiency in concurrent programming.",
        "Swift was created by Apple for iOS and macOS application development.",
        "Kotlin is interoperable with Java and is used for Android development.",
    ]
    
    for i, content in enumerate(knowledge_base):
        metadata = {"doc_id": i, "category": "programming_languages"}
        await gasa_agent.add_document(content=content, metadata=metadata)
        await standard_agent.add_document(content=content, metadata=metadata)
    
    print("\n3. Testing identical queries on both agents...")
    
    query = "Which programming languages would you recommend for a beginner who wants to eventually do web development?"
    
    print("   Testing GASA-enabled agent...")
    import time
    
    start_time = time.time()
    gasa_response = await gasa_agent.run(query)
    gasa_time = time.time() - start_time
    
    print("   Testing standard agent...")
    start_time = time.time()
    standard_response = await standard_agent.run(query)
    standard_time = time.time() - start_time
    
    print(f"\nQuery: {query}")
    print(f"\nGASA Agent Response ({gasa_time:.2f}s):\n{gasa_response}")
    print(f"\nStandard Agent Response ({standard_time:.2f}s):\n{standard_response}")
    print(f"\nTime difference: {abs(gasa_time - standard_time):.2f}s")


async def gasa_configuration_tuning():
    """Demonstrate different GASA configuration options."""
    print("\n=== GASA Configuration Tuning ===\n")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Skipping - OPENAI_API_KEY not set")
        return
    
    print("1. Testing different GASA strategies...")
    
    # Test different GASA configurations
    configs = [
        {
            "name": "Conservative (max_hops=1)",
            "gasa_max_hops": 1,
            "gasa_strategy": "binary",
            "gasa_fallback": "block_diagonal"
        },
        {
            "name": "Balanced (max_hops=3)",
            "gasa_max_hops": 3,
            "gasa_strategy": "binary", 
            "gasa_fallback": "prompt_composer"
        },
        {
            "name": "Aggressive (max_hops=5)",
            "gasa_max_hops": 5,
            "gasa_strategy": "mask",
            "gasa_fallback": "block_diagonal"
        }
    ]
    
    # Add some test documents about technology
    test_docs = [
        "Artificial Intelligence involves creating systems that can perform tasks requiring human intelligence.",
        "Machine Learning is a method of teaching computers to learn patterns from data.",
        "Natural Language Processing enables computers to understand and generate human language.",
        "Computer Vision allows machines to interpret and understand visual information.",
        "Robotics combines AI with physical systems to create autonomous machines.",
    ]
    
    for config in configs:
        print(f"\n2. Testing {config['name']} configuration...")
        
        agent = AgentBuilder.for_openai(
            "gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
            **{k: v for k, v in config.items() if k != "name"}
        ).build()
        
        # Add documents
        for doc in test_docs:
            await agent.add_document(content=doc, metadata={"category": "tech"})
        
        # Test query
        query = "How do AI, ML, and NLP work together in modern applications?"
        response = await agent.run(query)
        
        print(f"   Query: {query}")
        print(f"   Response: {response[:150]}...")


async def gasa_fallback_strategies():
    """Demonstrate GASA fallback strategies."""
    print("\n=== GASA Fallback Strategies ===\n")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Skipping - OPENAI_API_KEY not set")
        return
    
    print("1. Testing different fallback strategies...")
    
    fallback_strategies = [
        "block_diagonal",
        "prompt_composer", 
        "full_attention"
    ]
    
    for strategy in fallback_strategies:
        print(f"\n2. Testing '{strategy}' fallback strategy...")
        
        agent = AgentBuilder.for_openai(
            "gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
            gasa_enabled=True,
            gasa_strategy="binary",
            gasa_fallback=strategy,
            gasa_max_hops=2,
        ).build()
        
        # Add a challenging document set that might trigger fallback
        complex_docs = [
            "Quantum computing uses quantum mechanical phenomena like superposition and entanglement.",
            "Classical computers use bits that are either 0 or 1, while quantum computers use qubits.",
            "Quantum algorithms like Shor's algorithm can factor large numbers exponentially faster.",
            "Quantum machine learning combines quantum computing with machine learning techniques.",
        ]
        
        for doc in complex_docs:
            await agent.add_document(content=doc, metadata={"category": "quantum"})
        
        # Query that might challenge GASA attention selection
        query = "Explain how quantum computing could revolutionize machine learning algorithms."
        response = await agent.run(query)
        
        print(f"   Fallback: {strategy}")
        print(f"   Response: {response[:200]}...")


async def main():
    """Run all GASA OpenAI examples."""
    await basic_gasa_setup()
    await gasa_shadow_model_example()
    await gasa_performance_comparison()
    await gasa_configuration_tuning()
    await gasa_fallback_strategies()
    
    print("\n=== GASA OpenAI Examples Complete ===")
    print("\nKey Benefits Demonstrated:")
    print("- Improved performance through focused attention")
    print("- Cost reduction by processing only relevant context")
    print("- Better reasoning on interconnected information")
    print("- Flexible configuration for different use cases")
    print("\nNext: Check 04_gasa_qwen3_transformers_example.py for local model GASA usage")


if __name__ == "__main__":
    asyncio.run(main())