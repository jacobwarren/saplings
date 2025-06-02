#!/usr/bin/env python3
"""
GASA with Qwen3 via Transformers Provider Example

This example demonstrates how to use GASA with local Qwen3 models through
the transformers provider for enhanced performance without API costs.
"""

import asyncio
import os
from saplings import AgentBuilder


async def setup_local_qwen3_with_gasa():
    """Set up Qwen3 with GASA optimization."""
    print("=== Local Qwen3 with GASA Setup ===\n")
    
    print("1. Creating agent with local Qwen3 model and GASA...")
    
    # Create agent with Qwen3 via transformers provider
    agent = AgentBuilder.for_vllm(
        "Qwen/Qwen3-7B-Instruct",
        # GASA configuration optimized for local models
        gasa_enabled=True,
        gasa_strategy="binary",
        gasa_fallback="block_diagonal",
        gasa_max_hops=4,  # Can afford more hops with local models
        gasa_shadow_model=False,  # Disable shadow model for local setup
    ).build()
    
    print("2. Testing basic functionality...")
    
    # Test basic generation
    response = await agent.run("Explain what makes Qwen3 a good choice for local deployment.")
    print(f"Basic response: {response}")
    
    return agent


async def gasa_with_local_documents():
    """Demonstrate GASA effectiveness with local document processing."""
    print("\n=== GASA with Local Document Processing ===\n")
    
    print("1. Setting up Qwen3 agent for document processing...")
    
    # Create agent optimized for document processing
    doc_agent = AgentBuilder.for_vllm(
        "Qwen/Qwen3-7B-Instruct",
        gasa_enabled=True,
        gasa_strategy="mask",  # Mask strategy for better document reasoning
        gasa_max_hops=3,
        memory_path="./qwen3_documents",
    ).build()
    
    print("2. Adding comprehensive document set...")
    
    # Add documents about machine learning topics
    ml_documents = [
        {
            "content": "Linear regression is a statistical method used to model the relationship between a dependent variable and independent variables.",
            "metadata": {"topic": "linear_regression", "category": "algorithms", "difficulty": "beginner", "source": "ml_textbook"}
        },
        {
            "content": "Logistic regression is used for binary classification problems where the output is categorical.",
            "metadata": {"topic": "logistic_regression", "category": "algorithms", "difficulty": "beginner", "relates_to": "linear_regression", "source": "ml_textbook"}
        },
        {
            "content": "Decision trees make predictions by splitting data based on feature values to create a tree-like model.",
            "metadata": {"topic": "decision_trees", "category": "algorithms", "difficulty": "intermediate", "source": "ml_textbook"}
        },
        {
            "content": "Random forests combine multiple decision trees to improve prediction accuracy and reduce overfitting.",
            "metadata": {"topic": "random_forest", "category": "algorithms", "difficulty": "intermediate", "relates_to": "decision_trees", "source": "ml_textbook"}
        },
        {
            "content": "Support Vector Machines find the optimal boundary between classes by maximizing the margin.",
            "metadata": {"topic": "svm", "category": "algorithms", "difficulty": "advanced", "source": "ml_textbook"}
        },
        {
            "content": "K-means clustering groups data into k clusters based on similarity without labeled examples.",
            "metadata": {"topic": "k_means", "category": "unsupervised", "difficulty": "intermediate", "source": "ml_textbook"}
        },
        {
            "content": "Neural networks consist of interconnected layers of neurons that can learn complex patterns.",
            "metadata": {"topic": "neural_networks", "category": "deep_learning", "difficulty": "advanced", "source": "ml_textbook"}
        },
        {
            "content": "Gradient descent is an optimization algorithm used to minimize loss functions in machine learning.",
            "metadata": {"topic": "gradient_descent", "category": "optimization", "difficulty": "intermediate", "source": "ml_textbook"}
        }
    ]
    
    for doc_info in ml_documents:
        await doc_agent.add_document(
            content=doc_info["content"],
            metadata=doc_info["metadata"]
        )
    
    print("3. Testing GASA-optimized document retrieval...")
    
    # Complex query that should benefit from GASA's attention focusing
    query = "Compare supervised learning algorithms like linear regression and decision trees. How do ensemble methods like random forests improve upon individual decision trees?"
    
    response = await doc_agent.run(query)
    print(f"Query: {query}")
    print(f"GASA Response: {response}")


async def local_model_performance_optimization():
    """Demonstrate performance optimizations for local models."""
    print("\n=== Local Model Performance Optimization ===\n")
    
    print("1. Comparing different GASA configurations for local models...")
    
    # Configuration 1: Conservative (faster, less comprehensive)
    conservative_agent = AgentBuilder.for_vllm(
        "Qwen/Qwen3-7B-Instruct",
        gasa_enabled=True,
        gasa_strategy="binary",
        gasa_max_hops=2,
        gasa_fallback="block_diagonal",
        memory_path="./conservative_memory",
    ).build()
    
    # Configuration 2: Aggressive (slower, more comprehensive)
    aggressive_agent = AgentBuilder.for_vllm(
        "Qwen/Qwen3-7B-Instruct", 
        gasa_enabled=True,
        gasa_strategy="mask",
        gasa_max_hops=5,
        gasa_fallback="full_attention",
        memory_path="./aggressive_memory",
    ).build()
    
    print("2. Adding test documents to both agents...")
    
    # Programming concepts for testing
    programming_docs = [
        "Object-oriented programming organizes code into classes and objects with properties and methods.",
        "Functional programming treats computation as evaluation of mathematical functions.",
        "Procedural programming organizes code into procedures or functions that operate on data.",
        "Event-driven programming responds to events like user interactions or system notifications.",
        "Concurrent programming allows multiple tasks to execute simultaneously.",
        "Asynchronous programming enables non-blocking operations that don't wait for completion.",
    ]
    
    for i, doc in enumerate(programming_docs):
        metadata = {"doc_id": i, "category": "programming_paradigms"}
        await conservative_agent.add_document(content=doc, metadata=metadata)
        await aggressive_agent.add_document(content=doc, metadata=metadata)
    
    print("3. Testing response quality and speed...")
    
    query = "Explain the key differences between object-oriented and functional programming paradigms. When would you choose one over the other?"
    
    import time
    
    # Test conservative configuration
    print("   Testing conservative GASA configuration...")
    start_time = time.time()
    conservative_response = await conservative_agent.run(query)
    conservative_time = time.time() - start_time
    
    # Test aggressive configuration  
    print("   Testing aggressive GASA configuration...")
    start_time = time.time()
    aggressive_response = await aggressive_agent.run(query)
    aggressive_time = time.time() - start_time
    
    print(f"\nQuery: {query}")
    print(f"\nConservative GASA ({conservative_time:.2f}s):\n{conservative_response}")
    print(f"\nAggressive GASA ({aggressive_time:.2f}s):\n{aggressive_response}")
    print(f"\nTime difference: {abs(aggressive_time - conservative_time):.2f}s")


async def specialized_gasa_strategies():
    """Demonstrate specialized GASA strategies for different use cases."""
    print("\n=== Specialized GASA Strategies ===\n")
    
    strategies = [
        {
            "name": "Code Analysis",
            "strategy": "binary",
            "max_hops": 3,
            "fallback": "block_diagonal",
            "use_case": "Analyzing code relationships and dependencies"
        },
        {
            "name": "Research Documents", 
            "strategy": "mask",
            "max_hops": 4,
            "fallback": "prompt_composer",
            "use_case": "Processing academic papers and research materials"
        },
        {
            "name": "Technical Documentation",
            "strategy": "binary",
            "max_hops": 2,
            "fallback": "full_attention",
            "use_case": "Quick lookups in technical manuals"
        }
    ]
    
    for strategy_config in strategies:
        print(f"1. Testing {strategy_config['name']} strategy...")
        print(f"   Use case: {strategy_config['use_case']}")
        
        agent = AgentBuilder.for_vllm(
            "Qwen/Qwen3-7B-Instruct",
            gasa_enabled=True,
            gasa_strategy=strategy_config["strategy"],
            gasa_max_hops=strategy_config["max_hops"],
            gasa_fallback=strategy_config["fallback"],
            memory_path=f"./strategy_{strategy_config['name'].lower().replace(' ', '_')}",
        ).build()
        
        # Add relevant documents based on strategy
        if strategy_config["name"] == "Code Analysis":
            docs = [
                "Python functions are defined using the 'def' keyword followed by function name and parameters.",
                "Classes in Python are blueprints for creating objects with shared attributes and methods.",
                "Modules allow code organization by grouping related functions and classes into separate files.",
                "Packages are directories containing multiple modules with an __init__.py file.",
            ]
            query = "How do Python functions, classes, modules, and packages work together in a typical application structure?"
            
        elif strategy_config["name"] == "Research Documents":
            docs = [
                "Transformer models revolutionized NLP by using self-attention mechanisms to process sequences.",
                "BERT introduced bidirectional training to better understand context in both directions.",
                "GPT models use autoregressive generation to predict the next token in a sequence.",
                "T5 frames all NLP tasks as text-to-text generation problems.",
            ]
            query = "Compare the architectural differences between BERT and GPT models in terms of their training objectives and use cases."
            
        else:  # Technical Documentation
            docs = [
                "REST APIs use HTTP methods (GET, POST, PUT, DELETE) to perform operations on resources.",
                "JSON is a lightweight data interchange format commonly used in web APIs.",
                "Authentication tokens verify user identity and authorize access to protected resources.",
                "Rate limiting prevents abuse by restricting the number of requests per time period.",
            ]
            query = "What are the key components needed to implement a secure REST API with proper authentication and rate limiting?"
        
        # Add documents
        for doc in docs:
            metadata = {"complexity": "high", "domain": "research", "source": "research_papers"}
            await agent.add_document(content=doc, metadata=metadata)
        
        # Test the strategy
        response = await agent.run(query)
        print(f"   Query: {query}")
        print(f"   Response: {response[:200]}...\n")


async def memory_efficient_gasa():
    """Demonstrate memory-efficient GASA usage for resource-constrained environments."""
    print("\n=== Memory-Efficient GASA Configuration ===\n")
    
    print("1. Creating memory-optimized Qwen3 agent...")
    
    # Configuration optimized for lower memory usage
    memory_efficient_agent = AgentBuilder.for_vllm(
        "Qwen/Qwen3-7B-Instruct",
        gasa_enabled=True,
        gasa_strategy="binary",  # Binary is more memory efficient than mask
        gasa_max_hops=2,  # Fewer hops = less memory
        gasa_fallback="block_diagonal",  # Memory efficient fallback
        memory_path="./memory_efficient",
    ).build()
    
    print("2. Testing with large document set...")
    
    # Simulate a larger document set
    large_doc_set = []
    topics = ["ai", "ml", "data_science", "programming", "math", "statistics"]
    
    for i in range(20):  # Create 20 documents
        topic = topics[i % len(topics)]
        content = f"Document {i+1} about {topic}: This document contains detailed information about {topic} concepts and applications."
        large_doc_set.append({
            "content": content,
            "metadata": {"doc_id": i, "topic": topic, "batch": i // 5}
        })
    
    # Add documents in batches to simulate real-world usage
    for doc_info in large_doc_set:
        await memory_efficient_agent.add_document(
            content=doc_info["content"],
            metadata=doc_info["metadata"]
        )
    
    print("3. Testing memory-efficient retrieval...")
    
    query = "What are the connections between AI, machine learning, and data science? How do programming and mathematics support these fields?"
    response = await memory_efficient_agent.run(query)
    
    print(f"Query: {query}")
    print(f"Memory-efficient response: {response}")


async def main():
    """Run all Qwen3 GASA examples."""
    try:
        await setup_local_qwen3_with_gasa()
        await gasa_with_local_documents()
        await local_model_performance_optimization()
        await specialized_gasa_strategies()
        await memory_efficient_gasa()
        
        print("\n=== Qwen3 GASA Examples Complete ===")
        print("\nKey Benefits for Local Models:")
        print("- No API costs - all processing done locally")
        print("- Enhanced privacy - data never leaves your system")
        print("- Customizable performance vs. resource trade-offs")
        print("- Specialized GASA strategies for different document types")
        print("- Memory-efficient configurations for resource-constrained environments")
        
    except Exception as e:
        print(f"\nNote: This example requires a local model server or transformers library.")
        print(f"Error: {e}")
        print("\nTo run this example:")
        print("1. Install transformers: pip install transformers torch")
        print("2. Or set up vLLM server with Qwen3 model")
        print("3. Ensure sufficient GPU memory for 7B model")


if __name__ == "__main__":
    asyncio.run(main())