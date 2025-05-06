# Examples

This document provides examples of using Saplings for various tasks and use cases. For a step-by-step introduction to the library, see the [Quick Start Guide](./quick_start.md).

## Basic Examples

### Basic Agent Setup

```python
import asyncio
from saplings import Agent, AgentConfig
from saplings.memory import MemoryStore

async def main():
    # Create a memory store
    memory = MemoryStore()

    # Add a simple document
    await memory.add_document(
        "Saplings is a graph-first, self-improving agent framework that takes root in your repository or knowledge base, builds a structural map, and grows smarter each day."
    )

    # Create an agent with basic configuration
    agent = Agent(
        config=AgentConfig(
            provider="openai",
            model_name="gpt-4o",
        )
    )

    # Set the memory store
    agent.memory_store = memory

    # Run a simple query
    result = await agent.run("What is Saplings?")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

### Using Tools

```python
import asyncio
from saplings import Agent, AgentConfig
from saplings.tools import PythonInterpreterTool, WikipediaSearchTool

async def main():
    # Create tools
    python_tool = PythonInterpreterTool()
    wiki_tool = WikipediaSearchTool()

    # Create an agent with tools
    agent = Agent(
        config=AgentConfig(
            provider="openai",
            model_name="gpt-4o",
            tools=[python_tool, wiki_tool],
        )
    )

    # Run a task that requires using tools
    result = await agent.run(
        "Search for information about Graph Attention Networks on Wikipedia, "
        "then write a Python function that creates a simple representation of "
        "a graph attention mechanism."
    )
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

## Advanced Examples

### Multimodal Agent

```python
import asyncio
import os
from saplings import Agent, AgentConfig
from saplings.memory import MemoryStore, DependencyGraph
from saplings.gasa import GASAConfig
from saplings.core.model_adapter import LLM

async def main():
    # Create memory components
    memory = MemoryStore()
    graph = DependencyGraph()

    # Process text files
    text_files = [f for f in os.listdir("./dataset") if f.endswith(".txt")]
    for filename in text_files:
        with open(os.path.join("./dataset", filename), "r") as f:
            content = f.read()
            await memory.add_document(
                content=content,
                metadata={"type": "text", "source": filename}
            )

    # Process images
    image_files = [f for f in os.listdir("./dataset") if f.endswith((".jpg", ".png"))]
    for filename in image_files:
        image_path = os.path.join("./dataset", filename)
        await memory.add_document(
            content=f"Image file: {filename}",
            metadata={"type": "image", "source": filename, "image_path": image_path}
        )

    # Build dependency graph
    await graph.build_from_memory(memory)

    # Configure GASA
    gasa_config = GASAConfig(
        max_hops=3,
        mask_strategy="binary",
        add_summary_token=True,
    )

    # Create a multi-modal agent
    agent = Agent(
        config=AgentConfig(
            provider="openai",
            model_name="gpt-4o",
            enable_gasa=True,
            gasa_strategy=gasa_config.mask_strategy,
            gasa_max_hops=gasa_config.max_hops,
            supported_modalities=["text", "image"],
            memory_path="./multimodal_agent_memory",
        )
    )

    # Set memory components
    agent.memory_store = memory
    agent.dependency_graph = graph

    # Run a task that requires multimodal reasoning
    result = await agent.run(
        "Analyze all the documents in memory and describe the relationships between text and images.",
        input_modalities=["text"],
        output_modalities=["text"]
    )
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

### Research Paper Analyzer

```python
import asyncio
import os
from saplings import Agent, AgentConfig
from saplings.memory import MemoryStore, DependencyGraph
from saplings.memory.paper_chunker import PaperChunker
from saplings.core.model_adapter import LLM

async def main():
    # Create memory components
    memory = MemoryStore()
    graph = DependencyGraph()
    chunker = PaperChunker()

    # Process research papers
    papers_dir = "./papers"
    for filename in os.listdir(papers_dir):
        if filename.endswith(".pdf"):
            paper_path = os.path.join(papers_dir, filename)

            # Extract text and metadata from the paper
            paper_text, metadata = chunker.process_paper(paper_path)

            # Add the paper to memory
            await memory.add_document(
                content=paper_text,
                metadata=metadata
            )

    # Build dependency graph
    await graph.build_from_memory(memory)

    # Create an agent
    agent = Agent(
        config=AgentConfig(
            provider="openai",
            model_name="gpt-4o",
            enable_gasa=True,
        )
    )

    # Set memory components
    agent.memory_store = memory
    agent.dependency_graph = graph

    # Run analysis tasks
    tasks = [
        "Summarize the key findings across all papers",
        "Identify common themes and research gaps",
        "Compare methodologies used in different papers",
        "Suggest potential future research directions"
    ]

    for task in tasks:
        print(f"\n--- Task: {task} ---\n")
        result = await agent.run(task)
        print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

### Self-Improving Agent

```python
import asyncio
from saplings import Agent, AgentConfig
from saplings.judge import JudgeAgent, JudgeConfig, Rubric, RubricCriterion
from saplings.core.model_adapter import LLM
from saplings.self_heal import PatchGenerator

async def main():
    # Create model
    model = LLM.create("openai", "gpt-4o")

    # Create evaluation rubric
    rubric = Rubric(
        name="QA Rubric",
        description="Rubric for evaluating question answering",
        criteria=[
            RubricCriterion(
                name="correctness",
                description="How factually correct the answer is",
                weight=0.5,
                scoring_guide="5: Perfectly correct, 1: Completely incorrect"
            ),
            RubricCriterion(
                name="completeness",
                description="How complete the answer is",
                weight=0.3,
                scoring_guide="5: Fully complete, 1: Extremely incomplete"
            ),
            RubricCriterion(
                name="conciseness",
                description="How concise the answer is",
                weight=0.2,
                scoring_guide="5: Perfectly concise, 1: Extremely verbose"
            ),
        ]
    )

    # Create JudgeAgent
    judge_config = JudgeConfig(enable_detailed_feedback=True)
    judge = JudgeAgent(model=model, config=judge_config)

    # Create patch generator
    patch_generator = PatchGenerator(model=model)

    # Create agent
    agent = Agent(
        config=AgentConfig(
            provider="openai",
            model_name="gpt-4o",
            enable_self_healing=True,
        )
    )

    # Learning loop
    questions = [
        {"question": "What is the capital of France?", "expected": "Paris"},
        {"question": "What is the largest planet in our solar system?", "expected": "Jupiter"},
        {"question": "What causes rust on metal?", "expected": "Oxidation, specifically when iron reacts with oxygen and water"},
    ]

    for iteration in range(2):
        print(f"\n=== Iteration {iteration + 1} ===")

        all_results = []
        all_judgments = []

        # Test agent on all questions
        for q_idx, question_data in enumerate(questions):
            question = question_data["question"]
            expected = question_data["expected"]

            # Get agent's answer
            result = await agent.run(question)

            # Judge the answer
            judgment = await judge.judge_with_rubric(
                prompt=question,
                response=result,
                rubric=rubric,
                reference=expected
            )

            # Store results and judgments
            all_results.append({
                "question": question,
                "expected": expected,
                "answer": result,
                "judgment": judgment
            })

            # Print results
            print(f"\nQuestion: {question}")
            print(f"Expected: {expected}")
            print(f"Agent: {result}")
            print(f"Score: {judgment.score}/5.0")
            print(f"Feedback: {judgment.feedback}")

        # Generate patches based on judgments
        if iteration < 1:  # Only generate patches for the first iteration
            patches = []
            for result in all_results:
                if result["judgment"].score < 4.5:
                    patch = await patch_generator.generate_patch(
                        prompt=result["question"],
                        response=result["answer"],
                        expected=result["expected"],
                        feedback=result["judgment"].feedback
                    )
                    patches.append(patch)

            # Apply patches to the agent
            for patch in patches:
                await agent.apply_patch(patch)

            print(f"\nApplied {len(patches)} patches to the agent")

if __name__ == "__main__":
    asyncio.run(main())
```

### Dynamic Tool Creation

```python
import asyncio
from saplings import Agent, AgentConfig
from saplings.tool_factory import (
    ToolFactory, ToolFactoryConfig, ToolTemplate, ToolSpecification,
    SecurityLevel, SandboxType
)
from saplings.core.model_adapter import LLM

async def main():
    # Create model
    model = LLM.create("openai", "gpt-4o")

    # Create a tool factory
    tool_factory = ToolFactory(
        model=model,
        config=ToolFactoryConfig(
            output_dir="./tools",
            security_level=SecurityLevel.MEDIUM,
            sandbox_type=SandboxType.DOCKER,
        )
    )

    # Register a template
    template = ToolTemplate(
        id="python_tool",
        name="Python Tool",
        description="A generic Python tool",
        template_code="""
def {{function_name}}({{parameters}}):
    \"\"\"{{description}}\"\"\"
    {{code_body}}
""",
        required_parameters=["function_name", "parameters", "description", "code_body"],
    )
    tool_factory.register_template(template)

    # Create a tool specification
    spec = ToolSpecification(
        template_id="python_tool",
        function_name="calculate_circle_area",
        parameters="radius: float",
        description="Calculate the area of a circle given its radius",
        code_body="""
import math
return math.pi * radius ** 2
""",
    )

    # Generate the tool
    circle_tool = await tool_factory.create_tool(spec)

    # Create an agent with the dynamic tool
    agent = Agent(
        config=AgentConfig(
            provider="openai",
            model_name="gpt-4o",
            tools=[circle_tool],
        )
    )

    # Run a task that uses the dynamic tool
    result = await agent.run(
        "Calculate the area of a circle with radius 5 units."
    )
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

### Using MCP Client

```python
import asyncio
from saplings import Agent, AgentConfig
from saplings.tools import MCPClient

async def main():
    # Connect to an MCP server using stdio
    with MCPClient({"command": "path/to/mcp/server"}) as mcp_tools:
        # Create an agent with the MCP tools
        agent = Agent(
            config=AgentConfig(
                provider="openai",
                model_name="gpt-4o",
                tools=mcp_tools
            )
        )

        # Run a task that uses the MCP tools
        result = await agent.run(
            "Use the MCP tools to perform a task."
        )
        print(result)

    # Connect to multiple MCP servers
    server_parameters = [
        {"command": "path/to/first/mcp/server"},
        {"url": "http://localhost:8000/sse"},  # SSE server
    ]

    with MCPClient(server_parameters) as mcp_tools:
        # Create an agent with tools from both MCP servers
        agent = Agent(
            config=AgentConfig(
                provider="openai",
                model_name="gpt-4o",
                tools=mcp_tools
            )
        )

        # Run a task that uses the MCP tools
        result = await agent.run(
            "Use the MCP tools to perform a task."
        )
        print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

### Model Caching for Performance

```python
import asyncio
import time
from saplings import Agent, AgentConfig
from saplings.core.caching import ModelCache, CacheConfig
from saplings.core.model_adapter import LLM

async def main():
    # Create a model with caching enabled
    cache_config = CacheConfig(
        enable=True,
        ttl=3600,  # Cache entries expire after 1 hour
        max_size=1000,  # Maximum entries in the cache
        storage_path="./model_cache",  # Persist cache to disk
    )

    # Create model with caching
    model = LLM.create("openai", "gpt-4o")
    model_cache = ModelCache(config=cache_config)
    model.set_cache(model_cache)

    # Create an agent with the cached model
    agent = Agent(
        config=AgentConfig(
            provider="openai",
            model_name="gpt-4o",
            enable_model_caching=True,
            cache_ttl=cache_config.ttl,
            cache_max_size=cache_config.max_size,
            cache_storage_path=cache_config.storage_path,
        )
    )

    # Define some queries
    queries = [
        "What is the capital of France?",
        "What is the largest planet in our solar system?",
        "What is the capital of France?",  # Repeated query to demonstrate caching
        "What is the largest planet in our solar system?",  # Repeated query
    ]

    # Run queries and measure performance
    for i, query in enumerate(queries):
        print(f"\nQuery {i+1}: {query}")

        # Time the execution
        start_time = time.time()
        result = await agent.run(query)
        end_time = time.time()

        # Get cache info
        cache_hit = model_cache.was_last_request_cached()

        # Print results
        print(f"Result: {result}")
        print(f"Time: {(end_time - start_time):.4f} seconds")
        print(f"Cache: {'HIT' if cache_hit else 'MISS'}")

    # Print cache statistics
    stats = model_cache.get_statistics()
    print("\nCache Statistics:")
    print(f"Total requests: {stats['total_requests']}")
    print(f"Cache hits: {stats['hits']}")
    print(f"Cache misses: {stats['misses']}")
    print(f"Hit rate: {stats['hit_rate']:.2%}")
    print(f"Estimated tokens saved: {stats['tokens_saved']:,}")
    print(f"Estimated cost saved: ${stats['cost_saved']:.4f}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Example Applications

### Research Paper Analysis

Analyze research papers and generate insights:

```python
# See the Research Paper Analyzer example above
```

### Knowledge Base Assistant

Create an assistant that can answer questions about a knowledge base:

```python
import asyncio
import os
from saplings import Agent, AgentConfig
from saplings.memory import MemoryStore, DependencyGraph
from saplings.tools import PythonInterpreterTool

async def main():
    # Create memory components
    memory = MemoryStore()
    graph = DependencyGraph()

    # Process knowledge base files
    kb_dir = "./knowledge_base"
    for filename in os.listdir(kb_dir):
        if filename.endswith((".md", ".txt")):
            file_path = os.path.join(kb_dir, filename)
            with open(file_path, "r") as f:
                content = f.read()
                await memory.add_document(
                    content=content,
                    metadata={"source": filename}
                )

    # Build dependency graph
    await graph.build_from_memory(memory)

    # Create tools
    python_tool = PythonInterpreterTool()

    # Create an agent
    agent = Agent(
        config=AgentConfig(
            provider="openai",
            model_name="gpt-4o",
            enable_gasa=True,
            tools=[python_tool],
        )
    )

    # Set memory components
    agent.memory_store = memory
    agent.dependency_graph = graph

    # Interactive Q&A loop
    print("Knowledge Base Assistant (type 'exit' to quit)")
    while True:
        question = input("\nQuestion: ")
        if question.lower() == "exit":
            break

        result = await agent.run(question)
        print(f"\nAnswer: {result}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Code Assistant

Create an assistant that can help with coding tasks:

```python
import asyncio
import os
from saplings import Agent, AgentConfig
from saplings.memory import MemoryStore, DependencyGraph
from saplings.tools import PythonInterpreterTool

async def main():
    # Create memory components
    memory = MemoryStore()
    graph = DependencyGraph()

    # Process codebase files
    code_dir = "./codebase"
    for root, _, files in os.walk(code_dir):
        for filename in files:
            if filename.endswith((".py", ".js", ".java", ".cpp", ".h")):
                file_path = os.path.join(root, filename)
                with open(file_path, "r") as f:
                    content = f.read()
                    await memory.add_document(
                        content=content,
                        metadata={
                            "source": os.path.relpath(file_path, code_dir),
                            "type": "code",
                            "language": os.path.splitext(filename)[1][1:]
                        }
                    )

    # Build dependency graph
    await graph.build_from_memory(memory)

    # Create tools
    python_tool = PythonInterpreterTool()

    # Create an agent
    agent = Agent(
        config=AgentConfig(
            provider="openai",
            model_name="gpt-4o",
            enable_gasa=True,
            tools=[python_tool],
        )
    )

    # Set memory components
    agent.memory_store = memory
    agent.dependency_graph = graph

    # Interactive coding assistant loop
    print("Code Assistant (type 'exit' to quit)")
    while True:
        question = input("\nQuestion: ")
        if question.lower() == "exit":
            break

        result = await agent.run(question)
        print(f"\nAnswer: {result}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Conclusion

These examples demonstrate the versatility and power of Saplings for building intelligent agents. For more examples, see the [examples](../examples) directory in the repository.
