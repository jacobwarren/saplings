# Examples

This document provides examples of using the Saplings framework for various tasks.

## Basic Usage

```python
from saplings import Agent, AgentConfig
from saplings.memory import MemoryStore
import asyncio

# Initialize a memory store with your repository
memory = MemoryStore()
memory.index_repository("/path/to/your/repo")

# Create an agent configuration
config = AgentConfig(
    model_uri="openai://gpt-4",
    memory_path="./agent_memory",
)

# Create an agent
agent = Agent(config=config)

# Set the memory store
agent.memory_store = memory

# Run a task (note: this is an async method)
async def main():
    result = await agent.run("Explain the architecture of this codebase")
    print(result["final_result"])

# Run the async function
asyncio.run(main())
```

## Using Self-Healing Capabilities

```python
from saplings import Agent, AgentConfig
from saplings.self_heal import PatchGenerator, SuccessPairCollector
import asyncio

# Create a success pair collector
collector = SuccessPairCollector(storage_dir="success_pairs")

# Create a patch generator with the collector
patch_generator = PatchGenerator(
    max_retries=3,
    success_pair_collector=collector
)

# Create an agent configuration with self-healing enabled
config = AgentConfig(
    model_uri="openai://gpt-4",
    enable_self_healing=True,
)

# Create an agent
agent = Agent(config=config)

# Set the patch generator
agent.patch_generator = patch_generator

# Run a task that might generate code with errors
async def main():
    result = await agent.run("Write a function to calculate the factorial of a number")

    # Check for errors in the result
    if "error" in result:
        print(f"Error: {result['error']}")
        if "patched_code" in result:
            print(f"Patched code:\n{result['patched_code']}")

    # Export collected success pairs for training
    await collector.export_to_jsonl("training_data.jsonl")

# Run the async function
asyncio.run(main())
```

## Fine-Tuning with LoRA

```python
from saplings.self_heal import LoRaTrainer, LoRaConfig
import asyncio

# Create a LoRA configuration
config = LoRaConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    learning_rate=5e-5,
    num_train_epochs=3
)

# Create a LoRA trainer
trainer = LoRaTrainer(
    model_name="gpt2",
    output_dir="lora_output",
    config=config
)

# Train the model
async def main():
    metrics = await trainer.train("training_data.jsonl")
    print(f"Training loss: {metrics.train_loss}")
    print(f"Evaluation loss: {metrics.eval_loss}")

    # Save the model
    model = await trainer.load_model()
    await trainer.save_model(model)

# Run the async function
asyncio.run(main())
```

## Using Adapters

```python
from saplings import Agent, AgentConfig
from saplings.self_heal import AdapterManager
import asyncio

# Create an adapter manager
adapter_manager = AdapterManager(
    model_name="gpt2",
    adapters_dir="adapters"
)

# Create an agent configuration with self-healing enabled
config = AgentConfig(
    model_uri="openai://gpt-4",
    enable_self_healing=True,
)

# Create an agent
agent = Agent(config=config)

# Set the adapter manager
agent.adapter_manager = adapter_manager

# Run a task
async def main():
    result = await agent.run("Write a function to calculate the factorial of a number")
    print(result["final_result"])

    # The adapter manager will automatically use the appropriate adapter for any errors

# Run the async function
asyncio.run(main())
```

## Advanced Usage

For more advanced usage examples, see the [examples](../examples) directory in the repository.
