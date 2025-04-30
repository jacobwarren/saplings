# Self-Healing and Adaptation Capabilities

Saplings includes powerful self-healing and adaptation capabilities that allow your agents to learn from errors, automatically fix issues, and improve over time through fine-tuning.

## Overview

The self-healing system consists of several components:

1. **PatchGenerator**: Analyzes errors in code and generates patches to fix them
2. **SuccessPairCollector**: Collects successful error-fix pairs for training
3. **LoRA Fine-Tuning**: Fine-tunes models using Low-Rank Adaptation (LoRA)
4. **AdapterManager**: Manages LoRA adapters for different error types

These components work together to create a continuous improvement loop:

```
Error → PatchGenerator → Fix → SuccessPairCollector → LoRA Fine-Tuning → Improved Models
```

## PatchGenerator

The `PatchGenerator` class analyzes errors in code and generates patches to fix them.

```python
from saplings import PatchGenerator

# Create a patch generator
patch_generator = PatchGenerator(max_retries=3)

# Generate a patch for an error
code = "def foo():\n    print(bar)\n"
error = "NameError: name 'bar' is not defined"
patch = patch_generator.generate_patch(code, error)

# Apply the patch
result = patch_generator.apply_patch(patch)
if result.success:
    patched_code = result.patched_code
    print(f"Patched code:\n{patched_code}")
else:
    print(f"Failed to apply patch: {result.error}")

# Validate the patch
is_valid, error_msg = patch_generator.validate_patch(patch.patched_code)
if is_valid:
    print("Patch is valid")
else:
    print(f"Patch is invalid: {error_msg}")
```

The `PatchGenerator` supports various error types:
- SyntaxError
- NameError
- TypeError
- ImportError
- IndentationError
- And more...

## SuccessPairCollector

The `SuccessPairCollector` class collects successful error-fix pairs for training.

```python
from saplings import SuccessPairCollector, PatchGenerator

# Create a success pair collector
collector = SuccessPairCollector(storage_dir="success_pairs")

# Create a patch generator with the collector
patch_generator = PatchGenerator(
    max_retries=3,
    success_pair_collector=collector
)

# When a patch is successful, it will be automatically collected
# You can also manually collect a patch
patch_generator.after_success(patch)

# Get statistics about collected pairs
stats = collector.get_statistics()
print(f"Total pairs: {stats['total_pairs']}")
print(f"Error types: {stats['error_types']}")

# Export pairs for training
collector.export_to_jsonl("training_data.jsonl")
```

## LoRA Fine-Tuning (Optional)

The LoRA fine-tuning pipeline allows you to fine-tune models using collected error-fix pairs. This feature requires additional dependencies:

```bash
# Install the required dependencies
pip install saplings[lora]
```

```python
from saplings import LoRaTrainer, LoRaConfig

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
metrics = trainer.train("training_data.jsonl")
print(f"Training loss: {metrics.train_loss}")
print(f"Evaluation loss: {metrics.eval_loss}")

# Save the model
model = trainer.load_model()
trainer.save_model(model)
```

## AdapterManager

The `AdapterManager` class manages LoRA adapters for different error types.

```python
from saplings import AdapterManager, AdapterMetadata, AdapterPriority

# Create an adapter manager
adapter_manager = AdapterManager(
    model_name="gpt2",
    adapters_dir="adapters"
)

# Register an adapter
metadata = AdapterMetadata(
    adapter_id="syntax_error_adapter",
    model_name="gpt2",
    description="Adapter for fixing syntax errors",
    version="1.0.0",
    created_at="2023-01-01T00:00:00",
    success_rate=0.85,
    priority=AdapterPriority.MEDIUM,
    error_types=["SyntaxError"],
    tags=["python"]
)
adapter_manager.register_adapter("adapters/syntax_error_adapter", metadata)

# Find adapters for a specific error type
adapters = adapter_manager.find_adapters_for_error("SyntaxError")
for adapter in adapters:
    print(f"Found adapter: {adapter.metadata.adapter_id}")

# Activate an adapter
adapter_manager.activate_adapter("syntax_error_adapter")

# Process feedback from JudgeAgent
adapter_manager.process_judge_feedback(0.9, "Good patch")

# Deactivate the adapter
adapter_manager.deactivate_adapter()

# Prune underperforming adapters
adapter_manager.prune_adapters(min_success_rate=0.5)
```

## Integration with Executor

The self-healing capabilities can be integrated with the `Executor` class to automatically fix errors in code:

```python
from saplings import Executor, ExecutorConfig, PatchGenerator

# Create a patch generator
patch_generator = PatchGenerator(max_retries=3)

# Create an executor with the patch generator
executor = Executor(
    model=model,
    config=ExecutorConfig(),
    patch_generator=patch_generator
)

# Execute a task
result = await executor.execute("Write a function to calculate the factorial of a number")

# If there's an error in the generated code, the patch generator will try to fix it
if result.error:
    print(f"Error: {result.error}")
    if result.patched_code:
        print(f"Patched code:\n{result.patched_code}")
```

## Continuous Improvement

The self-healing system is designed to improve over time:

1. As more errors are encountered and fixed, the `SuccessPairCollector` builds a larger dataset of error-fix pairs
2. This dataset can be used to fine-tune models using the `LoRaTrainer`
3. The fine-tuned models are managed by the `AdapterManager` and can be used to generate better patches
4. The system becomes more effective at fixing errors over time

You can schedule nightly training to continuously improve your models:

```python
from saplings import LoRaTrainer

# Create a LoRA trainer
trainer = LoRaTrainer(
    model_name="gpt2",
    output_dir="lora_output"
)

# Schedule nightly training
trainer.schedule_nightly_training(
    data_path="success_pairs.jsonl",
    cron_expression="0 0 * * *"  # Run at midnight every day
)
```

## Conclusion

The self-healing and adaptation capabilities in Saplings provide a powerful way to make your agents more robust and effective over time. By automatically fixing errors and learning from them, your agents can continuously improve without manual intervention.
