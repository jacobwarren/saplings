"""
Example of using the self-healing capabilities in Saplings.

This example demonstrates how to use the PatchGenerator, SuccessPairCollector,
and AdapterManager to automatically fix errors in code and improve over time.
"""

import os
from saplings import PatchGenerator, SuccessPairCollector, AdapterManager, AdapterMetadata, AdapterPriority

# Create a directory for success pairs
os.makedirs("success_pairs", exist_ok=True)

# Create a success pair collector
collector = SuccessPairCollector(storage_dir="success_pairs")

# Create a patch generator with the collector
patch_generator = PatchGenerator(
    max_retries=3,
    success_pair_collector=collector
)

# Example 1: Fix a syntax error
print("Example 1: Fix a syntax error")
code_with_syntax_error = """
def calculate_factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * calculate_factorial(n - 1
"""
error_message = "SyntaxError: unexpected EOF while parsing"

# Generate a patch
patch = patch_generator.generate_patch(code_with_syntax_error, error_message)
print(f"Original code:\n{patch.original_code}")
print(f"Error: {patch.error}")
print(f"Patched code:\n{patch.patched_code}")

# Apply the patch
result = patch_generator.apply_patch(patch)
if result.success:
    print("Patch applied successfully")
    
    # Validate the patch
    is_valid, error_msg = patch_generator.validate_patch(patch.patched_code)
    if is_valid:
        print("Patch is valid")
        
        # Collect the successful patch
        patch_generator.after_success(patch)
    else:
        print(f"Patch is invalid: {error_msg}")
else:
    print(f"Failed to apply patch: {result.error}")

# Example 2: Fix a name error
print("\nExample 2: Fix a name error")
code_with_name_error = """
def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total += num
    return result
"""
error_message = "NameError: name 'result' is not defined"

# Generate a patch
patch = patch_generator.generate_patch(code_with_name_error, error_message)
print(f"Original code:\n{patch.original_code}")
print(f"Error: {patch.error}")
print(f"Patched code:\n{patch.patched_code}")

# Apply the patch
result = patch_generator.apply_patch(patch)
if result.success:
    print("Patch applied successfully")
    
    # Validate the patch
    is_valid, error_msg = patch_generator.validate_patch(patch.patched_code)
    if is_valid:
        print("Patch is valid")
        
        # Collect the successful patch
        patch_generator.after_success(patch)
    else:
        print(f"Patch is invalid: {error_msg}")
else:
    print(f"Failed to apply patch: {result.error}")

# Get statistics about collected pairs
stats = collector.get_statistics()
print(f"\nTotal pairs: {stats['total_pairs']}")
print(f"Error types: {stats['error_types']}")
print(f"Error patterns: {stats['error_patterns']}")

# Export pairs for training
collector.export_to_jsonl("training_data.jsonl")
print("\nExported success pairs to training_data.jsonl")

# Optional: Fine-tune a model with the collected pairs
try:
    from saplings import LoRaTrainer, LoRaConfig
    
    print("\nFine-tuning a model with the collected pairs")
    
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
    
    # Create an adapter manager
    os.makedirs("adapters", exist_ok=True)
    adapter_manager = AdapterManager(
        model_name="gpt2",
        adapters_dir="adapters"
    )
    
    # Register the adapter
    metadata = AdapterMetadata(
        adapter_id="error_fix_adapter",
        model_name="gpt2",
        description="Adapter for fixing errors",
        version="1.0.0",
        created_at="2023-01-01T00:00:00",
        success_rate=0.85,
        priority=AdapterPriority.MEDIUM,
        error_types=["SyntaxError", "NameError"],
        tags=["python"]
    )
    adapter_manager.register_adapter("adapters/error_fix_adapter", metadata)
    
    print("\nRegistered adapter: error_fix_adapter")
    
    # Find adapters for a specific error type
    adapters = adapter_manager.find_adapters_for_error("SyntaxError")
    for adapter in adapters:
        print(f"Found adapter for SyntaxError: {adapter.metadata.adapter_id}")
except ImportError:
    print("\nSkipping fine-tuning (LoRA dependencies not installed)")
    print("Install with: pip install saplings[lora]")
