# Self-Healing System

The Self-Healing system in Saplings provides a powerful framework for automatically detecting, diagnosing, and fixing errors, enabling agents to improve over time through continuous learning.

## Overview

The Self-Healing system consists of several key components:

- **PatchGenerator**: Generates patches to fix code errors
- **SuccessPairCollector**: Collects successful error-fix pairs for training
- **LoRaTrainer**: Fine-tunes models using Low-Rank Adaptation (LoRA)
- **AdapterManager**: Manages LoRA adapters for model improvements
- **SelfHealingService**: Orchestrates self-healing operations

This system enables agents to automatically fix errors, learn from successful fixes, and continuously improve their performance through fine-tuning.

## Core Concepts

### Patch Generation

Patch generation is the process of automatically fixing errors in code:

- **Error Analysis**: Analyze error messages and code context to identify the root cause
- **Patch Creation**: Generate a patch to fix the error
- **Patch Validation**: Validate the patch by analyzing and/or executing the patched code
- **Patch Application**: Apply the patch to fix the error

The PatchGenerator uses a combination of static analysis, pattern matching, and model-based generation to create patches for various types of errors.

### Success Pair Collection

Success pair collection is the process of collecting successful error-fix pairs for training:

- **Pair Collection**: Collect input-output pairs where the input is an error and the output is a fix
- **Pair Storage**: Store the pairs in a structured format for later use
- **Pair Export**: Export the pairs for training models

The SuccessPairCollector manages the collection, storage, and export of success pairs, providing a foundation for continuous learning.

### LoRA Fine-Tuning

LoRA fine-tuning is the process of adapting models to specific tasks using Low-Rank Adaptation:

- **Data Preparation**: Prepare training data from success pairs
- **Model Configuration**: Configure the model and LoRA parameters
- **Training**: Train the model using LoRA
- **Adapter Management**: Save and manage the trained adapters

The LoRaTrainer provides a complete pipeline for fine-tuning models, enabling continuous improvement through learning from successful fixes.

### Scheduled Training

Scheduled training enables automatic, periodic fine-tuning of models:

- **Schedule Configuration**: Configure the training schedule using cron expressions
- **Job Execution**: Execute training jobs according to the schedule
- **Result Tracking**: Track training results and metrics
- **Graceful Shutdown**: Handle shutdown signals for clean termination

The LoRaTrainer includes scheduling capabilities using APScheduler, enabling nightly training and continuous improvement.

## API Reference

### PatchGenerator

```python
class PatchGenerator:
    def __init__(
        self,
        max_retries: int = 3,
        success_pair_collector: Optional[Any] = None,
    ):
        """Initialize the patch generator."""

    async def generate_patch(
        self,
        error_message: str,
        code_context: str,
    ) -> Dict[str, Any]:
        """Generate a patch for a failed execution."""

    def validate_patch(self, patched_code: str) -> tuple[bool, Optional[str]]:
        """Validate a patched code by analyzing and/or executing it."""

    def apply_patch(self, patch: Patch) -> bool:
        """Apply a patch to fix an error."""

    def after_success(self, patch: Patch) -> None:
        """Process a successful patch for collection and learning."""

    def reset(self) -> None:
        """Reset the patch generator state."""
```

### Patch

```python
class Patch:
    def __init__(
        self,
        original_code: str,
        patched_code: str,
        error: str,
        error_info: Dict[str, Any],
        status: PatchStatus = PatchStatus.GENERATED,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize a patch."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert the patch to a dictionary."""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Patch":
        """Create a patch from a dictionary."""
```

### SuccessPairCollector

```python
class SuccessPairCollector:
    def __init__(
        self,
        output_dir: str = "success_pairs",
        max_pairs: int = 1000,
    ):
        """Initialize the success pair collector."""

    async def collect(
        self,
        input_text: str,
        output_text: str,
        context: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Collect a successful pair."""

    async def collect_patch(
        self,
        patch: Patch,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Collect a patch as a success pair (legacy format)."""

    def get_all_pairs(self) -> List[Dict[str, Any]]:
        """Get all collected pairs."""

    def export_to_jsonl(self, output_path: str) -> None:
        """Export all pairs to a JSONL file."""
```

### LoRaTrainer

```python
class LoRaTrainer:
    def __init__(
        self,
        model_name: str,
        output_dir: str,
        config: Optional[LoRaConfig] = None,
        gasa_tune: bool = False,
    ):
        """Initialize the LoRA trainer."""

    def train(self, data_path: str) -> TrainingMetrics:
        """Train the model using LoRA."""

    def load_data(self, data_path: str) -> Dataset:
        """Load training data from a file."""

    def preprocess_data(self, dataset: Dataset) -> Dataset:
        """Preprocess the training data."""

    def save_model(self, model: Any) -> None:
        """Save the trained model."""

    def schedule_nightly_training(self, data_path: str, cron_expression: str = "0 0 * * *") -> None:
        """Schedule nightly training using APScheduler."""
```

### LoRaConfig

```python
class LoRaConfig:
    r: int = 8  # LoRA rank
    lora_alpha: int = 16  # LoRA alpha
    lora_dropout: float = 0.05  # LoRA dropout
    bias: str = "none"  # Bias type
    target_modules: List[str] = ["q_proj", "v_proj"]  # Target modules for LoRA
    learning_rate: float = 3e-4  # Learning rate
    batch_size: int = 4  # Batch size
    num_epochs: int = 3  # Number of training epochs
    warmup_steps: int = 100  # Number of warmup steps
    gradient_accumulation_steps: int = 1  # Number of gradient accumulation steps
    max_grad_norm: float = 1.0  # Maximum gradient norm
    weight_decay: float = 0.01  # Weight decay
    fp16: bool = True  # Whether to use mixed precision training
    bf16: bool = False  # Whether to use bfloat16 precision training
    optim: str = "adamw_torch"  # Optimizer type
    lr_scheduler_type: str = "cosine"  # Learning rate scheduler type
    max_seq_length: int = 2048  # Maximum sequence length

    @classmethod
    def default(cls) -> "LoRaConfig":
        """Create a default configuration."""

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "LoRaConfig":
        """Create a configuration from a dictionary."""
```

### AdapterManager

```python
class AdapterManager:
    def __init__(
        self,
        model: Optional[Any] = None,
        adapter_dir: Optional[str] = None,
    ):
        """Initialize the adapter manager."""

    def load_adapters(self) -> None:
        """Load adapters from the adapters directory."""

    def register_adapter(self, path: str, metadata: AdapterMetadata) -> None:
        """Register an adapter."""

    def get_adapter(self, adapter_id: str) -> Optional[Adapter]:
        """Get an adapter by ID."""

    def list_adapters(self) -> List[str]:
        """List all registered adapter IDs."""

    def activate_adapter(self, adapter_id: str) -> bool:
        """Activate an adapter."""

    def deactivate_adapter(self) -> bool:
        """Deactivate the active adapter."""

    def update_adapter_metadata(self, adapter_id: str, metadata: Dict[str, Any]) -> bool:
        """Update adapter metadata."""

    def find_adapters_for_error(self, error_type: str) -> List[Adapter]:
        """Find adapters for a specific error type."""
```

### SelfHealingService

The SelfHealingService provides comprehensive error handling and integration with the success pair collection system:

- **Patch Application**: Validates patches before applying them and provides detailed feedback
- **Success Pair Collection**: Automatically collects successful patches for future learning
- **Error Analysis**: Provides detailed error analysis to guide patch generation

```python
class SelfHealingService:
    def __init__(
        self,
        patch_generator: Optional[IPatchGenerator] = None,
        success_pair_collector: Optional[ISuccessPairCollector] = None,
        adapter_manager: Optional[IAdapterManager] = None,
        enabled: bool = True,
        trace_manager: Optional["TraceManager"] = None,
    ):
        """Initialize the self-healing service."""

    async def collect_success_pair(
        self,
        input_text: str,
        output_text: str,
        context: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Collect a success pair for future improvements."""

    async def get_all_success_pairs(
        self,
        trace_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get all collected success pairs."""

    async def generate_patch(
        self,
        error_message: str,
        code_context: str,
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a patch for a failed execution."""

    async def apply_patch(
        self,
        patch_id: str,
        trace_id: Optional[str] = None,
    ) -> bool:
        """Apply a patch."""

    async def train_adapter(
        self,
        pairs: List[Dict[str, Any]],
        adapter_name: str,
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Train an adapter using success pairs."""

    async def load_adapter(
        self,
        adapter_name: str,
        trace_id: Optional[str] = None,
    ) -> bool:
        """Load an adapter."""

    async def unload_adapter(
        self,
        trace_id: Optional[str] = None,
    ) -> bool:
        """Unload the current adapter."""
```

### Enums

```python
class PatchStatus(str, Enum):
    """Status of a patch."""
    GENERATED = "generated"  # Patch has been generated
    VALIDATED = "validated"  # Patch has been validated
    APPLIED = "applied"  # Patch has been applied
    FAILED = "failed"  # Patch failed validation or application
    REJECTED = "rejected"  # Patch was rejected by the user

class AdapterPriority(str, Enum):
    """Priority of an adapter."""
    LOW = "low"  # Low priority
    MEDIUM = "medium"  # Medium priority
    HIGH = "high"  # High priority
```

## Usage Examples

### Basic Patch Generation

```python
from saplings.self_heal import PatchGenerator, Patch, PatchStatus

# Create a patch generator
patch_generator = PatchGenerator(max_retries=3)

# Generate a patch for a failed execution
import asyncio
patch_result = asyncio.run(patch_generator.generate_patch(
    error_message="NameError: name 'data' is not defined",
    code_context="""
def process_data():
    result = data.process()
    return result
"""
))

# Check the patch result
if patch_result["success"]:
    patch = Patch.from_dict(patch_result["patch"])
    print(f"Original code:\n{patch.original_code}")
    print(f"Patched code:\n{patch.patched_code}")
    print(f"Error: {patch.error}")
    print(f"Status: {patch.status}")

    # Validate the patch
    is_valid, error = patch_generator.validate_patch(patch.patched_code)
    if is_valid:
        print("Patch is valid!")

        # Apply the patch
        success = patch_generator.apply_patch(patch)
        if success:
            print("Patch applied successfully!")

            # Process successful patch
            patch.status = PatchStatus.APPLIED
            patch_generator.after_success(patch)
        else:
            print(f"Failed to apply patch: {error}")
    else:
        print(f"Patch is invalid: {error}")
else:
    print(f"Failed to generate patch: {patch_result['error']}")
```

### Success Pair Collection

```python
from saplings.self_heal import SuccessPairCollector

# Create a success pair collector
collector = SuccessPairCollector(output_dir="./success_pairs")

# Collect a success pair
import asyncio
asyncio.run(collector.collect(
    input_text="NameError: name 'data' is not defined\n\ndef process_data():\n    result = data.process()\n    return result",
    output_text="def process_data():\n    data = get_data()\n    result = data.process()\n    return result",
    metadata={
        "error_type": "NameError",
        "fix_type": "variable_initialization",
    }
))

# Get all collected pairs
pairs = collector.get_all_pairs()
print(f"Collected {len(pairs)} pairs")

# Export pairs to a JSONL file
collector.export_to_jsonl("./success_pairs.jsonl")
```

### LoRA Fine-Tuning

```python
from saplings.self_heal import LoRaTrainer, LoRaConfig

# Create a LoRA configuration
config = LoRaConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    learning_rate=3e-4,
    batch_size=4,
    num_epochs=3,
)

# Create a LoRA trainer
trainer = LoRaTrainer(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    output_dir="./lora_adapters",
    config=config,
)

# Train the model
metrics = trainer.train("./success_pairs.jsonl")
print(f"Training metrics: {metrics}")

# Schedule nightly training
trainer.schedule_nightly_training(
    data_path="./success_pairs.jsonl",
    cron_expression="0 0 * * *",  # Midnight every day
)
```

### Adapter Management

```python
from saplings.self_heal import AdapterManager, AdapterMetadata, AdapterPriority

# Create an adapter manager
adapter_manager = AdapterManager(adapter_dir="./lora_adapters")

# Register an adapter
adapter_manager.register_adapter(
    path="./lora_adapters/error_fix_adapter",
    metadata=AdapterMetadata(
        adapter_id="error_fix_adapter",
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        description="Adapter for fixing common errors",
        version="1.0.0",
        created_at="2023-01-01T00:00:00Z",
        success_rate=0.85,
        priority=AdapterPriority.HIGH,
        error_types=["NameError", "TypeError", "AttributeError"],
        tags=["error_fix", "code_repair"],
    ),
)

# List all adapters
adapter_ids = adapter_manager.list_adapters()
print(f"Available adapters: {adapter_ids}")

# Activate an adapter
success = adapter_manager.activate_adapter("error_fix_adapter")
if success:
    print("Adapter activated successfully!")
else:
    print("Failed to activate adapter")

# Find adapters for a specific error type
adapters = adapter_manager.find_adapters_for_error("NameError")
print(f"Found {len(adapters)} adapters for NameError")
```

### Integration with Agent

```python
from saplings import Agent, AgentConfig
from saplings.self_heal import PatchGenerator, SuccessPairCollector

# Create a patch generator
patch_generator = PatchGenerator(max_retries=3)

# Create a success pair collector
collector = SuccessPairCollector(output_dir="./success_pairs")

# Create an agent with self-healing enabled
agent = Agent(
    config=AgentConfig(
        provider="openai",
        model_name="gpt-4o",
        enable_self_healing=True,
        self_healing_max_retries=3,
    )
)

# Set the patch generator and success pair collector
agent.patch_generator = patch_generator
agent.success_pair_collector = collector

# Run a task
import asyncio
result = asyncio.run(agent.run(
    "Write a Python function to calculate the factorial of a number."
))

# The agent will automatically fix any errors in the generated code
print(result)
```

## Advanced Features

### Intelligent Error Handling

The PatchGenerator includes intelligent error handling for common issues:

- **Variable Initialization**: Automatically initializes variables with appropriate default values based on variable names
- **Type Conversion**: Handles type conversion errors with appropriate fallback mechanisms
- **Function Arguments**: Fixes function argument errors by providing appropriate default values
- **None Type Handling**: Adds proper null checks and initialization for None values

```python
from saplings.self_heal import PatchGenerator

# Create a patch generator
patch_generator = PatchGenerator()

# Generate a patch for a type error
import asyncio
patch_result = asyncio.run(patch_generator.generate_patch(
    error_message="TypeError: can only concatenate str (not 'int') to str",
    code_context="""
def format_message(name, age):
    return "Name: " + name + ", Age: " + age
"""
))

print(patch_result["patch"]["patched_code"])
# Output:
# def format_message(name, age):
#     return "Name: " + name + ", Age: " + str(age)
```

### Error Pattern Recognition

The PatchGenerator includes pattern recognition for common error types:

```python
from saplings.self_heal import PatchGenerator

# Create a patch generator
patch_generator = PatchGenerator()

# Analyze an error
error_info = patch_generator.analyze_error(
    code_context="""
def process_data():
    result = data.process()
    return result
""",
    error_message="NameError: name 'data' is not defined",
)

# Check the recognized patterns
print(f"Error type: {error_info['type']}")
print(f"Patterns: {error_info['patterns']}")
print(f"Variable: {error_info.get('variable')}")
```

The PatchGenerator recognizes various error patterns, including:

- **undefined_variable**: Variable is used before it's defined
- **attribute_error**: Attribute access on an object that doesn't have it
- **type_error**: Operation on incompatible types
- **import_error**: Module import fails
- **syntax_error**: Syntax errors in the code
- **index_error**: Index out of range
- **key_error**: Key not found in dictionary
- **value_error**: Invalid value for operation
- **zero_division_error**: Division by zero
- **file_not_found_error**: File not found
- **permission_error**: Permission denied
- **timeout_error**: Operation timed out
- **memory_error**: Out of memory
- **recursion_error**: Maximum recursion depth exceeded
- **assertion_error**: Assertion failed
- **not_implemented_error**: Method not implemented
- **runtime_error**: Generic runtime error
- **exception**: Generic exception

### Static Analysis

The PatchGenerator uses static analysis to identify issues in code:

```python
from saplings.self_heal import PatchGenerator

# Create a patch generator
patch_generator = PatchGenerator()

# Perform static analysis
analysis_result = patch_generator.static_analyze(
    code="""
def process_data():
    result = data.process()
    return result
"""
)

# Check the analysis result
print(f"Issues found: {len(analysis_result['issues'])}")
for issue in analysis_result["issues"]:
    print(f"Line {issue['line']}: {issue['message']} ({issue['symbol']})")
```

Static analysis helps identify issues like:

- Undefined variables
- Unused variables
- Missing imports
- Syntax errors
- Style issues
- Type mismatches
- Logic errors

### Scheduled Training

The LoRaTrainer supports scheduled training using APScheduler:

```python
from saplings.self_heal import LoRaTrainer, LoRaConfig

# Create a LoRA trainer
trainer = LoRaTrainer(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    output_dir="./lora_adapters",
)

# Schedule training with a custom cron expression
trainer.schedule_nightly_training(
    data_path="./success_pairs.jsonl",
    cron_expression="0 3 * * 1-5",  # 3 AM on weekdays
)
```

The scheduler supports various cron expressions:

- **Daily**: `0 0 * * *` (midnight every day)
- **Weekdays**: `0 0 * * 1-5` (midnight on weekdays)
- **Weekly**: `0 0 * * 0` (midnight on Sunday)
- **Monthly**: `0 0 1 * *` (midnight on the 1st of each month)
- **Custom**: Any valid cron expression

### GASA-Specific Tuning

The LoRaTrainer supports GASA-specific tuning:

```python
from saplings.self_heal import LoRaTrainer, LoRaConfig

# Create a LoRA configuration for GASA
config = LoRaConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Include attention modules
)

# Create a LoRA trainer with GASA tuning
trainer = LoRaTrainer(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    output_dir="./gasa_adapters",
    config=config,
    gasa_tune=True,  # Enable GASA-specific tuning
)

# Train the model
trainer.train("./success_pairs.jsonl")
```

GASA-specific tuning includes attention modules in the target modules, enabling the model to better utilize the Graph-Aligned Sparse Attention mechanism.

## Implementation Details

### Patch Generation Process

The patch generation process works as follows:

1. **Error Analysis**: Analyze the error message and code context to identify the root cause
2. **Pattern Recognition**: Recognize common error patterns
3. **Static Analysis**: Perform static analysis to identify issues
4. **Patch Generation**: Generate a patch based on the analysis
5. **Patch Validation**: Validate the patch by analyzing and/or executing the patched code
6. **Patch Application**: Apply the patch to fix the error

### Success Pair Collection Process

The success pair collection process works as follows:

1. **Pair Creation**: Create a pair from input and output
2. **Metadata Addition**: Add metadata to the pair
3. **Pair Storage**: Store the pair in the output directory
4. **Pair Export**: Export pairs for training

### LoRA Fine-Tuning Process

The LoRA fine-tuning process works as follows:

1. **Data Loading**: Load training data from a file
2. **Data Preprocessing**: Preprocess the data for training
3. **Model Loading**: Load the base model
4. **LoRA Configuration**: Configure LoRA parameters
5. **Training**: Train the model using LoRA
6. **Evaluation**: Evaluate the trained model
7. **Model Saving**: Save the trained model and adapter

### Scheduled Training Process

The scheduled training process works as follows:

1. **Scheduler Creation**: Create a background scheduler
2. **Job Definition**: Define the training job
3. **Schedule Configuration**: Configure the schedule using a cron expression
4. **Scheduler Start**: Start the scheduler
5. **Signal Handling**: Register signal handlers for clean shutdown

### Adapter Management Process

The adapter management process works as follows:

1. **Adapter Loading**: Load adapters from the adapters directory
2. **Adapter Registration**: Register new adapters
3. **Adapter Activation**: Activate an adapter for use
4. **Adapter Deactivation**: Deactivate the active adapter
5. **Adapter Selection**: Select adapters based on error type

## Extension Points

The Self-Healing system is designed to be extensible:

### Custom Patch Generator

You can create a custom patch generator by implementing the `IPatchGenerator` interface:

```python
from saplings.self_heal.interfaces import IPatchGenerator
from typing import Dict, Any, Optional, Tuple

class CustomPatchGenerator(IPatchGenerator):
    async def generate_patch(
        self,
        error_message: str,
        code_context: str,
    ) -> Dict[str, Any]:
        # Custom patch generation logic
        # ...
        return {
            "success": True,
            "patch": {
                "original_code": code_context,
                "patched_code": patched_code,
                "error": error_message,
                "error_info": error_info,
                "status": "generated",
            }
        }

    def validate_patch(self, patched_code: str) -> Tuple[bool, Optional[str]]:
        # Custom patch validation logic
        # ...
        return True, None
```

### Custom Success Pair Collector

You can create a custom success pair collector by implementing the `ISuccessPairCollector` interface:

```python
from saplings.self_heal.interfaces import ISuccessPairCollector
from typing import Dict, Any, List, Optional

class CustomSuccessPairCollector(ISuccessPairCollector):
    async def collect(
        self,
        input_text: str,
        output_text: str,
        context: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        # Custom collection logic
        # ...
        pass

    def get_all_pairs(self) -> List[Dict[str, Any]]:
        # Custom retrieval logic
        # ...
        return pairs

    def export_to_jsonl(self, output_path: str) -> None:
        # Custom export logic
        # ...
        pass
```

### Custom Adapter Manager

You can create a custom adapter manager by implementing the `IAdapterManager` interface:

```python
from saplings.self_heal.interfaces import IAdapterManager
from typing import Dict, Any, List, Optional

class CustomAdapterManager(IAdapterManager):
    def load_adapters(self) -> None:
        # Custom adapter loading logic
        # ...
        pass

    def register_adapter(self, path: str, metadata: Dict[str, Any]) -> None:
        # Custom adapter registration logic
        # ...
        pass

    def activate_adapter(self, adapter_id: str) -> bool:
        # Custom adapter activation logic
        # ...
        return True

    def deactivate_adapter(self) -> bool:
        # Custom adapter deactivation logic
        # ...
        return True
```

## Conclusion

The Self-Healing system in Saplings provides a powerful framework for automatically detecting, diagnosing, and fixing errors, enabling agents to improve over time through continuous learning. By combining patch generation, success pair collection, and LoRA fine-tuning, it enables agents to automatically fix errors, learn from successful fixes, and continuously improve their performance.
