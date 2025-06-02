# Saplings Self-Healing

This package provides self-healing capabilities for Saplings agents.

## API Structure

The self-healing module follows the Saplings API separation pattern:

1. **Public API**: Exposed through `saplings.api.self_heal` and re-exported at the top level of the `saplings` package
2. **Internal Implementation**: Located in the `_internal` directory

## Usage

To use the self-healing components, import them from the public API:

```python
# Recommended: Import from the top-level package
from saplings import (
    Adapter,
    AdapterManager,
    PatchGenerator,
    SelfHealingConfig
)

# Alternative: Import directly from the API module
from saplings.api.self_heal import (
    Adapter,
    AdapterManager,
    PatchGenerator,
    SelfHealingConfig
)
```

Do not import directly from the internal implementation:

```python
# Don't do this
from saplings.self_heal._internal import PatchGenerator  # Wrong
```

## Available Components

The following components are available in the public API:

### Adapter Management

- `Adapter`: A LoRA adapter for model fine-tuning
- `AdapterManager`: Manager for LoRA adapters
- `AdapterMetadata`: Metadata for a LoRA adapter
- `AdapterPriority`: Priority levels for adapters

### Configuration

- `RetryStrategy`: Strategy for retrying failed operations
- `SelfHealingConfig`: Configuration for self-healing capabilities

### LoRA Fine-Tuning

- `LoRaConfig`: Configuration for LoRA fine-tuning
- `LoRaTrainer`: Trainer for LoRA fine-tuning
- `TrainingMetrics`: Metrics for LoRA training

### Patch Generation

- `Patch`: Representation of a code patch
- `PatchGenerator`: Generator for code patches
- `PatchResult`: Result of a patch generation operation
- `PatchStatus`: Status of a patch

### Success Pair Collection

- `SuccessPairCollector`: Collector for success pairs

## Example Usage

```python
from saplings import (
    AdapterManager,
    PatchGenerator,
    SelfHealingConfig
)

# Create a patch generator
patch_generator = PatchGenerator()

# Generate a patch for a failed execution
patch_result = await patch_generator.generate_patch(
    error_message="NameError: name 'pandas' is not defined",
    code_context="import pandas as pd\ndf = pd.DataFrame()"
)

# Create an adapter manager
adapter_manager = AdapterManager(
    model_name="gpt-3.5-turbo",
    adapter_dir="./adapters"
)

# Configure self-healing
config = SelfHealingConfig(
    enabled=True,
    max_retries=3,
    retry_strategy="exponential_backoff",
    enable_patch_generation=True
)
```

## Integration with Other Components

The self-healing module integrates with other Saplings components:

- **Agent**: The agent can use the self-healing capabilities to recover from errors
- **Judge**: The judge can provide feedback to improve adapters
- **Services**: The `SelfHealingService` provides a high-level interface to self-healing capabilities
