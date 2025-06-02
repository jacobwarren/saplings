from __future__ import annotations

"""
Public API for Self-Healing capabilities.

This module provides the public API for self-healing capabilities in Saplings,
including patch generation, adapter management, and success pair collection.
"""

from saplings.api.stability import beta, stable

# Import from internal modules
from saplings.self_heal._internal.adapters import (
    Adapter as _Adapter,
)
from saplings.self_heal._internal.adapters import (
    AdapterManager as _AdapterManager,
)
from saplings.self_heal._internal.adapters import (
    AdapterMetadata as _AdapterMetadata,
)
from saplings.self_heal._internal.adapters import (
    AdapterPriority as _AdapterPriority,
)
from saplings.self_heal._internal.collectors import (
    SuccessPairCollector as _SuccessPairCollector,
)
from saplings.self_heal._internal.config import (
    RetryStrategy as _RetryStrategy,
)
from saplings.self_heal._internal.config import (
    SelfHealingConfig as _SelfHealingConfig,
)
from saplings.self_heal._internal.interfaces import (
    IAdapterManager as _IAdapterManager,
)
from saplings.self_heal._internal.interfaces import (
    IPatchGenerator as _IPatchGenerator,
)
from saplings.self_heal._internal.interfaces import (
    ISuccessPairCollector as _ISuccessPairCollector,
)
from saplings.self_heal._internal.patches import (
    Patch as _Patch,
)
from saplings.self_heal._internal.patches import (
    PatchGenerator as _PatchGenerator,
)
from saplings.self_heal._internal.patches import (
    PatchResult as _PatchResult,
)
from saplings.self_heal._internal.patches import (
    PatchStatus as _PatchStatus,
)
from saplings.self_heal._internal.tuning import (
    LoRaConfig as _LoRaConfig,
)
from saplings.self_heal._internal.tuning import (
    LoRaTrainer as _LoRaTrainer,
)
from saplings.self_heal._internal.tuning import (
    TrainingMetrics as _TrainingMetrics,
)

# Re-export with stability annotations
# Use direct assignment for enums to avoid extending them
RetryStrategy = _RetryStrategy
# Add stability annotation
stable(RetryStrategy)


# Use direct assignment for enums to avoid extending them
AdapterPriority = _AdapterPriority
# Add stability annotation
stable(AdapterPriority)


@stable
class SelfHealingConfig(_SelfHealingConfig):
    """
    Configuration for self-healing capabilities.

    This class provides configuration options for self-healing features including:
    - Retry strategies and limits
    - Success pair collection
    - LoRA fine-tuning parameters
    - Adapter management
    - Patch generation
    """


@stable
class Adapter(_Adapter):
    """
    A LoRA adapter for model fine-tuning.

    Adapters are lightweight fine-tuning layers that can be applied to a model
    to improve its performance on specific tasks or error types.
    """


@stable
class AdapterMetadata(_AdapterMetadata):
    """
    Metadata for a LoRA adapter.

    This class stores information about an adapter, including its success rate,
    priority, and the types of errors it can handle.
    """


@stable
class AdapterManager(_AdapterManager):
    """
    Manager for LoRA adapters.

    This class provides functionality for managing LoRA adapters, including:
    - Loading and unloading adapters
    - Registering new adapters
    - Finding adapters for specific error types
    - Processing feedback to update adapter success rates
    """


@beta
class LoRaConfig(_LoRaConfig):
    """
    Configuration for LoRA fine-tuning.

    This class provides configuration options for LoRA (Low-Rank Adaptation)
    fine-tuning, a parameter-efficient fine-tuning technique.
    """


@beta
class LoRaTrainer(_LoRaTrainer):
    """
    Trainer for LoRA fine-tuning.

    This class provides functionality for fine-tuning models using LoRA,
    a parameter-efficient fine-tuning technique.
    """


@beta
class TrainingMetrics(_TrainingMetrics):
    """
    Metrics for LoRA training.

    This class stores metrics from LoRA training, including loss values
    and evaluation results.
    """


# Use direct assignment for enums to avoid extending them
PatchStatus = _PatchStatus
# Add stability annotation
stable(PatchStatus)


@stable
class Patch(_Patch):
    """
    Representation of a code patch.

    A patch is a proposed fix for a code error, with metadata about its
    status, confidence, and other attributes.
    """


@stable
class PatchResult(_PatchResult):
    """
    Result of a patch generation operation.

    This class stores the result of a patch generation operation, including
    the generated patch and metadata about the generation process.
    """


@stable
class PatchGenerator(_PatchGenerator):
    """
    Generator for code patches.

    This class provides functionality for generating patches to fix code errors,
    including:
    - Analyzing error messages
    - Generating fixes for common error types
    - Validating generated patches
    """


@beta
class SuccessPairCollector(_SuccessPairCollector):
    """
    Collector for success pairs.

    This class provides functionality for collecting and managing success pairs,
    which are pairs of input and output texts that can be used for training
    adapters or other self-improvement mechanisms.
    """


@stable
class IAdapterManager(_IAdapterManager):
    """
    Interface for managing model adapters for self-healing.

    This interface defines the contract for adapter management components,
    allowing for different implementations while maintaining a consistent API.
    """


@stable
class IPatchGenerator(_IPatchGenerator):
    """
    Interface for generating patches to fix code errors.

    This interface defines the contract for patch generator components,
    allowing for different implementations while maintaining a consistent API.
    """


@stable
class ISuccessPairCollector(_ISuccessPairCollector):
    """
    Interface for collecting and managing success pairs for future learning.

    This interface defines the contract for success pair collector components,
    allowing for different implementations while maintaining a consistent API.
    """


__all__ = [
    # Retry strategy
    "RetryStrategy",
    # Adapter management
    "Adapter",
    "AdapterManager",
    "AdapterMetadata",
    "AdapterPriority",
    "IAdapterManager",
    # Configuration
    "SelfHealingConfig",
    # LoRA fine-tuning
    "LoRaConfig",
    "LoRaTrainer",
    "TrainingMetrics",
    # Patch generation
    "Patch",
    "PatchGenerator",
    "PatchResult",
    "PatchStatus",
    "IPatchGenerator",
    # Success pair collection
    "SuccessPairCollector",
    "ISuccessPairCollector",
]
