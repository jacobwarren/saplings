from __future__ import annotations

"""
Internal implementation of the Self-Healing module.
"""

from saplings.self_heal._internal.adapters import (
    Adapter,
    AdapterManager,
    AdapterMetadata,
    AdapterPriority,
)
from saplings.self_heal._internal.collectors import (
    SuccessPairCollector,
)
from saplings.self_heal._internal.config import (
    RetryStrategy,
    SelfHealingConfig,
)
from saplings.self_heal._internal.interfaces import (
    IPatchGenerator,
    ISuccessPairCollector,
)
from saplings.self_heal._internal.patches import (
    Patch,
    PatchGenerator,
    PatchResult,
    PatchStatus,
)
from saplings.self_heal._internal.tuning import (
    LoRaConfig,
    LoRaTrainer,
    TrainingMetrics,
)

__all__ = [
    # Adapter management
    "Adapter",
    "AdapterManager",
    "AdapterMetadata",
    "AdapterPriority",
    # Configuration
    "RetryStrategy",
    "SelfHealingConfig",
    # Interfaces
    "IPatchGenerator",
    "ISuccessPairCollector",
    # LoRA tuning
    "LoRaConfig",
    "LoRaTrainer",
    "TrainingMetrics",
    # Patch generation
    "Patch",
    "PatchGenerator",
    "PatchResult",
    "PatchStatus",
    # Success pair collection
    "SuccessPairCollector",
]
