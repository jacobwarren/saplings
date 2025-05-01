"""
Self-healing module for Saplings.

This module provides self-healing capabilities for Saplings, including:
- PatchGenerator for auto-fixing errors
- Retry mechanism with capping
- Success pair collection for training
- LoRA fine-tuning pipeline
- Adapter management for model improvements
- Integration with JudgeAgent feedback
"""

from saplings.self_heal.adapter_manager import (
    Adapter,
    AdapterManager,
    AdapterMetadata,
    AdapterPriority,
)
from saplings.self_heal.lora_tuning import LoRaConfig, LoRaTrainer, TrainingMetrics
from saplings.self_heal.patch_generator import Patch, PatchGenerator, PatchResult, PatchStatus
from saplings.self_heal.success_pair_collector import SuccessPairCollector

__all__ = [
    "PatchGenerator",
    "PatchResult",
    "PatchStatus",
    "Patch",
    "SuccessPairCollector",
    "LoRaTrainer",
    "LoRaConfig",
    "TrainingMetrics",
    "AdapterManager",
    "AdapterMetadata",
    "AdapterPriority",
    "Adapter",
]
