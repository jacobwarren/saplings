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

from saplings.self_heal.patch_generator import PatchGenerator, PatchResult, PatchStatus, Patch
from saplings.self_heal.success_pair_collector import SuccessPairCollector
from saplings.self_heal.lora_tuning import LoRaTrainer, LoRaConfig, TrainingMetrics
from saplings.self_heal.adapter_manager import AdapterManager, AdapterMetadata, AdapterPriority, Adapter

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
