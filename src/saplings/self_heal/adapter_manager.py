"""
Adapter manager module for Saplings.

This module provides the AdapterManager class for managing LoRA adapters.
"""

import json
import logging
import os
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from saplings.self_heal.lora_tuning import LoRaTrainer

logger = logging.getLogger(__name__)


class AdapterPriority(str, Enum):
    """Priority of an adapter."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class AdapterMetadata:
    """Metadata for a LoRA adapter."""
    
    adapter_id: str
    model_name: str
    description: str
    version: str
    created_at: str
    success_rate: float
    priority: AdapterPriority
    error_types: List[str]
    tags: List[str]
    
    def __post_init__(self):
        """Initialize default values and convert types."""
        if isinstance(self.priority, str):
            self.priority = AdapterPriority(self.priority)
        
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


class Adapter:
    """A LoRA adapter."""
    
    def __init__(
        self,
        path: str,
        metadata: AdapterMetadata,
    ):
        """
        Initialize the adapter.
        
        Args:
            path: Path to the adapter
            metadata: Metadata for the adapter
        """
        self.path = path
        self.metadata = metadata
        self.model = None
    
    def load(self, model_name: str) -> Any:
        """
        Load the adapter.
        
        Args:
            model_name: Name of the base model
            
        Returns:
            Any: Loaded model with the adapter
        """
        try:
            # Create a LoRaTrainer to load the model
            trainer = LoRaTrainer(
                model_name=model_name,
                output_dir=self.path,
            )
            
            # Load the model
            self.model = trainer.load_model()
            
            logger.info(f"Loaded adapter {self.metadata.adapter_id} from {self.path}")
            
            return self.model
        except Exception as e:
            logger.error(f"Error loading adapter {self.metadata.adapter_id}: {e}")
            return None
    
    def unload(self) -> None:
        """Unload the adapter."""
        self.model = None
        
        logger.info(f"Unloaded adapter {self.metadata.adapter_id}")


class AdapterManager:
    """
    Manager for LoRA adapters.
    
    This class provides functionality for managing LoRA adapters.
    """
    
    def __init__(
        self,
        model_name: str,
        adapters_dir: str,
    ):
        """
        Initialize the adapter manager.
        
        Args:
            model_name: Name of the base model
            adapters_dir: Directory to store adapters
        """
        self.model_name = model_name
        self.adapters_dir = adapters_dir
        self.adapters: Dict[str, Adapter] = {}
        self.active_adapter: Optional[str] = None
        
        # Create the adapters directory if it doesn't exist
        os.makedirs(adapters_dir, exist_ok=True)
        
        # Load existing adapters
        self.load_adapters()
    
    def load_adapters(self) -> None:
        """Load adapters from the adapters directory."""
        # Get all subdirectories in the adapters directory
        for adapter_id in os.listdir(self.adapters_dir):
            adapter_path = os.path.join(self.adapters_dir, adapter_id)
            
            # Skip if not a directory
            if not os.path.isdir(adapter_path):
                continue
            
            # Check if metadata file exists
            metadata_path = os.path.join(adapter_path, "metadata.json")
            if not os.path.exists(metadata_path):
                logger.warning(f"No metadata found for adapter {adapter_id}, skipping")
                continue
            
            # Load metadata
            try:
                with open(metadata_path, "r") as f:
                    metadata_dict = json.load(f)
                
                # Create metadata object
                metadata = AdapterMetadata(**metadata_dict)
                
                # Create adapter
                adapter = Adapter(
                    path=adapter_path,
                    metadata=metadata,
                )
                
                # Add to adapters dictionary
                self.adapters[adapter_id] = adapter
                
                logger.info(f"Loaded adapter {adapter_id} from {adapter_path}")
            except Exception as e:
                logger.error(f"Error loading adapter {adapter_id}: {e}")
    
    def register_adapter(self, path: str, metadata: AdapterMetadata) -> None:
        """
        Register an adapter.
        
        Args:
            path: Path to the adapter
            metadata: Metadata for the adapter
        """
        # Create adapter
        adapter = Adapter(
            path=path,
            metadata=metadata,
        )
        
        # Add to adapters dictionary
        self.adapters[metadata.adapter_id] = adapter
        
        # Save metadata
        metadata_path = os.path.join(path, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(asdict(metadata), f, indent=2)
        
        logger.info(f"Registered adapter {metadata.adapter_id} at {path}")
    
    def get_adapter(self, adapter_id: str) -> Optional[Adapter]:
        """
        Get an adapter by ID.
        
        Args:
            adapter_id: ID of the adapter
            
        Returns:
            Optional[Adapter]: The adapter, or None if not found
        """
        return self.adapters.get(adapter_id)
    
    def activate_adapter(self, adapter_id: str) -> bool:
        """
        Activate an adapter.
        
        Args:
            adapter_id: ID of the adapter to activate
            
        Returns:
            bool: Whether the adapter was activated
        """
        # Check if adapter exists
        adapter = self.get_adapter(adapter_id)
        if not adapter:
            logger.error(f"Adapter {adapter_id} not found")
            return False
        
        # Deactivate current adapter if any
        if self.active_adapter:
            self.deactivate_adapter()
        
        # Load the adapter
        model = adapter.load(self.model_name)
        if not model:
            logger.error(f"Failed to load adapter {adapter_id}")
            return False
        
        # Set as active adapter
        self.active_adapter = adapter_id
        
        logger.info(f"Activated adapter {adapter_id}")
        
        return True
    
    def deactivate_adapter(self) -> None:
        """Deactivate the current adapter."""
        if not self.active_adapter:
            return
        
        # Get the active adapter
        adapter = self.get_adapter(self.active_adapter)
        if adapter:
            # Unload the adapter
            adapter.unload()
        
        # Clear active adapter
        self.active_adapter = None
        
        logger.info("Deactivated adapter")
    
    def update_adapter_metadata(self, adapter_id: str, metadata_dict: Dict[str, Any]) -> bool:
        """
        Update adapter metadata.
        
        Args:
            adapter_id: ID of the adapter
            metadata_dict: Dictionary with metadata fields to update
            
        Returns:
            bool: Whether the metadata was updated
        """
        # Check if adapter exists
        adapter = self.get_adapter(adapter_id)
        if not adapter:
            logger.error(f"Adapter {adapter_id} not found")
            return False
        
        # Update metadata
        for key, value in metadata_dict.items():
            if hasattr(adapter.metadata, key):
                setattr(adapter.metadata, key, value)
        
        # Save metadata
        metadata_path = os.path.join(adapter.path, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(asdict(adapter.metadata), f, indent=2)
        
        logger.info(f"Updated metadata for adapter {adapter_id}")
        
        return True
    
    def find_adapters_for_error(self, error_type: str) -> List[Adapter]:
        """
        Find adapters for a specific error type.
        
        Args:
            error_type: Type of error
            
        Returns:
            List[Adapter]: List of adapters that can handle the error
        """
        matching_adapters = []
        
        for adapter in self.adapters.values():
            if error_type in adapter.metadata.error_types:
                matching_adapters.append(adapter)
        
        # Sort by priority and success rate
        matching_adapters.sort(
            key=lambda a: (
                # Sort by priority (HIGH > MEDIUM > LOW)
                {"high": 2, "medium": 1, "low": 0}[a.metadata.priority.value],
                # Then by success rate
                a.metadata.success_rate,
            ),
            reverse=True,
        )
        
        return matching_adapters
    
    def prune_adapters(self, min_success_rate: float = 0.5) -> None:
        """
        Prune underperforming adapters.
        
        Args:
            min_success_rate: Minimum success rate to keep an adapter
        """
        adapters_to_remove = []
        
        for adapter_id, adapter in self.adapters.items():
            if adapter.metadata.success_rate < min_success_rate:
                adapters_to_remove.append(adapter_id)
        
        # Remove adapters
        for adapter_id in adapters_to_remove:
            self.remove_adapter(adapter_id)
        
        logger.info(f"Pruned {len(adapters_to_remove)} adapters with success rate < {min_success_rate}")
    
    def remove_adapter(self, adapter_id: str) -> bool:
        """
        Remove an adapter.
        
        Args:
            adapter_id: ID of the adapter to remove
            
        Returns:
            bool: Whether the adapter was removed
        """
        # Check if adapter exists
        adapter = self.get_adapter(adapter_id)
        if not adapter:
            logger.error(f"Adapter {adapter_id} not found")
            return False
        
        # Deactivate if active
        if self.active_adapter == adapter_id:
            self.deactivate_adapter()
        
        # Remove from adapters dictionary
        del self.adapters[adapter_id]
        
        # Remove directory
        try:
            shutil.rmtree(adapter.path)
            logger.info(f"Removed adapter {adapter_id}")
            return True
        except Exception as e:
            logger.error(f"Error removing adapter {adapter_id}: {e}")
            return False
    
    def process_judge_feedback(self, score: float, feedback: str) -> None:
        """
        Process feedback from JudgeAgent.
        
        Args:
            score: Score from JudgeAgent
            feedback: Feedback from JudgeAgent
        """
        if not self.active_adapter:
            logger.warning("No active adapter to process feedback for")
            return
        
        # Get the active adapter
        adapter = self.get_adapter(self.active_adapter)
        if not adapter:
            logger.error(f"Active adapter {self.active_adapter} not found")
            return
        
        # Update success rate with exponential moving average
        alpha = 0.1  # Weight for new data
        old_rate = adapter.metadata.success_rate
        new_rate = (1 - alpha) * old_rate + alpha * score
        
        # Update metadata
        self.update_adapter_metadata(
            adapter_id=self.active_adapter,
            metadata_dict={
                "success_rate": new_rate,
            },
        )
        
        logger.info(
            f"Updated success rate for adapter {self.active_adapter} "
            f"from {old_rate:.2f} to {new_rate:.2f} based on score {score}"
        )
