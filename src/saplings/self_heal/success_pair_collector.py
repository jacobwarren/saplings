"""
Success pair collector module for Saplings.

This module provides the SuccessPairCollector class for collecting successful error-fix pairs.
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Union

from saplings.self_heal.patch_generator import Patch

logger = logging.getLogger(__name__)


class SuccessPairCollector:
    """
    Collector for successful error-fix pairs.
    
    This class collects successful error-fix pairs for training models to generate patches.
    """
    
    def __init__(
        self,
        storage_dir: str = "success_pairs",
        max_pairs: int = 1000,
    ):
        """
        Initialize the success pair collector.
        
        Args:
            storage_dir: Directory to store success pairs
            max_pairs: Maximum number of pairs to store
        """
        self.storage_dir = storage_dir
        self.max_pairs = max_pairs
        self.pairs: List[Dict] = []
        
        # Create the storage directory if it doesn't exist
        os.makedirs(storage_dir, exist_ok=True)
        
        # Load existing pairs
        self._load_pairs()
    
    def collect(self, patch: Patch) -> None:
        """
        Collect a successful patch.
        
        Args:
            patch: Successful patch
        """
        # Create a pair from the patch
        pair = {
            "original_code": patch.original_code,
            "patched_code": patch.patched_code,
            "error": patch.error,
            "error_info": patch.error_info,
            "timestamp": datetime.now().isoformat(),
        }
        
        # Add the pair to the list
        self.pairs.append(pair)
        
        # If we've exceeded the maximum number of pairs, remove the oldest one
        if len(self.pairs) > self.max_pairs:
            self.pairs.pop(0)
        
        # Save the pairs
        self._save_pairs()
        
        logger.info(f"Collected success pair, total pairs: {len(self.pairs)}")
    
    def get_pairs(self) -> List[Dict]:
        """
        Get all collected pairs.
        
        Returns:
            List[Dict]: List of collected pairs
        """
        return self.pairs
    
    def clear(self) -> None:
        """Clear all collected pairs."""
        self.pairs = []
        self._save_pairs()
        
        logger.info("Cleared all success pairs")
    
    def _load_pairs(self) -> None:
        """Load pairs from storage."""
        pairs_file = os.path.join(self.storage_dir, "success_pairs.json")
        
        if os.path.exists(pairs_file):
            try:
                with open(pairs_file, "r") as f:
                    self.pairs = json.load(f)
                
                logger.info(f"Loaded {len(self.pairs)} success pairs from {pairs_file}")
            except Exception as e:
                logger.error(f"Error loading success pairs: {e}")
                self.pairs = []
    
    def _save_pairs(self) -> None:
        """Save pairs to storage."""
        pairs_file = os.path.join(self.storage_dir, "success_pairs.json")
        
        try:
            with open(pairs_file, "w") as f:
                json.dump(self.pairs, f, indent=2)
            
            logger.debug(f"Saved {len(self.pairs)} success pairs to {pairs_file}")
        except Exception as e:
            logger.error(f"Error saving success pairs: {e}")
    
    def export_to_jsonl(self, output_file: str) -> None:
        """
        Export pairs to a JSONL file for training.
        
        Args:
            output_file: Output file path
        """
        try:
            with open(output_file, "w") as f:
                for pair in self.pairs:
                    f.write(json.dumps(pair) + "\n")
            
            logger.info(f"Exported {len(self.pairs)} success pairs to {output_file}")
        except Exception as e:
            logger.error(f"Error exporting success pairs: {e}")
    
    def import_from_jsonl(self, input_file: str, append: bool = False) -> None:
        """
        Import pairs from a JSONL file.
        
        Args:
            input_file: Input file path
            append: Whether to append to existing pairs
        """
        if not append:
            self.pairs = []
        
        try:
            with open(input_file, "r") as f:
                for line in f:
                    pair = json.loads(line.strip())
                    self.pairs.append(pair)
            
            # If we've exceeded the maximum number of pairs, remove the oldest ones
            if len(self.pairs) > self.max_pairs:
                self.pairs = self.pairs[-self.max_pairs:]
            
            # Save the pairs
            self._save_pairs()
            
            logger.info(f"Imported {len(self.pairs)} success pairs from {input_file}")
        except Exception as e:
            logger.error(f"Error importing success pairs: {e}")
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about the collected pairs.
        
        Returns:
            Dict: Statistics about the collected pairs
        """
        if not self.pairs:
            return {
                "total_pairs": 0,
                "error_types": {},
                "error_patterns": {},
            }
        
        # Count error types
        error_types = {}
        error_patterns = {}
        
        for pair in self.pairs:
            error_info = pair.get("error_info", {})
            error_type = error_info.get("type", "Unknown")
            
            # Count error type
            error_types[error_type] = error_types.get(error_type, 0) + 1
            
            # Count error patterns
            for pattern in error_info.get("patterns", []):
                error_patterns[pattern] = error_patterns.get(pattern, 0) + 1
        
        return {
            "total_pairs": len(self.pairs),
            "error_types": error_types,
            "error_patterns": error_patterns,
        }
