from __future__ import annotations

"""
Success pair collector module for Saplings.

This module provides the SuccessPairCollector class for collecting successful error-fix pairs.
"""


import json
import logging
import os
from datetime import datetime
from typing import TYPE_CHECKING, Any

from saplings.core.exceptions import DataError, SelfHealingError
from saplings.core.resilience import Validation, retry
from saplings.self_heal.interfaces import ISuccessPairCollector

if TYPE_CHECKING:
    from saplings.self_heal.patch_generator import Patch

logger = logging.getLogger(__name__)


class SuccessPairCollector(ISuccessPairCollector):
    """
    Collector for successful error-fix pairs.

    This class collects successful error-fix pairs for training models to generate patches.
    It handles both legacy patch objects and modern input/output pairs.
    """

    def __init__(
        self,
        output_dir: str = "success_pairs",
        max_pairs: int = 1000,
        *,
        storage_dir: str | None = None,  # Back-compat param name expected by tests
    ) -> None:
        """
        Initialize the success pair collector.

        Args:
        ----
            output_dir: Directory to store success pairs (preferred name)
            max_pairs: Maximum number of pairs to store
            storage_dir: **Deprecated**. Alias for ``output_dir`` kept for backward-compatibility.

        Raises:
        ------
            ValueError: If parameters are invalid

        """
        # Validate parameters
        Validation.require(max_pairs > 0, "max_pairs must be positive")

        # Accept either parameter name
        if storage_dir is not None:
            output_dir = storage_dir

        Validation.require_not_empty(output_dir, "output_dir")

        self.storage_dir = output_dir  # legacy attribute expected by old code/tests
        self.output_dir = output_dir  # Also store as output_dir for new code
        self.max_pairs = max_pairs
        self.pairs: list[dict[str, Any]] = []

        # Create the storage directory if it doesn't exist
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            msg = f"Failed to create output directory: {output_dir}"
            raise SelfHealingError(msg, cause=e)

        # Load existing pairs
        self._load_pairs()

    async def collect(
        self,
        input_text: str,
        output_text: str,
        context: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Collect a successful pair.

        Args:
        ----
            input_text: Input text (e.g., prompt, error, etc.)
            output_text: Output text (e.g., response, fix, etc.)
            context: Context documents
            metadata: Additional metadata

        Raises:
        ------
            ValueError: If input or output text is empty
            SelfHealingError: If there is a problem with collection or storage

        """
        # Validate inputs
        Validation.require_not_empty(input_text, "input_text")
        Validation.require_not_empty(output_text, "output_text")

        # Create the pair
        pair = {
            "input_text": input_text,
            "output_text": output_text,
            "context": context or [],
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
        }

        # Add the pair to the list
        self.pairs.append(pair)

        # If we've exceeded the maximum number of pairs, remove the oldest one
        if len(self.pairs) > self.max_pairs:
            self.pairs.pop(0)

        # Save the pairs
        try:
            await self._save_pairs()
            logger.info(f"Collected success pair, total pairs: {len(self.pairs)}")
        except Exception as e:
            logger.exception(f"Failed to save success pair: {e}")
            msg = f"Failed to save success pair: {e!s}"
            raise SelfHealingError(msg, cause=e)

    # Legacy method for backward compatibility
    async def collect_patch(self, patch: Patch, metadata: dict[str, Any] | None = None) -> None:
        """
        Collect a patch as a success pair (legacy format).

        Args:
        ----
            patch: Successful patch
            metadata: Additional metadata

        Raises:
        ------
            ValueError: If patch is None
            SelfHealingError: If there is a problem with collection or storage

        """
        # Validate inputs
        Validation.require_not_none(patch, "patch")

        # Convert patch to input/output format
        await self.collect(
            input_text=patch.error + "\n\n" + patch.original_code,
            output_text=patch.patched_code,
            context=[],
            metadata={
                "error_info": patch.error_info,
                "timestamp": patch.timestamp or datetime.now().isoformat(),
                **(metadata or {}),
            },
        )

    @retry(max_attempts=3)
    async def get_all_pairs(self):
        """
        Get all collected pairs.

        Returns
        -------
            List[Dict[str, Any]]: List of collected pairs

        Raises
        ------
            SelfHealingError: If there is a problem retrieving pairs

        """
        return self.pairs.copy()

    def clear(self):
        """
        Clear all collected pairs.

        Raises
        ------
            SelfHealingError: If there is a problem clearing pairs

        """
        self.pairs = []

        try:
            self._save_pairs_sync()
            logger.info("Cleared all success pairs")
        except Exception as e:
            logger.exception(f"Failed to clear success pairs: {e}")
            msg = f"Failed to clear success pairs: {e!s}"
            raise SelfHealingError(msg, cause=e)

    def _load_pairs(self):
        """
        Load pairs from storage.

        Raises
        ------
            SelfHealingError: If there is a problem loading pairs

        """
        pairs_file = os.path.join(self.storage_dir, "success_pairs.json")

        if os.path.exists(pairs_file):
            try:
                with open(pairs_file) as f:
                    self.pairs = json.load(f)

                logger.info(f"Loaded {len(self.pairs)} success pairs from {pairs_file}")
            except json.JSONDecodeError as e:
                logger.exception(f"Error parsing success pairs file: {e}")
                self.pairs = []
                msg = f"Invalid JSON in success pairs file: {e!s}"
                raise DataError(msg, cause=e)
            except Exception as e:
                logger.exception(f"Error loading success pairs: {e}")
                self.pairs = []
                msg = f"Failed to load success pairs: {e!s}"
                raise SelfHealingError(msg, cause=e)

    def _save_pairs_sync(self):
        """
        Save pairs to storage synchronously.

        Raises
        ------
            SelfHealingError: If there is a problem saving pairs

        """
        pairs_file = os.path.join(self.storage_dir, "success_pairs.json")

        try:
            # Ensure directory exists
            os.makedirs(self.storage_dir, exist_ok=True)

            with open(pairs_file, "w") as f:
                json.dump(self.pairs, f, indent=2)

            logger.debug(f"Saved {len(self.pairs)} success pairs to {pairs_file}")
        except TypeError as e:  # This is what json.dump raises for non-serializable objects
            logger.exception(f"Error encoding success pairs to JSON: {e}")
            msg = f"Failed to encode success pairs to JSON: {e!s}"
            raise DataError(msg, cause=e)
        except Exception as e:
            logger.exception(f"Error saving success pairs: {e}")
            msg = f"Failed to save success pairs: {e!s}"
            raise SelfHealingError(msg, cause=e)

    @retry(max_attempts=3)
    async def _save_pairs(self):
        """
        Save pairs to storage asynchronously.

        Raises
        ------
            SelfHealingError: If there is a problem saving pairs

        """
        self._save_pairs_sync()

    @retry(max_attempts=2)
    async def export_to_jsonl(self, output_file: str) -> None:
        """
        Export pairs to a JSONL file for training.

        Args:
        ----
            output_file: Output file path

        Raises:
        ------
            ValueError: If output_file is empty
            SelfHealingError: If there is a problem exporting pairs

        """
        # Validate parameters
        Validation.require_not_empty(output_file, "output_file")

        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

            with open(output_file, "w") as f:
                for pair in self.pairs:
                    f.write(json.dumps(pair) + "\n")

            logger.info(f"Exported {len(self.pairs)} success pairs to {output_file}")
        except TypeError as e:  # This is what json.dumps raises for non-serializable objects
            logger.exception(f"Error encoding success pairs to JSON: {e}")
            msg = f"Failed to encode success pairs to JSON: {e!s}"
            raise DataError(msg, cause=e)
        except Exception as e:
            logger.exception(f"Error exporting success pairs: {e}")
            msg = f"Failed to export success pairs: {e!s}"
            raise SelfHealingError(msg, cause=e)

    @retry(max_attempts=2)
    async def import_from_jsonl(self, input_file: str, append: bool = False) -> None:
        """
        Import pairs from a JSONL file.

        Args:
        ----
            input_file: Input file path
            append: Whether to append to existing pairs

        Raises:
        ------
            ValueError: If input_file is empty or doesn't exist
            DataError: If file contains invalid JSON
            SelfHealingError: If there is a problem importing pairs

        """
        # Validate parameters
        Validation.require_not_empty(input_file, "input_file")
        Validation.require(os.path.exists(input_file), f"Input file does not exist: {input_file}")

        if not append:
            self.pairs = []

        try:
            imported_count = 0
            with open(input_file) as f:
                for line in f:
                    if not line.strip():
                        continue

                    try:
                        pair = json.loads(line.strip())
                        self.pairs.append(pair)
                        imported_count += 1
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSON line: {e}")

            # If we've exceeded the maximum number of pairs, remove the oldest ones
            if len(self.pairs) > self.max_pairs:
                self.pairs = self.pairs[-self.max_pairs :]

            # Save the pairs
            await self._save_pairs()

            logger.info(f"Imported {imported_count} success pairs from {input_file}")
        except Exception as e:
            logger.exception(f"Error importing success pairs: {e}")
            msg = f"Failed to import success pairs: {e!s}"
            raise SelfHealingError(msg, cause=e)

    def get_statistics(self):
        """
        Get statistics about the collected pairs.

        Returns
        -------
            Dict[str, Any]: Statistics about the collected pairs

        """
        if not self.pairs:
            return {
                "total_pairs": 0,
                "error_types": {},
                "error_patterns": {},
                "pair_types": {},
            }

        # Count error types and patterns
        error_types = {}
        error_patterns = {}
        pair_types = {"input_output": 0, "error_fix": 0, "other": 0}

        for pair in self.pairs:
            # Categorize pair type
            if "input_text" in pair and "output_text" in pair:
                pair_types["input_output"] += 1
            elif "original_code" in pair and "patched_code" in pair:
                pair_types["error_fix"] += 1
            else:
                pair_types["other"] += 1

            # Count error types and patterns
            error_info = pair.get("error_info", {})
            if isinstance(error_info, dict):
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
            "pair_types": pair_types,
            "earliest_pair": self.pairs[0].get("timestamp", "unknown") if self.pairs else None,
            "latest_pair": self.pairs[-1].get("timestamp", "unknown") if self.pairs else None,
        }
