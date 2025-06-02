from __future__ import annotations

"""
Null implementation of the GASA service.

This module provides a null implementation of the GASA service that can be used
when GASA is disabled. It follows the IGASAService interface but performs no
actual GASA operations, simply passing through inputs unchanged.
"""

import logging
from typing import Any, Dict, List, Optional, Protocol, Union, runtime_checkable

import numpy as np
import scipy.sparse as sp

from saplings.api.core.interfaces import IGasaService as IGASAService
from saplings.gasa._internal.config import GASAConfig


# Define Document protocol locally to avoid circular imports
@runtime_checkable
class Document(Protocol):
    """Protocol for document objects."""

    id: str
    content: str
    metadata: Dict[str, Any] | None


logger = logging.getLogger(__name__)


class NullGASAService(IGASAService):
    """
    Null implementation of the GASA service.

    This class implements the IGASAService interface but performs no actual GASA
    operations. It's used when GASA is disabled to maintain the same interface
    without the overhead of GASA processing.
    """

    def __init__(self, config: Optional[GASAConfig] = None) -> None:
        """
        Initialize the null GASA service.

        Args:
        ----
            config: Optional GASA configuration (ignored in this implementation)

        """
        if config is None:
            # Create a default config with enabled=False
            default_config = GASAConfig.default()
            default_config.enabled = False
            self._config = default_config
        else:
            self._config = config
        logger.debug("Initialized NullGASAService (GASA disabled)")

    @property
    def enabled(self) -> bool:
        """
        Check if GASA is enabled.

        Returns
        -------
            bool: Always False for NullGASAService

        """
        return False

    def build_mask(
        self,
        documents: List[Document],
        prompt: str,
        format: str = "dense",
        mask_type: str = "attention",
    ) -> Union[np.ndarray, sp.spmatrix, List[Dict[str, Any]]]:
        """
        Build a dummy attention mask (no-op).

        Args:
        ----
            documents: Documents used in the prompt (ignored)
            prompt: Prompt text (ignored)
            format: Output format for the mask (ignored)
            mask_type: Type of attention mask (ignored)

        Returns:
        -------
            np.ndarray: Empty attention mask

        """
        logger.debug("NullGASAService.build_mask called (no-op)")
        # Return an empty numpy array as a placeholder
        return np.array([])

    def apply_gasa(
        self,
        documents: List[Document],
        prompt: str,
        input_ids: Optional[List[int]] = None,
        attention_mask: Optional[np.ndarray] = None,
        model_supports_sparse_attention: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Apply GASA to a prompt (no-op).

        Args:
        ----
            documents: Documents used in the prompt (ignored)
            prompt: Prompt text
            input_ids: Token IDs (passed through)
            attention_mask: Attention mask (passed through)
            model_supports_sparse_attention: Whether the model supports sparse attention (ignored)
            **kwargs: Additional parameters (ignored)

        Returns:
        -------
            Dict[str, Any]: Unchanged inputs

        """
        logger.debug("NullGASAService.apply_gasa called (no-op)")
        # Simply pass through the inputs unchanged
        return {
            "prompt": prompt,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def compose_prompt(
        self,
        documents: List[Document],
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Compose a prompt (no-op).

        Args:
        ----
            documents: Documents used in the prompt (ignored)
            prompt: Prompt text
            system_prompt: Optional system prompt (ignored)

        Returns:
        -------
            str: Unchanged prompt

        """
        logger.debug("NullGASAService.compose_prompt called (no-op)")
        # Simply return the original prompt unchanged
        return prompt
