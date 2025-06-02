from __future__ import annotations

"""Interface for GASA (Graph-Aligned Sparse Attention) operations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy
import scipy

# Always define MaskType at module scope using fully qualified string-based forward references
MaskType = Union["numpy.ndarray", "scipy.sparse.spmatrix", list[dict[str, Any]], Any]


@dataclass
class GasaConfig:
    """Configuration for GASA operations."""

    max_hops: int = 2
    use_shadow_model: bool = False
    shadow_model_name: str = "Qwen/Qwen3-0.6B"
    mask_format: str = "dense"
    mask_type: str = "attention"


class IGasaService(ABC):
    """Interface for GASA operations."""

    @property
    @abstractmethod
    def enabled(self):
        """Whether GASA is enabled."""

    @abstractmethod
    def build_mask(
        self,
        documents: list[Any],
        prompt: str,
        format: str = "dense",
        mask_type: str = "attention",
    ) -> MaskType:
        """
        Build an attention mask based on the dependency graph.

        Args:
        ----
            documents: Documents used in the prompt
            prompt: Prompt text
            format: Output format for the mask
            mask_type: Type of attention mask

        Returns:
        -------
            Union[np.ndarray, sp.spmatrix, List[Dict[str, Any]]]: Attention mask

        """

    @abstractmethod
    def apply_gasa(
        self,
        documents: list[Any],
        prompt: str,
        input_ids: list[int] | None = None,
        attention_mask: Any | None = None,
        model_supports_sparse_attention: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Apply GASA to a prompt and inputs based on model capabilities.

        Args:
        ----
            documents: Documents used in the prompt
            prompt: Prompt text
            input_ids: Token IDs (optional)
            attention_mask: Attention mask (optional)
            model_supports_sparse_attention: Whether the model supports sparse attention
            **kwargs: Additional parameters

        Returns:
        -------
            Dict[str, Any]: Result containing the modified prompt, input_ids, attention_mask,
                           and any additional information needed by the model

        """

    @abstractmethod
    def compose_prompt(
        self,
        documents: list[Any],
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        """
        Compose a prompt with graph-aware context.

        Args:
        ----
            documents: Documents used in the prompt
            prompt: Prompt text
            system_prompt: Optional system prompt

        Returns:
        -------
            str: Composed prompt

        """

    @abstractmethod
    async def create_attention_mask(
        self,
        documents: List[Any],
        config: Optional[GasaConfig] = None,
    ) -> Dict[str, Any]:
        """
        Create an attention mask from documents.

        Args:
        ----
            documents: Documents to create mask from
            config: Optional GASA configuration

        Returns:
        -------
            Dict[str, Any]: Attention mask and related data

        """
