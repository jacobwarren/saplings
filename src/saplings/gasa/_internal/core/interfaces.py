from __future__ import annotations

"""
Interfaces for Graph-Aligned Sparse Attention (GASA).

This module provides interfaces (abstract base classes) for the GASA module,
defining contracts that implementations must adhere to.
"""


import abc
from typing import TYPE_CHECKING, Any

from saplings.gasa._internal.core.types import MaskFormat, MaskType

if TYPE_CHECKING:
    import numpy as np
    import scipy.sparse as sp

    from saplings.memory._internal.document import Document


class MaskBuilderInterface(abc.ABC):
    """Interface for mask builders."""

    @abc.abstractmethod
    def build_mask(
        self,
        documents: list[Document],
        prompt: str,
        format: MaskFormat = MaskFormat.DENSE,
        mask_type: MaskType = MaskType.ATTENTION,
    ) -> np.ndarray | sp.spmatrix | list[dict[str, Any]]:
        """
        Build an attention mask based on the document dependency graph.

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

    @abc.abstractmethod
    def save_mask(
        self,
        mask: np.ndarray | sp.spmatrix | list[dict[str, Any]],
        file_path: str,
        format: MaskFormat,
        mask_type: MaskType,
    ) -> None:
        """
        Save a mask to disk.

        Args:
        ----
            mask: Attention mask
            file_path: Path to save the mask
            format: Format of the mask
            mask_type: Type of attention mask

        """

    @abc.abstractmethod
    def load_mask(
        self,
        file_path: str,
    ) -> tuple[np.ndarray | sp.spmatrix | list[dict[str, Any]], MaskFormat, MaskType]:
        """
        Load a mask from disk.

        Args:
        ----
            file_path: Path to load the mask from

        Returns:
        -------
            Tuple[Union[np.ndarray, sp.spmatrix, List[Dict[str, Any]]], MaskFormat, MaskType]:
                Mask, format, and type

        """

    @abc.abstractmethod
    def clear_cache(self):
        """Clear the mask cache."""
