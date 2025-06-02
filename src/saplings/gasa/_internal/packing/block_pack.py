from __future__ import annotations

"""
Auto-selecting block diagonal packer for GASA.

This module provides a unified interface for block-diagonal packing that automatically
selects the best available implementation:
1. PyTorch (fastest, GPU-accelerated)
2. SciPy (fast CPU-only version)
3. Pure Python (fallback)
"""


import logging
from typing import Any, List, Union

import numpy as np

# Type definitions for better type checking
ArrayType = np.ndarray

# Use centralized lazy import system
from saplings._internal.optional_deps import OPTIONAL_DEPENDENCIES

HAS_TORCH = OPTIONAL_DEPENDENCIES["torch"].available


# Lazy import function for PyTorch components using centralized system
def _get_torch_components():
    """Lazy import PyTorch components using centralized system."""
    torch_module = OPTIONAL_DEPENDENCIES["torch"].require()

    from saplings.gasa._internal.packing.torch_block_packer import (
        block_pack_torch,
        convert_to_torch,
    )

    return torch_module, block_pack_torch, convert_to_torch


# Try to import SciPy implementation
try:
    from scipy import linalg

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

logger = logging.getLogger(__name__)

# Define type variables for better type hints
MaskList = List[Any]  # Use Any to handle both numpy arrays and torch tensors


def block_pack(
    mask_list: MaskList,
) -> Union[ArrayType, Any]:
    """
    Combine attention masks along block diagonal using the best available backend.

    Args:
    ----
        mask_list: List of mask matrices (numpy arrays or torch tensors)

    Returns:
    -------
        Block diagonal matrix

    """
    # Check if input is empty
    if not mask_list:
        if HAS_TORCH:
            torch, _, _ = _get_torch_components()
            return torch.zeros((0, 0), device="cpu")
        return np.empty((0, 0))

    # Determine input type
    using_torch = False
    if HAS_TORCH:
        torch, _, _ = _get_torch_components()
        # Check if any input is a torch tensor
        for m in mask_list:
            if isinstance(m, torch.Tensor):
                using_torch = True
                break

    # PyTorch implementation (fastest, especially with GPU)
    if using_torch:
        logger.debug("Using PyTorch-based block packer")
        # Get torch components
        torch, block_pack_torch, convert_to_torch = _get_torch_components()
        # Convert numpy arrays to tensors if mixed
        torch_masks = convert_to_torch(list(mask_list))
        return block_pack_torch(torch_masks)

    # Convert any torch tensors to numpy arrays
    mask_list_np = []
    for m in mask_list:
        # Check if it's a torch tensor
        if HAS_TORCH:
            torch, _, _ = _get_torch_components()
            if isinstance(m, torch.Tensor):
                mask_list_np.append(m.detach().cpu().numpy())
                continue
        # If not a torch tensor or torch not available
        mask_list_np.append(m)

    # SciPy implementation (fast CPU implementation)
    if HAS_SCIPY:
        logger.debug("Using SciPy-based block packer")
        return linalg.block_diag(*mask_list_np)

    # Fallback to pure Python implementation
    logger.debug(
        "Using pure Python block packer (consider installing PyTorch or SciPy for better performance)"
    )
    return block_pack_python(mask_list_np)


def block_pack_python(mask_list: list[np.ndarray]) -> np.ndarray:
    """
    Pure Python implementation of block packing.

    Args:
    ----
        mask_list: List of mask matrices

    Returns:
    -------
        Block diagonal matrix

    """
    if not mask_list:
        return np.empty((0, 0))

    # Determine the total size
    total_rows = sum(mask.shape[0] for mask in mask_list)
    total_cols = sum(mask.shape[1] for mask in mask_list)

    # Create the output matrix
    result = np.zeros((total_rows, total_cols), dtype=mask_list[0].dtype)

    # Fill in the blocks
    row_offset = 0
    col_offset = 0

    for mask in mask_list:
        rows, cols = mask.shape
        result[row_offset : row_offset + rows, col_offset : col_offset + cols] = mask
        row_offset += rows
        col_offset += cols

    return result


def get_block_diagonal_packer(use_torch: bool = True):
    """
    Get the appropriate block diagonal packer based on available backends.

    Args:
    ----
        use_torch: Whether to prefer PyTorch backend (if available)

    Returns:
    -------
        Block diagonal packer function

    """
    if use_torch and HAS_TORCH:
        _, block_pack_torch, _ = _get_torch_components()
        return block_pack_torch
    if HAS_SCIPY:
        return lambda masks: linalg.block_diag(
            *[m.numpy() if hasattr(m, "numpy") else m for m in masks]
        )
    return block_pack_python
