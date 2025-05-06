from __future__ import annotations

"""
PyTorch-based implementation of block-diagonal packing for GASA.

This module provides a high-performance implementation of block-diagonal
packing using PyTorch's built-in functions. It is significantly faster than
the pure Python implementation, especially when used with GPU acceleration.
"""


import logging
from typing import Any, List

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None  # type: ignore

logger = logging.getLogger(__name__)


def block_pack_torch(mask_list: List[Any]) -> Any:
    """
    Combine attention masks along the block diagonal using PyTorch.

    This implementation leverages PyTorch's built-in torch.block_diag function
    for high-performance block-diagonal packing. When run on a GPU, this can be
    orders of magnitude faster than the pure Python implementation.

    Args:
    ----
        mask_list: List of PyTorch tensor mask matrices

    Returns:
    -------
        torch.Tensor: Block diagonal matrix

    """
    if not HAS_TORCH or torch is None:
        msg = "PyTorch is not installed. Please install torch to use this function."
        raise ImportError(msg)

    if not mask_list:
        return torch.zeros((0, 0), device="cpu")

    # Use PyTorch's built-in block_diag function
    return torch.block_diag(*mask_list)


def convert_to_torch(mask_list: List[Any], device: str = "cpu") -> List[Any]:
    """
    Convert a list of masks to PyTorch tensors.

    Args:
    ----
        mask_list: List of masks (torch tensors or numpy arrays)
        device: PyTorch device to place tensors on

    Returns:
    -------
        List of PyTorch tensors

    """
    if not HAS_TORCH or torch is None:
        msg = "PyTorch is not installed. Please install torch to use this function."
        raise ImportError(msg)

    torch_masks = []
    for mask in mask_list:
        if isinstance(mask, torch.Tensor):
            # Move to specified device if needed
            if mask.device.type != device:
                mask = mask.to(device)
            torch_masks.append(mask)
        else:
            # Convert numpy array to torch tensor
            torch_masks.append(torch.tensor(mask, device=device))

    return torch_masks


def check_gpu_availability() -> bool:
    """
    Check if GPU acceleration is available for PyTorch.

    Returns
    -------
        bool: True if GPU is available, False otherwise

    """
    if not HAS_TORCH or torch is None:
        return False

    return torch.cuda.is_available()
