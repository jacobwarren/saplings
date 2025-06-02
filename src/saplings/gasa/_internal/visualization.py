from __future__ import annotations

"""
Visualization module for Graph-Aligned Sparse Attention (GASA).

This module provides visualization utilities for debugging attention masks.
"""


import json
import logging
import os
from typing import TYPE_CHECKING, Any, Union, cast

import numpy as np

from saplings.gasa.config import GASAConfig
from saplings.gasa.mask_builder import MaskFormat, MaskType

if TYPE_CHECKING:
    import scipy.sparse as sp

logger = logging.getLogger(__name__)

try:
    import matplotlib  # noqa: F401
    import matplotlib.pyplot as plt
    from matplotlib import cm

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

    # Create dummy modules for type hints
    class DummyModule:
        def __getattr__(self, _name: str) -> Any:
            return None

    plt = DummyModule()
    cm = DummyModule()


class MaskVisualizer:
    """
    Visualizer for Graph-Aligned Sparse Attention (GASA) masks.

    This class provides visualization utilities for debugging attention masks.
    """

    def __init__(
        self,
        config: GASAConfig | None = None,
    ) -> None:
        """
        Initialize the mask visualizer.

        Args:
        ----
            config: GASA configuration

        """
        # Use provided config or create a default one with explicit values
        if config is None:
            from saplings.gasa.config import FallbackStrategy, MaskStrategy

            self.config = GASAConfig(
                enabled=True,  # Default GASA settings for visualization context
                max_hops=2,
                mask_strategy=MaskStrategy.BINARY,
                fallback_strategy=FallbackStrategy.BLOCK_DIAGONAL,
                global_tokens=["[CLS]", "[SEP]", "<s>", "</s>", "[SUM]"],
                summary_token="[SUM]",
                add_summary_token=True,
                block_size=512,
                overlap=64,
                soft_mask_temperature=0.1,
                cache_masks=False,  # Caching not relevant for visualizer itself
                cache_dir=None,
                visualize=True,  # Default to visualize=True contextually
                visualization_dir=None,
                enable_shadow_model=False,
                shadow_model_name="Qwen/Qwen3-1.8B",  # Match default from config.py
                shadow_model_device="cpu",
                shadow_model_cache_dir=None,
                enable_prompt_composer=False,
                focus_tags=False,
                core_tag="[CORE_CTX]",
                near_tag="[NEAR_CTX]",
                summary_tag="[SUMMARY_CTX]",
            )
        else:
            self.config = config

        if not MATPLOTLIB_AVAILABLE:
            logger.warning(
                "Matplotlib not installed. Visualizations will not be available. "
                "Install matplotlib with: pip install matplotlib"
            )

    def visualize_mask(
        self,
        mask: np.ndarray | sp.spmatrix | list[dict[str, Any]],
        format: MaskFormat,
        mask_type: MaskType,
        output_path: str | None = None,
        title: str | None = None,
        show: bool = False,
        token_labels: list[str] | None = None,
        highlight_tokens: list[int] | None = None,
        figsize: tuple[int, int] = (10, 10),
    ) -> Any | None:
        """
        Visualize an attention mask.

        Args:
        ----
            mask: Attention mask
            format: Format of the mask
            mask_type: Type of attention mask
            output_path: Path to save the visualization
            title: Title for the visualization
            show: Whether to show the visualization
            token_labels: Labels for tokens
            highlight_tokens: Indices of tokens to highlight
            figsize: Figure size

        Returns:
        -------
            Optional[plt.Figure]: Figure if matplotlib is available

        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not installed. Cannot visualize mask.")
            return None

        # Convert mask to dense format
        dense_mask = self._convert_to_dense(mask, format)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Set title
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"Attention Mask ({mask_type.value})")

        # Create colormap
        cmap = cm.get_cmap("Blues")

        # Plot mask
        im = ax.imshow(dense_mask, cmap=cmap, interpolation="nearest")

        # Add colorbar
        plt.colorbar(im, ax=ax)

        # Set tick labels
        if token_labels:
            # Limit the number of labels to avoid overcrowding
            max_labels = 20
            if len(token_labels) > max_labels:
                # Show labels at regular intervals
                step = len(token_labels) // max_labels
                indices = list(range(0, len(token_labels), step))
                if indices[-1] != len(token_labels) - 1:
                    indices.append(len(token_labels) - 1)

                # Create tick positions and labels
                tick_positions = indices
                tick_labels = [token_labels[i] for i in indices]
            else:
                tick_positions = list(range(len(token_labels)))
                tick_labels = token_labels

            # Set tick positions and labels
            ax.set_xticks(tick_positions)
            ax.set_yticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=90)
            ax.set_yticklabels(tick_labels)

        # Highlight tokens
        if highlight_tokens:
            for idx in highlight_tokens:
                # Highlight row and column
                ax.axhline(y=idx, color="red", linestyle="--", alpha=0.5)
                ax.axvline(x=idx, color="red", linestyle="--", alpha=0.5)

        # Add grid
        ax.grid(False)

        # Tight layout
        plt.tight_layout()

        # Save figure
        if output_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Save figure
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved visualization to {output_path}")

        # Show figure
        if show:
            plt.show()

        return fig

    def visualize_mask_sparsity(
        self,
        mask: np.ndarray | sp.spmatrix | list[dict[str, Any]],
        format: MaskFormat,
        mask_type: MaskType,
        output_path: str | None = None,
        title: str | None = None,
        show: bool = False,
        figsize: tuple[int, int] = (10, 5),
    ) -> Any | None:
        """
        Visualize the sparsity of an attention mask.

        Args:
        ----
            mask: Attention mask
            format: Format of the mask
            mask_type: Type of attention mask
            output_path: Path to save the visualization
            title: Title for the visualization
            show: Whether to show the visualization
            figsize: Figure size

        Returns:
        -------
            Optional[plt.Figure]: Figure if matplotlib is available

        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not installed. Cannot visualize mask sparsity.")
            return None

        # Convert mask to dense format
        dense_mask = self._convert_to_dense(mask, format)

        # Calculate sparsity
        total_elements = dense_mask.size
        nonzero_elements = np.count_nonzero(dense_mask)
        zero_elements = total_elements - nonzero_elements
        sparsity = zero_elements / total_elements

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Set title
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"Attention Mask Sparsity: {sparsity:.2%}")

        # Create bar chart
        labels = ["Non-zero", "Zero"]
        values = [nonzero_elements, zero_elements]
        colors = ["#1f77b4", "#ff7f0e"]

        ax.bar(labels, values, color=colors)

        # Add value labels
        for i, v in enumerate(values):
            ax.text(i, v + 0.1, f"{v} ({v / total_elements:.2%})", ha="center")

        # Add grid
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        # Tight layout
        plt.tight_layout()

        # Save figure
        if output_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Save figure
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved sparsity visualization to {output_path}")

        # Show figure
        if show:
            plt.show()

        return fig

    def visualize_mask_comparison(
        self,
        masks: list[
            tuple[np.ndarray | sp.spmatrix | list[dict[str, Any]], MaskFormat, MaskType, str]
        ],
        output_path: str | None = None,
        title: str | None = None,
        show: bool = False,
        figsize: tuple[int, int] = (15, 10),
    ) -> Any | None:
        """
        Visualize a comparison of multiple attention masks.

        Args:
        ----
            masks: List of (mask, format, type, label) tuples
            output_path: Path to save the visualization
            title: Title for the visualization
            show: Whether to show the visualization
            figsize: Figure size

        Returns:
        -------
            Optional[plt.Figure]: Figure if matplotlib is available

        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not installed. Cannot visualize mask comparison.")
            return None

        # Create figure
        n_masks = len(masks)
        fig, axes = plt.subplots(1, n_masks, figsize=figsize)

        # Set title
        if title:
            fig.suptitle(title, fontsize=16)

        # Ensure axes is always iterable, even for a single subplot
        if n_masks == 1:
            axes = [axes]

        # Plot each mask
        for i, (mask, format, _mask_type, label) in enumerate(masks):
            # Convert mask to dense format
            dense_mask = self._convert_to_dense(mask, format)

            # Plot mask
            im = axes[i].imshow(dense_mask, cmap=cm.get_cmap("Blues"), interpolation="nearest")

            # Set title
            axes[i].set_title(label)

            # Add colorbar
            plt.colorbar(im, ax=axes[i])

            # Remove ticks
            axes[i].set_xticks([])
            axes[i].set_yticks([])

        # Tight layout
        plt.tight_layout(rect=(0, 0, 1, 0.95))

        # Save figure
        if output_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Save figure
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved comparison visualization to {output_path}")

        # Show figure
        if show:
            plt.show()

        return fig

    def _convert_to_dense(
        self,
        mask: np.ndarray | sp.spmatrix | list[dict[str, Any]],
        format: MaskFormat,
    ) -> np.ndarray:
        """
        Convert a mask to dense format.

        Args:
        ----
            mask: Attention mask
            format: Format of the mask

        Returns:
        -------
            np.ndarray: Dense mask

        """
        if format == MaskFormat.DENSE:
            if not isinstance(mask, np.ndarray):
                logger.warning(
                    f"Expected np.ndarray for DENSE format, got {type(mask)}. Attempting conversion."
                )
            # Ensure it's a numpy array
            return np.asarray(mask, dtype=np.int32)

        if format == MaskFormat.SPARSE:
            # Handle sparse matrices
            import scipy.sparse as sp_local  # Use local import to avoid top-level dependency if matplotlib fails

            if not isinstance(mask, sp_local.spmatrix):
                logger.warning(
                    f"Expected sp.spmatrix for SPARSE format, got {type(mask)}. Attempting conversion."
                )
                # Attempt conversion, might fail if not convertible
                try:
                    mask = sp_local.csr_matrix(mask)
                except Exception as e:
                    logger.error(f"Failed to convert mask to sparse matrix: {e}")
                    # Return empty array as fallback
                    return np.zeros((0, 0), dtype=np.int32)

            # Handle known sparse types with toarray
            if isinstance(mask, (sp_local.csr_matrix, sp_local.csc_matrix, sp_local.coo_matrix)):
                known_sparse_mask = cast(
                    "Union[sp_local.csr_matrix, sp_local.csc_matrix, sp_local.coo_matrix]", mask
                )
                return np.asarray(known_sparse_mask.toarray(), dtype=np.int32)

            # Fallback for other sparse types (less common)
            logger.warning(
                f"Attempting conversion for potentially unsupported sparse matrix type: {type(mask)}. Using toarray() if available."
            )
            if hasattr(mask, "toarray"):
                # Attempt to call toarray directly if it exists
                try:
                    # Cast mask to Any to satisfy Pylance before calling toarray
                    return np.asarray(cast("Any", mask).toarray(), dtype=np.int32)
                except Exception as toarray_error:
                    logger.error(f"Calling toarray failed for {type(mask)}: {toarray_error}")

            # Last resort: direct conversion (might fail)
            logger.warning(
                f"No standard conversion method found for {type(mask)}. Attempting direct np.array conversion."
            )
            try:
                return np.asarray(np.array(mask), dtype=np.int32)
            except Exception as direct_conversion_error:
                logger.error(
                    f"Direct conversion failed for {type(mask)}: {direct_conversion_error}"
                )
                shape = mask.shape if hasattr(mask, "shape") else (0, 0)
                return np.zeros(shape, dtype=np.int32)  # Return empty on failure

        if format == MaskFormat.BLOCK_SPARSE:
            # Ensure mask is a list of dictionaries
            if not isinstance(mask, list):
                msg = f"Block sparse mask must be a list of dictionaries, got {type(mask)}"
                raise TypeError(msg)

            # Determine mask dimensions
            max_row = max(block["row"] + block["size_row"] for block in mask)
            max_col = max(block["col"] + block["size_col"] for block in mask)

            # Create dense mask
            dense_mask = np.zeros((max_row, max_col), dtype=np.int32)

            # Fill in blocks
            for block in mask:
                row = block["row"]
                col = block["col"]
                size_row = block["size_row"]
                size_col = block["size_col"]
                block_data = block["block"]

                dense_mask[row : row + size_row, col : col + size_col] = block_data

            return dense_mask

        msg = f"Unsupported format: {format}"
        raise ValueError(msg)

    def calculate_mask_statistics(
        self,
        mask: np.ndarray | sp.spmatrix | list[dict[str, Any]],
        format: MaskFormat,
    ) -> dict[str, Any]:
        """
        Calculate statistics for an attention mask.

        Args:
        ----
            mask: Attention mask
            format: Format of the mask

        Returns:
        -------
            Dict[str, Any]: Mask statistics

        """
        # Convert mask to dense format
        dense_mask = self._convert_to_dense(mask, format)

        # Calculate statistics
        total_elements = dense_mask.size
        nonzero_elements = np.count_nonzero(dense_mask)
        zero_elements = total_elements - nonzero_elements
        sparsity = zero_elements / total_elements

        # Calculate row and column statistics
        row_nonzeros = np.count_nonzero(dense_mask, axis=1)
        col_nonzeros = np.count_nonzero(dense_mask, axis=0)

        row_sparsity = 1 - (row_nonzeros / dense_mask.shape[1])
        col_sparsity = 1 - (col_nonzeros / dense_mask.shape[0])

        # Return statistics
        return {
            "total_elements": int(total_elements),
            "nonzero_elements": int(nonzero_elements),
            "zero_elements": int(zero_elements),
            "sparsity": float(sparsity),
            "shape": dense_mask.shape,
            "row_nonzeros": row_nonzeros.tolist(),
            "col_nonzeros": col_nonzeros.tolist(),
            "row_sparsity": row_sparsity.tolist(),
            "col_sparsity": col_sparsity.tolist(),
            "min_row_nonzeros": int(np.min(row_nonzeros)),
            "max_row_nonzeros": int(np.max(row_nonzeros)),
            "avg_row_nonzeros": float(np.mean(row_nonzeros)),
            "min_col_nonzeros": int(np.min(col_nonzeros)),
            "max_col_nonzeros": int(np.max(col_nonzeros)),
            "avg_col_nonzeros": float(np.mean(col_nonzeros)),
        }

    def save_statistics(
        self,
        statistics: dict[str, Any],
        output_path: str,
    ) -> None:
        """
        Save mask statistics to disk.

        Args:
        ----
            statistics: Mask statistics
            output_path: Path to save the statistics

        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Convert numpy arrays to lists
        for key, value in statistics.items():
            if isinstance(value, np.ndarray):
                statistics[key] = value.tolist()

        # Save statistics
        with open(output_path, "w") as f:
            json.dump(statistics, f, indent=2)

        logger.info(f"Saved mask statistics to {output_path}")
