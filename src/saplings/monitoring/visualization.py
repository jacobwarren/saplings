from __future__ import annotations

"""
Visualization module for Saplings monitoring.

This module provides visualization components for monitoring data,
including GASA heatmap and performance visualizations.
"""


import json
import logging
import os
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

from saplings.gasa.config import FallbackStrategy, GASAConfig, MaskStrategy
from saplings.monitoring.config import MonitoringConfig, VisualizationFormat

if TYPE_CHECKING:
    from saplings.gasa.mask_builder import MaskType


# Define MaskFormat enum locally to avoid import issues
class MaskFormat(str, Enum):
    """Format of attention masks."""

    DENSE = "dense"  # Dense matrix (numpy array)
    SPARSE = "sparse"  # Sparse matrix (scipy.sparse)
    SPARSE_TENSOR = "sparse_tensor"  # Sparse tensor format (list of dicts)


logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    from matplotlib import ticker

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning(
        "Matplotlib not installed. Static visualizations will not be available. "
        "Install matplotlib with: pip install matplotlib"
    )

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning(
        "Plotly not installed. Interactive visualizations will not be available. "
        "Install plotly with: pip install plotly"
    )


class GASAHeatmap:
    """
    GASA heatmap visualization component.

    This class provides visualization utilities for GASA attention masks,
    including heatmaps, token highlighting, and mask analysis.
    """

    def __init__(
        self,
        config: MonitoringConfig | None = None,
        gasa_config: GASAConfig | None = None,
    ) -> None:
        """
        Initialize the GASA heatmap visualizer.

        Args:
        ----
            config: Monitoring configuration
            gasa_config: GASA configuration

        """
        self.config = config or MonitoringConfig()
        self.gasa_config = gasa_config or GASAConfig(
            enabled=True,
            max_hops=2,
            mask_strategy=MaskStrategy.BINARY,
            fallback_strategy=FallbackStrategy.BLOCK_DIAGONAL,
            global_tokens=["[CLS]", "[SEP]", "<s>", "</s>", "[SUM]"],
            summary_token="[SUM]",
            add_summary_token=True,
            block_size=512,
            overlap=64,
            soft_mask_temperature=0.1,
            cache_masks=True,
            cache_dir=None,
            visualize=False,
            visualization_dir=None,
            enable_shadow_model=False,
            shadow_model_name="Qwen/Qwen3-1.8B",
            shadow_model_device="cpu",
            shadow_model_cache_dir=None,
            enable_prompt_composer=False,
            focus_tags=True,
            core_tag="[CORE_CTX]",
            near_tag="[NEAR_CTX]",
            summary_tag="[SUMMARY_CTX]",
        )

        # Create output directory if it doesn't exist
        os.makedirs(self.config.visualization_output_dir, exist_ok=True)

    def visualize_mask(
        self,
        mask: np.ndarray | list[dict[str, Any]],
        format: MaskFormat,
        mask_type: MaskType,
        output_path: str | None = None,
        title: str | None = None,
        show: bool = False,
        token_labels: list[str] | None = None,
        highlight_tokens: list[int] | None = None,
        interactive: bool = True,
    ) -> Any | None:
        """
        Visualize a GASA attention mask.

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
            interactive: Whether to create an interactive visualization

        Returns:
        -------
            Optional[Any]: Visualization object if available

        """
        if interactive and PLOTLY_AVAILABLE:
            return self._visualize_mask_interactive(
                mask=mask,
                format=format,
                mask_type=mask_type,
                output_path=output_path,
                title=title,
                show=show,
                token_labels=token_labels,
                highlight_tokens=highlight_tokens,
            )
        if MATPLOTLIB_AVAILABLE:
            return self._visualize_mask_static(
                mask=mask,
                format=format,
                mask_type=mask_type,
                output_path=output_path,
                title=title,
                show=show,
                token_labels=token_labels,
                highlight_tokens=highlight_tokens,
            )
        logger.warning("Neither Plotly nor Matplotlib is installed. Cannot visualize mask.")
        return None

    def visualize_mask_comparison(
        self,
        masks: list[tuple[np.ndarray | list[dict[str, Any]], MaskFormat, MaskType, str]],
        output_path: str | None = None,
        title: str | None = None,
        show: bool = False,
        interactive: bool = True,
    ) -> Any | None:
        """
        Visualize a comparison of multiple attention masks.

        Args:
        ----
            masks: List of (mask, format, type, label) tuples
            output_path: Path to save the visualization
            title: Title for the visualization
            show: Whether to show the visualization
            interactive: Whether to create an interactive visualization

        Returns:
        -------
            Optional[Any]: Visualization object if available

        """
        if interactive and PLOTLY_AVAILABLE:
            return self._visualize_mask_comparison_interactive(
                masks=masks,
                output_path=output_path,
                title=title,
                show=show,
            )
        if MATPLOTLIB_AVAILABLE:
            return self._visualize_mask_comparison_static(
                masks=masks,
                output_path=output_path,
                title=title,
                show=show,
            )
        logger.warning(
            "Neither Plotly nor Matplotlib is installed. Cannot visualize mask comparison."
        )
        return None

    def visualize_mask_sparsity(
        self,
        mask: np.ndarray | list[dict[str, Any]],
        format: MaskFormat,
        mask_type: MaskType,
        output_path: str | None = None,
        title: str | None = None,
        show: bool = False,
        interactive: bool = True,
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
            interactive: Whether to create an interactive visualization

        Returns:
        -------
            Optional[Any]: Visualization object if available

        """
        # Convert mask to dense format
        dense_mask = self._convert_to_dense(mask, format)

        # Calculate sparsity
        total_elements = dense_mask.size
        nonzero_elements = np.count_nonzero(dense_mask)
        zero_elements = total_elements - nonzero_elements
        sparsity = zero_elements / total_elements

        if interactive and PLOTLY_AVAILABLE:
            # Create figure
            fig = go.Figure()

            # Add bar chart
            fig.add_trace(
                go.Bar(
                    x=["Non-zero", "Zero"],
                    y=[nonzero_elements, zero_elements],
                    marker_color=["#1f77b4", "#ff7f0e"],
                    text=[
                        f"{nonzero_elements} ({nonzero_elements / total_elements:.2%})",
                        f"{zero_elements} ({sparsity:.2%})",
                    ],
                    textposition="auto",
                )
            )

            # Update layout
            fig.update_layout(
                title=title or f"Attention Mask Sparsity: {sparsity:.2%}",
                xaxis_title="Element Type",
                yaxis_title="Count",
                height=500,
                width=700,
            )

            # Save figure
            if output_path:
                self._save_visualization(fig, output_path, VisualizationFormat.HTML)

            # Show figure
            if show:
                fig.show()

            return fig

        if MATPLOTLIB_AVAILABLE:
            # Create figure
            fig, ax = plt.subplots(figsize=(7, 5))

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

            # Add labels
            ax.set_xlabel("Element Type")
            ax.set_ylabel("Count")

            # Tight layout
            plt.tight_layout()

            # Save figure
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches="tight")
                logger.info(f"Saved visualization to {output_path}")

            # Show figure
            if show:
                plt.show()

            return fig

        logger.warning(
            "Neither Plotly nor Matplotlib is installed. Cannot visualize mask sparsity."
        )
        return None

    def _visualize_mask_interactive(
        self,
        mask: np.ndarray | list[dict[str, Any]],
        format: MaskFormat,
        mask_type: MaskType,
        output_path: str | None = None,
        title: str | None = None,
        show: bool = False,
        token_labels: list[str] | None = None,
        highlight_tokens: list[int] | None = None,
    ) -> Any | None:
        """
        Create an interactive visualization of an attention mask.

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

        Returns:
        -------
            Optional[Any]: Visualization object if available

        """
        # Convert mask to dense format
        dense_mask = self._convert_to_dense(mask, format)

        # Create figure
        fig = go.Figure()

        # Add heatmap
        fig.add_trace(
            go.Heatmap(
                z=dense_mask,
                x=token_labels or list(range(dense_mask.shape[1])),
                y=token_labels or list(range(dense_mask.shape[0])),
                colorscale="Blues",
                showscale=True,
                hovertemplate="From: %{y}<br>To: %{x}<br>Value: %{z}<extra></extra>",
            )
        )

        # Highlight tokens
        if highlight_tokens:
            for idx in highlight_tokens:
                # Add horizontal line
                fig.add_shape(
                    type="line",
                    x0=-0.5,
                    x1=dense_mask.shape[1] - 0.5,
                    y0=idx,
                    y1=idx,
                    line={"color": "red", "width": 2, "dash": "dash"},
                )

                # Add vertical line
                fig.add_shape(
                    type="line",
                    x0=idx,
                    x1=idx,
                    y0=-0.5,
                    y1=dense_mask.shape[0] - 0.5,
                    line={"color": "red", "width": 2, "dash": "dash"},
                )

        # Update layout
        fig.update_layout(
            title=title or f"Attention Mask ({mask_type.value})",
            xaxis_title="To Token",
            yaxis_title="From Token",
            height=800,
            width=800,
        )

        # Save figure
        if output_path:
            self._save_visualization(fig, output_path, VisualizationFormat.HTML)

        # Show figure
        if show:
            fig.show()

        return fig

    def _visualize_mask_static(
        self,
        mask: np.ndarray | list[dict[str, Any]],
        format: MaskFormat,
        mask_type: MaskType,
        output_path: str | None = None,
        title: str | None = None,
        show: bool = False,
        token_labels: list[str] | None = None,
        highlight_tokens: list[int] | None = None,
    ) -> Any | None:
        """
        Create a static visualization of an attention mask.

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

        Returns:
        -------
            Optional[Any]: Visualization object if available

        """
        # Convert mask to dense format
        dense_mask = self._convert_to_dense(mask, format)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10))

        # Set title
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"Attention Mask ({mask_type.value})")

        # Create colormap
        cmap = "Blues"  # Use string name instead of attribute access

        # Plot mask
        im = ax.imshow(dense_mask, cmap=cmap, interpolation="nearest")

        # Add colorbar
        plt.colorbar(im, ax=ax)

        # Add token labels if provided
        if token_labels:
            # Limit the number of labels to avoid overcrowding
            if len(token_labels) > 20:
                step = len(token_labels) // 20
                indices = list(range(0, len(token_labels), step))
                if indices[-1] != len(token_labels) - 1:
                    indices.append(len(token_labels) - 1)

                x_labels = [
                    token_labels[i] if i in indices else "" for i in range(len(token_labels))
                ]
                y_labels = [
                    token_labels[i] if i in indices else "" for i in range(len(token_labels))
                ]
            else:
                x_labels = token_labels
                y_labels = token_labels

            ax.set_xticks(range(len(x_labels)))
            ax.set_yticks(range(len(y_labels)))
            ax.set_xticklabels(x_labels, rotation=90)
            ax.set_yticklabels(y_labels)

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
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved visualization to {output_path}")

        # Show figure
        if show:
            plt.show()

        return fig

    def _visualize_mask_comparison_interactive(
        self,
        masks: list[tuple[np.ndarray | list[dict[str, Any]], MaskFormat, MaskType, str]],
        output_path: str | None = None,
        title: str | None = None,
        show: bool = False,
    ) -> Any | None:
        """
        Create an interactive visualization comparing multiple attention masks.

        Args:
        ----
            masks: List of (mask, format, type, label) tuples
            output_path: Path to save the visualization
            title: Title for the visualization
            show: Whether to show the visualization

        Returns:
        -------
            Optional[Any]: Visualization object if available

        """
        # Create figure
        n_masks = len(masks)
        fig = make_subplots(
            rows=1,
            cols=n_masks,
            subplot_titles=[label for _, _, _, label in masks],
        )

        # Add each mask
        for i, (mask, format, _mask_type, _) in enumerate(masks):
            # Convert mask to dense format
            dense_mask = self._convert_to_dense(mask, format)

            # Add heatmap
            fig.add_trace(
                go.Heatmap(
                    z=dense_mask,
                    colorscale="Blues",
                    showscale=(i == n_masks - 1),  # Only show colorbar for last mask
                ),
                row=1,
                col=i + 1,
            )

        # Update layout
        fig.update_layout(
            title=title or "Attention Mask Comparison",
            height=600,
            width=300 * n_masks,
        )

        # Save figure
        if output_path:
            self._save_visualization(fig, output_path, VisualizationFormat.HTML)

        # Show figure
        if show:
            fig.show()

        return fig

    def _visualize_mask_comparison_static(
        self,
        masks: list[tuple[np.ndarray | list[dict[str, Any]], MaskFormat, MaskType, str]],
        output_path: str | None = None,
        title: str | None = None,
        show: bool = False,
    ) -> Any | None:
        """
        Create a static visualization comparing multiple attention masks.

        Args:
        ----
            masks: List of (mask, format, type, label) tuples
            output_path: Path to save the visualization
            title: Title for the visualization
            show: Whether to show the visualization

        Returns:
        -------
            Optional[Any]: Visualization object if available

        """
        # Create figure
        n_masks = len(masks)
        fig, axes = plt.subplots(1, n_masks, figsize=(5 * n_masks, 5))

        # Set title
        if title:
            fig.suptitle(title, fontsize=16)

            # Convert to single axis if only one mask\1# Index for n masks\1N_MASKS_INDEX = 1\1if n_masks == N_MASKS_INDEX:
            axes = [axes]

        # Plot each mask
        for i, (mask, format, _mask_type, label) in enumerate(masks):
            # Convert mask to dense format
            dense_mask = self._convert_to_dense(mask, format)

            # Plot mask
            im = axes[i].imshow(dense_mask, cmap="Blues", interpolation="nearest")

            # Set title
            axes[i].set_title(label)

            # Add colorbar
            plt.colorbar(im, ax=axes[i])

            # Remove ticks
            axes[i].set_xticks([])
            axes[i].set_yticks([])

        # Tight layout
        plt.tight_layout()

        # Save figure
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved visualization to {output_path}")

        # Show figure
        if show:
            plt.show()

        return fig

    def _convert_to_dense(
        self,
        mask: np.ndarray | list[dict[str, Any]],
        format: MaskFormat,
    ) -> np.ndarray:
        """
        Convert a mask to dense format.

        Args:
        ----
            mask: Mask to convert
            format: Format of the mask

        Returns:
        -------
            np.ndarray: Dense mask

        """
        if format == MaskFormat.DENSE:
            return np.array(mask)
        if format == MaskFormat.SPARSE_TENSOR:
            # Convert sparse tensor format to dense
            # Assuming mask is a list of dictionaries with 'i', 'j', and 'v' keys
            indices = [(item["i"], item["j"]) for item in mask]
            values = [item["v"] for item in mask]

            # Determine shape
            shape = (
                max(idx[0] for idx in indices) + 1,
                max(idx[1] for idx in indices) + 1,
            )

            # Create dense mask
            dense_mask = np.zeros(shape)
            for (i, j), v in zip(indices, values):
                dense_mask[i, j] = v

            return dense_mask
        msg = f"Unsupported mask format: {format}"
        raise ValueError(msg)

    def _save_visualization(
        self,
        fig: Any,
        output_path: str,
        format: VisualizationFormat,
    ) -> None:
        """
        Save a visualization to disk.

        Args:
        ----
            fig: Visualization to save
            output_path: Path to save the visualization
            format: Format of the visualization

        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save based on the type of figure
        if PLOTLY_AVAILABLE and hasattr(fig, "write_html"):
            # Plotly figure
            if format == VisualizationFormat.HTML:
                fig.write_html(output_path)
            elif format in (VisualizationFormat.PNG, VisualizationFormat.SVG):
                fig.write_image(output_path)
            elif format == VisualizationFormat.JSON:
                with open(output_path, "w") as f:
                    json.dump(fig.to_dict(), f, indent=2)
            else:
                logger.warning(f"Unsupported visualization format for Plotly: {format}")
                # Default to HTML
                fig.write_html(output_path)
        elif MATPLOTLIB_AVAILABLE and hasattr(fig, "savefig"):
            # Matplotlib figure
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
        else:
            logger.warning(f"Unsupported figure type for saving: {type(fig)}")

        logger.info(f"Saved visualization to {output_path}")


class PerformanceVisualizer:
    """
    Performance visualization component.

    This class provides visualization utilities for performance metrics,
    including latency, throughput, and resource usage.
    """

    def __init__(
        self,
        config: MonitoringConfig | None = None,
    ) -> None:
        """
        Initialize the performance visualizer.

        Args:
        ----
            config: Monitoring configuration

        """
        self.config = config or MonitoringConfig()

        # Create output directory if it doesn't exist
        os.makedirs(self.config.visualization_output_dir, exist_ok=True)

    def visualize_latency(
        self,
        latencies: dict[str, list[float]],
        output_path: str | None = None,
        title: str | None = None,
        show: bool = False,
        interactive: bool = True,
    ) -> Any | None:
        """
        Visualize latency metrics.

        Args:
        ----
            latencies: Dictionary mapping component names to latency lists (in ms)
            output_path: Path to save the visualization
            title: Title for the visualization
            show: Whether to show the visualization
            interactive: Whether to create an interactive visualization

        Returns:
        -------
            Optional[Any]: Visualization object if available

        """
        if interactive and PLOTLY_AVAILABLE:
            # Create figure
            fig = go.Figure()

            # Add box plots for each component
            for component, values in latencies.items():
                fig.add_trace(
                    go.Box(
                        y=values,
                        name=component,
                        boxpoints="all",
                        jitter=0.3,
                        pointpos=-1.8,
                    )
                )

            # Update layout
            fig.update_layout(
                title=title or "Latency by Component",
                yaxis_title="Latency (ms)",
                height=600,
                width=800,
            )

            # Save figure
            if output_path:
                self._save_visualization(fig, output_path, VisualizationFormat.HTML)

            # Show figure
            if show:
                fig.show()

            return fig

        if MATPLOTLIB_AVAILABLE:
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))

            # Create box plot
            components = list(latencies.keys())
            ax.boxplot(
                [latencies[component] for component in components],
                showfliers=True,
            )
            ax.set_xticklabels(components)

            # Set title and labels
            if title:
                ax.set_title(title)
            else:
                ax.set_title("Latency by Component")

            ax.set_ylabel("Latency (ms)")

            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45, ha="right")

            # Tight layout
            plt.tight_layout()

            # Save figure
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches="tight")
                logger.info(f"Saved visualization to {output_path}")

            # Show figure
            if show:
                plt.show()

            return fig

        logger.warning("Neither Plotly nor Matplotlib is installed. Cannot visualize latency.")
        return None

    def visualize_throughput(
        self,
        throughputs: dict[str, list[float]],
        output_path: str | None = None,
        title: str | None = None,
        show: bool = False,
        interactive: bool = True,
    ) -> Any | None:
        """
        Visualize throughput metrics.

        Args:
        ----
            throughputs: Dictionary mapping component names to throughput lists (in requests/sec)
            output_path: Path to save the visualization
            title: Title for the visualization
            show: Whether to show the visualization
            interactive: Whether to create an interactive visualization

        Returns:
        -------
            Optional[Any]: Visualization object if available

        """
        if interactive and PLOTLY_AVAILABLE:
            # Create figure
            fig = go.Figure()

            # Add bar charts for each component
            for component, values in throughputs.items():
                fig.add_trace(
                    go.Bar(
                        y=[np.mean(values)],
                        x=[component],
                        name=component,
                        error_y={"type": "data", "array": [np.std(values)], "visible": True},
                    )
                )

            # Update layout
            fig.update_layout(
                title=title or "Throughput by Component",
                yaxis_title="Throughput (requests/sec)",
                height=600,
                width=800,
            )

            # Save figure
            if output_path:
                self._save_visualization(fig, output_path, VisualizationFormat.HTML)

            # Show figure
            if show:
                fig.show()

            return fig

        if MATPLOTLIB_AVAILABLE:
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))

            # Get components and mean throughputs
            components = list(throughputs.keys())
            means = [np.mean(throughputs[component]) for component in components]
            stds = [np.std(throughputs[component]) for component in components]

            # Create bar chart
            ax.bar(components, means, yerr=stds, capsize=10)

            # Set title and labels
            if title:
                ax.set_title(title)
            else:
                ax.set_title("Throughput by Component")

            ax.set_ylabel("Throughput (requests/sec)")

            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45, ha="right")

            # Tight layout
            plt.tight_layout()

            # Save figure
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches="tight")
                logger.info(f"Saved visualization to {output_path}")

            # Show figure
            if show:
                plt.show()

            return fig

        logger.warning("Neither Plotly nor Matplotlib is installed. Cannot visualize throughput.")
        return None

    def visualize_error_rate(
        self,
        error_rates: dict[str, list[float]],
        output_path: str | None = None,
        title: str | None = None,
        show: bool = False,
        interactive: bool = True,
    ) -> Any | None:
        """
        Visualize error rate metrics.

        Args:
        ----
            error_rates: Dictionary mapping component names to error rate lists (as fractions)
            output_path: Path to save the visualization
            title: Title for the visualization
            show: Whether to show the visualization
            interactive: Whether to create an interactive visualization

        Returns:
        -------
            Optional[Any]: Visualization object if available

        """
        if interactive and PLOTLY_AVAILABLE:
            # Create figure
            fig = go.Figure()

            # Add line charts for each component
            for component, values in error_rates.items():
                fig.add_trace(
                    go.Scatter(
                        y=values,
                        mode="lines+markers",
                        name=component,
                    )
                )

            # Update layout
            fig.update_layout(
                title=title or "Error Rate by Component",
                yaxis_title="Error Rate",
                height=600,
                width=800,
                yaxis={
                    "tickformat": ".1%",
                    "range": [0, max([max(rates) for rates in error_rates.values()]) * 1.1],
                },
            )

            # Save figure
            if output_path:
                self._save_visualization(fig, output_path, VisualizationFormat.HTML)

            # Show figure
            if show:
                fig.show()

            return fig

        if MATPLOTLIB_AVAILABLE:
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot error rates for each component
            for component, values in error_rates.items():
                ax.plot(range(len(values)), values, marker="o", label=component)

            # Set title and labels
            if title:
                ax.set_title(title)
            else:
                ax.set_title("Error Rate by Component")

            ax.set_ylabel("Error Rate")
            ax.set_xlabel("Sample")

            # Format y-axis as percentage
            ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))

            # Add legend
            ax.legend()

            # Tight layout
            plt.tight_layout()

            # Save figure
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches="tight")
                logger.info(f"Saved visualization to {output_path}")

            # Show figure
            if show:
                plt.show()

            return fig

        logger.warning("Neither Plotly nor Matplotlib is installed. Cannot visualize error rate.")
        return None

    def _save_visualization(
        self,
        fig: Any,
        output_path: str,
        format: VisualizationFormat,
    ) -> None:
        """
        Save a visualization to disk.

        Args:
        ----
            fig: Visualization to save
            output_path: Path to save the visualization
            format: Format of the visualization

        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save based on the type of figure
        if PLOTLY_AVAILABLE and hasattr(fig, "write_html"):
            # Plotly figure
            if format == VisualizationFormat.HTML:
                fig.write_html(output_path)
            elif format in (VisualizationFormat.PNG, VisualizationFormat.SVG):
                fig.write_image(output_path)
            elif format == VisualizationFormat.JSON:
                with open(output_path, "w") as f:
                    json.dump(fig.to_dict(), f, indent=2)
            else:
                logger.warning(f"Unsupported visualization format for Plotly: {format}")
                # Default to HTML
                fig.write_html(output_path)
        elif MATPLOTLIB_AVAILABLE and hasattr(fig, "savefig"):
            # Matplotlib figure
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
        else:
            logger.warning(f"Unsupported figure type for saving: {type(fig)}")

        logger.info(f"Saved visualization to {output_path}")
