from __future__ import annotations

"""
GASA service for integrating Graph-Aligned Sparse Attention functionality.

This module provides a central service for managing GASA functionality across
the Saplings framework, serving as an integration point for other components.
"""


import logging
from typing import TYPE_CHECKING, Any

from saplings.gasa.builder.standard_mask_builder import StandardMaskBuilder
from saplings.gasa.config import FallbackStrategy, GASAConfig
from saplings.gasa.core.graph_distance import GraphDistanceCalculator
from saplings.gasa.core.types import MaskFormat, MaskType
from saplings.gasa.packing.block_diagonal_packer import BlockDiagonalPacker
from saplings.gasa.prompt.prompt_composer import GASAPromptComposer
from saplings.gasa.visualization.mask_visualizer import MaskVisualizer
from saplings.memory.document import Document
from saplings.memory.graph import DependencyGraph

if TYPE_CHECKING:
    import numpy as np
    import scipy.sparse as sp

logger = logging.getLogger(__name__)


class GASAService:
    """
    Service for managing Graph-Aligned Sparse Attention functionality.

    This class serves as a central point of integration for GASA, managing
    the creation and coordination of mask builders, packers, and prompt composers.
    It also provides methods for applying GASA to prompts and inputs based on
    model capabilities and configuration.
    """

    def __init__(
        self,
        graph: DependencyGraph | None = None,
        config: GASAConfig | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        """
        Initialize the GASA service.

        Args:
        ----
            graph: Dependency graph
            config: GASA configuration
            tokenizer: Tokenizer for converting text to tokens

        """
        self.graph = graph
        # Use default configuration if none is provided
        self.config = config or GASAConfig.default()
        self.tokenizer = tokenizer

        # Initialize components as None - they'll be created on demand
        self._mask_builder = None
        self._block_packer = None
        self._prompt_composer = None
        self._visualizer = None
        self._graph_distance_calculator = None

    @property
    def graph_distance_calculator(self):
        """Get or create the graph distance calculator."""
        if self._graph_distance_calculator is None:
            if self.graph is None:
                msg = "Dependency graph is required for graph distance calculation"
                raise ValueError(msg)

            self._graph_distance_calculator = GraphDistanceCalculator(graph=self.graph)

        return self._graph_distance_calculator

    @property
    def mask_builder(self):
        """Get or create the mask builder."""
        if self._mask_builder is None:
            if self.graph is None:
                msg = "Dependency graph is required for mask building"
                raise ValueError(msg)

            self._mask_builder = StandardMaskBuilder(
                graph=self.graph,
                config=self.config,
                tokenizer=self.tokenizer,
            )

        return self._mask_builder

    @property
    def block_packer(self):
        """Get or create the block diagonal packer."""
        if self._block_packer is None:
            if self.graph is None:
                msg = "Dependency graph is required for block packing"
                raise ValueError(msg)

            self._block_packer = BlockDiagonalPacker(
                graph=self.graph,
                config=self.config,
                tokenizer=self.tokenizer,
            )

        return self._block_packer

    @property
    def prompt_composer(self):
        """Get or create the prompt composer."""
        if self._prompt_composer is None:
            if self.graph is None:
                msg = "Dependency graph is required for prompt composition"
                raise ValueError(msg)

            self._prompt_composer = GASAPromptComposer(
                graph=self.graph,
                config=self.config,
            )

        return self._prompt_composer

    @property
    def visualizer(self):
        """Get or create the mask visualizer."""
        if self._visualizer is None:
            self._visualizer = MaskVisualizer(config=self.config)

        return self._visualizer

    def build_mask(
        self,
        documents: list[Document],
        prompt: str,
        format: MaskFormat = MaskFormat.DENSE,
        mask_type: MaskType = MaskType.ATTENTION,
    ) -> np.ndarray | sp.spmatrix | list[dict[str, Any]]:
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
        # Convert documents to the expected type if needed
        return self.mask_builder.build_mask(
            documents=documents,  # type: ignore
            prompt=prompt,
            format=format,  # type: ignore
            mask_type=mask_type,  # type: ignore
        )

    def reorder_tokens(
        self,
        documents: list[Document],
        prompt: str,
        input_ids: list[int],
        attention_mask: np.ndarray | None = None,
    ) -> tuple[list[int], np.ndarray | list[int] | None, dict[int, int]]:
        """
        Reorder tokens to create a block-diagonal structure.

        Args:
        ----
            documents: Documents used in the prompt
            prompt: Prompt text
            input_ids: Token IDs
            attention_mask: Attention mask (optional)

        Returns:
        -------
            Tuple[List[int], Union[np.ndarray, List[int], None], Dict[int, int]]:
                Reordered token IDs, reordered attention mask, and mapping from
                original to reordered positions

        """
        return self.block_packer.reorder_tokens(
            documents=documents,  # type: ignore
            prompt=prompt,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    def restore_order(
        self,
        reordered_output: list[Any],
        position_mapping: dict[int, int],
    ) -> list[Any]:
        """
        Restore the original order of tokens.

        Args:
        ----
            reordered_output: Output with reordered tokens
            position_mapping: Mapping from original to reordered positions

        Returns:
        -------
            List[Any]: Output with original token order

        """
        return self.block_packer.restore_order(
            reordered_output=reordered_output,
            position_mapping=position_mapping,
        )

    def compose_prompt(
        self,
        documents: list[Document],
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        """
        Compose a prompt based on graph relationships.

        Args:
        ----
            documents: Documents to include in the prompt
            prompt: User prompt
            system_prompt: System prompt

        Returns:
        -------
            str: Composed prompt

        """
        return self.prompt_composer.compose_prompt(
            documents=documents,
            prompt=prompt,
            system_prompt=system_prompt,
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
            Optional[Any]: Figure if matplotlib is available

        """
        return self.visualizer.visualize_mask(
            mask=mask,
            format=format,
            mask_type=mask_type,
            output_path=output_path,
            title=title,
            show=show,
            token_labels=token_labels,
            highlight_tokens=highlight_tokens,
            figsize=figsize,
        )

    def clear_cache(self):
        """Clear all caches in GASA components."""
        if self._mask_builder is not None:
            self._mask_builder.clear_cache()

        if self._graph_distance_calculator is not None:
            self._graph_distance_calculator.distance_cache.clear()

        # Clear other component caches as needed

    def apply_gasa(
        self,
        documents: list[Document],
        prompt: str,
        input_ids: list[int] | None = None,
        attention_mask: np.ndarray | None = None,
        model_supports_sparse_attention: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Apply GASA to a prompt and inputs based on model capabilities.

        This method automatically selects the appropriate strategy based on
        the model's capabilities and the GASA configuration.

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
        if not self.config.enabled:
            # Return the original prompt and inputs if GASA is disabled
            return {
                "prompt": prompt,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

        # Select strategy based on model capabilities and configuration
        if model_supports_sparse_attention:
            # Model supports sparse attention, use mask builder
            mask = self.build_mask(
                documents=documents,
                prompt=prompt,
                format=MaskFormat.DENSE,
                mask_type=MaskType.ATTENTION,
            )

            return {
                "prompt": prompt,
                "input_ids": input_ids,
                "attention_mask": mask,
            }

        if (
            self.config.fallback_strategy == FallbackStrategy.BLOCK_DIAGONAL
            and input_ids is not None
        ):
            # Model doesn't support sparse attention, use block-diagonal packing
            reordered_input_ids, reordered_attention_mask, position_mapping = self.reorder_tokens(
                documents=documents,
                prompt=prompt,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            return {
                "prompt": prompt,
                "input_ids": reordered_input_ids,
                "attention_mask": reordered_attention_mask,
                "position_mapping": position_mapping,
            }

        if self.config.fallback_strategy == FallbackStrategy.PROMPT_COMPOSER:
            # Use prompt composer for third-party APIs
            system_prompt = kwargs.get("system_prompt")
            composed_prompt = self.compose_prompt(
                documents=documents,
                prompt=prompt,
                system_prompt=system_prompt,
            )

            return {
                "prompt": composed_prompt,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

        # No applicable strategy, return original
        logger.warning(
            f"No applicable GASA strategy for fallback_strategy={self.config.fallback_strategy}"
            f" and model_supports_sparse_attention={model_supports_sparse_attention}"
        )
        return {
            "prompt": prompt,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def set_graph(self, graph: DependencyGraph) -> None:
        """
        Set or update the dependency graph.

        Args:
        ----
            graph: New dependency graph

        """
        self.graph = graph

        # Reset components to force recreation with the new graph
        self._mask_builder = None
        self._block_packer = None
        self._prompt_composer = None
        self._graph_distance_calculator = None

    def set_config(self, config: GASAConfig) -> None:
        """
        Set or update the GASA configuration.

        Args:
        ----
            config: New GASA configuration

        """
        self.config = config

        # Reset components to force recreation with the new config
        self._mask_builder = None
        self._block_packer = None
        self._prompt_composer = None
        self._visualizer = None

    def set_tokenizer(self, tokenizer: Any) -> None:
        """
        Set or update the tokenizer.

        Args:
        ----
            tokenizer: New tokenizer

        """
        self.tokenizer = tokenizer

        # Reset components to force recreation with the new tokenizer
        self._mask_builder = None
        self._block_packer = None
