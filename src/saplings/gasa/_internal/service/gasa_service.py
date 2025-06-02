from __future__ import annotations

"""
GASA service for integrating Graph-Aligned Sparse Attention functionality.

This module provides a central service for managing GASA functionality across
the Saplings framework, serving as an integration point for other components.
"""


import logging
from collections.abc import Sequence
from typing import Any, Callable, Protocol, runtime_checkable

from saplings.api.core.interfaces import IGasaService as IGASAService
from saplings.gasa._internal.config import FallbackStrategy, GASAConfig
from saplings.gasa._internal.core.types import MaskFormat, MaskType

# Forward declarations for type hints
try:
    import numpy as np
except ImportError:
    np = Any  # type: ignore

try:
    import scipy.sparse as sp
except ImportError:
    sp = Any  # type: ignore

logger = logging.getLogger(__name__)


@runtime_checkable
class Document(Protocol):
    """Protocol for document objects."""

    id: str
    content: str
    metadata: dict[str, Any] | None

    def __getattr__(self, name: str) -> Any: ...


@runtime_checkable
class DependencyGraph(Protocol):
    """Protocol for dependency graph objects."""

    def get_neighbors(self, node_id: str) -> list[str]: ...
    def get_distance(self, source_id: str, target_id: str) -> int | float: ...
    def get_subgraph(self, node_ids: list[str], max_hops: int = 2) -> "DependencyGraph": ...
    def add_node(self, node_id: str, metadata: dict[str, Any] | None = None) -> None: ...
    def add_edge(
        self, source_id: str, target_id: str, metadata: dict[str, Any] | None = None
    ) -> None: ...
    def __getattr__(self, name: str) -> Any: ...


class GASAService(IGASAService):
    """
    Service for managing Graph-Aligned Sparse Attention functionality.

    This class serves as a central point of integration for GASA, managing
    the creation and coordination of mask builders, packers, and prompt composers.
    It also provides methods for applying GASA to prompts and inputs based on
    model capabilities and configuration.
    """

    def __init__(
        self,
        graph: Any | None = None,
        config: GASAConfig | None = None,
        tokenizer: Any | None = None,
        graph_provider: Callable[[], Any] | None = None,
        tokenizer_provider: Callable[[], Any] | None = None,
    ) -> None:
        """
        Initialize the GASA service.

        Args:
        ----
            graph: Dependency graph
            config: GASA configuration
            tokenizer: Tokenizer for converting text to tokens
            graph_provider: Optional provider function for lazy loading of dependency graph
            tokenizer_provider: Optional provider function for lazy loading of tokenizer

        """
        self._graph = graph
        self._graph_provider = graph_provider
        # Use default configuration if none is provided
        self.config = config or GASAConfig.default()
        self._tokenizer = tokenizer
        self._tokenizer_provider = tokenizer_provider
        self._initialized = False

        # Initialize components as None - they'll be created on demand
        self._mask_builder = None
        self._block_packer = None
        self._prompt_composer = None
        self._visualizer = None
        self._graph_distance_calculator = None

        # Track initialization status
        self._components_initialized = {
            "mask_builder": False,
            "block_packer": False,
            "prompt_composer": False,
            "visualizer": False,
            "graph_distance_calculator": False,
        }

    @property
    def enabled(self) -> bool:
        """
        Check if GASA is enabled.

        Returns
        -------
            bool: Whether GASA is enabled

        """
        return self.config.enabled

    @property
    def graph(self) -> Any:
        """
        Get the dependency graph, loading it on demand if necessary.

        Returns
        -------
            Any: The dependency graph (implements DependencyGraph protocol)

        Raises
        ------
            ValueError: If no graph or graph provider is available

        """
        if self._graph is None and self._graph_provider is not None:
            logger.debug("Lazy loading dependency graph for GASA service")
            self._graph = self._graph_provider()

        if self._graph is None:
            msg = (
                "Dependency graph is required but not available. Provide a graph or graph_provider."
            )
            raise ValueError(msg)

        return self._graph

    @property
    def tokenizer(self) -> Any:
        """
        Get the tokenizer, loading it on demand if necessary.

        Returns
        -------
            Any: The tokenizer

        Raises
        ------
            ValueError: If no tokenizer or tokenizer provider is available when needed

        """
        if self._tokenizer is None and self._tokenizer_provider is not None:
            logger.debug("Lazy loading tokenizer for GASA service")
            self._tokenizer = self._tokenizer_provider()

        return self._tokenizer

    @property
    def graph_distance_calculator(self):
        """
        Get or create the graph distance calculator.

        Returns
        -------
            GraphDistanceCalculator: The graph distance calculator

        Raises
        ------
            ValueError: If dependency graph is not available

        """
        if self._graph_distance_calculator is None:
            # Import here to avoid circular imports
            from saplings.gasa._internal.core.graph_distance import GraphDistanceCalculator

            logger.debug("Initializing graph distance calculator")
            self._graph_distance_calculator = GraphDistanceCalculator(graph=self.graph)
            self._components_initialized["graph_distance_calculator"] = True

        return self._graph_distance_calculator

    @property
    def mask_builder(self):
        """
        Get or create the mask builder.

        Returns
        -------
            StandardMaskBuilder: The mask builder

        Raises
        ------
            ValueError: If dependency graph is not available

        """
        if self._mask_builder is None:
            # Import here to avoid circular imports
            from saplings.gasa._internal.builder.standard_mask_builder import StandardMaskBuilder

            logger.debug("Initializing mask builder")
            self._mask_builder = StandardMaskBuilder(
                graph=self.graph,
                config=self.config,
                tokenizer=self.tokenizer,
            )
            self._components_initialized["mask_builder"] = True

        return self._mask_builder

    @property
    def block_packer(self):
        """
        Get or create the block diagonal packer.

        Returns
        -------
            BlockDiagonalPacker: The block diagonal packer

        Raises
        ------
            ValueError: If dependency graph is not available

        """
        if self._block_packer is None:
            # Import here to avoid circular imports
            from saplings.gasa._internal.packing.block_diagonal_packer import BlockDiagonalPacker

            logger.debug("Initializing block diagonal packer")
            self._block_packer = BlockDiagonalPacker(
                graph=self.graph,
                config=self.config,
                tokenizer=self.tokenizer,
            )
            self._components_initialized["block_packer"] = True

        return self._block_packer

    @property
    def prompt_composer(self):
        """
        Get or create the prompt composer.

        Returns
        -------
            GASAPromptComposer: The prompt composer

        Raises
        ------
            ValueError: If dependency graph is not available

        """
        if self._prompt_composer is None:
            # Import here to avoid circular imports
            from saplings.gasa._internal.prompt_composer import GASAPromptComposer

            logger.debug("Initializing prompt composer")
            self._prompt_composer = GASAPromptComposer(
                graph=self.graph,
                config=self.config,
            )
            self._components_initialized["prompt_composer"] = True

        return self._prompt_composer

    @property
    def visualizer(self):
        """
        Get or create the mask visualizer.

        Returns
        -------
            MaskVisualizer: The mask visualizer

        """
        if self._visualizer is None:
            # Import here to avoid circular imports
            from saplings.gasa._internal.visualization.mask_visualizer import MaskVisualizer

            logger.debug("Initializing mask visualizer")
            self._visualizer = MaskVisualizer(config=self.config)
            self._components_initialized["visualizer"] = True

        return self._visualizer

    def build_mask(
        self,
        documents: list[Any],
        prompt: str,
        format: MaskFormat = MaskFormat.DENSE,
        mask_type: MaskType = MaskType.ATTENTION,
    ) -> Any:
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
            Any: Attention mask (numpy array, scipy sparse matrix, or list of dicts)

        """
        # Convert documents to the expected type if needed
        return self.mask_builder.build_mask(
            documents=documents,
            prompt=prompt,
            format=format,
            mask_type=mask_type,
        )

    def reorder_tokens(
        self,
        documents: list[Any],
        prompt: str,
        input_ids: list[int],
        attention_mask: Any | None = None,
    ) -> tuple[list[int], Any | None, dict[int, int]]:
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
            Tuple[List[int], Any, Dict[int, int]]:
                Reordered token IDs, reordered attention mask, and mapping from
                original to reordered positions

        """
        return self.block_packer.reorder_tokens(
            documents=documents,
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
        documents: Sequence[Any],
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
        documents: list[Any],
        prompt: str,
        input_ids: list[int] | None = None,
        attention_mask: Any | None = None,
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

    def set_graph(self, graph: Any) -> None:
        """
        Set or update the dependency graph.

        Args:
        ----
            graph: New dependency graph

        """
        self._graph = graph
        self._graph_provider = None

        # Reset components to force recreation with the new graph
        self._mask_builder = None
        self._block_packer = None
        self._prompt_composer = None
        self._graph_distance_calculator = None

        # Reset initialization status
        self._components_initialized["mask_builder"] = False
        self._components_initialized["block_packer"] = False
        self._components_initialized["prompt_composer"] = False
        self._components_initialized["graph_distance_calculator"] = False

        logger.debug("Updated dependency graph for GASA service")

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

        # Reset initialization status
        self._components_initialized["mask_builder"] = False
        self._components_initialized["block_packer"] = False
        self._components_initialized["prompt_composer"] = False
        self._components_initialized["visualizer"] = False

        logger.debug("Updated configuration for GASA service")

    def set_tokenizer(self, tokenizer: Any) -> None:
        """
        Set or update the tokenizer.

        Args:
        ----
            tokenizer: New tokenizer

        """
        self._tokenizer = tokenizer
        self._tokenizer_provider = None

        # Reset components to force recreation with the new tokenizer
        self._mask_builder = None
        self._block_packer = None

        # Reset initialization status
        self._components_initialized["mask_builder"] = False
        self._components_initialized["block_packer"] = False

        logger.debug("Updated tokenizer for GASA service")
