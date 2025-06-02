from __future__ import annotations

"""
Token tracking mask builder for Graph-Aligned Sparse Attention (GASA).

This module provides an enhanced implementation of the MaskBuilderInterface that
builds attention masks based on document dependency graphs using a token tracking
approach for more robust chunk-to-token mapping.
"""

import logging
from typing import TYPE_CHECKING, Any

from saplings.gasa._internal.builder.standard_mask_builder import StandardMaskBuilder
from saplings.gasa._internal.core.chunk_info import ChunkInfo
from saplings.gasa._internal.core.token_mapper import TokenMapper

if TYPE_CHECKING:
    from saplings.gasa._internal.config import GASAConfig
    from saplings.memory._internal.document import Document
    from saplings.memory._internal.graph import DependencyGraph

logger = logging.getLogger(__name__)


class TokenTrackingMaskBuilder(StandardMaskBuilder):
    """
    Enhanced mask builder that tracks tokens during prompt construction.

    This class extends the StandardMaskBuilder with more robust token tracking
    to ensure accurate mapping between tokens and document chunks.
    """

    def __init__(
        self,
        graph: DependencyGraph,
        config: GASAConfig | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        """
        Initialize the token tracking mask builder.

        Args:
        ----
            graph: Dependency graph
            config: GASA configuration
            tokenizer: Tokenizer for converting text to tokens

        """
        super().__init__(graph, config, tokenizer)
        self.token_mapper = None
        if self.tokenizer is not None:
            self.token_mapper = TokenMapper(self.tokenizer)

    def set_token_mapper(self, token_mapper: TokenMapper) -> None:
        """
        Set the token mapper for this builder.

        Args:
        ----
            token_mapper: Token mapper to use

        """
        self.token_mapper = token_mapper

    def _map_tokens_to_chunks(
        self,
        documents: list[Document],
        prompt: str,
        input_ids: list[int] | None = None,
    ) -> list[ChunkInfo]:
        """
        Map tokens to document chunks using the token mapper if available.

        Args:
        ----
            documents: Documents used in the prompt
            prompt: Prompt text
            input_ids: Token IDs

        Returns:
        -------
            List[ChunkInfo]: Chunk information for each token

        """
        # If we have a token mapper with chunk infos, use it
        if self.token_mapper is not None and self.token_mapper.chunk_infos:
            logger.info(f"Using token mapper with {len(self.token_mapper.chunk_infos)} chunk infos")
            return self.token_mapper.chunk_infos

        # Otherwise, fall back to the standard implementation
        logger.info("Token mapper not available or empty, falling back to standard implementation")
        return super()._map_tokens_to_chunks(documents, prompt, input_ids)
