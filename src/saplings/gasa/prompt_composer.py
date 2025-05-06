from __future__ import annotations

"""
Graph-aware prompt composer for GASA with third-party LLM APIs.

This module provides a prompt composer that structures prompts based on
graph relationships for use with third-party LLM APIs like OpenAI and Anthropic.
"""


import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from saplings.gasa.config import FallbackStrategy, GASAConfig, MaskStrategy

if TYPE_CHECKING:
    from saplings.memory import DependencyGraph, Document


def _extract_metadata_value(metadata: Any, *keys: str) -> str | None:
    """
    Extract value from metadata object or dict with nested attribute support.

    Args:
    ----
        metadata: Metadata object or dictionary
        *keys: Keys to search for in order of preference

    Returns:
    -------
        str | None: Found value or None if not found

    """
    if metadata is None:
        return None

    try:
        if isinstance(metadata, dict):
            # Search in main dict
            for key in keys:
                if key in metadata:
                    value = metadata[key]
                    if isinstance(value, (str, int, float, bool)):
                        return str(value)

            # Search in nested custom dict
            custom = metadata.get("custom", {})
            if isinstance(custom, dict):
                for key in keys:
                    if key in custom:
                        value = custom[key]
                        if isinstance(value, (str, int, float, bool)):
                            return str(value)

            # Search in nested metadata
            for key in keys:
                parts = key.split(".")
                value = metadata
                for part in parts:
                    if isinstance(value, dict) and part in value:
                        value = value[part]
                    else:
                        value = None
                        break
                if value is not None and isinstance(value, (str, int, float, bool)):
                    return str(value)

        else:
            # Search in object attributes
            for key in keys:
                if hasattr(metadata, key):
                    value = getattr(metadata, key)
                    if isinstance(value, (str, int, float, bool)):
                        return str(value)

            # Search in nested custom object
            if hasattr(metadata, "custom"):
                custom = metadata.custom
                if isinstance(custom, dict):
                    for key in keys:
                        if key in custom:
                            value = custom[key]
                            if isinstance(value, (str, int, float, bool)):
                                return str(value)
                elif custom is not None:
                    for key in keys:
                        if hasattr(custom, key):
                            value = getattr(custom, key)
                            if isinstance(value, (str, int, float, bool)):
                                return str(value)

            # Search in nested attributes
            for key in keys:
                parts = key.split(".")
                value = metadata
                for part in parts:
                    if hasattr(value, part):
                        value = getattr(value, part)
                    else:
                        value = None
                        break
                if value is not None and isinstance(value, (str, int, float, bool)):
                    return str(value)

    except Exception as e:
        logger.warning(f"Error extracting metadata value: {e!s}")

    return None


logger = logging.getLogger(__name__)


class GASAPromptComposer:
    """
    Graph-aware prompt composer for GASA with third-party LLM APIs.

    This class structures prompts based on graph relationships for use with
    third-party LLM APIs like OpenAI and Anthropic. It implements the block-diagonal
    packing strategy and adds focus tags to important context.
    """

    def __init__(
        self,
        graph: DependencyGraph,
        config: GASAConfig | None = None,
    ) -> None:
        """
        Initialize the prompt composer.

        Args:
        ----
            graph: Dependency graph
            config: GASA configuration

        """
        self.graph = graph
        self.config = config or GASAConfig(
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
            enable_prompt_composer=True,
            focus_tags=True,
            core_tag="[CORE_CTX]",
            near_tag="[NEAR_CTX]",
            summary_tag="[SUMMARY_CTX]",
        )

        # Cache for distance calculations
        self._distance_cache: dict[tuple[str, str], int] = {}

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
        # Sort documents by graph distance
        sorted_docs = self._sort_documents_by_graph_distance(documents)

        # Group documents by distance category
        core_docs, near_docs, far_docs = self._group_documents_by_distance(sorted_docs)

        # Create summaries for far documents
        summaries = self._create_summaries(far_docs)

        # Compose the final prompt
        return self._build_composed_prompt(core_docs, near_docs, summaries, prompt, system_prompt)

    def _sort_documents_by_graph_distance(
        self, documents: list[Document]
    ) -> list[tuple[Document, int]]:
        """
        Sort documents by graph distance.

        Args:
        ----
            documents: Documents to sort

        Returns:
        -------
            List[Tuple[Document, int]]: Sorted documents with distances

        """
        # Calculate the average distance from each document to all others
        doc_distances = []
        for doc in documents:
            total_distance = 0
            valid_connections = 0
            for other_doc in documents:
                if doc.id != other_doc.id:
                    distance = self._get_graph_distance(doc.id, other_doc.id)
                    if distance is not None:
                        total_distance += distance
                        valid_connections += 1

            # Calculate average distance or use a large value if no connections
            avg_distance = (
                total_distance / valid_connections if valid_connections > 0 else float("inf")
            )
            doc_distances.append((doc, avg_distance))

        # Sort by average distance (ascending)
        return sorted(doc_distances, key=lambda x: x[1])

    def _group_documents_by_distance(
        self, sorted_docs: list[tuple[Document, int]]
    ) -> tuple[list[Document], list[Document], list[Document]]:
        """
        Group documents by distance category.

        Args:
        ----
            sorted_docs: Sorted documents with distances

        Returns:
        -------
            Tuple[List[Document], List[Document], List[Document]]: Core, near, and far documents

        """
        # Extract documents from sorted list
        docs = [doc for doc, _ in sorted_docs]

        # Determine cutoffs based on max_hops

        # Core documents are within max_hops
        core_cutoff = int(len(docs) * 0.4)  # Top 40% are core
        near_cutoff = int(len(docs) * 0.7)  # Next 30% are near

        # Ensure at least one document in each category
        core_cutoff = max(1, min(core_cutoff, len(docs) - 2))
        near_cutoff = max(core_cutoff + 1, min(near_cutoff, len(docs) - 1))

        # Split documents into categories
        core_docs = docs[:core_cutoff]
        near_docs = docs[core_cutoff:near_cutoff]
        far_docs = docs[near_cutoff:]

        return core_docs, near_docs, far_docs

    def _create_summaries(self, far_docs: list[Document]) -> list[str]:
        """
        Create summaries for far documents.

        Args:
        ----
            far_docs: Far documents to summarize

        Returns:
        -------
            List[str]: Summaries of far documents

        """
        summaries = []

        # Group documents by source using defaultdict
        grouped_docs = defaultdict(list)
        for doc in far_docs:
            source = _extract_metadata_value(doc.metadata, "source") or "unknown"
            grouped_docs[source].append(doc)

        # Create a summary for each group
        for source, docs in grouped_docs.items():
            # Extract titles or first lines
            titles = []
            for doc in docs:
                # Extract title or use first line
                title = _extract_metadata_value(doc.metadata, "title")
                if title is None:
                    # Use first line, truncated if needed
                    first_line = doc.content.split("\n")[0][:50]
                    title = first_line + "..." if len(first_line) == 50 else first_line
                titles.append(title)

            # Create a summary
            summary = f"• {source}: {', '.join(titles[:3])}"
            if len(titles) > 3:
                summary += f" and {len(titles) - 3} more"

            summaries.append(summary)

        return summaries

    def _build_composed_prompt(
        self,
        core_docs: list[Document],
        near_docs: list[Document],
        summaries: list[str],
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        """
        Build the composed prompt.

        Args:
        ----
            core_docs: Core documents
            near_docs: Near documents
            summaries: Summaries of far documents
            prompt: User prompt
            system_prompt: System prompt

        Returns:
        -------
            str: Composed prompt

        """
        # Start with system prompt if provided
        composed_parts = []
        if system_prompt:
            composed_parts.append(f"SYSTEM: {system_prompt}")

        # Add core documents with focus tags
        if core_docs:
            composed_parts.append(self.config.core_tag)
            for doc in core_docs:
                # Build source info from metadata
                source = _extract_metadata_value(doc.metadata, "source")
                title = _extract_metadata_value(doc.metadata, "title")

                source_info = []
                if source:
                    source_info.append(f"# Source: {source}")
                if title:
                    source_info.append(f"# Title: {title}")

                # Add document with source info if available
                content = doc.content
                if source_info:
                    header = "\n".join(source_info)
                    content = f"{header}\n{content}"
                composed_parts.append(content)
            composed_parts.append(f"[/{self.config.core_tag[1:]}")  # Remove leading [ and add /

        # Add near documents with focus tags
        if near_docs:
            composed_parts.append(self.config.near_tag)
            for doc in near_docs:
                # Build source info from metadata
                source = _extract_metadata_value(doc.metadata, "source")
                title = _extract_metadata_value(doc.metadata, "title")

                source_info = []
                if source:
                    source_info.append(f"# Source: {source}")
                if title:
                    source_info.append(f"# Title: {title}")

                # Add document with source info if available
                content = doc.content
                if source_info:
                    header = "\n".join(source_info)
                    content = f"{header}\n{content}"
                composed_parts.append(content)
            composed_parts.append(f"[/{self.config.near_tag[1:]}")  # Remove leading [ and add /

        # Add summaries
        if summaries:
            composed_parts.append(self.config.summary_tag)
            composed_parts.append("Unrelated chunks were summarized:")
            composed_parts.extend(summaries)
            composed_parts.append(f"[/{self.config.summary_tag[1:]}")  # Remove leading [ and add /

        # Add user prompt
        composed_parts.append(f"USER: {prompt}")

        # Join all parts with double newlines
        return "\n\n".join(composed_parts)

    def _get_graph_distance(self, source_id: str, target_id: str) -> int | None:
        """
        Get the distance between two nodes in the graph.

        Args:
        ----
            source_id: Source node ID
            target_id: Target node ID

        Returns:
        -------
            Optional[int]: Distance between nodes or None if not connected

        """
        # Check cache first
        cache_key = (source_id, target_id)
        if cache_key in self._distance_cache:
            return self._distance_cache[cache_key]

        # If source and target are the same, distance is 0
        if source_id == target_id:
            self._distance_cache[cache_key] = 0
            return 0

        # Check if both nodes exist in the graph
        if source_id not in self.graph.nodes or target_id not in self.graph.nodes:
            return None

        # Calculate distance using shortest path
        try:
            # Find paths between nodes
            paths = self.graph.find_paths(
                source_id=source_id,
                target_id=target_id,
                max_hops=self.config.max_hops,
            )

            if not paths:
                # No path found within max_hops
                return None

            # Get the shortest path length
            distance = min(len(path) for path in paths)
            self._distance_cache[cache_key] = distance
            return distance
        except ValueError:
            # No path exists
            return None
