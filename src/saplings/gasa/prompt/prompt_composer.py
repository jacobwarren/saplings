from __future__ import annotations

"""
Graph-aware prompt composer for GASA with third-party LLM APIs.

This module provides a prompt composer that structures prompts based on
graph relationships for use with third-party LLM APIs like OpenAI and Anthropic.
"""


import logging

from saplings.gasa.config import FallbackStrategy, GASAConfig, MaskStrategy
from saplings.memory.document import Document
from saplings.memory.graph import DependencyGraph
from saplings.security.sanitizer import sanitize_prompt

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
        # Use default configuration if none is provided
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
            enable_prompt_composer=False,
            focus_tags=False,
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

        # Group documents by source or type
        grouped_docs = {}
        for doc in far_docs:
            source = "unknown"
            if doc.metadata:
                # Get source
                if isinstance(doc.metadata, dict) and "source" in doc.metadata:
                    source = doc.metadata["source"]
                elif hasattr(doc.metadata, "source"):
                    # Use getattr to avoid type checking issues
                    source = getattr(doc.metadata, "source", None)

            if source not in grouped_docs:
                grouped_docs[source] = []
            grouped_docs[source].append(doc)

        # Create a summary for each group
        for source, docs in grouped_docs.items():
            # Extract titles or first lines
            titles = []
            for doc in docs:
                title = None
                if doc.metadata:
                    # Get title from metadata or custom field
                    if isinstance(doc.metadata, dict):
                        if "title" in doc.metadata:
                            title = doc.metadata["title"]
                        elif (
                            "custom" in doc.metadata
                            and isinstance(doc.metadata["custom"], dict)
                            and "title" in doc.metadata["custom"]
                        ):
                            title = doc.metadata["custom"]["title"]
                    # Check if metadata is DocumentMetadata
                    elif (
                        hasattr(doc.metadata, "custom")
                        and isinstance(doc.metadata.custom, dict)
                        and "title" in doc.metadata.custom
                    ):
                        title = doc.metadata.custom["title"]

                # If no title found, use first line
                if not title:
                    # Use first line as title
                    first_line = doc.content.split("\n")[0]
                    if len(first_line) > 50:
                        first_line = first_line[:50] + "..."
                    title = first_line

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
            composed_parts.append(f"{self.config.core_tag}")
            for doc in core_docs:
                # Add source information if available
                source_info = ""
                # Handle both DocumentMetadata and dictionary metadata
                if doc.metadata:
                    # Get source
                    source = None
                    if isinstance(doc.metadata, dict) and "source" in doc.metadata:
                        source = doc.metadata["source"]
                    elif hasattr(doc.metadata, "source"):
                        # Use getattr to avoid type checking issues
                        source = getattr(doc.metadata, "source", None)

                    if source:
                        source_info = f"# Source: {source}"

                        # Get title from metadata or custom field
                        title = None
                        # Check if metadata is a dictionary
                        if isinstance(doc.metadata, dict):
                            if "title" in doc.metadata:
                                title = doc.metadata["title"]
                            elif (
                                "custom" in doc.metadata
                                and isinstance(doc.metadata["custom"], dict)
                                and "title" in doc.metadata["custom"]
                            ):
                                title = doc.metadata["custom"]["title"]
                        # Check if metadata is DocumentMetadata
                        elif (
                            hasattr(doc.metadata, "custom")
                            and isinstance(doc.metadata.custom, dict)
                            and "title" in doc.metadata.custom
                        ):
                            title = doc.metadata.custom["title"]

                        if title:
                            source_info += f"\n# Title: {title}"

                # Add the document content with source info
                if source_info:
                    composed_parts.append(f"{source_info}\n{doc.content}")
                else:
                    composed_parts.append(doc.content)
            composed_parts.append(f"[/{self.config.core_tag[1:]}]")

        # Add near documents
        if near_docs:
            composed_parts.append(f"{self.config.near_tag}")
            for doc in near_docs:
                # Add source information if available
                source_info = ""
                # Handle both DocumentMetadata and dictionary metadata
                if doc.metadata:
                    # Get source
                    source = None
                    if isinstance(doc.metadata, dict) and "source" in doc.metadata:
                        source = doc.metadata["source"]
                    elif hasattr(doc.metadata, "source"):
                        # Use getattr to avoid type checking issues
                        source = getattr(doc.metadata, "source", None)

                    if source:
                        source_info = f"# Source: {source}"

                        # Get title from metadata or custom field
                        title = None
                        # Check if metadata is a dictionary
                        if isinstance(doc.metadata, dict):
                            if "title" in doc.metadata:
                                title = doc.metadata["title"]
                            elif (
                                "custom" in doc.metadata
                                and isinstance(doc.metadata["custom"], dict)
                                and "title" in doc.metadata["custom"]
                            ):
                                title = doc.metadata["custom"]["title"]
                        # Check if metadata is DocumentMetadata
                        elif (
                            hasattr(doc.metadata, "custom")
                            and isinstance(doc.metadata.custom, dict)
                            and "title" in doc.metadata.custom
                        ):
                            title = doc.metadata.custom["title"]

                        if title:
                            source_info += f"\n# Title: {title}"

                # Add the document content with source info
                if source_info:
                    composed_parts.append(f"{source_info}\n{doc.content}")
                else:
                    composed_parts.append(doc.content)
            composed_parts.append(f"[/{self.config.near_tag[1:]}]")

        # Add summaries
        if summaries:
            composed_parts.append(f"{self.config.summary_tag}")
            composed_parts.append("Unrelated chunks were summarized:")
            composed_parts.extend(summaries)
            composed_parts.append(f"[/{self.config.summary_tag[1:]}]")

        # Add sanitized user prompt
        safe_prompt = sanitize_prompt(prompt)
        composed_parts.append(f"USER: {safe_prompt}")

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
        # Use nodes dictionary instead of has_node method
        if source_id not in self.graph.nodes or target_id not in self.graph.nodes:
            return None

        # Calculate distance using shortest path
        try:
            # Use networkx shortest_path_length method directly
            import networkx as nx

            distance = nx.shortest_path_length(self.graph.graph, source=source_id, target=target_id)
            self._distance_cache[cache_key] = distance
            return distance
        except (nx.NetworkXNoPath, nx.NetworkXError, ValueError):
            # No path exists
            return None
