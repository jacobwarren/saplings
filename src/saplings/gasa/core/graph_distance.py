from __future__ import annotations

"""
Graph distance module for Graph-Aligned Sparse Attention (GASA).

This module provides efficient algorithms for calculating distances between nodes
in the dependency graph, optimizing the graph distance queries needed for GASA.
"""


import heapq
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from saplings.memory.graph import DependencyGraph

logger = logging.getLogger(__name__)


class GraphDistanceCalculator:
    """
    Efficient calculator for graph distances.

    This class provides optimized algorithms for calculating distances between nodes
    in a dependency graph, reducing the computational complexity compared to per-pair
    breadth-first search.
    """

    def __init__(self, graph: DependencyGraph) -> None:
        """
        Initialize the graph distance calculator.

        Args:
        ----
            graph: Dependency graph to calculate distances for

        """
        self.graph = graph
        self.distance_cache: dict[tuple[str, str], int | float] = {}

    def build_distance_matrix(
        self, node_ids: list[str], max_hops: int
    ) -> dict[tuple[str, str], int | float]:
        """
        Build a distance matrix for a set of nodes.

        Instead of computing distances on-demand with BFS for each pair (which is O(n²·BFS)),
        this computes all pairwise distances using a more efficient algorithm.

        Args:
        ----
            node_ids: List of node IDs to calculate distances for
            max_hops: Maximum number of hops to consider

        Returns:
        -------
            Dict[Tuple[str, str], Union[int, float]]: Dictionary mapping node ID pairs to their distances,
                with float('inf') for unreachable pairs

        """
        if not node_ids:
            return {}

        # Create a new distance cache
        distance_matrix = {}

        # If we have only a few nodes, use single-source Dijkstra for each node
        if len(node_ids) <= 100:  # Threshold for using Dijkstra over Floyd-Warshall
            for source_id in node_ids:
                distances = self._dijkstra(source_id, node_ids, max_hops)
                for target_id, distance in distances.items():
                    distance_matrix[(source_id, target_id)] = distance
                    # Cache the reverse direction too
                    distance_matrix[(target_id, source_id)] = distance
        else:
            # For larger graphs, use Floyd-Warshall
            distance_matrix = self._floyd_warshall(node_ids, max_hops)

        # Cache the results
        self.distance_cache.update(distance_matrix)

        return distance_matrix

    def get_distance(self, source_id: str, target_id: str, max_hops: int) -> int | float:
        """
        Get the distance between two nodes.

        Args:
        ----
            source_id: Source node ID
            target_id: Target node ID
            max_hops: Maximum number of hops to consider

        Returns:
        -------
            Union[int, float]: Distance between the nodes, or float('inf') if no path exists
                 within max_hops

        """
        # Check cache first
        cache_key = (source_id, target_id)
        if cache_key in self.distance_cache:
            return self.distance_cache[cache_key]

        # If source and target are the same, distance is 0
        if source_id == target_id:
            self.distance_cache[cache_key] = 0
            return 0

        # Calculate distance using Dijkstra's algorithm
        distances = self._dijkstra(source_id, [target_id], max_hops)
        distance = distances.get(target_id, float("inf"))

        # Cache the result
        self.distance_cache[cache_key] = distance
        self.distance_cache[(target_id, source_id)] = distance  # Cache reverse direction too

        return distance

    def _dijkstra(
        self, source_id: str, target_ids: list[str], max_hops: int
    ) -> dict[str, int | float]:
        """
        Calculate distances from source to targets using Dijkstra's algorithm.

        Args:
        ----
            source_id: Source node ID
            target_ids: List of target node IDs
            max_hops: Maximum number of hops to consider

        Returns:
        -------
            Dict[str, Union[int, float]]: Dictionary mapping target IDs to their distances from the source,
                with float('inf') for unreachable targets

        """
        # Check if the source node exists
        if source_id not in self.graph.nodes:
            return {target_id: float("inf") for target_id in target_ids}

        # Set of target IDs for fast lookup
        target_set = set(target_ids)

        # Initialize distances
        distances = {source_id: 0}

        # Priority queue for Dijkstra's algorithm
        queue = [(0, source_id)]

        # Set of visited nodes
        visited = set()

        # Process the queue
        while queue and len(visited.intersection(target_set)) < len(target_set):
            distance, node_id = heapq.heappop(queue)

            # Skip if we've already visited this node or exceeded max_hops
            if node_id in visited or distance > max_hops:
                continue

            # Mark as visited
            visited.add(node_id)

            # Process neighbors
            try:
                neighbors = self.graph.get_neighbors(node_id)

                for neighbor in neighbors:
                    neighbor_id = neighbor.id
                    new_distance = distance + 1

                    # Update distance if we found a shorter path
                    if neighbor_id not in distances or new_distance < distances[neighbor_id]:
                        distances[neighbor_id] = new_distance

                        # Add to queue if within max_hops
                        if new_distance <= max_hops:
                            heapq.heappush(queue, (new_distance, neighbor_id))
            except ValueError:
                # Node not found or other error
                pass

        # Filter results to only include targets
        return {target_id: distances.get(target_id, float("inf")) for target_id in target_ids}

    def _floyd_warshall(
        self, node_ids: list[str], max_hops: int
    ) -> dict[tuple[str, str], int | float]:
        """
        Calculate all-pairs shortest paths using Floyd-Warshall algorithm.

        This is more efficient than running Dijkstra for each node when we have
        a large, dense graph and need all-pairs distances.

        Args:
        ----
            node_ids: List of node IDs to calculate distances for
            max_hops: Maximum number of hops to consider

        Returns:
        -------
            Dict[Tuple[str, str], Union[int, float]]: Dictionary mapping node ID pairs to their distances,
                with float('inf') for unreachable pairs

        """
        # Create a distance matrix
        distances = {}

        # Initialize distances
        for _i, source_id in enumerate(node_ids):
            for _j, target_id in enumerate(node_ids):
                if source_id == target_id:
                    # Distance to self is 0
                    distances[(source_id, target_id)] = 0
                elif self._are_adjacent(source_id, target_id):
                    # Adjacent nodes have distance 1
                    distances[(source_id, target_id)] = 1
                else:
                    # Other nodes have infinite distance initially
                    distances[(source_id, target_id)] = float("inf")

        # Floyd-Warshall algorithm
        for _k, k_id in enumerate(node_ids):
            for _i, i_id in enumerate(node_ids):
                for _j, j_id in enumerate(node_ids):
                    # Check if going through k provides a shorter path
                    if distances.get((i_id, k_id), float("inf")) < float("inf") and distances.get(
                        (k_id, j_id), float("inf")
                    ) < float("inf"):
                        new_distance = distances[(i_id, k_id)] + distances[(k_id, j_id)]
                        current_distance = distances.get((i_id, j_id), float("inf"))

                        if new_distance < current_distance:
                            distances[(i_id, j_id)] = new_distance

        # Filter out distances greater than max_hops
        return {
            (source_id, target_id): min(distance, max_hops + 1)
            for (source_id, target_id), distance in distances.items()
            if distance <= max_hops or source_id == target_id
        }

    def _are_adjacent(self, node_id1: str, node_id2: str) -> bool:
        """
        Check if two nodes are adjacent in the graph.

        Args:
        ----
            node_id1: First node ID
            node_id2: Second node ID

        Returns:
        -------
            bool: True if the nodes are adjacent, False otherwise

        """
        try:
            neighbors = self.graph.get_neighbors(node_id1)
            return any(neighbor.id == node_id2 for neighbor in neighbors)
        except ValueError:
            return False
