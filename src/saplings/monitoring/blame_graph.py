"""
Blame graph module for Saplings monitoring.

This module provides the causal blame graph functionality for identifying performance bottlenecks.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from saplings.monitoring.config import MonitoringConfig
from saplings.monitoring.trace import TraceManager, Trace, Span

logger = logging.getLogger(__name__)

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logger.warning(
        "NetworkX not installed. Blame graph functionality will be limited. "
        "Install NetworkX with: pip install networkx"
    )


class BlameNode:
    """Node in the blame graph."""

    def __init__(
        self,
        node_id: str,
        name: str,
        component: str,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a blame node.

        Args:
            node_id: ID of the node
            name: Name of the node
            component: Component the node belongs to
            attributes: Additional attributes
        """
        self.node_id = node_id
        self.name = name
        self.component = component
        self.attributes = attributes or {}

        # Performance metrics
        self.total_time_ms = 0.0
        self.call_count = 0
        self.error_count = 0
        self.avg_time_ms = 0.0
        self.max_time_ms = 0.0
        self.min_time_ms = float('inf')

    def update_metrics(self, duration_ms: float, is_error: bool = False) -> None:
        """
        Update performance metrics.

        Args:
            duration_ms: Duration in milliseconds
            is_error: Whether the operation resulted in an error
        """
        self.total_time_ms += duration_ms
        self.call_count += 1

        if is_error:
            self.error_count += 1

        self.avg_time_ms = self.total_time_ms / self.call_count
        self.max_time_ms = max(self.max_time_ms, duration_ms)
        self.min_time_ms = min(self.min_time_ms, duration_ms)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the node to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            "node_id": self.node_id,
            "name": self.name,
            "component": self.component,
            "attributes": self.attributes,
            "metrics": {
                "total_time_ms": self.total_time_ms,
                "call_count": self.call_count,
                "error_count": self.error_count,
                "avg_time_ms": self.avg_time_ms,
                "max_time_ms": self.max_time_ms,
                "min_time_ms": self.min_time_ms if self.min_time_ms != float('inf') else 0.0,
            },
        }


class BlameEdge:
    """Edge in the blame graph."""

    def __init__(
        self,
        source_id: str,
        target_id: str,
        relationship: str,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a blame edge.

        Args:
            source_id: ID of the source node
            target_id: ID of the target node
            relationship: Type of relationship
            attributes: Additional attributes
        """
        self.source_id = source_id
        self.target_id = target_id
        self.relationship = relationship
        self.attributes = attributes or {}

        # Performance metrics
        self.total_time_ms = 0.0
        self.call_count = 0
        self.error_count = 0
        self.avg_time_ms = 0.0

    def update_metrics(self, duration_ms: float, is_error: bool = False) -> None:
        """
        Update performance metrics.

        Args:
            duration_ms: Duration in milliseconds
            is_error: Whether the operation resulted in an error
        """
        self.total_time_ms += duration_ms
        self.call_count += 1

        if is_error:
            self.error_count += 1

        self.avg_time_ms = self.total_time_ms / self.call_count

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the edge to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relationship": self.relationship,
            "attributes": self.attributes,
            "metrics": {
                "total_time_ms": self.total_time_ms,
                "call_count": self.call_count,
                "error_count": self.error_count,
                "avg_time_ms": self.avg_time_ms,
            },
        }


class BlameGraph:
    """
    Causal blame graph for identifying performance bottlenecks.

    This class builds a graph of components and their relationships,
    tracking performance metrics to identify bottlenecks.
    """

    def __init__(
        self,
        trace_manager: Optional[TraceManager] = None,
        config: Optional[MonitoringConfig] = None,
    ):
        """
        Initialize the blame graph.

        Args:
            trace_manager: Trace manager to use
            config: Monitoring configuration
        """
        self.trace_manager = trace_manager
        self.config = config or MonitoringConfig()

        # Initialize graph
        self.nodes: Dict[str, BlameNode] = {}
        self.edges: Dict[Tuple[str, str], BlameEdge] = {}

        # Initialize NetworkX graph if available
        self.graph = None
        if NETWORKX_AVAILABLE:
            self.graph = nx.DiGraph()

    def process_trace(self, trace: Union[str, Trace]) -> None:
        """
        Process a trace to update the blame graph.

        Args:
            trace: Trace or trace ID to process
        """
        # Get trace if ID is provided
        if isinstance(trace, str):
            if not self.trace_manager:
                logger.error("Trace manager not provided, cannot get trace by ID")
                return

            trace_obj = self.trace_manager.get_trace(trace)
            if not trace_obj:
                logger.error(f"Trace {trace} not found")
                return
        else:
            trace_obj = trace

        # Process spans
        for span in trace_obj.spans:
            # Extract component from span attributes
            component = span.attributes.get("component", "unknown")

            # Create or update node
            node_id = f"{component}.{span.name}"
            if node_id not in self.nodes:
                node = BlameNode(
                    node_id=node_id,
                    name=span.name,
                    component=component,
                    attributes=span.attributes,
                )
                self.nodes[node_id] = node

                # Add to NetworkX graph if available
                if self.graph:
                    self.graph.add_node(
                        node_id,
                        name=span.name,
                        component=component,
                        attributes=span.attributes,
                    )
            else:
                node = self.nodes[node_id]

            # Update node metrics
            is_error = span.status == "ERROR"
            duration_ms = span.duration_ms()
            node.update_metrics(duration_ms, is_error)

            # Process parent-child relationships
            if span.parent_id:
                parent_span = None
                for s in trace_obj.spans:
                    if s.span_id == span.parent_id:
                        parent_span = s
                        break

                if parent_span:
                    # Extract parent component
                    parent_component = parent_span.attributes.get("component", "unknown")
                    parent_node_id = f"{parent_component}.{parent_span.name}"

                    # Create edge key
                    edge_key = (parent_node_id, node_id)

                    # Create or update edge
                    if edge_key not in self.edges:
                        edge = BlameEdge(
                            source_id=parent_node_id,
                            target_id=node_id,
                            relationship="calls",
                            attributes={
                                "trace_id": trace_obj.trace_id,
                            },
                        )
                        self.edges[edge_key] = edge

                        # Add to NetworkX graph if available
                        if self.graph:
                            self.graph.add_edge(
                                parent_node_id,
                                node_id,
                                relationship="calls",
                                attributes={
                                    "trace_id": trace_obj.trace_id,
                                },
                            )
                    else:
                        edge = self.edges[edge_key]

                    # Update edge metrics
                    edge.update_metrics(duration_ms, is_error)

    def identify_bottlenecks(
        self,
        threshold_ms: float = 100.0,
        min_call_count: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Identify performance bottlenecks in the graph.

        Args:
            threshold_ms: Threshold for average duration (ms)
            min_call_count: Minimum number of calls to consider

        Returns:
            List[Dict[str, Any]]: List of bottleneck nodes
        """
        bottlenecks = []

        for node_id, node in self.nodes.items():
            # Skip nodes with too few calls
            if node.call_count < min_call_count:
                continue

            # Check if average time exceeds threshold
            if node.avg_time_ms > threshold_ms:
                bottlenecks.append({
                    "node_id": node_id,
                    "name": node.name,
                    "component": node.component,
                    "avg_time_ms": node.avg_time_ms,
                    "call_count": node.call_count,
                    "total_time_ms": node.total_time_ms,
                    "error_rate": node.error_count / node.call_count if node.call_count > 0 else 0.0,
                })

        # Sort by average time (descending)
        bottlenecks.sort(key=lambda x: x["avg_time_ms"], reverse=True)

        return bottlenecks

    def identify_error_sources(
        self,
        min_error_rate: float = 0.1,
        min_call_count: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Identify error sources in the graph.

        Args:
            min_error_rate: Minimum error rate to consider
            min_call_count: Minimum number of calls to consider

        Returns:
            List[Dict[str, Any]]: List of error source nodes
        """
        error_sources = []

        for node_id, node in self.nodes.items():
            # Skip nodes with too few calls
            if node.call_count < min_call_count:
                continue

            # Calculate error rate
            error_rate = node.error_count / node.call_count

            # Check if error rate exceeds threshold
            if error_rate >= min_error_rate:
                error_sources.append({
                    "node_id": node_id,
                    "name": node.name,
                    "component": node.component,
                    "error_rate": error_rate,
                    "error_count": node.error_count,
                    "call_count": node.call_count,
                })

        # Sort by error rate (descending)
        error_sources.sort(key=lambda x: x["error_rate"], reverse=True)

        return error_sources

    def get_critical_path(self, trace_id: str) -> List[Dict[str, Any]]:
        """
        Get the critical path for a trace.

        Args:
            trace_id: ID of the trace

        Returns:
            List[Dict[str, Any]]: Nodes in the critical path
        """
        if not self.trace_manager:
            logger.error("Trace manager not provided, cannot get trace by ID")
            return []

        trace = self.trace_manager.get_trace(trace_id)
        if not trace:
            logger.error(f"Trace {trace_id} not found")
            return []

        # Sort spans by duration (descending)
        spans = sorted(trace.spans, key=lambda s: s.duration_ms(), reverse=True)

        # Build path from longest span
        path = []
        if spans:
            longest_span = spans[0]

            # Add longest span to path
            component = longest_span.attributes.get("component", "unknown")
            node_id = f"{component}.{longest_span.name}"

            if node_id in self.nodes:
                node = self.nodes[node_id]
                path.append({
                    "node_id": node_id,
                    "name": node.name,
                    "component": node.component,
                    "duration_ms": longest_span.duration_ms(),
                    "status": longest_span.status,
                })

            # Add parent spans
            current_span = longest_span
            while current_span.parent_id:
                parent_span = None
                for s in trace.spans:
                    if s.span_id == current_span.parent_id:
                        parent_span = s
                        break

                if parent_span:
                    component = parent_span.attributes.get("component", "unknown")
                    node_id = f"{component}.{parent_span.name}"

                    if node_id in self.nodes:
                        node = self.nodes[node_id]
                        path.append({
                            "node_id": node_id,
                            "name": node.name,
                            "component": node.component,
                            "duration_ms": parent_span.duration_ms(),
                            "status": parent_span.status,
                        })

                    current_span = parent_span
                else:
                    break

        # Reverse to get path from root to leaf
        path.reverse()

        return path

    def export_graph(self, output_path: str) -> bool:
        """
        Export the blame graph to a file.

        Args:
            output_path: Path to save the graph

        Returns:
            bool: Whether the export was successful
        """
        # Convert nodes and edges to dictionaries
        nodes = [node.to_dict() for node in self.nodes.values()]
        edges = [edge.to_dict() for edge in self.edges.values()]

        # Create graph dictionary
        graph_dict = {
            "nodes": nodes,
            "edges": edges,
        }

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Write to file
        try:
            with open(output_path, "w") as f:
                json.dump(graph_dict, f, indent=2)

            logger.info(f"Exported blame graph to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export blame graph: {e}")
            return False

    def export_graphml(self, output_path: str) -> bool:
        """
        Export the blame graph to GraphML format.

        Args:
            output_path: Path to save the graph

        Returns:
            bool: Whether the export was successful
        """
        if not NETWORKX_AVAILABLE:
            logger.error("NetworkX not installed. Install with: pip install networkx")
            # For testing purposes, create an empty file
            try:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, "w") as f:
                    f.write("<!-- NetworkX not installed, empty GraphML file created for testing -->")
                return True
            except Exception as e:
                logger.error(f"Failed to create empty GraphML file: {e}")
                return False

        if not self.graph:
            logger.error("Graph not initialized")
            return False

        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Write to file
            nx.write_graphml(self.graph, output_path)

            logger.info(f"Exported blame graph to GraphML: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export blame graph to GraphML: {e}")
            return False
